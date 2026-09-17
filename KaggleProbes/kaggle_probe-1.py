"""
Kaggriculture — ONE-SUBMISSION PROBE agent

WHAT THIS IS
    A legal agent that plays a season, never touches the farm (always PASS),
    and spends its per-turn compute budget measuring the machine the
    competition grades on.

    IT SPLITS ITS WORK BY SEAT, because in the validation episode both seats
    are us. From the competition's own pages:

        "When you upload a submission, a Validation Episode is run where your
         agent plays against a copy of itself to ensure it runs without errors."

    So one submission is two agents, and they divide the job:

      SEAT 0 - the clean seat. Runs the full measurement suite (steps 0-23),
               drift all season, the contention windows, and publishes the
               summary at step 290. It never misbehaves and it finishes the
               season, so it is the seat that carries the results home.

      SEAT 1 - the expendable seat. A short census to confirm both seats see
               the same machine, then it makes CPU noise for seat 0 to measure
               against.

    That split buys three things a single-seat probe cannot have:

      1. The error ladder stops being dangerous. Seat 1 dying costs nothing,
         because seat 0 has the measurements and keeps going to step 719.
      2. CONTENTION becomes measurable. Both agents think inside the same turn
         on Kaggle; seat 1 burns CPU in a background thread while seat 0 runs
         its reference bench, and the ratio says how much of our compute the
         opponent's thinking actually takes.
      3. OPPONENT DEATH becomes an experiment. Seat 0 knows the step seat 1 is
         scheduled to raise, so it snapshots what is observable about the
         opponent before and after and reports what changed.

    Set LADDER_ENABLED = False to submit the same file as a clean measurement
    run: without the ladder, validation passes and we also get logs from real
    leaderboard games.

WHAT THIS AGENT MEASURES
    1. Machine census: cores, cgroup CPU quota, memory ceiling, CPU model and
       SIMD flags, BLAS backend, thread env vars, installed packages.
    2. How much CPU we actually get: CPU-seconds per wall-second, at 1/2/3/4
       threads. This is the "is 2 threads better than 1" question, measured.
    3. scipy.optimize.milp: import cost, then solve time vs problem size.
    4. MLP forward pass: numpy vs torch vs jax, at 1/2/4 threads, same shapes.
    5. multiprocessing: fork cost and whether extra processes buy throughput.
    6. The real billing function: deliberately overruns by a known amount and
       reads `remainingOverageTime` back to derive what an overrun costs HERE.
    7. Drift: re-runs one benchmark every 50 turns until the ladder arms, so we
       know how much this machine wanders under a noisy neighbour.

SAFETY
    - Always returns {"farmer": ["PASS"], "hands": [], "market": []}.
    - Never raises: every task is wrapped, a failed task is logged and skipped.
    - Self-limits to SAFE_SECONDS of work per turn, with a cooldown turn after
      any turn that ran long.
    - Watches its own overage bank and drops remaining work below BANK_FLOOR.
    - Heavy imports are LAZY and timed, never at module load.
    - Emits a rolling summary every 200 steps, so an episode that dies early
      still returns most of its findings.

READING THE LOG
    Every line is:  PROBE|<tag>|<json>
    Filter with:    grep '^PROBE|' agent_0.log
    The interesting tags, in order: machine, packages, cpu_capacity,
    thread_sweep_subprocess, milp, nn_numpy, nn_torch, nn_jax, billing,
    summary, ladder_armed, ATTEMPT, SURVIVED, LADDER_COMPLETE.
"""

import gc
import json
import os
import platform
import sys
import time

# --------------------------------------------------------------------------- #
# module-load markers: taken before anything heavy, so "how long did Kaggle
# spend loading us, and was that inside a turn budget" is answerable.
# --------------------------------------------------------------------------- #
_T_MODULE_START = time.perf_counter()
_WALL_MODULE_START = time.time()

try:
    import numpy as np
    _NUMPY_OK = True
except Exception as _exc:  # pragma: no cover - the finding IS the failure
    np = None
    _NUMPY_OK = False
    _NUMPY_ERR = repr(_exc)

_T_MODULE_END = time.perf_counter()

# --------------------------------------------------------------------------- #
# budget policy
# --------------------------------------------------------------------------- #
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}

SAFE_SECONDS = 0.70   # self-time ceiling for ordinary turns (free turn is 1.0)
BANK_FLOOR = 25.0     # stop all heavy work below this much overage bank left
DRIFT_EVERY = 50      # re-measure the reference bench every N steps
QUIET_AFTER = 690     # no new work this late; just finish the season cleanly

# The error ladder runs LAST, and only once the measurement queue is empty and
# its summary is on the wire. Rung 11 (`raise`) ended our participation in the
# local dry run, so anything scheduled before the ladder must already be done
# and logged - the ladder is allowed to be the last thing this agent ever does.
# The ladder runs in SEAT 1 only, so seat 0 keeps measuring all season whatever
# happens to seat 1. Set False to submit the same file as a clean measurement
# run: without the ladder the validation episode passes, the agent reaches the
# leaderboard, and we also get logs from real games. See PROBE_README.
LADDER_ENABLED = True
LADDER_START = 300    # earliest step for the first rung (seat 1)
LADDER_GAP = 6        # steps between rungs: enough to see the aftermath
SUMMARY_AT = 290      # seat 0 puts everything on the wire BEFORE seat 1 breaks


def emit(tag, **fields):
    """One JSON line per measurement. Flushed, because the episode may end."""
    try:
        print("PROBE|%s|%s\n" % (tag, json.dumps(fields, default=str)))
    except Exception:
        pass


def _read(path, limit=4096):
    try:
        with open(path, "r") as fh:
            return fh.read(limit).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# thread control: three mechanisms, in order of reliability. We report WHICH
# one worked, because "we set the threads" is not the same as "the threads
# changed" - re-reading the variable you just set measures your own assignment.
# --------------------------------------------------------------------------- #
_THREADCTL = {"mechanism": None, "checked": False}
_GLOBAL_SHARE_VAR = -1


def _thread_mechanism():
    if _THREADCTL["checked"]:
        return _THREADCTL["mechanism"]
    _THREADCTL["checked"] = True
    try:
        import threadpoolctl  # noqa: F401
        _THREADCTL["mechanism"] = "threadpoolctl"
        return "threadpoolctl"
    except Exception:
        pass
    # OpenBLAS exposes a runtime setter. find_library() will NOT find it: the
    # wheel bundles it inside site-packages (scipy_openblas64, numpy.libs).
    # So read the shared objects this process has ALREADY mapped and use the
    # real one - which is also proof it is the library numpy is calling.
    try:
        import ctypes
        maps = _read("/proc/self/maps", 1 << 20) or ""
        seen = set()
        for line in maps.splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            path = parts[-1]
            low = path.lower()
            if path in seen or ".so" not in low:
                continue
            seen.add(path)
            if "openblas" not in low and "libmkl" not in low and "libomp" not in low:
                continue
            try:
                lib = ctypes.CDLL(path)
            except Exception:
                continue
            for setter in ("openblas_set_num_threads", "openblas_set_num_threads64_",
                           "MKL_Set_Num_Threads", "omp_set_num_threads"):
                if hasattr(lib, setter):
                    _THREADCTL["mechanism"] = "ctypes:" + setter
                    _THREADCTL["lib"] = lib
                    _THREADCTL["setter"] = setter
                    _THREADCTL["lib_path"] = path
                    return _THREADCTL["mechanism"]
    except Exception:
        pass
    _THREADCTL["mechanism"] = "none"
    return "none"


class blas_threads(object):
    """Context manager that limits BLAS threads, or admits it could not."""

    def __init__(self, n):
        self.n = n
        self._ctx = None
        self.effective = None

    def __enter__(self):
        mech = _thread_mechanism()
        try:
            if mech == "threadpoolctl":
                import threadpoolctl
                self._ctx = threadpoolctl.threadpool_limits(limits=self.n)
                self.effective = self.n
            elif mech.startswith("ctypes:"):
                getattr(_THREADCTL["lib"], _THREADCTL["setter"])(self.n)
                self.effective = self.n
            else:
                self.effective = None  # uncontrolled; measurement still valid
        except Exception:
            self.effective = None
        return self

    def __exit__(self, *exc):
        try:
            if self._ctx is not None:
                self._ctx.__exit__(*exc)
            elif str(_THREADCTL.get("mechanism", "")).startswith("ctypes:"):
                getattr(_THREADCTL["lib"], _THREADCTL["setter"])(
                    os.cpu_count() or 1)  # restore
        except Exception:
            pass
        return False


def cpu_ratio(fn, min_seconds=0.12):
    """Run fn repeatedly for >= min_seconds; return (wall, cpu/wall, reps).

    cpu/wall is process CPU-seconds per wall-second == how many cores this
    workload actually got. It is the single most informative number here: on a
    1.6 vCPU quota, a perfectly parallel workload tops out near 1.6.
    """
    reps = 0
    t0 = time.perf_counter()
    c0 = time.process_time()
    while time.perf_counter() - t0 < min_seconds:
        fn()
        reps += 1
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    return wall, (cpu / wall if wall > 0 else 0.0), reps


# --------------------------------------------------------------------------- #
# tasks. Each returns None; each is called at most once unless re-queued.
# --------------------------------------------------------------------------- #

def task_machine_census():
    cg = {
        "cpu.max": _read("/sys/fs/cgroup/cpu.max"),
        "cfs_quota_us": _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "cfs_period_us": _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        "memory.max": _read("/sys/fs/cgroup/memory.max"),
        "memory.limit_in_bytes": _read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    }
    quota_cores = None
    try:
        if cg["cpu.max"]:
            q, p = cg["cpu.max"].split()
            if q != "max":
                quota_cores = int(q) / int(p)
        elif cg["cfs_quota_us"] and int(cg["cfs_quota_us"]) > 0:
            quota_cores = int(cg["cfs_quota_us"]) / int(cg["cfs_period_us"])
    except Exception:
        pass

    model, flags = None, []
    info = _read("/proc/cpuinfo", 65536) or ""
    for line in info.splitlines():
        if line.startswith("model name") and model is None:
            model = line.split(":", 1)[1].strip()
        if line.startswith("flags") and not flags:
            have = set(line.split(":", 1)[1].split())
            flags = [f for f in ("avx", "avx2", "avx512f", "avx512bw",
                                 "fma", "sse4_2", "amx_tile") if f in have]

    try:
        affinity = len(os.sched_getaffinity(0))
    except Exception:
        affinity = None

    emit("machine",
         cpu_count=os.cpu_count(),
         sched_affinity=affinity,
         cgroup_quota_cores=quota_cores,
         cgroup=cg,
         cpu_model=model,
         simd=flags,
         python=sys.version.split()[0],
         platform=platform.platform(),
         machine=platform.machine(),
         module_load_seconds=round(_T_MODULE_END - _T_MODULE_START, 4),
         numpy_ok=_NUMPY_OK,
         numpy_version=(np.__version__ if _NUMPY_OK else _NUMPY_ERR),
         thread_env={k: os.environ.get(k) for k in (
             "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE")},
         thread_control=_thread_mechanism())


def task_package_census():
    """find_spec does NOT import - this is cheap and safe, and tells us the
    whole image in one turn."""
    try:
        import importlib.util
        from importlib.metadata import version as _ver
    except Exception:
        _ver = lambda _n: None  # noqa: E731

    names = ["numpy", "scipy", "torch", "jax", "jaxlib", "ortools", "numba",
             "threadpoolctl", "cvxpy", "highspy",
             "networkx", "numexpr", "psutil", "kaggle_environments"]
    found = {}
    for n in names:
        try:
            spec = importlib.util.find_spec(n)
        except Exception:
            spec = None
        if spec is None:
            found[n] = None
            continue
        try:
            found[n] = _ver(n)
        except Exception:
            found[n] = "present"
    emit("packages", **found)

    if _NUMPY_OK:
        try:
            import io
            buf = io.StringIO()
            old, sys.stdout = sys.stdout, buf
            try:
                np.show_config()
            finally:
                sys.stdout = old
            emit("numpy_config", raw=buf.getvalue()[:1500])
        except Exception as exc:
            emit("numpy_config", error=repr(exc))


def _matmul_bench(n):
    a = np.random.rand(n, n).astype(np.float64)
    b = np.random.rand(n, n).astype(np.float64)

    def run():
        return a @ b
    return run


def task_cpu_capacity():
    """THE question: how many cores does this container really give a single
    process, and does asking for more threads get us more work done?"""
    if not _NUMPY_OK:
        emit("cpu_capacity", error="numpy unavailable")
        return
    run = _matmul_bench(320)
    rows = []
    for n_threads in (1, 2, 3, 4):
        with blas_threads(n_threads) as bt:
            wall, cores, reps = cpu_ratio(run, 0.10)
        flops = 2.0 * (320 ** 3) * reps
        rows.append({
            "threads_requested": n_threads,
            "threads_applied": bt.effective,
            "wall_s": round(wall, 4),
            "reps": reps,
            "cores_used": round(cores, 3),
            "gflops": round(flops / wall / 1e9, 3),
        })
    base = rows[0]["gflops"] or 1e-9
    for r in rows:
        r["speedup_vs_1"] = round((r["gflops"] or 0) / base, 3)
    emit("cpu_capacity", control=_thread_mechanism(), rows=rows)


def task_matmul_sizes():
    if not _NUMPY_OK:
        return
    rows = []
    for n in (64, 128, 256, 512):
        run = _matmul_bench(n)
        run()  # warm
        wall, cores, reps = cpu_ratio(run, 0.06)
        rows.append({"n": n, "gflops": round(2.0 * n ** 3 * reps / wall / 1e9, 3),
                     "cores_used": round(cores, 3),
                     "us_per_call": round(wall / reps * 1e6, 1)})
    emit("matmul_sizes", rows=rows)


_LAZY = {}


def _timed_import(modname, attr=None):
    t0 = time.perf_counter()
    try:
        mod = __import__(modname, fromlist=["_"] if "." in modname else [])
        dt = time.perf_counter() - t0
        _LAZY[modname] = mod
        emit("import", module=modname, seconds=round(dt, 4), ok=True,
             version=getattr(mod, "__version__", None))
        return mod
    except Exception as exc:
        dt = time.perf_counter() - t0
        emit("import", module=modname, seconds=round(dt, 4), ok=False,
             error=repr(exc)[:300])
        return None


def task_import_scipy():
    _timed_import("scipy")
    _timed_import("scipy.optimize")


def _milp_problem(n_items, n_groups):
    """A small assignment-with-capacity MILP, shaped like the tile problem:
    pick at most one action per tile, respect a global budget."""
    rng = np.random.default_rng(0)
    c = -rng.random(n_items * n_groups)          # maximise value
    weights = rng.random(n_items * n_groups) + 0.1
    return c, weights


def task_milp_sweep():
    sp = _LAZY.get("scipy.optimize")
    if sp is None:
        sp = _timed_import("scipy.optimize")
    if sp is None or not _NUMPY_OK:
        emit("milp", error="scipy.optimize unavailable")
        return
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
    except Exception as exc:
        emit("milp", error=repr(exc)[:300])
        return

    rows = []
    for n_items, n_groups in ((25, 4), (50, 8), (100, 8), (200, 8)):
        nv = n_items * n_groups
        try:
            c, w = _milp_problem(n_items, n_groups)
            # one action per item
            r_idx, c_idx = [], []
            for i in range(n_items):
                for g in range(n_groups):
                    r_idx.append(i)
                    c_idx.append(i * n_groups + g)
            A_one = csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)),
                               shape=(n_items, nv))
            budget = csr_matrix(w.reshape(1, -1))
            cons = [LinearConstraint(A_one, lb=0, ub=1),
                    LinearConstraint(budget, lb=0, ub=0.35 * w.sum())]
            t0 = time.perf_counter()
            res = milp(c=c, constraints=cons,
                       integrality=np.ones(nv),
                       bounds=Bounds(0, 1),
                       options={"time_limit": 0.45, "presolve": True})
            dt = time.perf_counter() - t0
            rows.append({"vars": nv, "rows": n_items + 1,
                         "seconds": round(dt, 4),
                         "status": int(getattr(res, "status", -1)),
                         "success": bool(getattr(res, "success", False))})
        except Exception as exc:
            rows.append({"vars": nv, "error": repr(exc)[:200]})
            break
        if dt > 0.45:
            break  # do not push the next size into the bank
    emit("milp", rows=rows)


# --------------------------------------------------------------------------- #
# the same MLP, three ways. Identical shapes so the numbers are comparable.
# --------------------------------------------------------------------------- #
NN_SHAPE = {"batch": 64, "din": 256, "h": 512, "dout": 64}


def _nn_flops():
    s = NN_SHAPE
    macs = s["din"] * s["h"] + s["h"] * s["h"] + s["h"] * s["dout"]
    return 2.0 * macs * s["batch"]


def task_nn_numpy():
    if not _NUMPY_OK:
        return
    s = NN_SHAPE
    rng = np.random.default_rng(1)
    x = rng.standard_normal((s["batch"], s["din"])).astype(np.float32)
    w1 = rng.standard_normal((s["din"], s["h"])).astype(np.float32) * 0.05
    w2 = rng.standard_normal((s["h"], s["h"])).astype(np.float32) * 0.05
    w3 = rng.standard_normal((s["h"], s["dout"])).astype(np.float32) * 0.05

    def fwd():
        h = np.maximum(x @ w1, 0.0)
        h = np.maximum(h @ w2, 0.0)
        return h @ w3

    rows = []
    for n_threads in (1, 2, 4):
        with blas_threads(n_threads) as bt:
            fwd()
            wall, cores, reps = cpu_ratio(fwd, 0.08)
        rows.append({"threads": n_threads, "applied": bt.effective,
                     "us_per_fwd": round(wall / reps * 1e6, 1),
                     "cores_used": round(cores, 3),
                     "gflops": round(_nn_flops() * reps / wall / 1e9, 3)})
    emit("nn_numpy", shape=NN_SHAPE, rows=rows)


def task_import_torch():
    _timed_import("torch")


def task_nn_torch():
    torch = _LAZY.get("torch")
    if torch is None:
        emit("nn_torch", error="torch unavailable")
        return
    try:
        s = NN_SHAPE
        torch.manual_seed(1)
        x = torch.randn(s["batch"], s["din"])
        w1 = torch.randn(s["din"], s["h"]) * 0.05
        w2 = torch.randn(s["h"], s["h"]) * 0.05
        w3 = torch.randn(s["h"], s["dout"]) * 0.05

        def fwd():
            with torch.no_grad():
                h = torch.relu(x @ w1)
                h = torch.relu(h @ w2)
                return h @ w3

        rows = []
        default_threads = torch.get_num_threads()
        for n_threads in (1, 2, 4):
            torch.set_num_threads(n_threads)
            applied = torch.get_num_threads()   # read BACK, not the request
            fwd()
            wall, cores, reps = cpu_ratio(fwd, 0.08)
            rows.append({"threads_requested": n_threads,
                         "threads_readback": applied,
                         "us_per_fwd": round(wall / reps * 1e6, 1),
                         "cores_used": round(cores, 3),
                         "gflops": round(_nn_flops() * reps / wall / 1e9, 3)})
        torch.set_num_threads(default_threads)
        emit("nn_torch", shape=NN_SHAPE, default_threads=default_threads,
             interop=torch.get_num_interop_threads(), rows=rows)
    except Exception as exc:
        emit("nn_torch", error=repr(exc)[:300])


def task_import_jax():
    _timed_import("jax")


def task_nn_jax():
    jax = _LAZY.get("jax")
    if jax is None:
        emit("nn_jax", error="jax unavailable")
        return
    try:
        import jax.numpy as jnp
        s = NN_SHAPE
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (s["batch"], s["din"]))
        w1 = jax.random.normal(key, (s["din"], s["h"])) * 0.05
        w2 = jax.random.normal(key, (s["h"], s["h"])) * 0.05
        w3 = jax.random.normal(key, (s["h"], s["dout"])) * 0.05

        def fwd_raw(x, w1, w2, w3):
            h = jnp.maximum(x @ w1, 0.0)
            h = jnp.maximum(h @ w2, 0.0)
            return h @ w3

        t0 = time.perf_counter()
        fwd_jit = jax.jit(fwd_raw)
        fwd_jit(x, w1, w2, w3).block_until_ready()
        compile_s = time.perf_counter() - t0
        _LAZY["_jax_fwd"] = lambda: fwd_jit(x, w1, w2, w3).block_until_ready()
        emit("nn_jax_compile", shape=NN_SHAPE,
             devices=str(jax.devices())[:200],
             jit_compile_seconds=round(compile_s, 4),
             jax_default_threads=os.environ.get("XLA_FLAGS"))
    except Exception as exc:
        emit("nn_jax_compile", error=repr(exc)[:300])


def task_nn_jax_bench():
    fwd = _LAZY.get("_jax_fwd")
    if fwd is None:
        emit("nn_jax", error="jax not compiled (unavailable or compile failed)")
        return
    try:
        wall, cores, reps = cpu_ratio(fwd, 0.08)
        emit("nn_jax", shape=NN_SHAPE,
             us_per_fwd=round(wall / reps * 1e6, 1),
             cores_used=round(cores, 3),
             gflops=round(_nn_flops() * reps / wall / 1e9, 3))
    except Exception as exc:
        emit("nn_jax", error=repr(exc)[:300])


_CHILD_BENCH = (
    "import os,sys,json,time\n"
    "t0=time.perf_counter()\n"
    "import numpy as np\n"
    "imp=time.perf_counter()-t0\n"
    "n=320\n"
    "a=np.random.rand(n,n);b=np.random.rand(n,n)\n"
    "a@b\n"
    "r=0;t0=time.perf_counter();c0=time.process_time()\n"
    "while time.perf_counter()-t0<0.25:\n"
    "    a@b; r+=1\n"
    "w=time.perf_counter()-t0; c=time.process_time()-c0\n"
    "print(json.dumps({'import_s':round(imp,4),'reps':r,'wall':round(w,4),"
    "'cores_used':round(c/w,3),'gflops':round(2.0*n**3*r/w/1e9,3),"
    "'omp':os.environ.get('OMP_NUM_THREADS')}))\n"
)


_CHILD_ROWS = []


def _make_thread_child(n_threads, last=False):
    """One child per turn. The only thread sweep that always works.

    In-process limiters need threadpoolctl or a reachable BLAS symbol. A child
    that sets OMP_NUM_THREADS *before* importing numpy needs neither - the
    variable is read at library load, which is the one moment it is honoured.
    It also prices the child itself, which is what any multi-process plan pays.
    """
    def task():
        import subprocess
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = str(n_threads)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, "-c", _CHILD_BENCH], env=env,
                               capture_output=True, text=True, timeout=8)
            data = json.loads(p.stdout.strip().splitlines()[-1])
            data["child_total_s"] = round(time.perf_counter() - t0, 4)
            data["threads_env"] = n_threads
        except Exception as exc:
            data = {"threads_env": n_threads, "error": repr(exc)[:200],
                    "stderr": (locals().get("p").stderr[-200:]
                               if locals().get("p") is not None else None),
                    "child_total_s": round(time.perf_counter() - t0, 4)}
        _CHILD_ROWS.append(data)
        if last:
            base = next((r.get("gflops") for r in _CHILD_ROWS if r.get("gflops")), None)
            for r in _CHILD_ROWS:
                if r.get("gflops") and base:
                    r["speedup_vs_1"] = round(r["gflops"] / base, 3)
            emit("thread_sweep_subprocess", rows=_CHILD_ROWS)
    return task


def task_multiprocessing():
    """Does a second PROCESS buy throughput, or does the cgroup quota just
    split the same CPU-seconds two ways?"""
    try:
        import multiprocessing as mp
    except Exception as exc:
        emit("multiprocessing", error=repr(exc)[:200])
        return
    try:
        t0 = time.perf_counter()
        ctx = mp.get_context("fork")
        p = ctx.Process(target=_mp_noop)
        p.start()
        p.join(5)
        fork_s = time.perf_counter() - t0
        emit("multiprocessing", start_method="fork",
             spawn_one_process_seconds=round(fork_s, 4),
             cpu_count=mp.cpu_count(), alive_ok=(p.exitcode == 0))
    except Exception as exc:
        emit("multiprocessing", error=repr(exc)[:300])


def _mp_noop():
    return None


def task_memory():
    if not _NUMPY_OK:
        return
    peak = None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux
    except Exception:
        pass
    # A cgroup OOM kills the process; it does not raise MemoryError. So never
    # probe toward the ceiling - cap at a quarter of the stated limit and stop.
    ceiling_mb = 384
    try:
        raw = _read("/sys/fs/cgroup/memory.max") or ""
        if raw.isdigit():
            ceiling_mb = min(ceiling_mb, int(raw) // (1024 * 1024) // 4)
    except Exception:
        pass
    # Touching 384 MB of fresh pages cost 2.2 s in the local dry run - over the
    # free turn. Allocation is not the question here; the ceiling is read from
    # the cgroup. So allocate a token amount only, to prove it works at all.
    blocks, ok_mb = [], 0
    try:
        for mb in (16, 32):
            if ok_mb + mb > ceiling_mb:
                break
            blocks.append(np.empty(mb * 1024 * 1024 // 8, dtype=np.float64))
            blocks[-1][::65536] = 1.0
            ok_mb += mb
    except MemoryError:
        pass
    except Exception:
        pass
    finally:
        del blocks
        gc.collect()
    emit("memory", allocated_ok_mb=ok_mb, peak_rss_kib=peak,
         cgroup_max=_read("/sys/fs/cgroup/memory.max"),
         meminfo_total=(_read("/proc/meminfo", 200) or "").split("\n")[0])


# --------------------------------------------------------------------------- #
# the billing probe: overrun by a known amount, read the bank back next turn.
# This is the only way to learn what an overrun really costs on THIS machine.
# --------------------------------------------------------------------------- #
_BILLING = {"pending": None, "rows": []}


def _make_overrun(extra):
    def task():
        bank_before = _STATE.get("bank")
        target = 1.0 + extra
        t0 = time.perf_counter()
        # burn, do not sleep: sleeping may be billed differently from working
        x = 0.0
        while time.perf_counter() - t0 < target:
            x += 1.0
        _BILLING["pending"] = {"requested_self_s": round(target, 4),
                               "measured_self_s": round(time.perf_counter() - t0, 4),
                               "bank_before": bank_before,
                               "burn": x}
    return task


def _settle_billing(bank_now, step):
    p = _BILLING.pop("pending", None)
    _BILLING["pending"] = None
    if not p or bank_now is None or p["bank_before"] is None:
        return
    billed = p["bank_before"] - bank_now
    over = p["measured_self_s"] - 1.0
    row = {"step": step,
           "self_s": p["measured_self_s"],
           "over_1s": round(over, 4),
           "bank_before": p["bank_before"],
           "bank_after": bank_now,
           "billed": round(billed, 4),
           "harness_surcharge_ms": round((billed - max(0.0, over)) * 1000, 1)}
    _BILLING["rows"].append(row)
    emit("billing", **row)


# --------------------------------------------------------------------------- #
# drift: the same reference bench, every DRIFT_EVERY steps, all season.
# --------------------------------------------------------------------------- #
_DRIFT = []


def task_drift():
    if not _NUMPY_OK:
        return
    run = _matmul_bench(256)
    run()
    wall, cores, reps = cpu_ratio(run, 0.05)
    g = 2.0 * 256 ** 3 * reps / wall / 1e9
    _DRIFT.append(round(g, 3))
    emit("drift", seat=_STATE.get("seat"), step=_STATE.get("step"), gflops=round(g, 3),
         cores_used=round(cores, 3), bank=_STATE.get("bank"))


# --------------------------------------------------------------------------- #
# the queue
# --------------------------------------------------------------------------- #
TASKS = [
    ("machine_census", task_machine_census),
    ("package_census", task_package_census),
    ("cpu_capacity", task_cpu_capacity),
    ("matmul_sizes", task_matmul_sizes),
    ("nn_numpy", task_nn_numpy),
    # the sweep that works with no threadpoolctl and no reachable BLAS symbol
    ("thread_child_1", _make_thread_child(1)),
    ("thread_child_2", _make_thread_child(2)),
    ("thread_child_4", _make_thread_child(4, last=True)),
    ("import_scipy", task_import_scipy),
    ("milp_sweep", task_milp_sweep),
    ("import_torch", task_import_torch),
    ("nn_torch", task_nn_torch),
    ("import_jax", task_import_jax),
    ("nn_jax_compile", task_nn_jax),
    ("nn_jax_bench", task_nn_jax_bench),
    ("multiprocessing", task_multiprocessing),
    ("memory", task_memory),
    ("cpu_capacity_again", task_cpu_capacity),   # after the heavy imports
    ("billing_+0.10s", _make_overrun(0.10)),
    ("billing_+0.30s", _make_overrun(0.30)),
    ("billing_+0.60s", _make_overrun(0.60)),
]

# Seat 1 runs a short census only: enough to confirm both seats see the same
# machine and to give a second, concurrent sample of it. Its turns are then
# spent making noise for seat 0 to measure against, and finally on the ladder.
SEAT1_TASKS = [
    ("machine_census", task_machine_census),
    ("cpu_capacity", task_cpu_capacity),
]


def _new_seat_state(seat):
    return {
        "seat": seat, "n": 0, "step": None, "bank": None,
        "first_call_wall": None, "self_ms": [], "summarised": False,
        "config_seen": None, "opp_last_money": None, "opp_frozen_since": None,
        "queue": list(TASKS if seat == 0 else SEAT1_TASKS),
    }


# State is per seat, because in the validation episode BOTH seats are us.
# Locally the harness calls seat 0 then seat 1 in the same process, so one
# shared module would have them overwrite each other; on Kaggle they are
# separate processes and the dict is simply never shared. Keying by seat is
# correct in both worlds. `_STATE` is rebound to the calling seat on entry -
# safe because the two seats are never inside agent() at the same time.
_SEATS = {}
_STATE = _new_seat_state(0)


def _bind_seat(seat):
    global _STATE
    if seat not in _SEATS:
        _SEATS[seat] = _new_seat_state(seat)
    _STATE = _SEATS[seat]
    return _STATE


# --------------------------------------------------------------------------- #
# CONTENTION - the measurement that needs two seats.
#
# On Kaggle both agents think inside the same turn. If they share a CPU quota,
# the compute we actually get depends on whether the OPPONENT is thinking too -
# and nothing a single-seat probe does can reveal that. Here seat 1 starts a
# background burner and returns immediately (so it never overruns its own
# budget), while seat 0 runs the reference bench. Both seats know `step`, so
# they agree on the schedule without exchanging a word.
#
# A burn that uses numpy releases the GIL, so it is a real CPU competitor
# whether the seats share a process (local) or not (Kaggle).
# --------------------------------------------------------------------------- #
# Two facts drive this schedule, and the first version got both wrong:
#
#   1. WITHIN a step, seat 0 is called BEFORE seat 1 (measured: the harness
#      walks the seats in order). So a burner seat 1 starts at step N cannot
#      affect seat 0's measurement at step N - only at N+1 and after. The first
#      version measured "contended" at the same step the burn started, i.e.
#      always just before it.
#   2. A PASS turn costs microseconds, so STEP distance is not WALL distance.
#      Five steps after a 2.5 s burn is still deep inside that burn. The first
#      version's "quiet" samples were the contended ones and vice versa - which
#      is why it reported the opponent's load making us 2.7x FASTER.
#
# So: the quiet block runs first and completely, before any burner has ever
# existed; then ONE burn covers the whole contended block, which is measured on
# the steps after it starts.
QUIET_FIRST = 150           # quiet block: nothing has burned yet, ever
CONTENTION_WINDOWS = 6
BURN_AT = 199               # seat 1 starts one burn here
CONTENTION_FIRST = 200      # seat 0 measures here onward, inside that burn
CONTENTION_BURN_S = 3.0     # must outlast the whole contended block
CONTENTION_SECONDS = 0.20   # per-window measurement: long enough to be stable

_CONTENTION = {"contended": [], "quiet": []}


def _quiet_steps():
    return {QUIET_FIRST + i for i in range(CONTENTION_WINDOWS)}


def _contended_steps():
    return {CONTENTION_FIRST + i for i in range(CONTENTION_WINDOWS)}


def _start_burner(seconds=CONTENTION_BURN_S):
    """Seat 1: burn CPU in the background, return this turn immediately."""
    import threading

    def burn():
        try:
            if not _NUMPY_OK:
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < seconds:
                    pass
                return
            a = np.random.rand(256, 256)
            b = np.random.rand(256, 256)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                a @ b
        except Exception:
            pass

    t = threading.Thread(target=burn, daemon=True)
    t.start()
    return t


def _measure_window(kind):
    """Seat 0: the reference bench, labelled by whether the opponent is busy."""
    if not _NUMPY_OK:
        return
    run = _matmul_bench(256)
    run()
    wall, cores, reps = cpu_ratio(run, CONTENTION_SECONDS)
    g = 2.0 * 256 ** 3 * reps / wall / 1e9
    _CONTENTION[kind].append(round(g, 3))
    emit("contention", kind=kind, step=_STATE.get("step"),
         gflops=round(g, 3), cores_used=round(cores, 3))


def _contention_verdict():
    c, q = _CONTENTION["contended"], _CONTENTION["quiet"]
    if not c or not q:
        emit("contention_summary", error="one side never sampled",
             contended=c, quiet=q)
        return
    mc, mq = sum(c) / len(c), sum(q) / len(q)
    ratio = round(mc / mq, 3) if mq else None
    emit("contention_summary",
         contended_gflops=round(mc, 3), quiet_gflops=round(mq, 3),
         contended_samples=c, quiet_samples=q,
         ratio=ratio,
         reading=("ratio ~1.0: the seats do not compete - either the harness "
                  "runs them sequentially or the quota is not shared. "
                  "ratio well below 1.0: the opponent's thinking costs us "
                  "compute, and the per-turn budget must be planned for the "
                  "contended case, not the idle one. "
                  "ratio ABOVE ~1.1 means this measurement is broken, not that "
                  "load made us faster - distrust it and check the schedule."))


# --------------------------------------------------------------------------- #
# Can we see the opponent die? Directly useful to #16: if an opponent errors or
# times out, our agent should notice and play the rest of the season unopposed.
# --------------------------------------------------------------------------- #
# Seat 1 dies at a KNOWN step - both seats run this same file, so seat 0 can
# read the ladder schedule and snapshot the opponent before and after.
#
# The first version watched for "opponent's money stopped changing" and called
# that death. In self-play both seats PASS, so the money is frozen from step 1
# and it fired at step 26 on a perfectly healthy opponent. That false positive
# is itself the finding: an idle opponent and a dead one look identical from
# inside the observation. So this version stops guessing and runs the actual
# experiment - snapshot, compare, and report what did or did not change.
def _opp_digest(obs):
    try:
        player = obs.get("player")
        farms = obs.get("farms") or []
        if not isinstance(player, int) or len(farms) != 2:
            return None
        f = farms[1 - player]
        tiles = f.get("tiles") or []
        planted = sum(1 for row in tiles for t in row
                      if isinstance(t, dict) and t.get("plant"))
        return {"money": f.get("money"), "planted_tiles": planted,
                "hands": len(f.get("hands") or []),
                "hires_today": f.get("hires_today"),
                "quadrants": sorted(f.get("unlocked_quadrants") or []),
                "farmer": str(f.get("farmer"))[:80]}
    except Exception:
        return None


def _opp_snapshot_steps():
    """Around the step seat 1 is scheduled to raise (rung index 11)."""
    death = LADDER_START + 11 * LADDER_GAP
    return {death - 10: "before", death + 20: "after",
            death + 150: "later", QUIET_AFTER - 5: "end"}


def _watch_opponent(obs, step):
    if _STATE.get("seat") != 0 or step is None:
        return
    try:
        label = _opp_snapshot_steps().get(step)
        if label is None:
            return
        digest = _opp_digest(obs)
        _STATE.setdefault("opp_snaps", {})[label] = digest
        emit("opponent_snapshot", step=step, when=label, digest=digest)
        if label == "end":
            snaps = _STATE.get("opp_snaps", {})
            before, after = snaps.get("before"), snaps.get("after")
            changed = None
            if before and after:
                changed = sorted(k for k in before if before[k] != after[k])
            emit("opponent_death_visibility",
                 snapshots=snaps, fields_that_changed_across_the_death=changed,
                 note=("empty list = an opponent that ERRORed is "
                       "indistinguishable from one that is merely idle, using "
                       "only the observation. If so, our agent cannot detect a "
                       "dead opponent directly and #16 must infer it from mass "
                       "balance - the opponent's sales - rather than from any "
                       "status field, because there is none."))
    except Exception:
        pass


def _summary(final=True):
    """Rolling summaries matter: if the episode dies early we still have one."""
    if final:
        if _STATE["summarised"]:
            return
        _STATE["summarised"] = True
    s = sorted(_STATE["self_ms"])
    pick = lambda q: (s[min(len(s) - 1, int(q * len(s)))] if s else None)  # noqa: E731
    emit("summary" if final else "summary_partial",
         seat=_STATE.get("seat"),
         turns_seen=_STATE["n"],
         last_step=_STATE["step"],
         bank_left=_STATE["bank"],
         self_ms_p50=pick(0.50), self_ms_p95=pick(0.95),
         self_ms_max=(s[-1] if s else None),
         drift_gflops=_DRIFT,
         drift_spread=(round(max(_DRIFT) - min(_DRIFT), 3) if _DRIFT else None),
         billing_rows=_BILLING["rows"],
         tasks_left=[t[0] for t in _STATE["queue"]],
         configuration=_STATE["config_seen"])


# --------------------------------------------------------------------------- #
# PHASE 2 - the error ladder. Deliberate misbehaviour, least destructive first.
#
# Each rung logs whether the harness CALLED US AGAIN afterwards, so the rung the
# log stops on is the answer. We learn everything up to and including the first
# fatal mode; if one is fatal, delete it from LADDER and resubmit to reach the
# rungs behind it.
#
# Ordering is by measured, not assumed, destructiveness:
#   - `raise` ended our participation in the local dry run, so everything we
#     actually expect to survive is scheduled before it;
#   - `nested_too_deep` at depth 200 raised RecursionError inside the harness's
#     own deepcopy and took the DRIVER process down, losing the log with it. It
#     is now depth 30 and sits behind `raise`;
#   - `burn_the_bank` is last because it is meant to end the episode.
# --------------------------------------------------------------------------- #

def mode_return_none():
    return None


def mode_return_string():
    return "PASS"


def mode_return_empty_dict():
    return {}


def mode_missing_keys():
    return {"farmer": ["PASS"]}                 # no 'hands', no 'market'


def mode_wrong_inner_types():
    return {"farmer": "PASS", "hands": 0, "market": None}


def mode_unknown_op():
    return {"farmer": ["TELEPORT"], "hands": [], "market": []}


def mode_known_op_wrong_arity():
    return {"farmer": ["PLANT"], "hands": [], "market": []}   # PLANT needs a crop


def mode_bad_crop_name():
    return {"farmer": ["PLANT", "UNOBTAINIUM"], "hands": [], "market": []}


def mode_burn_3s():
    """Overrun the free second by 2 s. Expected to be absorbed by the bank -
    which is exactly what we want confirmed on the grading machine."""
    t0 = time.perf_counter()
    x = 0.0
    while time.perf_counter() - t0 < 3.0:
        x += 1.0
    return dict(PASS_ACTION)


def mode_not_serialisable():
    """A set where JSON expects a list. Harmless in-process; on Kaggle the
    action crosses a JSON boundary, so this may behave completely differently."""
    return {"farmer": ["PASS"], "hands": [], "market": [{1, 2, 3}]}


def mode_raise():
    raise ValueError("deliberate probe failure: what does Kaggle do with this?")


def mode_nested_too_deep():
    d = {"farmer": ["PASS"], "hands": [], "market": []}
    cur = d
    for _ in range(30):
        cur["market"] = [{"x": cur.get("market")}]
        cur = cur["market"][0]
    return d


def mode_burn_the_bank():
    """70 s in one turn exceeds the whole 60 s bank. Last rung on purpose."""
    t0 = time.perf_counter()
    x = 0.0
    while time.perf_counter() - t0 < 70.0:
        x += 1.0
    return dict(PASS_ACTION)


LADDER = [
    ("return_none", mode_return_none),
    ("return_string", mode_return_string),
    ("return_empty_dict", mode_return_empty_dict),
    # ("missing_keys", mode_missing_keys),
    # ("wrong_inner_types", mode_wrong_inner_types),
    # ("unknown_op", mode_unknown_op),
    # ("known_op_wrong_arity", mode_known_op_wrong_arity),
    # ("bad_crop_name", mode_bad_crop_name),
    ("burn_3s", mode_burn_3s),
    # ("not_serialisable", mode_not_serialisable),
    # ("nested_too_deep", mode_nested_too_deep),   # depth 30 survived locally
    # `raise` ended our participation in the local dry run (status -> ERROR,
    # never called again), so everything we expect to survive is above it and
    # `burn_the_bank` below it will almost certainly NOT be reached this run.
    # That is a deliberate trade: one episode cannot test both an uncaught
    # exception and an exhausted bank. To test the bank instead, move
    # ("burn_the_bank", ...) above ("raise", ...) and resubmit.
    # ("raise", mode_raise),
    ("burn_the_bank", mode_burn_the_bank),
]

_LADDER = {"schedule": None, "pending": None, "complete": False}


def _farm_snapshot(obs):
    try:
        player = obs.get("player")
        farms = obs.get("farms") or []
        return {"step": obs.get("step"), "day": obs.get("day"), "hour": obs.get("hour"),
                "player": player,
                "money": (farms[player]["money"]
                          if isinstance(player, int) and player < len(farms) else None),
                "opp_money": (farms[1 - player]["money"]
                              if isinstance(player, int) and len(farms) == 2 else None),
                "bank": obs.get("remainingOverageTime")}
    except Exception as exc:
        return {"snapshot_error": repr(exc)[:200]}


def _arm_ladder(step):
    """Arm only in seat 1, and only once its own queue is empty.

    Seat 0 never breaks: it is the seat that carries the measurements home.
    """
    if not LADDER_ENABLED or _STATE.get("seat") != 1:
        return None
    if _LADDER["schedule"] is not None or step is None:
        return _LADDER["schedule"]
    if step < LADDER_START - 1 or _STATE["queue"]:
        return None
    base = step + 1
    _LADDER["schedule"] = {base + i * LADDER_GAP: rung
                           for i, rung in enumerate(LADDER)}
    _summary(final=True)          # everything we measured, on the wire, first
    emit("ladder_armed", first_rung_at=base, gap=LADDER_GAP,
         schedule={str(k): v[0] for k, v in sorted(_LADDER["schedule"].items())},
         note="from here on this agent misbehaves on purpose")
    return _LADDER["schedule"]


# --------------------------------------------------------------------------- #
# the agent. Must be the LAST function in this file.
# --------------------------------------------------------------------------- #
first_run = True
def agent(obs, config=None):
    try:
        global _GLOBAL_SHARE_VAR, first_run
        if first_run:
            print(_GLOBAL_SHARE_VAR)
            _GLOBAL_SHARE_VAR = obs.get("player")
            first_run = False
    except:
        print("\n_GLOBAL_SHARE_VAR\n")
    
    t_turn = time.perf_counter()

    try:
        step = obs.get("step") if hasattr(obs, "get") else getattr(obs, "step", None)
        bank = (obs.get("remainingOverageTime") if hasattr(obs, "get")
                else getattr(obs, "remainingOverageTime", None))
        seat = obs.get("player") if hasattr(obs, "get") else getattr(obs, "player", 0)
    except Exception:
        step, bank, seat = None, None, 0
    if seat not in (0, 1):
        seat = 0
    _bind_seat(seat)
    _STATE["n"] += 1
    _STATE["step"] = step
    _STATE["bank"] = bank

    if _STATE["first_call_wall"] is None:
        _STATE["first_call_wall"] = time.time()
        emit("boot", seat=seat,
             module_to_first_call_s=round(_STATE["first_call_wall"] - _WALL_MODULE_START, 4),
             step=step, bank=bank,
             role=("measure" if seat == 0 else "census + contention + ladder"),
             obs_keys=sorted(list(obs.keys()))[:20] if hasattr(obs, "keys") else None,
             config=(dict(config) if isinstance(config, dict) else str(config)[:300]))
        try:
            _STATE["config_seen"] = dict(config) if isinstance(config, dict) else None
        except Exception:
            pass

    _watch_opponent(obs, step)

    # settle any overrun from last turn before spending more
    try:
        _settle_billing(bank, step)
    except Exception:
        pass

    # ---------------------------------------------------------------- PHASE 2
    # Did we survive the previous rung? Being called at all is the evidence.
    #
    # The rung itself is SELECTED in this try block but CALLED outside it. An
    # earlier version called it inside, and the except below caught the
    # deliberate ValueError before the harness ever saw it - then logged
    # "SURVIVED raise", which was false. A guard that quietly voids a
    # measurement while still printing a confident line is the failure R005
    # exists to stop, so the call site is now where it has to be.
    rung_to_run = None
    try:
        pending = _LADDER["pending"]
        if pending:
            _LADDER["pending"] = None
            emit("SURVIVED", mode=pending["mode"], attempted_at=pending["step"],
                 now=_farm_snapshot(obs), turns_since=_STATE["n"] - pending["n"],
                 note="the harness kept calling us after this mode")

        schedule = _arm_ladder(step)
        if schedule:
            rung = schedule.get(step)
            if rung is not None:
                name, fn = rung
                emit("ATTEMPT", mode=name, at=_farm_snapshot(obs),
                     note="if the log ends here, this mode is fatal")
                _LADDER["pending"] = {"mode": name, "step": step, "n": _STATE["n"]}
                rung_to_run = fn
            elif (not _LADDER["complete"] and step is not None
                    and step > max(schedule) + 2):
                _LADDER["complete"] = True
                emit("LADDER_COMPLETE", at=_farm_snapshot(obs),
                     note="every mode survived; read the episode result to see "
                          "what each one cost in score")
    except Exception as exc:
        emit("ladder_error", step=step, error=repr(exc)[:300])

    if rung_to_run is not None:
        _STATE["self_ms"].append(round((time.perf_counter() - t_turn) * 1000.0, 3))
        return rung_to_run()      # outside the try: it MUST reach the harness

    try:
        quiet = (step is not None and step >= QUIET_AFTER)
        broke = (bank is not None and bank < BANK_FLOOR)
        if broke and _STATE["queue"]:
            emit("bank_floor", step=step, bank=bank,
                 dropped=[t[0] for t in _STATE["queue"]])
            _STATE["queue"] = []

        # Cooldown: after a turn that ran past the free second, do nothing this
        # turn. It lets the bank settle, and it keeps the billing readout clean
        # - the turn that reads the bank back must not be spending it too.
        prev = _STATE["self_ms"][-1] if _STATE["self_ms"] else 0.0
        cooling = prev > 900.0

        if not quiet and not broke and not cooling:
            if _STATE["queue"]:
                name, fn = _STATE["queue"].pop(0)
                t0 = time.perf_counter()
                try:
                    fn()
                except Exception as exc:
                    emit("task_failed", task=name, error=repr(exc)[:300])
                emit("task_done", seat=seat, task=name, step=step,
                     seconds=round(time.perf_counter() - t0, 4), bank=bank)
            elif seat == 1 and step == BURN_AT:
                # one burn covering the whole contended block; return at once so
                # this seat's own turn budget stays clean
                _start_burner()
                emit("burner_started", step=step, seconds=CONTENTION_BURN_S,
                     covers_steps=sorted(_contended_steps()))
            elif seat == 0 and step in _contended_steps():
                _measure_window("contended")
            elif seat == 0 and step in _quiet_steps():
                _measure_window("quiet")
            elif step is not None and step % DRIFT_EVERY == 0 and step > 0:
                task_drift()

        if seat == 0 and step == CONTENTION_FIRST + CONTENTION_WINDOWS:
            _contention_verdict()

        # Seat 0 publishes everything BEFORE seat 1 starts breaking things, so
        # an episode aborted by the ladder still carries the measurements.
        if seat == 0 and step == SUMMARY_AT and not _STATE["summarised"]:
            _summary(final=True)
        elif step is not None and step >= QUIET_AFTER and not _STATE["summarised"]:
            _summary(final=True)
        elif seat == 0 and step == QUIET_AFTER:
            _summary(final=False)   # the whole season, after the ladder
        elif step is not None and step > 0 and step % 200 == 0:
            _summary(final=False)   # insurance against an episode that dies early
    except Exception as exc:
        emit("turn_failed", step=step, error=repr(exc)[:300])

    dt_ms = (time.perf_counter() - t_turn) * 1000.0
    _STATE["self_ms"].append(round(dt_ms, 3))
    if step is not None and step % 100 == 0:
        emit("heartbeat", seat=seat, step=step, bank=bank, self_ms=round(dt_ms, 2),
             queue=len(_STATE["queue"]))

    return dict(PASS_ACTION)
