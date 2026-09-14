"""Pull a competitor's agent out of a published Kaggle notebook, without running it.

roadmap 2.3. `random` and `starter` are trivial, so beating them says nothing;
the arena only starts meaning something once it has opponents that actually
play.

**Nothing in a notebook is executed.** Notebooks are arbitrary code published by
strangers, and running one to find out what it does is the wrong order. Every
cell is parsed with `ast` instead, string literals are resolved with
`literal_eval`, and the handful of decode calls competitors use to pack their
submission are applied explicitly. If a notebook packs its agent some way this
does not understand, it is skipped and said so -- never executed as a fallback.

Three packing styles cover every notebook seen so far:

  * **blob** -- `MAIN_BLOB = ("...")` then
    `zlib.decompress(base64.b85decode(MAIN_BLOB))`. Base85 or base64.
  * **writefile** -- a `%%writefile main.py` cell whose body is the source.
  * **inline** -- the notebook simply defines `def agent(obs)` at top level.

The extracted source is written verbatim to `opponents/<slug>/agent.py` and
hashed, so it stays evidence rather than a paraphrase -- nothing here reformats
it, and `tests/test_opponents.py` checks every payload against its recorded
SHA-256.
"""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

# Modules a farming agent has any business importing. Anything else is reported
# for a human to look at before the agent is ever called.
SAFE_IMPORTS = {
    "base64", "bisect", "collections", "copy", "dataclasses", "enum", "functools",
    "heapq", "itertools", "json", "math", "operator", "random", "re", "statistics",
    "struct", "sys", "types", "typing", "zlib", "array", "abc", "numbers",
    "fractions", "decimal", "string", "textwrap", "warnings", "contextlib",
    "kaggle_environments", "numpy", "__future__", "gzip", "binascii", "time",
}

# Calls that reach outside the process. A competitor's agent has no reason to
# make any of them, and the arena runs this code in our own interpreter.
FORBIDDEN_CALLS = {
    "open", "__import__", "eval", "input", "breakpoint", "compile",
}
FORBIDDEN_MODULES = {
    "subprocess", "socket", "urllib", "requests", "http", "ftplib", "smtplib",
    "shutil", "pathlib", "os", "pickle", "marshal", "ctypes", "multiprocessing",
    "importlib", "tempfile", "sqlite3", "asyncio", "threading",
}


@dataclass
class Extraction:
    slug: str
    notebook: Path
    source: str
    style: str
    sha256: str
    warnings: list[str] = field(default_factory=list)

    @property
    def lines(self) -> int:
        return self.source.count("\n") + 1

    @property
    def bytes(self) -> int:
        return len(self.source.encode("utf-8"))


class NotExtractable(Exception):
    pass


# --------------------------------------------------------------------------
# safe evaluation of the constants a notebook builds its payload from
# --------------------------------------------------------------------------
def _decode_chain(node: ast.AST, consts: dict[str, object]) -> str | None:
    """Resolve the decode expressions competitors actually use.

    Every packing style seen so far is some composition of: split the payload
    across a list of string literals, join it, base85- or base64-decode it,
    optionally zlib- or gzip-decompress it, and decode to text. Anything outside
    that vocabulary returns None rather than guessing -- the notebook is skipped,
    never executed to find out.

    Bytes are carried through as latin-1 text, which round-trips every byte
    value exactly, so `.encode("latin-1")` recovers the original payload.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        value = consts.get(node.id)
        return value if isinstance(value, str) else None
    if isinstance(node, ast.JoinedStr):
        return None
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in ("b85decode", "b64decode", "decompress", "decode", "encode", "join"):
            if name == "join" and isinstance(func, ast.Attribute):
                sep = func.value.value if isinstance(func.value, ast.Constant) else ""
                if not node.args:
                    return None
                arg = node.args[0]
                if isinstance(arg, (ast.List, ast.Tuple)):
                    parts = [
                        e.value for e in arg.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    ]
                    return sep.join(parts)
                # "".join(PARTS) where PARTS was assigned a list of literals
                if isinstance(arg, ast.Name):
                    value = consts.get(arg.id)
                    if isinstance(value, list):
                        return sep.join(value)
                return None
            # Where the payload lives depends on the call shape:
            # `base64.b85decode(X)` carries it in args[0], but `X.decode("utf-8")`
            # carries it in the receiver and args[0] is the *encoding*. Reading
            # args[0] for both silently resolves the whole chain to "utf-8".
            if name in ("decode", "encode"):
                inner = (
                    _decode_chain(func.value, consts)
                    if isinstance(func, ast.Attribute) else None
                )
            else:
                inner = _decode_chain(node.args[0], consts) if node.args else None
            if inner is None:
                return None
            try:
                if name == "b85decode":
                    return base64.b85decode(inner).decode("latin-1")
                if name == "b64decode":
                    return base64.b64decode(inner).decode("latin-1")
                if name == "decompress":
                    raw = inner.encode("latin-1")
                    mod = ""
                    if isinstance(func, ast.Attribute):
                        mod = getattr(func.value, "id", "")
                    out = gzip.decompress(raw) if mod == "gzip" else zlib.decompress(raw)
                    return out.decode("utf-8", "replace")
                if name in ("decode", "encode"):
                    return inner
            except Exception:
                return None
    return None


def _constants(code: str) -> dict[str, object]:
    """Module-level names bound to a string, or to a list of strings.

    Lists matter: several notebooks split the payload as
    `_AGENT_B85_PARTS = ["...", "..."]` and join it at the point of use.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    consts: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                consts[target.id] = value
                continue
            if isinstance(value, (list, tuple)) and value and all(
                isinstance(v, str) for v in value
            ):
                consts[target.id] = list(value)
                continue
        except (ValueError, SyntaxError, TypeError):
            pass
        decoded = _decode_chain(node.value, consts)
        if decoded is not None:
            consts[target.id] = decoded
    return consts


# --------------------------------------------------------------------------
# the three packing styles
# --------------------------------------------------------------------------
def _strip_magics(cell: str) -> str:
    return "\n".join(
        "" if ln.lstrip().startswith(("%", "!")) else ln for ln in cell.splitlines()
    )


def _looks_like_an_agent(source: str) -> bool:
    """A top-level `def agent(...)` that parses. That is the whole contract."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(n, ast.FunctionDef) and n.name == "agent" for n in tree.body
    )


def _from_writefile(cells: list[str]) -> str | None:
    for cell in cells:
        first = cell.lstrip().splitlines()[0] if cell.strip() else ""
        if first.startswith("%%writefile"):
            body = "\n".join(cell.splitlines()[1:])
            if _looks_like_an_agent(body):
                return body
    return None


def _from_blob(cells: list[str]) -> str | None:
    """The largest resolvable payload that parses as an agent.

    Both the named constants and every decode expression in the notebook are
    considered: several competitors never bind the decoded source to a name at
    all, going straight from the literal into `main_path.write_bytes(...)`.
    """
    code = "\n".join(_strip_magics(c) for c in cells)
    consts = _constants(code)
    candidates = [v for v in consts.values() if isinstance(v, str)]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                decoded = _decode_chain(node, consts)
                if decoded:
                    candidates.append(decoded)
    best = None
    for value in candidates:
        if len(value) > 500 and _looks_like_an_agent(value):
            if best is None or len(value) > len(best):
                best = value
    return best


def _from_inline(cells: list[str]) -> str | None:
    """Cells that define `agent` directly, concatenated in notebook order."""
    kept = [_strip_magics(c) for c in cells]
    joined = "\n\n".join(kept)
    if _looks_like_an_agent(joined):
        return joined
    for cell in reversed(kept):
        if _looks_like_an_agent(cell):
            return cell
    return None


def extract(notebook: Path, slug: str | None = None) -> Extraction:
    """Return the competitor's agent source, or raise NotExtractable."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    for style, fn in (("blob", _from_blob), ("writefile", _from_writefile),
                      ("inline", _from_inline)):
        source = fn(cells)
        if source:
            if style == "blob":
                # The decoded payload IS the published file. Do not normalise it
                # -- competitors publish its SHA-256, `SOURCE.md` records that
                # hash, and NOTICE.md's Apache 4(b) claim ("no changes") rests
                # on it. Stripping a trailing newline silently cost one byte and
                # broke all three at once.
                pass
            else:
                # Assembled from notebook cells, so the surrounding whitespace is
                # our own artefact rather than theirs.
                source = source.strip() + "\n"
            return Extraction(
                slug=slug or slugify(notebook),
                notebook=notebook,
                source=source,
                style=style,
                sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                warnings=audit(source),
            )
    raise NotExtractable(f"{notebook.name}: no cell yields a top-level `def agent`")


def slugify(notebook: Path) -> str:
    name = re.sub(r"\[[^\]]*\]", "", notebook.stem)
    name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"^kaggriculture-", "", name)


# --------------------------------------------------------------------------
# audit -- read before running, because the arena runs this in our interpreter
# --------------------------------------------------------------------------
def _self_bundled(tree: ast.AST) -> set[str]:
    """Module names the agent registers in `sys.modules` itself.

    Several competitors ship one flat file that rebuilds a package tree at
    import time -- `sys.modules["v44.gold_floor"] = ...` then
    `from v44.gold_floor import ...`. Those imports never leave the file, so
    flagging them as unknown third-party code is noise that hides the findings
    that matter.
    """
    bundled: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "modules"
                ):
                    bundled.add("*")  # dynamic name; treat the file as self-bundling
        if isinstance(node, ast.Call):
            fn = node.func
            if getattr(fn, "attr", "") == "ModuleType" or getattr(fn, "id", "") == "ModuleType":
                bundled.add("*")
    return bundled


def _audit_source(source: str, where: str = "",
                  bundling: bool | None = None) -> list[str]:
    """Audit ONE unit of source. `where` labels findings from a nested payload.

    `bundling` overrides the self-bundling detection. A packed payload is a
    module of a package its loader rebuilt at import, so it imports its own
    siblings by names that exist nowhere else -- `v23.state_encoder`,
    `scripts.v22_market_impact`. Detected in isolation it bundles nothing, and
    every one of those reads as unknown third-party code: seventeen such lines
    for one agent, which is how a real finding gets buried. The caller knows the
    payload came out of a self-bundling file and says so.
    """
    findings: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [f"{where}payload does not parse as Python"]
    if bundling is None:
        bundling = bool(_self_bundled(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if name in FORBIDDEN_CALLS:
                if name == "compile" and bundling:
                    # Its own module loader, not an escape hatch. Say so rather
                    # than staying silent: `audit` decodes and scans what it
                    # loads, and a suppressed finding that is never replaced by
                    # a real one is how an unaudited payload slips through.
                    findings.append(
                        f"{where}line {node.lineno}: compile() -- self-bundled "
                        "module loader, payload audited separately")
                    continue
                findings.append(f"{where}line {node.lineno}: calls {name}()")
            continue
        else:
            continue
        for full in names:
            top = full.split(".")[0]
            if top in FORBIDDEN_MODULES:
                findings.append(
                    f"{where}line {node.lineno}: imports {full}  << REACHES OUTSIDE")
            elif top and top not in SAFE_IMPORTS:
                if bundling:
                    continue  # resolved from the agent's own sys.modules entries
                findings.append(
                    f"{where}line {node.lineno}: imports {full} (unrecognised)")
    return findings


def _try_decode(text: str) -> str | None:
    """Decode one string literal if it is a packed payload, else None.

    Deliberately blunter than `_decode_chain`: it ignores how the file decodes
    the constant and just tries the ladder competitors use. That matters because
    several of them decode through a helper -- `_load_payload(_MOON_PAYLOAD)`
    -- where the call site's argument is a parameter name and resolving the
    chain yields nothing. An audit that only follows call sites reports those
    agents as carrying no payload at all.

    Guessing wrong is cheap here: a constant that decodes to something which is
    neither source nor a module map is dropped by `_nested_sources`. Guessing
    wrong in `_decode_chain` would corrupt an extraction, which is why that one
    stays strict and this one does not share its vocabulary.
    """
    if len(text) < 512:
        return None
    try:
        payload = text.encode("ascii")
    except UnicodeEncodeError:
        return None
    for decode in (base64.b85decode, base64.b64decode):
        try:
            raw = decode(payload)
        except Exception:  # noqa: BLE001 - any malformed literal is simply not a payload
            continue
        for decompress in (zlib.decompress, gzip.decompress, None):
            try:
                data = raw if decompress is None else decompress(raw)
                return data.decode("utf-8")
            except Exception:  # noqa: BLE001 - wrong guess, try the next rung
                continue
    return None


def _payloads(source: str) -> list[str]:
    """Every packed payload in one file, decoded in memory.

    Two ways in, because agents pack two ways: the decode chain at a call site
    (what `unpack` follows), and any module-level string constant that decodes
    on its own (what a helper-function loader leaves behind). Kept in memory
    rather than written out -- making the audit depend on a side effect on disk
    would let the audit and the thing audited drift apart.
    """
    out: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    consts = _constants(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            payload = _decode_chain(node, consts)
            if payload and len(payload) >= 512:
                out.append(payload)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            payload = _try_decode(node.value)
            if payload:
                out.append(payload)
    return out


def _nested_sources(payload: str) -> list[str]:
    """Python source carried inside one decoded payload.

    A payload is either module source itself, or a JSON map of module name to
    source -- the shape competitors use to rebuild a package tree from one file.
    Route tables and action schedules are data and carry no code to audit.
    """
    # JSON first, and not via `_classify`: a module MAP is a JSON object whose
    # values are source, and `ast.parse` accepts that object as a dict literal
    # while `"def "` appears inside one of its strings -- so `_classify` calls it
    # a module and the sources inside it are never scanned. That misread is the
    # whole reason five agents' payloads looked empty.
    if payload.lstrip()[:1] in "[{":
        try:
            obj = json.loads(payload)
        except ValueError:
            pass
        else:
            if isinstance(obj, dict):
                return [v for v in obj.values()
                        if isinstance(v, str) and _classify(v)[0] == "module"]
            return []  # a route table: data, no code to audit
    return [payload] if _classify(payload)[0] == "module" else []


def audit(source: str, depth: int = 3) -> list[str]:
    """Imports and calls worth a human's attention before this agent is played.

    Tuned to surface what actually matters -- anything reaching the network, the
    filesystem or a subprocess -- rather than every unfamiliar name.

    **Recurses into packed payloads.** Six of the vendored agents carry their
    real logic base85- or zlib-packed and `exec` it at import, so auditing only
    the outer file describes a loader, not an agent. Every payload is decoded
    (never executed), and any Python source inside it is audited too; findings
    from a payload are labelled with its digest. `depth` bounds payload-inside-
    payload nesting.
    """
    findings = _audit_source(source)
    for digest, kind, text in packed_payloads(source, depth):
        if kind == "module":
            findings += _audit_source(text, f"embedded {digest}: ", bundling=True)
    return findings


def packed_payloads(source: str, depth: int = 3) -> list[tuple[str, str, str]]:
    """(digest, kind, text) for every payload packed into this file.

    `kind` is `module` for Python source -- the payload itself, or each source
    inside a JSON module map -- and `table` for anything else, which in practice
    means a decoded route or action schedule. Only a `module` carries code for
    `audit` to read; a `table` is data, and calling it unaudited would be a false
    alarm on the ten agents whose entire payload is a precomputed route.
    """
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    frontier = [(source, 1)]
    while frontier:
        current, level = frontier.pop()
        if level > depth:
            continue
        for payload in _payloads(current):
            sources = _nested_sources(payload)
            kind = "module" if sources else "table"
            for text in sources or [payload]:
                digest = hashlib.sha256(
                    text.encode("utf-8", "replace")).hexdigest()[:12]
                if digest in seen:
                    continue
                seen.add(digest)
                found.append((digest, kind, text))
                if kind == "module":
                    frontier.append((text, level + 1))
    return found


def payload_digests(source: str, depth: int = 3) -> list[str]:
    """Digests of every payload decoded out of this file, code or data.

    Coverage, reported separately from findings, because "audited and clean" and
    "never decoded" both render as an empty finding list -- and for a third of
    the vendored agents the second was silently true until the decoder learned
    about JSON module maps and helper-function loaders.
    `tests/test_opponents.py` asserts this stays non-empty for the packed ones.
    """
    return [digest for digest, _, _ in packed_payloads(source, depth)]


# --------------------------------------------------------------------------
# unpacking -- a derived view, never the thing we play
# --------------------------------------------------------------------------
def unpack(agent_py: Path, out_dir: Path) -> list[tuple[str, int, str]]:
    """Decode every payload embedded in a vendored agent into readable files.

    `agent.py` stays byte-identical -- it is what the arena plays, and its
    SHA-256 in SOURCE.md is the guarantee that we are playing what was
    published. This writes a *separate* view beside it for reading.

    Most of what comes out is data, not code: precomputed route tables and
    action schedules. That is usually the more useful half. `v111-8c4s`, joint
    strongest agent in the field, is a 720-entry list of action dicts -- its
    entire strategy, in the open.

    Returns [(filename, bytes, kind)]. Output is regenerable and can be large
    (one agent unpacks to 7.6 MB), so it belongs in .gitignore, not in a commit.
    """
    import json as _json

    code = agent_py.read_text(encoding="utf-8", errors="replace")
    consts = _constants(code)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    seen: set[str] = set()
    written: list[tuple[str, int, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        payload = _decode_chain(node, consts)
        if not payload or len(payload) < 512:
            continue
        digest = hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:12]
        if digest in seen:
            continue
        seen.add(digest)
        kind, suffix = _classify(payload)
        if kind == "opaque":
            continue
        name = f"{kind}-{digest}{suffix}"
        text = payload
        if kind == "table":
            try:
                text = _json.dumps(_json.loads(payload), indent=1)
            except ValueError:
                pass
        (out_dir / name).write_text(text, encoding="utf-8")
        written.append((name, len(text), kind))
    return written


def _classify(payload: str) -> tuple[str, str]:
    import json as _json

    try:
        ast.parse(payload)
        if "def " in payload or "import " in payload:
            return "module", ".py"
    except SyntaxError:
        pass
    if payload.lstrip()[:1] in "[{":
        try:
            _json.loads(payload)
            return "table", ".json"
        except ValueError:
            pass
    return "opaque", ""
