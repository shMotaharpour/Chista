import importlib.util
spec = importlib.util.spec_from_file_location("chista_agent", "agent/main.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from kaggle_environments import make
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1})
env.run([mod.agent, "random"])
for d in (1, 3, 8, 15, 20, 25, 29):
    step = min(d * 24, len(env.steps) - 1)
    o = env.steps[step][0].observation
    me = o.farms[0]
    plants = sum(1 for _, _, t in mod.tiles(o) if mod.is_plant(t))
    weeds = sum(1 for _, _, t in mod.tiles(o) if isinstance(t, dict) and t.get("kind") == "WEED")
    print(f"day{d}: money={me['money']:.0f} plants={plants} weeds={weeds} "
          f"hands={len(me['hands'])} unlocked={len(me['unlocked_quadrants'])} "
          f"shed_melon={o.private['shed'].get('MELON', 0)} "
          f"shed_carrot={o.private['shed'].get('CARROT', 0)}")
print("final:", env.steps[-1][0].observation.farms[0]["money"])
