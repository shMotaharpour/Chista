import importlib.util
spec = importlib.util.spec_from_file_location("chista_agent", "agent/main.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from kaggle_environments import make
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1})
buys = []
def ag(obs):
    a = mod.agent(obs)
    for o in a.get("market", []):
        if o[0] in ("HIRE", "BUY_LAND", "BUY_ANIMAL", "BUY_SEED"):
            buys.append((obs.get("day"), o))
    return a
env.run([ag, "random"])
print("buys days 0-6:")
for b in buys:
    if b[0] <= 6:
        print("  ", b)
print("final:", env.steps[-1][0].observation.farms[0]["money"])
