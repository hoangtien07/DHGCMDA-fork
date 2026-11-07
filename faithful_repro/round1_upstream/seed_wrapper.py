# Runs the UNMODIFIED author script, only remapping its hardcoded global seed 1234 -> $DHG_SEED.
import os, sys, random, runpy, numpy as np, torch
sys.path.insert(0, os.getcwd())
S = int(os.environ["DHG_SEED"])
def wrap(f):
    def g(s=None, *a, **k):
        return f(S if s == 1234 else s, *a, **k)
    return g
random.seed = wrap(random.seed); np.random.seed = wrap(np.random.seed)
torch.manual_seed = wrap(torch.manual_seed); torch.cuda.manual_seed = wrap(torch.cuda.manual_seed)
torch.cuda.manual_seed_all = wrap(torch.cuda.manual_seed_all)
print(f"[seed_wrapper] global seed 1234 -> {S}", flush=True)
sys.argv = ["main_experiments_hetero1.py"] + sys.argv[1:]
runpy.run_path("main_experiments_hetero1.py", run_name="__main__")
