import re, glob, os, json, statistics as st
PAPER = {"Top-1 Precision":0.5842,"Top-1 Recall":0.6341,"Top-1 F1":0.5970,"AUPR":0.9738,"AUC":0.9669,"F1":0.9278}
rows = {}
for d in sorted(glob.glob(os.path.expanduser("~/dhg_runs/*/run.log"))):
    name = d.split("/")[-2]; t = open(d, errors="ignore").read()
    if "FINAL COMPREHENSIVE" not in t: continue
    blk = t.split("FINAL COMPREHENSIVE")[1]
    g = lambda k: float(re.search(r"\n\s+" + re.escape(k) + r": ([0-9.]+)", blk).group(1))
    rows[name] = {k: g(k) for k in ["AUC","AUPR","F1","Top-1 Precision","Top-1 Recall","Top-1 F1"]}
    rows[name]["folds_top1"] = [float(x) for x in re.findall(r"Fold \d completed - AUC: [0-9.]+, Top-1 F1: ([0-9.]+)", t)]
json.dump(rows, open(os.path.expanduser("~/dhg_runs/summary.json"),"w"), indent=1)
ks = ["AUC","AUPR","F1","Top-1 Precision","Top-1 Recall","Top-1 F1"]
print("run | " + " | ".join(ks)); print("paper | " + " | ".join(f"{PAPER[k]:.4f}" for k in ks))
for n,r in rows.items(): print(n + " | " + " | ".join(f"{r[k]:.4f}" for k in ks))
seeds = [rows[n] for n in ["def1","seed1","seed2","seed3"] if n in rows]
if len(seeds) > 1:
    print(f"default mean±sd (n={len(seeds)}) | " + " | ".join(f"{st.mean(r[k] for r in seeds):.4f}±{st.stdev(r[k] for r in seeds):.4f}" for k in ks))
