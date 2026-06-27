#!/usr/bin/env python3
"""Aggregate Plan L council results into a comparison table + JSON.

Reads results/council_*.json (+ prior j1 baseline) and reports:
  - v2.0 push: each full_bilinear config vs paper 0.5970 and vs J-1 baseline 0.6350
  - seed band: full_bilinear seeds {1234,0,42}(+1 if present) mean+-std + significance vs paper
  - ablations under full_bilinear vs baseline (Fig.4 reproduce check)
  - v3.2 honest numbers (two_head vs softmax5)
Outputs results/council_summary.json and prints a table.
"""
import json, os, glob, math

PAPER_TOP1 = 0.5970
PAPER_AUC = 0.9669
J1_BASELINE = 0.6350  # Windows full_bilinear seed1234/K13/exist0.1

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def g(d, *keys):
    for k in keys:
        if d and k in d and d[k] is not None:
            return d[k]
    return None

def top1(d):  return g(d, "top1_f1", "Top-1 F1", "top1_F1")
def auc(d):   return g(d, "AUC", "auc")

def main():
    rows = []
    for p in sorted(glob.glob("results/council_*.json")):
        cid = os.path.basename(p)[len("council_"):-len(".json")]
        if cid in ("matrix", "matrix_wave2", "summary"):
            continue
        d = load(p)
        if not d:
            continue
        rows.append({"id": cid, "top1_f1": top1(d), "auc": auc(d),
                     "aupr": g(d, "AUPR"), "f1": g(d, "F1")})

    # --- v2.0 push table ---
    v2 = [r for r in rows if not r["id"].startswith("C") and "v32" not in r["id"]]
    print("="*92)
    print("PLAN L — v2.0 full_bilinear matrix (paper Top-1 F1=%.4f, J-1 baseline=%.4f)" % (PAPER_TOP1, J1_BASELINE))
    print("="*92)
    print(f"{'config':<24}{'Top-1 F1':>10}{'vs paper':>11}{'vs J-1':>10}{'AUC':>9}")
    best = None
    for r in sorted(v2, key=lambda x: (x["top1_f1"] or 0), reverse=True):
        t = r["top1_f1"]
        if t is None: continue
        dp = f"{(t-PAPER_TOP1)/PAPER_TOP1*100:+.1f}%"
        dj = f"{(t-J1_BASELINE)/J1_BASELINE*100:+.1f}%"
        print(f"{r['id']:<24}{t:>10.4f}{dp:>11}{dj:>10}{(r['auc'] or 0):>9.4f}")
        if best is None or t > best["top1_f1"]:
            best = r

    # --- seed band (significance) ---
    seed_ids = {"R0_baseline_fb", "A4_ens_fb_s0", "A5_ens_fb_s42"}
    seeds = [r["top1_f1"] for r in v2 if r["id"] in seed_ids and r["top1_f1"] is not None]
    band = None
    if len(seeds) >= 2:
        mean = sum(seeds)/len(seeds)
        std = (sum((x-mean)**2 for x in seeds)/(len(seeds)-1))**0.5 if len(seeds) > 1 else 0.0
        # one-sample t vs paper
        t_stat = (mean - PAPER_TOP1)/(std/math.sqrt(len(seeds))) if std > 0 else float('inf')
        band = {"n": len(seeds), "seeds_top1": seeds, "mean": round(mean,4),
                "std": round(std,4), "t_vs_paper": round(t_stat,2)}
        print("-"*92)
        print(f"SEED BAND (full_bilinear baseline config, n={len(seeds)}): "
              f"mean={mean:.4f} +- {std:.4f}  | t vs paper={t_stat:.2f} "
              f"({'SIGNIFICANT' if abs(t_stat)>2 else 'n.s.'})")

    # --- ablations under full_bilinear (Fig.4) ---
    base_t = next((r["top1_f1"] for r in v2 if r["id"] == "R0_baseline_fb"), None)
    abls = [r for r in v2 if "_abl_" in r["id"]]
    if base_t and abls:
        print("-"*92)
        print(f"FIG.4 ABLATIONS under full_bilinear (baseline Top-1={base_t:.4f}); paper: all should HURT")
        for r in sorted(abls, key=lambda x: x["id"]):
            t = r["top1_f1"]
            if t is None: continue
            delta = (t-base_t)/base_t*100
            verdict = "HURT(paper-consistent)" if delta < -0.5 else ("HELP(reversal)" if delta > 0.5 else "~flat")
            print(f"  {r['id']:<22}{t:>9.4f}  delta={delta:+6.1f}%  {verdict}")

    # --- v3.2 honest ---
    v32 = [r for r in rows if r["id"].startswith("C") or "v32" in r["id"]]
    # also pull the repro_v32 file if present
    for extra in ["results/repro_v32_honest_linux.json"]:
        d = load(extra)
        if d:
            v32.append({"id": "R2_v32_two_head", "top1_f1": top1(d), "auc": auc(d)})
    if v32:
        print("-"*92)
        print("v3.2 HONEST (corrected metric; paper 0.86 needs unreleased 411x271 data)")
        for r in v32:
            print(f"  {r['id']:<24}{(r['top1_f1'] or 0):>9.4f}  AUC={(r['auc'] or 0):.4f}")

    out = {"v2_rows": v2, "best_v2": best, "seed_band": band,
           "paper_top1": PAPER_TOP1, "j1_baseline": J1_BASELINE,
           "v32_rows": v32}
    json.dump(out, open("results/council_summary.json", "w"), indent=2, ensure_ascii=False)
    print("="*92)
    if best:
        print(f"BEST v2.0: {best['id']} Top-1 F1={best['top1_f1']:.4f} "
              f"({(best['top1_f1']-J1_BASELINE)/J1_BASELINE*100:+.1f}% vs J-1, "
              f"{(best['top1_f1']-PAPER_TOP1)/PAPER_TOP1*100:+.1f}% vs paper)")
    print("Saved results/council_summary.json")

if __name__ == "__main__":
    main()
