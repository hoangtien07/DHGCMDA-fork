#!/usr/bin/env python3
"""Conformal / uncertainty-aware miRNA-disease TYPE prediction (post-hoc).

Breakthrough direction (branch breakthrough-conformal). Reads per-fold held-out
predictions dumped by main_experiments_hetero1.py (--dump_scores DIR) and produces
distribution-free prediction SETS with a marginal coverage guarantee (APS / RAPS),
instead of a single Top-1 type. No retraining; pure numpy/scipy/sklearn.

Dump schema (one file per CV fold, written by the additive hook in test_optimized):
    fold{k}.npz  keys:
        mirna_idx    [N] int
        disease_idx  [N] int
        type_probs   [N, C] float   (per-type probabilities, columns = type 1..C)
        existence    [N]   float
        true_type    [N] int in {1..C}
        num_types    scalar int (C)

Validity note: each CV fold k comes from a DIFFERENT trained model M_k (trained on
the other folds). Split-conformal requires calibration & test from the SAME model,
so we split EACH fold's held-out set internally into calibration/test halves and
aggregate coverage across folds. This preserves exchangeability within each fold.
"""
import argparse
import glob
import json
import os

import numpy as np


# ----------------------------- data loading -----------------------------------
def load_folds(dump_dir):
    folds = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "fold*.npz"))):
        d = np.load(f)
        tp = d["type_probs"].astype(np.float64)
        yt = d["true_type"].astype(int)
        # keep only rows with a valid positive type label (1..C) and finite probs
        C = int(d["num_types"]) if "num_types" in d else tp.shape[1]
        mask = (yt >= 1) & (yt <= C) & np.isfinite(tp).all(axis=1) & (tp.sum(axis=1) > 0)
        tp = tp[mask]
        yt = yt[mask]
        # renormalise to a proper simplex (defensive)
        tp = tp / tp.sum(axis=1, keepdims=True)
        folds.append({"probs": tp, "y": yt - 1, "C": C, "file": os.path.basename(f)})
    if not folds:
        raise SystemExit(f"[ERR] no fold*.npz found in {dump_dir}")
    return folds


# ----------------------------- conformal core ---------------------------------
def aps_scores(probs, labels, rng, randomize=True, lam=0.0, k_reg=1):
    """APS non-conformity score for the TRUE label of each row (Romano et al. 2020).
    E_i = cumulative prob mass of classes ranked >= true class (more-probable-first),
    with optional uniform randomisation of the true class's own mass.
    RAPS (lam>0): + lam*max(0, rank_true - k_reg). The SAME reg is applied here
    (calibration) and in aps_predict_sets (prediction) so coverage holds."""
    n = probs.shape[0]
    order = np.argsort(-probs, axis=1)               # most->least probable
    sorted_p = np.take_along_axis(probs, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    # rank (0-based) of the true label within the sorted order
    rank_true = np.argmax(order == labels[:, None], axis=1)
    cum_incl = csum[np.arange(n), rank_true]         # mass up to & incl. true class
    p_true = probs[np.arange(n), labels]
    reg = lam * np.maximum(0, (rank_true + 1) - k_reg)   # 1-based rank penalty
    if randomize:
        u = rng.uniform(size=n)
        scores = cum_incl - u * p_true + reg
    else:
        scores = cum_incl + reg
    return scores


def aps_predict_sets(probs, tau, rng, randomize=True, lam=0.0, k_reg=1):
    """Build APS/RAPS prediction sets given calibrated threshold tau.
    lam>0 => RAPS regularisation penalising classes beyond rank k_reg."""
    n, C = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    reg = lam * np.maximum(0, np.arange(1, C + 1) - k_reg)[None, :]  # RAPS penalty
    csum_reg = csum + reg
    sets = np.zeros((n, C), dtype=bool)
    for i in range(n):
        if randomize:
            # include classes until reg-cumsum reaches tau; randomise last inclusion
            incl = 0
            acc = 0.0
            for r in range(C):
                acc = csum_reg[i, r]
                incl = r + 1
                if acc >= tau:
                    # randomised removal of the boundary class (APS exactness)
                    prev = csum_reg[i, r - 1] if r > 0 else 0.0
                    if (acc - tau) > rng.uniform() * (acc - prev + 1e-12):
                        incl = r  # drop boundary class
                    break
            sets[i, order[i, :incl]] = True
        else:
            incl = int(np.searchsorted(csum_reg[i], tau) + 1)
            incl = min(incl, C)
            sets[i, order[i, :incl]] = True
    return sets


def conformal_quantile(scores, alpha):
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return np.quantile(scores, level, method="higher")


# ----------------------------- evaluation -------------------------------------
def eval_sets(sets, labels, C):
    covered = sets[np.arange(len(labels)), labels]
    marginal = float(covered.mean())
    size = float(sets.sum(axis=1).mean())
    per_class = {}
    for c in range(C):
        m = labels == c
        if m.any():
            per_class[c + 1] = {
                "coverage": float(sets[m, c].mean()),
                "avg_size": float(sets[m].sum(axis=1).mean()),
                "n": int(m.sum()),
            }
    return {"marginal_coverage": marginal, "avg_set_size": size, "per_class": per_class}


def run_method(folds, alpha, seed, lam=0.0, k_reg=1, shuffle=False):
    rng = np.random.default_rng(seed)
    all_sets, all_labels = [], []
    C = folds[0]["C"]
    for fd in folds:
        probs, y = fd["probs"], fd["y"].copy()
        if shuffle:
            y = rng.permutation(y)              # negative control
        n = len(y)
        if n < 4:
            continue
        idx = rng.permutation(n)
        half = n // 2
        cal, tst = idx[:half], idx[half:]
        cal_scores = aps_scores(probs[cal], y[cal], rng, lam=lam, k_reg=k_reg)
        tau = conformal_quantile(cal_scores, alpha)
        sets = aps_predict_sets(probs[tst], tau, rng, lam=lam, k_reg=k_reg)
        all_sets.append(sets)
        all_labels.append(y[tst])
    sets = np.vstack(all_sets)
    labels = np.concatenate(all_labels)
    res = eval_sets(sets, labels, C)
    res["target_coverage"] = 1 - alpha
    res["n_test"] = int(len(labels))
    return res


def mondrian_thresholds(probs, labels, alpha, C, rng, lam=0.0, k_reg=1):
    """Class-conditional (Mondrian) thresholds: a separate tau per TRUE class so
    each class gets >= 1-alpha coverage (fixes marginal conformal hiding a
    minority-class collapse)."""
    taus = np.full(C, np.inf)
    # non-randomized score to match the deterministic inclusion rule in mondrian_predict
    scores = aps_scores(probs, labels, rng, randomize=False, lam=lam, k_reg=k_reg)
    for c in range(C):
        m = labels == c
        if m.sum() >= 1:
            taus[c] = conformal_quantile(scores[m], alpha)
    return taus


def mondrian_predict(probs, taus, lam=0.0, k_reg=1):
    """Include class k iff its APS inclusion-score <= tau_k (deterministic)."""
    n, C = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    reg = lam * np.maximum(0, np.arange(1, C + 1) - k_reg)[None, :]
    csum_reg = csum + reg
    ranks = np.argsort(order, axis=1)                 # ranks[i,k] = position of class k
    incl_score = np.take_along_axis(csum_reg, ranks, axis=1)  # score to include class k
    sets = np.zeros((n, C), dtype=bool)
    for k in range(C):
        sets[:, k] = incl_score[:, k] <= taus[k]
    return sets


def run_mondrian(folds, alpha, seed, lam=0.0, k_reg=1):
    rng = np.random.default_rng(seed)
    all_sets, all_labels = [], []
    C = folds[0]["C"]
    for fd in folds:
        probs, y = fd["probs"], fd["y"].copy()
        n = len(y)
        if n < 4:
            continue
        idx = rng.permutation(n)
        half = n // 2
        cal, tst = idx[:half], idx[half:]
        taus = mondrian_thresholds(probs[cal], y[cal], alpha, C, rng, lam=lam, k_reg=k_reg)
        sets = mondrian_predict(probs[tst], taus, lam=lam, k_reg=k_reg)
        all_sets.append(sets)
        all_labels.append(y[tst])
    sets = np.vstack(all_sets)
    labels = np.concatenate(all_labels)
    res = eval_sets(sets, labels, C)
    res["target_coverage"] = 1 - alpha
    res["n_test"] = int(len(labels))
    return res


def top1_accuracy(folds):
    correct = tot = 0
    for fd in folds:
        pred = fd["probs"].argmax(axis=1)
        correct += int((pred == fd["y"]).sum())
        tot += len(fd["y"])
    return correct / max(tot, 1)


# ----------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.05])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--raps_lambda", type=float, default=0.05)
    ap.add_argument("--raps_kreg", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    folds = load_folds(args.dump_dir)
    C = folds[0]["C"]
    n_total = sum(len(f["y"]) for f in folds)
    print(f"Loaded {len(folds)} folds, {n_total} positive test pairs, C={C} types")
    print(f"Model Top-1 type accuracy (held-out): {top1_accuracy(folds):.4f}\n")

    report = {"dump_dir": args.dump_dir, "num_types": C, "n_total": n_total,
              "top1_accuracy": top1_accuracy(folds), "results": []}

    def pc(res):
        return ", ".join(f"T{k}:{v['coverage']:.2f}(n{v['n']})"
                         for k, v in res["per_class"].items())

    for alpha in args.alpha:
        aps = run_method(folds, alpha, args.seed)
        raps = run_method(folds, alpha, args.seed, lam=args.raps_lambda, k_reg=args.raps_kreg)
        mond = run_mondrian(folds, alpha, args.seed)          # class-conditional
        ctrl = run_method(folds, alpha, args.seed, shuffle=True)
        row = {"alpha": alpha, "target": 1 - alpha,
               "APS": aps, "RAPS": raps, "Mondrian": mond, "shuffle_control": ctrl}
        report["results"].append(row)
        print(f"alpha={alpha}  target coverage={1-alpha:.3f}")
        print(f"  APS      : cov={aps['marginal_coverage']:.4f}  size={aps['avg_set_size']:.3f}")
        print(f"  RAPS     : cov={raps['marginal_coverage']:.4f}  size={raps['avg_set_size']:.3f}"
              f"  (lambda={args.raps_lambda})")
        print(f"  Mondrian : cov={mond['marginal_coverage']:.4f}  size={mond['avg_set_size']:.3f}  (class-conditional)")
        print(f"  CTRL(shuf): cov={ctrl['marginal_coverage']:.4f}  size={ctrl['avg_set_size']:.3f}")
        print(f"  APS per-class      : {pc(aps)}")
        print(f"  Mondrian per-class : {pc(mond)}")
        print()

    out = args.out or os.path.join(args.dump_dir, "conformal_report.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
