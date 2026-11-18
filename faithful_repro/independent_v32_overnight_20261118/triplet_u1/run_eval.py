# -*- coding: utf-8 -*-
"""CVtriplet-U1 REV2 eval driver — scores pinned eval sets with the
trained U1X checkpoint (Task T triplet + Task P pair), runs the matched
train-only logistic baseline on the SAME replicates, stratifies, writes
eval JSON. Called by train.py after sanity; standalone via CLI."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OVER = HERE.parent
ROOT = OVER.parents[1]
PKG = ROOT / 'faithful_repro' / 'colab_package_f0_selection_20261115'
SRC = ROOT / 'faithful_repro' / 'paper_guided' / \
    'u1_horizon650_fold0_rev1' / 'source'
for _p in (str(SRC), str(SRC / 'stages'), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import split as S                                   # noqa: E402
import runner as s1r                                # noqa: E402
from eval_cvtriplet import (binary_metrics, joint_score,  # noqa: E402
                            pair_score, stratified_scores)
from model_u1x import PaperGuidedU1X                # noqa: E402
import baseline_logistic as BL                     # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
_arr = lambda a: hashlib.sha256(
    np.ascontiguousarray(a).tobytes()).hexdigest()
STRATA = ('neg_on_train_pair', 'neg_on_heldout_pair', 'neg_pair_zero')


def load_Y(strict_path, expect_sha):
    assert sha(strict_path) == expect_sha, 'strict npz pin'
    z = np.load(strict_path, allow_pickle=True)
    return np.asarray(z['Y']).astype(bool)


def eval_pinned_sets(Y, fold):
    """Returns (task_T_replicates, task_P_replicates, mask, seg)."""
    seg = S.fold_segments(Y, fold)
    train_pairs = S.pairs_of(seg['inner_train'])
    mask = S.pair_mask(*Y.shape[:2], train_pairs)
    test = seg['test']
    unk = S.unknown_universe(Y)
    t_reps = [(seed, test, S.eval_unknown_sample(unk, len(test), seed))
              for seed in S.EVAL_SEEDS]
    ppos = S.pair_unseen_test_pairs(Y, test)
    pz = S.pair_zero_index(Y)
    p_reps = [(seed, ppos,
               pz[np.random.default_rng(seed).choice(
                   len(pz), size=len(ppos), replace=False)])
              for seed in S.EVAL_SEEDS]
    return t_reps, p_reps, mask, seg


def model_scores(model, inputs, G, ei, pairs, torch):
    """exist probs + type probs for a list of (i,j) pairs."""
    pt = torch.tensor(np.asarray(pairs), dtype=torch.long)
    with torch.no_grad():
        out = model(inputs, G, ei, pt)
    probs = torch.softmax(out['logits'].cpu(), dim=-1).numpy()
    return out['exist'].cpu().numpy(), probs


def run_eval(bundle_dir, ckpt_path, strict_path, fold, out_path):
    import torch
    torch.set_num_threads(2)
    data = s1r.load_data_npz(str(Path(bundle_dir)
                               / 'candidate_fold_data.npz'))
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    views = data['views']
    inputs = {k: torch.tensor(np.asarray(views[k]),
                              dtype=torch.float32)
              for k in s1r.VIEW_KEYS}
    model = PaperGuidedU1X(
        {k: inputs[k].shape[1] for k in s1r.VIEW_KEYS},
        n_types=int(cfg['n_types']), hidden=int(cfg['hidden']),
        readout=int(cfg['readout']), proj=int(cfg['proj']),
        hgt_layers=int(cfg['hgt_layers']),
        hgt_heads=int(cfg['hgt_heads']),
        hgt_dropout=float(cfg['hgt_dropout']),
        readout_dropout=float(cfg['readout_dropout']),
        leaky_slope=float(cfg['leaky_slope']),
        temperature_init=float(cfg['temperature_init']),
        temperature_min=float(cfg['temperature_min']),
        temperature_max=float(cfg['temperature_max']),
        recon_enabled=(float(cfg['lam_recon']) != 0.0),
        recon_seed=int(cfg['seed']), exist_seed=int(cfg['seed']))
    model.load_state_dict(ck['model_state'])
    model.eval()
    G, ei = ck['G'], ck['ei']

    Y = load_Y(strict_path, json.loads(
        (PKG / 'package_manifest.json').read_text(
            encoding='utf-8'))['files'][
            'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'])
    t_reps, p_reps, mask, seg = eval_pinned_sets(Y, fold)

    # ---- Task T ----
    t_rows = []
    for seed, pos, neg in t_reps:
        ep = np.unique(np.concatenate([pos[:, :2], neg[:, :2]]), axis=0)
        row_of = {(i, j): r for r, (i, j) in enumerate(ep.tolist())}
        exist, tprobs = model_scores(model, inputs, G, ei, ep, torch)
        trip = np.concatenate([pos, neg])
        pr = np.asarray([row_of[(i, j)] for i, j, k in trip],
                        dtype=np.int64)
        y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
        s = joint_score(exist, tprobs, trip, pr)
        m = binary_metrics(y, s)
        ncodes = S.unknown_strata(neg, mask, Y)
        neg_scores = s[len(pos):]
        strat = {}
        for code, name in enumerate(STRATA):
            sel = ncodes == code
            strat[name] = {'n': int(sel.sum()),
                           'score_mean': float(neg_scores[sel].mean())
                           if sel.any() else None}
        ps = S.positive_strata(pos, mask)
        row = {'seed': seed, **m, 'strata': strat,
               'pos_pair_seen': int(ps.sum()),
               'pos_pair_unseen': int((1 - ps).sum()),
               'pos_seen_score_mean': float(s[:len(pos)][ps == 1].mean())
               if ps.any() else None,
               'pos_unseen_score_mean':
                   float(s[:len(pos)][ps == 0].mean())
               if (1 - ps).any() else None,
               'eval_neg_sha256': _arr(neg)}
        t_rows.append(row)
    # ---- Task P ----
    p_rows = []
    for seed, pos, neg in p_reps:
        ep = np.unique(np.concatenate([pos, neg]), axis=0)
        row_of = {(i, j): r for r, (i, j) in enumerate(ep.tolist())}
        exist, _ = model_scores(model, inputs, G, ei, ep, torch)
        pr = np.asarray([row_of[(i, j)] for i, j in
                         np.concatenate([pos, neg]).tolist()],
                        dtype=np.int64)
        y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
        s = pair_score(exist, pr)
        m = binary_metrics(y, s)
        p_rows.append({'seed': seed, **m, 'n_pos': int(len(pos)),
                       'eval_neg_sha256': _arr(S._sort_rows(neg))})

    bl = BL.run(Y, fold,
                [(pos, neg) for seed, pos, neg in t_reps])
    out = {
        'status': 'PASS', 'fold': fold,
        'task_T': {
            'definition': 's(i,j,k)=sigmoid(exist)*softmax(logits/T)[k]',
            'rows': t_rows,
            'mean': {k: float(np.mean([r[k] for r in t_rows]))
                     for k in ('auroc', 'average_precision',
                               'f1_at_0_5')}},
        'task_P': {
            'definition': 's_pair(i,j)=sigmoid(exist) on pair-unseen '
                          'positives vs pair-zero negatives',
            'rows': p_rows,
            'mean': {k: float(np.mean([r[k] for r in p_rows]))
                     for k in ('auroc', 'average_precision',
                               'f1_at_0_5')}},
        'baseline_logistic': bl}
    with open(out_path, 'x', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    return out
