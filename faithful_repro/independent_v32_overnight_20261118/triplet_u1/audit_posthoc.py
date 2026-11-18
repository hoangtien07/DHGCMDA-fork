# -*- coding: utf-8 -*-
"""WP1 post-hoc audit — locked eval_arrays.npz only, NO retrain.

Outputs audit_posthoc.json:
  * per-fold stratified Task-T AUROC: all positives vs negatives of
    each stratum (0 on-train-pair, 1 on-heldout-pair, 2 pair-zero),
    for U1X and both baseline pools; n_pos/n_neg/prevalence per
    stratum; AP reported only with prevalence context.
  * positive-side split: seen vs unseen pairs.
  * sensitivity: eval negatives overlapping each baseline's train
    negative set are dropped (recomputed deterministically), metrics
    recomputed for U1X and that baseline on the reduced set.
  * paired differences U1X - BL[mask_zero_pairs] per fold (mean over
    seeds) with std.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1] / 'colab_package_f0_selection_20261115'
STRICT = PKG / 'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
sys.path.insert(0, str(HERE))
import split as S                                   # noqa: E402
import baseline_logistic as BL                      # noqa: E402
from eval_cvtriplet import binary_metrics           # noqa: E402

STRATA = {0: 'neg_on_train_pair', 1: 'neg_on_heldout_pair',
          2: 'neg_pair_zero'}


def auroc(y, s):
    return binary_metrics(np.asarray(y), np.asarray(s))['auroc']


def main():
    z = np.load(STRICT, allow_pickle=True)
    Y = np.asarray(z['Y']).astype(bool)
    out = {'scope': 'post-hoc audit on locked eval_arrays.npz — '
                    'no retrain, no metric re-definition',
           'strata_meaning': STRATA,
           'folds': {}}
    for fold in range(5):
        seg = S.fold_segments(Y, fold)
        mask = S.pair_mask(*Y.shape[:2],
                           S.pairs_of(seg['inner_train']))
        # deterministically rebuild each baseline's train negatives
        trneg_sets = {}
        for pool in BL.POOLS:
            cells = BL.negative_pool(Y, mask, pool)
            sel = np.random.default_rng(
                S.BASELINE_NEG_SEED_BASE + fold).choice(
                len(cells), size=len(seg['inner_train']),
                replace=False)
            trneg_sets[pool] = S.triplet_set(
                S._sort_rows(cells[sel]))
        fkey = f'fold{fold}'
        out['folds'][fkey] = {'seeds': {}, 'paired': {}}
        agg = {k: [] for k in ('u1x', 'bl_mz', 'bl_or')}
        paired_mz, paired_or = [], []
        for si, seed in enumerate(S.EVAL_SEEDS):
            # average over the 3 model seeds for the fold
            u_rows, m_rows, o_rows = [], [], []
            hard_rows = []
            for run_key in ['fold%d_seed%d' % (fold, ms)
                            for ms in (1234, 42, 2025)]:
                arr = np.load(HERE / 'runs' / run_key /
                              'eval_arrays.npz')
                idx = arr[f'T_rep{si}_idx']
                y = arr[f'T_rep{si}_y']
                codes = arr[f'T_rep{si}_neg_codes']
                seen = arr[f'T_rep{si}_pos_seen']
                s_u = arr[f'T_rep{si}_s_u1x']
                s_m = arr[f'T_rep{si}_s_bl_mask_zero_pairs']
                s_o = arr[f'T_rep{si}_s_bl_all_unknown']
                u_rows.append(auroc(y, s_u))
                m_rows.append(auroc(y, s_m))
                o_rows.append(auroc(y, s_o))
            agg['u1x'].append(float(np.mean(u_rows)))
            agg['bl_mz'].append(float(np.mean(m_rows)))
            agg['bl_or'].append(float(np.mean(o_rows)))
            paired_mz.append(float(np.mean(u_rows)
                                   - np.mean(m_rows)))
            paired_or.append(float(np.mean(u_rows)
                                   - np.mean(o_rows)))
            # strata detail from seed1234 run only (identical eval
            # sets across model seeds — eval sampling uses eval seed)
            arr = np.load(HERE / 'runs' /
                          f'fold{fold}_seed1234' /
                          'eval_arrays.npz')
            idx = arr[f'T_rep{si}_idx']
            y = arr[f'T_rep{si}_y']
            codes = arr[f'T_rep{si}_neg_codes']
            seen = arr[f'T_rep{si}_pos_seen']
            s_u = arr[f'T_rep{si}_s_u1x']
            s_m = arr[f'T_rep{si}_s_bl_mask_zero_pairs']
            s_o = arr[f'T_rep{si}_s_bl_all_unknown']
            pos = y == 1
            neg_sel = y == 0
            st_rows = {}
            for code, name in STRATA.items():
                sel = pos | ((neg_sel) & (codes == code))
                st_rows[name] = {
                    'n_pos': int(pos.sum()),
                    'n_neg': int(sel.sum() - pos.sum()),
                    'prevalence': float(
                        pos.sum() / max(int(sel.sum()), 1)),
                    'auroc_u1x': float(auroc(y[sel], s_u[sel])),
                    'auroc_bl_mz': float(auroc(y[sel], s_m[sel])),
                    'auroc_bl_or': float(auroc(y[sel], s_o[sel]))}
            # positive-side split vs ALL negatives
            hard = {}
            for label, psel in (
                    ('pos_pair_seen', pos & (seen == 1)),
                    ('pos_pair_unseen', pos & (seen == 0))):
                sel = psel | neg_sel
                hard[label] = {
                    'n_pos': int(psel.sum()),
                    'n_neg': int(neg_sel.sum()),
                    'prevalence': float(
                        psel.sum() / max(int(sel.sum()), 1)),
                    'auroc_u1x': float(auroc(y[sel], s_u[sel])),
                    'auroc_bl_mz': float(auroc(y[sel], s_m[sel])),
                    'auroc_bl_or': float(auroc(y[sel], s_o[sel]))}
            # sensitivity: drop eval negs that overlap each baseline's
            # train negatives, recompute on reduced set
            sens = {}
            neg_list = [tuple(t) for t in idx[neg_sel].tolist()]
            for pool, tag in (('mask_zero_pairs', 'bl_mz'),
                              ('all_unknown', 'bl_or')):
                trs = trneg_sets[pool]
                keep_neg = np.array(
                    [tuple(t) not in trs for t in neg_list])
                sel = np.r_[np.full(int(pos.sum()), True),
                            keep_neg]
                s_bl = s_m if pool == 'mask_zero_pairs' else s_o
                sens[pool] = {
                    'dropped_negs': int((~keep_neg).sum()),
                    'auroc_u1x': float(auroc(y[sel], s_u[sel])),
                    'auroc_bl': float(auroc(y[sel], s_bl[sel]))}
            out['folds'][fkey]['seeds'][f'seed{seed}'] = {
                'auroc_u1x_mean3': agg['u1x'][-1],
                'auroc_bl_mz_mean3': agg['bl_mz'][-1],
                'auroc_bl_or_mean3': agg['bl_or'][-1],
                'strata': st_rows, 'pos_split': hard,
                'sensitivity_no_overlap': sens}
        out['folds'][fkey]['paired'] = {
            'u1x_minus_bl_mz': {
                'per_seed': paired_mz,
                'mean': float(np.mean(paired_mz)),
                'std': float(np.std(paired_mz))},
            'u1x_minus_bl_or': {
                'per_seed': paired_or,
                'mean': float(np.mean(paired_or)),
                'std': float(np.std(paired_or))},
            'u1x_mean': float(np.mean(agg['u1x'])),
            'bl_mz_mean': float(np.mean(agg['bl_mz'])),
            'bl_or_mean': float(np.mean(agg['bl_or']))}
    mz = [out['folds'][f]['paired']['u1x_minus_bl_mz']['mean']
          for f in out['folds']]
    out['paired_summary'] = {
        'u1x_minus_bl_mz_per_fold_mean': mz,
        'grand_mean': float(np.mean(mz)),
        'note': 'paired = same eval replicates, same fold; '
                'negative on every fold = baseline dominates'}
    with open(HERE / 'audit_posthoc.json', 'x', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    print('AUDIT DONE')
    for f, d in out['folds'].items():
        p = d['paired']
        print(f, 'U1X %.4f BLm %.4f BLo %.4f d_mz %+.4f d_or %+.4f'
              % (p['u1x_mean'], p['bl_mz_mean'], p['bl_or_mean'],
                 p['u1x_minus_bl_mz']['mean'],
                 p['u1x_minus_bl_or']['mean']))


if __name__ == '__main__':
    main()
