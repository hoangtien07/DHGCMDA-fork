import itertools, numpy as np, pandas as pd
exec(open('entity_rule.py').read().split("for nm, F, ref in")[0])
for nm, F, ref in [('miRNA', Fm, rm), ('disease', Fd, rd)]:
    y = F.index.isin(ref); res = []
    print(f'== {nm}: ref min per-type deg', {t: int(F.loc[y, 'deg_' + t].min()) for t in T5}, 'frac ref with type>=1', {t: round(float((F.loc[y, 'deg_' + t] >= 1).mean()), 3) for t in T5})
    for r in range(1, 6):
        for S in itertools.combinations(T5, r):
            for k in [1, 2, 3]:
                for dmin in [0, 3]:
                    p = np.all([F['deg_' + t] >= k for t in S], 0) & (F.deg_trip >= dmin)
                    if nm == 'disease': p = p & F.has_C_tree.values.astype(bool)
                    res.append((int((p & ~y).sum()) + int((~p & y).sum()), S, k, dmin, int(p.sum()), int((p & ~y).sum()), int((~p & y).sum())))
    for x in sorted(res)[:6]: print('  errors', x[0], 'types', x[1], 'min per type', x[2], 'min total', x[3], 'selected', x[4], 'FP', x[5], 'FN', x[6])
