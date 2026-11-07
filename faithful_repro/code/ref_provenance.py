"""Provenance check of the 411x271 reference artifact (fork MDAv3.2-3) against CUILab files."""
import numpy as np, pandas as pd
O = '/home/ubuntu/dhg_runs/lit/cuilab_old/'
T5 = ['genetics', 'epigenetics', 'circulation', 'target', 'tissue']
R = dict(np.load('../data/v3.2_ref_MDAv3.2-3.npz', allow_pickle=True))
rm = [str(x).lower() for x in R['mir_names']]; rd = [str(x).lower().strip() for x in R['dis_names']]; T = list(R['type_names'])
ref = set((rm[i], rd[j], T[k]) for i, j, k in np.argwhere(R['Y'] > 0))
def load(prefix):
    fr = []
    for t in T5:
        d = pd.read_csv(O + prefix + t + '.txt', sep='\t', dtype=str, encoding='latin-1', usecols=[0, 1, 2]); d.columns = ['cat', 'mir', 'dis']; d['type'] = t; fr.append(d)
    d = pd.concat(fr); d['mir'] = d.mir.str.strip().str.lower(); d['dis'] = d.dis.str.strip().str.strip('"').str.lower()
    return d[['mir', 'dis', 'type']].drop_duplicates()
for prefix in ['v3_', '']:
    tri = load(prefix); s = set(map(tuple, tri.values))
    sub = tri[tri.mir.isin(set(rm)) & tri.dis.isin(set(rd))]; ss = set(map(tuple, sub.values))
    print(f'[{prefix or "unprefixed"}] raw {tri.mir.nunique()}x{tri.dis.nunique()}x{len(tri)} | ref mirs present {len(set(rm)&set(tri.mir))}/411, dis {len(set(rd)&set(tri.dis))}/271'
          f' | ref triplets found {len(ref&s)}/{len(ref)} | raw triplets among ref entities {len(ss)}, not in ref {len(ss-ref)}')
mh = set(); ent = set()
for l in open('../ext/d2025.bin', encoding='latin-1'):
    if l.startswith('MH = '): mh.add(l[5:].strip().lower())
    elif l.startswith('ENTRY = ') or l.startswith('PRINT ENTRY = '): ent.add(l.split('=', 1)[1].split('|')[0].strip().lower())
alld = set(load('').dis); non = alld - set(rd)
print(f'MeSH 2025 name match (MH or entry): ref {len(set(rd)&(mh|ent))}/271, non-ref {len(non&(mh|ent))}/{len(non)}')
