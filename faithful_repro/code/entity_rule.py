"""B1: can a simple rule over entity attributes recover the 411 miRNA / 271 disease selection of the reference artifact?"""
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
O = '/home/ubuntu/dhg_runs/lit/cuilab_old/'; T5 = ['genetics', 'epigenetics', 'circulation', 'target', 'tissue']
R = dict(np.load('../data/v3.2_ref_MDAv3.2-3.npz', allow_pickle=True))
rm = set(str(x).lower() for x in R['mir_names']); rd = set(str(x).lower().strip() for x in R['dis_names'])
def rd_(p, cols=(0, 1, 2, 3)):
    d = pd.read_csv(p, sep='\t', dtype=str, encoding='latin-1', usecols=list(cols)); d.columns = ['cat', 'mir', 'dis', 'pmid'][:len(cols)]
    d['mir'] = d.mir.str.strip().str.lower(); d['dis'] = d.dis.str.strip().str.strip('"').str.lower(); return d
fr = []
for t in T5:
    d = rd_(O + t + '.txt'); d['type'] = t; fr.append(d)
d = pd.concat(fr); tri = d[['mir', 'dis', 'type']].drop_duplicates()
v3 = pd.concat([rd_(O + 'v3_' + t + '.txt') for t in T5]); old = rd_(O + 'old_alldata.txt', (0, 1, 2, 3)); al = rd_(O + 'alldata.txt')
mh = {}; ent = {}; ui = None; trees = {}
for l in open('../ext/d2025.bin', encoding='latin-1'):
    if l.startswith('*NEWRECORD'): ui = None; cur = []; names = []
    elif l.startswith('MH = '): mhn = l[5:].strip().lower(); names.append(mhn)
    elif l.startswith('ENTRY = ') or l.startswith('PRINT ENTRY = '): names.append(l.split('=', 1)[1].split('|')[0].strip().lower())
    elif l.startswith('MN = '): cur.append(l[5:].strip())
    elif l.startswith('UI = '):
        for i, n in enumerate(names): (mh if i == 0 else ent)[n] = l[5:].strip()
        trees[l[5:].strip()] = cur
mis = set(x.strip().lower() for x in open('../ext/misim/miRNA_name.txt'))
seqs = set(); 
for l in open('../ext/seqgene/mature.fa.gz'):
    if l.startswith('>hsa'): seqs.add(l[1:].split()[0].lower())
hp = set(l[1:].split()[0].lower() for l in open('../ext/seqgene/hairpin.fa') if l.startswith('>hsa'))
def feats(key, other):
    g = tri.groupby(key); F = pd.DataFrame({'deg_trip': g.size(), 'deg_pair': tri.drop_duplicates(['mir', 'dis']).groupby(key).size()})
    for t in T5: F['deg_' + t] = tri[tri.type == t].groupby(key).size()
    F['n_pmid'] = d.groupby(key).pmid.nunique(); F['n_rows_alldata'] = al.groupby(key).size()
    F['in_v3'] = F.index.isin(set(v3[key])); F['in_old'] = F.index.isin(set(old[key]))
    return F.fillna(0)
Fm = feats('mir', 'dis'); Fm['in_misim'] = Fm.index.isin(mis); Fm['in_mature'] = Fm.index.map(lambda m: any(s == m or s.startswith(m + '-') for s in seqs) if False else (m in hp))
Fm['is_arm_or_cluster'] = Fm.index.str.count('-') > 2
Fd = feats('dis', 'mir'); Fd['mesh_mh'] = Fd.index.isin(set(mh)); Fd['mesh_any'] = Fd.index.isin(set(mh) | set(ent))
Fd['has_C_tree'] = [any(t.startswith('C') for t in trees.get(mh.get(n, ent.get(n, '')), [])) for n in Fd.index]
Fd['unspecific'] = Fd.index.str.contains(r'\[')
for nm, F, ref in [('miRNA', Fm, rm), ('disease', Fd, rd)]:
    y = F.index.isin(ref); X = F.astype(float)
    print(f'===== {nm}: {len(F)} candidates, {y.sum()} in ref')
    print(X.groupby(y).mean().T.round(2).to_string())
    for depth in [2, 3, 5]:
        t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y); p = t.predict(X)
        print(f'-- depth {depth}: train acc {np.mean(p == y):.3f}, FP {int((p & ~y).sum())}, FN {int((~p & y).sum())}')
    t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y); print(export_text(t, feature_names=list(X.columns)))
