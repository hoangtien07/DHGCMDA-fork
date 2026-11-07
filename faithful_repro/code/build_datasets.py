"""Build datasets for the paper-faithful DHGCMDA run.

Outputs <out>/<name>.npz with: mir_names, dis_names, type_names, Y (nm, nd, C) multi-hot,
sim_m1, sim_m2, sim_d1, sim_d2 (None-able: 'GIP' marker => recomputed per fold from A_train).
"""
import json, os, re, sys, hashlib
import numpy as np, pandas as pd

ROOT = os.path.expanduser('~/dhg_runs/faithful')
UP = os.path.expanduser('~/repos/DHGCMDA/v2.0_495m383D')
REF = os.path.expanduser('~/repos/DHGCMDA-fork/HMDD_data/MDAv3.2-3')
RAW = os.path.join(ROOT, 'hmdd_v32_raw')
EXT = os.path.join(ROOT, 'ext')
OUT = os.path.join(ROOT, 'data')
TYPES32 = ['genetics', 'epigenetics', 'circulation', 'target', 'tissue']


def save(name, **kw):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + '.npz')
    np.savez_compressed(p, **kw)
    Y = kw['Y']
    stats = dict(name=name, n_mir=int(Y.shape[0]), n_dis=int(Y.shape[1]), n_types=int(Y.shape[2]),
                 triplets=int(Y.sum()), pairs=int((Y.sum(2) > 0).sum()),
                 per_type=dict(zip([str(t) for t in kw['type_names']], Y.sum((0, 1)).astype(int).tolist())),
                 min_mir_pair_deg=int((Y.sum(2) > 0).sum(1).min()), min_dis_pair_deg=int((Y.sum(2) > 0).sum(0).min()))
    for k in ['sim_m1', 'sim_m2', 'sim_d1', 'sim_d2']:
        v = kw[k]
        stats[k] = 'GIP(per-fold A_train)' if v.ndim == 0 else f'{v.shape} mean_offdiag={float((v.sum()-np.trace(v))/(v.size-len(v))):.4f}'
    stats.update(kw.get('extra', np.array({})).item() if 'extra' in kw else {})
    print(json.dumps(stats, indent=1))
    with open(p.replace('.npz', '.json'), 'w') as f:
        json.dump(stats, f, indent=1)


GIP = np.array('GIP')


def build_v20():
    d = pd.read_csv(os.path.join(UP, 'multi_all_mirna_disease_pairs_without_negative.csv'), header=None)
    Y = np.zeros((495, 383, 4), np.float32)
    Y[d[0].values - 1, d[1].values - 1, d[2].values - 1] = 1
    # author view mapping (prepareData.py): m view1 = M_GSM (GIP of full A -> recomputed per fold), m view2 = M_FSM,
    # d view1 = D_SSM2, d view2 = D_SSM1
    save('v2.0', Y=Y, mir_names=np.arange(1, 496), dis_names=np.arange(1, 384),
         type_names=np.array(['circulation', 'epigenetics', 'target', 'genetics']),
         sim_m1=GIP, sim_m2=np.loadtxt(os.path.join(UP, 'M_FSM.txt')).astype(np.float32),
         sim_d1=np.loadtxt(os.path.join(UP, 'D_SSM2.txt')).astype(np.float32),
         sim_d2=np.loadtxt(os.path.join(UP, 'D_SSM1.txt')).astype(np.float32))


def parse_mesh():
    ui2trees, tree2ui, cur = {}, {}, {}
    with open(os.path.join(EXT, 'd2025.bin'), encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '*NEWRECORD':
                cur = {'MN': []}
            elif line.startswith('MN = '):
                cur['MN'].append(line[5:])
            elif line.startswith('UI = '):
                ui = line[5:]
                ui2trees[ui] = cur['MN']
                for t in cur['MN']:
                    tree2ui[t] = ui
    return ui2trees, tree2ui


def dag(ui, ui2trees, tree2ui):
    """Wang DAG contributions: node -> max over paths of 0.5**depth_diff."""
    contrib = {ui: 1.0}
    for t in ui2trees.get(ui, []):
        parts = t.split('.')
        for k in range(1, len(parts)):
            anc = tree2ui.get('.'.join(parts[:k]))
            if anc is None:
                continue
            w = 0.5 ** (len(parts) - k)
            contrib[anc] = max(contrib.get(anc, 0.0), w)
    return contrib


def semantic_sims(uis, ui2trees, tree2ui):
    n = len(uis)
    dags = [dag(u, ui2trees, tree2ui) if u else {} for u in uis]
    # SS2 (Xuan et al.): information content over all MeSH descriptors in category C
    c_uis = [u for u, ts in ui2trees.items() if any(t.startswith('C') for t in ts)]
    cnt = {}
    for u in c_uis:
        for a in set(dag(u, ui2trees, tree2ui)):
            cnt[a] = cnt.get(a, 0) + 1
    N = len(c_uis)
    ic = lambda a: -np.log(cnt.get(a, 1) / N)
    dags2 = [{a: ic(a) for a in g} for g in dags]
    S1 = np.eye(n, dtype=np.float32); S2 = np.eye(n, dtype=np.float32)
    for i in range(n):
        if not dags[i]:
            continue
        for j in range(i + 1, n):
            if not dags[j]:
                continue
            common = dags[i].keys() & dags[j].keys()
            if not common:
                continue
            s1 = sum(dags[i][a] + dags[j][a] for a in common) / (sum(dags[i].values()) + sum(dags[j].values()))
            den2 = sum(dags2[i].values()) + sum(dags2[j].values())
            s2 = sum(dags2[i][a] + dags2[j][a] for a in common) / den2 if den2 > 0 else 0.0
            S1[i, j] = S1[j, i] = s1; S2[i, j] = S2[j, i] = s2
    return S1, S2


def load_raw32():
    frames = []
    for t in TYPES32:
        e = pd.read_csv(os.path.join(RAW, f'v3_{t}.txt'), sep='\t', dtype=str, encoding='latin-1')
        e['type'] = t
        frames.append(e)
    d = pd.concat(frames)
    d['mir'] = d['mir'].str.strip().str.lower()
    d['disease'] = d['disease'].str.strip()
    return d


def build_v32_cuilab(min_deg=2):
    d = load_raw32()
    raw_tri = d[['mir', 'disease', 'type']].drop_duplicates()
    raw_stats = dict(raw_rows=len(d), raw_mir=raw_tri.mir.nunique(), raw_dis=raw_tri.disease.nunique(),
                     raw_triplets=len(raw_tri), raw_pairs=len(raw_tri[['mir', 'disease']].drop_duplicates()))
    # paper 3.1: exclude miRNAs and diseases with fewer than two experimentally verified associations
    # (single pass; association = unique miRNA-disease pair over the five types)
    pairs = raw_tri[['mir', 'disease']].drop_duplicates()
    mdeg = pairs.mir.value_counts(); ddeg = pairs.disease.value_counts()
    keep = raw_tri[raw_tri.mir.map(mdeg).ge(min_deg) & raw_tri.disease.map(ddeg).ge(min_deg)]
    mirs = sorted(keep.mir.unique()); diss = sorted(keep.disease.unique())
    mi = {m: i for i, m in enumerate(mirs)}; di = {x: i for i, x in enumerate(diss)}
    Y = np.zeros((len(mirs), len(diss), 5), np.float32)
    Y[keep.mir.map(mi).values, keep.disease.map(di).values, keep.type.map(TYPES32.index).values] = 1
    # disease semantic views from MeSH 2025 (UI given by HMDD 'mesh' column; first UI if several)
    ui_map = (d.dropna(subset=['mesh']).groupby('disease')['mesh'].agg(lambda s: s.value_counts().index[0]))
    uis = [re.split(r'[,;| ]+', str(ui_map.get(x, '')).strip())[0] if x in ui_map else '' for x in diss]
    ui2trees, tree2ui = parse_mesh()
    uis = [u if u in ui2trees else '' for u in uis]
    S1, S2 = semantic_sims(uis, ui2trees, tree2ui)
    # miRNA functional view: MISIM 2.0 (cuilab/lirmed), matched case-insensitively
    names = [l.strip().lower() for l in open(os.path.join(EXT, 'misim', 'miRNA_name.txt'))]
    F = np.loadtxt(os.path.join(EXT, 'misim', 'similarity.txt'), dtype=np.float32)
    idx = {n: i for i, n in enumerate(names)}
    FM = np.eye(len(mirs), dtype=np.float32)
    hit = [idx.get(m) for m in mirs]
    sel = [i for i, h in enumerate(hit) if h is not None]
    hsel = [hit[i] for i in sel]
    FM[np.ix_(sel, sel)] = F[np.ix_(hsel, hsel)]
    np.fill_diagonal(FM, 1.0)
    extra = dict(raw_stats, min_deg=min_deg, mesh_covered=int(sum(1 for u in uis if u)),
                 misim_covered=len(sel), paper_target='411 mir x 271 dis x 11748 triplets')
    save('v3.2_cuilab_deg2', Y=Y, mir_names=np.array(mirs), dis_names=np.array(diss), type_names=np.array(TYPES32),
         sim_m1=GIP, sim_m2=FM, sim_d1=S2, sim_d2=S1, extra=np.array(extra))
    # sensitivity table (counts only) for other literal readings of the filter
    rows = []
    for mode in ['pair', 'triplet']:
        for iterative in [False, True]:
            k = raw_tri.copy()
            while True:
                base = k[['mir', 'disease']].drop_duplicates() if mode == 'pair' else k
                md, dd = base.mir.value_counts(), base.disease.value_counts()
                k2 = k[k.mir.map(md).ge(min_deg) & k.disease.map(dd).ge(min_deg)]
                done = len(k2) == len(k) or not iterative
                k = k2
                if done:
                    break
            rows.append(dict(degree=mode, iterative=iterative, mir=k.mir.nunique(), dis=k.disease.nunique(), triplets=len(k)))
    print(pd.DataFrame(rows).to_string())
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'v3.2_filter_sensitivity.csv'), index=False)


def build_v32_ref():
    Y = np.stack([pd.read_csv(os.path.join(REF, f'{t}.csv'), index_col=0).values for t in TYPES32], 2).astype(np.float32)
    save('v3.2_ref_MDAv3.2-3', Y=Y, mir_names=pd.read_csv(os.path.join(REF, 'mi_name_3.2_3.csv'), index_col=0).iloc[:, 0].values.astype(str),
         dis_names=pd.read_csv(os.path.join(REF, 'dis_name_3.2_3.csv'), index_col=0).iloc[:, 0].values.astype(str),
         type_names=np.array(TYPES32), sim_m1=GIP,
         sim_m2=pd.read_csv(os.path.join(REF, 'mi_fun_sim_3.2_3.csv'), index_col=0).values.astype(np.float32),
         sim_d1=GIP, sim_d2=pd.read_csv(os.path.join(REF, 'DSSM3.2_3.csv'), index_col=0).values.astype(np.float32))


if __name__ == '__main__':
    which = sys.argv[1:] or ['v2.0', 'v3.2_cuilab', 'v3.2_ref']
    if 'v2.0' in which: build_v20()
    if 'v3.2_cuilab' in which: build_v32_cuilab()
    if 'v3.2_ref' in which: build_v32_ref()
