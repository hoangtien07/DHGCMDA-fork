"""Build the paper's four similarity views for an existing faithful-run dataset."""
import argparse, collections, json, os, re, numpy as np, pandas as pd
import build_datasets as B
from Bio import Align

SG = os.path.join(B.EXT, 'seqgene')


def mesh_name_to_ui():
    aliases = collections.defaultdict(set)
    record = {'terms': []}

    def add_record():
        ui = record.get('ui')
        if ui:
            for term in record['terms']:
                aliases[term.strip().lower()].add(ui)

    with open(os.path.join(B.EXT, 'd2025.bin'), encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '*NEWRECORD':
                add_record()
                record = {'terms': []}
            elif line.startswith('UI = '):
                record['ui'] = line[5:].strip()
            elif line.startswith(('MH = ', 'ENTRY = ', 'PRINT ENTRY = ')):
                record['terms'].append(line.split('=', 1)[1].split('|', 1)[0].strip())
    add_record()
    return {term: next(iter(uis)) for term, uis in aliases.items() if len(uis) == 1}


def build(base='v3.2_cuilab_deg2', out='v3.2_cuilab_deg2_4view'):
    D = dict(np.load(os.path.join(B.ROOT, 'data', base + '.npz'), allow_pickle=True))
    mirs, diss = [str(x) for x in D['mir_names']], [str(x) for x in D['dis_names']]

    # ---- m_ss
    mat, n = {}, None
    for l in open(os.path.join(SG, 'mature.fa.gz')):
        if l.startswith('>'):
            n = l[1:].split()[0]; mat[n] = ''
        else:
            mat[n] += l.strip()
    arms = collections.defaultdict(set)
    for k, s in mat.items():
        if k.startswith('hsa-'):
            arms[re.sub(r'-(5p|3p)$', '', k.lower())].add(s)

    def seqs(m):
        for c in (m, re.sub(r'-\d+$', '', m)):
            if c in arms:
                return sorted(arms[c])
        return []

    ms = [seqs(m) for m in mirs]
    al = Align.PairwiseAligner(mode='global', match_score=1, mismatch_score=-1, open_gap_score=-2, extend_gap_score=-1)
    cov = [i for i, s in enumerate(ms) if s]
    R = np.full((len(mirs), len(mirs)), np.nan)
    for a_, i in enumerate(cov):
        for j in cov[a_ + 1:]:
            R[i, j] = R[j, i] = max(al.score(x, y) for x in ms[i] for y in ms[j])
    v = R[~np.isnan(R)]; lo, hi = v.min(), v.max()
    MSS = np.where(np.isnan(R), 0, (R - lo) / (hi - lo)).astype(np.float32); np.fill_diagonal(MSS, 1)

    # ---- d_gs
    if base == 'v3.2_cuilab_deg2':
        d = B.load_raw32()
        ui_map = d.dropna(subset=['mesh']).groupby('disease')['mesh'].agg(lambda s: s.value_counts().index[0])
        uis = [re.split(r'[,;| ]+', str(ui_map.get(x, '')).strip())[0] if x in ui_map else '' for x in diss]
        name_mapped = None
    else:
        name_to_ui = mesh_name_to_ui()
        uis = [name_to_ui.get(x.strip().lower(), '') for x in diss]
        name_mapped = sum(bool(u) for u in uis)

    ctd = pd.read_csv(os.path.join(SG, 'CTD_curated_genes_diseases.tsv.gz'), sep='\t', comment='#', header=None, dtype=str)
    ctd = ctd[ctd[4].str.contains('marker/mechanism', na=False)]
    ctd['ui'] = ctd[3].str.replace('MESH:', '', regex=False)
    g_of = ctd.groupby('ui')[1].agg(lambda s: sorted(set(s.astype(int))))
    G = [g_of.get(u, []) if u else [] for u in uis]
    hn = pd.read_csv(os.path.join(SG, 'HumanNet-XC.tsv'), sep='\t', header=None, comment='#', names=['a', 'b', 'lls'])
    genes = sorted({g for gs in G for g in gs}); gi = {g: i for i, g in enumerate(genes)}
    hn = hn[hn.a.isin(gi) & hn.b.isin(gi)]
    lmin, lmax = hn.lls.min(), hn.lls.max()
    S = np.zeros((len(genes), len(genes)), np.float32)
    ia, ib = hn.a.map(gi).values, hn.b.map(gi).values
    S[ia, ib] = S[ib, ia] = ((hn.lls - lmin) / (lmax - lmin)).values; np.fill_diagonal(S, 1)
    dcov = [i for i, gs in enumerate(G) if gs]
    M = np.zeros((len(genes), len(diss)), np.float32)
    for i in dcov:
        M[:, i] = S[:, [gi[g] for g in G[i]]].max(1)
    Bm = np.zeros((len(diss), len(genes)), np.float32)
    for i in dcov:
        Bm[i, [gi[g] for g in G[i]]] = 1
    T = Bm @ M; nG = Bm.sum(1)
    DGS = np.zeros((len(diss), len(diss)), np.float32)
    den = nG[:, None] + nG[None, :]
    mask = den > 0; DGS[mask] = ((T + T.T)[mask] / den[mask])
    nz = np.array([bool(g) for g in G]); DGS[~nz] = 0; DGS[:, ~nz] = 0; np.fill_diagonal(DGS, 1)

    extra = dict(D['extra'].item() if 'extra' in D else {},
                 view_m1='m_ss NW miRBase mature (match1 mismatch-1 gapopen-2 ext-1, max over arms, minmax)',
                 view_d1='d_gs Cheng2014 FunSim CTD curated marker/mechanism + HumanNet-XC LLS minmax',
                 mss_covered=len(cov), dgs_covered=len(dcov), hn_lls_range=[float(lmin), float(lmax)],
                 nw_raw_range=[float(lo), float(hi)],
                 sources=dict(mirbase='https://www.mirbase.org/download/mature.fa',
                              ctd='http://ctdbase.org/reports/CTD_curated_genes_diseases.tsv.gz',
                              humannet='https://www.inetbio.org/humannetv3/networks/HumanNet-XC.tsv'))
    if name_mapped is not None:
        extra['mesh_name_mapped'] = int(name_mapped)
        extra['mesh_name_mapping'] = 'MeSH 2025 MH/ENTRY/PRINT ENTRY exact case-insensitive unique-name match'

    B.save(out, Y=D['Y'], mir_names=D['mir_names'], dis_names=D['dis_names'], type_names=D['type_names'],
           sim_m1=MSS, sim_m2=D['sim_m2'], sim_d1=DGS, sim_d2=D['sim_d2'], extra=np.array(extra))
    off = lambda X: float((X.sum() - np.trace(X)) / (X.size - len(X)))
    print(json.dumps(dict(base=base, out=out, mss_cov=len(cov), dgs_cov=len(dcov),
                          name_to_ui_mapped=name_mapped, n_genes=len(genes), hn_edges=len(hn),
                          mss_mean=off(MSS), dgs_mean=off(DGS))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='v3.2_cuilab_deg2')
    p.add_argument('--out', default='v3.2_cuilab_deg2_4view')
    args = p.parse_args()
    build(args.base, args.out)
