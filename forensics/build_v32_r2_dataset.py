"""build_v32_r2_dataset.py — build the R2 (leakage-free) v3.2 approximation artifact.

Pipeline = closest documented reconstruction of SPLD MDAv3.2-3 lineage:
  1. raw CuiLab v3_*.txt -> unique (mir, disease, type) triplets
  2. eligibility: mir in (miRBase hairpin ∩ MISIM 2.0), disease in MeSH category 'C'
  3. iterative typed-degree pruning: mir>=4, dis>=7  (best match: 406x266x11,970;
     mir>=4 alone already yields exactly 411 miRNAs)

Outputs to v3.2_lineage_approx/:
  - Y_multilabel.npy        float32 [406, 266, 5] multi-hot type tensor
  - triplets.csv            mi_idx, dis_idx, type_idx (0-based, type order below)
  - miRNA_names.txt / disease_names.txt
  - M_MISIM.npy             [406,406] real MISIM 2.0 functional similarity slice
  - D_MESH.npy              [266,266] Wang-style MeSH-tree semantic similarity
  - manifest.json           sha256 of sources + stats

Type order (canonical, paper Table): [genetics, epigenetics, circulation, target, tissue]
"""
import pandas as pd
import numpy as np
import json, hashlib, os
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'v3.2_lineage_approx')
os.makedirs(OUT, exist_ok=True)

FILES = {
    'genetics': 'HMDD_data/MDAv3.2/v3_genetics.txt',
    'epigenetics': 'HMDD_data/MDAv3.2/v3_epigenetics.txt',
    'circulation': 'HMDD_data/MDAv3.2/v3_circulation.txt',
    'target': 'HMDD_data/MDAv3.2/v3_target.txt',
    'tissue': 'HMDD_data/MDAv3.2/v3_tissue.txt',
}
ORDER = ['genetics', 'epigenetics', 'circulation', 'target', 'tissue']
TIDX = {t: i for i, t in enumerate(ORDER)}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

# ---------- 1. raw triplets ----------
trip = set()
dis_mesh = defaultdict(set)
src_hashes = {}
for t, rel in FILES.items():
    p = os.path.join(REPO, rel)
    src_hashes[rel] = sha256(p)
    df = pd.read_csv(p, sep='\t', dtype=str)
    for m, d, mesh in zip(df['mir'].str.strip().str.lower(),
                          df['disease'].str.strip().str.lower(),
                          df['mesh'].fillna('')):
        trip.add((m, d, t))
        if mesh.strip():
            dis_mesh[d].add(mesh.strip())

# ---------- 2. eligibility ----------
trees = json.load(open('/tmp/mesh_trees.json'))
C_uis = {ui for ui, v in trees.items() if 'C' in v['letters']}
dis_C = {d for d, uis in dis_mesh.items() if uis & C_uis}
mirbase = set(json.load(open('/tmp/mirbase_names.json')))
misim_names = [l.strip() for l in open('/tmp/misim/miRNA_name.txt') if l.strip()]
misim = {n.lower() for n in misim_names}
mir_ok = mirbase & misim

base = {(m, d, t) for m, d, t in trip if m in mir_ok and d in dis_C}

# ---------- 3. iterative typed-degree pruning mir>=4, dis>=7 ----------
def prune(tr, tm, td):
    tr = set(tr)
    while True:
        dm = Counter(x[0] for x in tr)
        dd = Counter(x[1] for x in tr)
        new = {(m, d, t) for m, d, t in tr if dm[m] >= tm and dd[d] >= td}
        if len(new) == len(tr):
            return new
        tr = new

final = prune(base, 4, 7)

mirs = sorted({x[0] for x in final})
diss = sorted({x[1] for x in final})
mi_i = {m: i for i, m in enumerate(mirs)}
di_i = {d: i for i, d in enumerate(diss)}
per = Counter(x[2] for x in final)
print(f"dataset: {len(mirs)} mir x {len(diss)} dis x {len(final)} triplets")
print("per-type:", {t: per[t] for t in ORDER})

# ---------- 4. Y tensor + triplets csv ----------
Y = np.zeros((len(mirs), len(diss), len(ORDER)), dtype=np.float32)
rows = []
for m, d, t in sorted(final):
    i, j, k = mi_i[m], di_i[d], TIDX[t]
    Y[i, j, k] = 1.0
    rows.append((i, j, k))
np.save(os.path.join(OUT, 'Y_multilabel.npy'), Y)
pd.DataFrame(rows, columns=['mi', 'di', 'type']).to_csv(
    os.path.join(OUT, 'triplets.csv'), index=False)
open(os.path.join(OUT, 'miRNA_names.txt'), 'w').write('\n'.join(mirs) + '\n')
open(os.path.join(OUT, 'disease_names.txt'), 'w').write('\n'.join(diss) + '\n')

# ---------- 5. M_MISIM slice ----------
sim = np.loadtxt('/tmp/misim/similarity.txt')
assert sim.shape[0] == len(misim_names), (sim.shape, len(misim_names))
name2idx = {n.lower(): i for i, n in enumerate(misim_names)}
missing = [m for m in mirs if m not in name2idx]
print("mirs missing in MISIM names:", len(missing), missing[:10])
idx = np.array([name2idx[m] for m in mirs if m in name2idx])
M_MISIM = np.zeros((len(mirs), len(mirs)), dtype=np.float32)
keep = [i for i, m in enumerate(mirs) if m in name2idx]
sub = sim[np.ix_(idx, idx)].astype(np.float32)
for a, ia in enumerate(keep):
    for b, ib in enumerate(keep):
        M_MISIM[ia, ib] = sub[a, b]
np.fill_diagonal(M_MISIM, 1.0)
np.save(os.path.join(OUT, 'M_MISIM.npy'), M_MISIM)

# ---------- 6. D_MESH Wang-style tree similarity ----------
ui2trees = {}
for ui, v in trees.items():
    tns = []
    for t in v.get('tree', []):
        tn = t.rsplit('/', 1)[-1]
        if tn and tn[0].isalpha() and '.' in tn:
            tns.append(tn)
    ui2trees[ui] = tns

def dis_tree_paths(d):
    paths = []
    for ui in dis_mesh.get(d, ()):
        for tn in ui2trees.get(ui, []):
            paths.append(tn.split('.'))
    return paths

DECAY = 0.5
def tree_contrib(path):
    # ancestor at index a has distance (len-1-a) from node
    L = len(path)
    return [DECAY ** (L - 1 - a) for a in range(L)]

def pair_sim(pa, pb):
    best = 0.0
    for a in pa:
        ca = tree_contrib(a)
        sva = sum(ca)
        for b in pb:
            cb = tree_contrib(b)
            svb = sum(cb)
            shared = 0.0
            for k in range(min(len(a), len(b))):
                if a[k] != b[k]:
                    break
                shared += ca[k] + cb[k]
            best = max(best, shared / (sva + svb))
    return best

paths = {d: dis_tree_paths(d) for d in diss}
nopath = [d for d in diss if not paths[d]]
print("diseases without tree path:", len(nopath), nopath[:10])
D_MESH = np.eye(len(diss), dtype=np.float32)
for i in range(len(diss)):
    for j in range(i + 1, len(diss)):
        if paths[diss[i]] and paths[diss[j]]:
            D_MESH[i, j] = D_MESH[j, i] = pair_sim(paths[diss[i]], paths[diss[j]])
np.save(os.path.join(OUT, 'D_MESH.npy'), D_MESH)

# ---------- 7. manifest ----------
manifest = {
    'pipeline': 'misim∩mirbase(mir) x mesh-category-C(dis) -> iterative typed-degree mir>=4 dis>=7',
    'mir': len(mirs), 'dis': len(diss), 'triplets': len(final),
    'per_type': {t: per[t] for t in ORDER},
    'type_order': ORDER,
    'misim_missing_mirs': missing,
    'diseases_no_treepath': nopath,
    'source_hashes': src_hashes,
    'paper_target': {'mir': 411, 'dis': 271, 'triplets': 11748,
                     'per_type': {'genetics': 1155, 'epigenetics': 403,
                                  'circulation': 2293, 'target': 3997, 'tissue': 3900}},
}
json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=2)
print("wrote", OUT)
