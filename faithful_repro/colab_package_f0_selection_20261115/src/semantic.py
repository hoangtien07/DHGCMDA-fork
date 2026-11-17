# -*- coding: utf-8 -*-
"""Disease semantic values from a declared external MeSH DAG — REV2.

Formula: the inspected MISIM v2.0 default pipeline (author archive script
1, read — never executed). Per disease, all MeSH tree codes are gathered;
each code contributes 1 to itself and theta**depth to each PREFIX ancestor
(theta = 0.5), with max retained across codes; contributions are keyed by
ancestor descriptor NAME with first-in-order dedup. DV(d) = 1 + sum of
ancestor contributions; T(d1,d2) = sum_{shared ancestors}(c1[a]+c2[a]) /
(DV1+DV2) — the theta_pair = 1.0 branch of author script 3.1.

DESCRIPTOR KEYS (REV2): the author formula keys own-DV and pair terms by
disease NAME, not by dataset row. Rows whose resolved DescriptorName is
identical are ALIASES and must contribute once downstream. Each row gets
an explicit `desc_key`:
  ('D', name)    single resolved descriptor
  ('D+', atoms)  explicit multi-descriptor union (only when
                 multi_descriptor='expand'; recorded atoms metadata —
                 an engineering expansion, NOT a claim of author-join
                 equivalence)
  ('U', i)       uncovered row — unique per row by declared policy so
                 unrelated unmapped names can never silently merge.

Entity chain (declared): raw name -> archive TERM -> MeSH UI ->
DescriptorName -> 2019 tree codes; pass-through when the name itself is a
tree DescriptorName with no TERM resolution. Multi-UI iteration and
shared-ancestor summations are SORTED for byte-determinism across
PYTHONHASHSEED; per-code file order is preserved where the author's
order-dependent self-name crediting quirk needs it.

All sources read-only; nothing here executes downloaded code.
"""
from dataclasses import dataclass, field
import csv
import hashlib
import os
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

THETA_DAG = 0.5
THETA_PAIR = 1.0
M_ = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


# ---------- external source loading (stdlib only) ------------------------
def _xlsx_rows(path, sheet_xml):
    z = zipfile.ZipFile(path)
    ss = ET.fromstring(z.read('xl/sharedStrings.xml'))
    strings = [''.join(t.text or '' for t in si.iter(M_ + 't'))
               for si in ss.iter(M_ + 'si')]
    root = ET.fromstring(z.read(sheet_xml))
    for row in root.iter(M_ + 'row'):
        cells = {}
        for c in row.iter(M_ + 'c'):
            ref = c.attrib.get('r', '')
            col = re.match(r'[A-Z]+', ref).group(0)
            v = c.find(M_ + 'v')
            if v is None:
                continue
            cells[col] = (strings[int(v.text)]
                          if c.attrib.get('t') == 's' else v.text)
        yield cells


def load_mesh_source(xlsx_path, meshtoid_path, mtrees_path):
    """Load the frozen 2019 entity chain (paths + sha256 provenance)."""
    term2uis = {}
    for r in _xlsx_rows(xlsx_path, 'xl/worksheets/sheet1.xml'):
        if r.get('A') == 'TERM':
            continue
        if r.get('D'):
            term2uis.setdefault(r.get('A', ''), set()).add(r['D'])
    ui2name = {}
    for row in csv.DictReader(open(meshtoid_path, encoding='utf-8')):
        ui2name[row['DescriptorUI']] = row['DescriptorName']
    name2codes, code2name = {}, {}
    with open(mtrees_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            nm, code = line.split(';', 1)
            name2codes.setdefault(nm, []).append(code)
            code2name.setdefault(code, []).append(nm)
    prov = {
        'declared_mesh_year': '2019',
        'chain': 'raw_name->TERM->MeSH_UI->DescriptorName->tree_codes '
                 '(author-archive embedded maps; no inferred identity)',
        'files': {os.path.basename(p): hashlib.sha256(open(p, 'rb')
                                                      .read()).hexdigest()
                  for p in (xlsx_path, meshtoid_path, mtrees_path)},
    }
    return {'term2uis': term2uis, 'ui2name': ui2name,
            'name2codes': name2codes, 'code2name': code2name,
            'provenance': prov}


def mesh_source_from_dicts(term2uis, ui2name, name2codes, code2name,
                           provenance=None):
    """Fixture/injection path: identical bundle from in-memory maps."""
    return {'term2uis': term2uis, 'ui2name': ui2name,
            'name2codes': name2codes, 'code2name': code2name,
            'provenance': provenance or {'declared_mesh_year': 'synthetic'}}


# ---------- descriptor atom resolution (mapping only, no matrix) --------
def resolve_atoms(names, mesh_source):
    """Per-name sorted descriptor-atom list (deterministic).

    TERM chain first; if it yields no descriptor and the name itself is a
    2019 tree DescriptorName, pass through. Returns list of sorted lists.
    """
    atoms_per_row = []
    for nm in names:
        uis = sorted(mesh_source['term2uis'].get(str(nm), set()))
        atoms = sorted({mesh_source['ui2name'][u] for u in uis
                        if u in mesh_source['ui2name']})
        if not atoms and nm in mesh_source['name2codes']:
            atoms = [str(nm)]
        atoms_per_row.append(atoms)
    return atoms_per_row


# ---------- per-disease DAG values (author script-1 semantics) -----------
def _dv_values(A, index, trss):
    A_tmp = '.'.join(A.split('.')[:index])
    if len(A.split('.')) == index:
        trss[A_tmp] = 1
        return 1
    cand = THETA_DAG * _dv_values(A, index + 1, trss)
    if A_tmp in trss:
        if cand > trss[A_tmp]:
            trss[A_tmp] = cand
    else:
        trss[A_tmp] = cand
    return cand


def disease_dv_tree(codes, code2name):
    """DV + ancestor-name->contribution dict for ONE descriptor's code
    list (file order preserved — the author's order-dependent self-name
    crediting quirk lives here). Returns (1.0, {}) when uncovered."""
    if not codes:
        return 1.0, {}
    trss = {}
    for c in codes:
        for j in range(len(c.split('.')), 0, -1):
            _dv_values(c, j, trss)
    tree = {}
    DV = 1.0
    for c in codes:
        tree[code2name[c][0]] = 1.0
        for code, val in trss.items():
            nm = code2name[code][0]
            if nm not in tree:
                tree[nm] = val
                DV += val
    return DV, tree


@dataclass(frozen=True)
class SemanticResult:
    """Disease semantic bundle, per-row aligned to the input order.

    T:         (n,n) float64 formula semantic similarity (pair-θ=1).
               Uncovered-key rows are all-zero INCLUDING the diagonal.
    S:         (n,n) float64 feature matrix = T with the declared fixed
               missing fill applied to uncovered rows (diag 1, off-diag
               0). This is a fixed fill, NOT a learned imputation.
    dv:        (n,) float64 per-key DAG semantic value (1.0 uncovered).
    coverage:  (n,) bool — True iff the row resolved to >=1 tree code.
    desc_keys: (n,) tuple of canonical keys ('D',name)/('D+',atoms)/
               ('U',i). Rows sharing a 'D' key are aliases — identical
               dv/T/coverage guaranteed (asserted).
    key_rep:   dict canonical key -> representative row index (first).
    atoms:     (n,) tuple of sorted descriptor atoms per row.
    provenance: source files/hashes/declared year/chain/policies.
    """
    T: np.ndarray
    S: np.ndarray
    dv: np.ndarray
    coverage: np.ndarray
    desc_keys: tuple
    key_rep: dict
    atoms: tuple
    provenance: dict = field(default_factory=dict)


def semantic_similarity(disease_names, mesh_source,
                        multi_descriptor='reject'):
    """Compute the semantic bundle for an ordered disease-name list.

    multi_descriptor: 'reject' (default) raises on names resolving to
    more than one descriptor, carrying the offending names in the error;
    'expand' unions their codes under an explicit ('D+', atoms) key —
    recorded as an engineering choice, not author-join equivalence.
    """
    if multi_descriptor not in ('reject', 'expand'):
        raise ValueError('multi_descriptor must be reject|expand')
    names = [str(x) for x in disease_names]
    n = len(names)
    atoms = tuple(resolve_atoms(names, mesh_source))
    multi = [names[i] for i, a in enumerate(atoms) if len(a) > 1]
    if multi and multi_descriptor == 'reject':
        raise ValueError('multi-descriptor names under declared chain '
                         '(no silent union): %r' % sorted(set(multi)))
    keys = tuple(
        ('U', i) if not a else
        (('D', a[0]) if len(a) == 1 else ('D+', tuple(a)))
        for i, a in enumerate(atoms))
    key_rep = {}
    for i, k in enumerate(keys):
        key_rep.setdefault(k, i)
    # per unique key: codes + DV/tree
    key_dv, key_tree = {}, {}
    for k in key_rep:
        if k[0] == 'D':
            codes = mesh_source['name2codes'].get(k[1], [])
        elif k[0] == 'D+':
            codes = []
            for a_ in k[1]:
                codes.extend(mesh_source['name2codes'].get(a_, []))
        else:
            codes = []
        key_dv[k], key_tree[k] = disease_dv_tree(
            codes, mesh_source['code2name'])
    dv = np.array([key_dv[k] for k in keys])
    coverage = np.array([k[0] != 'U' for k in keys])
    # T via unique-key trees (deterministic sorted shared-ancestor sum)
    uk = list(key_rep)
    T = np.zeros((n, n))
    for i in range(n):
        ti = key_tree[keys[i]]
        si = set(ti)
        for j in range(i, n):
            shared = sorted(si & set(key_tree[keys[j]]))
            inter = sum(ti[a] + key_tree[keys[j]][a] for a in shared)
            T[i, j] = T[j, i] = (THETA_PAIR * inter
                                 / (dv[i] + dv[j])) if inter else 0.0
    # consistency: rows sharing a key must carry identical dv/T
    for i in range(n):
        r = key_rep[keys[i]]
        assert dv[i] == dv[r]
        assert (T[i] == T[r]).all()
    S = T.copy()
    for i in range(n):
        if not coverage[i]:
            S[i, i] = 1.0
    prov = dict(mesh_source['provenance'])
    prov['multi_descriptor_policy'] = multi_descriptor
    prov['n_multi_descriptor_names'] = len(set(multi))
    prov['uncovered_key_policy'] = "unique ('U',row) keys — never merged"
    return SemanticResult(T=T, S=S, dv=dv, coverage=coverage,
                          desc_keys=keys, key_rep=key_rep, atoms=atoms,
                          provenance=prov)
