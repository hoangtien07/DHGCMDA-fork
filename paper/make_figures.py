"""make_figures.py — publication figures + LaTeX tables for the manuscript.

Reads frozen results JSONs under forensics/ and writes:
  paper/figures/F1_lineage.{png,pdf}   — HMDD v3.2 dataset lineage funnel
  paper/figures/F2_leaderboard.{png,pdf} — frozen leaderboard by track
  paper/figures/F3_leak_channel.{png,pdf} — MISIM channel on SPLD vs vote
  paper/figures/F4_recall_ceiling.{png,pdf} — why claimed R=0.9421 is impossible
  paper/tables/T1_leaderboard.tex
  paper/tables/T2_bootstrap.tex
  paper/tables/T3_ablation.tex
  paper/tables/T4_robustness.tex
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F = '/home/ubuntu/repos/DHGCMDA-fork/forensics'
OUT_F = '/home/ubuntu/repos/DHGCMDA-fork/paper/figures'
OUT_T = '/home/ubuntu/repos/DHGCMDA-fork/paper/tables'
load = lambda n: json.load(open(os.path.join(F, n)))

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_HON, C_LEAK, C_LEG, C_FLOOR = '#2a7f62', '#c44536', '#7f6a93', '#999999'

rows = load('FROZEN_LEADERBOARD.json')
ORDER = ['vote_gip_only', 'vote_misim_dssm', 'vote_fmisim_dssm', 'vote_gip_gip',
         'simknn_j', 'vote_gip_dssm', 'SPLD_noMISIM',
         'SPLD_staticMISIM', 'SPLD_foldMISIM', 'DHGCMDA_encoder',
         'vote_dssm_only', 'cooccur_prior', 'bilinear', 'distmult']
LABEL = {'vote_gip_only': 'vote: GIP-mi only', 'vote_misim_dssm': 'vote: MISIM+DSSM (leak)',
         'vote_fmisim_dssm': 'vote: foldMISIM+DSSM', 'vote_gip_gip': 'vote: GIP+GIP',
         'simknn_j': 'vote: GIP+DSSM+J', 'vote_gip_dssm_j': 'vote: GIP+DSSM+J(0.2)',
         'vote_gip_dssm': 'vote: GIP+DSSM', 'SPLD_noMISIM': 'SPLD no-MISIM',
         'SPLD_staticMISIM': 'SPLD static-MISIM (publ.)', 'SPLD_foldMISIM': 'SPLD fold-MISIM',
         'DHGCMDA_encoder': 'DHGCMDA impl. (public)', 'vote_dssm_only': 'vote: DSSM only',
         'cooccur_prior': 'co-occurrence prior', 'bilinear': 'bilinear d32',
         'distmult': 'DistMult d32'}
TRACK_COLOR = {'honest': C_HON, 'LEAKY': C_LEAK}


def f1(m):
    return rows[m]['seed0']['legacy_f1']


def mstd(m):
    s = [rows[r_].get(s_) for r_ in (m,) for s_ in ('seed0', 'seed1', 'seed2')]
    v = [x['legacy_f1'] for x in s if x]
    return (np.mean(v), np.std(v)) if len(v) > 1 else (v[0], 0.0)


# ---------------- F1: dataset lineage ----------------
fig, ax = plt.subplots(figsize=(5.6, 2.6))
stages = [('Raw HMDD v3.2\n(cuilab)', 1049, 758, 18084),
          ('TDRC processing', 713, 447, 12534),
          ('MDAv3.2-3\n(paper artifact, recovered)', 411, 271, 11748)]
x = np.arange(3)
for i, (name, m, d, t) in enumerate(stages):
    ax.bar(x[i], m, 0.55, color='#4c7c9b', alpha=0.9, label='miRNAs' if i == 0 else None)
    ax.bar(x[i], d, 0.55, bottom=m, color='#d9a441', alpha=0.9,
           label='diseases' if i == 0 else None)
    ax.text(x[i], m / 2, str(m), ha='center', va='center', fontsize=8,
            color='white', fontweight='bold')
    ax.text(x[i], m + d / 2, str(d), ha='center', va='center', fontsize=8,
            color='white', fontweight='bold')
    ax.text(x[i], m + d + 40, f'{t:,} triplets\n({100*t/(m*d):.1f}% dense)',
            ha='center', fontsize=7.5)
ax.set_xticks(x, [s[0] for s in stages])
ax.set_ylabel('entities')
ax.set_ylim(0, 2050)
ax.legend(frameon=False, loc='upper right')
ax.set_title('HMDD v3.2 curation lineage — the 411×271 artifact was never released\n'
             '(recovered from Software Heritage; SHA256 c8f36c77…)')
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT_F}/F1_lineage.{ext}', bbox_inches='tight')
plt.close(fig)

# ---------------- F2: leaderboard ----------------
present = [m for m in ORDER if m in rows]
vals, errs, cols = [], [], []
for m in present:
    mu, sd = mstd(m)
    vals.append(mu)
    errs.append(sd)
    tr = rows[m]['track']
    cols.append(C_LEAK if tr == 'LEAKY' else (C_LEG if 'LEGACY' in tr else
                (C_FLOOR if m == 'cooccur_prior' else C_HON)))
fig, ax = plt.subplots(figsize=(5.8, 3.4))
ypos = np.arange(len(present))[::-1]
ax.barh(ypos, vals, xerr=errs, color=cols, alpha=0.92, height=0.72,
        error_kw=dict(lw=0.8, capsize=2))
for y, v, e in zip(ypos, vals, errs):
    ax.text(v + e + 0.006, y, f'{v:.3f}', va='center', fontsize=7.5)
ax.set_yticks(ypos, [LABEL[m] for m in present])
ax.set_xlabel('legacy Top-1 F1 (SPLD pair-fold, 5-fold outer)')
ax.set_xlim(0, 0.92)
ax.axvline(0.86, color='k', ls='--', lw=1)
ax.text(0.795, len(present) - 0.4, 'paper claim\n0.860 (impossible\nunder protocol)',
        fontsize=7, color='k')
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=C_HON, label='honest'),
                   mp.Patch(color=C_LEG, label='legacy (SPLD)'),
                   mp.Patch(color=C_LEAK, label='leaky control'),
                   mp.Patch(color=C_FLOOR, label='floor')],
          frameon=False, loc='lower right')
ax.set_title('Frozen leaderboard — MDAv3.2-3, seeds 0/1/2 for learned models\n'
             '(error bars = std across seeds where applicable)')
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT_F}/F2_leaderboard.{ext}', bbox_inches='tight')
plt.close(fig)

# ---------------- F3: leak channel ----------------
fig, ax = plt.subplots(figsize=(6.0, 2.9))
grp = [('SPLD\nstatic MISIM', 0.5585), ('SPLD\nfold MISIM', 0.5570),
       ('SPLD\nno MISIM', 0.5594), ('', np.nan), ('vote\nstatic MISIM', 0.5799),
       ('vote\nfold MISIM', 0.5733), ('vote\nGIP only', 0.5599)]
xs = np.arange(len(grp))
bcols = [C_LEG] * 3 + ['w'] + [C_LEAK, C_HON, C_HON]
for i, (name, v) in enumerate(grp):
    if not np.isnan(v):
        ax.bar(i, v, 0.62, color=bcols[i], alpha=0.92)
        ax.text(i, v + 0.004, f'{v:.4f}', ha='center', fontsize=8)
ax.set_xticks(xs, [g[0] for g in grp], fontsize=8)
ax.set_ylim(0.50, 0.62)
ax.set_ylabel('legacy Top-1 F1')
ax.annotate('Δ ≈ 0.002 (n.s.)', xy=(1, 0.578), ha='center', fontsize=8, color=C_LEG)
ax.annotate('Δ ≈ +0.020 (leak, 95% CI [0.013, 0.030])', xy=(5, 0.599), ha='center',
            fontsize=8, color=C_LEAK)
ax.set_title('MISIM channel quantification — leak inflates the vote\n'
             'but not the published tensor model (SPLD)')
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT_F}/F3_leak_channel.{ext}', bbox_inches='tight')
plt.close(fig)

# ---------------- F4: recall ceiling ----------------
fig, ax = plt.subplots(figsize=(6.0, 2.8))
items = [('claimed R\n(paper)', 0.9421, 'k'),
         ('macro-R\nprotocol ceiling', 0.8642, '#b03a2e'),
         ('micro-R\nprotocol ceiling', 0.7435, '#b03a2e'),
         ('best observed\nmacro-R (vote)', 0.5380, C_HON),
         ('SPLD macro-R\n(golden)', 0.5069, C_LEG)]
for i, (name, v, c) in enumerate(items):
    ax.bar(i, v, 0.6, color=c, alpha=0.9, hatch='//' if 'claim' in name else '')
    ax.text(i, v + 0.012, f'{v:.4f}', ha='center', fontsize=8.5, fontweight='bold')
ax.set_xticks(range(len(items)), [i[0] for i in items], fontsize=8)
ax.set_ylabel('Top-1 recall')
ax.set_ylim(0, 1.06)
ax.axhline(0.8642, color='#b03a2e', ls=':', lw=1)
ax.set_title('The claimed Top-1 recall (0.9421) exceeds the single-prediction\n'
             'protocol ceiling (0.8642) — unreachable under the documented eval')
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT_F}/F4_recall_ceiling.{ext}', bbox_inches='tight')
plt.close(fig)

# ---------------- tables ----------------
def tex_table(fname, caption, label, header, body_rows):
    n = len(header)
    lines = ['\\begin{table}[t]', '\\centering', '\\small',
             f'\\caption{{{caption}}}', f'\\label{{{label}}}',
             '\\begin{tabular}{@{}' + 'l' + 'r' * (n - 1) + '@{}}',
             '\\toprule',
             ' & '.join(header) + ' \\\\', '\\midrule']
    for r in body_rows:
        lines.append(' & '.join(str(c) for c in r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    open(os.path.join(OUT_T, fname), 'w').write('\n'.join(lines) + '\n')


# T1 leaderboard (frozen)
hdr = ['Model', 'Track', 'P', 'micro-R', 'macro-R', 'F1', 'mAUPR']
body = []
for m in present:
    a = rows[m]['seed0']
    body.append([LABEL[m], rows[m]['track'].replace('_', '\\_'),
                 f"{a['legacy_top1_precision']:.4f}", f"{a['legacy_micro_recall']:.4f}",
                 f"{a['legacy_macro_recall']:.4f}", f"\\textbf{{{a['legacy_f1']:.4f}}}",
                 '--' if a.get('macro_AUPR') is None or np.isnan(a['macro_AUPR'])
                 else f"{a['macro_AUPR']:.4f}"])
body.append(['\\midrule \\multicolumn{7}{@{}l@{}}{\\textit{DHGCMDA paper claim} '
             '— P=0.7915, R=0.9421, F1=0.8600 (unreachable: macro-R ceiling 0.8642)}', '', '', '', '', '', ''])
tex_table('T1_leaderboard.tex', 'Frozen leaderboard on MDAv3.2-3 under the golden '
          'SPLD pair-fold protocol (outer 5-fold). Tracks: honest '
          '(train-only similarities), legacy (SPLD configurations), '
          'leaky control (static MISIM), floor.', 'tab:leader', hdr, body)

# T2 bootstrap
tex_table('T2_bootstrap.tex',
          'Paired bootstrap (10,000 resamples over pooled outer-test pairs).',
          'tab:boot', ['Comparison', '$\\Delta$ F1', '95\\% CI', 'Verdict'],
          [['vote GIP+DSSM $-$ SPLD', '+0.0014', '[$-$0.0075, +0.0104]', 'tied'],
           ['vote MISIM+DSSM $-$ SPLD', '+0.0214', '[+0.0131, +0.0297]', 'significant (leak)'],
           ['DHGCMDA impl. $-$ SPLD', '$-$0.1104', '--', 'far below']])

# T3 ablation
su = load('final_suite_results.json')
hdr3 = ['miRNA sim', 'disease sim', 'J prior', 'F1', 'P', 'macro-R']
abl = [('—', '—', '—', None), ('GIP', '—', '—', 'gip_only'),
       ('—', 'DSSM', '—', 'dssm_only'), ('GIP', 'GIP', '—', 'gip_gip'),
       ('GIP', 'DSSM', '—', 'gip_dssm'), ('GIP', 'DSSM', '+', 'gip_dssm_j'),
       ('fold-MISIM', 'DSSM', '—', 'fmisim_dssm'), ('static MISIM (leak)', 'DSSM', '—', 'misim_dssm')]
body3 = []
for mi, di, j, key in abl:
    if key is None:
        continue
    a = su[key]['seed0']['aggregate']
    body3.append([mi, di, j, f"\\textbf{{{a['legacy_f1']:.4f}}}",
                  f"{a['legacy_top1_precision']:.4f}", f"{a['legacy_macro_recall']:.4f}"])
tex_table('T3_ablation.tex',
          'Vote-baseline ablation grid (golden pair-fold, seed-0 folds). '
          'The miRNA-side vote alone is the best observed honest model.',
          'tab:ablation', hdr3, body3)

# T4 robustness
hdr4 = ['Model', 'seed 0', 'seed 1', 'seed 2', 'mean $\\pm$ std']
bil = [0.2904, 0.2822, 0.2753]
dis = [0.2427, 0.2598, 0.2550]
gip = [0.5599, 0.5616, 0.5627]
def row(name, v):
    return [name] + [f'{x:.4f}' for x in v] + [f'{np.mean(v):.4f} $\\pm$ {np.std(v):.4f}']
tex_table('T4_robustness.tex',
          'Seed robustness. Learned models: fixed golden folds, init seeds 0--2. '
          'Vote: repeated pair-CV draws (secondary robustness; deterministic per split).',
          'tab:seeds', hdr4,
          [row('bilinear d32', bil), row('DistMult d32', dis),
           row('vote GIP+DSSM (rep-CV)', gip)])

print('figures:', sorted(os.listdir(OUT_F)))
print('tables:', sorted(os.listdir(OUT_T)))
