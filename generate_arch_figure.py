"""Sinh hình minh hoạ kiến trúc DHGCMDA bằng matplotlib (không cần PowerPoint)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path


def draw():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def box(x, y, w, h, label, color):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle='round,pad=0.05',
                           linewidth=1.5,
                           edgecolor='#333',
                           facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label,
                ha='center', va='center', fontsize=9, fontweight='bold')

    def arrow(x1, y1, x2, y2, label=None):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->,head_length=0.3,head_width=0.2',
                            mutation_scale=12, color='#444', linewidth=1.2)
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                    ha='center', fontsize=7, style='italic')

    # Layer 0: Input similarity matrices
    box(0.2, 5.7, 1.8, 0.6, 'M_GSM\n(seq sim)', '#E6F0FA')
    box(0.2, 4.9, 1.8, 0.6, 'M_FSM\n(func sim)', '#E6F0FA')
    box(0.2, 1.2, 1.8, 0.6, 'D_SSM2\n(gene sim)', '#FAF0E6')
    box(0.2, 0.4, 1.8, 0.6, 'D_SSM1\n(sem sim)', '#FAF0E6')

    # Layer 1: Augmented features (concat with A)
    box(2.4, 5.3, 1.8, 0.6, 'X_m^(1) = [S_m^seq, A]', '#CFE2F3')
    box(2.4, 4.5, 1.8, 0.6, 'X_m^(2) = [S_m^func, A]', '#CFE2F3')
    box(2.4, 1.6, 1.8, 0.6, 'X_d^(1) = [S_d^gene, Aᵀ]', '#F4D9C0')
    box(2.4, 0.8, 1.8, 0.6, 'X_d^(2) = [S_d^sem, Aᵀ]', '#F4D9C0')

    # Layer 2: HGCN (KNN hypergraph + spectral conv)
    box(4.7, 5.3, 2.0, 0.6, 'HGCN view-1 miRNA', '#A4C8E1')
    box(4.7, 4.5, 2.0, 0.6, 'HGCN view-2 miRNA', '#A4C8E1')
    box(4.7, 1.6, 2.0, 0.6, 'HGCN view-1 disease', '#E8B89A')
    box(4.7, 0.8, 2.0, 0.6, 'HGCN view-2 disease', '#E8B89A')

    # Intra-CL labels
    ax.annotate('', xy=(6.7, 5.6), xytext=(6.7, 4.8),
                arrowprops=dict(arrowstyle='<->', color='#1A5490', linewidth=1.3))
    ax.text(6.85, 5.2, 'Intra-CL\n(view 1↔2)', fontsize=7, color='#1A5490', va='center')
    ax.annotate('', xy=(6.7, 1.9), xytext=(6.7, 1.1),
                arrowprops=dict(arrowstyle='<->', color='#A0522D', linewidth=1.3))
    ax.text(6.85, 1.5, 'Intra-CL\n(view 1↔2)', fontsize=7, color='#A0522D', va='center')

    # Layer 3: Attention-guided View Fusion
    box(7.7, 4.9, 2.0, 0.7, 'AVF: α_v σ + β_v sum\n→ Z_M', '#5A9BD3')
    box(7.7, 1.2, 2.0, 0.7, 'AVF: α_v σ + β_v sum\n→ Z_D', '#D98A5B')

    # Inter-modality CL
    ax.annotate('', xy=(8.7, 4.8), xytext=(8.7, 1.95),
                arrowprops=dict(arrowstyle='<->', color='#3F8E3F',
                                linewidth=1.5, linestyle='--'))
    ax.text(8.9, 3.4, 'Cross-modal CL\nInfoNCE + Margin', fontsize=8,
            color='#3F8E3F', va='center', fontweight='bold')

    # Layer 4: HGT
    box(10.3, 3.0, 1.7, 0.7, 'HGT (2 layers,\n4 heads)', '#FFD966')

    # Connections from AVF to HGT
    arrow(9.7, 5.0, 10.3, 3.4)
    arrow(9.7, 1.6, 10.3, 3.2)

    # Layer 5: Predictor
    box(10.3, 1.6, 1.7, 0.7, 'Bilinear\nPredictor', '#FFB266')
    arrow(11.1, 3.0, 11.1, 2.3)
    box(10.3, 0.4, 1.7, 0.7, 'P(exist) +\nP(type 1..C)', '#9FCC9F')
    arrow(11.1, 1.6, 11.1, 1.1)

    # Connect inputs through layers
    for y_in, y_out in [(5.9, 5.6), (5.1, 4.8), (1.5, 1.9), (0.7, 1.1)]:
        arrow(2.0, y_in, 2.4, y_out)
    for y_in, y_out in [(5.6, 5.6), (4.8, 4.8), (1.9, 1.9), (1.1, 1.1)]:
        arrow(4.2, y_in, 4.7, y_out)
    arrow(6.7, 5.6, 7.7, 5.3)
    arrow(6.7, 4.8, 7.7, 5.1)
    arrow(6.7, 1.9, 7.7, 1.5)
    arrow(6.7, 1.1, 7.7, 1.3)

    # Title
    ax.text(6.5, 6.8,
            'KIẾN TRÚC TỔNG QUAN DHGCMDA', ha='center', fontsize=13,
            fontweight='bold', color='#1F3A5F')
    ax.text(6.5, 6.5, '(Dual-View Hypergraph + Contrastive Learning + HGT)',
            ha='center', fontsize=10, style='italic', color='#555')

    # Legend
    legend_items = [
        mpatches.Patch(color='#A4C8E1', label='miRNA modality'),
        mpatches.Patch(color='#E8B89A', label='Disease modality'),
        mpatches.Patch(color='#FFD966', label='Heterogeneous fusion (HGT)'),
        mpatches.Patch(color='#9FCC9F', label='Output (existence + type)'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=8,
              framealpha=0.9, ncol=4)

    plt.tight_layout()
    out_path = Path('results') / 'architecture_overview.png'
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    draw()
