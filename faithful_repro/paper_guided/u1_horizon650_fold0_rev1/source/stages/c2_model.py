# -*- coding: utf-8 -*-
"""C2 model — PaperGuidedS1 subclass exposing fused PRE-HGT embeddings
ZM/ZD (paper Eq. 10 input) from the SAME single encoder forward.

Scope: constructor and initialization are inherited UNCHANGED from
model.PaperGuidedS1 — no new parameters, no second stochastic forward,
no monkeypatching of the frozen class. Only node_embeddings/forward are
overridden to carry the pre-HGT AVF outputs (z_mi, z_dis — computed
inside the same forward) through to the output dict under 'zm_pre' /
'zd_pre'. lam_inter==0 callers never read them; the override does not
alter computation order, RNG consumption, or the graph.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import PaperGuidedS1          # frozen S1, read-only import


class PaperGuidedC2(PaperGuidedS1):
    """Same architecture/init as PaperGuidedS1; forward additionally
    returns 'zm_pre'/'zd_pre': the fused PRE-HGT embeddings (outputs of
    AdaptiveViewFusion, Eq. 16-18) that feed the HGT stack — the paper
    Eq. 10 inputs ZM/ZD for the inter-modality contrastive term."""

    def node_embeddings(self, inputs, G, edge_index_dict):
        z_mi, z1m, z2m, gm = self.mi_view(
            inputs['mi_v1'], G['mi_v1'], inputs['mi_v2'], G['mi_v2'])
        z_dis, z1d, z2d, gd = self.dis_view(
            inputs['dis_v1'], G['dis_v1'], inputs['dis_v2'], G['dis_v2'])
        zm_pre, zd_pre = z_mi, z_dis          # fused PRE-HGT (Eq. 10)
        x_dict = {'mir': z_mi, 'dis': z_dis}
        for layer in self.hgt:
            x_dict = layer(x_dict, edge_index_dict)
        return (x_dict['mir'], x_dict['dis'],
                (z1m, z2m, z1d, z2d), (gm, gd), zm_pre, zd_pre)

    def forward(self, inputs, G, edge_index_dict, pair_idx):
        z_mi, z_dis, views, gates, zm_pre, zd_pre = \
            self.node_embeddings(inputs, G, edge_index_dict)
        logits = self.readout(z_mi, z_dis, pair_idx)
        return {'logits': logits, 'z_mi': z_mi, 'z_dis': z_dis,
                'views': views, 'gates': gates,
                'zm_pre': zm_pre, 'zd_pre': zd_pre}
