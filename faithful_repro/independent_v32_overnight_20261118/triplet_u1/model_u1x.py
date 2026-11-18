# -*- coding: utf-8 -*-
"""PaperGuidedU1X — frozen C4 encoder + diag-bilinear existence channel.

Port of upstream SimplifiedTypePredictor channel 0
(upstream_hetero_model.py L490-493): exist = sigmoid(mi_feat . (r_exist
* dis_feat)), computed on the SAME readout projections that feed the
type logits — single readout pass, shared mi_feat/dis_feat (upstream
shares them too; no temperature on the existence score, as upstream).

exist_rel is drawn inside torch.random.fork_rng under seed+4007 AFTER
the unchanged PaperGuidedC4 constructor, so every baseline parameter
and the caller's global RNG stay byte-identical to a C4 run (same
discipline as ReconstructionAdapter's seed+4001 fork). Fixture G2:
zero exist_rel -> forward logits bitwise-equal C4 logits.
"""
import math
import torch
import torch.nn as nn

from stages.c4_model import PaperGuidedC4

EXIST_SEED_OFFSET = 4007


class PaperGuidedU1X(PaperGuidedC4):
    """C4 + exist_rel (readout-dim) existence head; +exist in forward."""

    def __init__(self, view_dims, *args, exist_seed=0, readout=128,
                 **kwargs):
        super().__init__(view_dims, *args, readout=readout, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(exist_seed) + EXIST_SEED_OFFSET)
            self.readout.exist_rel = nn.Parameter(
                torch.randn(int(readout))
                * math.sqrt(2.0 / int(readout)))

    def _readout_x(self, z_mi, z_dis, pair_idx):
        ro = self.readout
        mi = ro.mi_proj(z_mi)[pair_idx[:, 0]]
        di = ro.dis_proj(z_dis)[pair_idx[:, 1]]
        logits = torch.einsum('pc,tc,pc->pt', mi, ro.type_rel, di)
        logits = logits / ro.temperature.clamp(ro.t_min, ro.t_max)
        exist = torch.sigmoid((mi * ro.exist_rel * di).sum(-1))
        return logits, exist

    def forward(self, inputs, G, edge_index_dict, pair_idx):
        (z_mi, z_dis, views, gates,
         zm_pre, zd_pre) = self.node_embeddings(
            inputs, G, edge_index_dict)
        logits, exist = self._readout_x(z_mi, z_dis, pair_idx)
        return {'logits': logits, 'exist': exist, 'z_mi': z_mi,
                'z_dis': z_dis, 'zm_pre': zm_pre, 'zd_pre': zd_pre,
                'views': views, 'gates': gates}
