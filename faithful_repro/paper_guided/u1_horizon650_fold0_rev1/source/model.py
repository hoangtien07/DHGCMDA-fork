# -*- coding: utf-8 -*-
"""S1 supervised-only model — independent paper-guided engineering track.

Stage S1 (per docs/superpowers/specs/2026-11-14-paper-guided-dhgc-design.md):
dual-view hypergraph convolution + paper Eq16-18 adaptive view fusion +
static 4-relation PyG HGT stack + diagonal bilinear TYPE logits. NO
existence head, NO contrastive or reconstruction objectives, NO dynamic
graph updates — auxiliary branches cannot inject gradients in this stage
(projection heads exist but are never called by forward; fixtures assert
zero/None grads).

Scoped classes mirror inspected semantics (hetero_model.py +
main_experiments_hetero1.py, read-only — never imported/monkeypatched):

  HGNNConv:      out = G @ (x @ W + b)          (author HGNN_conv)
  HGCN branch:   LeakyReLU(0.25) after one conv (author HGCN; the author's
                 HGCN applies NO dropout — none added here)
  Views:         FOUR DISTINCT augmented inputs, per author lines
                 ~905-933:  mi_v1 = [S_seq | A_train],
                            mi_v2 = [S_func | A_train],
                            dis_v1 = [S_gene | A_train^T],
                            dis_v2 = [S_sem  | A_train^T].
                 Each view has its own KNN hypergraph Laplacian.
  Fusion (AVF):  paper Eq16-18 — per-view scalar gate
                 alpha_v = sigmoid(W2 ReLU(W1 GAP(Z_v))),
                 Z~_v = ReLU(alpha_v * Z_v),
                 Z_fused = sum_v beta_v Z~_v  with learnable beta scalars
                 (beta init 0.5 each, NOT normalized — declared; paper
                 specifies learnable coefficients without a constraint).
  HGT:           torch_geometric.nn.HGTConv over the static typed graph
                 (pinned PyG 2.6.1) + residual + LayerNorm + dropout
                 per destination node type. NOT a custom layer.
  Readout:       per-entity Linear+LayerNorm+Dropout projection, then
                 per-type diagonal bilinear
                 logit[p,t] = z_mi[i_p] . (r_t * z_dis[j_p]),
                 divided by a learned temperature (init 2.0, clamped
                 [0.5, 5.0] in forward; author softmax(logits/T)).
                 No type bias term (declared — absent justification).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class HGNNConv(nn.Module):
    """out = G @ (x @ W + b) — hypergraph-Laplacian propagation."""

    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_ft, out_ft))
        self.bias = nn.Parameter(torch.empty(out_ft)) if bias else None
        stdv = 1.0 / math.sqrt(out_ft)
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, G):
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return G @ x


class AdaptiveViewFusion(nn.Module):
    """Paper Eqs 16-18 attention-guided adaptive view fusion (one
    modality). Input-dependent per-view scalar gate:

        alpha_v = sigmoid(W2 @ ReLU(W1 @ GAP(Z_v)))     (Eq16)
        Z~_v    = ReLU(alpha_v * Z_v)                   (Eq17)
        Z_fused = sum_v beta_v * Z~_v                   (Eq18)

    GAP pools over the node axis -> d-vector; W1: d->d, W2: d->1.
    beta_v are learnable scalars initialized 0.5 (declared init; the
    paper does not constrain or normalize them).
    """

    def __init__(self, dim, n_views=2, beta_init=0.5):
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, 1)
        self.beta = nn.Parameter(torch.full((n_views,), float(beta_init)))

    def forward(self, z_list):
        zl = list(z_list)
        if len(zl) != self.beta.shape[0]:
            raise ValueError('view count mismatch')
        fused = 0.0
        gates = []
        for v, z in enumerate(zl):
            alpha = torch.sigmoid(self.W2(F.relu(self.W1(z.mean(0)))))
            gates.append(alpha)
            fused = fused + self.beta[v] * F.relu(alpha * z)
        return fused, torch.stack(gates).flatten()


class DualViewHGCN(nn.Module):
    """Two HGCN branches over DISTINCT view inputs + AVF fusion.

    forward(x1, G1, x2, G2) -> (z_fused, z1, z2, gates). z1/z2 kept for
    inspection and later-stage contrastive use; S1 uses only z_fused.
    """

    def __init__(self, in1, in2, hidden, leaky_slope=0.25):
        super().__init__()
        self.conv1 = HGNNConv(in1, hidden)
        self.conv2 = HGNNConv(in2, hidden)
        self.act = nn.LeakyReLU(leaky_slope)
        self.avf = AdaptiveViewFusion(hidden, n_views=2)

    def forward(self, x1, G1, x2, G2):
        z1 = self.act(self.conv1(x1, G1))
        z2 = self.act(self.conv2(x2, G2))
        z, gates = self.avf([z1, z2])
        return z, z1, z2, gates


class ProjectionHead(nn.Module):
    """fc1 hidden->proj (ELU), fc2 proj->hidden — S2 contrastive only.
    NEVER called by the S1 forward; fixtures assert zero/None grads."""

    def __init__(self, hidden, proj):
        super().__init__()
        self.fc1 = nn.Linear(hidden, proj)
        self.fc2 = nn.Linear(proj, hidden)

    def forward(self, z):
        return self.fc2(F.elu(self.fc1(z)))


class HGTLayer(nn.Module):
    """Pinned PyG HGTConv + residual + LayerNorm + dropout per dst type.

    conv(x, ei) supplies per-dst-type outputs; the wrapper applies
        z'_t = LayerNorm_t( x_t + Dropout(conv_out_t) )
    preserving node types that receive no messages (None passthrough).
    """

    def __init__(self, dim, metadata, heads=4, dropout=0.3):
        super().__init__()
        node_types, edge_types = metadata
        in_ch = {t: dim for t in node_types}
        self.conv = HGTConv(in_ch, dim, metadata, heads=heads)
        self.norm = nn.ModuleDict(
            {t: nn.LayerNorm(dim) for t in node_types})
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        res = {}
        for t in x_dict:
            o = out.get(t)
            res[t] = x_dict[t] if o is None else \
                self.norm[t](x_dict[t] + self.drop(o))
        return res


class DiagBilinearType(nn.Module):
    """Per-type diagonal bilinear readout on projected embeddings.
    logit[p, t] = z_mi[i_p] . (r_t * z_dis[j_p]) / T
    T: learned temperature, init 2.0, clamped to [0.5, 5.0] in forward
    (author divides logits by temperature before softmax). NO type
    bias — declared, absent justification.
    """

    def __init__(self, node_dim, readout, n_types, dropout=0.2,
                 temperature_init=2.0, temperature_min=0.5,
                 temperature_max=5.0):
        super().__init__()
        self.mi_proj = nn.Sequential(nn.Linear(node_dim, readout),
                                     nn.LayerNorm(readout),
                                     nn.Dropout(dropout))
        self.dis_proj = nn.Sequential(nn.Linear(node_dim, readout),
                                      nn.LayerNorm(readout),
                                      nn.Dropout(dropout))
        self.type_rel = nn.Parameter(torch.randn(n_types, readout) * 0.1)
        self.temperature = nn.Parameter(
            torch.tensor(float(temperature_init)))
        self.t_min = float(temperature_min)
        self.t_max = float(temperature_max)

    def forward(self, z_mi, z_dis, pair_idx):
        mi = self.mi_proj(z_mi)[pair_idx[:, 0]]
        di = self.dis_proj(z_dis)[pair_idx[:, 1]]
        logits = torch.einsum('pc,tc,pc->pt', mi, self.type_rel, di)
        return logits / self.temperature.clamp(self.t_min, self.t_max)


class PaperGuidedS1(nn.Module):
    """Full S1 forward graph.

    forward(inputs, G, edge_index_dict, pair_idx) ->
        dict(logits=(n_pairs, C), z_mi, z_dis, views, gates)

    inputs: {'mi_v1','mi_v2','dis_v1','dis_v2'} float tensors — the four
        DISTINCT augmented view inputs (train-only A inside, caller's
        contract). Per-view widths may differ.
    G: same keys, dense hypergraph Laplacians (static).
    edge_index_dict: {(nt,rel,nt): LongTensor(2,E)} static, train-only.
    """

    NODE_TYPES = ('mir', 'dis')
    RELS = (('mir', 'sim', 'mir'), ('dis', 'sim', 'dis'),
            ('mir', 'assoc', 'dis'), ('dis', 'rev_assoc', 'mir'))
    METADATA = (list(NODE_TYPES), list(RELS))

    def __init__(self, view_dims, n_types=5, hidden=256, readout=128,
                 proj=64, hgt_layers=2, hgt_heads=4, hgt_dropout=0.3,
                 readout_dropout=0.2, leaky_slope=0.25,
                 temperature_init=2.0, temperature_min=0.5,
                 temperature_max=5.0):
        super().__init__()
        for k in ('mi_v1', 'mi_v2', 'dis_v1', 'dis_v2'):
            if k not in view_dims:
                raise ValueError('view_dims missing ' + k)
        self.cfg = dict(view_dims=dict(view_dims), hidden=hidden,
                        readout=readout, proj=proj,
                        hgt_layers=hgt_layers, hgt_heads=hgt_heads,
                        n_types=n_types, hgt_dropout=hgt_dropout,
                        readout_dropout=readout_dropout,
                        temperature_init=temperature_init,
                        temperature_min=temperature_min,
                        temperature_max=temperature_max)
        self.mi_view = DualViewHGCN(view_dims['mi_v1'],
                                    view_dims['mi_v2'], hidden,
                                    leaky_slope)
        self.dis_view = DualViewHGCN(view_dims['dis_v1'],
                                     view_dims['dis_v2'], hidden,
                                     leaky_slope)
        self.mi_proj_head = ProjectionHead(hidden, proj)
        self.dis_proj_head = ProjectionHead(hidden, proj)
        self.hgt = nn.ModuleList([
            HGTLayer(hidden, self.METADATA, hgt_heads, hgt_dropout)
            for _ in range(hgt_layers)])
        self.readout = DiagBilinearType(
            hidden, readout, n_types, readout_dropout,
            temperature_init, temperature_min, temperature_max)

    def node_embeddings(self, inputs, G, edge_index_dict):
        z_mi, z1m, z2m, gm = self.mi_view(
            inputs['mi_v1'], G['mi_v1'], inputs['mi_v2'], G['mi_v2'])
        z_dis, z1d, z2d, gd = self.dis_view(
            inputs['dis_v1'], G['dis_v1'], inputs['dis_v2'], G['dis_v2'])
        x_dict = {'mir': z_mi, 'dis': z_dis}
        for layer in self.hgt:
            x_dict = layer(x_dict, edge_index_dict)
        return (x_dict['mir'], x_dict['dis'],
                (z1m, z2m, z1d, z2d), (gm, gd))

    def forward(self, inputs, G, edge_index_dict, pair_idx):
        z_mi, z_dis, views, gates = self.node_embeddings(
            inputs, G, edge_index_dict)
        logits = self.readout(z_mi, z_dis, pair_idx)
        return {'logits': logits, 'z_mi': z_mi, 'z_dis': z_dis,
                'views': views, 'gates': gates}
