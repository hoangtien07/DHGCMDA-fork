import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
import random
import os
import numpy as np
from functools import lru_cache
import time

# Check if PyTorch Geometric is available
try:
    from torch_geometric.nn import HGTConv, GCNConv, SAGEConv, HeteroDictLinear
    from torch_geometric.data import HeteroData

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


    class HGTConv(nn.Module):
        def __init__(self, in_channels, out_channels, metadata, heads):
            super().__init__()
            self.linear = nn.Linear(in_channels['miRNA'], out_channels)

        def forward(self, x_dict, edge_index_dict):
            return {key: self.linear(x) for key, x in x_dict.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InterViewContrastiveLoss(nn.Module):
    """跨模态视图间对比学习损失 - miRNA vs Disease (修复版)

    [VN] Cross-modal contrastive loss giữa miRNA và disease embedding.
    Mục tiêu: với cặp (miRNA i, disease j) CÓ association → kéo gần nhau
    trong embedding space; cặp KHÔNG association → đẩy xa.
    Kết hợp InfoNCE (SimCLR-style) + Margin ranking loss.
    Chi tiết toán học: xem docs/NOTES_MODEL.md §5
    """

    def __init__(self, temperature=0.5, margin=0.5):
        super(InterViewContrastiveLoss, self).__init__()
        self.tau = temperature
        self.margin = margin
        self.eps = 1e-8

    def forward(self, mi_embeddings, dis_embeddings, association_matrix):
        """
        计算miRNA-Disease跨模态对比学习损失

        Args:
            mi_embeddings: miRNA嵌入 [495, dim]
            dis_embeddings: Disease嵌入 [383, dim]
            association_matrix: 关联矩阵 [495, 383]

        Returns:
            inter_view_loss: 跨模态对比学习损失
        """
        # 确保在正确设备上
        target_device = mi_embeddings.device
        association_matrix = association_matrix.to(target_device).float()

        # 归一化嵌入
        mi_norm = F.normalize(mi_embeddings, dim=1, eps=self.eps)
        dis_norm = F.normalize(dis_embeddings, dim=1, eps=self.eps)

        # 计算跨模态相似度矩阵 [495 × 383]
        similarity = torch.mm(mi_norm, dis_norm.t()) / self.tau

        # 正样本mask：有关联的配对
        positive_mask = (association_matrix > 0).float()

        # 检查是否有正样本
        num_positives = positive_mask.sum()
        if num_positives == 0:
            # 如果没有正样本，返回零损失
            return torch.tensor(0.0, device=target_device, requires_grad=True)

        # 负样本mask：无关联的配对
        negative_mask = (association_matrix == 0).float()

        # 转换为概率分布（使用更稳定的方式）
        similarity_exp = torch.exp(torch.clamp(similarity, min=-10, max=10))

        # 对每个miRNA，计算InfoNCE损失
        # 正样本相似度
        positive_sim = similarity_exp * positive_mask

        # 负样本相似度（每行求和）
        negative_sim_sum = (similarity_exp * negative_mask).sum(dim=1, keepdim=True)

        # 对每个正样本计算损失
        # 分母 = 当前正样本 + 所有负样本
        losses = []
        for i in range(similarity.shape[0]):
            # 找到这个miRNA的所有正样本
            pos_indices = (positive_mask[i] > 0).nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue

            for j in pos_indices:
                # 正样本相似度
                pos_sim = similarity_exp[i, j]

                # 分母 = 这个正样本 + 这个miRNA的所有负样本
                denominator = pos_sim + negative_sim_sum[i, 0]

                # InfoNCE损失
                loss = -torch.log((pos_sim / (denominator + self.eps)) + self.eps)
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=target_device, requires_grad=True)

        inter_view_loss = torch.stack(losses).mean()

        # 额外的margin loss：正样本相似度 > 负样本相似度 + margin
        positive_pairs = similarity[positive_mask.bool()]
        negative_pairs = similarity[negative_mask.bool()]

        if len(positive_pairs) > 0 and len(negative_pairs) > 0:
            # 采样负样本以平衡计算
            num_neg_samples = min(len(negative_pairs), len(positive_pairs) * 10)
            sampled_neg = negative_pairs[torch.randperm(len(negative_pairs), device=target_device)[:num_neg_samples]]

            # Margin ranking loss
            pos_mean = positive_pairs.mean()
            neg_mean = sampled_neg.mean()
            margin_loss = F.relu(self.margin - (pos_mean - neg_mean))

            # 组合损失
            total_loss = inter_view_loss + 0.1 * margin_loss
        else:
            total_loss = inter_view_loss

        return total_loss


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_torch()


class HGNN_conv(nn.Module):
    """优化的超图神经网络卷积层 - 设备兼容版本

    [VN] 1 layer hypergraph convolution.
    Công thức: output = G · (x · W) + b
      - x [N, F]: feature của N node
      - W [F, F']: linear transform
      - G [N, N]: hypergraph Laplacian, xây offline bằng KNN
                  G = D_v^(-½) · H · W_e · D_e^(-1) · H^T · D_v^(-½)
    Khác GCN: H là INCIDENCE matrix (1 hyperedge chứa K node),
    cho phép capture high-order relation.
    """

    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        # 确保所有张量在同一设备上
        target_device = next(self.parameters()).device
        x = x.to(target_device).float()
        G = G.to(target_device).float()

        # 使用更高效的矩阵乘法
        x = torch.mm(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.mm(G, x)
        return x


class HGCN(nn.Module):
    """优化的超图卷积网络 - 设备兼容版本"""

    def __init__(self, in_dim, hidden_list, dropout=0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv(in_dim, hidden_list[0])
        self.activation = nn.LeakyReLU(0.25, inplace=True)

    def forward(self, x, G):
        # 确保设备一致性
        target_device = next(self.parameters()).device
        x = x.to(target_device).float()
        G = G.to(target_device).float()

        x_embed = self.hgnn1(x, G)
        x_embed_1 = self.activation(x_embed)
        return x_embed_1


class CL_HGCN(nn.Module):
    """优化的对比学习超图卷积网络 - 设备兼容版本

    [VN] Dual-view contrastive hypergraph conv network.
    Nhận 2 view (x1, adj1) và (x2, adj2) cùng 1 loại node (ví dụ miRNA).
    Mỗi view qua 1 HGCN riêng → z1, z2.
    Contrastive: kéo gần z1[i] và z2[i] (cùng node, khác view),
    đẩy xa khỏi z1[j]/z2[j] (node khác).
    Dùng SimCLR/NT-Xent loss với tau=0.5.
    """

    def __init__(self, in_size, hid_list, num_proj_hidden, alpha=0.5):
        super(CL_HGCN, self).__init__()
        self.hgcn1 = HGCN(in_size, hid_list)
        self.hgcn2 = HGCN(in_size, hid_list)

        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])

        self.tau = 0.5
        self.alpha = alpha
        self.eps = 1e-8

    def forward(self, x1, adj1, x2, adj2):
        # 确保设备一致性
        target_device = next(self.parameters()).device
        x1 = x1.to(target_device).float()
        adj1 = adj1.to(target_device).float()
        x2 = x2.to(target_device).float()
        adj2 = adj2.to(target_device).float()

        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)

        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)

        loss = self.alpha * self.sim(h1, h2) + (1 - self.alpha) * self.sim(h2, h1)
        return z1, z2, loss

    def projection(self, z):
        z = F.elu(self.fc1(z), inplace=True)
        return self.fc2(z)

    def norm_sim(self, z1, z2):
        # 优化的归一化相似度计算 - 确保设备一致
        target_device = next(self.parameters()).device
        z1 = F.normalize(z1.to(target_device).float(), dim=1, eps=self.eps)
        z2 = F.normalize(z2.to(target_device).float(), dim=1, eps=self.eps)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        # 优化的对比损失计算 - 确保设备一致
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))

        numerator = between_sim.diag()
        denominator = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()

        loss = -torch.log(numerator / (denominator + self.eps))
        return loss.mean()


class HGCN_Attention_Mechanism(nn.Module):
    """优化的注意力机制 - 设备兼容版本

    [VN] CẢNH BÁO: tên class gây hiểu nhầm — KHÔNG phải softmax attention!
    Thực chất chỉ là weighted sum với weights cố định 0.6/0.4.
    Đơn giản hóa để stable hơn trên dynamic graph.
    Muốn dùng attention thật → thay bằng multi-head attention module.
    """

    def __init__(self):
        super(HGCN_Attention_Mechanism, self).__init__()

    def forward(self, input_list):
        if not isinstance(input_list, list) or len(input_list) < 2:
            raise ValueError("Input must be a list with at least two elements")

        # 确定目标设备
        if hasattr(input_list[0], 'device'):
            target_device = input_list[0].device
        else:
            target_device = device

        feature1 = input_list[0].to(target_device).float()
        feature2 = input_list[1].to(target_device).float()

        # [VN] Weighted sum tĩnh. View 1 nhận weight cao hơn (0.6)
        # vì code coi View 1 (sequence/gene) là "primary", View 2 là "auxiliary".
        weight1 = 0.6
        weight2 = 0.4

        return weight1 * feature1 + weight2 * feature2


class SimpleHypergraphDecoder(nn.Module):
    """优化的超图解码器 - 设备兼容版本"""

    def __init__(self, hidden_dim, output_dim):
        super(SimpleHypergraphDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 确保设备一致性
        target_device = next(self.parameters()).device
        x = x.to(target_device).float()

        projected = self.projection(x)

        features_normalized = F.normalize(projected, p=2, dim=1, eps=1e-8)
        similarity = torch.mm(features_normalized, features_normalized.t())

        # 高效的输出矩阵创建
        output_sim = torch.eye(self.output_dim, device=target_device, dtype=torch.float32)

        valid_size = min(similarity.size(0), self.output_dim)
        if valid_size > 0:
            output_sim[:valid_size, :valid_size] = similarity[:valid_size, :valid_size]

        return output_sim


class EnhancedHGTLayer(nn.Module):


    def __init__(self, in_channels, out_channels, metadata, heads, dropout=0.1):
        super(EnhancedHGTLayer, self).__init__()
        self.hgt = HGTConv(in_channels, out_channels, metadata, heads)
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(out_channels)
            for node_type in metadata[0]
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # 确保所有输入在同一设备
        target_device = next(self.parameters()).device

        # 移动x_dict到目标设备
        x_dict_device = {}
        for node_type, x in x_dict.items():
            x_dict_device[node_type] = x.to(target_device).float()

        # 移动edge_index_dict到目标设备
        edge_index_dict_device = {}
        for edge_type, edge_index in edge_index_dict.items():
            edge_index_dict_device[edge_type] = edge_index.to(target_device)

        # 应用HGT卷积
        out_dict = self.hgt(x_dict_device, edge_index_dict_device)

        # 批量应用归一化和dropout
        for node_type in out_dict:
            out_dict[node_type] = self.dropout(self.norms[node_type](out_dict[node_type]))

        return out_dict


class AssociationPredictor(nn.Module):


    def __init__(self, node_dim, hidden_dim):
        super(AssociationPredictor, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        self.mi_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.dis_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, mi_embeddings, dis_embeddings):
        # 确保设备一致性
        target_device = next(self.parameters()).device
        mi_embeddings = mi_embeddings.to(target_device).float()
        dis_embeddings = dis_embeddings.to(target_device).float()

        # 预计算投影
        mi_proj = self.mi_projection(mi_embeddings)
        dis_proj = self.dis_projection(dis_embeddings)

        # 使用广播进行高效的配对计算
        mi_expanded = mi_proj.unsqueeze(1)
        dis_expanded = dis_proj.unsqueeze(0)

        # 广播拼接
        pair_embeds = torch.cat([
            mi_expanded.expand(-1, dis_proj.size(0), -1),
            dis_expanded.expand(mi_proj.size(0), -1, -1)
        ], dim=2)

        # 重塑并批量评分
        batch_size = pair_embeds.shape[0] * pair_embeds.shape[1]
        pair_embeds_flat = pair_embeds.view(batch_size, -1)

        scores_flat = self.scorer(pair_embeds_flat).squeeze(-1)
        scores = scores_flat.view(mi_embeddings.size(0), dis_embeddings.size(0))

        return scores




class SimplifiedTypePredictor(nn.Module):
    """
 类型预测器 - 使用双线性模型


    - score = miRNA^T @ diag(r_type) @ disease  (BilinearDiag风格)
    - 每个类型有独立的关系向量 (ComplEx风格)

    [VN] Classifier cuối. Cho mỗi cặp (miRNA i, disease j) tính:
      existence_score[i,j] = sigmoid(mi_feat[i] · diag(r_exist) · dis_feat[j]^T)
      type_logit[i,j,t]    = mi_feat[i] · diag(r_type_t) · dis_feat[j]^T
      type_prob[i,j]       = softmax(type_logit / T)  với T learnable
    Output shape: [mi_num, dis_num, 1 + num_types] = [..., 5]
    Type relation vectors được init ORTHOGONAL để tránh collapse.
    """

    def __init__(self, node_dim, hidden_dim, num_types=4, dropout=0.2, loss_mode='two_head'):
        super(SimplifiedTypePredictor, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_types = num_types
        self.loss_mode = loss_mode

        # 节点投影层
        self.mi_projector = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.dis_projector = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # 存在性预测的关系向量 (two_head mode dùng để compute exist_score riêng)
        self.exist_relation = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # 类型特定的关系向量 (核心创新)
        self.type_relations = nn.Parameter(torch.randn(num_types, hidden_dim) * 0.01)

        # Plan D Fix A++: r_no_assoc cho softmax_5class mode (class 0 = no association).
        # Chỉ dùng khi loss_mode='softmax_5class'; với two_head thì parameter này vẫn tồn tại
        # nhưng không xuất hiện trong loss path → gradient = 0, không ảnh hưởng.
        self.r_no_assoc = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # 🔥 新增: 可学习的temperature (提高多分类准确性)
        self.temperature = nn.Parameter(torch.tensor(2.0))

        self._initialize_parameters()

        print(f"   节点维度: {node_dim} → {hidden_dim}")
        print(f"   类型数量: {num_types}")
        print(f"   Temperature: 2.0 (可学习)")
        print(f"   Loss mode: {loss_mode}")

    def _initialize_parameters(self):
        """正交初始化类型关系向量"""
        with torch.no_grad():
            vectors = torch.randn(self.num_types, self.hidden_dim)
            q, _ = torch.linalg.qr(vectors.T)
            orthogonal = q.T[:self.num_types]
            scale = math.sqrt(2.0 / self.hidden_dim)
            self.type_relations.data = orthogonal * scale
            self.exist_relation.data = torch.randn(self.hidden_dim) * scale
            # Plan D: r_no_assoc init với cùng scale, KHÔNG orthogonal với type_relations
            # (no_assoc là class anti-correlated với types, không cần orthogonal)
            self.r_no_assoc.data = torch.randn(self.hidden_dim) * scale

    def forward(self, mi_embeddings, dis_embeddings):
        """
        双线性打分 (BilinearDiag风格)

        Args:
            mi_embeddings: [mi_num, node_dim]
            dis_embeddings: [dis_num, node_dim]

        Returns:
            scores: [mi_num, dis_num, 1 + num_types]
        """
        device = mi_embeddings.device
        mi_num, dis_num = mi_embeddings.size(0), dis_embeddings.size(0)

        # 特征投影
        mi_feat = self.mi_projector(mi_embeddings)  # [mi_num, hidden_dim]
        dis_feat = self.dis_projector(dis_embeddings)  # [dis_num, hidden_dim]

        temperature = torch.clamp(self.temperature, min=0.5, max=5.0)

        if self.loss_mode == 'softmax_5class':
            # Plan D Fix A++: 5-class softmax CE (Eq. 32 aligned).
            # Class 0 = no association, classes 1-4 = 4 types.
            # Trả về RAW LOGITS (không apply softmax) — F.cross_entropy tự handle.
            # Downstream code (Calculate_Metrics, case_study) detect bằng shape + dùng softmax derive.
            no_assoc_logit = torch.mm(mi_feat * self.r_no_assoc, dis_feat.t())  # [mi, dis]
            type_logits_list = []
            for type_idx in range(self.num_types):
                r_type = self.type_relations[type_idx]
                type_logits_list.append(torch.mm(mi_feat * r_type, dis_feat.t()))
            type_logits = torch.stack(type_logits_list, dim=2)  # [mi, dis, 4]
            # Concat no_assoc làm channel 0
            scores = torch.cat([
                no_assoc_logit.unsqueeze(-1),  # [mi, dis, 1] = no_assoc logit
                type_logits  # [mi, dis, 4] = type logits
            ], dim=-1) / temperature
            # scores shape [mi, dis, 5] — RAW LOGITS, downstream phải softmax
            return scores

        # two_head mode (Plan A/B/C path) — giữ nguyên behavior
        # 存在性预测: score = mi^T @ diag(r_exist) @ dis
        exist_scores = torch.sigmoid(
            torch.mm(mi_feat * self.exist_relation, dis_feat.t())
        )

        # 类型预测: 对每个类型计算 score_type = mi^T @ diag(r_type) @ dis
        type_logits = []
        for type_idx in range(self.num_types):
            r_type = self.type_relations[type_idx]
            type_score = torch.mm(mi_feat * r_type, dis_feat.t())
            type_logits.append(type_score)

        type_logits = torch.stack(type_logits, dim=2)  # [mi_num, dis_num, num_types]

        # 🔥 关键修改: 使用temperature scaling (提高多分类准确性)
        # 避免所有类型概率都接近0.25
        type_probs = F.softmax(type_logits / temperature, dim=2)

        # 组合结果
        scores = torch.cat([
            exist_scores.unsqueeze(-1),  # [mi_num, dis_num, 1]
            type_probs  # [mi_num, dis_num, num_types]
        ], dim=-1)

        return scores


class HeterogenousGraphCLAMIR(nn.Module):
    """
    [VN] Model chính của DHGCMDA.
    Kiến trúc 3 khối:
      1. Dual-view hypergraph encoder: CL_HGCN_mi + CL_HGCN_dis
      2. Inter-view contrastive + similarity reconstruction decoders
      3. HGT layers (nlayer=2) + SimplifiedTypePredictor

    Sơ đồ đầy đủ: docs/ARCHITECTURE.md
    """

    def __init__(self, mi_num, dis_num, hidden_list, num_proj_hidden, args):
        super(HeterogenousGraphCLAMIR, self).__init__()

        # 元数据定义
        self.node_types = ['miRNA', 'disease']
        self.edge_types = [
            ('miRNA', 'associates', 'disease'),
            ('disease', 'associates', 'miRNA'),
            ('miRNA', 'similar', 'miRNA'),
            ('disease', 'similar', 'disease')
        ]
        self.metadata = (self.node_types, self.edge_types)

        # 模型维度
        self.hidden_dim = hidden_list[0]
        self.mi_num = mi_num
        self.dis_num = dis_num

        # 缓存和优化参数
        self.edge_index_cache = {}
        self.similarity_threshold = 0.5
        self.graph_update_frequency = 5
        self.current_epoch = 0

        # [Ablation] paper Fig. 4 reproduction switch — default 'none' preserves original behavior
        self.ablation_mode = getattr(args, 'ablation', 'none')
        if self.ablation_mode != 'none':
            print(f"[ABLATION] Running with mode = {self.ablation_mode}")

        print(f"优化模型初始化: miRNA数量 = {mi_num}, 疾病数量 = {dis_num}")

        # 核心网络组件
        self.CL_HGCN_mi = CL_HGCN(mi_num + dis_num, hidden_list, num_proj_hidden)
        self.CL_HGCN_dis = CL_HGCN(dis_num + mi_num, hidden_list, num_proj_hidden)

        # 🆕 添加视图间对比学习
        # 🆕 添加跨模态视图间对比学习
        self.enable_inter_view_cl = getattr(args, 'enable_inter_view_cl', True)
        self.inter_view_weight = getattr(args, 'inter_view_weight', 0.3)
        if self.enable_inter_view_cl:
            # 跨模态对比学习：miRNA vs Disease
            self.inter_view_cl = InterViewContrastiveLoss(
                temperature=0.5,
                margin=0.5  # margin loss的边界值
            )

        # 简化的注意力机制
        self.AM_mi = HGCN_Attention_Mechanism()
        self.AM_dis = HGCN_Attention_Mechanism()

        # 优化的解码器
        self.miRNA_decoder = SimpleHypergraphDecoder(hidden_list[0], mi_num)
        self.disease_decoder = SimpleHypergraphDecoder(hidden_list[0], dis_num)

        # 节点特征变换器
        self.node_transformers = nn.ModuleDict({
            'miRNA': nn.Linear(hidden_list[0], self.hidden_dim),
            'disease': nn.Linear(hidden_list[0], self.hidden_dim)
        })

        # HGT层
        self.hgt_layers = nn.ModuleList()
        out_channels = (self.hidden_dim // args.n_head) * args.n_head
        if out_channels == 0:
            out_channels = args.n_head

        for _ in range(args.nlayer):
            layer = EnhancedHGTLayer(
                in_channels={node_type: self.hidden_dim for node_type in self.node_types},
                out_channels=out_channels,
                metadata=self.metadata,
                heads=args.n_head,
                dropout=args.dropout
            )
            self.hgt_layers.append(layer)

        # ============================================================
        # Plan E: True ablation rebuild modules (Fix C, backwards-compat)
        # ============================================================
        # no_cl_rebuild: HGCN plain single-view (no dual + no contrastive)
        if self.ablation_mode == 'no_cl_rebuild':
            self.HGCN_mi_plain = HGCN(mi_num + dis_num, hidden_list)
            self.HGCN_dis_plain = HGCN(dis_num + mi_num, hidden_list)
            print(f"[ABLATION REBUILD] no_cl_rebuild — HGCN plain single-view, no contrastive")

        # no_hgcn_rebuild: GCNConv thay HGNN_conv (edge_index thay G matrix)
        if self.ablation_mode == 'no_hgcn_rebuild':
            self.gcn_mi_v1 = GCNConv(mi_num + dis_num, hidden_list[0])
            self.gcn_mi_v2 = GCNConv(mi_num + dis_num, hidden_list[0])
            self.gcn_dis_v1 = GCNConv(dis_num + mi_num, hidden_list[0])
            self.gcn_dis_v2 = GCNConv(dis_num + mi_num, hidden_list[0])
            self._ei_cache = None  # Lazy init trong forward (build từ G matrix lần đầu)
            print(f"[ABLATION REBUILD] no_hgcn_rebuild — GCNConv với edge_index (threshold G > 0.1)")

        # no_hgt_rebuild: skip node_transformers + hgt_layers HOÀN TOÀN
        # Permanent projection layer (tránh per-step nn.Linear allocation bug ở line 819)
        if self.ablation_mode == 'no_hgt_rebuild':
            self.skip_proj_mi = nn.Linear(hidden_list[0], out_channels)
            self.skip_proj_dis = nn.Linear(hidden_list[0], out_channels)
            print(f"[ABLATION REBUILD] no_hgt_rebuild — skip transformers + HGT, projection trực tiếp")

        # 优化的关联预测器
        self.association_predictor = SimplifiedTypePredictor(
            node_dim=out_channels,
            hidden_dim=128,
            num_types=4,
            loss_mode=getattr(args, 'loss_mode', 'two_head')
        )

        self.dropout = nn.Dropout(args.dropout)

        # 4种类型的分布跟踪（基于实际数据）
        self.register_buffer('type_distribution',
                             torch.tensor([0.219, 0.094, 0.175, 0.407]))
        self.update_type_counts = True

    def _g_to_edge_index(self, G, threshold=0.1):
        """Plan E (no_hgcn_rebuild): convert hypergraph Laplacian G [N, N] → edge_index [2, E].

        Threshold để giảm số edges (tránh OOM). G được tính từ KNN nên đã sparse rồi —
        threshold=0.1 chỉ giữ edges có weight đáng kể. GCNConv tự thêm self-loops nên
        ta loại bỏ diagonal.
        """
        n = G.shape[0]
        edge_mask = (G.abs() > threshold)
        # Remove self-loops (GCNConv add_self_loops=True by default)
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=G.device)
        edge_mask = edge_mask & diag_mask
        edge_index = torch.nonzero(edge_mask, as_tuple=False).t().contiguous()
        return edge_index.long()

    def _get_association_matrix(self, concat_mi_tensor, concat_dis_tensor):
        """
        从拼接的特征张量中提取关联矩阵

        concat_mi_tensor: [mi_num, mi_num + dis_num]
        concat_dis_tensor: [dis_num, dis_num + mi_num]

        关联矩阵在 concat_mi_tensor 的前 dis_num 列
        """
        target_device = concat_mi_tensor.device

        # 从 concat_mi_tensor 中提取关联部分
        # concat_mi_tensor = [association_matrix, mi_features]
        association_matrix = concat_mi_tensor[:, :self.dis_num]

        return association_matrix.to(target_device).float()

    def forward(self, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km, hetero_data=None):
        """
        [VN] Forward pass chính — 5 giai đoạn:
          Stage 1: HGCN 2 view cho miRNA + disease (intra-view CL)
          Stage 2: Fusion weighted sum (0.6/0.4)
          Stage 3: Inter-view CL (miRNA ↔ disease)
          Stage 4: Similarity reconstruction (self-supervised)
          Stage 5: HGT layers + bilinear classifier

        Input shapes:
          concat_mi_tensor  [495, 878] = cat(assoc_matrix, m_ss) — View 1 của miRNA
          concat_dis_tensor [383, 878] = cat(assoc_matrix.T, d_gs) — View 1 của disease
          G_mi_Kn, G_mi_Km  [495, 495] — 2 hypergraph Laplacian cho miRNA
          G_dis_Kn, G_dis_Km [383, 383] — 2 hypergraph Laplacian cho disease

        Output:
          score [495, 383, 5] — [exist_prob, p_type1, p_type2, p_type3, p_type4]
          mi_cl_loss, dis_cl_loss — scalar (intra + inter_view_weight·inter)
          mi_sim_recon [495, 495], dis_sim_recon [383, 383]
        Chi tiết: xem docs/NOTES_DATAFLOW.md §6 và docs/ARCHITECTURE.md
        """
        # 确定目标设备
        target_device = next(self.parameters()).device

        # 确保所有输入张量在正确设备上
        concat_mi_tensor = concat_mi_tensor.to(target_device).float()
        concat_dis_tensor = concat_dis_tensor.to(target_device).float()
        G_mi_Kn = G_mi_Kn.to(target_device).float()
        G_mi_Km = G_mi_Km.to(target_device).float()
        G_dis_Kn = G_dis_Kn.to(target_device).float()
        G_dis_Km = G_dis_Km.to(target_device).float()

        # [Ablation] no_dv: dùng cùng 1 hypergraph cho cả 2 view → mất tính dual-view
        if self.ablation_mode == 'no_dv':
            G_mi_Km = G_mi_Kn
            G_dis_Km = G_dis_Kn

        # [Ablation] no_hgcn: thay G (hypergraph Laplacian) bằng identity → degenerate HGCN thành MLP
        if self.ablation_mode == 'no_hgcn':
            G_mi_Kn = torch.eye(self.mi_num, device=target_device, dtype=torch.float32)
            G_mi_Km = torch.eye(self.mi_num, device=target_device, dtype=torch.float32)
            G_dis_Kn = torch.eye(self.dis_num, device=target_device, dtype=torch.float32)
            G_dis_Km = torch.eye(self.dis_num, device=target_device, dtype=torch.float32)

        try:
            # 阶段1: 超图特征提取（视图内对比学习）
            # Plan E rebuild branches:
            if self.ablation_mode == 'no_cl_rebuild':
                # HGCN plain single-view, no dual + no contrastive
                mi_feature1 = self.HGCN_mi_plain(concat_mi_tensor, G_mi_Kn)
                mi_feature2 = mi_feature1
                mi_intra_loss = torch.tensor(0.0, device=target_device, requires_grad=True)
                dis_feature1 = self.HGCN_dis_plain(concat_dis_tensor, G_dis_Kn)
                dis_feature2 = dis_feature1
                dis_intra_loss = torch.tensor(0.0, device=target_device, requires_grad=True)
            elif self.ablation_mode == 'no_hgcn_rebuild':
                # GCNConv với edge_index thay HGNN_conv (G matrix). Lazy-init edge_index cache.
                if self._ei_cache is None:
                    self._ei_cache = {
                        'mi_v1': self._g_to_edge_index(G_mi_Kn, threshold=0.1),
                        'mi_v2': self._g_to_edge_index(G_mi_Km, threshold=0.1),
                        'dis_v1': self._g_to_edge_index(G_dis_Kn, threshold=0.1),
                        'dis_v2': self._g_to_edge_index(G_dis_Km, threshold=0.1),
                    }
                    print(f"[no_hgcn_rebuild] edge counts: mi_v1={self._ei_cache['mi_v1'].shape[1]}, "
                          f"mi_v2={self._ei_cache['mi_v2'].shape[1]}, "
                          f"dis_v1={self._ei_cache['dis_v1'].shape[1]}, "
                          f"dis_v2={self._ei_cache['dis_v2'].shape[1]}")
                mi_feature1 = F.leaky_relu(self.gcn_mi_v1(concat_mi_tensor, self._ei_cache['mi_v1']), 0.25)
                mi_feature2 = F.leaky_relu(self.gcn_mi_v2(concat_mi_tensor, self._ei_cache['mi_v2']), 0.25)
                mi_intra_loss = torch.tensor(0.0, device=target_device, requires_grad=True)
                dis_feature1 = F.leaky_relu(self.gcn_dis_v1(concat_dis_tensor, self._ei_cache['dis_v1']), 0.25)
                dis_feature2 = F.leaky_relu(self.gcn_dis_v2(concat_dis_tensor, self._ei_cache['dis_v2']), 0.25)
                dis_intra_loss = torch.tensor(0.0, device=target_device, requires_grad=True)
            else:
                # Original CL_HGCN dual-view + contrastive (default + no_hgt_rebuild + legacy ablations)
                mi_feature1, mi_feature2, mi_intra_loss = self.CL_HGCN_mi(
                    concat_mi_tensor, G_mi_Kn, concat_mi_tensor, G_mi_Km)
                dis_feature1, dis_feature2, dis_intra_loss = self.CL_HGCN_dis(
                    concat_dis_tensor, G_dis_Kn, concat_dis_tensor, G_dis_Km)

            # [Ablation] no_avf: thay attention-guided view fusion bằng simple average
            if self.ablation_mode == 'no_avf':
                mi_feature_fused = (mi_feature1 + mi_feature2) / 2.0
                dis_feature_fused = (dis_feature1 + dis_feature2) / 2.0
            else:
                mi_feature_fused = self.AM_mi([mi_feature1, mi_feature2])
                dis_feature_fused = self.AM_dis([dis_feature1, dis_feature2])

            # 🆕 跨模态视图间对比学习（miRNA vs Disease）
            # Plan E: disable inter-view CL khi no_cl_rebuild để full bypass contrastive
            inter_view_loss = torch.tensor(0.0, device=target_device)
            if self.enable_inter_view_cl and self.training and self.ablation_mode != 'no_cl_rebuild':
                # 直接从concat_mi_tensor的前dis_num列提取关联矩阵
                # 但要确保数值在合理范围内（0或正数）
                association_matrix_extracted = concat_mi_tensor[:, :self.dis_num].clone()

                # 二值化：将所有正值视为1（有关联），0保持为0（无关联）
                association_matrix_binary = (association_matrix_extracted > 0).float()

                # 检查是否有正样本
                if association_matrix_binary.sum() > 0:
                    # 使用融合后的特征进行跨模态对比学习
                    inter_view_loss = self.inter_view_cl(
                        mi_feature_fused,
                        dis_feature_fused,
                        association_matrix_binary
                    )
                else:
                    print("⚠️ Warning: No positive associations found for inter-view CL")

            # 合并视图内和跨模态视图间损失
            mi_cl_loss = mi_intra_loss + self.inter_view_weight * inter_view_loss
            dis_cl_loss = dis_intra_loss + self.inter_view_weight * inter_view_loss

            # [Ablation] no_cl: zero-out cả intra + inter contrastive losses
            if self.ablation_mode == 'no_cl':
                mi_cl_loss = torch.tensor(0.0, device=target_device, requires_grad=True)
                dis_cl_loss = torch.tensor(0.0, device=target_device, requires_grad=True)

            # 阶段2: 相似性重构（仅在训练时）
            if self.training:
                mi_sim_reconstructed = self.miRNA_decoder(mi_feature_fused)
                dis_sim_reconstructed = self.disease_decoder(dis_feature_fused)
            else:
                # 测试时使用缓存的单位矩阵
                mi_sim_reconstructed = torch.eye(self.mi_num, device=target_device, dtype=torch.float32)
                dis_sim_reconstructed = torch.eye(self.dis_num, device=target_device, dtype=torch.float32)

            # 阶段3: 异构图处理
            # Plan E no_hgt_rebuild: bypass node_transformers + hgt_layers HOÀN TOÀN
            if self.ablation_mode == 'no_hgt_rebuild':
                # Project fused features trực tiếp về out_channels (skip transformers + HGT)
                mi_embeddings = self.skip_proj_mi(mi_feature_fused)
                dis_embeddings = self.skip_proj_dis(dis_feature_fused)
                # Predictor needs to be called outside this branch — fall through bằng cách set hetero_data=None tạm
                # và đi vào fallback branch dưới? Không — đặt flag riêng.
                _no_hgt_rebuild_path = True
                score = self.association_predictor(mi_embeddings, dis_embeddings)
                return score, mi_cl_loss, dis_cl_loss, mi_sim_reconstructed, dis_sim_reconstructed

            if hetero_data is not None:
                # 变换节点特征
                x_dict = {
                    'miRNA': self.node_transformers['miRNA'](mi_feature_fused),
                    'disease': self.node_transformers['disease'](dis_feature_fused)
                }

                # [Ablation] no_hgt: skip HGT layers, dùng node_transformers output trực tiếp
                if self.ablation_mode == 'no_hgt':
                    mi_embeddings = x_dict['miRNA']
                    dis_embeddings = x_dict['disease']
                    # Project to out_channels nếu shape khác (cho safe khi hidden_dim != out_channels)
                    expected_dim = self.association_predictor.node_dim if hasattr(
                        self.association_predictor, 'node_dim') else mi_embeddings.shape[-1]
                    if mi_embeddings.shape[-1] != expected_dim:
                        proj = nn.Linear(mi_embeddings.shape[-1], expected_dim).to(target_device)
                        mi_embeddings = proj(mi_embeddings)
                        dis_embeddings = proj(dis_embeddings)
                else:
                    # 获取边索引并确保设备一致
                    edge_index_dict = self._get_edge_index_dict_device_safe(hetero_data, target_device)

                    # 通过HGT层处理
                    for layer in self.hgt_layers:
                        x_dict = layer(x_dict, edge_index_dict)

                    # 获取最终节点嵌入
                    mi_embeddings = x_dict['miRNA']
                    dis_embeddings = x_dict['disease']

                # 阶段4: 关联预测
                score = self.association_predictor(mi_embeddings, dis_embeddings)

            else:
                # 备份方案：简单点积
                print("WARNING: No heterogeneous graph data provided. Using fallback prediction method.")
                mi_proj = F.normalize(mi_feature_fused, p=2, dim=1)
                dis_proj = F.normalize(dis_feature_fused, p=2, dim=1)
                binary_score = torch.mm(mi_proj, dis_proj.t())
                binary_score = (binary_score + 1) / 2  # 缩放到[0, 1]

                # 创建默认类型概率
                type_probs = torch.zeros(binary_score.shape[0], binary_score.shape[1], 3,
                                         device=target_device, dtype=torch.float32)
                type_probs[:, :, 0] = 0.2
                type_probs[:, :, 1] = 0.1
                type_probs[:, :, 2] = 0.7

                score = torch.cat([binary_score.unsqueeze(-1), type_probs], dim=-1)

            return score, mi_cl_loss, dis_cl_loss, mi_sim_reconstructed, dis_sim_reconstructed

        except Exception as e:
            print(f"Forward pass error: {e}")
            import traceback
            traceback.print_exc()

            # 创建默认结果
            score = torch.zeros(self.mi_num, self.dis_num, 4, device=target_device, dtype=torch.float32)
            score[:, :, 0] = 0.5  # 默认存在概率
            score[:, :, 1] = 0.2
            score[:, :, 2] = 0.1
            score[:, :, 3] = 0.7

            mi_cl_loss = torch.tensor(0.0, device=target_device, dtype=torch.float32)
            dis_cl_loss = torch.tensor(0.0, device=target_device, dtype=torch.float32)
            mi_sim_reconstructed = torch.eye(self.mi_num, device=target_device, dtype=torch.float32)
            dis_sim_reconstructed = torch.eye(self.dis_num, device=target_device, dtype=torch.float32)

            return score, mi_cl_loss, dis_cl_loss, mi_sim_reconstructed, dis_sim_reconstructed

    def _get_edge_index_dict_device_safe(self, hetero_data, target_device):
        """获取边索引字典并确保设备一致性"""
        edge_index_dict = {}

        if hasattr(hetero_data, 'edge_index_dict'):
            for edge_type, edge_index in hetero_data.edge_index_dict.items():
                edge_index_dict[edge_type] = edge_index.to(target_device)
        else:
            # 处理PyTorch Geometric格式
            for edge_type in self.edge_types:
                try:
                    if hasattr(hetero_data, '__getitem__') and edge_type in hetero_data:
                        edge_index_dict[edge_type] = hetero_data[edge_type].edge_index.to(target_device)
                    else:
                        # 创建空边索引
                        edge_index_dict[edge_type] = torch.zeros((2, 0), dtype=torch.long, device=target_device)
                except:
                    # 创建空边索引作为后备
                    edge_index_dict[edge_type] = torch.zeros((2, 0), dtype=torch.long, device=target_device)

        return edge_index_dict

    def _update_hetero_data_optimized(self, hetero_data, mi_sim, dis_sim):
        if not self.training:
            return hetero_data

        # 检查是否需要更新
        self.current_epoch += 1
        if self.current_epoch % self.graph_update_frequency != 0:
            return hetero_data

        try:
            target_device = next(self.parameters()).device

            import copy

            # 高效的深拷贝
            if TORCH_GEOMETRIC_AVAILABLE:
                new_hetero_data = copy.deepcopy(hetero_data)
            else:
                new_hetero_data = type('HeteroDataSimple', (), {})()
                new_hetero_data.edge_index_dict = {}
                if hasattr(hetero_data, 'x_dict'):
                    new_hetero_data.x_dict = copy.deepcopy(hetero_data.x_dict)

            threshold = self.similarity_threshold

            # 确保相似性矩阵在正确设备上
            mi_sim = mi_sim.to(target_device)
            dis_sim = dis_sim.to(target_device)

            # 矢量化边创建
            mi_mask = (mi_sim > threshold) & (torch.eye(self.mi_num, device=target_device) == 0)
            mi_indices = torch.nonzero(mi_mask, as_tuple=True)

            dis_mask = (dis_sim > threshold) & (torch.eye(self.dis_num, device=target_device) == 0)
            dis_indices = torch.nonzero(dis_mask, as_tuple=True)

            # 根据可用性更新边索引
            if TORCH_GEOMETRIC_AVAILABLE:
                if len(mi_indices[0]) > 0:
                    new_hetero_data['miRNA', 'similar', 'miRNA'].edge_index = torch.stack(mi_indices)
                else:
                    new_hetero_data['miRNA', 'similar', 'miRNA'].edge_index = torch.zeros((2, 0), dtype=torch.long,
                                                                                          device=target_device)

                if len(dis_indices[0]) > 0:
                    new_hetero_data['disease', 'similar', 'disease'].edge_index = torch.stack(dis_indices)
                else:
                    new_hetero_data['disease', 'similar', 'disease'].edge_index = torch.zeros((2, 0), dtype=torch.long,
                                                                                              device=target_device)
            else:
                if len(mi_indices[0]) > 0:
                    new_hetero_data.edge_index_dict[('miRNA', 'similar', 'miRNA')] = torch.stack(mi_indices)
                else:
                    new_hetero_data.edge_index_dict[('miRNA', 'similar', 'miRNA')] = torch.zeros((2, 0),
                                                                                                 dtype=torch.long,
                                                                                                 device=target_device)

                if len(dis_indices[0]) > 0:
                    new_hetero_data.edge_index_dict[('disease', 'similar', 'disease')] = torch.stack(dis_indices)
                else:
                    new_hetero_data.edge_index_dict[('disease', 'similar', 'disease')] = torch.zeros((2, 0),
                                                                                                     dtype=torch.long,
                                                                                                     device=target_device)

                # 保留现有的关联边
                if hasattr(hetero_data, 'edge_index_dict'):
                    for k, v in hetero_data.edge_index_dict.items():
                        if k not in [('miRNA', 'similar', 'miRNA'), ('disease', 'similar', 'disease')]:
                            new_hetero_data.edge_index_dict[k] = v.to(target_device)

            return new_hetero_data

        except Exception as e:
            print(f"Error updating heterogeneous graph: {e}")
            return hetero_data


# 导出的类
__all__ = [
    'HGNN_conv', 'HGCN', 'CL_HGCN', 'HGCN_Attention_Mechanism',
    'SimpleHypergraphDecoder', 'EnhancedHGTLayer', 'AssociationPredictor',
    'SimplifiedTypePredictor', 'HeterogenousGraphCLAMIR'
]


AssociationTypePredictor = SimplifiedTypePredictor
BayesianTypePredictor = SimplifiedTypePredictor  # 保持向后兼容
OptimizedBayesianTypePredictor = SimplifiedTypePredictor  # 兼容性别名