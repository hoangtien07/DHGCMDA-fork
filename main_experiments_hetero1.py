import os
# CPU thread tuning — phải set trước torch/numpy import để MKL/OpenMP nhận ra
# Xeon E5-2680 v4: 14 physical cores; HyperThreading thường ko giúp compute-bound
_N_THREADS = os.environ.get('DHGCMDA_N_THREADS', '14')
os.environ.setdefault('OMP_NUM_THREADS', _N_THREADS)
os.environ.setdefault('MKL_NUM_THREADS', _N_THREADS)
os.environ.setdefault('OPENBLAS_NUM_THREADS', _N_THREADS)
os.environ.setdefault('NUMEXPR_NUM_THREADS', _N_THREADS)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from functools import lru_cache
import time
from collections import defaultdict

from torch_geometric.data import HeteroData

import hypergraph_construct_KNN
import hypergraph_construct_kmeans
from hetero_model import TORCH_GEOMETRIC_AVAILABLE, HeterogenousGraphCLAMIR

# Runtime intra-/inter-op threads (override PyTorch default heuristic)
try:
    torch.set_num_threads(int(_N_THREADS))
    torch.set_num_interop_threads(min(4, int(_N_THREADS)))
    print(f"[CPU] torch.set_num_threads({_N_THREADS}), interop=min(4, {_N_THREADS})")
except RuntimeError as _e:
    # set_num_interop_threads chỉ gọi được 1 lần per process
    print(f"[CPU] thread tuning skipped: {_e}")

# 启用JIT编译以加速
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
warnings.filterwarnings('ignore')

# 检查是否可以使用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 启用优化选项
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# 设置随机种子
def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_torch()


# ============================================================================
# 简化的损失函数 - 受 MRFGMDA 项目启发
# 使用简单的 Cross Entropy + Focal Loss
# ============================================================================

class SimplifiedMultiTypeAssociationLoss(nn.Module):
    """简化的多类型损失 - 简洁高效

    [VN] Loss tổng hợp cho bài toán multi-type association prediction.
    Tổng loss = 0.3 * existence_loss + 0.7 * type_loss

    existence_loss: FOCAL LOSS xử lý class imbalance giữa có/không associaton.
      pos: -(1-p)^gamma · log(p)  với gamma=2.0
      neg: -p^gamma · log(1-p) · alpha  với alpha=0.5

    type_loss: WEIGHTED CROSS-ENTROPY với label smoothing 0.1
      class_weights computed bằng Effective Number formula (beta=0.99999)
      → minority classes (Epigenetics: 157 samples) được weight cao hơn
      → majority (Genetics: 681 samples) weight thấp hơn

    Chi tiết: xem docs/NOTES_MODEL.md §10
    """

    def __init__(self, args, model=None):
        super(SimplifiedMultiTypeAssociationLoss, self).__init__()
        self.alpha = args.alpha
        self.device = device
        self.loss_mode = getattr(args, 'loss_mode', 'two_head')

        # Effective Number 类别权重 — chuyển dynamic theo dataset
        ds = getattr(args, 'dataset', 'v2.0_495m383D')
        if ds == 'v3.2_processed':
            counts = [3216, 575, 6052, 1820, 5952]  # 5 types v3.2 GIP
        elif ds == 'v3.2_wang':
            # v3.2 Wang TDRC preprocessing — actual counts after preprocess_v32_wang.py
            # circu=3310, epic=411, target=4810, genetic=850, tissue=3153
            counts = [3310, 411, 4810, 850, 3153]
        elif ds == 'v3.2_wang_multilabel':
            # Plan H-3: multi-label preserved counts (per-type sum, not collapsed)
            # circu=3310, epic=519, target=5844, genetic=1581, tissue=5087
            counts = [3310, 519, 5844, 1581, 5087]
        else:
            counts = [367, 157, 293, 681]  # 4 types v2.0
        beta = 0.99999
        effective_nums = [(1 - beta ** n) / (1 - beta) for n in counts]
        raw_weights = [1.0 / en for en in effective_nums]
        sum_weights = sum(raw_weights)
        normalized_weights = [w * len(counts) / sum_weights for w in raw_weights]

        self.register_buffer('class_weights',
                             torch.tensor(normalized_weights, device=self.device))

        # Plan D: 5-class weights cho softmax_5class mode.
        # neg_count xấp xỉ 10× positive (alpha=10 ratio sampling typical).
        # Tổng positive ~1498, neg sampled ~14980. neg weight nhỏ hơn types để minority types không bị neg ăn hết signal.
        neg_count = sum(counts) * 10  # ratio sampling mặc định
        counts_5 = [neg_count] + counts  # [neg, circ, epi, target, genetic]
        effective_nums_5 = [(1 - beta ** n) / (1 - beta) for n in counts_5]
        raw_weights_5 = [1.0 / en for en in effective_nums_5]
        sum_weights_5 = sum(raw_weights_5)
        normalized_weights_5 = [w * len(counts_5) / sum_weights_5 for w in raw_weights_5]
        self.register_buffer('class_weights_5',
                             torch.tensor(normalized_weights_5, device=self.device))

        self.focal_gamma = 2.0
        # Eq. 32 alignment sweep: exist_weight ∈ {0.0, 0.05, 0.1, 0.3} via CLI
        self.label_smoothing = 0.1
        self.exist_weight = float(getattr(args, 'exist_weight', 0.3))
        self.type_weight = 1.0 - self.exist_weight if self.exist_weight < 1.0 else 0.0

        # Plan F: paper_literal mode override — strict Eq. 32 implementation
        # Disable mọi trick: focal, class_weights, label_smoothing, existence loss.
        # Chỉ giữ plain CE cho type prediction. CL + recon đã match paper bên ngoài forward.
        if self.loss_mode == 'paper_literal':
            self.class_weights = torch.ones_like(self.class_weights)
            self.class_weights_5 = torch.ones_like(self.class_weights_5)
            self.focal_gamma = 0.0
            self.label_smoothing = 0.0
            self.exist_weight = 0.0
            self.type_weight = 1.0
            print(f"   [Plan F paper_literal] OVERRIDE: uniform class_weights, focal_gamma=0, "
                  f"label_smoothing=0, exist_weight=0 → plain CE cho type only")

        print(f"   loss_mode={self.loss_mode}")
        if self.loss_mode == 'two_head':
            print(f"   exist_weight={self.exist_weight:.3f}, type_weight={self.type_weight:.3f}")
        elif self.loss_mode == 'paper_literal':
            print(f"   [paper_literal] exist_weight=0.0, type_weight=1.0, plain CE, no focal/CW/LS")
        else:
            print(f"   class_weights_5 (neg+4types): {[f'{w:.3f}' for w in normalized_weights_5]}")

        print(f"\n✅ SimplifiedMultiTypeAssociationLoss 初始化 (改进版)")
        print(f"   类别权重 (beta=0.99999): {[f'{w:.3f}' for w in normalized_weights]}")

    def forward(self, one_index, zero_index, predictions, targets):
        predictions = predictions.to(self.device).float()
        targets = targets.to(self.device).float()

        # Plan D Fix A++: 5-class softmax CE branch
        if self.loss_mode == 'softmax_5class':
            return self._compute_softmax5_loss(one_index, zero_index, predictions, targets)

        # Plan G-1: Multi-label BCE branch
        if self.loss_mode == 'multilabel_bce':
            return self._compute_multilabel_bce_loss(one_index, zero_index, predictions, targets)

        # two_head mode (Plan A/B/C path) — giữ nguyên
        # 兼容2D和3D输入
        if len(predictions.shape) == 2:
            # 2D输入 [mi_num, dis_num] - 只有存在性预测
            exist_pred = predictions
            type_pred = None
            print("⚠️ 警告: 损失函数接收到2D输入，只计算存在性损失")
        elif len(predictions.shape) == 3:
            # 3D输入 [mi_num, dis_num, 5] - 存在性 + 类型预测
            exist_pred = predictions[:, :, 0]
            type_pred = predictions[:, :, 1:]
        else:
            raise ValueError(f"predictions shape错误: {predictions.shape}, 期望2D或3D")

        # 存在性损失 (Focal Loss)
        exist_loss = self._compute_existence_loss(
            one_index, zero_index, exist_pred, targets
        )

        # 类型损失 (Weighted CE + Label Smoothing)
        if type_pred is not None:
            type_loss = self._compute_type_loss(
                one_index, type_pred, targets
            )
            return self.exist_weight * exist_loss + self.type_weight * type_loss
        else:
            # 只有存在性损失
            return exist_loss

    def _compute_multilabel_bce_loss(self, one_index, zero_index, predictions, targets):
        """Plan G-1: Multi-label BCE — mỗi (mi, dis) cell có thể có multiple types active.

        predictions: [mi, dis, K+1] (K=num_types). Channel 0 = existence, 1..K = type logits.
        targets: hỗ trợ 2 formats:
          - 3D [mi, dis, K] — multi-hot tensor đã build từ preprocess (preferred).
          - 2D [mi, dis] — single-label int (backward compat, sẽ convert sang multi-hot trivial).

        Loss = BCE(existence_logit, has_any_type) + BCE(type_logits[1:], type_targets) trên positive.
        Reuse existing existence_loss focal pattern + type BCE.
        """
        num_types = predictions.shape[2] - 1  # exclude existence channel

        # Build multi-hot target tensor
        if len(targets.shape) == 3:
            target_multi = targets  # already [mi, dis, K]
        else:
            # convert from single-label int [mi, dis] → multi-hot [mi, dis, K]
            target_multi = torch.zeros((predictions.shape[0], predictions.shape[1], num_types),
                                       device=predictions.device, dtype=torch.float32)
            for t in range(1, num_types + 1):
                target_multi[:, :, t - 1] = (targets == t).float()

        # Existence loss (focal) on channel 0
        exist_pred = predictions[:, :, 0]
        existence_target = (target_multi.sum(dim=-1) > 0).float()
        exist_loss = self._compute_existence_loss(
            one_index, zero_index, exist_pred, existence_target.long())

        # Type BCE on channels 1..K — chỉ tính trên positive cells
        pos_indices = self._process_indices(one_index, predictions.shape[0], predictions.shape[1])
        if len(pos_indices) == 0:
            return exist_loss

        # Lấy logits + multi-hot targets cho positive cells
        type_logits = predictions[pos_indices[:, 0], pos_indices[:, 1], 1:]  # [N_pos, K]
        type_targets = target_multi[pos_indices[:, 0], pos_indices[:, 1], :]  # [N_pos, K]

        # BCE per channel với class_weights để boost minority
        # F.binary_cross_entropy_with_logits expects raw logits — predictor may have sigmoid/softmax already
        # Để safe, dùng plain BCE assuming logits đã được processed bởi predictor
        pos_weight = self.class_weights  # [K] — boost minority types
        bce_loss = F.binary_cross_entropy(
            type_logits.clamp(1e-7, 1 - 1e-7),
            type_targets,
            weight=pos_weight.unsqueeze(0).expand_as(type_targets),
            reduction='mean'
        )

        return self.exist_weight * exist_loss + self.type_weight * bce_loss

    def _compute_softmax5_loss(self, one_index, zero_index, logits, targets):
        """Plan D Fix A++: single 5-class CrossEntropy.

        logits: [mi_num, dis_num, 5] RAW LOGITS từ predictor (channel 0=no_assoc, 1-4=types)
        targets: [mi_num, dis_num] giá trị ∈ {0=no_assoc, 1..4=types}

        Build target classes 0..4 cho cả positive (one_index) + sampled negative (zero_index).
        F.cross_entropy với class_weights_5 + label_smoothing → đơn loss thay 2-head fight.
        """
        if len(logits.shape) != 3 or logits.shape[2] != 5:
            raise ValueError(f"softmax_5class expects logits shape [mi, dis, 5], got {logits.shape}")

        pos_indices = self._process_indices(one_index, logits.shape[0], logits.shape[1])
        neg_indices = self._process_indices(zero_index, logits.shape[0], logits.shape[1])

        if len(pos_indices) == 0 and len(neg_indices) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Combine positive + negative indices
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            combined_idx = torch.cat([pos_indices, neg_indices], dim=0)
        elif len(pos_indices) > 0:
            combined_idx = pos_indices
        else:
            combined_idx = neg_indices

        # Extract logits for combined indices: [N, 5]
        combined_logits = logits[combined_idx[:, 0], combined_idx[:, 1], :]

        # Build target classes from `targets` matrix (đã có values 0..4)
        # targets[i,j] = 0 cho no_assoc, 1..4 cho 4 types — match class indices của 5-class softmax
        combined_targets = targets[combined_idx[:, 0], combined_idx[:, 1]].long()

        # Clip để đảm bảo target trong [0, 4] (defensive)
        combined_targets = torch.clamp(combined_targets, 0, 4)

        return F.cross_entropy(
            combined_logits, combined_targets,
            weight=self.class_weights_5,
            label_smoothing=self.label_smoothing,
            reduction='mean'
        )

    def _compute_existence_loss(self, one_index, zero_index, exist_pred, targets):
        pos_indices = self._process_indices(one_index, exist_pred.shape[0], exist_pred.shape[1])
        neg_indices = self._process_indices(zero_index, exist_pred.shape[0], exist_pred.shape[1])

        total_loss, count = 0.0, 0

        if len(pos_indices) > 0:
            pos_scores = exist_pred[pos_indices[:, 0], pos_indices[:, 1]]
            pos_loss = -(1 - pos_scores) ** self.focal_gamma * torch.log(pos_scores + 1e-7)
            total_loss += pos_loss.sum()
            count += len(pos_indices)

        if len(neg_indices) > 0:
            neg_scores = exist_pred[neg_indices[:, 0], neg_indices[:, 1]]
            neg_loss = -neg_scores ** self.focal_gamma * torch.log(1 - neg_scores + 1e-7)
            total_loss += neg_loss.sum() * self.alpha
            count += len(neg_indices)

        return total_loss / (count + 1e-7)

    def _compute_type_loss(self, one_index, type_pred, targets):
        pos_indices = self._process_indices(one_index, type_pred.shape[0], type_pred.shape[1])

        if len(pos_indices) == 0:
            return torch.tensor(0.0, device=self.device)

        pos_targets = targets[pos_indices[:, 0], pos_indices[:, 1]]
        type_indices = torch.zeros_like(pos_targets, dtype=torch.long)
        type_indices[pos_targets == 1] = 0
        type_indices[pos_targets == 2] = 1
        type_indices[pos_targets == 3] = 2
        type_indices[pos_targets == 4] = 3

        valid_mask = pos_targets > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        valid_indices = pos_indices[valid_mask]
        valid_type_indices = type_indices[valid_mask]
        type_logits = type_pred[valid_indices[:, 0], valid_indices[:, 1], :]

        return F.cross_entropy(
            type_logits, valid_type_indices,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction='mean'
        )

    def _process_indices(self, indices, max_rows, max_cols):
        if isinstance(indices, torch.Tensor):
            indices = indices.to(self.device)
            if indices.dim() == 2 and indices.size(1) == 2:
                valid_mask = (indices[:, 0] >= 0) & (indices[:, 0] < max_rows) & \
                             (indices[:, 1] >= 0) & (indices[:, 1] < max_cols)
                return indices[valid_mask]
        return torch.empty((0, 2), dtype=torch.long, device=self.device)


class CachedHypergraphBuilder:
    """缓存超图构建结果以避免重复计算 - 设备兼容版本"""

    def __init__(self, cache_size=10):
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = defaultdict(int)
        self.device = device

    def _get_cache_key(self, X, K_neigs, method):
        """生成缓存键"""
        X_hash = hash(X.data.tobytes())
        return (X_hash, tuple(K_neigs) if isinstance(K_neigs, list) else K_neigs, method)

    def construct_knn(self, X, K_neigs, is_probH=False):
        """缓存的KNN超图构建 - 设备兼容版本"""
        cache_key = self._get_cache_key(X, K_neigs, 'knn')

        if cache_key in self.cache:
            self.access_count[cache_key] += 1
            result = self.cache[cache_key]
            # 确保返回的张量在正确设备上
            return result.to(self.device) if isinstance(result, torch.Tensor) else result

        # 计算新的超图
        X_float32 = X.astype(np.float32)
        H = hypergraph_construct_KNN.construct_H_with_KNN(X_float32, K_neigs, is_probH)
        G = hypergraph_construct_KNN._generate_G_from_H(H)

        # 转换为tensor并移动到正确设备
        if isinstance(G, np.ndarray):
            G = torch.from_numpy(G)
        G = G.to(self.device).float()

        # 缓存结果
        if len(self.cache) >= self.cache_size:
            # 移除最少使用的缓存
            least_used = min(self.cache.keys(), key=lambda k: self.access_count[k])
            del self.cache[least_used]
            del self.access_count[least_used]

        self.cache[cache_key] = G.clone()
        self.access_count[cache_key] = 1

        return G

    def construct_kmeans(self, X, clusters):
        """缓存的K-means超图构建 - 设备兼容版本"""
        cache_key = self._get_cache_key(X, clusters, 'kmeans')

        if cache_key in self.cache:
            self.access_count[cache_key] += 1
            result = self.cache[cache_key]
            # 确保返回的张量在正确设备上
            return result.to(self.device) if isinstance(result, torch.Tensor) else result

        # 计算新的超图
        X_float32 = X.astype(np.float32)
        H = hypergraph_construct_kmeans.construct_H_with_Kmeans(X_float32, clusters)
        G = hypergraph_construct_kmeans._generate_G_from_H(H)

        # 转换为tensor并移动到正确设备
        if isinstance(G, np.ndarray):
            G = torch.from_numpy(G)
        G = G.to(self.device).float()

        # 缓存结果
        if len(self.cache) >= self.cache_size:
            least_used = min(self.cache.keys(), key=lambda k: self.access_count[k])
            del self.cache[least_used]
            del self.access_count[least_used]

        self.cache[cache_key] = G.clone()
        self.access_count[cache_key] = 1

        return G


# 全局缓存实例
hypergraph_builder = CachedHypergraphBuilder(cache_size=20)


def constructHW_knn(X, K_neigs, is_probH):
    """优化的KNN超图构建"""
    return hypergraph_builder.construct_knn(X, K_neigs, is_probH)


def constructHW_kmean(X, clusters):
    """优化的K-means超图构建"""
    return hypergraph_builder.construct_kmeans(X, clusters)


# 损失函数
from param import parameter_parser


def get_L2reg(parameters):
    """优化的L2正则化计算"""
    reg = 0
    for param in parameters:
        reg += 0.5 * (param.float() ** 2).sum()
    return reg


class Myloss(nn.Module):
    def __init__(self, args):
        super(Myloss, self).__init__()
        self.alpha = args.alpha

    def forward(self, one_index, zero_index, input, target):
        input = input.float()
        target = target.float()
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1 - self.alpha) * loss_sum[one_index].sum() + self.alpha * loss_sum[zero_index].sum()


# 优化的异构图构建函数
@lru_cache(maxsize=5)
def create_hetero_data_cached(train_data_key, mi_sim_key=None, dis_sim_key=None):
    """缓存版本的异构图创建函数"""
    pass


def create_hetero_data_optimized(train_data, mi_sim_recon=None, dis_sim_recon=None):
    """优化的异构图数据创建 - 设备兼容版本"""
    try:
        # 快速数据提取
        if isinstance(train_data, list) and len(train_data) >= 5:
            association_matrix = train_data[4].to(device).float()
        else:
            association_matrix = torch.zeros((495, 380), device=device).float()

        target_mi_num = association_matrix.shape[0]
        target_dis_num = association_matrix.shape[1]

        # 使用重构的相似性矩阵或原始矩阵
        if mi_sim_recon is not None:
            mi_sim = mi_sim_recon.to(device).float()
        else:
            mi_sim = train_data[1].to(device).float() if len(train_data) >= 2 else torch.eye(target_mi_num,
                                                                                             device=device).float()

        if dis_sim_recon is not None:
            dis_sim = dis_sim_recon.to(device).float()
        else:
            dis_sim = train_data[0].to(device).float() if len(train_data) >= 1 else torch.eye(target_dis_num,
                                                                                              device=device).float()

        # 维度检查和调整
        mi_num = mi_sim.shape[0]
        dis_num = dis_sim.shape[0]

        threshold = 0.5

        if TORCH_GEOMETRIC_AVAILABLE:
            hetero_data = HeteroData()

            # 创建节点特征
            hetero_data['miRNA'].x = torch.eye(mi_num, device=device).float()
            hetero_data['disease'].x = torch.eye(dis_num, device=device).float()

            # 高效的边创建
            # miRNA-disease关联边
            md_indices = torch.nonzero(association_matrix > 0, as_tuple=True)
            if len(md_indices[0]) > 0:
                md_edges = torch.stack(md_indices).t()
                md_edge_attr = association_matrix[md_indices].unsqueeze(1)

                hetero_data['miRNA', 'associates', 'disease'].edge_index = md_edges.t()
                hetero_data['miRNA', 'associates', 'disease'].edge_attr = md_edge_attr

                # 反向边
                dm_edges = torch.stack([md_indices[1], md_indices[0]]).t()
                hetero_data['disease', 'associates', 'miRNA'].edge_index = dm_edges.t()
                hetero_data['disease', 'associates', 'miRNA'].edge_attr = md_edge_attr
            else:
                # 空边
                hetero_data['miRNA', 'associates', 'disease'].edge_index = torch.zeros((2, 0), dtype=torch.long,
                                                                                       device=device)
                hetero_data['miRNA', 'associates', 'disease'].edge_attr = torch.zeros((0, 1), device=device)
                hetero_data['disease', 'associates', 'miRNA'].edge_index = torch.zeros((2, 0), dtype=torch.long,
                                                                                       device=device)
                hetero_data['disease', 'associates', 'miRNA'].edge_attr = torch.zeros((0, 1), device=device)

            # miRNA相似性边(矢量化)
            mm_mask = (mi_sim > threshold) & (torch.eye(mi_num, device=device) == 0)
            mm_indices = torch.nonzero(mm_mask, as_tuple=True)
            if len(mm_indices[0]) > 0:
                mm_edges = torch.stack(mm_indices).t()
                mm_edge_attr = mi_sim[mm_indices].unsqueeze(1)
                hetero_data['miRNA', 'similar', 'miRNA'].edge_index = mm_edges.t()
                hetero_data['miRNA', 'similar', 'miRNA'].edge_attr = mm_edge_attr
            else:
                hetero_data['miRNA', 'similar', 'miRNA'].edge_index = torch.zeros((2, 0), dtype=torch.long,
                                                                                  device=device)
                hetero_data['miRNA', 'similar', 'miRNA'].edge_attr = torch.zeros((0, 1), device=device)

            # 疾病相似性边(矢量化)
            dd_mask = (dis_sim > threshold) & (torch.eye(dis_num, device=device) == 0)
            dd_indices = torch.nonzero(dd_mask, as_tuple=True)
            if len(dd_indices[0]) > 0:
                dd_edges = torch.stack(dd_indices).t()
                dd_edge_attr = dis_sim[dd_indices].unsqueeze(1)
                hetero_data['disease', 'similar', 'disease'].edge_index = dd_edges.t()
                hetero_data['disease', 'similar', 'disease'].edge_attr = dd_edge_attr
            else:
                hetero_data['disease', 'similar', 'disease'].edge_index = torch.zeros((2, 0), dtype=torch.long,
                                                                                      device=device)
                hetero_data['disease', 'similar', 'disease'].edge_attr = torch.zeros((0, 1), device=device)

        else:
            # 简化版异构图数据结构
            hetero_data = type('HeteroDataSimple', (), {})()
            hetero_data.edge_index_dict = {}
            hetero_data.x_dict = {
                'miRNA': torch.eye(mi_num, device=device).float(),
                'disease': torch.eye(dis_num, device=device).float()
            }

            # 使用矢量化操作创建边索引
            md_indices = torch.nonzero(association_matrix > 0, as_tuple=True)
            if len(md_indices[0]) > 0:
                hetero_data.edge_index_dict[('miRNA', 'associates', 'disease')] = torch.stack(md_indices)
                hetero_data.edge_index_dict[('disease', 'associates', 'miRNA')] = torch.stack(
                    [md_indices[1], md_indices[0]])
            else:
                hetero_data.edge_index_dict[('miRNA', 'associates', 'disease')] = torch.zeros((2, 0), dtype=torch.long,
                                                                                              device=device)
                hetero_data.edge_index_dict[('disease', 'associates', 'miRNA')] = torch.zeros((2, 0), dtype=torch.long,
                                                                                              device=device)

            # miRNA和疾病相似性边
            mm_mask = (mi_sim > threshold) & (torch.eye(mi_num, device=device) == 0)
            mm_indices = torch.nonzero(mm_mask, as_tuple=True)
            if len(mm_indices[0]) > 0:
                hetero_data.edge_index_dict[('miRNA', 'similar', 'miRNA')] = torch.stack(mm_indices)
            else:
                hetero_data.edge_index_dict[('miRNA', 'similar', 'miRNA')] = torch.zeros((2, 0), dtype=torch.long,
                                                                                         device=device)

            dd_mask = (dis_sim > threshold) & (torch.eye(dis_num, device=device) == 0)
            dd_indices = torch.nonzero(dd_mask, as_tuple=True)
            if len(dd_indices[0]) > 0:
                hetero_data.edge_index_dict[('disease', 'similar', 'disease')] = torch.stack(dd_indices)
            else:
                hetero_data.edge_index_dict[('disease', 'similar', 'disease')] = torch.zeros((2, 0), dtype=torch.long,
                                                                                             device=device)

        return hetero_data

    except Exception as e:
        print(f"Error creating heterogeneous graph: {e}")
        # 返回最小化的备份结构
        if TORCH_GEOMETRIC_AVAILABLE:
            hetero_data = HeteroData()
            hetero_data['miRNA'].x = torch.eye(100, device=device).float()
            hetero_data['disease'].x = torch.eye(100, device=device).float()
            # 空边
            for edge_type in [('miRNA', 'associates', 'disease'), ('disease', 'associates', 'miRNA'),
                              ('miRNA', 'similar', 'miRNA'), ('disease', 'similar', 'disease')]:
                hetero_data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                hetero_data[edge_type].edge_attr = torch.zeros((0, 1), device=device)
        else:
            hetero_data = type('HeteroDataSimple', (), {})()
            hetero_data.edge_index_dict = {edge_type: torch.zeros((2, 0), dtype=torch.long, device=device)
                                           for edge_type in
                                           [('miRNA', 'associates', 'disease'), ('disease', 'associates', 'miRNA'),
                                            ('miRNA', 'similar', 'miRNA'), ('disease', 'similar', 'disease')]}
            hetero_data.x_dict = {
                'miRNA': torch.eye(100, device=device).float(),
                'disease': torch.eye(100, device=device).float()
            }
        return hetero_data


# 📍 添加详细调试函数
def analyze_predictions_detailed(score, association_matrix, epoch):
    """详细分析预测分布"""
    print(f"\n{'=' * 90}")
    print(f"📊 Epoch {epoch} - 详细预测分析")
    print(f"{'=' * 90}")

    # 检查输出维度
    print(f"\n🔍 输出形状检查:")
    print(f"  score.shape: {score.shape}")

    if len(score.shape) != 3:
        print(f"  ❌ 错误: 输出应为3维 [N, M, channels]")
        return

    if score.shape[2] < 5:
        print(f"  ❌ 错误: 通道数为 {score.shape[2]}, 应为5 (1存在性 + 4类型)")
        return

    print(f"  ✅ 输出维度正确: [miRNA={score.shape[0]}, Disease={score.shape[1]}, Channels={score.shape[2]}]")

    # 分离存在性和类型分数
    existence_scores = score[:, :, 0]
    type_scores = score[:, :, 1:5]  # 4种类型

    print(f"\n📈 存在性分数统计:")
    print(f"  范围: [{existence_scores.min():.4f}, {existence_scores.max():.4f}]")
    print(f"  均值: {existence_scores.mean():.4f}")
    print(f"  标准差: {existence_scores.std():.4f}")

    print(f"\n🎯 类型分数统计 (原始logits):")
    for i in range(4):
        type_i_scores = type_scores[:, :, i]
        print(f"  类型{i + 1}: 范围=[{type_i_scores.min():.4f}, {type_i_scores.max():.4f}], "
              f"均值={type_i_scores.mean():.4f}, 标准差={type_i_scores.std():.4f}")

    # 找到有关联的样本
    has_association = association_matrix > 0
    num_associations = has_association.sum().item()

    if num_associations == 0:
        print(f"\n⚠️ 警告: 没有找到正样本!")
        return

    print(f"\n🔢 正样本分析 (共 {num_associations} 个):")

    # 对正样本进行类型预测
    pos_type_scores = type_scores[has_association]  # [num_pos, 4]
    pos_type_preds = torch.argmax(pos_type_scores, dim=1)

    # 统计预测分布
    type_names = ["循环(1)", "表观遗传(2)", "靶标(3)", "遗传学(4)"]
    pred_counts = [(pos_type_preds == i).sum().item() for i in range(4)]

    print(f"\n  预测分布:")
    for i, (name, count) in enumerate(zip(type_names, pred_counts)):
        pct = count / num_associations * 100
        bar = '█' * int(pct / 2)
        print(f"    {name:15} {count:4d}/{num_associations} ({pct:5.1f}%) {bar}")

    # 真实类型分布
    true_types = association_matrix[has_association]
    true_type_counts = defaultdict(int)
    for t in true_types:
        t_val = int(t.item())
        if t_val > 0:
            true_type_counts[t_val] += 1

    print(f"\n  真实分布:")
    for type_id in [1, 2, 3, 4]:
        count = true_type_counts[type_id]
        pct = count / num_associations * 100 if num_associations > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"    {type_names[type_id - 1]:15} {count:4d}/{num_associations} ({pct:5.1f}%) {bar}")

    # 分布差异
    print(f"\n  ⚖️ 分布对比:")
    print(f"  {'类型':<15} {'真实比例':>10} {'预测比例':>10} {'差异':>10} {'状态':<10}")
    print(f"  {'-' * 60}")

    total_diff = 0
    for i, name in enumerate(type_names):
        true_count = true_type_counts[i + 1]
        pred_count = pred_counts[i]

        true_pct = true_count / num_associations * 100 if num_associations > 0 else 0
        pred_pct = pred_count / num_associations * 100 if num_associations > 0 else 0
        diff = abs(true_pct - pred_pct)
        total_diff += diff

        if diff < 5:
            status = "✅ 接近"
        elif diff < 15:
            status = "⚠️ 偏差"
        else:
            status = "❌ 较大偏差"

        print(f"  {name:<15} {true_pct:>9.1f}% {pred_pct:>9.1f}% {diff:>9.1f}% {status:<10}")

    avg_diff = total_diff / 4
    print(f"\n  平均分布偏差: {avg_diff:.1f}%")

    if avg_diff > 25:
        print(f"  ❌ 严重: 模型预测分布与真实分布严重不匹配!")
    elif avg_diff > 15:
        print(f"  ⚠️ 中等: 模型预测分布有较大偏差")
    else:
        print(f"  ✅ 良好: 模型预测分布基本匹配真实分布")

    # Softmax后的概率分布
    print(f"\n🎲 Softmax概率分析 (正样本):")
    type_probs = pos_type_scores  # ✅ 直接使用

    for i, name in enumerate(type_names):
        probs_i = type_probs[:, i]
        print(f"  {name:15} 均值={probs_i.mean():.4f}, "
              f"标准差={probs_i.std():.4f}, "
              f"最大={probs_i.max():.4f}")

    # 检查是否所有概率都集中在一个类型
    max_probs = type_probs.max(dim=1)[0]
    avg_max_prob = max_probs.mean().item()
    print(f"\n  平均最大概率: {avg_max_prob:.4f}")

    if avg_max_prob > 0.95:
        print(f"  ❌ 严重: 模型过于自信,几乎总是100%预测某一类")
    elif avg_max_prob > 0.85:
        print(f"  ⚠️ 注意: 模型比较自信,可能缺乏多样性")
    else:
        print(f"  ✅ 正常: 模型预测有适当的不确定性")

    # 检查是否有类型从未被预测
    never_predicted = [i for i, count in enumerate(pred_counts) if count == 0]
    if never_predicted:
        print(f"\n  ⚠️ 警告: 以下类型从未被预测: {[type_names[i] for i in never_predicted]}")

    print(f"{'=' * 90}\n")


def check_gradients(model, epoch):
    """检查梯度流动情况"""
    print(f"\n🔧 Epoch {epoch} 梯度检查:")
    print(f"{'=' * 80}")

    # 检查SimplifiedTypePredictor的梯度
    if hasattr(model, 'association_predictor'):
        predictor = model.association_predictor

        grad_info = []

        # 检查SimplifiedTypePredictor的核心参数
        if hasattr(predictor, 'type_relations') and predictor.type_relations.grad is not None:
            grad_norm = predictor.type_relations.grad.norm().item()
            grad_mean = predictor.type_relations.grad.abs().mean().item()
            grad_info.append(('type_relations', grad_norm, grad_mean))

            if grad_norm < 1e-6:
                print(f"  ⚠️ 警告: type_relations梯度几乎为0 (norm={grad_norm:.8f})")
                print(f"     → 类型分类器可能没有学习!")

        if hasattr(predictor, 'exist_relation') and predictor.exist_relation.grad is not None:
            grad_norm = predictor.exist_relation.grad.norm().item()
            grad_mean = predictor.exist_relation.grad.abs().mean().item()
            grad_info.append(('exist_relation', grad_norm, grad_mean))

        # 检查投影层 (mi_projector和dis_projector)
        if hasattr(predictor, 'mi_projector'):
            for i, layer in enumerate(predictor.mi_projector):
                if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                    grad_norm = layer.weight.grad.norm().item()
                    grad_mean = layer.weight.grad.abs().mean().item()
                    grad_info.append((f'mi_proj_layer{i}', grad_norm, grad_mean))

        if hasattr(predictor, 'dis_projector'):
            for i, layer in enumerate(predictor.dis_projector):
                if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                    grad_norm = layer.weight.grad.norm().item()
                    grad_mean = layer.weight.grad.abs().mean().item()
                    grad_info.append((f'dis_proj_layer{i}', grad_norm, grad_mean))

        if grad_info:
            print(f"\n  {'参数名称':<25} {'梯度范数':>12} {'梯度均值':>12} {'状态':<15}")
            print(f"  {'-' * 70}")
            for name, norm, mean in grad_info:
                if norm < 1e-7:
                    status = "❌ 过小"
                elif norm > 100:
                    status = "⚠️ 过大"
                else:
                    status = "✅ 正常"
                print(f"  {name:<25} {norm:>12.6f} {mean:>12.6f} {status:<15}")
        else:
            print("  ℹ️ SimplifiedTypePredictor: 所有参数梯度正常流动")

    print(f"{'=' * 80}\n")


# 训练和测试函数
def train_epoch_optimized(model, train_data, optim, args):
    """优化的训练函数 - 真正的双视图实现 + 详细调试

    [VN] Training loop cho 1 fold của CV:
      1. Lấy 12-tuple từ train_data (xem docs/NOTES_DATAFLOW.md §3)
      2. Negative sampling: chọn 10x số positive làm negative
      3. Build 4 hypergraph Laplacian G từ 4 view khác nhau (KNN K=13)
      4. Build heterogeneous graph (4 edge types)
      5. Loop args.epoch lần:
         - forward model → score [495,383,5] + losses
         - compute total loss = recover + CL + 0.15·recon + 1e-4·reg
         - backward + clip grad + step
         - mỗi 50 epoch: check gradient + print progress
         - mỗi 5 epoch (graph_update_frequency): rebuild edges nếu sim thay đổi >0.01
      6. Sau training loop: chạy test_optimized để eval
    """
    model.train()
    regression_crit = SimplifiedMultiTypeAssociationLoss(args, model)

    # 预处理索引并确保在正确设备上
    one_index_tensor = train_data[2][0].to(device)
    zero_index_tensor = train_data[2][1].to(device)

    print(f"Original positive index shape: {one_index_tensor.shape}")
    print(f"Original negative index shape: {zero_index_tensor.shape}")

    # 负采样优化
    neg_sample_ratio = 10
    if zero_index_tensor.shape[0] > one_index_tensor.shape[0] * neg_sample_ratio:
        perm = torch.randperm(zero_index_tensor.shape[0], device=device)
        sampled_indices = perm[:one_index_tensor.shape[0] * neg_sample_ratio]
        zero_index_tensor = zero_index_tensor[sampled_indices]
        print(f"采样后负样本数量: {zero_index_tensor.shape[0]}")

    # 获取关联矩阵
    association_matrix = train_data[4].to(device).float()
    mi_num = association_matrix.shape[0]
    dis_num = association_matrix.shape[1]

    args.mi_num = mi_num
    args.dis_num = dis_num

    # Plan H-3: load multi-label target tensor [m, d, K] nếu flag set.
    # Preserve 23.3% multi-label signal mất khi collapse single-label (M1 fix).
    multilabel_target = None
    ml_path = getattr(args, 'multilabel_target_path', '')
    if ml_path and os.path.exists(ml_path):
        ml_np = np.load(ml_path)
        multilabel_target = torch.from_numpy(ml_np).float().to(device)
        print(f"[H-3 multilabel] Loaded target tensor {multilabel_target.shape} từ {ml_path}")

    # 🎯 真正的双视图构建 - 四个不同的特征源
    print("🎯 Building TRUE DUAL VIEWS from four different sources...")

    try:
        # 获取四个不同的特征/相似性源
        dis_sem_data = train_data[0].to(device).float()  # 疾病语义相似性 (Disease View 2)
        mi_fun_data = train_data[1].to(device).float()  # miRNA功能相似性 (miRNA View 2)
        d_gs_data = train_data[8].to(device).float()  # 疾病-基因特征 (Disease View 1)
        m_ss_data = train_data[9].to(device).float()  # miRNA-序列特征 (miRNA View 1)

        print(f"📊 Data sources loaded:")
        print(f"  Disease semantic: {dis_sem_data.shape}")
        print(f"  miRNA functional: {mi_fun_data.shape}")
        print(f"  Disease-gene: {d_gs_data.shape}")
        print(f"  miRNA-sequence: {m_ss_data.shape}")

        # 🧬 构建miRNA的真正双视图
        # miRNA View 1: 关联矩阵 + 序列特征 (来自m_ss.xlsx)
        concat_miRNA_view1 = torch.cat([association_matrix, m_ss_data], dim=1)
        concat_mi_tensor_view1 = concat_miRNA_view1.to(device).float()

        # miRNA View 2: 关联矩阵 + 功能相似性 (来自mi_fun_sim_2.0.csv)
        concat_miRNA_view2 = torch.cat([association_matrix, mi_fun_data], dim=1)
        concat_mi_tensor_view2 = concat_miRNA_view2.to(device).float()

        print("🧬 Building miRNA dual-view hypergraphs...")
        print(f"  📊 miRNA View 1 (Sequence): {concat_mi_tensor_view1.shape}")
        print(f"  📊 miRNA View 2 (Function): {concat_mi_tensor_view2.shape}")

        G_mi_view1 = constructHW_knn(concat_mi_tensor_view1.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)
        G_mi_view2 = constructHW_knn(concat_mi_tensor_view2.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)

        # 🦠 构建Disease的真正双视图
        # Disease View 1: 关联矩阵转置 + 基因特征 (来自d_gs.xlsx)
        concat_dis_view1 = torch.cat([association_matrix.t(), d_gs_data], dim=1)
        concat_dis_tensor_view1 = concat_dis_view1.to(device).float()

        # Disease View 2: 关联矩阵转置 + 语义相似性 (来自dis_sem_sim_2.0.csv)
        concat_dis_view2 = torch.cat([association_matrix.t(), dis_sem_data], dim=1)
        concat_dis_tensor_view2 = concat_dis_view2.to(device).float()

        print("🦠 Building disease dual-view hypergraphs...")
        print(f"  📊 Disease View 1 (Gene): {concat_dis_tensor_view1.shape}")
        print(f"  📊 Disease View 2 (Semantic): {concat_dis_tensor_view2.shape}")

        G_dis_view1 = constructHW_knn(concat_dis_tensor_view1.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)
        G_dis_view2 = constructHW_knn(concat_dis_tensor_view2.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)

        print("✅ Successfully built TRUE DUAL VIEWS:")
        print("   🧬 miRNA: Sequence Features (m_ss.xlsx) vs Functional Similarity (mi_fun_sim_2.0.csv)")
        print("   🦠 Disease: Gene Features (d_gs.xlsx) vs Semantic Similarity (dis_sem_sim_2.0.csv)")

    except Exception as e:
        print(f"❌ Error in dual view construction: {e}")
        print("🔄 Falling back to single source approach...")

        # 备份方案:使用单一源构建视图
        # 获取整合相似性数据
        if len(train_data) > 10:
            dis_sim_integrate_tensor = train_data[10].to(device).float()  # 整合疾病相似性
            mi_sim_integrate_tensor = train_data[11].to(device).float()  # 整合miRNA相似性
        else:
            dis_sim_integrate_tensor = train_data[0].to(device).float()
            mi_sim_integrate_tensor = train_data[1].to(device).float()

        concat_miRNA = torch.cat([association_matrix, mi_sim_integrate_tensor], dim=1)
        concat_mi_tensor_view1 = concat_miRNA.to(device).float()
        concat_mi_tensor_view2 = concat_mi_tensor_view1

        G_mi_view1 = constructHW_knn(concat_mi_tensor_view1.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)
        G_mi_view2 = constructHW_kmean(concat_mi_tensor_view1.detach().cpu().numpy(), clusters=[9])

        concat_dis = torch.cat([association_matrix.t(), dis_sim_integrate_tensor], dim=1)
        concat_dis_tensor_view1 = concat_dis.to(device).float()
        concat_dis_tensor_view2 = concat_dis_tensor_view1

        G_dis_view1 = constructHW_knn(concat_dis_tensor_view1.detach().cpu().numpy(), K_neigs=args.K_neigs, is_probH=False)
        G_dis_view2 = constructHW_kmean(concat_dis_tensor_view1.detach().cpu().numpy(), clusters=[9])

    # 确保所有超图在正确设备上
    G_mi_view1 = G_mi_view1.to(device).float()
    G_mi_view2 = G_mi_view2.to(device).float()
    G_dis_view1 = G_dis_view1.to(device).float()
    G_dis_view2 = G_dis_view2.to(device).float()

    # 创建初始异构图
    train_data_list = [dis_sem_data, mi_fun_data, None, None, association_matrix]
    hetero_data = create_hetero_data_optimized(train_data_list)

    # 训练循环
    start_time = time.time()
    for epoch in range(1, args.epoch + 1):
        # 前向传播 - 使用真正的双视图
        score, mi_cl_loss, dis_cl_loss, mi_sim_recon, dis_sim_recon = model(
            concat_mi_tensor_view1, concat_dis_tensor_view1,
            G_mi_view1, G_mi_view2, G_dis_view1, G_dis_view2,
            hetero_data
        )

        # 确保重构结果在正确设备上
        mi_sim_recon = mi_sim_recon.to(device)
        dis_sim_recon = dis_sim_recon.to(device)

        # Dynamic hypergraph update — every args.update_graph_frequency epochs (paper: 5)
        update_graph = (epoch > 0 and epoch % args.update_graph_frequency == 0)
        if update_graph:
            hetero_data = create_hetero_data_optimized(train_data_list, mi_sim_recon, dis_sim_recon)
            if epoch <= 50 or epoch % 50 == 0:
                print(f"[INFO] Hypergraph updated at epoch {epoch}")

        # 损失计算
        # 使用原始相似性数据作为重构目标
        mi_recon_loss = F.mse_loss(mi_sim_recon, mi_fun_data)  # 重构miRNA功能相似性
        dis_recon_loss = F.mse_loss(dis_sim_recon, dis_sem_data)  # 重构疾病语义相似性

        # 处理多维分数张量
        if len(score.shape) == 3:
            existence_score = score[:, :, 0]
        else:
            existence_score = score

        # 计算关联预测损失 (传入完整的score以支持类型预测)
        binary_target = (association_matrix != 0).float()
        # 🔥 修复: 传入完整的score (3D) 而不是existence_score (2D)
        # 这样SimplifiedMultiTypeAssociationLoss才能同时计算存在性和类型损失
        # Plan H-3: dùng multi-label target [m,d,K] nếu có (multilabel_bce mode), else single-label matrix.
        loss_target = multilabel_target if multilabel_target is not None else association_matrix
        recover_loss = regression_crit(one_index_tensor, zero_index_tensor, score, loss_target)

        # 正则化损失
        reg_loss = get_L2reg(model.parameters())

        # 🆕 提取inter_view_loss用于输出
        # mi_cl_loss和dis_cl_loss已经包含了inter_view_loss
        # 我们需要单独计算inter_view_loss以便输出
        inter_view_loss = torch.tensor(0.0, device=device)
        if model.enable_inter_view_cl and model.training:
            # 从concat_mi_tensor_view1提取关联矩阵
            association_matrix_extracted = concat_mi_tensor_view1[:, :model.dis_num].clone()
            association_matrix_binary = (association_matrix_extracted > 0).float()

            if association_matrix_binary.sum() > 0:
                # 获取融合特征(需要重新计算或从模型获取)
                # 这里我们只是为了输出,实际已在forward中计算
                pass

        # 总损失
        # 🔥 SimplifiedTypePredictor不需要额外的类别分离损失
        # 类型关系向量已经通过正交初始化确保了分离性

        tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 1.0 * (
                mi_recon_loss + dis_recon_loss) + 0.0001 * reg_loss

        # 🔧 检查是否有nan(在反向传播之前)
        if torch.isnan(tol_loss) or torch.isnan(mi_cl_loss) or torch.isnan(dis_cl_loss):
            print("⚠️ NaN detected before backward! Loss breakdown:")
            print(f"  Total Loss: {tol_loss.item() if not torch.isnan(tol_loss) else 'NaN'}")
            print(f"  Recover Loss: {recover_loss.item()}")
            print(f"  miRNA CL: {mi_cl_loss.item() if not torch.isnan(mi_cl_loss) else 'NaN'}")
            print(f"  Disease CL: {dis_cl_loss.item() if not torch.isnan(dis_cl_loss) else 'NaN'}")
            print(f"  Recon Loss: {(mi_recon_loss + dis_recon_loss).item()}")
            print(f"  Reg Loss: {reg_loss.item()}")
            print("Stopping training due to NaN...")
            break

        # 反向传播和优化
        optim.zero_grad()
        tol_loss.backward()

        # 🔍 每50轮检查梯度
        if epoch % 50 == 0 or epoch == 1:
            check_gradients(model, epoch)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        # 周期性输出
        if epoch % 50 == 0 or epoch == 1:
            elapsed_time = time.time() - start_time

            # 计算inter_view的实际贡献(从总CL loss中分离)
            # mi_cl_loss = mi_intra_loss + inter_view_weight * inter_view_loss
            # 假设inter_view_weight = 0.3
            estimated_inter_view = (mi_cl_loss.item() + dis_cl_loss.item()) * 0.3 / (1 + 0.3)

            print(f"Epoch {epoch}, Total Loss: {tol_loss.item():.4f}, "
                  f"Recover Loss: {recover_loss.item():.4f}, "
                  f"miRNA CL: {mi_cl_loss.item():.4f}, "
                  f"Disease CL: {dis_cl_loss.item():.4f}, "
                  f"Inter-view (est): {estimated_inter_view:.4f}, "
                  f"Recon Loss: {(mi_recon_loss + dis_recon_loss).item():.4f}, "
                  f"Time: {elapsed_time:.2f}s")

            # 检查是否有nan
            if torch.isnan(tol_loss):
                print("⚠️ NaN detected! Loss breakdown:")
                print(f"  Recover Loss: {recover_loss.item()}")
                print(f"  miRNA CL: {mi_cl_loss.item()}")
                print(f"  Disease CL: {dis_cl_loss.item()}")
                print(f"  Recon: {(mi_recon_loss + dis_recon_loss).item()}")
                break

        # 学习率调度器更新 (如果使用)
        if hasattr(optim, 'scheduler') and optim.scheduler is not None:
            optim.scheduler.step()

    # ✅ 修复: 在训练循环完成后进行测试
    # 测试应该在所有epoch完成后进行,而不是在循环内部
    print("\n" + "=" * 80)
    print("Training completed. Running final test...")
    print("=" * 80)

    model.eval()
    true_value_one, true_value_zero, pre_value_one, pre_value_zero = test_optimized(
        model, train_data, concat_mi_tensor_view1, concat_dis_tensor_view1,
        G_mi_view1, G_mi_view2, G_dis_view1, G_dis_view2, hetero_data
    )

    return true_value_one, true_value_zero, pre_value_one, pre_value_zero


def test_optimized(model, data, concat_mi_tensor, concat_dis_tensor,
                   G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km, hetero_data=None):
    """优化的测试函数 - 设备兼容版本"""
    model.eval()
    with torch.no_grad():
        # 确保所有输入在正确设备上
        concat_mi_tensor = concat_mi_tensor.to(device).float()
        concat_dis_tensor = concat_dis_tensor.to(device).float()
        G_mi_Kn = G_mi_Kn.to(device).float()
        G_mi_Km = G_mi_Km.to(device).float()
        G_dis_Kn = G_dis_Kn.to(device).float()
        G_dis_Km = G_dis_Km.to(device).float()

        score, _, _, _, _ = model(
            concat_mi_tensor, concat_dis_tensor,
            G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km,
            hetero_data
        )

    # Plan D: nếu loss_mode='softmax_5class', score là raw logits → softmax + transform
    # về 5-channel format [exist_score, type1_prob, type2_prob, type3_prob, type4_prob]
    # để downstream evaluation code không phải đổi
    predictor_loss_mode = getattr(model.association_predictor, 'loss_mode', 'two_head')
    if predictor_loss_mode == 'softmax_5class' and len(score.shape) == 3 and score.shape[2] == 5:
        probs = F.softmax(score, dim=-1)  # [mi, dis, 5] probabilities
        # exist = 1 - P(no_assoc), type_probs = P(class=k|k>0) RE-NORMALIZED qua type-only softmax
        # Để top-1 ranking giữa types không bị nhiễu bởi "no_assoc" mass.
        existence = 1.0 - probs[..., 0:1]  # [mi, dis, 1]
        type_probs_renorm = probs[..., 1:5] / (probs[..., 1:5].sum(dim=-1, keepdim=True) + 1e-12)
        score = torch.cat([existence, type_probs_renorm], dim=-1)  # [mi, dis, 5]

    # 批量处理测试索引
    test_one_index = data[3][0]
    test_zero_index = data[3][1]

    # 快速索引处理
    def process_test_indices(indices):
        if isinstance(indices, torch.Tensor) and indices.dim() == 2:
            return indices.to(device)
        else:
            if hasattr(indices, 't'):
                return indices.t().to(device)
            else:
                return torch.tensor(indices, device=device, dtype=torch.long).t()

    test_one_indices = process_test_indices(test_one_index)
    test_zero_indices = process_test_indices(test_zero_index)

    # 批量提取真实值和预测值
    true_one = torch.zeros(len(test_one_indices), device=device).float()
    true_zero = torch.zeros(len(test_zero_indices), device=device).float()

    # 批量处理真实值
    if len(test_one_indices) > 0:
        valid_one_mask = (test_one_indices[:, 0] < data[5].shape[0]) & (test_one_indices[:, 1] < data[5].shape[1])
        valid_one_indices = test_one_indices[valid_one_mask]
        if len(valid_one_indices) > 0:
            true_one[valid_one_mask] = data[5][valid_one_indices[:, 0], valid_one_indices[:, 1]].to(device).float()

    if len(test_zero_indices) > 0:
        valid_zero_mask = (test_zero_indices[:, 0] < data[5].shape[0]) & (test_zero_indices[:, 1] < data[5].shape[1])
        valid_zero_indices = test_zero_indices[valid_zero_mask]
        if len(valid_zero_indices) > 0:
            true_zero[valid_zero_mask] = data[5][valid_zero_indices[:, 0], valid_zero_indices[:, 1]].to(device).float()

    # 批量提取预测分数
    is_multiclass = len(score.shape) == 3 and score.shape[2] > 1

    if is_multiclass:
        pre_one = torch.zeros((len(test_one_indices), score.shape[2]), device=device).float()
        pre_zero = torch.zeros((len(test_zero_indices), score.shape[2]), device=device).float()
    else:
        pre_one = torch.zeros(len(test_one_indices), device=device).float()
        pre_zero = torch.zeros(len(test_zero_indices), device=device).float()

    # 批量处理预测值
    if len(test_one_indices) > 0:
        valid_one_mask = (test_one_indices[:, 0] < score.shape[0]) & (test_one_indices[:, 1] < score.shape[1])
        valid_one_indices = test_one_indices[valid_one_mask]
        if len(valid_one_indices) > 0:
            if is_multiclass:
                pre_one[valid_one_mask] = score[valid_one_indices[:, 0], valid_one_indices[:, 1]]
            else:
                pre_one[valid_one_mask] = score[valid_one_indices[:, 0], valid_one_indices[:, 1]].float()

    if len(test_zero_indices) > 0:
        valid_zero_mask = (test_zero_indices[:, 0] < score.shape[0]) & (test_zero_indices[:, 1] < score.shape[1])
        valid_zero_indices = test_zero_indices[valid_zero_mask]
        if len(valid_zero_indices) > 0:
            if is_multiclass:
                pre_zero[valid_zero_mask] = score[valid_zero_indices[:, 0], valid_zero_indices[:, 1]]
            else:
                pre_zero[valid_zero_mask] = score[valid_zero_indices[:, 0], valid_zero_indices[:, 1]].float()

    return true_one, true_zero, pre_one, pre_zero


# 评估函数
from Calculate_Metrics import Metric_fun


def evaluate_optimized(true_one, true_zero, pre_one, pre_zero):
    """优化的评估函数 - 设备兼容版本"""
    from Calculate_Metrics import Metric_fun
    import numpy as np
    import torch

    Metric = Metric_fun()
    binary_metrics_tensor = np.zeros((1, 7))

    # 多分类指标
    multiclass_metrics = {
        'accuracy': 0.0,
        'precision_macro': 0.0,
        'recall_macro': 0.0,
        'f1_macro': 0.0,
        'precision_weighted': 0.0,
        'recall_weighted': 0.0,
        'f1_weighted': 0.0
    }
    has_multiclass = False

    valid_seeds = 0

    # 减少种子数量以加速评估
    num_seeds = 5  # 从10减少到5

    for seed in range(num_seeds):
        try:
            test_po_num = true_one.shape[0]

            # 数据预处理 - 确保在CPU上进行numpy操作
            if isinstance(true_zero, torch.Tensor):
                true_zero_cpu = true_zero.cpu()
            else:
                true_zero_cpu = true_zero

            if isinstance(true_one, torch.Tensor):
                true_one_cpu = true_one.cpu()
            else:
                true_one_cpu = true_one

            # 寻找真负样本
            test_index = np.array(np.where(true_zero_cpu.numpy() == 0))

            if test_index.size == 0:
                test_index = np.array([range(len(true_zero_cpu))])

            # 快速重采样
            np.random.seed(seed)
            np.random.shuffle(test_index.T)

            sample_size = min(test_po_num, test_index.shape[1])
            if sample_size == 0:
                continue

            test_ne_index = tuple(test_index[:, :sample_size])

            # 提取评估数据
            eval_true_zero = true_zero[test_ne_index]
            eval_pre_zero = pre_zero[test_ne_index]

            if len(eval_true_zero) == 0 or len(eval_pre_zero) == 0:
                continue

            # 合并数据
            eval_true_data = torch.cat([true_one, eval_true_zero])

            # 处理多分类预测
            if len(pre_one.shape) > 1 and pre_one.shape[-1] > 1:
                has_multiclass = True
                binary_pre_one = pre_one[:, 0]
            else:
                binary_pre_one = pre_one

            if len(eval_pre_zero.shape) > 1 and eval_pre_zero.shape[-1] > 1:
                binary_pre_zero = eval_pre_zero[:, 0]
            else:
                binary_pre_zero = eval_pre_zero

            eval_pre_data_binary = torch.cat([binary_pre_one, binary_pre_zero])

            # 检查数据有效性
            if torch.isnan(eval_true_data).any() or torch.isnan(eval_pre_data_binary).any():
                continue

            # 计算二分类指标
            binary_metrics = Metric.get_metrics(
                (eval_true_data != 0).float(),
                eval_pre_data_binary
            )
            binary_metrics_tensor += binary_metrics
            valid_seeds += 1

            # 多分类指标计算
            if has_multiclass and len(np.unique(eval_true_data.cpu().numpy())) > 1:
                non_zero_mask = eval_true_data != 0
                if non_zero_mask.sum() > 0:
                    true_multiclass = eval_true_data[non_zero_mask]

                    if len(pre_one.shape) > 1 and pre_one.shape[-1] > 1:
                        all_preds = torch.cat([pre_one, eval_pre_zero], dim=0)
                        pred_multiclass = all_preds[non_zero_mask]

                        mc_metrics = Metric.get_multiclass_metrics(true_multiclass, pred_multiclass)

                        for key in multiclass_metrics:
                            if key in mc_metrics:
                                multiclass_metrics[key] += mc_metrics[key]

        except Exception as e:
            print(f"Error in evaluation seed {seed}: {e}")
            continue

    # 计算平均值
    if valid_seeds == 0:
        valid_seeds = 1

    binary_metrics_avg = binary_metrics_tensor / valid_seeds

    if has_multiclass and valid_seeds > 0:
        for key in multiclass_metrics:
            multiclass_metrics[key] /= valid_seeds
        return binary_metrics_avg, multiclass_metrics
    else:
        return binary_metrics_avg, None


def evaluate_optimized_with_comprehensive_metrics(true_one, true_zero, pre_one, pre_zero):
    """
    包含Top-1指标的综合评估函数
    Returns: (二分类指标, CV_type风格指标, Top-1指标, 多分类指标)
    """
    from Calculate_Metrics import Metric_fun
    import numpy as np
    import torch

    Metric = Metric_fun()

    # 原有的二分类指标累积
    binary_metrics_tensor = np.zeros((1, 7))

    # CV_type风格指标累积
    cv_type_metrics_sum = np.zeros(
        9)  # [avg_precision, avg_recall, AUC, AUPR, F1, Accuracy, Recall, Specificity, Precision]

    # Top-1指标累积
    top1_metrics_sum = {'top1_precision': 0.0, 'top1_recall': 0.0, 'top1_f1': 0.0}

    # 多分类指标
    multiclass_metrics = {
        'accuracy': 0.0,
        'precision_macro': 0.0,
        'recall_macro': 0.0,
        'f1_macro': 0.0,
        'precision_weighted': 0.0,
        'recall_weighted': 0.0,
        'f1_weighted': 0.0
    }

    # 检查是否有多分类输出
    has_multiclass = isinstance(pre_one, torch.Tensor) and len(pre_one.shape) > 1 and pre_one.shape[-1] > 1

    valid_seeds = 0

    # 减少种子数量以加速评估
    num_seeds = 5

    for seed in range(num_seeds):
        try:
            test_po_num = true_one.shape[0]

            # 数据预处理 - 确保在CPU上进行numpy操作
            if isinstance(true_zero, torch.Tensor):
                true_zero_cpu = true_zero.cpu()
            else:
                true_zero_cpu = true_zero

            if isinstance(true_one, torch.Tensor):
                true_one_cpu = true_one.cpu()
            else:
                true_one_cpu = true_one

            # 寻找真负样本
            test_index = np.array(np.where(true_zero_cpu.numpy() == 0))

            if test_index.size == 0:
                test_index = np.array([range(len(true_zero_cpu))])

            # 快速重采样
            np.random.seed(seed)
            np.random.shuffle(test_index.T)

            sample_size = min(test_po_num, test_index.shape[1])
            if sample_size == 0:
                continue

            test_ne_index = tuple(test_index[:, :sample_size])

            # 提取评估数据
            eval_true_zero = true_zero[test_ne_index]
            eval_pre_zero = pre_zero[test_ne_index]

            if len(eval_true_zero) == 0 or len(eval_pre_zero) == 0:
                continue

            # 合并数据
            eval_true_data = torch.cat([true_one, eval_true_zero])

            # 处理多分类预测
            if has_multiclass:
                if len(pre_one.shape) > 1 and pre_one.shape[-1] > 1:
                    binary_pre_one = pre_one[:, 0]
                else:
                    binary_pre_one = pre_one

                if len(eval_pre_zero.shape) > 1 and eval_pre_zero.shape[-1] > 1:
                    binary_pre_zero = eval_pre_zero[:, 0]
                else:
                    binary_pre_zero = eval_pre_zero

                eval_pre_data_binary = torch.cat([binary_pre_one, binary_pre_zero])
            else:
                eval_pre_data_binary = torch.cat([pre_one, eval_pre_zero])

            # 检查数据有效性
            if torch.isnan(eval_true_data).any() or torch.isnan(eval_pre_data_binary).any():
                continue

            # 计算二分类指标
            binary_metrics = Metric.get_metrics(
                (eval_true_data != 0).float(),
                eval_pre_data_binary
            )
            binary_metrics_tensor += binary_metrics

            # 🎯 计算CV_type风格指标和Top-1指标
            # 准备CV_type和Top-1风格的数据格式
            real_scores_list = []
            predict_scores_list = []

            # 对于正样本
            for i in range(len(true_one)):
                if has_multiclass and len(pre_one.shape) > 1:
                    real_scores_list.append(true_one[i].cpu().numpy())
                    predict_scores_list.append(pre_one[i].cpu().numpy())
                else:
                    # 对于单值,转换为向量形式
                    real_val = true_one[i].cpu().numpy() if isinstance(true_one[i], torch.Tensor) else true_one[i]
                    pred_val = pre_one[i].cpu().numpy() if isinstance(pre_one[i], torch.Tensor) else pre_one[i]
                    real_scores_list.append([real_val])
                    predict_scores_list.append([pred_val])

            # 对于负样本
            for i in range(len(eval_true_zero)):
                if has_multiclass and len(eval_pre_zero.shape) > 1:
                    real_scores_list.append(eval_true_zero[i].cpu().numpy())
                    predict_scores_list.append(eval_pre_zero[i].cpu().numpy())
                else:
                    real_val = eval_true_zero[i].cpu().numpy() if isinstance(eval_true_zero[i], torch.Tensor) else \
                        eval_true_zero[i]
                    pred_val = eval_pre_zero[i].cpu().numpy() if isinstance(eval_pre_zero[i], torch.Tensor) else \
                        eval_pre_zero[i]
                    real_scores_list.append([real_val])
                    predict_scores_list.append([pred_val])

            # 计算CV_type风格指标
            cv_type_metrics = Metric.compute_cv_type_style_metrics(real_scores_list, predict_scores_list)
            cv_type_metrics_sum += cv_type_metrics

            # 🎯 计算Top-1指标
            top1_metrics = Metric.compute_top1_metrics(real_scores_list, predict_scores_list)
            for key in top1_metrics_sum:
                if key in top1_metrics:
                    top1_metrics_sum[key] += top1_metrics[key]

            valid_seeds += 1

            # 多分类指标计算
            if has_multiclass and len(np.unique(eval_true_data.cpu().numpy())) > 1:
                non_zero_mask = eval_true_data != 0
                if non_zero_mask.sum() > 0:
                    true_multiclass = eval_true_data[non_zero_mask]

                    if len(pre_one.shape) > 1 and pre_one.shape[-1] > 1:
                        all_preds = torch.cat([pre_one, eval_pre_zero], dim=0)
                        pred_multiclass = all_preds[non_zero_mask]

                        mc_metrics = Metric.get_multiclass_metrics(true_multiclass, pred_multiclass)

                        for key in multiclass_metrics:
                            if key in mc_metrics:
                                multiclass_metrics[key] += mc_metrics[key]

        except Exception as e:
            print(f"Error in evaluation seed {seed}: {e}")
            continue

    # 计算平均值
    if valid_seeds == 0:
        valid_seeds = 1

    binary_metrics_avg = binary_metrics_tensor / valid_seeds
    cv_type_metrics_avg = cv_type_metrics_sum / valid_seeds

    # Top-1指标平均值
    top1_metrics_avg = {k: v / valid_seeds for k, v in top1_metrics_sum.items()}

    if has_multiclass and valid_seeds > 0:
        for key in multiclass_metrics:
            multiclass_metrics[key] /= valid_seeds
        return binary_metrics_avg, cv_type_metrics_avg, top1_metrics_avg, multiclass_metrics
    else:
        return binary_metrics_avg, cv_type_metrics_avg, top1_metrics_avg, None


# 主函数
from prepareData import prepare_data
from trainData import Dataset


def main_optimized(args):
    """优化的主函数 - 双视图设备兼容版本"""
    print("🎯 Loading dual-view data...")
    dataset = prepare_data(args)

    if dataset is None:
        print("❌ Failed to load dataset. Please check data files.")
        return None

    train_data = Dataset(args, dataset)

    metrics_cross = np.zeros((1, 7))
    cv_type_metrics_cross = np.zeros(9)  # CV_type风格指标
    top1_metrics_cross = {'top1_precision': 0.0, 'top1_recall': 0.0, 'top1_f1': 0.0}  # Top-1指标

    multiclass_metrics_sum = {
        'accuracy': 0.0,
        'precision_macro': 0.0,
        'recall_macro': 0.0,
        'f1_macro': 0.0,
        'precision_weighted': 0.0,
        'recall_weighted': 0.0,
        'f1_weighted': 0.0
    }
    has_multiclass = False

    # 执行交叉验证
    for i in range(args.validation):
        print(f"=== Cross Validation Fold {i + 1}/{args.validation} ===")
        fold_start_time = time.time()

        # 优化的模型参数
        hidden_list = [256, 256]
        num_proj_hidden = 64

        # 确保隐藏维度与注意力头数兼容
        if args.n_head > 0:
            hidden_list = [dim - (dim % args.n_head) for dim in hidden_list]
            if hidden_list[0] == 0:
                hidden_list = [args.n_head * 5, args.n_head * 5]

        print(f"Using hidden dimensions: {hidden_list} with {args.n_head} attention heads")

        # 初始化模型并确保在正确设备上
        model = HeterogenousGraphCLAMIR(
            args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args
        )
        model.to(device)

        # 确保模型参数是float类型
        for param in model.parameters():
            param.data = param.data.float()

        # 优化器(使用AdamW和学习率调度)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50)

        # 训练模型
        fold_data = train_data[i]
        true_value_one, true_value_zero, pre_value_one, pre_value_zero = train_epoch_optimized(
            model, fold_data, optimizer, args
        )

        # 🎯 使用增强评估函数(包含Top-1指标)
        metrics_result = evaluate_optimized_with_comprehensive_metrics(
            true_value_one, true_value_zero, pre_value_one, pre_value_zero
        )

        # 处理结果
        if isinstance(metrics_result, tuple) and len(metrics_result) >= 3:
            binary_metrics, cv_type_metrics, top1_metrics = metrics_result[:3]
            multiclass_metrics = metrics_result[3] if len(metrics_result) > 3 else None

            has_multiclass = multiclass_metrics is not None

            if has_multiclass:
                for key in multiclass_metrics_sum:
                    if key in multiclass_metrics:
                        multiclass_metrics_sum[key] += multiclass_metrics[key]

            # 累积各种指标
            metrics_cross += binary_metrics
            cv_type_metrics_cross += cv_type_metrics

            # 累积Top-1指标
            for key in top1_metrics_cross:
                if key in top1_metrics:
                    top1_metrics_cross[key] += top1_metrics[key]

            # 去掉详细的fold结果输出,只保留简要信息
            print(f"Fold {i + 1} completed - AUC: {binary_metrics[0][0]:.4f}, "
                  f"Top-1 F1: {top1_metrics['top1_f1']:.4f}")

        else:
            metrics_cross += metrics_result
            print(f"Fold {i + 1} completed")

        fold_time = time.time() - fold_start_time
        print(f"Fold {i + 1} training time: {fold_time:.2f} seconds")

        # 内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算交叉验证的平均性能
    metrics_cross_avg = metrics_cross / args.validation
    cv_type_metrics_cross_avg = cv_type_metrics_cross / args.validation

    # Top-1指标平均值
    top1_metrics_cross_avg = {k: v / args.validation for k, v in top1_metrics_cross.items()}

    print('Average binary metrics across folds:', metrics_cross_avg)
    print('🎯 Average Top-1 metrics across folds:', top1_metrics_cross_avg)

    # 打印详细的CV_type风格指标(去掉前两行)
    cv_type_metric_names = ['AUC', 'AUPR', 'F1', 'Accuracy', 'Recall', 'Specificity', 'Precision']
    print("\n📊 CV_type Style Metrics Details:")
    for name, value in zip(cv_type_metric_names, cv_type_metrics_cross_avg[2:]):  # 跳过前两个值
        print(f"  {name:15}: {value:.4f}")

    # 打印Top-1指标详情
    print("\n🎯 Top-1 Metrics Details:")
    for name, value in top1_metrics_cross_avg.items():
        print(f"  {name:15}: {value:.4f}")

    return metrics_cross_avg, cv_type_metrics_cross_avg, top1_metrics_cross_avg


# 运行主程序
if __name__ == '__main__':
    from param import parameter_parser

    args = parameter_parser()

    # 🐛 FIX BUG (2026-05-11): seed_torch() ở line 65 dùng default 1234, KHÔNG đọc args.seed.
    # Multi-seed experiment broken trước khi fix này — mọi run dùng cùng seed.
    # Re-seed RNG state với args.seed sau khi parse args.
    seed_torch(args.seed)
    print(f"[SEED] Re-seeded với args.seed = {args.seed}")

    # 启用性能分析
    start_time = time.time()

    # 选择使用增强评估版本(包含Top-1指标)
    print("🚀 Running with comprehensive evaluation including Top-1 metrics...")
    result = main_optimized(args)

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    if result is not None:
        print("\n" + "=" * 90)
        print("🎯 FINAL COMPREHENSIVE RESULTS WITH TOP-1 METRICS")
        print("=" * 90)

        if isinstance(result, tuple) and len(result) >= 3:
            binary_metrics, cv_type_metrics, top1_metrics = result[:3]
            multiclass_metrics = result[3] if len(result) > 3 else None

            print("📊 Binary Classification Metrics (Original):")
            print(f"  AUC: {binary_metrics[0][0]:.4f}")
            print(f"  AUPR: {binary_metrics[0][1]:.4f}")
            print(f"  F1: {binary_metrics[0][2]:.4f}")
            print(f"  Accuracy: {binary_metrics[0][3]:.4f}")
            print(f"  Recall: {binary_metrics[0][4]:.4f}")
            print(f"  Specificity: {binary_metrics[0][5]:.4f}")
            print(f"  Precision: {binary_metrics[0][6]:.4f}")

            print("\n🎯 CV_type Style Metrics (Enhanced):")
            cv_type_names = ['AUC', 'AUPR', 'F1', 'Accuracy', 'Recall', 'Specificity', 'Precision']
            for name, value in zip(cv_type_names, cv_type_metrics[2:]):  # 跳过前两个值
                print(f"  {name}: {value:.4f}")

            print("\n🏆 Top-1 Metrics (NEW):")
            print(f"  Top-1 Precision: {top1_metrics['top1_precision']:.4f}")
            print(f"  Top-1 Recall: {top1_metrics['top1_recall']:.4f}")
            print(f"  Top-1 F1: {top1_metrics['top1_f1']:.4f}")


        else:
            print("📊 Binary Classification Metrics:")
            print(f"  AUC: {result[0][0]:.4f}")
            print(f"  AUPR: {result[0][1]:.4f}")
            print(f"  F1: {result[0][2]:.4f}")
            print(f"  Accuracy: {result[0][3]:.4f}")
            print(f"  Recall: {result[0][4]:.4f}")
            print(f"  Specificity: {result[0][5]:.4f}")
            print(f"  Precision: {result[0][6]:.4f}")

        print("=" * 90)
    else:
        print("❌ Execution failed. Please check data files and configurations.")