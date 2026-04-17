import numpy as np
import torch
import math


def Eu_dis(x):
    # 确保输入数据是float32类型
    x = np.asmatrix(x).astype(np.float32)

    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0

    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    features = None
    for f in F_list:
        if f is not None and f != []:
            # 确保输入数据是float32类型
            f = np.array(f).astype(np.float32)

            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])

            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max

            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        # if h is not None and h != []:
        if h is not None and len(h) != 0:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    # 确保H是float32类型
    H = np.array(H).astype(np.float32)

    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge, dtype=np.float32)

    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    invDE = np.asmatrix(np.diag(np.power(DE, -1)))
    DV2 = np.asmatrix(np.diag(np.power(DV, -0.5)))

    W = np.asmatrix(np.diag(W))
    H = np.asmatrix(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        G = torch.FloatTensor(G)  # 使用FloatTensor确保是float32类型
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    # 确保距离矩阵是float32类型
    dis_mat = dis_mat.astype(np.float32)

    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge), dtype=np.float32)
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(
                    -dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)  ## affinity matrix的计算公式
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs, split_diff_scale=False, is_probH=False, m_prob=1):
    # 确保X是float32类型
    X = np.array(X).astype(np.float32)

    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)

        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)

    return H