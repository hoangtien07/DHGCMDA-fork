import torch
import numpy as np


def create_enhanced_hetero_data(train_data, mi_sim_recon=None, dis_sim_recon=None, device=None):
    """
    创建增强的异构图数据结构，支持边特征，方便独立导入

    Args:
        train_data: 包含原始矩阵的训练数据
        mi_sim_recon: 重构的miRNA相似性矩阵（可选）
        dis_sim_recon: 重构的疾病相似性矩阵（可选）
        device: 计算设备（可选）

    Returns:
        hetero_data: 异构图数据结构
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查PyTorch Geometric是否可用
    try:
        from torch_geometric.data import HeteroData
        TORCH_GEOMETRIC_AVAILABLE = True
    except ImportError:
        TORCH_GEOMETRIC_AVAILABLE = False
        print("Warning: PyTorch Geometric not available. Using simplified hetero data structure.")

    # 处理train_data数据，确保按照预期格式提取数据
    try:
        # 使用关联存在矩阵而不是类型矩阵
        # 确保索引正确
        if isinstance(train_data, list) and len(train_data) >= 8:
            # 尝试获取association_matrix
            if isinstance(train_data[7], torch.Tensor):
                association_matrix = train_data[7].to(device).float()  # 关联存在矩阵
            else:
                print("Warning: train_data[7] is not a tensor. Trying to fall back to train_data[4].")
                association_matrix = train_data[4].to(device).float()

            # 尝试获取association_type_matrix
            if isinstance(train_data[4], torch.Tensor):
                association_type_matrix = train_data[4].to(device).float()  # 关联类型矩阵
            else:
                print("Warning: train_data[4] is not a tensor. Using same as association_matrix.")
                association_type_matrix = association_matrix

        elif isinstance(train_data, list) and len(train_data) >= 5:
            # 尝试获取association_matrix和association_type_matrix
            if isinstance(train_data[4], torch.Tensor):
                association_matrix = train_data[4].to(device).float()  # 关联存在矩阵 (可能只有一种类型)
                association_type_matrix = train_data[4].to(device).float()  # 作为兜底使用相同矩阵
            else:
                print("Warning: train_data[4] is not a tensor. Creating empty matrix.")
                # 创建一个空矩阵作为兜底
                association_matrix = torch.zeros((100, 100), device=device).float()  # 使用默认大小
                association_type_matrix = association_matrix
        else:
            print("Warning: train_data does not have sufficient elements. Creating empty matrices.")
            association_matrix = torch.zeros((100, 100), device=device).float()  # 使用默认大小
            association_type_matrix = association_matrix
    except Exception as e:
        print(f"Error extracting matrices from train_data: {e}")
        # 创建一个空矩阵作为兜底
        association_matrix = torch.zeros((100, 100), device=device).float()
        association_type_matrix = association_matrix

    # 将关联矩阵二值化确保表示存在
    association_exists = (association_matrix != 0).float()

    # 使用重构的相似性矩阵（如果提供），否则使用原始矩阵
    if mi_sim_recon is not None:
        mi_sim = mi_sim_recon.to(device).float()  # 重构的miRNA相似性矩阵
    else:
        if isinstance(train_data, list) and len(train_data) >= 2:
            if isinstance(train_data[1], torch.Tensor):
                mi_sim = train_data[1].to(device).float()  # 原始miRNA相似性矩阵
            else:
                print("Warning: train_data[1] is not a tensor. Creating identity matrix.")
                mi_sim = torch.eye(association_matrix.shape[0], device=device).float()
        else:
            mi_sim = torch.eye(association_matrix.shape[0], device=device).float()

    if dis_sim_recon is not None:
        dis_sim = dis_sim_recon.to(device).float()  # 重构的疾病相似性矩阵
    else:
        if isinstance(train_data, list) and len(train_data) >= 1:
            if isinstance(train_data[0], torch.Tensor):
                dis_sim = train_data[0].to(device).float()  # 原始疾病相似性矩阵
            else:
                print("Warning: train_data[0] is not a tensor. Creating identity matrix.")
                dis_sim = torch.eye(association_matrix.shape[1], device=device).float()
        else:
            dis_sim = torch.eye(association_matrix.shape[1], device=device).float()

    # 节点数量
    mi_num = association_matrix.shape[0]
    dis_num = association_matrix.shape[1]

    # 设置边存在的阈值
    threshold = 0.5  # 可根据需要调整

    # 创建异构图数据结构
    if TORCH_GEOMETRIC_AVAILABLE:
        from torch_geometric.data import HeteroData
        hetero_data = HeteroData()

        # 创建节点特征 - 使用独热编码作为初始特征
        hetero_data['miRNA'].x = torch.eye(mi_num, device=device).float()
        hetero_data['disease'].x = torch.eye(dis_num, device=device).float()

        # 创建miRNA-disease关联边
        md_edges = []
        md_edge_attr = []
        dm_edges = []
        dm_edge_attr = []

        # 当创建边属性时，使用关联类型作为额外特征
        for i in range(mi_num):
            for j in range(dis_num):
                if association_exists[i, j] > 0:
                    # miRNA -> disease边
                    md_edges.append([i, j])
                    # 边属性包含关联存在和类型
                    md_edge_attr.append([float(association_exists[i, j]), float(association_type_matrix[i, j])])

                    # disease -> miRNA边（反向）
                    dm_edges.append([j, i])
                    dm_edge_attr.append([float(association_exists[i, j]), float(association_type_matrix[i, j])])

        # 添加miRNA-disease关联边
        if len(md_edges) > 0:
            hetero_data['miRNA', 'associates', 'disease'].edge_index = torch.tensor(md_edges, device=device).t()
            hetero_data['miRNA', 'associates', 'disease'].edge_attr = torch.tensor(md_edge_attr, device=device)

            hetero_data['disease', 'associates', 'miRNA'].edge_index = torch.tensor(dm_edges, device=device).t()
            hetero_data['disease', 'associates', 'miRNA'].edge_attr = torch.tensor(dm_edge_attr, device=device)
        else:
            # 添加空边以保持结构
            hetero_data['miRNA', 'associates', 'disease'].edge_index = torch.zeros((2, 0), dtype=torch.long,
                                                                                   device=device)
            hetero_data['miRNA', 'associates', 'disease'].edge_attr = torch.zeros((0, 2), device=device)

            hetero_data['disease', 'associates', 'miRNA'].edge_index = torch.zeros((2, 0), dtype=torch.long,
                                                                                   device=device)
            hetero_data['disease', 'associates', 'miRNA'].edge_attr = torch.zeros((0, 2), device=device)

        # miRNA相似性边
        mm_edges = []
        mm_edge_attr = []
        for i in range(mi_num):
            for j in range(mi_num):
                if i != j and mi_sim[i, j] > threshold:
                    mm_edges.append([i, j])
                    mm_edge_attr.append([float(mi_sim[i, j])])

        if len(mm_edges) > 0:
            hetero_data['miRNA', 'similar', 'miRNA'].edge_index = torch.tensor(mm_edges, device=device).t()
            hetero_data['miRNA', 'similar', 'miRNA'].edge_attr = torch.tensor(mm_edge_attr, device=device)
        else:
            hetero_data['miRNA', 'similar', 'miRNA'].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            hetero_data['miRNA', 'similar', 'miRNA'].edge_attr = torch.zeros((0, 1), device=device)

        # 疾病相似性边
        dd_edges = []
        dd_edge_attr = []
        for i in range(dis_num):
            for j in range(dis_num):
                if i != j and dis_sim[i, j] > threshold:
                    dd_edges.append([i, j])
                    dd_edge_attr.append([float(dis_sim[i, j])])

        if len(dd_edges) > 0:
            hetero_data['disease', 'similar', 'disease'].edge_index = torch.tensor(dd_edges, device=device).t()
            hetero_data['disease', 'similar', 'disease'].edge_attr = torch.tensor(dd_edge_attr, device=device)
        else:
            hetero_data['disease', 'similar', 'disease'].edge_index = torch.zeros((2, 0), dtype=torch.long,
                                                                                  device=device)
            hetero_data['disease', 'similar', 'disease'].edge_attr = torch.zeros((0, 1), device=device)

    else:
        # 简化版异构图数据结构（无PyTorch Geometric）
        hetero_data = type('HeteroDataSimple', (), {})()
        hetero_data.edge_index_dict = {}
        hetero_data.x_dict = {
            'miRNA': torch.eye(mi_num, device=device).float(),
            'disease': torch.eye(dis_num, device=device).float()
        }

        # 添加关联边
        md_edges = []
        md_edge_attr = []
        dm_edges = []
        dm_edge_attr = []

        for i in range(mi_num):
            for j in range(dis_num):
                if association_exists[i, j] > 0:
                    # miRNA -> disease边
                    md_edges.append([i, j])
                    md_edge_attr.append([float(association_exists[i, j]), float(association_type_matrix[i, j])])

                    # disease -> miRNA边（反向）
                    dm_edges.append([j, i])
                    dm_edge_attr.append([float(association_exists[i, j]), float(association_type_matrix[i, j])])

        if len(md_edges) > 0:
            hetero_data.edge_index_dict[('miRNA', 'associates', 'disease')] = torch.tensor(md_edges, device=device).t()
            hetero_data.edge_attr_dict = {('miRNA', 'associates', 'disease'): torch.tensor(md_edge_attr, device=device)}

            hetero_data.edge_index_dict[('disease', 'associates', 'miRNA')] = torch.tensor(dm_edges, device=device).t()
            hetero_data.edge_attr_dict[('disease', 'associates', 'miRNA')] = torch.tensor(dm_edge_attr, device=device)
        else:
            hetero_data.edge_index_dict[('miRNA', 'associates', 'disease')] = torch.zeros((2, 0), dtype=torch.long,
                                                                                          device=device)
            hetero_data.edge_attr_dict = {('miRNA', 'associates', 'disease'): torch.zeros((0, 2), device=device)}

            hetero_data.edge_index_dict[('disease', 'associates', 'miRNA')] = torch.zeros((2, 0), dtype=torch.long,
                                                                                          device=device)
            hetero_data.edge_attr_dict[('disease', 'associates', 'miRNA')] = torch.zeros((0, 2), device=device)

        # 添加miRNA相似性边
        mm_edges = []
        mm_edge_attr = []
        for i in range(mi_num):
            for j in range(mi_num):
                if i != j and mi_sim[i, j] > threshold:
                    mm_edges.append([i, j])
                    mm_edge_attr.append([float(mi_sim[i, j])])

        if len(mm_edges) > 0:
            hetero_data.edge_index_dict[('miRNA', 'similar', 'miRNA')] = torch.tensor(mm_edges, device=device).t()
            hetero_data.edge_attr_dict[('miRNA', 'similar', 'miRNA')] = torch.tensor(mm_edge_attr, device=device)
        else:
            hetero_data.edge_index_dict[('miRNA', 'similar', 'miRNA')] = torch.zeros((2, 0), dtype=torch.long,
                                                                                     device=device)
            hetero_data.edge_attr_dict[('miRNA', 'similar', 'miRNA')] = torch.zeros((0, 1), device=device)

        # 添加疾病相似性边
        dd_edges = []
        dd_edge_attr = []
        for i in range(dis_num):
            for j in range(dis_num):
                if i != j and dis_sim[i, j] > threshold:
                    dd_edges.append([i, j])
                    dd_edge_attr.append([float(dis_sim[i, j])])

        if len(dd_edges) > 0:
            hetero_data.edge_index_dict[('disease', 'similar', 'disease')] = torch.tensor(dd_edges, device=device).t()
            hetero_data.edge_attr_dict[('disease', 'similar', 'disease')] = torch.tensor(dd_edge_attr, device=device)
        else:
            hetero_data.edge_index_dict[('disease', 'similar', 'disease')] = torch.zeros((2, 0), dtype=torch.long,
                                                                                         device=device)
            hetero_data.edge_attr_dict[('disease', 'similar', 'disease')] = torch.zeros((0, 1), device=device)

    return hetero_data