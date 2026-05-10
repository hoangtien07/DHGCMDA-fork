import csv
import os
import torch as t
import numpy as np
from math import e
import pandas as pd
from scipy import io
from functools import lru_cache
import pickle
import hashlib
import time


def read_csv(path):
    """CSV读取函数"""
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    """文本文件读取函数"""
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def read_mat(path, name):
    """MAT文件读取函数"""
    matrix = io.loadmat(path)
    matrix = t.FloatTensor(matrix[name])
    return matrix


def read_excel(path, sheet_name=0):
    """读取Excel文件"""
    try:
        data = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
        return t.FloatTensor(data.values).float()
    except Exception as e:
        print(f"Error reading Excel file {path}: {e}")
        return None


def read_association_csv(path):
    """
    读取多类型关联CSV文件并转换为关联矩阵

    关联类型说明（保持原始生物学编码）:
    1 -> 循环类型 (Circulation)
    2 -> 表观遗传学类型 (Epigenetics)
    3 -> 靶标类型 (Target)
    4 -> 遗传学类型 (Genetics)
    0 -> 无关联
    """
    try:
        data = pd.read_csv(path, header=None)
        print(f"✅ Loaded multi-type association pairs from {path}: {data.shape}")

        # 提取数据（转换为0-based索引）
        mirna_indices = data.iloc[:, 0].values - 1
        disease_indices = data.iloc[:, 1].values - 1
        association_types = data.iloc[:, 2].values  # 保持原始类型编码 1,2,3,4

        # 创建关联矩阵 - 495个miRNA × 383个疾病
        max_mirna = 495
        max_disease = 383
        association_matrix = np.zeros((max_mirna, max_disease), dtype=np.float32)

        # 填充关联矩阵（保持原始类型值 1,2,3,4）
        for i, j, atype in zip(mirna_indices, disease_indices, association_types):
            if 0 <= i < max_mirna and 0 <= j < max_disease:
                association_matrix[i, j] = atype  # 不做转换，保持1,2,3,4

        print(f"✅ Created multi-type association matrix: {association_matrix.shape}")

        # 统计分析（带生物学含义）
        print(f"📊 Multi-type Association Statistics:")
        type_names = {
            0: "无关联 (No association)",
            1: "循环类型 (Circulation)",
            2: "表观遗传学 (Epigenetics)",
            3: "靶标类型 (Target)",
            4: "遗传学类型 (Genetics)"
        }

        unique_vals, counts = np.unique(association_matrix, return_counts=True)
        for val, count in zip(unique_vals, counts):
            type_name = type_names.get(int(val), f"未知类型 {int(val)}")
            percentage = count / association_matrix.size * 100
            print(f"   {type_name}: {count} pairs ({percentage:.2f}%)")

        return t.FloatTensor(association_matrix)

    except Exception as e:
        print(f"❌ Error reading multi-type association CSV {path}: {e}")
        return None


def read_md_data(path, validation):
    """批量读取MD数据"""
    result = [{} for _ in range(validation)]
    for filename in os.listdir(path):
        data_type = filename[filename.index('_') + 1:filename.index('.') - 1]
        num = int(filename[filename.index('.') - 1])
        result[num - 1][data_type] = read_csv(os.path.join(path, filename))
    return result


def get_edge_index(matrix):
    """边索引提取"""
    indices = t.nonzero(matrix, as_tuple=True)
    edge_index = t.stack(indices, dim=0)
    return edge_index


@lru_cache(maxsize=5)
def cached_gauss_calculation(matrix_hash, N, matrix_type):
    """缓存的高斯计算"""
    pass


def Gauss_M_optimized(adj_matrix, N):
    """高斯相似性计算 - miRNA"""
    adj_matrix = np.array(adj_matrix).astype(np.float32)
    adj_squared = adj_matrix * adj_matrix
    adj_sum = np.sum(adj_squared, axis=1, keepdims=True)
    diff_matrix = adj_sum + adj_sum.T - 2 * np.dot(adj_matrix, adj_matrix.T)
    rm = N * 1. / np.sum(adj_squared)
    GM = np.exp(-rm * diff_matrix)
    return GM.astype(np.float32)


def Gauss_D_optimized(adj_matrix, M):
    """高斯相似性计算 - 疾病"""
    adj_matrix = np.array(adj_matrix).astype(np.float32)
    T = adj_matrix.T
    T_squared = T * T
    T_sum = np.sum(T_squared, axis=1, keepdims=True)
    diff_matrix = T_sum + T_sum.T - 2 * np.dot(T, T.T)
    rd = M * 1. / np.sum(T_squared)
    GD = np.exp(-rd * diff_matrix)
    return GD.astype(np.float32)


class OptimizedDataPreprocessor:
    """数据预处理器"""

    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, cache_key):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _compute_hash(self, data):
        """计算数据哈希"""
        if isinstance(data, (pd.DataFrame, np.ndarray)):
            return hashlib.md5(data.tobytes()).hexdigest()
        elif isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()

    def _load_from_cache(self, cache_key):
        """从缓存加载数据"""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def _save_to_cache(self, cache_key, data):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass

    def preprocess_similarity_matrices(self, dd_mat, mm_mat, association_matrix):
        """相似性矩阵预处理"""
        dd_hash = self._compute_hash(dd_mat)
        mm_hash = self._compute_hash(mm_mat)
        assoc_hash = self._compute_hash(association_matrix)
        cache_key = f"sim_matrices_{dd_hash}_{mm_hash}_{assoc_hash}"

        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            print("Loaded similarity matrices from cache")
            return cached_result

        print("Computing similarity matrices...")
        start_time = time.time()

        nd = association_matrix.shape[1]
        nm = association_matrix.shape[0]

        DGSM = Gauss_D_optimized(association_matrix, nd)
        MGSM = Gauss_M_optimized(association_matrix, nm)

        ID = np.zeros([nd, nd], dtype=np.float32)
        IM = np.zeros([nm, nm], dtype=np.float32)

        zero_mask_d = (dd_mat == 0)
        ID = np.where(zero_mask_d, DGSM, (dd_mat + DGSM) / 2)

        zero_mask_m = (mm_mat == 0)
        IM = np.where(zero_mask_m, MGSM, (mm_mat + MGSM) / 2)

        result = {
            'ID': t.from_numpy(ID).float(),
            'IM': t.from_numpy(IM).float(),
            'DGSM': t.from_numpy(DGSM).float(),
            'MGSM': t.from_numpy(MGSM).float()
        }

        self._save_to_cache(cache_key, result)
        elapsed_time = time.time() - start_time
        print(f"Similarity matrices computed in {elapsed_time:.2f} seconds")
        return result

    def preprocess_indices(self, association_matrix, validation, seed=0):
        """索引预处理.

        🐛 FIX (2026-05-11): Trước đây `np.random.seed(0)` hardcoded → train/test split
        FIXED across mọi --seed. Giờ accept `seed` parameter và include vào cache_key
        để multi-seed thực sự cho different splits.
        """
        cache_key = f"indices_{self._compute_hash(association_matrix)}_{validation}_seed{seed}"

        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            print(f"Loaded indices from cache (seed={seed})")
            return cached_result

        print(f"Computing indices (seed={seed})...")
        start_time = time.time()

        type_indices = {
            'zero': [],
            'type1': [],
            'type2': [],
            'type3': [],
            'type4': [],
            'nonzero': []
        }

        indices_i, indices_j = np.where(association_matrix != 0)
        for i, j in zip(indices_i, indices_j):
            value = association_matrix[i, j].item()
            type_indices['nonzero'].append([i, j])

            if value == 1:
                type_indices['type1'].append([i, j])
            elif value == 2:
                type_indices['type2'].append([i, j])
            elif value == 3:
                type_indices['type3'].append([i, j])
            elif value == 4:
                type_indices['type4'].append([i, j])

        zero_indices_i, zero_indices_j = np.where(association_matrix == 0)
        for i, j in zip(zero_indices_i, zero_indices_j):
            type_indices['zero'].append([i, j])

        # FIX: dùng `seed` parameter thay vì hardcoded 0 → multi-seed get different splits
        np.random.seed(seed)
        for key in type_indices:
            np.random.shuffle(type_indices[key])
            type_indices[key] = t.LongTensor(type_indices[key])

        zero_tensor = type_indices['zero']
        nonzero_tensor = type_indices['nonzero']

        zero_splits = zero_tensor.split(int(zero_tensor.size(0) / 10), dim=0)
        nonzero_splits = nonzero_tensor.split(int(nonzero_tensor.size(0) / 10), dim=0)

        cross_zero_index = t.cat([zero_splits[i] for i in range(9)])
        cross_nonzero_index = t.cat([nonzero_splits[j] for j in range(9)])

        new_zero_splits = cross_zero_index.split(int(cross_zero_index.size(0) / validation), dim=0)
        new_nonzero_splits = cross_nonzero_index.split(int(cross_nonzero_index.size(0) / validation), dim=0)

        cv_data = []
        for i in range(validation):
            train_indices = [j for j in range(validation) if j != i]
            cv_fold = {
                'test': [new_nonzero_splits[i], new_zero_splits[i]],
                'train': [
                    t.cat([new_nonzero_splits[j] for j in train_indices]),
                    t.cat([new_zero_splits[j] for j in train_indices])
                ]
            }
            cv_data.append(cv_fold)

        independent_test = {
            'test': [nonzero_splits[-2], zero_splits[-2]],
            'train': [cross_nonzero_index, cross_zero_index]
        }

        result = {
            'cv_data': cv_data,
            'independent': [independent_test],
            'type_indices': {
                'type1': type_indices['type1'],
                'type2': type_indices['type2'],
                'type3': type_indices['type3'],
                'type4': type_indices['type4'],
            }
        }

        self._save_to_cache(cache_key, result)
        elapsed_time = time.time() - start_time
        print(f"Indices computed in {elapsed_time:.2f} seconds")
        return result


def prepare_data_optimized(opt):
    """双视图数据准备函数 - 适配新的文件格式"""
    print("Starting dual-view data preparation with new file format...")
    total_start_time = time.time()

    dataset = {}
    preprocessor = OptimizedDataPreprocessor()

    print("Loading all similarity and feature files from v2.0_495m383D...")

    # 1. 加载疾病语义相似性 (D_SSM1.txt) - Disease View 2的基础
    dd_sem_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'D_SSM1.txt')
    if os.path.exists(dd_sem_path):
        dd_sem_data = read_txt(dd_sem_path)
        dd_sem_mat = dd_sem_data.numpy().astype(np.float32)
        print(f"✅ Loaded disease semantic similarity from D_SSM1.txt: {dd_sem_mat.shape}")
    else:
        print("❌ D_SSM1.txt not found!")
        return None

    # 2. 加载miRNA功能相似性 (M_FSM.txt) - miRNA View 2的基础
    mm_fun_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'M_FSM.txt')
    if os.path.exists(mm_fun_path):
        mm_fun_data = read_txt(mm_fun_path)
        mm_fun_mat = mm_fun_data.numpy().astype(np.float32)
        print(f"✅ Loaded miRNA functional similarity from M_FSM.txt: {mm_fun_mat.shape}")
    else:
        print("❌ M_FSM.txt not found!")
        return None

    # 3. 加载关联数据并转换为矩阵 (multi_all_mirna_disease_pairs_without_negative.csv)
    mi_dis_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'multi_all_mirna_disease_pairs_without_negative.csv')
    if os.path.exists(mi_dis_path):
        association_matrix_tensor = read_association_csv(mi_dis_path)
        if association_matrix_tensor is not None:
            association_matrix = association_matrix_tensor.numpy()
            print(f"✅ Created association matrix from CSV: {association_matrix.shape}")
        else:
            print("❌ Failed to create association matrix!")
            return None
    else:
        print("❌ multi_all_mirna_disease_pairs_without_negative.csv not found!")
        return None

    dataset['md_p'] = t.FloatTensor(association_matrix)
    dataset['md_true'] = dataset['md_p']

    # 4. 加载疾病-基因特征 (D_SSM2.txt) - Disease View 1的基础
    d_gs_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'D_SSM2.txt')
    if os.path.exists(d_gs_path):
        d_gs_data = read_txt(d_gs_path)
        dataset['d_gs'] = d_gs_data
        print(f"✅ Loaded disease-gene features from D_SSM2.txt: {d_gs_data.shape}")
    else:
        print("⚠️ D_SSM2.txt not found, using semantic similarity as fallback")
        dataset['d_gs'] = t.FloatTensor(dd_sem_mat)

    # 5. 加载miRNA-序列特征 (M_GSM.txt) - miRNA View 1的基础
    m_ss_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'M_GSM.txt')
    if os.path.exists(m_ss_path):
        m_ss_data = read_txt(m_ss_path)
        dataset['m_ss'] = m_ss_data
        print(f"✅ Loaded miRNA-sequence features from M_GSM.txt: {m_ss_data.shape}")
    else:
        print("⚠️ M_GSM.txt not found, using functional similarity as fallback")
        dataset['m_ss'] = t.FloatTensor(mm_fun_mat)

    # 存储四个不同的相似性/特征源
    dataset['dis_sem'] = t.FloatTensor(dd_sem_mat)  # 疾病语义相似性
    dataset['mi_fun'] = t.FloatTensor(mm_fun_mat)  # miRNA功能相似性

    # 为了向后兼容，使用语义相似性作为主要疾病相似性，功能相似性作为主要miRNA相似性
    print("Processing integrated similarity matrices...")
    similarity_results = preprocessor.preprocess_similarity_matrices(
        dd_sem_mat, mm_fun_mat, association_matrix)

    dataset['ID'] = similarity_results['ID']  # 整合的疾病相似性
    dataset['IM'] = similarity_results['IM']  # 整合的miRNA相似性

    # 预处理索引 — truyền seed để multi-seed thực sự cho different train/test split
    print("Processing indices...")
    seed = getattr(opt, 'seed', 0)
    index_results = preprocessor.preprocess_indices(
        association_matrix, opt.validation, seed=seed)

    dataset['md'] = index_results['cv_data']
    dataset['independent'] = index_results['independent']
    dataset['type_indices'] = index_results['type_indices']

    total_time = time.time() - total_start_time
    print(f"✅ Dual-view data preparation completed in {total_time:.2f} seconds")

    # 打印双视图配置摘要
    print("\n" + "=" * 60)
    print("🎯 DUAL-VIEW CONFIGURATION SUMMARY (v2.0_495m383D)")
    print("=" * 60)
    print("miRNA Dual Views:")
    print(f"  📊 View 1 (Sequence): M_GSM.txt {dataset['m_ss'].shape}")
    print(f"  📊 View 2 (Function): M_FSM.txt {dataset['mi_fun'].shape}")
    print("\nDisease Dual Views:")
    print(f"  📊 View 1 (Gene): D_SSM2.txt {dataset['d_gs'].shape}")
    print(f"  📊 View 2 (Semantic): D_SSM1.txt {dataset['dis_sem'].shape}")
    print(f"\nAssociation Matrix: {dataset['md_p'].shape}")
    print(f"Data source: v2.0_495m383D (495 miRNAs × 383 Diseases)")
    print("=" * 60)

    return dataset


def prepare_data(opt):
    """保持原代码兼容的接口"""
    return prepare_data_optimized(opt)


def Gauss_M(adj_matrix, N):
    """兼容性包装器"""
    return Gauss_M_optimized(adj_matrix, N)


def Gauss_D(adj_matrix, M):
    """兼容性包装器"""
    return Gauss_D_optimized(adj_matrix, M)


if __name__ == "__main__":
    from param import parameter_parser

    print("Testing dual-view data preparation with new file format...")
    args = parameter_parser()

    args.data_path = './'

    start_time = time.time()
    dataset = prepare_data_optimized(args)
    end_time = time.time()

    if dataset is not None:
        print(f"✅ Test completed in {end_time - start_time:.2f} seconds")
        print(f"Dataset keys: {list(dataset.keys())}")
        print(f"Association matrix shape: {dataset['md_p'].shape}")
        print(f"Number of cross-validation folds: {len(dataset['md'])}")

        # 打印数据集统计信息
        association_matrix = dataset['md_p'].numpy()
        unique_vals, counts = np.unique(association_matrix, return_counts=True)
        print(f"\nAssociation statistics:")
        for val, count in zip(unique_vals, counts):
            print(f"  Value {val}: {count} pairs ({count / association_matrix.size * 100:.2f}%)")
    else:
        print("❌ Test failed - missing required data files")