from __future__ import division
import torch
import numpy as np
from functools import lru_cache
import time
from collections import defaultdict


class OptimizedDataset(object):
    """数据集类，支持双视图架构"""

    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 缓存机制
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # 预处理数据到GPU（如果可用）
        self._preload_to_device()

        print(f"OptimizedDataset initialized with {self.nums} folds")
        print(f"Using device: {self.device}")

    def _preload_to_device(self):
        """预加载数据到设备内存 - 支持双视图"""
        print("Preloading dual-view data to device...")
        start_time = time.time()

        # 预加载相似性矩阵（这些在所有fold中都相同）
        if 'ID' in self.data_set:
            self.data_set['ID'] = self.data_set['ID'].to(self.device).float()
        if 'IM' in self.data_set:
            self.data_set['IM'] = self.data_set['IM'].to(self.device).float()
        if 'md_p' in self.data_set:
            self.data_set['md_p'] = self.data_set['md_p'].to(self.device).float()
        if 'md_true' in self.data_set:
            self.data_set['md_true'] = self.data_set['md_true'].to(self.device).float()

        # 预加载四个双视图特征源
        if 'd_gs' in self.data_set:
            self.data_set['d_gs'] = self.data_set['d_gs'].to(self.device).float()
            print(f"✅ Loaded disease-gene features to device: {self.data_set['d_gs'].shape}")

        if 'm_ss' in self.data_set:
            self.data_set['m_ss'] = self.data_set['m_ss'].to(self.device).float()
            print(f"✅ Loaded miRNA-sequence features to device: {self.data_set['m_ss'].shape}")

        if 'dis_sem' in self.data_set:
            self.data_set['dis_sem'] = self.data_set['dis_sem'].to(self.device).float()
            print(f"✅ Loaded disease semantic similarity to device: {self.data_set['dis_sem'].shape}")

        if 'mi_fun' in self.data_set:
            self.data_set['mi_fun'] = self.data_set['mi_fun'].to(self.device).float()
            print(f"✅ Loaded miRNA functional similarity to device: {self.data_set['mi_fun'].shape}")

        # 预处理索引数据
        for i in range(self.nums):
            if 'md' in self.data_set and i < len(self.data_set['md']):
                # 训练索引
                if 'train' in self.data_set['md'][i]:
                    train_indices = self.data_set['md'][i]['train']
                    if len(train_indices) >= 2:
                        self.data_set['md'][i]['train'][0] = train_indices[0].to(self.device)
                        self.data_set['md'][i]['train'][1] = train_indices[1].to(self.device)

                # 测试索引
                if 'test' in self.data_set['md'][i]:
                    test_indices = self.data_set['md'][i]['test']
                    if len(test_indices) >= 2:
                        self.data_set['md'][i]['test'][0] = test_indices[0].to(self.device)
                        self.data_set['md'][i]['test'][1] = test_indices[1].to(self.device)

        # 预处理独立测试集
        if 'independent' in self.data_set and len(self.data_set['independent']) > 0:
            independent = self.data_set['independent'][0]
            if 'train' in independent and len(independent['train']) >= 2:
                independent['train'][0] = independent['train'][0].to(self.device)
                independent['train'][1] = independent['train'][1].to(self.device)
            if 'test' in independent and len(independent['test']) >= 2:
                independent['test'][0] = independent['test'][0].to(self.device)
                independent['test'][1] = independent['test'][1].to(self.device)

        elapsed_time = time.time() - start_time
        print(f"✅ Dual-view data preloading completed in {elapsed_time:.2f} seconds")

    @lru_cache(maxsize=32)
    def _get_cached_item(self, index):
        """缓存版本的数据获取 - 支持双视图"""
        self._cache_misses += 1

        # 确保索引有效
        if index >= self.nums:
            raise IndexError(f"Index {index} out of range for {self.nums} folds")

        try:
            # 构建返回的数据元组，包含四个双视图特征源
            result = (
                self.data_set['dis_sem'],  # 0: 疾病语义相似性 (Disease View 2)
                self.data_set['mi_fun'],  # 1: miRNA功能相似性 (miRNA View 2)
                self.data_set['md'][index]['train'],  # 2: 训练索引
                self.data_set['md'][index]['test'],  # 3: 测试索引
                self.data_set['md_p'],  # 4: 关联矩阵
                self.data_set['md_true'],  # 5: 真实关联矩阵
                self.data_set['independent'][0]['train'],  # 6: 独立训练集
                self.data_set['independent'][0]['test'],  # 7: 独立测试集
                self.data_set.get('d_gs', torch.eye(100, device=self.device).float()),  # 8: 疾病-基因特征 (Disease View 1)
                self.data_set.get('m_ss', torch.eye(100, device=self.device).float()),  # 9: miRNA-序列特征 (miRNA View 1)
                self.data_set['ID'],  # 10: 整合疾病相似性
                self.data_set['IM'],  # 11: 整合miRNA相似性
            )

            return result

        except KeyError as e:
            print(f"KeyError accessing data for index {index}: {e}")
            # 返回默认值以避免程序崩溃
            default_tensor = torch.zeros((100, 100), device=self.device).float()
            default_indices = [torch.zeros((10, 2), device=self.device).long(),
                               torch.zeros((10, 2), device=self.device).long()]

            return (default_tensor, default_tensor, default_indices, default_indices,
                    default_tensor, default_tensor, default_indices, default_indices,
                    default_tensor, default_tensor, default_tensor, default_tensor)

    def __getitem__(self, index):
        """优化的数据获取方法"""
        # 检查缓存
        cache_key = f"item_{index}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        # 获取数据
        result = self._get_cached_item(index)

        # 缓存结果（限制缓存大小）
        if len(self._cache) < 16:
            self._cache[cache_key] = result

        return result

    def __len__(self):
        """返回数据集大小"""
        return self.nums

    def get_cache_statistics(self):
        """获取缓存统计信息"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        }

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def prefetch_all(self):
        """预取所有数据到缓存"""
        print("Prefetching all dual-view data...")
        start_time = time.time()

        for i in range(self.nums):
            _ = self[i]  # 触发数据加载和缓存

        elapsed_time = time.time() - start_time
        print(f"Prefetching completed in {elapsed_time:.2f} seconds")

        cache_stats = self.get_cache_statistics()
        print(f"Cache statistics: {cache_stats}")


class BatchedDataLoader:
    """批处理数据加载器"""

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __iter__(self):
        """迭代器实现"""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = []

            for idx in batch_indices:
                batch_data.append(self.dataset[idx])

            if len(batch_data) == 1:
                yield batch_data[0]
            else:
                yield self._collate_batch(batch_data)

    def _collate_batch(self, batch_data):
        """整理批处理数据"""
        if len(batch_data) == 1:
            return batch_data[0]
        return batch_data

    def __len__(self):
        """返回批次数量"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DatasetValidator:
    """数据集验证器，用于检查双视图数据完整性"""

    @staticmethod
    def validate_dataset(dataset_obj):
        """验证双视图数据集的完整性"""
        print("Validating dual-view dataset...")

        issues = []

        # 检查基本数据结构
        required_keys = ['ID', 'IM', 'md_p', 'md_true', 'md', 'independent']
        for key in required_keys:
            if key not in dataset_obj.data_set:
                issues.append(f"Missing required key: {key}")

        # 检查双视图特征源
        dual_view_keys = ['d_gs', 'm_ss', 'dis_sem', 'mi_fun']
        for key in dual_view_keys:
            if key not in dataset_obj.data_set:
                issues.append(f"Missing dual-view key: {key}")

        # 检查数据维度
        if 'ID' in dataset_obj.data_set and 'IM' in dataset_obj.data_set:
            id_shape = dataset_obj.data_set['ID'].shape
            im_shape = dataset_obj.data_set['IM'].shape

            print(f"Disease similarity matrix shape: {id_shape}")
            print(f"miRNA similarity matrix shape: {im_shape}")

        # 检查关联矩阵
        if 'md_p' in dataset_obj.data_set:
            md_shape = dataset_obj.data_set['md_p'].shape
            print(f"Association matrix shape: {md_shape}")

        # 检查双视图数据
        if 'd_gs' in dataset_obj.data_set:
            d_gs_shape = dataset_obj.data_set['d_gs'].shape
            print(f"Disease-gene view shape: {d_gs_shape}")

        if 'm_ss' in dataset_obj.data_set:
            m_ss_shape = dataset_obj.data_set['m_ss'].shape
            print(f"miRNA-sequence view shape: {m_ss_shape}")

        if 'dis_sem' in dataset_obj.data_set:
            dis_sem_shape = dataset_obj.data_set['dis_sem'].shape
            print(f"Disease semantic view shape: {dis_sem_shape}")

        if 'mi_fun' in dataset_obj.data_set:
            mi_fun_shape = dataset_obj.data_set['mi_fun'].shape
            print(f"miRNA functional view shape: {mi_fun_shape}")

        # 检查交叉验证数据
        if 'md' in dataset_obj.data_set:
            cv_folds = len(dataset_obj.data_set['md'])
            print(f"Cross-validation folds: {cv_folds}")

        # 报告验证结果
        if issues:
            print("⚠️  Dataset validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Dual-view dataset validation passed!")
            return True

    @staticmethod
    def print_dataset_statistics(dataset_obj):
        """打印双视图数据集统计信息"""
        print("\n" + "=" * 60)
        print("DUAL-VIEW DATASET STATISTICS")
        print("=" * 60)

        if 'md_p' in dataset_obj.data_set:
            md_matrix = dataset_obj.data_set['md_p']
            total_pairs = md_matrix.numel()

            # 统计不同类型的关联
            zero_count = (md_matrix == 0).sum().item()
            pos_count = (md_matrix == 1).sum().item()
            neg_count = (md_matrix == -1).sum().item()
            other_count = (md_matrix == 2).sum().item()

            print(f"Total miRNA-disease pairs: {total_pairs:,}")
            print(f"No association (0): {zero_count:,} ({zero_count / total_pairs * 100:.2f}%)")
            print(f"Up-regulation (1): {pos_count:,} ({pos_count / total_pairs * 100:.2f}%)")
            print(f"Down-regulation (-1): {neg_count:,} ({neg_count / total_pairs * 100:.2f}%)")
            print(f"Other association (2): {other_count:,} ({other_count / total_pairs * 100:.2f}%)")

        print(f"\n🎯 DUAL-VIEW ARCHITECTURE:")
        print(f"miRNA Views: Sequence (m_ss) + Functional (mi_fun)")
        print(f"Disease Views: Gene (d_gs) + Semantic (dis_sem)")

        if 'ID' in dataset_obj.data_set:
            print(f"Number of diseases: {dataset_obj.data_set['ID'].shape[0]}")

        if 'IM' in dataset_obj.data_set:
            print(f"Number of miRNAs: {dataset_obj.data_set['IM'].shape[0]}")

        print(f"Cross-validation folds: {dataset_obj.nums}")

        # 缓存统计
        cache_stats = dataset_obj.get_cache_statistics()
        if cache_stats['cache_hits'] + cache_stats['cache_misses'] > 0:
            print(f"\nCache Statistics:")
            print(f"  Cache hits: {cache_stats['cache_hits']}")
            print(f"  Cache misses: {cache_stats['cache_misses']}")
            print(f"  Hit rate: {cache_stats['hit_rate'] * 100:.1f}%")


# 保持向后兼容性的别名
Dataset = OptimizedDataset

# 导出类和函数
__all__ = ['OptimizedDataset', 'Dataset', 'BatchedDataLoader', 'DatasetValidator']

if __name__ == "__main__":
    print("Dual-view dataset module loaded successfully!")