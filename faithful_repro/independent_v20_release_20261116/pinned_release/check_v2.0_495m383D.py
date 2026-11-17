#!/usr/bin/env python3
"""
测试新数据格式加载的脚本
验证v2.0_495m383D数据集的加载是否正常
"""

import os
import sys
import torch
import numpy as np
from param import parameter_parser
from prepareData import prepare_data_optimized


def test_data_loading():
    """测试新数据格式的加载"""
    print("🔍 Testing v2.0_495m383D Data Loading")
    print("=" * 60)

    # 获取参数
    args = parameter_parser()

    # 检查数据目录是否存在
    data_dir = os.path.join(args.data_path, 'v2.0_495m383D')
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure v2.0_495m383D folder is in the current directory")
        return False

    print(f"📁 Data directory: {data_dir}")

    # 检查必需文件
    required_files = {
        'D_SSM1.txt': '疾病语义相似性矩阵',
        'D_SSM2.txt': '疾病基因相似性矩阵',
        'M_FSM.txt': 'miRNA功能相似性矩阵',
        'M_GSM.txt': 'miRNA序列相似性矩阵',
        'multi_all_mirna_disease_pairs_without_negative.csv': 'miRNA-疾病关联对'
    }

    print("\n📋 Checking required files...")
    missing_files = []
    for filename, description in required_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✅ {filename} ({size_mb:.2f} MB) - {description}")
        else:
            print(f"  ❌ {filename} - NOT FOUND - {description}")
            missing_files.append(filename)

    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    # 测试数据加载
    print("\n🚀 Starting data loading test...")
    try:
        dataset = prepare_data_optimized(args)

        if dataset is None:
            print("❌ Data loading failed - dataset is None")
            return False

        print("✅ Data loading successful!")

        # 验证数据结构
        print("\n📊 Validating data structure...")

        expected_keys = ['md_p', 'md_true', 'dis_sem', 'mi_fun', 'd_gs', 'm_ss', 'ID', 'IM', 'md', 'independent']
        missing_keys = []

        for key in expected_keys:
            if key in dataset:
                if isinstance(dataset[key], torch.Tensor):
                    print(f"  ✅ {key}: {dataset[key].shape} - {dataset[key].dtype}")
                elif isinstance(dataset[key], list):
                    print(f"  ✅ {key}: list with {len(dataset[key])} elements")
                else:
                    print(f"  ✅ {key}: {type(dataset[key])}")
            else:
                missing_keys.append(key)
                print(f"  ❌ {key}: MISSING")

        if missing_keys:
            print(f"\n⚠️ Warning: Missing {len(missing_keys)} expected keys")

        # 验证数据维度
        print("\n📐 Validating data dimensions...")

        # 检查关联矩阵维度
        if 'md_p' in dataset:
            md_shape = dataset['md_p'].shape
            expected_shape = (495, 383)
            if md_shape == expected_shape:
                print(f"  ✅ Association matrix shape: {md_shape} (matches expected {expected_shape})")
            else:
                print(f"  ⚠️ Association matrix shape: {md_shape} (expected {expected_shape})")

        # 检查相似性矩阵维度
        sim_matrices = {
            'dis_sem': (383, 383),
            'mi_fun': (495, 495),
            'd_gs': (383, 383),
            'm_ss': (495, 495),
            'ID': (383, 383),
            'IM': (495, 495)
        }

        for key, expected_shape in sim_matrices.items():
            if key in dataset:
                actual_shape = dataset[key].shape
                if actual_shape == expected_shape:
                    print(f"  ✅ {key} shape: {actual_shape} (matches expected)")
                else:
                    print(f"  ⚠️ {key} shape: {actual_shape} (expected {expected_shape})")

        # 检查关联统计
        print("\n📈 Association statistics...")
        if 'md_p' in dataset:
            association_matrix = dataset['md_p'].numpy()
            unique_vals, counts = np.unique(association_matrix, return_counts=True)
            total_pairs = association_matrix.size

            print("  Association value distribution:")
            for val, count in zip(unique_vals, counts):
                percentage = count / total_pairs * 100
                print(f"    Value {val}: {count:,} pairs ({percentage:.2f}%)")

            # 检查非零关联数量
            nonzero_count = np.count_nonzero(association_matrix)
            print(f"  Total non-zero associations: {nonzero_count:,}")
            print(f"  Sparsity: {(1 - nonzero_count / total_pairs) * 100:.2f}%")

        # 检查交叉验证数据
        print("\n🔄 Cross-validation data...")
        if 'md' in dataset:
            cv_folds = len(dataset['md'])
            print(f"  ✅ Cross-validation folds: {cv_folds}")

            # 检查第一个fold的数据
            if cv_folds > 0:
                first_fold = dataset['md'][0]
                if 'train' in first_fold and 'test' in first_fold:
                    train_pos = len(first_fold['train'][0]) if len(first_fold['train']) > 0 else 0
                    train_neg = len(first_fold['train'][1]) if len(first_fold['train']) > 1 else 0
                    test_pos = len(first_fold['test'][0]) if len(first_fold['test']) > 0 else 0
                    test_neg = len(first_fold['test'][1]) if len(first_fold['test']) > 1 else 0

                    print(f"  First fold train: {train_pos} positive, {train_neg} negative")
                    print(f"  First fold test: {test_pos} positive, {test_neg} negative")

        print("\n🎉 Data loading test completed successfully!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Data loading test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_view_architecture():
    """测试双视图架构的数据访问"""
    print("\n🎯 Testing Dual-View Architecture Access")
    print("-" * 40)

    try:
        args = parameter_parser()
        dataset = prepare_data_optimized(args)

        if dataset is None:
            print("❌ Cannot test dual-view architecture - dataset loading failed")
            return False

        # 测试四个视图源
        view_sources = {
            'miRNA View 1 (Sequence)': 'm_ss',
            'miRNA View 2 (Function)': 'mi_fun',
            'Disease View 1 (Gene)': 'd_gs',
            'Disease View 2 (Semantic)': 'dis_sem'
        }

        print("📊 Dual-view data sources:")
        for view_name, key in view_sources.items():
            if key in dataset:
                shape = dataset[key].shape
                dtype = dataset[key].dtype
                print(f"  ✅ {view_name}: {shape} ({dtype})")

                # 检查数据范围
                data = dataset[key].numpy()
                min_val, max_val = data.min(), data.max()
                mean_val = data.mean()
                print(f"      Range: [{min_val:.4f}, {max_val:.4f}], Mean: {mean_val:.4f}")
            else:
                print(f"  ❌ {view_name}: MISSING ({key})")

        print("\n🔗 Association matrix for dual-view training:")
        if 'md_p' in dataset:
            md_shape = dataset['md_p'].shape
            print(f"  ✅ Shape: {md_shape}")
            print(f"  Expected: miRNA views ({md_shape[0]}) × Disease views ({md_shape[1]})")

        return True

    except Exception as e:
        print(f"❌ Dual-view architecture test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 v2.0_495m383D Data Loading Test Suite")
    print("=" * 60)

    # 基本数据加载测试
    success1 = test_data_loading()

    # 双视图架构测试
    success2 = test_dual_view_architecture()

    # 总结
    print("\n📋 Test Summary")
    print("-" * 20)
    print(f"Data Loading Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Dual-View Test: {'✅ PASSED' if success2 else '❌ FAILED'}")

    if success1 and success2:
        print("\n🎉 All tests passed! The new data format is ready for training.")
        print("You can now run the main training script with the v2.0_495m383D dataset.")
    else:
        print("\n❌ Some tests failed. Please check the data files and fix any issues.")

    print("\n💡 Next steps:")
    print("1. If tests passed, run: python main_experiments_hetero.py")
    print("2. If tests failed, check data file paths and formats")
    print("3. Ensure all required files are in the v2.0_495m383D folder")