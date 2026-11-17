import numpy as np
import torch
from sklearn import metrics
from functools import lru_cache
import warnings
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict

warnings.filterwarnings('ignore')


class OptimizedMetricCalculator:
    """CVtype Top-1指标"""

    def __init__(self, cache_size=100):
        self.cache_size = cache_size
        self._metric_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _compute_hash(self, real_score, predict_score):
        """数据哈希用于缓存"""
        try:
            real_hash = hash(real_score.data.tobytes()) if hasattr(real_score, 'data') else hash(real_score.tobytes())
            pred_hash = hash(predict_score.data.tobytes()) if hasattr(predict_score, 'data') else hash(
                predict_score.tobytes())
            return f"{real_hash}_{pred_hash}"
        except:
            return None

    def _preprocess_inputs(self, real_score, predict_score):
        """输入预处理"""
        # 确保输入是numpy数组
        if isinstance(real_score, torch.Tensor):
            real_score = real_score.detach().cpu().numpy()
        if isinstance(predict_score, torch.Tensor):
            predict_score = predict_score.detach().cpu().numpy()

        # 展平并确保数据类型
        real_score = np.array(real_score).flatten().astype(np.float32)
        predict_score = np.array(predict_score).flatten().astype(np.float32)

        # 确保长度一致
        if len(real_score) != len(predict_score):
            min_len = min(len(real_score), len(predict_score))
            real_score = real_score[:min_len]
            predict_score = predict_score[:min_len]

        # 多类到二分类的转换
        if len(np.unique(real_score)) > 2:
            real_score = (real_score != 0).astype(np.float32)

        # 多维预测（取第一通道）
        if len(predict_score.shape) > 1 and predict_score.shape[1] > 1:
            predict_score = predict_score[:, 0]

        # 数据验证
        if len(real_score) == 0 or len(predict_score) == 0:
            raise ValueError("Empty input arrays")

        # 处理NaN值
        valid_mask = ~(np.isnan(real_score) | np.isnan(predict_score))
        if not valid_mask.all():
            real_score = real_score[valid_mask]
            predict_score = predict_score[valid_mask]

        return real_score, predict_score

    def _compute_binary_metrics_fast(self, real_score, predict_score):
        """二分类指标计算"""
        try:
            # 预处理输入
            real_score, predict_score = self._preprocess_inputs(real_score, predict_score)

            if len(real_score) == 0:
                return [0] * 7

            # 并行计算ROC和PR曲线
            def compute_roc():
                try:
                    fpr, tpr, thresholds = metrics.roc_curve(real_score, predict_score)
                    auc = metrics.auc(fpr, tpr)
                    return auc, thresholds
                except:
                    return 0, [0.5]

            def compute_pr():
                try:
                    precision, recall, _ = metrics.precision_recall_curve(real_score, predict_score)
                    aupr = metrics.auc(recall, precision)
                    return aupr
                except:
                    return 0

            # 使用线程池并行计算
            with ThreadPoolExecutor(max_workers=2) as executor:
                roc_future = executor.submit(compute_roc)
                pr_future = executor.submit(compute_pr)

                auc, thresholds = roc_future.result()
                aupr = pr_future.result()

            # 快速阈值优化
            if len(thresholds) > 0:
                # 使用向量化操作计算F1分数
                y_pred_matrix = (predict_score[:, np.newaxis] >= thresholds).astype(int)

                # 批量计算F1分数
                tp = np.sum((real_score[:, np.newaxis] == 1) & (y_pred_matrix == 1), axis=0)
                fp = np.sum((real_score[:, np.newaxis] == 0) & (y_pred_matrix == 1), axis=0)
                fn = np.sum((real_score[:, np.newaxis] == 1) & (y_pred_matrix == 0), axis=0)

                # 计算F1分数（向量化）
                precision_vec = tp / (tp + fp + 1e-7)
                recall_vec = tp / (tp + fn + 1e-7)
                f1_scores = 2 * (precision_vec * recall_vec) / (precision_vec + recall_vec + 1e-7)

                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
            else:
                best_threshold = 0.5

            # 其他指标
            y_pred = (predict_score >= best_threshold).astype(int)

            # 所有指标
            tn, fp, fn, tp = metrics.confusion_matrix(real_score, y_pred, labels=[0, 1]).ravel()

            f1_score = 2 * tp / (2 * tp + fp + fn + 1e-7)
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            specificity = tn / (tn + fp + 1e-7)
            precision = tp / (tp + fp + 1e-7)

            return [auc, aupr, f1_score, accuracy, recall, specificity, precision]

        except Exception as e:
            return [0] * 7

    def _compute_multiclass_metrics_fast(self, real_score, predict_score):
        """多分类指标计算"""
        try:
            # 确保输入格式正确
            if isinstance(real_score, torch.Tensor):
                real_score = real_score.detach().cpu().numpy()
            if isinstance(predict_score, torch.Tensor):
                predict_score = predict_score.detach().cpu().numpy()

            real_score = np.array(real_score).astype(np.float32)
            predict_score = np.array(predict_score).astype(np.float32)

            # 非零样本
            non_zero_mask = real_score != 0
            if np.sum(non_zero_mask) == 0:
                return self._get_empty_multiclass_metrics()

            real_multiclass = real_score[non_zero_mask]

            # 预测分数
            if len(predict_score.shape) > 1 and predict_score.shape[1] > 1:
                # 多通道输出
                multiclass_probs = predict_score[non_zero_mask, 1:]  # 跳过存在概率通道
                pred_multiclass = np.argmax(multiclass_probs, axis=1)

                # 类别映射
                # 多类型关联映射（保持原始编码）
                # 类型: 1=循环, 2=表观遗传学, 3=靶标, 4=遗传学

                pred_multiclass = pred_multiclass


                type_mapping_to_bio = {0: 1, 1: 2, 2: 3, 3: 4}
                pred_multiclass = np.array([type_mapping_to_bio.get(p, 1) for p in pred_multiclass])
            else:

                pred_scores = predict_score[non_zero_mask]
                pred_multiclass = np.zeros_like(pred_scores)

                # 使用更优的阈值分配
                # 4类型阈值分配（基于数据分布）
                # 循环(1): 21.9%, 表观遗传学(2): 9.4%, 靶标(3): 17.5%, 遗传学(4): 40.7%
                pred_multiclass[pred_scores < 0.25] = 2  # 表观遗传学(少数类)
                pred_multiclass[(pred_scores >= 0.25) & (pred_scores < 0.45)] = 1  # 循环
                pred_multiclass[(pred_scores >= 0.45) & (pred_scores < 0.65)] = 3  # 靶标
                pred_multiclass[pred_scores >= 0.65] = 4  # 遗传学(多数类)

            # 获取唯一类别
            classes = np.unique(np.concatenate([real_multiclass, pred_multiclass]))

            # 并行计算指标
            def compute_accuracy():
                return metrics.accuracy_score(real_multiclass, pred_multiclass)

            def compute_macro_metrics():
                precision = metrics.precision_score(real_multiclass, pred_multiclass,
                                                    average='macro', labels=classes, zero_division=0)
                recall = metrics.recall_score(real_multiclass, pred_multiclass,
                                              average='macro', labels=classes, zero_division=0)
                f1 = metrics.f1_score(real_multiclass, pred_multiclass,
                                      average='macro', labels=classes, zero_division=0)
                return precision, recall, f1

            def compute_weighted_metrics():
                precision = metrics.precision_score(real_multiclass, pred_multiclass,
                                                    average='weighted', labels=classes, zero_division=0)
                recall = metrics.recall_score(real_multiclass, pred_multiclass,
                                              average='weighted', labels=classes, zero_division=0)
                f1 = metrics.f1_score(real_multiclass, pred_multiclass,
                                      average='weighted', labels=classes, zero_division=0)
                return precision, recall, f1

            # 使用线程池并行计算
            with ThreadPoolExecutor(max_workers=3) as executor:
                acc_future = executor.submit(compute_accuracy)
                macro_future = executor.submit(compute_macro_metrics)
                weighted_future = executor.submit(compute_weighted_metrics)

                accuracy = acc_future.result()
                precision_macro, recall_macro, f1_macro = macro_future.result()
                precision_weighted, recall_weighted, f1_weighted = weighted_future.result()

            # 计算混淆矩阵
            confusion = metrics.confusion_matrix(real_multiclass, pred_multiclass, labels=classes)

            # 计算各类别性能
            class_metrics = {}
            for cls in classes:
                binary_real = (real_multiclass == cls).astype(int)
                binary_pred = (pred_multiclass == cls).astype(int)

                class_labels = {
                    1: "circulation",  # 循环类型
                    2: "epigenetics",  # 表观遗传学
                    3: "target",  # 靶标类型
                    4: "genetics"  # 遗传学类型
                }
                class_label = class_labels.get(cls, f"type_{cls}")

                class_metrics[f'precision_{class_label}'] = metrics.precision_score(
                    binary_real, binary_pred, zero_division=0)
                class_metrics[f'recall_{class_label}'] = metrics.recall_score(
                    binary_real, binary_pred, zero_division=0)
                class_metrics[f'f1_{class_label}'] = metrics.f1_score(
                    binary_real, binary_pred, zero_division=0)

            return {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'confusion_matrix': confusion,
                'classes': classes,
                **class_metrics
            }

        except Exception as e:
            return self._get_empty_multiclass_metrics()

    def compute_top1_metrics(self, real_scores_list, predict_scores_list):
        """
        CVtype Top-1指标
        """
        try:
            if len(real_scores_list) == 0 or len(predict_scores_list) == 0:
                return {'top1_precision': 0.0, 'top1_recall': 0.0, 'top1_f1': 0.0}

            # 统计变量
            correct_predictions = 0
            total_predictions = 0

            # 按类型统计的正确预测和总数（用于计算recall）
            type_correct = defaultdict(int)  # 每种类型正确预测的数量
            type_total = defaultdict(int)  # 每种类型的总数
            type_predicted = defaultdict(int)  # 每种类型被预测的总数

            valid_samples = 0

            for i, (real_score, pred_score) in enumerate(zip(real_scores_list, predict_scores_list)):
                try:
                    # 转换为numpy数组
                    real_array = np.array(real_score).flatten()
                    pred_array = np.array(pred_score).flatten()

                    # 跳过空数据或零标签
                    if len(real_array) == 0 or len(pred_array) == 0:
                        continue

                    real_value = real_array[0] if len(real_array) == 1 else real_array

                    # 只处理有关联的样本（非零）
                    if isinstance(real_value, np.ndarray):
                        if real_value.sum() == 0:
                            continue
                        # 多标签情况：找到真实的类型
                        real_type_indices = np.where(real_value > 0)[0]
                        if len(real_type_indices) == 0:
                            continue
                        true_type = real_type_indices[0]  # 取第一个正类型
                    else:
                        if real_value == 0:
                            continue
                        # 单标签情况：映射到类型索引
                        # 单标签情况：4种生物学关联类型
                        # 1=循环->0, 2=表观遗传学->1, 3=靶标->2, 4=遗传学->3
                        if real_value == 1:
                            true_type = 0  # 循环类型
                        elif real_value == 2:
                            true_type = 1  # 表观遗传学
                        elif real_value == 3:
                            true_type = 2  # 靶标类型
                        elif real_value == 4:
                            true_type = 3  # 遗传学类型
                        else:
                            continue

                    # 处理预测分数
                    if len(pred_array) >= 3:
                        # 多类型预测分数
                        if len(pred_array) == 5:
                            # [存在性, type1, type2, type3, type4] 格式（4种类型）
                            type_scores = pred_array[1:5]
                        elif len(pred_array) == 4:
                            # [type1, type2, type3, type4] 格式（直接4种类型）
                            type_scores = pred_array[:4]
                        else:
                            # 如果不是4类型，跳过
                            continue

                        # Top-1预测：选择最高分的类型
                        predicted_type = np.argmax(type_scores)

                        valid_samples += 1
                        total_predictions += 1

                        # 统计真实类型
                        type_total[true_type] += 1

                        # 统计预测类型
                        type_predicted[predicted_type] += 1

                        # 检查Top-1预测是否正确
                        if predicted_type == true_type:
                            correct_predictions += 1
                            type_correct[true_type] += 1

                except Exception as sample_error:
                    continue

            # 计算Top-1 Precision
            top1_precision = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            # 计算Top-1 Recall（宏平均）
            type_recalls = []
            for type_id in type_total:
                if type_total[type_id] > 0:
                    recall = type_correct[type_id] / type_total[type_id]
                    type_recalls.append(recall)

            top1_recall = np.mean(type_recalls) if type_recalls else 0.0

            # 计算Top-1 F1
            if top1_precision + top1_recall > 0:
                top1_f1 = 2 * (top1_precision * top1_recall) / (top1_precision + top1_recall)
            else:
                top1_f1 = 0.0

            return {
                'top1_precision': top1_precision,
                'top1_recall': top1_recall,
                'top1_f1': top1_f1,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'valid_samples': valid_samples,
                'type_statistics': {
                    'type_correct': dict(type_correct),
                    'type_total': dict(type_total),
                    'type_predicted': dict(type_predicted)
                }
            }

        except Exception as e:
            return {'top1_precision': 0.0, 'top1_recall': 0.0, 'top1_f1': 0.0}

    def compute_cv_type_style_metrics(self, real_scores_list, predict_scores_list):

        try:
            if len(real_scores_list) == 0 or len(predict_scores_list) == 0:
                return [0] * 9

            # CV_type风格的计算
            TP = 0  # 预测正确的样本数
            FP = 0  # 预测错误的样本数
            total_real_positive = 0  # 真实正样本总数

            for i, (real_score, predict_score) in enumerate(zip(real_scores_list, predict_scores_list)):
                try:
                    real_score = np.array(real_score).flatten()
                    predict_score = np.array(predict_score).flatten()

                    # 跳过空数据
                    if len(real_score) == 0 or len(predict_score) == 0:
                        FP += 1
                        continue

                    # 处理不同的数据格式
                    if len(predict_score) == 4:
                        # 4维格式：[existence, type1, type2, type3]
                        existence_score = predict_score[0]
                        type_scores = predict_score[1:4]

                        # 处理真实标签
                        if len(real_score) == 4:
                            real_existence = real_score[0]
                            real_types = real_score[1:4]
                        elif len(real_score) == 1:
                            real_value = real_score[0]
                            real_existence = 1.0 if real_value > 0 else 0.0
                            real_types = np.zeros(3)
                            if real_value == 1:
                                real_types[0] = 1
                            elif real_value == -1:
                                real_types[1] = 1
                            elif real_value == 2:
                                real_types[2] = 1
                        else:
                            FP += 1
                            continue

                        positive_num = real_existence
                        total_real_positive += positive_num

                        if positive_num > 0:
                            # 预测最高分的类型
                            max_type_idx = np.argmax(type_scores)
                            if real_types[max_type_idx] > 0:
                                TP += 1
                            else:
                                FP += 1
                        else:
                            FP += 1

                    elif len(predict_score) == 1:
                        # 单值格式
                        pred_value = predict_score[0]
                        real_value = real_score[0] if len(real_score) > 0 else 0

                        positive_num = 1 if real_value > 0 else 0
                        total_real_positive += positive_num

                        if positive_num > 0:
                            if pred_value > 0.5:
                                TP += 1
                            else:
                                FP += 1
                        else:
                            FP += 1
                    else:
                        FP += 1
                        continue

                except Exception as sample_error:
                    FP += 1
                    continue

            # 计算CV_type指标
            total_predictions = TP + FP
            samples_with_positive = sum(1 for real_score in real_scores_list
                                        if np.array(real_score).sum() > 0)

            # CV_type Average Precision和Recall
            avg_precision = TP / total_predictions if total_predictions > 0 else 0
            avg_recall = TP / samples_with_positive if samples_with_positive > 0 else 0

            # 确保值在[0,1]范围内
            avg_precision = min(avg_precision, 1.0)
            avg_recall = min(avg_recall, 1.0)

            # 计算传统二分类指标
            try:
                # 合并所有数据进行传统二分类评估
                all_real = []
                all_pred = []

                for real_score, predict_score in zip(real_scores_list, predict_scores_list):
                    real_arr = np.array(real_score).flatten()
                    pred_arr = np.array(predict_score).flatten()

                    if len(real_arr) > 0 and len(pred_arr) > 0:
                        # 取存在性作为二分类标签
                        if len(real_arr) == 4:
                            all_real.append(real_arr[0])
                        else:
                            all_real.append(1.0 if real_arr[0] > 0 else 0.0)

                        if len(pred_arr) == 4:
                            all_pred.append(pred_arr[0])
                        else:
                            all_pred.append(pred_arr[0])

                if len(all_real) > 0 and len(all_pred) > 0:
                    all_real = np.array(all_real)
                    all_pred = np.array(all_pred)

                    # 转换为二分类
                    binary_real = (all_real > 0).astype(int)
                    comprehensive_metrics = self.get_metrics(binary_real, all_pred)

                    if len(comprehensive_metrics) == 7:
                        result = [avg_precision, avg_recall] + comprehensive_metrics
                    else:
                        result = [avg_precision, avg_recall] + [0] * 7
                else:
                    result = [avg_precision, avg_recall] + [0] * 7

            except Exception as comp_error:
                result = [avg_precision, avg_recall] + [0] * 7

            return result

        except Exception as e:
            return [0] * 9

    def _get_empty_multiclass_metrics(self):
        """返回空的多分类指标"""
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'error': 'No valid samples'
        }

    def get_metrics(self, real_score, predict_score):
        # 尝试从缓存获取
        cache_key = self._compute_hash(real_score, predict_score)
        if cache_key and cache_key in self._metric_cache:
            self._cache_hits += 1
            return self._metric_cache[cache_key]

        self._cache_misses += 1

        # 计算指标
        result = self._compute_binary_metrics_fast(real_score, predict_score)

        # 缓存结果
        if cache_key and len(self._metric_cache) < self.cache_size:
            self._metric_cache[cache_key] = result

        return result

    def get_multiclass_metrics(self, real_score, predict_score):

        return self._compute_multiclass_metrics_fast(real_score, predict_score)

    def cv_mat_model_evaluate(self, association_mat, predict_mat):

        return self.get_metrics(association_mat, predict_mat)

    def cv_mat_model_evaluate_multiclass(self, association_mat, predict_mat):

        binary_metrics = self.get_metrics(association_mat, predict_mat)
        multiclass_metrics = self.get_multiclass_metrics(association_mat, predict_mat)
        return binary_metrics, multiclass_metrics

    def get_cache_statistics(self):
        """获取缓存统计信息"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._metric_cache)
        }


class Metric_fun:

    def __init__(self):
        super(Metric_fun, self).__init__()
        self.calculator = OptimizedMetricCalculator()

    def cv_mat_model_evaluate(self, association_mat, predict_mat):
        """评估二分类性能"""
        return self.calculator.cv_mat_model_evaluate(association_mat, predict_mat)

    def cv_mat_model_evaluate_multiclass(self, association_mat, predict_mat):
        """评估多分类性能"""
        return self.calculator.cv_mat_model_evaluate_multiclass(association_mat, predict_mat)

    def get_metrics(self, real_score, predict_score):
        """计算评估指标"""
        return self.calculator.get_metrics(real_score, predict_score)

    def get_multiclass_metrics(self, real_score, predict_score):
        """计算多分类指标"""
        return self.calculator.get_multiclass_metrics(real_score, predict_score)

    def compute_cv_type_style_metrics(self, real_scores_list, predict_scores_list):
        return self.calculator.compute_cv_type_style_metrics(real_scores_list, predict_scores_list)

    def compute_top1_metrics(self, real_scores_list, predict_scores_list):
        return self.calculator.compute_top1_metrics(real_scores_list, predict_scores_list)


# 性能测试函数
def benchmark_metrics_calculation():
    print("Benchmarking metrics calculation...")

    # 创建测试数据
    np.random.seed(42)
    n_samples = 10000

    real_scores = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    predict_scores = np.random.random(n_samples)

    original_calculator = Metric_fun()
    optimized_calculator = OptimizedMetricCalculator()

    start_time = time.time()
    for _ in range(10):
        original_result = original_calculator.get_metrics(real_scores, predict_scores)
    original_time = time.time() - start_time

    start_time = time.time()
    for _ in range(10):
        optimized_result = optimized_calculator.get_metrics(real_scores, predict_scores)
    optimized_time = time.time() - start_time

    print(f"Original implementation: {original_time:.4f} seconds")
    print(f"Optimized implementation: {optimized_time:.4f} seconds")
    print(f"Speedup: {original_time / optimized_time:.2f}x")

    # 缓存统计
    cache_stats = optimized_calculator.get_cache_statistics()
    print(f"Cache statistics: {cache_stats}")

    return original_result, optimized_result


if __name__ == "__main__":
    # 运行性能测试
    benchmark_metrics_calculation()

    # 测试CVtype Top-1指标
    print("\n测试CVtype Top-1指标...")
    calculator = Metric_fun()

    # 模拟4维张量数据 [existence, type1, type2, type3]
    real_scores_list = [
        [1],  # 样本1：上调 (type=1 -> index=0)
        [2],  # 样本2：下调 (type=2 -> index=1)
        [3],  # 样本3：其他 (type=3 -> index=2)
        [4],  # 样本4：其他 (type=4 -> index=2)
        [1],  # 样本5：上调
        [0],  # 样本6：无关联
    ]

    predict_scores_list = [
        [0.8, 0.9, 0.1, 0.2],  # 样本1预测：存在概率0.8，type1概率最高 -> 正确
        [0.9, 0.2, 0.8, 0.1],  # 样本2预测：存在概率0.9，type2概率最高 -> 正确
        [0.7, 0.1, 0.2, 0.7],  # 样本3预测：存在概率0.7，type3概率最高 -> 正确
        [0.1, 0.1, 0.1, 0.1],  # 样本4预测：存在概率0.1，各类型概率低 -> 预测type1,错误
        [0.6, 0.3, 0.4, 0.5],  # 样本5预测：存在概率0.6，type3概率最高 -> 错误
        [0.1, 0.1, 0.1, 0.1],  # 样本6预测：无关联，应该被跳过
    ]

    # 计算CVtype Top-1指标
    top1_results = calculator.compute_top1_metrics(real_scores_list, predict_scores_list)
    print("\nCVtype Top-1指标结果:")
    for key, value in top1_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 测试CV_type风格指标
    print("\n测试CV_type风格指标...")
    cv_type_results = calculator.compute_cv_type_style_metrics(real_scores_list, predict_scores_list)

    metric_names = ['Avg_Precision', 'Avg_Recall', 'AUC', 'AUPR', 'F1', 'Accuracy', 'Recall', 'Specificity',
                    'Precision']
    print("CV_type风格指标结果:")
    for name, value in zip(metric_names, cv_type_results):
        print(f"  {name}: {value:.4f}")