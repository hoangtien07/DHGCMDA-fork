import pandas as pd
import numpy as np
import os
from collections import Counter


class DataDistributionAnalyzer:
    """分析数据分布并生成正确的权重配置"""

    def __init__(self, data_path='./v2.0_495m383D'):
        self.data_path = data_path
        self.association_file = 'multi_all_mirna_disease_pairs_without_negative.csv'
        # 4种生物学关联类型的名称
        self.type_names = {
            1: "循环类型 (Circulation)",
            2: "表观遗传学 (Epigenetics)",
            3: "靶标类型 (Target)",
            4: "遗传学类型 (Genetics)"
        }

    def analyze_and_generate_weights(self):
        """分析数据并生成推荐权重"""
        print("=" * 80)
        print("🔍 数据分布诊断工具 - 自动计算正确权重")
        print("=" * 80)

        # 1. 加载关联数据
        file_path = os.path.join(self.data_path, self.association_file)
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return None

        data = pd.read_csv(file_path, header=None)
        print(f"✅ 成功加载数据: {data.shape[0]} 条关联记录\n")

        # 2. 提取类型信息
        association_types = data.iloc[:, 2].values
        total_associations = len(association_types)

        # 3. 统计类型分布
        type_counter = Counter(association_types)

        print(f"📈 实际数据分布统计:")
        print(f"{'类型':<30} {'数量':<10} {'比例':<10} {'推荐权重':<15}")
        print("-" * 80)

        type_stats = {}
        for type_id in sorted(type_counter.keys()):
            count = type_counter[type_id]
            percentage = count / total_associations * 100

            weight = total_associations / (len(type_counter) * count)

            type_name = self.type_names.get(type_id, f"未知类型 {type_id}")
            print(f"{type_name:<30} {count:<10} {percentage:>6.2f}%   {weight:>6.3f}")

            type_stats[type_id] = {
                'count': count,
                'percentage': percentage,
                'weight': weight,
                'prior_prob': percentage / 100  # 先验概率 = 实际比例
            }

        # 4. 生成推荐配置
        print(f"\n" + "=" * 80)
        print(f"🎯 推荐的模型配置 (直接复制到代码中):")
        print("=" * 80)

        # 损失函数类别权重 (按类型1,2,3,4顺序)
        class_weights = [type_stats[i]['weight'] for i in range(1, 5)]

        print(f"\n1️⃣ 【最重要】损失函数类别权重:")
        print(f"   📍 位置: main_experiments_hetero.py 第70-76行")
        print(f"   📝 将以下代码替换现有的 self.register_buffer('class_weights', ...)")
        print(f"\n   self.register_buffer('class_weights',")
        print(f"                        torch.tensor({[round(w, 3) for w in class_weights]},")
        print(f"                                     device=self.device))")

        # 先验概率 (按类型1,2,3,4顺序)
        prior_probs = [type_stats[i]['prior_prob'] for i in range(1, 5)]

        print(f"\n2️⃣ 贝叶斯先验概率:")
        print(f"   📍 位置: hetero_model.py 第451-453行")
        print(f"   📝 将以下代码替换现有的 self.register_buffer('log_class_prior', ...)")
        print(f"\n   self.register_buffer('log_class_prior',")
        print(f"                        torch.log(torch.tensor({[round(p, 4) for p in prior_probs]})))")

        # 同步更新分布跟踪
        print(f"\n3️⃣ 分布跟踪 (同步更新):")
        print(f"   📍 位置: hetero_model.py 第675-677行")
        print(f"   📝 将以下代码替换现有的 self.register_buffer('type_distribution', ...)")
        print(f"\n   self.register_buffer('type_distribution',")
        print(f"                        torch.tensor({[round(p, 4) for p in prior_probs]}))")

        # 5. 对比当前配置
        print(f"\n" + "=" * 80)
        print(f"⚠️  当前代码中的配置 vs 推荐配置对比:")
        print("=" * 80)

        current_weights = [1.02, 2.38, 1.28, 0.55]
        current_priors = [0.219, 0.094, 0.175, 0.407]

        print(f"\n{'配置项':<20} {'当前值':<40} {'推荐值':<40}")
        print("-" * 100)

        weights_str_current = str([round(w, 3) for w in current_weights])
        weights_str_recommended = str([round(w, 3) for w in class_weights])
        print(f"{'类别权重':<20} {weights_str_current:<40} {weights_str_recommended:<40}")

        priors_str_current = str([round(p, 3) for p in current_priors])
        priors_str_recommended = str([round(p, 3) for p in prior_probs])
        print(f"{'先验概率':<20} {priors_str_current:<40} {priors_str_recommended:<40}")

        # 6. 计算配置差异
        weight_diff = np.abs(np.array(current_weights) - np.array(class_weights))
        prior_diff = np.abs(np.array(current_priors) - np.array(prior_probs))

        print(f"\n📉 配置差异分析:")
        print(f"   权重平均差异: {weight_diff.mean():.4f} (最大: {weight_diff.max():.4f})")
        print(f"   先验平均差异: {prior_diff.mean():.4f} (最大: {prior_diff.max():.4f})")

        if weight_diff.mean() > 0.3 or prior_diff.mean() > 0.1:
            print(f"   🚨 差异较大,强烈建议更新配置!")
            print(f"   💡 预期改进: Top-1 F1 从 0.31 提升到 0.60+")
        else:
            print(f"   ✅ 差异较小,当前配置基本合理")



        return {
            'class_weights': class_weights,
            'prior_probs': prior_probs,
            'type_stats': type_stats
        }


def main():
    """主函数"""
    print("\n🚀 启动权重诊断工具...\n")

    # 检查数据路径
    data_paths = [
        './v2.0_495m383D',
        '../v2.0_495m383D',
        'v2.0_495m383D'
    ]

    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            print(f"✅ 找到数据目录: {path}\n")
            break

    if data_path is None:
        print("❌ 未找到数据目录 v2.0_495m383D")
        print("📁 请确保数据文件夹在以下位置之一:")
        for path in data_paths:
            print(f"  - {os.path.abspath(path)}")
        return

    # 创建分析器并执行分析
    analyzer = DataDistributionAnalyzer(data_path)
    result = analyzer.analyze_and_generate_weights()

    if result:
        print("\n✅ 诊断完成!")
    else:
        print("\n❌ 诊断失败,请检查数据文件")


if __name__ == "__main__":
    main()