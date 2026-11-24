"""Produce the thesis results outline from completed runs and their gate evidence."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from clean_protocol import DATASETS, RUNS
from prepare_v32 import DATASET
from v32_runs import run_name


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def row(label, metrics, names):
    return "| " + " | ".join((label, *[f"{metrics[key]:.4f}" for key in names])) + " |"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=RUNS / "THESIS_RESULTS.md")
    options = parser.parse_args()
    device = options.device
    calibration = read(RUNS / "calibration_lock.json")
    decision = read(RUNS / "g3.json")
    rescored = read(RUNS / f"rescored-{device}.json")
    controls = read(RUNS / f"controls-{device}.json")
    improvements = read(RUNS / f"improvements-{device}.json")
    lines = [
        "# Kết quả luận văn DHGCMDA", "",
        "## 1. Phương pháp và dữ liệu", "",
        "Tái chạy mã phát hành DHGCMDA trên HMDD v2.0 và port năm loại trên MDAv3.2-3. "
        "Giao thức đánh giá có kiểm soát dùng cùng các cặp fit, validation và test cho mọi phương pháp.", "",
        "## 2. Tái hiện Bảng 3 và Bảng 4", "",
        f"Mã phát hành `{calibration['release']}`. Kết luận G3 `{decision['status']}`.", "",
        "| Run | AUC | AUPR | F1 nhị phân | Precision | Recall |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    expected = [f"c{n}-{device}" for n in range(1, 4)]
    expected.extend([run_name(profile, 1234, device) for profile in ("K_star", "T_star")])
    optional = sorted(set(rescored) - set(expected))
    for name in expected + optional:
        metrics = rescored[name]["binary_author_evaluation"]
        lines.append(row(name, metrics, ("auc", "aupr", "f1", "precision", "recall")))
    lines.extend(["", "| Đối chiếu | Sai lệch có dấu |", "| --- | --- |"])
    for label, metrics in (("v2.0 Bảng 4", calibration["g2"]["delta"]),
                           ("v2.0 Bảng 3", calibration["c2_table3_delta"]),
                           ("v3.2 Bảng 4", decision["table4_K_star"]["delta"]),
                           ("v3.2 Bảng 3", decision["table3_T_star"]["delta"])):
        delta = ", ".join(f"{key} {value:+.4f}" for key, value in metrics.items())
        lines.append(f"| {label} | {delta} |")

    lines.extend(["", "Chênh lệch C3 − C1 khi đổi seed, tính trên cùng cách chia fold của mã phát hành.", "",
                  "| Chỉ số | C3 − C1 |", "| --- | ---: |"])
    for metrics in (calibration["c3_minus_c1_binary"], calibration["c3_minus_c1_top1"]):
        lines.extend(f"| {key} | {value:+.4f} |" for key, value in metrics.items())
    if decision["status"] == "matched":
        profile = decision["closer_profile"]
        seed_metrics = [rescored[run_name(profile, seed, device)]["binary_author_evaluation"]
                        for seed in (1234, 42, 2025)]
        lines.extend(["", f"Dao động qua ba seed của cấu hình `{profile}` trên v3.2.", "",
                      "| Chỉ số | Trung bình | SD |", "| --- | ---: | ---: |"])
        for key in ("auc", "aupr", "f1", "precision", "recall"):
            values = [entry[key] for entry in seed_metrics]
            lines.append(f"| {key} | {mean(values):.4f} | {stdev(values):.4f} |")

    lines.extend(["", "## 3. Đánh giá lại và phân rã rò rỉ", "",
                  "| Run | Top-1 F1 TDRC | Top-1 mã gốc hoặc mở rộng | Macro-F1 đa nhãn |",
                  "| --- | ---: | ---: | ---: |"])
    for name in expected + optional:
        result = rescored[name]
        metrics = {**result["mean"]}
        metrics.setdefault("top1_f1", result["top1_author_evaluation"]["top1_f1"])
        lines.append(row(name, metrics,
                         ("tdrc_top1_f1", "top1_f1", "multilabel_macro_f1_uniform_cutoff")))
    lines.extend(["", "Macro-F1 trên các dự đoán tác giả dùng ngưỡng cố định 1/K "
                  "cho mỗi loại. Phép chấm Top-1 mở rộng áp dụng cho v3.2."])
    for dataset in DATASETS:
        lines.extend(["", f"### {dataset}", "",
                      "| Phương pháp | Top-1 F1 TDRC | SD | Chênh lệch với legacy |",
                      "| --- | ---: | ---: | ---: |"])
        results = controls[dataset]
        for method, stats in results["scores"].items():
            delta = results["controlled_difference_vs_legacy"].get(method)
            difference = f"{delta['mean']:+.4f}" if delta else "—"
            lines.append(f"| {method} | {stats['mean']:.4f} | {stats['std_ddof_1']:.4f} | {difference} |")

    lines.extend(["", "## 4. Thí nghiệm cải tiến", "",
                  "| Dữ liệu | Top-1 F1 TDRC trung bình | SD | Δ với DHGCMDA sạch | Δ với baseline chọn trên validation | DHGCMDA gốc theo giao thức tác giả |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        result = improvements[dataset]
        paired = result["paired_differences"]
        reference_run = run_name("K_star", 1234, device) if dataset == DATASET else f"c1-{device}"
        reference = rescored[reference_run]["mean"]["tdrc_top1_f1"]
        lines.append(f"| {dataset} | {result['selected']['mean']:.4f} | {result['selected']['std_ddof_1']:.4f} "
                     f"| {paired['clean']['mean']:+.4f} | {paired['strongest_baseline']['mean']:+.4f} | {reference:.4f} |")
    lines.extend(["", "Cột giao thức tác giả dùng cùng quy ước F1 TDRC để đối chiếu. "
                  "So sánh ghép cặp dùng các kết quả trên tập test chung của giao thức sạch.", "",
                  "| Dữ liệu | Macro-F1 đa nhãn | AUC | AUPR | Wilcoxon p so với DHGCMDA sạch | p kiểm định dấu |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        result = improvements[dataset]
        secondary = result["secondary"]["selected"]
        paired = result["paired_differences"]["clean"]
        lines.append(f"| {dataset} | {secondary['multilabel_macro_f1']['mean']:.4f} | "
                     f"{secondary['binary_auc']['mean']:.4f} | {secondary['binary_aupr']['mean']:.4f} | "
                     f"{paired['wilcoxon_p_descriptive']:.4f} | {paired['sign_p_descriptive']:.4f} |")
    if improvements["negative_result_on_v32"]:
        lines.extend(["", "Mô hình được chọn trên validation chưa vượt cả prior và logistic trên v3.2. "
                      "Trọng tâm kết luận là tái hiện và đánh giá lại."])
    lines.extend(["", "## 5. Hạn chế", "",
                  "- Đặc trưng GIP trong tái hiện dùng toàn bộ ma trận liên kết. Kết quả theo giao thức tác giả cần được diễn giải theo yếu tố này.",
                  "- View bệnh thứ hai v3.2 dùng bản sao D_MESH theo dữ liệu hiện có.",
                  "- Fold lặp qua seed dùng chung dữ liệu. Kiểm định ghép cặp mô tả kết quả trên benchmark này."])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(options.output)


if __name__ == "__main__":
    main()
