# Experiment State — Reproduce DHGCMDA

> File này ghi lại trạng thái cuối cùng của thí nghiệm reproduce. Cập nhật mỗi khi run xong một experiment.

## Lần cập nhật cuối
2026-05-05, sau khi 5/5 ablation đã hoàn thành.

## Kết quả tóm tắt — HMDD v2.0

### Baseline (full DHGCMDA) vs Paper

Source: [logs/baseline_v2.0_full.log](logs/baseline_v2.0_full.log) → [results/baseline_v2.0_metrics.json](results/baseline_v2.0_metrics.json)

| Metric | Reproduce | Paper | Δ |
|---|---:|---:|---:|
| AUC | 0.9738 | 0.9669 | **+0.71%** |
| AUPR | 0.9671 | 0.9738 | -0.69% |
| F1 | 0.9295 | 0.9278 | +0.18% |
| Accuracy | 0.9266 | — | — |
| Recall | 0.9681 | — | — |
| Specificity | 0.8851 | — | — |
| Precision | 0.8947 | — | — |
| Top-1 Precision | 0.5075 | 0.5842 | -13.13% |
| Top-1 Recall | 0.5979 | 0.6341 | -5.71% |
| Top-1 F1 | 0.5485 | 0.5970 | -8.12% |

Thời gian: 2981.31s (49.7 phút) trên Xeon E5-2680 v4 CPU. Paper báo 15.8 phút trên RTX 4060 Ti 16GB.

**Đánh giá**: Binary metrics (AUC/AUPR/F1) **rất sát paper** (±1%). Top-1 metrics thấp hơn -5 đến -13%, có thể do code-paper discrepancies + CPU non-determinism.

### Kết quả 5 ablations (full reproduce)

| Variant | AUC | AUPR | F1 | Top-1 P | Top-1 R | Top-1 F1 | Δ Top-1 F1 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full DHGCMDA** | 0.9738 | 0.9671 | 0.9295 | 0.5075 | 0.5979 | **0.5485** | — | 2981 |
| w/o CL | 0.9764 | 0.9691 | 0.9334 | 0.6164 | 0.6617 | 0.6381 | **+16.3%** | 1691 |
| w/o HGCN | 0.9758 | 0.9705 | 0.9293 | 0.5910 | 0.6490 | 0.6185 | **+12.8%** | 2632 |
| w/o AVF | 0.9733 | 0.9674 | 0.9290 | 0.5097 | 0.6095 | 0.5549 | +1.2% | 2759 |
| w/o HGT | 0.9703 | 0.9687 | 0.9195 | 0.6127 | 0.6711 | 0.6405 | **+16.8%** | 1910 |
| w/o DV | 0.9736 | 0.9684 | 0.9252 | 0.5403 | 0.6050 | 0.5705 | +4.0% | (n/a) |

**Phát hiện chính**: ngược lại paper Fig. 4 (claim TẤT CẢ ablation hurt performance), thực nghiệm trên fork hiện tại cho thấy **5/5 ablation đều cải thiện hoặc tương đương baseline trên Top-1 F1**. Top-1 F1 cao nhất ở `w/o HGT` (0.6405) và `w/o CL` (0.6381) — vượt xa baseline 0.5485.

Binary metrics (AUC/AUPR/F1) chỉ chênh ±0.01 giữa các variants — saturate trên dataset HMDD v2.0.

### Hội đồng chuyên gia đã consensus về cách present (báo cáo)

3 perspectives đã được mời (Strict reproducer / Skeptical reviewer / Methodology PM). Đồng thuận: chọn **(B) Balanced framing** — present trung thực với caveats rõ ràng:
- Đặt phát hiện trong section 3.4 với 5 subsection: Bối cảnh → Số liệu → Đối chiếu paper → Phân tích nguyên nhân → Hệ quả/Giới hạn.
- Dùng wording "quan sát/ghi nhận" thay vì "chứng minh/bác bỏ".
- Liệt kê 5 nguyên nhân khả dĩ (xếp theo xác suất): (i) Cơ chế ablation không tương đương paper; (ii) Code-paper discrepancies; (iii) Loss formulation khác paper; (iv) CPU vs GPU non-determinism; (v) Single-seed limitation.
- Khuyến nghị thực nghiệm bổ sung: sửa 4 discrepancies + multi-seed + GPU run.

## Cách tiếp tục lần sau

### Re-run từ đầu
```powershell
cd d:\Tien\DHGCMDA-fork
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = 1

# 1. Baseline (~50 phút)
python main_experiments_hetero1.py --device cpu *>&1 | Tee-Object logs\baseline_v2.0_full.log
python parse_metrics.py logs\baseline_v2.0_full.log results\baseline_v2.0_metrics.json

# 2. Ablations (~28-50 phút mỗi cái → 2.5-3 giờ total)
.\run_ablations.ps1

# 3. Compile + sinh báo cáo
.\compile_final.ps1
```

### Investigate phát hiện bất thường (nếu user muốn theo đuổi)
Ưu tiên thấp → cao theo độ rủi ro:
1. **Sửa `λ₃_recon` từ 0.15 → 1.0** ([main_experiments_hetero1.py:871](main_experiments_hetero1.py#L871)) → train lại baseline → so sánh.
2. **Sửa `n_head` từ 8 → 4** ([param.py:35](param.py#L35)) → train lại → so sánh.
3. Multi-seed: chạy `--seed 42`, `--seed 100`, `--seed 2024` để đo variance.
4. Liên hệ CDMBlab CDMBlab/DHGCMDA team trên GitHub Issues, share kết quả này, xin seed/config tái lập Fig. 4.

### Extension ideas (Phần 4 báo cáo)
Xem chi tiết trong `BaoCao_DHGCMDA.docx`:
- Cold-start evaluation (hide miRNA subset)
- Cross-dataset transfer (v2.0 train, v3.2 mới test)
- Replace HGT bằng NodeFormer/GraphGPS
- Áp dụng cho lncRNA-disease, circRNA-disease

## Files cần backup nếu di chuyển workspace

- `requirements.txt`, `CLAUDE.md`, `EXPERIMENT_STATE.md`
- `logs/baseline_v2.0_full.log`, `logs/ablation_*.log`
- `results/*.json`, `results/*.png`
- `BaoCao_DHGCMDA.docx`
- Tất cả source `.py` đã modify (param.py, hetero_model.py, generate_report.py, parse_metrics.py, generate_arch_figure.py)
- Plan: `C:\Users\hungld\.claude\plans\download-code-v-data-async-lovelace.md`
