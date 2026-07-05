# RESULTS — Conformal: Uncertainty-aware miRNA–disease TYPE prediction

> 2026-07-05, branch `breakthrough-conformal`. Hướng đột phá (ứng viên C-uq, `research/novelty_analysis.md`).
> Thay vì chỉ dự đoán 1 type (Top-1), sinh **tập dự đoán (prediction set)** kèm **đảm bảo coverage** —
> phân phối-tự-do, **post-hoc, KHÔNG train lại model**. Chỉ numpy/scipy.

## 1. Ý tưởng

Conformal prediction cho mỗi cặp miRNA–disease một *tập* các type khả dĩ sao cho xác suất chứa type thật
≥ 1−α (ví dụ ≥90%). Tập nhỏ = model tự tin; tập lớn = model bất định. Đây là cách định lượng **độ tin cậy**
trung thực, thứ mà một con số Top-1 F1 đơn lẻ không thể hiện.

## 2. Cơ chế (split-conformal)

- Model xuất per-sample softmax trên các type (`hetero_model.py`, two_head). Ta **dump** per-fold held-out
  (`--dump_scores DIR` → `foldK.npz`: `type_probs`, `true_type`, ...).
- Mỗi CV fold đến từ model M_k khác nhau → chia đôi held-out của *cùng* fold thành calibration/test
  (giữ exchangeability), rồi gộp coverage qua các fold.
- **APS** (Adaptive Prediction Sets): score = khối xác suất luỹ tích tới type thật; ngưỡng τ = quantile (1−α).
- **RAPS**: APS + regularization (cho large-C; ở đây C nhỏ nên không lợi).
- **Mondrian** (class-conditional): τ riêng cho từng type → đảm bảo coverage **per-class**.

Code: `conformal_type_prediction.py` (repo gốc). Bypass `Calculate_Metrics.py` → xử được cả 5-type v3.2.

## 3. Kết quả v2.0 (4-type, model K=2, held-out acc 0.70)

| α | Target | APS coverage | APS set-size /4 |
|---|---:|---:|---:|
| 0.10 | 0.900 | **0.924** ✓ | **2.29** |
| 0.05 | 0.950 | **0.966** ✓ | 2.91 |

- Coverage guarantee giữ. Set-size 2.29/4 = cắt gần nửa không gian nhãn mà vẫn đảm bảo 90%.
- **Negative control (shuffle nhãn)**: set-size phình → 3.57 ≈ C, còn APS chỉ 2.29 → **model info ≈ 1.3 lớp**.
  (Lưu ý: conformal *luôn* đạt coverage; tín hiệu phân biệt nằm ở SET-SIZE, không phải coverage sụp.)
- Per-class coverage đồng đều 0.89–0.94 (kể cả type hiếm Epigenetics n=64).

## 4. Kết quả v3.2 (5-type, model acc 0.30) — "problem → solution"

**Xác nhận độc lập metric-bug:** official Top-1 F1 = **0.0** (Calculate_Metrics bỏ type-5 Tissue) nhưng model
held-out **top-1 acc thật = 0.3024** (khớp Plan K ~0.30).

| Method @90% | Marginal cov | Set-size /5 | Per-class |
|---|---:|---:|---|
| APS | 0.907 ✓ | **4.68** (gần rỗng) | T1–4 ~1.0 **nhưng T5(Tissue) = 0.64** ✗ |
| **Mondrian** | 0.923 ✓ | 4.36 | **mọi class ≥ 0.90; T5 = 0.96** ✓ |

- Set-size 4.68/5 (v3.2) vs 2.29/4 (v2.0) → conformal định lượng trung thực rằng **v3.2 type-prediction trên
  public data là bất định cao**.
- **Marginal APS CHE GIẤU collapse type Tissue** (T5=0.64). **Class-conditional Mondrian phát lộ + phục hồi**
  (T5 0.64→0.96). → Bài học method: *conditional conformal là cần thiết cho MDA đa-type mất cân bằng*.

Biểu đồ: `results/conformal/perclass_and_compare.png` (per-class T5 fix + so v2.0/v3.2),
`results/conformal/coverage_setsize_v2.png`.

## 5. Ý nghĩa cho bài báo

- **Đột phá (bài 2 tiềm năng):** first demonstration conformal/uncertainty cho MDA-type (novelty R7, không thấy prior art).
- **Củng cố critique A:** (i) xác nhận metric-bug lần 4; (ii) model collapse Tissue = biểu hiện cùng gốc với gap v3.2 0.30→0.86.

## 6. Bug đã sửa trong quá trình (ghi lại để minh bạch)

1. **RAPS bất đối xứng**: regularization ban đầu chỉ áp ở prediction, không ở calibration → under-cover (0.76).
   Sửa: reg vào cả `aps_scores`. RAPS λ=0.05 vẫn degenerate ở C nhỏ → **APS là method chính**.
2. **Mondrian randomized-mismatch**: calibration randomized vs prediction deterministic → under-cover (0.50).
   Sửa: calibration dùng `randomize=False` khớp inclusion rule deterministic.

## 7. Giới hạn & mở rộng

- Split-conformal trong-fold hợp lệ; chỉ trên positive pairs (type có điều kiện đã có association).
- Mild over-coverage (~+0.02) = conservatism hữu hạn mẫu, chuẩn.
- Mở rộng: post-hoc temperature calibration so sánh; coverage plots đa-α; positioning conformal-in-biomedicine.

## 8. Files & lệnh

- Script: `conformal_type_prediction.py`. Hook: `--dump_scores` (param.py + main_experiments_hetero1.py, additive).
- Kết quả: `results/conformal/v2_conformal_report.json`, `v32_conformal_report.json`, `*_dump/`, `logs/conformal_*.log`.
- Lệnh: [REPRODUCE.md](REPRODUCE.md) §Conformal.
