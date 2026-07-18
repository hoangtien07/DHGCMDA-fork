# Triển khai Pha 0 (tiên quyết): P7 + P9 + P10

Hiện thực **additive / opt-in** — mặc định TẮT, mọi kết quả Plan A–M giữ nguyên. Smoke-test đã
qua (3 epoch × 2 fold, exit 0). Chi tiết đề xuất: [02-improvement-proposals.md](02-improvement-proposals.md);
căn cứ audit: [01-baseline-audit.md](01-baseline-audit.md).

## Những gì đã thay đổi

| Hạng mục | Cơ chế | File | Mặc định |
|---|---|---|---|
| **P7** leakage-free hoàn chỉnh (F1/F6) | `--leakage_free` giờ còn recompute miRNA GIP (`m_ss`) từ association ĐÃ MASK + assert 0 test-cell rò rỉ | [main_experiments_hetero1.py:932-975](../../main_experiments_hetero1.py#L932) | TẮT (leaky như cũ) |
| **P9a** CV toàn dữ liệu (F3) | `--cv_scheme full` = 5-fold chuẩn, dùng 100% cặp (legacy bỏ ~10%) + independent set disjoint (F8) | [param.py](../../param.py), [prepareData.py:232-360](../../prepareData.py#L232) | `legacy` (như cũ) |
| **P9b** thống kê trung thực (F4/F5/F7) | script mới: bootstrap CI + paired t/Wilcoxon + Holm; parse per-fold từ log | [summarize_stats.py](../../summarize_stats.py) | — |
| **P10** tái lập | manifest provenance/run (git SHA, md5 dataset, flags, versions) + `--deterministic` | [main_experiments_hetero1.py](../../main_experiments_hetero1.py) (`write_run_manifest`, `_apply_determinism`) | manifest luôn ghi; deterministic TẮT |

Manifest ghi tại `results/manifests/manifest_<dataset>_seed<seed>_<ts>.json` mỗi run.

## Đảm bảo additive (đã kiểm chứng)

- `--leakage_free` TẮT → 0 dòng mask, 0 dòng recompute GIP (log `smoke_leaky.log`).
- `--cv_scheme legacy` (mặc định) → giữ nguyên split cũ (cache key có hậu tố scheme nên không đụng cache cũ).
- Chỉ số headline cũ (K=2 0.697…) tái hiện được vì mặc định = hành vi cũ.

## Protocol đo "leakage gap" (chạy khi sẵn sàng — nhiều giờ CPU)

Giữ MỌI thứ cố định, chỉ bật/tắt `--leakage_free`, lặp qua nhiều seed. Ví dụ 3 seed:

```bash
cd /home/hungld/Documents/Tien/DHGCMDA-fork
for S in 0 42 1234; do
  # (a) leaky — baseline hiện tại
  ./venv/bin/python main_experiments_hetero1.py --device cpu \
     --predictor_mode full_bilinear --K_neigs 2 --cv_scheme full --seed $S \
     2>&1 | tee logs/gap_leaky_s$S.log
  # (b) leakage-free HOÀN CHỈNH — cùng seed, cùng mọi thứ
  ./venv/bin/python main_experiments_hetero1.py --device cpu \
     --predictor_mode full_bilinear --K_neigs 2 --cv_scheme full --seed $S \
     --leakage_free 2>&1 | tee logs/gap_leakfree_s$S.log
done

# Gộp per-fold của cả 3 seed cho mỗi điều kiện rồi so sánh có kiểm định
./venv/bin/python summarize_stats.py \
   --logs logs/gap_leaky_s0.log logs/gap_leaky_s42.log logs/gap_leaky_s1234.log \
          logs/gap_leakfree_s0.log logs/gap_leakfree_s42.log logs/gap_leakfree_s1234.log \
   --labels leaky leaky leaky leakage_free leakage_free leakage_free \
   --out results/leakage_gap.json
```

> Lưu ý: `summarize_stats.py` hiện coi mỗi log là một điều kiện; để gộp nhiều seed vào 1 điều
> kiện, dùng chế độ `--json` với schema `{"leaky": {"auc": [...tất cả fold×seed...], "top1_f1":[...]},
> "leakage_free": {...}}`, hoặc chạy so sánh theo từng seed rồi tổng hợp. (Nâng cấp gộp-nhiều-log
> vào-một-nhãn có thể thêm sau nếu cần.)

## KẾT QUẢ ĐO ĐƯỢC (2026-07-13) — `results/leakage_gap.json`

3 seed {0,42,1234} × 5 fold = n=15 mỗi điều kiện. full_bilinear, K=2, `--cv_scheme full`.

| Điều kiện | AUC | Top-1 F1 |
|---|---|---|
| leaky | 0.9875 ± 0.0016 | 0.6969 ± 0.0314 |
| leakage_free | 0.9361 ± 0.0072 | 0.6151 ± 0.0204 |
| **gap (leaky−free)** | **+0.0514** (CI .0479–.0548, t-p=7.8e-14) | **+0.0818** (CI .0662–.0979, t-p=1.45e-7) |

Per-seed (Top-1 F1 leaky → leakfree): s0 0.708→0.626 · s42 0.691→0.603 · s1234 0.692→0.616.
Gap dương ở CẢ 3 seed, có ý nghĩa thống kê (paired-t qua Holm, Wilcoxon p=6.1e-5).

**Kết luận:** baseline cũ thổi phồng do rò rỉ (F1). **Baseline TRUNG THỰC mới = Top-1 F1 0.615 ±
0.020, AUC 0.936 ± 0.007.** Mọi đề xuất P1–P6/P8 phải đo trên nền `--leakage_free --cv_scheme full`.

## Kỳ vọng & tiêu chí chấp nhận (nhắc lại từ P7/P9)

- **Kỳ vọng:** metric leakage-free ≤ leaky (gap > 0) → xác nhận baseline cũ bị thổi phồng; con số
  leakage-free là ước lượng generalize thật, dùng làm baseline MỚI cho P1–P6/P8.
- **Bằng chứng cần:** leakage gap định lượng kèm CI qua ≥3 seed; assert P7 (0 test-cell) pass mọi
  fold (đã tự động trong code); manifest gắn với mỗi số báo cáo.
- **KHÔNG** suy diễn sinh học/lâm sàng từ bất kỳ thay đổi metric nào.

## Chưa làm (nằm trong P9 nhưng cần quyết định thêm)

- Sửa F4 tận gốc (chọn ngưỡng F1 trên train thay vì test) cần truyền train-score vào eval — hiện
  khuyến nghị **báo cáo AUC/AUPR/Top-1 F1** làm chính (đều không tune ngưỡng trên test).
- Tách chạy riêng CVtriplet vs CVtype (F5) — hiện `--cv_scheme full` cải thiện độ phủ nhưng vẫn
  gộp 2 nhóm metric trong 1 vòng; tách hẳn là bước tăng cường tiếp theo.
