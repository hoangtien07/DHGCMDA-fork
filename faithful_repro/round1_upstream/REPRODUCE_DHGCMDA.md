# Tái hiện DHGCMDA (BMC Bioinformatics 2026) bằng code gốc của tác giả

**Kết luận: tái hiện được phần binary (CV_triplet) trên HMDD v2.0; KHÔNG tái hiện được Top-1 (CV_type).** Top-1 F1 đo được thấp hơn số trong paper khoảng 4 độ lệch chuẩn, và không có run nào trong 11 run chạm tới 0.5970. Các phần còn lại của paper (v3.2, Fig.3 K-sweep, trục t của Fig.2, ablation Fig.4) không chạy được nếu không sửa code: tác giả không phát hành dữ liệu v3.2, và các flag tương ứng không được nối vào code.

## 1. Thiết lập (đã đo/kiểm trên máy)
- Code: `github.com/CDMBlab/DHGCMDA` commit `06f9ba3` (repo chỉ có 1 commit). **Không sửa dòng nào.**
- Môi trường: Python 3.12, torch 2.5.1+cpu (cùng phiên bản PyTorch paper ghi), torch-geometric 2.6.1, numpy 1.26.4, pandas 2.2.3, scipy 1.13.1, scikit-learn 1.5.2. Chạy CPU (8 vCPU, 4 run song song, mỗi run 2 thread). Mỗi fold mất ~6–7 phút.
- Bắt buộc dùng **numpy < 2**. Code gọi `np.mat` (trong `hypergraph_construct_KNN.py:8`), hàm này đã bị xoá ở numpy 2.x nên chạy với numpy 2.4.3 bị crash. README không ghi chặn phiên bản.
- Protocol: mặc định của tác giả, 650 epoch × 5-fold, dataset `v2.0_495m383D`, lệnh `python main_experiments_hetero1.py --device cpu`. Số liệu lấy từ khối `FINAL COMPREHENSIVE RESULTS` trong log.
- Seed: code hardcode `seed_torch()` = 1234, flag `--seed` không có tác dụng. Để đo phương sai theo seed, tôi dùng wrapper `seed_wrapper.py`: nó chạy nguyên script gốc qua `runpy` và chỉ đổi lời gọi seed 1234 → seed mới. Split CV dùng `np.random.seed(0)` riêng (`prepareData.py:271`), nên giữ nguyên ở mọi run. Phương sai đo được vì vậy chỉ là phương sai của init/training, không gồm phương sai của split.
- Determinism: 2 run mặc định chạy độc lập cho kết quả trùng tới 4 chữ số ở 2/2 fold đã so (AUC 0.9771/0.9736, Top-1 F1 0.5050/0.5734). Trên CPU, cùng seed và cùng số thread thì kết quả lặp lại được.

## 2. Kết quả chính: HMDD v2.0 (Table 3 & Table 4 của paper)

| Metric | Paper | Code tác giả, mặc định, mean±sd (n=4 seed) | Δ | Kết luận |
|---|---:|---:|---:|---|
| AUC (CV_triplet) | 0.9669 | **0.9749 ± 0.0011** | +0.0080 | đạt/vượt |
| AUPR (CV_triplet) | 0.9738 | **0.9682 ± 0.0014** | −0.0056 | sát (−0.6%) |
| F1 (CV_triplet) | 0.9278 | **0.9332 ± 0.0023** | +0.0054 | đạt |
| Top-1 Precision (CV_type) | 0.5842 | **0.5156 ± 0.0106** | −0.0686 | **không đạt (−11.7%)** |
| Top-1 Recall (CV_type) | 0.6341 | **0.6063 ± 0.0084** | −0.0278 | không đạt (−4.4%) |
| Top-1 F1 (CV_type) | 0.5970 | **0.5570 ± 0.0095** | −0.0400 | **không đạt (−6.7%, ≈4.2 sd)** |

Chi tiết từng run (seed 1234 là mặc định của tác giả):

| Run | AUC | AUPR | F1 | Top-1 P | Top-1 R | Top-1 F1 |
|---|---:|---:|---:|---:|---:|---:|
| default seed 1234 | 0.9735 | 0.9665 | 0.9299 | 0.5119 | 0.6029 | 0.5533 |
| default seed 1 | 0.9754 | 0.9684 | 0.9349 | 0.5276 | 0.6125 | 0.5664 |
| default seed 2 | 0.9761 | 0.9698 | 0.9339 | 0.5030 | 0.5960 | 0.5454 |
| default seed 3 | 0.9747 | 0.9679 | 0.9343 | 0.5201 | 0.6138 | 0.5630 |
| `--n_head 4` (giá trị trong paper), seed 1234 | 0.9755 | 0.9698 | 0.9322 | 0.5187 | 0.6117 | 0.5611 |

Top-1 F1 theo từng fold ở run mặc định (seed 1234): 0.5050 / 0.5734 / 0.5844 / 0.5360 / 0.5677. Độ chênh giữa các fold lớn (~0.08).

### Nhận xét
1. **Binary đạt.** AUC/F1 vượt paper, AUPR thấp hơn 0.006.
   - Có một điểm lạ: AUC đo được (0.975) xấp xỉ AUPR của paper (0.974), còn AUPR đo được (0.968) xấp xỉ AUC của paper (0.967). Tôi đã kiểm tra code: `Calculate_Metrics.py` trả `[auc, aupr, ...]`, trong đó auc tính từ ROC và aupr tính từ đường PR, nên code gán nhãn đúng. [suy luận] Có thể Table 4 của paper bị đảo cột AUPR/AUC ở dòng v2.0. Tôi chưa xác minh được; cần tác giả xác nhận.
2. **Top-1 không đạt và không phải do nhiễu seed.** Số tốt nhất trong 11 run là 0.5732 (λ2=0.1, seed 1234), vẫn dưới 0.5970. Dùng `--n_head 4` theo paper (code mặc định là 8) cũng chỉ được 0.5611 (1 seed), không đóng được gap.
   - [suy luận] Nguyên nhân có thể nằm ở chỗ code tác giả phát hành không phải bản đã sinh ra số trong paper. Các bằng chứng gián tiếp:
     - README của repo tự ghi "Expected Performance" là AUC ~0.95, AUPR ~0.89, F1(macro) ~0.62, khác hẳn cả paper lẫn số đo được.
     - Loss trong code (`exist_weight=0.3/type_weight=0.7`, focal γ=2.0, label smoothing 0.1, class weight theo effective-number β=0.99999, ở `main_experiments_hetero1.py:59-79`) không khớp với Eq. 32/34 của paper (inverse-proportional weighting).
3. **Paper không tự nhất quán ở Top-1 F1.** Harmonic mean của P/R paper báo (0.5842/0.6341) là 0.6081, không phải 0.5970. Ngược lại, ở số đo của tôi, HM(0.5119, 0.6029) = 0.5537, khớp với 0.5533 mà code in ra. [suy luận] Có thể paper lấy trung bình theo cách khác, hoặc lấy P/R và F1 từ các run khác nhau.
4. Theo Table 3 của chính paper, DHGCMDA (0.5970) **không phải phương pháp tốt nhất** trên v2.0 Top-1 F1: TFLP đạt 0.5996 và MRFGMDA đạt 0.5979. Paper có thừa nhận điều này ("top-tier").

## 3. Fig. 2: độ nhạy theo λ2 (`--inter_view_weight`, flag này có tác dụng thật)

| λ2 | n seed | Top-1 F1 mean ± sd | Từng seed (1234/1/2/3) |
|---|---:|---:|---|
| 0.1 | 4 | **0.5654 ± 0.0081** | 0.5732 / 0.5654 / 0.5543 / 0.5688 |
| 0.3 (paper chọn, mặc định) | 4 | 0.5570 ± 0.0095 | 0.5533 / 0.5664 / 0.5454 / 0.5630 |
| 0.5 | 2 | 0.5529 ± 0.0095 | 0.5461 / 0.5596 / – / – (so với 0.3 cùng seed: −0.0072, −0.0068) |

- Paper kết luận "Top-1 đạt max tại λ2=0.3, λ2 lớn hơn thì giảm". Kết quả đo được:
  - Phần "λ2=0.5 giảm" khớp với chiều của paper.
  - Phần "0.3 là tối ưu" **không được xác nhận**: λ2=0.1 cao hơn 0.3 ở 3/4 cặp cùng seed, trung bình +0.0084 (≈ 1 sd).
  - Với n=4, dữ liệu chưa đủ để khẳng định 0.1 tốt hơn 0.3; chỉ nói được là 0.3 không vượt trội.
- Trục t (temperature) **không test được**: `--cl_temperature`/`--inter_view_temperature` không được dùng trong forward/loss. Grep chỉ thấy chúng ở `param.py`.

## 4. Những gì KHÔNG tái hiện được nếu không sửa code (kiểm bằng grep, không cần chạy)

| Phần paper | Lý do |
|---|---|
| Toàn bộ kết quả HMDD v3.2 (Table 3/4 cột v3.2, P/R/F1 0.79/0.94/0.86) | Repo chỉ có `v2.0_495m383D`; dữ liệu v3.2 (411×271) không được phát hành |
| Fig. 3: sweep K ∈ {1..15} | `--K_neigs` là flag chết, K hardcode `[13]` ở `main_experiments_hetero1.py:704-751` |
| Fig. 2: trục t | `cl_temperature` không được dùng |
| Fig. 2(b): so sánh chiến lược class weight | Weight hardcode (effective-number), `--class_weights` chỉ được in ra ở `param.py` |
| Fig. 4: ablation (w/o CL/HGCN/AVF/HGT/DV) | Code không có cơ chế ablation nào |
| Learning rate / weight decay | Optimizer hardcode `lr=1e-4, wd=1e-5` (`main_experiments_hetero1.py:1374`); `--lr` và `--weight_decay` không có tác dụng |
| Multi-seed | `--seed` không có tác dụng (seed cố định 1234) |

## 5. Tái lập
```bash
uv venv -p 3.12 venv_np1
VIRTUAL_ENV=venv_np1 uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=venv_np1 uv pip install numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.5.2 torch-geometric==2.6.1 openpyxl xlrd
cd DHGCMDA && PYTHONUTF8=1 ../venv_np1/bin/python main_experiments_hetero1.py --device cpu            # default
DHG_SEED=1 PYTHONUTF8=1 ../venv_np1/bin/python ../seed_wrapper.py --device cpu --inter_view_weight 0.1 # seed/λ2
```
Log đầy đủ của 12 run nằm trong `logs/`, số liệu tổng hợp trong `summary.json`.
