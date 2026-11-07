# Tái hiện DHGCMDA: bản cài đặt theo đúng paper (vòng 2)

## Kết luận

**Không tái hiện được.** Đã sửa code cho đúng thuật toán của paper, bỏ rò rỉ dữ liệu test, và chạy đúng 2 giao thức CV_type và CV_triplet. Trên cả 3 bộ dữ liệu, không metric nào đạt mức paper báo.

| Claim của paper | Paper | Đo được (faithful, 3 seed × 5 fold, mean±sd giữa các seed) | Kết luận |
|---|---:|---:|---|
| v2.0 CV_type Top-1 P/R/F1 | 0.5842 / 0.6341 / 0.5970 | 0.4010±0.0112 / 0.3788±0.0236 / **0.3878±0.0151** | Không tái hiện được (F1 −0.209) |
| v2.0 CV_triplet AUPR/AUC/F1 | 0.9738 / 0.9669 / 0.9278 | 0.9218±0.0041 / 0.9237±0.0031 / 0.8600±0.0031 | Không tái hiện được (−0.04 đến −0.07) |
| v3.2 (cuilab, lọc đúng như paper ghi) CV_type | 0.7915 / 0.9421 / 0.8600 | 0.2916±0.0232 / 0.2341±0.0062 / **0.2567±0.0101** | Không tái hiện được |
| v3.2 (cuilab) CV_triplet | 0.9271 / 0.9181 / 0.8674 | 0.8964±0.0030 / 0.9025±0.0026 / 0.8322±0.0025 | Không tái hiện được (−0.016 đến −0.035) |
| v3.2 (MDAv3.2-3, bản 411×271 paper dùng) CV_type | như trên | 0.3130±0.0224 / 0.2486±0.0084 / **0.2727±0.0111** | Không tái hiện được |
| v3.2 (MDAv3.2-3) CV_triplet | như trên | 0.8701±0.0026 / 0.8798±0.0015 / 0.8100±0.0014 | Không tái hiện được |
| Bộ v3.2 có 411×271×11,748 khi tải từ cuilab và lọc theo quy tắc paper ghi | 411×271×11,748 | 721×609×17,607 | Không tái hiện được |
| Fig.4: bỏ bất kỳ module nào cũng làm kết quả giảm | mọi biến thể đều < full | So cùng seed 1234: bỏ HGT (−0.036) và bỏ HGCN (−0.012) làm giảm, đúng chiều paper; bỏ AVF (+0.067), bỏ DV (+0.011), bỏ CL (+0.005) lại làm tăng (1 seed, bảng ở mục 5) | Bị phản bác ở AVF; HGT đúng chiều; HGCN, CL, DV chưa đủ bằng chứng |
| Fig.2 / Fig.3 (sweep λ, t, K) | — | Vòng này chưa chạy | Chưa đánh giá |

**Hai yếu tố lệch so với paper đã đo được (chạy chẩn đoán 1 seed, KHÔNG phải kết quả chính):**
- **Thang đo của recon loss.** Eq.21 viết theo đúng chữ là ‖S−Ŝ‖_F², tức tổng trên khoảng 495² phần tử. Loss này lớn cỡ 4,000–28,000 và lấn át L_type: type loss chỉ giảm từ 1.27 xuống 1.19 sau 300 epoch.
  - Nếu thay tổng bằng trung bình (cùng thang với MSE trong code tác giả), Top-1 F1 trên v2.0 tăng từ 0.388 lên 0.541. Con số này vẫn thấp hơn paper (0.597) và thấp hơn code gốc của tác giả (0.557).
  - Tôi không dùng biến thể này làm kết quả chính, vì paper ghi rõ là ‖·‖_F².
- **Rò rỉ dữ liệu test.** Bật lại rò rỉ như code tác giả (`--diag_leak 1`) thì Top-1 F1 trên v2.0 tăng từ 0.388 lên 0.458, và AUPR CV_triplet tăng từ 0.922 lên 0.943. Như vậy một phần điểm cao của code gốc đến từ việc dữ liệu test lọt vào feature và graph [suy luận, dựa trên số đo này].

**Kết quả code gốc của tác giả (vòng 1) không được tính là tái hiện.** Code gốc cho AUC 0.9749 và Top-1 F1 0.5570, nhưng có rò rỉ dữ liệu test và dùng giao thức không phải CV_triplet. Vì vậy mức khớp phần binary ở vòng 1 không phải là tái hiện hợp lệ claim CV_triplet của paper.

## 1. Nguồn và môi trường

- Code tác giả: CDMBlab/DHGCMDA commit `06f9ba3eed92e334d9a2b8003fc06d83596e5614` (repo chỉ có 1 commit).
- Bản cài đặt theo paper: `code/faithful.py`. Đây là một file riêng viết lại. Mỗi chỗ khác code tác giả đều ghi trong dict `DEVIATIONS` ở đầu file và được lưu vào từng file JSON kết quả.
- Môi trường: CPU (8 vCPU), Python 3.12.16, torch 2.5.1+cpu, torch-geometric 2.6.1, numpy 1.26.4, scikit-learn 1.5.2, scipy 1.13.1, pandas 2.2.3. Mỗi job chạy 1 thread.
- Test: `code/test_faithful.py` pass toàn bộ. Test kiểm tra:
  - các tập split rời nhau;
  - negative trong CV_triplet đều là triplet chưa biết, và có đúng số lượng bằng positive;
  - giữ đúng 161 cặp multi-type của v2.0;
  - metric Top-1 cho P/R/F1 trùng `Calculate_Metrics.compute_top1_metrics` của tác giả trên dữ liệu 4 loại (0.2768 = 0.2768);
  - với 5 loại, một predictor hoàn hảo cho F1 = 1.0.

## 2. Dữ liệu

### 2.1 HMDD v3.2, tải từ http://www.cuilab.cn/hmdd (theo yêu cầu)

Các file lấy về từ `/static/hmdd3/data/` (SHA256):

| File | SHA256 |
|---|---|
| v3_genetics.txt | f1994890f80dda0035daf62895c5cae71139e27660cbc4faf4de507ff68370b2 |
| v3_epigenetics.txt | f50e361bd62f456921339543db94c880e557c2790eafeac5b47f1aa3a70d5e39 |
| v3_circulation.txt | 46e572b2e9e0706cc6f0378bfcc77107dc9efdf0d564952563891814f61fbc2c |
| v3_target.txt | ce773c927e12649d55963ba0e546cd83e1718bc62b3fb41a8956dab66d576176 |
| v3_tissue.txt | 6221fe38d075c7fda241ccaca69c5ce7cb06386bc8fd69511fead4bb97a0b8f4 |
| v3_alldata.txt | dcc871c880a2351428d994752cd57caa5438b951cd42a37e58e63fda420c033e |
| v3_other.txt (không dùng: paper chỉ có 5 loại) | 48c9d99377d77491ec16845b3e2fe7bfdf7c18a579d6becdc7e6e5c79b3690c9 |
| MeSH 2025 d2025.bin (NLM) | 82a6d0f14f86afc5a4cdb14bf3732e4420637090dfa3f1916b1a6c1415a06291 |
| MISIM 2.0 similarity.zip (lirmed.com/misim) | 6b5da04cb58799ed8a18c0edd27e9581ecae2278a6f89e7a14ea2f0dc97759da |

**Cách xử lý (`code/build_datasets.py`):**
- Lấy 5 file theo loại; tên miRNA viết thường; khử trùng lặp các triplet (miRNA, bệnh, loại).
- Lọc theo đúng câu của paper: "exclude miRNAs and diseases with fewer than two experimentally verified associations". Tôi đếm bậc theo cặp (miRNA, bệnh) gộp qua 5 loại và chỉ lọc 1 lượt.
- Kết quả: **721 miRNA × 609 bệnh × 17,607 triplet** (13,744 cặp). Số triplet theo loại: genetics 1818, epigenetics 575, circulation 3215, target 6049, tissue 5950.
- Dữ liệu thô: 1049 × 758 × 18,084.
- Các cách hiểu khác của bộ lọc (đếm theo cặp hay theo triplet; lọc 1 lượt hay lặp đến khi ổn định) đều cho 720–722 × 606–612 × 17,603–17,615.
- **So với paper (411 × 271 × 11,748): KHÔNG khớp.** Không có cách hiểu nào của bộ lọc mà paper ghi ra được số của paper. Paper đã dùng thêm bước curation không mô tả, và tôi không bịa ra bước đó.

**Similarity (paper không ghi nguồn):**
- Bệnh, view 2: SS1 (Wang 2010). Bệnh, view 1: SS2 (Xuan 2013). Cả hai tính từ MeSH 2025 qua mã UI ở cột `mesh` của HMDD; 496/609 bệnh có MeSH, bệnh không có MeSH thì dùng hàng đơn vị.
- miRNA, view 2: functional similarity từ MISIM 2.0 (720/721 miRNA có trong MISIM).
- miRNA, view 1: GIP, tính lại theo từng fold chỉ từ A_train. Tôi dùng GIP vì với v2.0, file `M_GSM.txt` của tác giả chính là GIP của ma trận association đầy đủ 495×383 (sai khác tối đa 4.6e-16). Tôi giữ đúng loại similarity tác giả dùng, chỉ bỏ phần rò rỉ dữ liệu test.
- [giả định] Paper ghi tên view là "sequence" và "gene", nhưng code tác giả không dùng dữ liệu sequence hay gene nào cả.

### 2.2 v3.2 đối chứng: MDAv3.2-3 (không tải từ cuilab)

- `HMDD_data/MDAv3.2-3` trong fork (commit eaa2500) được khôi phục từ snapshot Software Heritage của repo SPLDHyperAWNTF.
- Kích thước 411×271×11,748, số theo loại 1155/403/2293/3997/3900, **khớp đúng Table 2 của paper**.
- Tuy vậy, bậc nhỏ nhất của miRNA trong bản này là 1, tức trái với quy tắc "<2 bị loại" mà paper ghi.
- Bản này chỉ có 2 similarity (miRNA functional, disease semantic), nên view 1 của cả miRNA và bệnh đều dùng GIP tính theo từng fold [giả định].
- Bản này chỉ dùng làm đối chứng: theo dấu vết thì đây là bộ dữ liệu paper thực sự dùng, nhưng nó không phải bản tải từ cuilab.

### 2.3 HMDD v2.0

- Dùng file `multi_all_mirna_disease_pairs_without_negative.csv` của repo: 1,679 triplet; số theo loại 443/199/356/681 khớp Table 2; có 1,498 cặp, trong đó 161 cặp có nhiều loại.
- Ma trận 495×383. Bậc nhỏ nhất bằng 0 vì chỉ 324 miRNA và 174 bệnh có association có loại. Như vậy quy tắc "<2 bị loại" của paper cũng không khớp với chính v2.0 mà paper dùng; tôi giữ nguyên 495×383 như paper.

## 3. Đối chiếu paper và code tác giả: những gì đã sửa

**Nguyên tắc:** chỗ nào paper ghi rõ thì làm theo paper. Chỗ nào paper không ghi thì giữ nguyên hành vi code tác giả và gắn nhãn [giả định]. Không có tham số nào được tune.

| # | Nội dung | Paper | Code tác giả | faithful.py |
|---|---|---|---|---|
| 1 | Rò rỉ dữ liệu test | CV: các cặp test phải chưa biết | Ma trận `association_matrix` đầy đủ (gồm cặp test) đi vào feature `[S,A]`, hypergraph, cạnh HGT; `M_GSM` là GIP tính trên toàn bộ ma trận | Chỉ dùng A_train; GIP tính lại theo từng fold |
| 2 | Hai view (Eq.1–2) | Nhánh v nhận X^(v) | Cả 2 nhánh nhận `concat_mi_tensor` (feature view 1) | Mỗi nhánh nhận view của mình |
| 3 | KNN (Sec 2.3) | Cosine, K=13 | Euclidean (`Eu_dis`), K=13 hardcode | Cosine, `--K` mặc định 13 |
| 4 | AVF (Eq.16–18) | Gate α_v và β_v học được | `0.6*z1 + 0.4*z2` cố định | Gate SE theo kênh + softmax β [giả định: hệ số reduction 4] |
| 5 | Reconstruction (Eq.19–21) | MLP 2 lớp, Ŝ=ZZᵀ, ‖S−Ŝ‖_F² | Decoder cosine, 0.15×MSE | Đúng như paper, λ3=1 |
| 6 | Cập nhật graph động (Sec 2.8) | Mỗi 5 epoch, ngưỡng 0.5, dựng lại hypergraph | Mỗi epoch khi MSE thay đổi > 0.01, không dựng lại hypergraph | Đúng như paper (Ŝ được min-max về [0,1] trước khi so ngưỡng [giả định]) |
| 7 | Hàm loss (Eq.32) | L_type + 1·L_intra + 0.3·L_inter + 1·L_recon | recover + cl_mi + cl_dis + 0.15·recon + 1e-4·L2, cộng label smoothing 0.1 | Đúng Eq.32, cộng thêm existence loss 0.3 (paper dự đoán existence nhưng không định nghĩa loss cho nó, nên giữ của tác giả [giả định]) |
| 8 | Multi-label (Eq.33) | Σ_k w_k y_k log p_k | Mỗi cặp chỉ giữ 1 nhãn; 161/1498 cặp v2.0 bị ghi đè nhãn | Tensor Y[m,d,C] multi-hot |
| 9 | Class weight (Eq.34) | Effective number, tính trên tập train | `counts=[367,157,293,681]` hardcode | n_k lấy từ train của từng fold |
| 10 | Số loại association | 4 (v2.0) / 5 (v3.2) | `num_types=4` hardcode; metric loại bỏ type 5 | C đọc từ data |
| 11 | HGT (Sec 2.7) | 2 lớp, 4 head, residual + LayerNorm | 8 head mặc định | 4 head, x = LN(x + Dropout(HGT(x))) |
| 12 | Optimizer | Adam, có early stopping | AdamW, không early stopping, chạy cố định 650 epoch | Adam lr 1e-4 (giữ của tác giả); early stopping trên 10% validation tách từ tập train, đánh giá mỗi 10 epoch, patience 10 [giả định] |
| 13 | Giao thức CV (Sec 3.2) | CV_triplet và CV_type là 2 giao thức riêng | 1 lần split theo cặp | Split riêng cho từng giao thức; CV_triplet có số triplet chưa biết bằng số positive |
| 14 | Lỗi bị nuốt | — | `forward()` có `except Exception` trả về điểm hằng | Bỏ |

**Giữ nguyên của tác giả vì paper không ghi rõ:**
- Inter-CL lấy toàn bộ cặp không liên kết làm negative; margin m = 0.5, trọng số margin 0.1.
- Predictor bilinear dạng đường chéo, kèm projector và temperature (paper chỉ ghi "bilinear").
- Existence loss dạng focal, negative 10:1, trọng số negative 0.5.
- Gradient clipping 1.0; hidden 256; dropout 0.3.
- Điểm của một triplet bằng p_exist(m,d) × p_type(m,d,k) [giả định].
- Metric dùng đúng định nghĩa của tác giả: AUPR = auc(recall, precision); F1 lấy ở ngưỡng cho F1 lớn nhất; Top-1 R là trung bình recall theo từng loại.

## 4. Toàn bộ kết quả (mỗi dòng: mean±sd giữa các seed của trung bình 5 fold; dòng tiếp theo: kết quả từng fold)

Cột metric: CV_type = P / R / F1 của Top-1; CV_triplet = AUPR / AUC / F1. Dòng recon=mean và dòng LEAK chỉ là chẩn đoán.

| data | protocol | ablation | recon | leak | seeds | metric1 | metric2 | metric3 |
|---|---|---|---|---|---|---|---|---|
| v2.0 | triplet | none | mean | - | [1234] | 0.9332±0.0000 (paper 0.9738, Δ -0.0406) | 0.9340±0.0000 (paper 0.9669, Δ -0.0329) | 0.8697±0.0000 (paper 0.9278, Δ -0.0581) |
|  |  |  |  |  | seed 1234 folds | 0.9327 / 0.9321 / 0.9272 / 0.9374 / 0.9364 | 0.9319 / 0.9432 / 0.9279 / 0.9318 / 0.9349 | 0.8683 / 0.8918 / 0.8543 / 0.8685 / 0.8656 |
| v2.0 | triplet | none | sum | - | [1, 1234, 2] | 0.9218±0.0041 (paper 0.9738, Δ -0.0520) | 0.9237±0.0031 (paper 0.9669, Δ -0.0432) | 0.8600±0.0031 (paper 0.9278, Δ -0.0678) |
|  |  |  |  |  | seed 1 folds | 0.9351 / 0.9153 / 0.9327 / 0.9254 / 0.9262 | 0.9352 / 0.9173 / 0.9292 / 0.9250 / 0.9307 | 0.8754 / 0.8584 / 0.8645 / 0.8580 / 0.8627 |
|  |  |  |  |  | seed 1234 folds | 0.9239 / 0.9272 / 0.9206 / 0.9081 / 0.9278 | 0.9280 / 0.9322 / 0.9184 / 0.9188 / 0.9219 | 0.8724 / 0.8701 / 0.8517 / 0.8600 / 0.8449 |
|  |  |  |  |  | seed 2 folds | 0.9221 / 0.9043 / 0.9199 / 0.9347 / 0.9041 | 0.9171 / 0.9119 / 0.9277 / 0.9335 / 0.9089 | 0.8502 / 0.8546 / 0.8618 / 0.8693 / 0.8455 |
| v2.0 | triplet | none | sum | LEAK | [1234] | 0.9433±0.0000 (paper 0.9738, Δ -0.0305) | 0.9515±0.0000 (paper 0.9669, Δ -0.0154) | 0.8949±0.0000 (paper 0.9278, Δ -0.0329) |
|  |  |  |  |  | seed 1234 folds | 0.9401 / 0.9531 / 0.9432 / 0.9210 / 0.9592 | 0.9497 / 0.9592 / 0.9439 / 0.9421 / 0.9629 | 0.8820 / 0.9034 / 0.8818 / 0.8994 / 0.9078 |
| v2.0 | type | no_avf | sum | - | [1234] | 0.4899±0.0000 (paper 0.5842, Δ -0.0943) | 0.4628±0.0000 (paper 0.6341, Δ -0.1713) | 0.4759±0.0000 (paper 0.5970, Δ -0.1211) |
|  |  |  |  |  | seed 1234 folds | 0.4833 / 0.5133 / 0.5333 / 0.4214 / 0.4983 | 0.4614 / 0.4903 / 0.4880 / 0.4159 / 0.4583 | 0.4721 / 0.5016 / 0.5097 / 0.4186 / 0.4775 |
| v2.0 | type | no_cl | sum | - | [1234] | 0.4132±0.0000 (paper 0.5842, Δ -0.1710) | 0.4154±0.0000 (paper 0.6341, Δ -0.2187) | 0.4141±0.0000 (paper 0.5970, Δ -0.1829) |
|  |  |  |  |  | seed 1234 folds | 0.3367 / 0.4433 / 0.5033 / 0.4114 / 0.3712 | 0.3416 / 0.4558 / 0.4764 / 0.4061 / 0.3969 | 0.3391 / 0.4495 / 0.4895 / 0.4087 / 0.3836 |
| v2.0 | type | no_dv | sum | - | [1234] | 0.4226±0.0000 (paper 0.5842, Δ -0.1616) | 0.4182±0.0000 (paper 0.6341, Δ -0.2159) | 0.4202±0.0000 (paper 0.5970, Δ -0.1768) |
|  |  |  |  |  | seed 1234 folds | 0.4000 / 0.4633 / 0.3933 / 0.4181 / 0.4381 | 0.3851 / 0.4732 / 0.3659 / 0.4258 / 0.4410 | 0.3924 / 0.4682 / 0.3791 / 0.4219 / 0.4395 |
| v2.0 | type | no_hgcn | sum | - | [1234] | 0.4066±0.0000 (paper 0.5842, Δ -0.1776) | 0.3876±0.0000 (paper 0.6341, Δ -0.2465) | 0.3968±0.0000 (paper 0.5970, Δ -0.2002) |
|  |  |  |  |  | seed 1234 folds | 0.3200 / 0.4267 / 0.3933 / 0.3980 / 0.4950 | 0.3041 / 0.4103 / 0.3852 / 0.3789 / 0.4594 | 0.3119 / 0.4183 / 0.3892 / 0.3882 / 0.4766 |
| v2.0 | type | no_hgt | sum | - | [1234] | 0.3745±0.0000 (paper 0.5842, Δ -0.2097) | 0.3716±0.0000 (paper 0.6341, Δ -0.2625) | 0.3728±0.0000 (paper 0.5970, Δ -0.2242) |
|  |  |  |  |  | seed 1234 folds | 0.3333 / 0.3433 / 0.4333 / 0.3813 / 0.3813 | 0.3201 / 0.3208 / 0.4277 / 0.3796 / 0.4098 | 0.3266 / 0.3317 / 0.4305 / 0.3804 / 0.3950 |
| v2.0 | type | none | mean | - | [1234] | 0.5533±0.0000 (paper 0.5842, Δ -0.0309) | 0.5298±0.0000 (paper 0.6341, Δ -0.1043) | 0.5412±0.0000 (paper 0.5970, Δ -0.0558) |
|  |  |  |  |  | seed 1234 folds | 0.5833 / 0.5933 / 0.5667 / 0.5151 / 0.5084 | 0.5716 / 0.5399 / 0.5475 / 0.4902 / 0.4996 | 0.5774 / 0.5654 / 0.5569 / 0.5023 / 0.5040 |
| v2.0 | type | none | sum | - | [1, 1234, 2] | 0.4010±0.0112 (paper 0.5842, Δ -0.1832) | 0.3788±0.0236 (paper 0.6341, Δ -0.2553) | 0.3878±0.0151 (paper 0.5970, Δ -0.2092) |
|  |  |  |  |  | seed 1 folds | 0.4267 / 0.3600 / 0.3400 / 0.3980 / 0.5184 | 0.2784 / 0.3547 / 0.3637 / 0.3147 / 0.4521 | 0.3370 / 0.3573 / 0.3515 / 0.3515 / 0.4830 |
|  |  |  |  |  | seed 1234 folds | 0.4033 / 0.3667 / 0.5033 / 0.3980 / 0.3746 | 0.3884 / 0.3681 / 0.4651 / 0.4147 / 0.4129 | 0.3957 / 0.3674 / 0.4835 / 0.4062 / 0.3928 |
|  |  |  |  |  | seed 2 folds | 0.3267 / 0.4900 / 0.4633 / 0.3712 / 0.2742 | 0.3363 / 0.4387 / 0.4097 / 0.3537 / 0.3315 | 0.3314 / 0.4629 / 0.4349 / 0.3622 / 0.3002 |
| v2.0 | type | none | sum | LEAK | [1234] | 0.4686±0.0000 (paper 0.5842, Δ -0.1156) | 0.4496±0.0000 (paper 0.6341, Δ -0.1845) | 0.4583±0.0000 (paper 0.5970, Δ -0.1387) |
|  |  |  |  |  | seed 1234 folds | 0.3800 / 0.5933 / 0.4633 / 0.5284 / 0.3779 | 0.3611 / 0.5084 / 0.4636 / 0.5130 / 0.4021 | 0.3703 / 0.5476 / 0.4635 / 0.5206 / 0.3896 |
| v3.2_cuilab_deg2 | triplet | none | mean | - | [1234] | 0.9032±0.0000 (paper 0.9271, Δ -0.0239) | 0.9087±0.0000 (paper 0.9181, Δ -0.0094) | 0.8358±0.0000 (paper 0.8674, Δ -0.0316) |
|  |  |  |  |  | seed 1234 folds | 0.9055 / 0.9015 / 0.8983 / 0.9035 / 0.9071 | 0.9126 / 0.9029 / 0.9016 / 0.9119 / 0.9146 | 0.8399 / 0.8283 / 0.8281 / 0.8395 / 0.8433 |
| v3.2_cuilab_deg2 | triplet | none | sum | - | [1, 1234, 2] | 0.8964±0.0030 (paper 0.9271, Δ -0.0307) | 0.9025±0.0026 (paper 0.9181, Δ -0.0156) | 0.8322±0.0025 (paper 0.8674, Δ -0.0352) |
|  |  |  |  |  | seed 1 folds | 0.8900 / 0.8916 / 0.8928 / 0.8999 / 0.8963 | 0.9028 / 0.8936 / 0.8981 / 0.9049 / 0.8998 | 0.8367 / 0.8226 / 0.8305 / 0.8337 / 0.8288 |
|  |  |  |  |  | seed 1234 folds | 0.8998 / 0.9016 / 0.8922 / 0.9004 / 0.9091 | 0.9065 / 0.9049 / 0.8981 / 0.9076 / 0.9133 | 0.8394 / 0.8364 / 0.8233 / 0.8385 / 0.8406 |
|  |  |  |  |  | seed 2 folds | 0.8950 / 0.8987 / 0.8969 / 0.8848 / 0.8965 | 0.9012 / 0.9020 / 0.9040 / 0.8939 / 0.9066 | 0.8285 / 0.8293 / 0.8305 / 0.8265 / 0.8371 |
| v3.2_cuilab_deg2 | type | none | mean | - | [1234] | 0.3529±0.0000 (paper 0.7915, Δ -0.4386) | 0.3047±0.0000 (paper 0.9421, Δ -0.6374) | 0.3267±0.0000 (paper 0.8600, Δ -0.5333) |
|  |  |  |  |  | seed 1234 folds | 0.2786 / 0.3027 / 0.4202 / 0.4678 / 0.2955 | 0.2427 / 0.2279 / 0.3919 / 0.4001 / 0.2609 | 0.2595 / 0.2600 / 0.4055 / 0.4313 / 0.2771 |
| v3.2_cuilab_deg2 | type | none | sum | - | [1, 1234, 2] | 0.2916±0.0232 (paper 0.7915, Δ -0.4999) | 0.2341±0.0062 (paper 0.9421, Δ -0.7080) | 0.2567±0.0101 (paper 0.8600, Δ -0.6033) |
|  |  |  |  |  | seed 1 folds | 0.2841 / 0.3008 / 0.2743 / 0.2965 / 0.4509 | 0.2253 / 0.2380 / 0.2405 / 0.2374 / 0.2089 | 0.2513 / 0.2658 / 0.2563 / 0.2637 / 0.2856 |
|  |  |  |  |  | seed 1234 folds | 0.2415 / 0.2881 / 0.2961 / 0.2397 / 0.2580 | 0.2638 / 0.2256 / 0.1645 / 0.2491 / 0.2444 | 0.2522 / 0.2530 / 0.2115 / 0.2443 / 0.2510 |
|  |  |  |  |  | seed 2 folds | 0.3161 / 0.2863 / 0.2539 / 0.3201 / 0.2671 | 0.2437 / 0.2336 / 0.2513 / 0.2508 / 0.2346 | 0.2752 / 0.2573 / 0.2526 / 0.2813 / 0.2498 |
| v3.2_ref_MDAv3.2-3 | triplet | none | mean | - | [1234] | 0.8673±0.0000 (paper 0.9271, Δ -0.0598) | 0.8807±0.0000 (paper 0.9181, Δ -0.0374) | 0.8102±0.0000 (paper 0.8674, Δ -0.0572) |
|  |  |  |  |  | seed 1234 folds | 0.8569 / 0.8699 / 0.8762 / 0.8632 / 0.8706 | 0.8773 / 0.8828 / 0.8831 / 0.8806 / 0.8797 | 0.8072 / 0.8163 / 0.8076 / 0.8130 / 0.8069 |
| v3.2_ref_MDAv3.2-3 | triplet | none | sum | - | [1, 1234, 2] | 0.8701±0.0026 (paper 0.9271, Δ -0.0570) | 0.8798±0.0015 (paper 0.9181, Δ -0.0383) | 0.8100±0.0014 (paper 0.8674, Δ -0.0574) |
|  |  |  |  |  | seed 1 folds | 0.8713 / 0.8687 / 0.8680 / 0.8692 / 0.8783 | 0.8814 / 0.8728 / 0.8786 / 0.8789 / 0.8808 | 0.8114 / 0.8020 / 0.8104 / 0.8122 / 0.8043 |
|  |  |  |  |  | seed 1234 folds | 0.8613 / 0.8700 / 0.8783 / 0.8725 / 0.8808 | 0.8726 / 0.8862 / 0.8906 / 0.8796 / 0.8804 | 0.8051 / 0.8233 / 0.8153 / 0.8089 / 0.8041 |
|  |  |  |  |  | seed 2 folds | 0.8733 / 0.8553 / 0.8690 / 0.8649 / 0.8704 | 0.8830 / 0.8755 / 0.8743 / 0.8832 / 0.8800 | 0.8156 / 0.8079 / 0.8011 / 0.8181 / 0.8102 |
| v3.2_ref_MDAv3.2-3 | triplet | none | sum | LEAK | [1234] | 0.8811±0.0000 (paper 0.9271, Δ -0.0460) | 0.8917±0.0000 (paper 0.9181, Δ -0.0264) | 0.8201±0.0000 (paper 0.8674, Δ -0.0473) |
|  |  |  |  |  | seed 1234 folds | 0.8719 / 0.8831 / 0.8823 / 0.8798 / 0.8884 | 0.8874 / 0.8958 / 0.8963 / 0.8885 / 0.8902 | 0.8198 / 0.8254 / 0.8222 / 0.8153 / 0.8179 |
| v3.2_ref_MDAv3.2-3 | type | none | mean | - | [1234] | 0.4434±0.0000 (paper 0.7915, Δ -0.3481) | 0.3467±0.0000 (paper 0.9421, Δ -0.5954) | 0.3885±0.0000 (paper 0.8600, Δ -0.4715) |
|  |  |  |  |  | seed 1234 folds | 0.4345 / 0.3795 / 0.4785 / 0.4894 / 0.4350 | 0.3352 / 0.3251 / 0.3615 / 0.3377 / 0.3741 | 0.3784 / 0.3502 / 0.4119 / 0.3996 / 0.4023 |
| v3.2_ref_MDAv3.2-3 | type | none | sum | - | [1, 1234, 2] | 0.3130±0.0224 (paper 0.7915, Δ -0.4785) | 0.2486±0.0084 (paper 0.9421, Δ -0.6935) | 0.2727±0.0111 (paper 0.8600, Δ -0.5873) |
|  |  |  |  |  | seed 1 folds | 0.2748 / 0.2730 / 0.3291 / 0.3715 / 0.2524 | 0.2434 / 0.2158 / 0.2842 / 0.2063 / 0.2357 | 0.2581 / 0.2410 / 0.3050 / 0.2653 / 0.2438 |
|  |  |  |  |  | seed 1234 folds | 0.3457 / 0.3366 / 0.2713 / 0.2873 / 0.4814 | 0.2509 / 0.3203 / 0.2245 / 0.2786 / 0.2109 | 0.2908 / 0.3282 / 0.2457 / 0.2829 / 0.2933 |
|  |  |  |  |  | seed 2 folds | 0.2902 / 0.2307 / 0.3795 / 0.2782 / 0.2936 | 0.2421 / 0.2569 / 0.2116 / 0.3003 / 0.2469 | 0.2640 / 0.2431 / 0.2717 / 0.2888 / 0.2683 |
| v3.2_ref_MDAv3.2-3 | type | none | sum | LEAK | [1234] | 0.3367±0.0000 (paper 0.7915, Δ -0.4548) | 0.2505±0.0000 (paper 0.9421, Δ -0.6916) | 0.2784±0.0000 (paper 0.8600, Δ -0.5816) |
|  |  |  |  |  | seed 1234 folds | 0.3686 / 0.3234 / 0.2106 / 0.2977 / 0.4831 | 0.2279 / 0.2581 / 0.2540 / 0.3018 / 0.2107 | 0.2816 / 0.2871 / 0.2303 / 0.2997 / 0.2934 |

## 5. Ablation (Fig.4), v2.0 CV_type, seed 1234, 5 fold

| Biến thể | Top-1 F1 | Δ so với full (cùng seed 1234) | Paper: biến thể < full |
|---|---:|---:|---|
| full | 0.4091 (3 seed: 0.3878±0.0151) | — | — |
| w/o CL (λ1=λ2=0) | 0.4141 | +0.0050 | NGƯỢC |
| w/o HGCN (GCN trên đồ thị KNN cặp đôi) | 0.3968 | -0.0123 | khớp |
| w/o AVF (ghép nối [Z1‖Z2]) | 0.4759 | +0.0668 | NGƯỢC |
| w/o HGT (bỏ HGT) | 0.3728 | -0.0363 | khớp |
| w/o DV (chỉ 1 view, không có intra-CL) | 0.4202 | +0.0111 | NGƯỢC |

Ablation chỉ chạy 1 seed. Độ lệch chuẩn giữa các seed của bản full là 0.015. Chênh lệch của w/o AVF (+0.067, khoảng 4.4 sd) lớn hơn rõ mức nhiễu này. w/o HGT (−0.036, khoảng 2.4 sd) đúng chiều paper. Các chênh lệch còn lại (≤0.012, dưới 1 sd) chưa đủ bằng chứng để kết luận.

## 6. Phân bố dự đoán (CV_type, seed 1234, fold 0)

- v2.0_type_none_s1234: số dự đoán theo từng loại [137, 72, 63, 28], recall theo loại [0.71, 0.38, 0.34, 0.13], best_epoch qua các fold [300, 250, 440, 340, 440]
- v3.2_ref_MDAv3.2-3_type_none_s1234: số dự đoán theo từng loại [21, 295, 813, 364, 254], recall theo loại [0.03, 0.3, 0.51, 0.3, 0.11], best_epoch qua các fold [30, 450, 30, 420, 10]
- v3.2_cuilab_deg2_type_none_s1234: số dự đoán theo từng loại [52, 745, 1508, 116, 328], recall theo loại [0.03, 0.52, 0.62, 0.04, 0.12], best_epoch qua các fold [380, 20, 10, 30, 30]

Trên v3.2, early stopping nhiều khi dừng rất sớm (best_epoch 10–30), vì Top-1 trên tập validation dao động mạnh. Đây là hệ quả của giả định early stopping (mục 3, #12) và có thể làm kết quả v3.2 thấp hơn [suy luận]. Tôi không thay đổi giả định này để tránh tune theo kết quả.

## 7. Lệnh chạy

```bash
PY=~/dhg_runs/venv_np1/bin/python   # numpy 1.26 env
$PY code/build_datasets.py                      # -> data/*.npz + *.json
$PY code/test_faithful.py                       # ALL TESTS PASSED
$PY code/faithful.py --data v2.0 --protocol type --seed 1234 --threads 1          # chính
$PY code/faithful.py --data v3.2_cuilab_deg2 --protocol triplet --seed 1
$PY code/faithful.py --data v2.0 --protocol type --ablation no_avf
$PY code/faithful.py --data v2.0 --protocol type --recon_reduction mean           # chẩn đoán
$PY code/faithful.py --data v2.0 --protocol type --diag_leak 1                    # chẩn đoán
$PY code/aggregate.py
```
Danh sách job: `jobs.txt` (23 job chính) và `jobs2.txt` (10 job chẩn đoán). Log nằm trong `logs/`, JSON kết quả từng fold nằm trong `results/` (kèm args và DEVIATIONS).

## 8. Những gì còn bỏ ngỏ / cần bằng chứng thêm

- Paper không mô tả bước curation để ra 411×271 từ HMDD v3.2, và không ghi nguồn của các view "sequence"/"gene". Cần tác giả công bố dữ liệu đã xử lý.
- Paper không ghi các thông số sau: tỉ lệ negative của inter-CL, giá trị margin m và hệ số λ của margin, existence loss, cấu hình early stopping và scheduler, chi tiết gate của AVF. Tôi giữ của tác giả hoặc ghi [giả định]; bất kỳ giá trị nào trong số này cũng có thể làm thay đổi kết quả.
- Fig.2/Fig.3 (sweep) chưa chạy trên bản faithful. Ablation mới có 1 seed.
- Kết luận cuối: các con số trong paper (đặc biệt v3.2 CV_type F1 0.86) không thu được từ code đã công bố, kể cả sau khi sửa code cho đúng mô tả của paper, trên cả dữ liệu cuilab lẫn bộ 411×271 khớp Table 2.
