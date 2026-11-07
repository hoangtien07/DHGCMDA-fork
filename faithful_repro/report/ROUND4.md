# DHGCMDA – Round 4: chuẩn hoá cách đánh giá (a) và thêm view sequence/gene từ literature (b)

**Kết luận: vẫn không tái hiện được paper.**
- (a) Đổi sang cách đánh giá kiểu TDRC (micro recall, trung bình 10 lần rút negative) chỉ làm metric thay đổi tối đa ±0.004. Mức đó không giải thích được khoảng cách 0.03–0.07 (CV_triplet) và 0.21–0.61 (Top-1 F1) so với paper.
- (b) Bản cuilab 4 view (sequence + functional, gene + semantic) có Top-1 F1 không đổi trong phạm vi nhiễu seed. AUPR CV_triplet giảm nhẹ nhưng giảm đều ở cả 3 seed (−0.004).
- Model, loss và hyperparameter giữ nguyên như round 3. Không tune gì. Chạy 3 seed (1234, 1, 2) × 5 fold, báo cáo toàn bộ, không chọn seed hay fold.

## 1. Thay đổi so với round 3
| Thay đổi | Nguồn | Phạm vi ảnh hưởng |
|---|---|---|
| Thêm `top1_recall_micro` = TP / Σ(số type đúng), và `top1_f1_micro` | TDRC `experiments.py` (https://github.com/BioMedicalBigDataMiningLab/TDRC, commit 1de1a87) | Chỉ thêm metric. Macro P/R/F1 giữ nguyên |
| CV_triplet: 10 lần rút negative (số negative bằng số positive), lấy trung bình AUC/AUPR/F1; vẫn giữ `*_draw0` | TDRC CV_triplet (rút negative 10 lần) | Các pair negative của cả 10 lần đều bị loại khỏi pool negative dùng để train, nên huấn luyện CV_triplet khác round 3 một chút |
| Dataset `v3.2_cuilab_deg2_4view` | MRGBMDAT Sec 2.2 (DOI 10.1093/bioinformatics/btag534), KBLTDARD Eq.1 (DOI 10.1371/journal.pcbi.1012287), Cheng 2014 FunSim | Chỉ ảnh hưởng dataset mới. Các dataset cũ giữ nguyên |

Code: `code/faithful.py` (bản round 3 lưu ở `code/faithful_round2.py`), `code/build_4view.py`, `code/compare_r4.py`, `code/aggregate.py` (đặt `RES=results_r4` để tổng hợp).

## 2. Dataset 4 view (b)
Y, danh sách entity, view functional miRNA (MISIM 2.0) và view semantic bệnh SS1 (MeSH 2025) giống hệt bản `v3.2_cuilab_deg2`: 721 miRNA × 609 bệnh × 17,607 triplet, 5 type.

| View | Bản 2 view (round 3) | Bản 4 view (round 4) | Độ phủ |
|---|---|---|---|
| miRNA view 1 | GIP tính lại trên A_train mỗi fold | `m_ss`: Needleman–Wunsch (Biopython PairwiseAligner, global) trên mature sequence miRBase, lấy max giữa các arm, min-max | 628/721 |
| miRNA view 2 | MISIM 2.0 | MISIM 2.0 | 720/721 |
| disease view 1 | SS2 semantic (Xuan) | `d_gs`: FunSim (Cheng 2014). Gene của bệnh lấy từ CTD curated (evidence marker/mechanism) theo MeSH UI; độ tương đồng gene–gene là LLS của HumanNet-XC, min-max (KBLTDARD Eq.1) | 439/609 |
| disease view 2 | SS1 semantic (Wang) | SS1 semantic (Wang) | 496/609 |

Các giả định [giả định], không paper nào ghi rõ:
- Tham số chấm điểm NW: match +1, mismatch −1, gap open −2, gap extend −1.
- Entity không có sequence hoặc gene: giữ hàng identity (đường chéo = 1, phần còn lại = 0).
- Chỉ lấy evidence CTD loại marker/mechanism.

Lưu ý: bản 4 view **đổi đồng thời** 2 thứ ở mỗi phía (bỏ GIP và thêm sequence; bỏ SS2 và thêm gene), nên không tách được đóng góp riêng của từng thay đổi.

[suy luận] MRGBMDAT cùng nhóm tác giả nên là tiền lệ gần nhất, nhưng không chứng minh được DHGCMDA dùng đúng các tham số này.

Nguồn và SHA256 (file nằm trong `ext/seqgene/SHA256SUMS`):
- miRBase `mature.fa` (https://www.mirbase.org/download/mature.fa, header file không ghi version): `52d69eeb…79ff`. File lưu tên `.gz` nhưng thực chất là text thường.
- miRBase `hairpin.fa`: `0b315c60…a5b`
- CTD `CTD_curated_genes_diseases.tsv.gz` (http://ctdbase.org/reports/): `0422d3bd…b896`
- HumanNet-XC (https://www.inetbio.org/humannetv3/networks/HumanNet-XC.tsv): `d45931ef…282c`. File 31 MB, không đưa vào zip.
- Dataset đầu ra: `v3.2_cuilab_deg2_4view.npz` `1528f40d…4b0c`. Bản 2 view `v3.2_cuilab_deg2.npz` `0a4036ca…f462`.

## 3. Kết quả (mean ± sd qua 3 seed; mỗi seed là trung bình 5 fold)
Kết quả từng fold có trong `report/results_table_r4.md`.

### CV_type (Top-1)
| Dataset | P | R macro | F1 macro | R micro | F1 micro | Paper P/R/F1 |
|---|---:|---:|---:|---:|---:|---|
| v2.0 | 0.4010±0.0112 | 0.3788±0.0236 | 0.3878±0.0151 | 0.3577±0.0099 | 0.3781±0.0105 | 0.5842/0.6341/0.5970 |
| v3.2 cuilab, 2 view | 0.2916±0.0232 | 0.2341±0.0062 | 0.2567±0.0101 | 0.2276±0.0182 | 0.2557±0.0204 | 0.7915/0.9421/0.8600 |
| v3.2 cuilab, 4 view | 0.3018±0.0163 | 0.2230±0.0042 | 0.2521±0.0055 | 0.2358±0.0127 | 0.2647±0.0143 | 0.7915/0.9421/0.8600 |
| v3.2 bản tham chiếu 411×271 | 0.3130±0.0224 | 0.2486±0.0084 | 0.2727±0.0111 | 0.2326±0.0168 | 0.2669±0.0192 | 0.7915/0.9421/0.8600 |

### CV_triplet
| Dataset | AUPR (10 lần rút) | AUC (10 lần rút) | F1 (10 lần rút) | AUPR / AUC / F1 lần rút 0 | Paper AUPR/AUC/F1 |
|---|---:|---:|---:|---|---|
| v2.0 | 0.9190±0.0011 | 0.9210±0.0004 | 0.8565±0.0013 | 0.9214 / 0.9220 / 0.8568 | 0.9738/0.9669/0.9278 |
| v3.2 cuilab, 2 view | 0.8975±0.0018 | 0.9017±0.0027 | 0.8300±0.0031 | 0.8961 / 0.9018 / 0.8314 | 0.9271/0.9181/0.8674 |
| v3.2 cuilab, 4 view | 0.8935±0.0015 | 0.8989±0.0019 | 0.8274±0.0017 | 0.8921 / 0.8990 / 0.8282 | 0.9271/0.9181/0.8674 |
| v3.2 bản tham chiếu 411×271 | 0.8710±0.0017 | 0.8800±0.0017 | 0.8101±0.0011 | 0.8703 / 0.8800 / 0.8099 | 0.9271/0.9181/0.8674 |

## 4. Kiểm tra hồi quy so với round 3
- **CV_type:** P, macro R và macro F1 của từng fold **trùng khớp hoàn toàn** với round 3 trên cả 3 dataset × 3 seed (kiểm bằng `np.allclose`, chạy `compare_r4.py`). Như vậy phần sửa đánh giá không làm thay đổi cách huấn luyện, và kết quả chạy lại cho ra đúng như cũ (deterministic).
- **CV_triplet:** khác round 3 tối đa 0.0035 (v2.0 AUPR: round 3 0.9218, round 4 0.9190, lần rút 0 là 0.9214). Lý do: pool negative để train giờ loại thêm các pair negative của cả 10 lần rút (mục 1).

## 5. Diễn giải
1. **Định nghĩa recall Top-1 (G9):**
   - Ở mỗi pair model chỉ dự đoán 1 type và mỗi pair có ≥1 type đúng, nên micro R = TP/Σtrue ≤ TP/n = P (chứng minh trực tiếp từ định nghĩa).
   - Paper báo R = 0.9421 > P = 0.7915 trên v3.2. Điều này không thể xảy ra với micro recall, nên DHGCMDA đang dùng macro recall như code gốc.
   - Các baseline trong Table 3 (TDRC, …) có R < P, khớp với micro recall.
   - Kết luận: cột R/F1 của Table 3 đang so hai đại lượng có định nghĩa khác nhau. Khi chuyển sang micro, F1 của chúng tôi thay đổi trong khoảng −0.010 đến +0.009, tức đổi định nghĩa không làm mất hay thu hẹp đáng kể khoảng cách với paper.
2. **Cách rút negative cho CV_triplet (G10):** lấy trung bình 10 lần rút thay vì 1 lần chỉ thay đổi ≤0.0024 so với lần rút 0. Cách rút negative không giải thích được khoảng cách AUPR 0.03–0.055 so với paper.
3. **Bốn view thật (G2):**
   - Top-1 F1 macro chênh −0.0047 (theo từng seed: +0.005, −0.016, −0.003), nhỏ hơn sd giữa các seed (~0.01), nên không có thay đổi đáng kể.
   - F1 micro chênh +0.009 nhưng dấu không đồng nhất giữa các seed (+0.013, −0.009, +0.023).
   - AUPR CV_triplet chênh −0.0040 và **âm ở cả 3 seed** (−0.005, −0.004, −0.003). Mức giảm nhỏ nhưng nhất quán.
   - [suy luận] Có thể vì bỏ view GIP: GIP lấy thông tin trực tiếp từ A_train. Chưa có thí nghiệm tách riêng để kiểm chứng.
   - Kết luận: dùng đủ 4 view theo mô tả paper **không** đưa kết quả lại gần paper.
4. **Các gap vẫn còn mở:**
   - G1: curation 411×271×11,748 không được mô tả.
   - Margin, tỉ lệ negative của inter-CL, thang đo recon loss: không tìm thấy nguồn.
   - Chẩn đoán round 3 (1 seed): đổi recon loss sang dạng trung bình làm Top-1 F1 v2.0 tăng từ 0.388 lên 0.541. Đây vẫn là yếu tố đơn lẻ có ảnh hưởng lớn nhất đã đo được, nhưng vẫn dưới mức paper và không phải cách đọc đúng chữ của Eq.21.

## 6. Trạng thái từng claim (cập nhật)
| Claim của paper | Trạng thái |
|---|---|
| v2.0 CV_triplet AUPR/AUC/F1 0.9738/0.9669/0.9278 | Không tái hiện được (0.919/0.921/0.857). Riêng code gốc, chạy cả phần rò rỉ dữ liệu test, thì khớp |
| v2.0 Top-1 0.5842/0.6341/0.5970 | Không tái hiện được, kể cả với code gốc (0.557) |
| v3.2 CV_triplet 0.9271/0.9181/0.8674 | Không tái hiện được trên cả 3 phiên bản dataset |
| v3.2 Top-1 0.7915/0.9421/0.8600 | Không tái hiện được (F1 0.25–0.27) |
| So sánh với baseline v3.2 (Table 3/4) | Không hợp lệ về phương pháp: số TDRC lấy từ dataset 713×447, và recall được tính theo định nghĩa khác (micro so với macro) |
| Dữ liệu v3.2 có kích thước 411×271 sau khi lọc <2 | Không tái hiện được từ dữ liệu cuilab (ra 721×609) |

## 7. Lệnh chạy
```bash
cd ~/dhg_runs/faithful/code
~/dhg_runs/venv_np1/bin/python build_4view.py
cd .. && xargs -P 8 -L 1 ./run_r4.sh < jobs_r4.txt     # 24 job, --threads 1
cd code && RES=~/dhg_runs/faithful/results_r4 ~/dhg_runs/venv_np1/bin/python aggregate.py > ../report/results_table_r4.md
~/dhg_runs/venv_np1/bin/python compare_r4.py
```
Môi trường: Python venv `venv_np1`, gồm torch 2.5.1+cpu, numpy 1.26.4, pandas 2.2.3, biopython 1.88.
