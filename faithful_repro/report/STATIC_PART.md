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
