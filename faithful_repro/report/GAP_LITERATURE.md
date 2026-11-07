# Lấp gap của DHGCMDA bằng phương pháp từ các paper liên quan

Nguyên tắc: phương án chỉ được chọn theo tài liệu, không chọn theo điểm số. Mỗi dòng ghi nguồn; chỗ nào là suy luận của tôi thì gắn nhãn [suy luận].

## Kết luận

- **Bảng so sánh baseline của paper dùng số liệu trên dataset khác.** Bằng chứng đo được: Table 3/4 của DHGCMDA ghi TDRC trên v3.2 là CV_type 0.4926/0.3671/0.4207 và CV_triplet 0.9059/0.8962/0.8309. Các số này trùng khớp từng chữ số với kết quả TDRC tự báo trong paper TDRC, vốn chạy trên dataset 713×447 (repo TDRC, bộ `HMDD3.2_processed`), không phải bộ 411×271.
- **Định nghĩa Top-1 recall khác với baseline** [suy luận, căn cứ chắc]:
  - TDRC tính micro recall = TP / Σ(số type thật) (`experiments.py` L13–65). Vì mỗi cặp có ít nhất 1 type nên luôn có R ≤ P.
  - DHGCMDA báo R > P (v3.2: 0.9421 > 0.7915), tức là dùng macro recall theo từng type như code đã phát hành.
  - Các baseline TDRC, NMCMDA, SPLDHyperAWNTF, TFLP, KBLTDARD đều có R < P.
  - Như vậy cột Top-1 R và Top-1 F1 trong Table 3 so hai đại lượng khác nhau.
- **Bộ 411×271×11,748:** không có paper liên quan nào mô tả bộ này. Tôi kiểm 7 bản xử lý HMDD v3.2 công khai, không bản nào khớp (bảng G1). Gap này vẫn chưa giải được nếu tác giả không công bố dữ liệu.
- **4 nguồn similarity:** paper MRGBMDAT (Bioinformatics 2026, cùng nhóm tác giả Sun/Shang/Liu, repo CDMBlab) mô tả cụ thể từng nguồn. README của MRGBMDAT dùng đúng các tên biến `m_ss`, `mi_fun`, `d_gs`, `dis_sem` như code DHGCMDA. Đây là nguồn tốt nhất để lấp gap G2.

## Bảng gap → phương pháp → nguồn

| # | Gap trong paper/code DHGCMDA | Các paper liên quan làm thế nào | Nguồn | Có áp dụng được không (không thiên về điểm số) |
|---|---|---|---|---|
| G1 | Curation để ra 411×271×11,748 | Kết quả các bản xử lý HMDD v3.2: TDRC/TFLP 713×447×12,534; WeightTDAIGN có MDA v3.2-5 125×65×4,785 (chỉ giữ miRNA có sequence trong miRBase và bệnh có trong MeSH); MRGBMDAT 853×591, 12,445 cặp, nhãn up/down/other; cuilab `v3_*` lọc deg≥2 lặp ra 720×606×17,603; cuilab `alldata.txt` (sha256 ef1a0c70…) lọc deg≥2 lặp ra 783×638×19,508. Riêng các file `old_*` trên cuilab là HMDD v2.0 | TDRC repo commit 1de1a87; TFLP commit 9de0ac3; WeightTDAIGN PMC9335924 Table 1; MRGBMDAT commit 133b3e3, `dataset.rar` sha256 e796067e…; cuilab.cn/static/hmdd3/data | **Không.** Không bản nào khớp. Chỉ còn cách xin dữ liệu từ tác giả |
| G2 | Nguồn của sequence sim (miRNA) và gene sim (bệnh) | Mô tả của MRGBMDAT: sequence = Needleman–Wunsch trên seed region (Rashed 2021); functional = Xiao 2018 (miRNA–gene kết hợp HumanNet); semantic = MeSH DAG; disease target/gene-based = số gene chung (Cheng 2014). KBLTDARD: gene sim từ HumanNet LLS, chuẩn hoá min-max (Eq.1), gene của miRNA lấy từ miRTarBase 8.0. WeightTDAIGN: Biostrings `pairwiseAlignment` rồi chuẩn hoá min-max | MRGBMDAT PMC13447018 §2.2; KBLTDARD PMC11257412; WeightTDAIGN PMC9335924 §2.5 | **Có.** Dựng `m_ss` (NW, miRBase) và `d_gs` (Cheng 2014, DisGeNET/HumanNet) cho bản cuilab, thay GIP ở view 1, nhờ đó loại bỏ hẳn rò rỉ ở view này |
| G3 | Tỉ lệ negative của inter-CL ("sampled negative pairs") | HGCLAMIR/GRACE dùng mọi node khác làm negative ở intra-CL. Không paper nào cho tỉ lệ negative của inter-CL | HGCLAMIR PMC11037542 | Không có nguồn, giữ ratio 10 [giả định] |
| G4 | Giá trị margin m và hệ số λ của margin | Không tìm thấy | — | Giữ giá trị của code [giả định] |
| G5 | Existence loss (paper có nêu task này nhưng không có phương trình) | HGCLAMIR dùng BCE có trọng số với bias α để xử lý mất cân bằng (Eq.28). MRGBMDAT dùng bilinear decoder kết hợp chọn negative | HGCLAMIR Eq.28 | Có căn cứ để giữ BCE existence; hệ số vẫn là [giả định] |
| G6 | Thiết kế gate của AVF | HGCLAMIR Eq.20: α = FNN(GAP(Z)), FNN 2 lớp với ReLU rồi Sigmoid | HGCLAMIR Eq.20 | **Đã khớp** với bản cài đặt faithful, không cần sửa |
| G7 | Scheduler và early stopping ("suitable") | TDRC không dùng early stopping. HGCLAMIR chia 9/10 cho CV và 1/10 làm independent test. MRGBMDAT sweep lr | TDRC; HGCLAMIR §Results | Chưa đủ bằng chứng để chọn. Đề xuất: báo thêm biến thể chạy số epoch cố định, không dùng early stopping, như một sensitivity |
| G8 | Thang đo của recon loss (‖·‖_F²) | Không có. HGCLAMIR reconstruct ma trận association chứ không phải ma trận similarity | — | Giữ cả 2 bản (sum là chính, mean là sensitivity) |
| G9 | Định nghĩa Top-1 recall | TDRC: micro, R ≤ P. KBLTDARD/SPLDHyperAWNTF/TFLP có R < P [suy luận: cùng định nghĩa micro] | TDRC `experiments.py` | **Có, chỉ sửa phần đánh giá:** báo thêm micro recall để so được với baseline |
| G10 | Cách lấy negative cho CV_triplet | TDRC lấy số negative bằng số positive, rút 10 lần rồi lấy trung bình. KBLTDARD rút 20 seed. F1 lấy giá trị tốt nhất trên 999 ngưỡng | TDRC L67–135; KBLTDARD | **Có, chỉ sửa phần đánh giá:** lấy trung bình trên 10 lần rút negative (bản faithful hiện chỉ rút 1 lần) |
| G11 | Khoảng cách dùng cho KNN | Paper DHGCMDA ghi cosine. HGCLAMIR và code upstream dùng Euclid | DHGCMDA PMC13169815; HGCLAMIR | Giữ cosine theo paper. Euclid là di sản từ HGCLAMIR [suy luận] |
| G12 | Similarity tính từ association (GIP) | TDRC tính lại functional sim từ ma trận train ở mỗi fold | TDRC L34 | Xác nhận cách sửa rò rỉ của bản faithful là đúng chuẩn |

## Repo và file đã kiểm

- TDRC: https://github.com/BioMedicalBigDataMiningLab/TDRC, commit 1de1a878.
- TFLP: https://github.com/nayu0419/TFLP, commit 9de0ac34.
- MRGBMDAT: https://github.com/CDMBlab/MRGBMDAT, commit 133b3e31.
- Không clone được: HGCLAMIR (GitHub API trả "Not Found"), SPLDHyperAWNTF_Model và KBLTDARD (proxy chặn hoặc bị rate-limit). Tôi chỉ đọc paper toàn văn của các repo này qua Europe PMC.
