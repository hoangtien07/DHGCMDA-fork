# Hội đồng chuyên gia: phương án gỡ các block của code DHGCMDA hiện tại

Hội đồng gồm 5 vai, cùng đánh giá trên một bộ bằng chứng:
- **[D]** Data curation / bioinformatics
- **[M]** Deep learning / tối ưu hoá
- **[E]** Đánh giá / thống kê
- **[R]** Reproducibility / liêm chính nghiên cứu
- **[S]** Kỹ thuật / tính toán

Nguyên tắc chung: chỉ chọn phương án dựa trên tài liệu hoặc code của tác giả, không dựa trên điểm số. Mọi biến thể không có trong tài liệu chỉ được báo cáo là sensitivity, có đăng ký trước và chạy đủ 3 seed.

## Phát hiện mới trong phiên hội đồng (đã đo, script `code/ref_provenance.py`, output `report/ref_provenance.txt`)
**Block lớn nhất (G1, nguồn của bộ 411×271) đã gỡ được một nửa.**
- Toàn bộ 11,748/11,748 triplet của artifact 411×271 có mặt trong **bộ file CUILab không có tiền tố** (`genetics/epigenetics/circulation/target/tissue.txt`, bản 1,151×802×20,041).
- Trong phạm vi 411×271 entity đó, bộ CUILab chỉ có thêm đúng 1 triplet không nằm trong artifact.
- So với bộ `v3_*` (bộ ta dùng làm primary) thì chỉ khớp 10,749/11,748, và 5 bệnh không có mặt.
- Map category → type khớp 100% (đếm theo pair):

| Type | Category CUILab gộp vào |
|---|---|
| genetics | genetics_* (5 subcategory) |
| circulation | circulation_biomarker_* |
| target | target gene, therapeutic target, transcription factor target, lncRNA target |
| tissue | tissue_expression_* |
| epigenetics | epigenetics |

- **Chọn entity:** toàn bộ 411 miRNA và 271 bệnh đều có degree ≥3 trên dữ liệu thô. Hiện tượng degree 1 trong artifact là do lọc degree trên dữ liệu thô trước rồi mới cắt xuống tập con [suy luận]. Điều này hoá giải phần lớn mâu thuẫn với câu "<2 bị loại".
- **Quy tắc chọn tập entity vẫn chưa biết.**
  - Không phải lọc theo degree thuần: có các bệnh degree rất cao bị loại, ví dụ gastric neoplasms (535), NSCLC (393), alzheimer disease (135).
  - Tên bệnh khớp tên MeSH 2025 (heading hoặc entry): 271/271 bệnh trong artifact, so với 270/531 bệnh ngoài artifact. Đây mới là điều kiện cần, chưa đủ.

Hệ quả [R]: artifact 411×271 **có nguồn gốc CUILab**, nên đạt một phần ràng buộc "data tải từ cuilab" của user, cụ thể là phần triplet. Danh sách entity thì vẫn lấy từ artifact của tác giả, chưa có quy tắc.

## Các block và phương án

| # | Block | Bằng chứng hiện có | Phương án hội đồng đề xuất | Loại |
|---|---|---|---|---|
| B1 | Quy tắc chọn 411 miRNA × 271 bệnh | Ở trên | **[D]** (i) Học quy tắc bằng một decision tree nông trên các feature của entity (degree thô theo từng type, có tên MeSH, có trong MISIM, có trong miRBase, có cây MeSH C-tree, năm PMID), rồi kiểm tra quy tắc tìm được có tái tạo đúng 411/271 không. Không ép cho khớp: nếu tree không tách được hai nhóm thì báo là không tìm ra. (ii) **[R]** Gửi email hỏi tác giả (Shang Junliang / Liu Jin-Xing, corresponding author), kèm số liệu cụ thể | Phân tích, chi phí thấp |
| B2 | Artifact chỉ có 2 similarity view (mi_fun, DSSM) | `data/v3.2_ref_MDAv3.2-3.npz` | **[D]** Dựng `ref_4view`: thêm m_ss (NW miRBase) và d_gs (CTD+HumanNet) bằng đúng pipeline của round 4, rồi chạy CV_type/CV_triplet với 3 seed | Tiền lệ MRGBMDAT |
| B3 | Thang đo recon loss: chữ của Eq.21 là ‖·‖²_F (tổng), code tác giả dùng 0.15×MSE (trung bình) | Chẩn đoán 1 seed: v2.0 Top-1 F1 tăng từ 0.388 lên 0.541 | **[M][R]** Hai nguồn mâu thuẫn và đều đáng tin ngang nhau: paper là đặc tả, code là bản cài đặt thật. Đăng ký trước cả hai bản (`sum` = primary, `mean` = cách đọc theo code), chạy đủ 3 seed × 3 dataset và báo cáo song song. Không chọn bản nào theo điểm | Sensitivity đăng ký trước |
| B4 | Predictor: paper chỉ ghi "bilinear predictor", không có phương trình. Code dùng bilinear chéo (diagonal) | Paper Sec 2 (`lit/dhgcmda.txt`). Thử nghiệm trước của user trong fork (CLAUDE.md Plan J-1): full bilinear cho v2.0 0.635, nhưng chạy trên code gốc còn rò rỉ dữ liệu test | **[M]** "Bilinear" theo cách hiểu chuẩn là xᵀW_t y với W_t đầy đủ [suy luận]; bản chéo là trường hợp riêng. Hiện chưa đủ bằng chứng để chọn → chạy full bilinear làm sensitivity đăng ký trước, giữ bản chéo (theo code tác giả) làm primary | Sensitivity |
| B5 | Trọng số existence loss: paper không định nghĩa loss này | CLAUDE.md Plan H: exist_weight 0.1 đạt 0.5996, nhưng cũng trên code rò rỉ | **[E][R]** Không chọn theo điểm. Code tác giả dùng 0.3/0.7 → giữ nguyên. Nếu chạy thêm {0.1, 0.3} thì chỉ báo là sensitivity, không dùng để cập nhật kết quả chính | Sensitivity, ưu tiên thấp |
| B6 | Mâu thuẫn về rò rỉ dữ liệu: CLAUDE.md M16 ghi "NO train/test leakage — CV SOUND" | Faithful: `train_data[4]` là ma trận association đầy đủ, được dùng cho feature/KNN/HGT/inter-CL (đọc code). Bật lại rò rỉ thì AUPR tăng 0.922→0.943 (đã đo) | **[R]** Hai kết luận ở hai mức tin cậy khác nhau: rò rỉ đã được xác minh bằng cả đọc code lẫn thí nghiệm can thiệp. [suy luận] M16 nhiều khả năng chỉ kiểm tra rò rỉ ở mức index của split. Đề xuất ghi đính chính vào CLAUDE.md của fork | Sửa tài liệu |
| B7 | So sánh baseline không công bằng: số TDRC lấy từ dataset 713×447 | Round 3/4 | **[E]** Chạy TDRC (clone sẵn, commit 1de1a87) trên đúng 3 dataset của ta, cùng split và metric (macro và micro), được baseline cùng điều kiện. Dù DHGCMDA không đạt số paper, vẫn kiểm được claim "vượt baseline" | Đánh giá |
| B8 | Threshold F1 chọn trên tập test (cách của tác giả, cho số lạc quan) | Calculate_Metrics.py:119 | **[E]** Báo thêm F1 với threshold chọn trên tập validation; giữ F1 kiểu tác giả để so được với paper | Đánh giá |
| B9 | Ablation (Fig.4) mới 1 seed; sweep Fig.2/3 chưa chạy trên code faithful | Round 3 | **[E]** 5 ablation × 3 seed trên v2.0 và bản 411×271. Sweep K và λ2 với 3 seed, chỉ để kiểm tra claim về hình dạng đường cong, không dùng để chọn tham số | Kiểm tra claim |
| B10 | Chạy chậm: load ~30 trên 8 core, mỗi process có 15 thread dù đã đặt `--threads 1` | `ps` lúc chạy round 4 | **[S]** Đặt `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` và `torch.set_num_interop_threads(1)`. Mức tăng tốc chưa đo [suy luận]; benchmark trước khi chạy hàng loạt | Kỹ thuật |

## Thứ tự đề xuất (hội đồng thống nhất)
1. **B10**: sửa luồng thread và benchmark (điều kiện để mọi bước sau chạy nhanh).
2. **B1(i)**: phân tích quy tắc chọn entity; **B1(ii)**: soạn email hỏi tác giả.
3. **B2 + B7**: `ref_4view` và baseline TDRC cùng điều kiện, 3 seed.
4. **B3 + B4**: sensitivity đăng ký trước (recon mean, full bilinear), 3 seed × 3 dataset, báo cáo tách riêng khỏi kết quả chính.
5. **B9, B8**: ablation và sweep nhiều seed, cùng F1 với threshold chọn trên validation.

**Không làm:** chọn tổ hợp nào cho điểm cao nhất rồi gọi đó là "kết quả tái hiện". Kết quả chính giữ nguyên định nghĩa đã chốt ở round 3/4.

Ý kiến bảo lưu [R]: kể cả khi B1 tìm ra đúng quy tắc chọn entity, khoảng cách Top-1 (0.27 so với 0.86) vẫn khó đóng. Đo trên chính artifact 411×271 chỉ được 0.27, nên dữ liệu không phải nguyên nhân chính của khoảng cách Top-1.
