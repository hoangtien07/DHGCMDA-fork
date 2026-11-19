# THESIS_REPRO_LIMITATIONS_VI — Giới hạn tái hiện (bản tiếng Việt cho luận văn)

## Những gì có thể khẳng định (đã chứng minh)

1. **Định nghĩa metric đánh giá là rào cản lớn nhất** (chứng minh mới ở WP1, REV F1.1):
   với mọi convention đánh giá suy ra được từ nguồn công khai — K5 scalar truth
   (first/last/bất kỳ phép gán), pooled hay fold-wise với precision mỗi fold
   tùy ý, multi-hot một dự đoán, K4 filtered — cặp số P=0.7915 / R=0.9421 của
   paper vượt quá cận trên chặt (0.758–0.917). Để đạt R=0.9421 với P=0.7915 cần
   một lớp scalar có support ≥ ~6.285 mẫu, trong khi lớn nhất khả dĩ là 3.997.
   → Kết luận đúng là "convention đánh giá không xác định được từ nguồn công
   khai" (UNRESOLVED_EVALUATION_CONVENTION), KHÔNG phải "paper sai".
   Lưu ý độ chặt: cận fold-wise 0.9123 tính trên partition fold đã khóa/đã
   audit với phân bổ lỗi tối ưu (greedy) — là cận trên hợp lệ kiểu relaxed
   (không phân bổ nào vượt quá), KHÔNG phải giá trị lớn nhất đã chứng minh
   đạt được.

2. **Không nhất quán trong bộ ba số P/R/F1**: harmonic(0.7915, 0.9421) = 0.86026
   làm tròn thành 0.8603, nhưng paper báo 0.8600. Không tồn tại P,R chưa làm
   tròn nào trong khoảng làm tròn 4 chữ số cho pooled harmonic = 0.8600. Điều
   này chỉ nhất quán với aggregation khác (ví dụ mean-of-fold-F1 — có nghiệm
   chứng minh được: 1 fold 0.6816/0.7504 + 4 fold 0.8190/0.9900 → mean F1 =
   0.8600 đúng tuyệt đối; chỉ là nghiệm tồn tại, không khẳng định đó là fold
   thật của tác giả hay dạng phân bố này là bắt buộc) hoặc giá trị gốc khác.
   Độ lệch 3e-4 cũng có thể chỉ là lỗi làm tròn/in ấn — không kết luận gian lận.

3. **Evaluator phát hành không chấm được v3.2** (K4 chỉ nhận 4 loại, bỏ hết
   hàng K5) — lỗi thật của code release, đã kiểm chứng bằng fixture.

4. **Dữ liệu khớp về mặt thống kê**: tensor 411×271×5, 11.748 links, density
   10.55% khớp mô tả paper (nguồn SPLD MDAv3.2-3, đồng nhất mức L1). Nhưng mọi
   view trong bundle vẫn là UNKNOWN_PROVENANCE — không được tuyên bố byte-identical.

5. **Giá trị đo được của ta (gắn nhãn "our interpretation", không so trực tiếp
   với paper)**: U1 top1 harmonic 0.456–0.492; baseline sửa lỗi tốt nhất
   0.5298; Track N A1 âm tính (−0.014 so với baseline).

6. **Provenance bốn view** (đã sửa theo code release thật, tách 3 lớp):
   (a) bốn input similarity thô `M_GSM/M_FSM/D_SSM2/D_SSM1` là file gốc được
   release nạp trực tiếp — không khẳng định mọi file nguồn này đều đã chứa
   GIP; (b) ma trận IM/ID tăng cường GIP tính trên TOÀN BỘ ma trận liên kết
   trong pipeline release → input model của release bị leak (kết luận leak
   đứng ở lớp này); (c) các feature/graph dựng trực tiếp trên ma trận liên
   kết đầy đủ → leak. MISIM (view chức năng) vốn là association-derived ở
   upstream — KHÔNG được gọi "clean" chỉ vì là database công khai. d2025.bin
   (NLM MeSH 2025) và mtrees2019.bin là hai artifact khác nhau cho view
   semantic.

## Những gì KHÔNG được khẳng định

- Paper sai / gian lận — không chứng minh được; các thoát hiểm còn mở (tập
  nhãn khác, đơn vị dự đoán nhiều type/cặp, micro recall, aggregation khác).
- Gọi convention nào cho F1 cao nhất là "convention của tác giả".
- Tuyên bố tương đương fold/feature/eval của tác giả khi artifact chưa public.
- Biến các lựa chọn thiết kế port (PAPER_LITERAL_K5) thành "sự thật của tác giả".

## Giới hạn không giải quyết được trong điều kiện hiện tại

- Evaluator convention và fold/input artifact: chỉ đóng được bằng tài liệu của
  tác giả — owner cấm liên hệ → blocker vĩnh viễn, ghi nhận minh bạch.
- Provenance các view trong bundle: UNKNOWN cho tới khi trace được công thức.

## Ba protocol tách biệt (không trộn kết quả)

- RELEASE_K4_V20: pipeline release nguyên trạng, evaluator K4, v2.0.
- PAPER_LITERAL_K5: dựng lại theo phương trình paper — mọi lựa chọn underdetermined
  phải nêu là option rõ ràng (scalar-first vs multi-hot target, fusion cố định
  vs học, recon formulation), không ngầm dùng heuristic của U1.
- CLEAN_REPRO_K5: suy luận độc lập mask-an-toàn trên PUBLIC_RECONSTRUCTED_4VIEW.

## Bằng chứng dùng được cho luận văn

- Bảng cận trên vs R công bố (JOINT_PR_THEOREM_REV_F1_1) — chứng minh dạng
  toán, kiểm chứng bằng brute-force fixture.
- Bảng fidelity trace (predictor/loss/graph — phân loại same/declared_port/
  adaptation/incompatible).
- Thí nghiệm tách leakage theo cặp đối chứng (v2.0 +0.049 / v3.2 ≈0).
