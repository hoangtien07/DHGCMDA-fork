# Giải thích paper DHGCMDA — theo từng chương

Series giải thích paper **DHGCMDA** (*"DHGCMDA: a dual-view heterogeneous graph contrastive learning framework for miRNA-disease association type prediction"*, Sun Y. et al., BMC Bioinformatics 2026) viết cho **người mới** — không yêu cầu nền GNN hay bioinformatics.

- Paper gốc: file PDF ở root repo (`[2026] DHGCMDA ... .pdf`), text trích xuất ở [`../../_pdf_text/`](../../_pdf_text/).
- Mỗi chương giải thích một phần của paper, kèm: **trực giác → toán → map sang code** trong repo này.
- Ký hiệu `Eq. (n)` / `Fig. n` / `Table n` = đánh số trong paper gốc.

## Lộ trình đọc

| Chương | Nội dung | Phần paper tương ứng |
|--------|----------|----------------------|
| [01 — Bối cảnh & bài toán](01-boi-canh-va-bai-toan.md) | miRNA là gì, tại sao phải dự đoán *loại* liên kết, phát biểu bài toán | §1 Introduction, §2.2 |
| [02 — Dữ liệu & hai "view"](02-du-lieu-va-hai-view.md) | 4 ma trận similarity, ma trận association A, augmented features | §2.2–2.3 |
| [03 — Hypergraph & HGCN](03-hypergraph-va-hgcn.md) | Siêu đồ thị, hyperedge KNN, incidence matrix H, Laplacian G, tích chập siêu đồ thị | §2.3–2.4 |
| [04 — Contrastive learning](04-contrastive-learning.md) | Intra-modality CL (NT-Xent) + cross-modality CL (InfoNCE + margin) | §2.4.1–2.4.2 |
| [05 — View fusion, reconstruction & dynamic update](05-fusion-reconstruction-dynamic-update.md) | Attention fusion, decoder tái tạo similarity, cập nhật đồ thị mỗi 5 epoch | §2.5–2.8 |
| [06 — HGT & bộ dự đoán](06-hgt-va-predictor.md) | Heterogeneous Graph Transformer, type-aware attention, bilinear predictor | §2.7 |
| [07 — Loss tổng hợp & training loop](07-loss-tong-hop-va-training.md) | Eq. 32–34, class weighting, negative sampling, tổng pipeline end-to-end | §2.9 |
| [08 — Đánh giá & kết quả](08-danh-gia-va-ket-qua.md) | HMDD v2.0/v3.2, CVtype vs CVtriplet, ablation, case study, limitations | §3 Results |
| [09 — Paper vs code trong repo này](09-paper-vs-code.md) | Map Eq/class → file, các chỗ code khác paper, kết quả reproduce | Toàn bộ |

## Đọc nhanh 5 phút

Mô hình trả lời 2 câu hỏi cho mỗi cặp (miRNA, bệnh): **(1)** có liên kết không? **(2)** nếu có, theo cơ chế nào (genetics / epigenetics / circulation / target / tissue)?

Pipeline 1 dòng:

```
4 ma trận similarity → 4 hypergraph KNN → 2 nhánh HGCN + contrastive learning
→ attention fusion → HGT trên heterogeneous graph → bilinear predictor → type
```

Nếu chỉ đọc 3 chương: **01 → 03 → 09**.

## Ghi chú cho người reproduce

Chương 09 là quan trọng nhất nếu bạn định chạy lại code: repo này là fork đã được audit kỹ, có nhiều chỗ **code ≠ paper** (vd. "attention" fusion thực chất là weighted sum cố định, metric Top-1 của tác giả không chấm được 5 types của v3.2). Các chương kỹ thuật (03–07) giải thích *paper nói gì*; chương 09 giải thích *code thực sự làm gì*.
