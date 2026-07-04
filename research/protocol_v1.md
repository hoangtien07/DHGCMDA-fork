# Giao thức Anti-Hallucination v1 (cập nhật SESSION 1 — 2026-07-05)

> Kế thừa Mục 2 của `LITERATURE_RESEARCH_PROMPT.md`, bổ sung kỹ thuật 2025–2026 rút từ 3 search Session 1
> (xem `search_log.md`). **Nhắc lại LUẬT VÀNG: không cite gì từ trí nhớ — mọi trích dẫn phải fetch LIVE.**

## Phần A — 13 luật gốc (R1–R13) giữ nguyên hiệu lực
Không thay đổi. Đây là xương sống.

## Phần B — 4 bổ sung từ best-practice 2025–2026 (R14–R17)

**R14 — Span/claim-level grounding.** Không chỉ gắn 1 URL cho cả đoạn. Mỗi CÂU khẳng định định lượng phải map tới **một span/ô bảng cụ thể** trong nguồn đã fetch. Nếu một câu không map được về span đã đọc → tách ra và hạ xuống `[RECALL]` hoặc xoá. (Nguồn kỹ thuật: span-level verification / claim-level grounding — SEARCH-HIT-only, dùng làm phương pháp, không làm citation.)

**R15 — Ba trục kiểm citation (precision / relevance / temporality).** Với mỗi bài trước khi lên map, tự hỏi:
  1. **Precision** — bài này CÓ TỒN TẠI đúng như mô tả? (khớp tiêu đề+tác giả+venue+ID).
  2. **Relevance** — nó thật sự chống lưng cho claim ta gán, hay chỉ liên quan mơ hồ?
  3. **Temporality** — con số/"SOTA" còn hiệu lực ở thời điểm ta viết, hay đã bị bài mới vượt?
  Ghi 3 tick này vào cột "Cờ đối kháng" của ledger.

**R16 — Đa nguồn cross-validate cho mỗi bài lõi.** Ưu tiên xác nhận sự tồn tại của một bài qua **≥2 cơ sở dữ liệu có cấu trúc độc lập** (OpenAlex + PubMed/PMC, hoặc + Semantic Scholar/DBLP). Một mình Google Scholar KHÔNG đủ để coi là VERIFIED. Chỉ đánh VERIFIED-fetched khi đã mở được trang thật (abstract tối thiểu) ở ≥1 nguồn có cấu trúc.

**R17 — Định lượng tỉ lệ bịa (self-audit).** Cuối mỗi session báo cáo: `#bài SEARCH-HIT / #bài fetch thử / #bài VERIFIED / #bài LOẠI vì không tồn tại`. Literature 2025–2026 ghi nhận tỉ lệ LLM bịa citation 18–95% tuỳ model → coi con số "#LOẠI vì không tồn tại > 0" là **bình thường và tốt** (bằng chứng ta đang lọc thật), không phải thất bại.

## Phần C — Thứ tự thao tác chuẩn cho mỗi bài (checklist thực thi)
1. WebSearch tên bài / chủ đề → lấy candidate.
2. WebSearch lại khớp `tiêu đề + tác giả + venue` (R3).
3. WebFetch trang thật (ưu tiên OpenAlex/PubMed/arXiv/DOI landing) → đọc abstract + copy quote đóng góp + copy ô metric (R4/R14).
4. Cross-check nguồn thứ 2 (R16).
5. Đối kháng: retracted? predatory? năm/tác giả lệch? (R6/R9/R15).
6. Ghi ledger với trạng thái + confidence. Chỉ VERIFIED mới ra map.

---

# KẾ HOẠCH TRUY VẤN CHI TIẾT (dùng cho SESSION 2–3)

> Trần: ≤~12 search + ≤~10 fetch mỗi session. Danh sách dưới là *pool* ưu tiên; chọn lọc theo trần.

## SESSION 2 — Cụm A: dòng dõi DHGCMDA & MDA methods (mục tiêu ~10 bài VERIFIED)

**A0. Bài gốc & baseline trực tiếp (bắt buộc thử fetch)**
- `DHGCMDA dual-view heterogeneous graph contrastive learning miRNA disease Sun 2026 BMC Bioinformatics`
- `NMCMDA neural multi-category miRNA disease association`
- `TDRC miRNA disease association tensor` (đã reproduce ~98% — cần bản gốc để cite chính xác)

**A1. MDA bằng GNN / hypergraph / contrastive (2022–2026)**
- `miRNA disease association prediction graph neural network 2024 2025`
- `hypergraph neural network miRNA disease association`
- `contrastive learning miRNA disease association prediction`
- `heterogeneous graph transformer miRNA disease`
- `multi-type miRNA disease association type prediction HMDD`

**A2. Benchmark & data**
- `HMDD v3.2 database miRNA disease association benchmark`
- `HMDD v2.0 495 miRNA 383 disease evaluation`

→ Fetch ưu tiên: bản gốc DHGCMDA, NMCMDA, TDRC, 2–3 MDA-GNN mới nhất. Output: `biblio_A.md`.

## SESSION 3 — Cụm B (phương pháp nền) + C (cross-cutting nơi novelty của ta trú)

**B. Phương pháp nền**
- `hypergraph convolution network HGCN link prediction`
- `KNN hypergraph construction neighbor size sensitivity`
- `heterogeneous graph transformer HGT Hu 2020`
- `bilinear / tensor factorization link predictor knowledge graph` (đối chiếu full vs diagonal bilinear)
- `graph contrastive learning survey 2024 2025`

**C. Cross-cutting — trực tiếp chống lưng novelty (A) & (B)**
- `over-smoothing graph neural network small sparse graph`
- `over-parameterization GNN low-data regime overfitting`
- `GNN ablation study component contribution reversal negative result`
- `hypergraph K nearest neighbor sensitivity link prediction biomedical`
- `reproducibility crisis machine learning bioinformatics released code bug`
- `evaluation pitfalls metric implementation bug published ML benchmark`
- `data leakage benchmark curation biomedical link prediction`
- `contrastive learning collapse small dataset`

→ Output: `biblio_B_C.md`.

## SESSION 5 — Truy vấn PHỦ ĐỊNH R7 (novelty) — soạn sẵn ≥3/ứng viên
- **Ứng viên A (predictor faithfulness / reproduction critique):**
  `reproducibility study miRNA disease association GNN critique`,
  `diagonal vs full bilinear predictor faithfulness reimplementation`,
  `ReScience replication biomedical link prediction negative result`.
- **Ứng viên B (K-sparsity / right-sizing hypergraph):**
  `optimal K neighbors hypergraph GNN biomedical small graph`,
  `hypergraph sparsity over-smoothing MDA`,
  `right-sizing GNN capacity sparse biological network`.
- **Ứng viên C (metric bug / eval audit):**
  `metric implementation bug released code multi-class F1 biomedical`,
  `evaluation code audit reproducibility miRNA disease`,
  `hardcoded number of classes evaluation bug GNN`.

---

# TÓM TẮT SESSION 1 (cho người dùng duyệt — R11)

- **Đã làm:** 3 web search (giao thức, không cite khoa học) → thêm R14–R17; tạo `ledger.md`, `search_log.md`, `protocol_v1.md`; soạn pool truy vấn chi tiết Session 2–3–5.
- **Self-audit (R17):** 0 bài fetch, 0 VERIFIED, 0 LOẠI (chưa vào giai đoạn cite). Đúng kỳ vọng cho session giao thức.
- **Chưa làm (đúng thiết kế):** chưa cite/fetch bài MDA nào — để dành Session 2.
- **DỪNG chờ duyệt.** Lệnh tiếp: `Bắt đầu SESSION 2`.
