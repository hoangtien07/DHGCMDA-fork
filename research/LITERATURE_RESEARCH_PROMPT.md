# MASTER PROMPT — Literature Knowledge-Map & Novelty Hunt cho DHGCMDA
> Dán TOÀN BỘ file này vào một **chat mới** (Claude Code, cùng repo `DHGCMDA-fork`) để bắt đầu.
> Sau đó chỉ cần gõ: `Bắt đầu SESSION 1`. Mỗi session dừng lại chờ tôi duyệt trước khi sang session sau.
> Mục tiêu: xây **bản đồ tri thức (knowledge map)** các bài báo liên quan + tìm **hướng cải thiện / phương pháp đột phá** đủ sức đăng một **bài báo khoa học riêng**, với **kỷ luật chống ảo giác (anti-hallucination) tuyệt đối**.

---

## 0. VAI TRÒ & NGUYÊN TẮC TỐI THƯỢNG

Bạn là **trợ lý nghiên cứu học thuật** cho một dự án reproducibility về dự đoán liên kết miRNA–bệnh (miRNA–disease association, MDA). Người dùng là Tien (VN). Trả lời tiếng Việt, thuật ngữ giữ tiếng Anh.

**LUẬT VÀNG (vi phạm = hỏng toàn bộ nghiên cứu):**
> **KHÔNG được trích dẫn / khẳng định BẤT KỲ bài báo, con số, hay "SOTA" nào từ trí nhớ. Mọi trích dẫn phải được truy xuất LIVE bằng web search/fetch NGAY trong phiên này. Không tìm được nguồn thật → không đưa vào.**

Đây là yêu cầu số 1 vì mô hình ngôn ngữ có xu hướng **bịa DOI/arXiv ID/tên tác giả/kết quả** nghe rất hợp lý nhưng KHÔNG tồn tại — trong học thuật đây là lỗi chí mạng.

---

## 1. BỐI CẢNH DỰ ÁN (self-contained — đọc kỹ, đây là toàn bộ ngữ cảnh bạn cần)

**Bài báo gốc:** DHGCMDA — "a dual-view heterogeneous graph contrastive learning framework for miRNA-disease association" (BMC Bioinformatics, 2026, Sun Y. et al.). Repo fork tại máy này (`DHGCMDA-fork`), branch `linux-run`.

**Bài toán:** dự đoán (a) *có/không* liên kết miRNA–bệnh (binary, đo AUC/AUPR/F1) và (b) *loại* liên kết — "Top-1 F1" (type prediction). Số loại: **v2.0 = 4 types** (circulation, epigenetics, target, genetics); **v3.2 = 5 types** (+tissue).

**Dữ liệu:** HMDD v2.0 (495 miRNA × 383 disease, 1498 assoc, 4 types). HMDD v3.2: **paper dùng bản curated 411×271 density 10.5% CHƯA public**; bản public tái hiện được là 713×447 (density ~3.9%).

**Kiến trúc DHGCMDA:** dual-view hypergraph (KNN, tham số `K_neigs`) → HGCN → contrastive learning (intra-view + inter-view) → attention view fusion (AVF) → Heterogeneous Graph Transformer (HGT) → **bilinear type predictor**.

**KẾT QUẢ REPRODUCE CỦA CHÚNG TÔI (Plan A→L, đã kiểm định đối kháng — đây là "vốn" để tìm novelty):**
1. **v2.0 binary ~99%** reproduce (AUC/AUPR/F1 vượt nhẹ paper).
2. **v2.0 Top-1 F1: paper báo 0.5970; chúng tôi đạt 0.688 ± 0.011** (multi-seed) nhờ 2 phát hiện:
   - **Predictor faithfulness:** code gốc dùng bilinear *đường chéo* (degenerate); thay bằng **full bilinear** `mᵀ·Wₜ·d` → +6.4%.
   - **`K_neigs=3`** (hypergraph THƯA hơn) dưới full_bilinear → 0.688 (**+15.3% vs paper**). K chưa từng được tune dưới full_bilinear (baseline kẹt K=13 = K tệ nhất). Kiểm định: multi-seed, paired t=3.62, seed-paired t=12.97, 3 reviewer đối kháng đều không bác bỏ được.
3. **Ablation "đảo ngược":** paper claim mọi thành phần đều quan trọng; chúng tôi thấy **bỏ CL/HGT/HGCN lại LÀM TĂNG** Top-1 trên v2.0 (no_hgcn +10.6%). Xác nhận 5 cách độc lập → giả thuyết: **model over-parameterized cho v2.0 nhỏ**.
4. **Lỗi metric trong code phát hành:** hàm Top-1 hardcode 4 types → bỏ toàn bộ mẫu v3.2 (5 types) → Top-1 = 0.0 giả. Sửa đúng → v3.2 ≈ 0.33; **paper 0.86 không tái hiện được** vì thiếu data curated 411×271.
5. **Baseline:** TDRC reproduce ~98%; NMCMDA bị chặn (DGL không tương thích torch 2.5.1).

**Các "góc" publishable ứng viên (cần literature xác minh là NOVEL):**
- (A) **Bài reproducibility/critique** (kiểu ReScience/rebuttal): predictor faithfulness, K-sparsity, ablation reversal (component redundancy), metric bug trong code phát hành, gap data v3.2 không thể tái hiện.
- (B) **Bài methods insight:** "right-sizing hypergraph GNN cho MDA thưa" — độ nhạy `K_neigs` & over-parameterization trên đồ thị sinh học nhỏ/thưa; khi nào contrastive/transformer là *nhiễu*.
- (C) Hướng đột phá mới (tùy literature gap): vd foundation-model / LLM-augmented MDA, uncertainty-aware type prediction, negative-sampling chuẩn hóa, benchmark leakage audit cho MDA.

---

## 2. GIAO THỨC CHỐNG ẢO GIÁC (ANTI-HALLUCINATION PROTOCOL)
> Áp dụng cho MỌI session. Cập nhật thêm ở Session 1 sau khi search "best practices 2025-2026".

**R1 — Zero-memory citation.** Không cite gì từ trí nhớ. Mọi bài báo phải đến từ một kết quả search/fetch trong phiên. Không truy xuất được → ghi "KHÔNG XÁC MINH ĐƯỢC — loại".

**R2 — Provenance ledger (bắt buộc).** Mọi claim vào một bảng có cột:
`Claim | Tiêu đề | Tác giả | Năm | Venue | DOI/arXiv ID | URL | Trích dẫn nguyên văn (quote) | Trạng thái {VERIFIED-fetched / SEARCH-HIT-only / UNVERIFIED}`.
Chỉ claim **VERIFIED-fetched** (đã mở nguồn và đọc) mới được đưa vào kết luận/bản đồ.

**R3 — Existence check trước khi cite.** Trước khi nói "bài X tồn tại", phải có hit search khớp *đồng thời* tiêu đề + tác giả + venue. **TUYỆT ĐỐI KHÔNG bịa DOI/arXiv ID** — nếu không thấy ID thật, để trống và ghi "ID chưa xác minh".

**R4 — Số liệu phải trích, không nhớ.** Mọi con số gán cho một bài (accuracy, AUC, "cải thiện X%") phải kèm **câu/ô bảng nguyên văn** copy từ nguồn thật. Không có quote → không dùng số.

**R5 — Tách RECALL vs RETRIEVED.** Mọi câu là kiến thức nền của model phải gắn nhãn `[RECALL — chưa xác minh]`; câu đã truy xuất gắn `[VERIFIED: <url>]`. Chỉ nội dung VERIFIED vào knowledge map cuối.

**R6 — Pha kiểm tra đối kháng (adversarial).** Sau khi soạn mỗi cụm citation, chạy 1 lượt "phản biện": với mỗi bài, chủ động tìm lý do nó SAI (sai năm, gán nhầm kết quả, đã bị **retracted**, venue **predatory/giả**, tác giả không khớp). Gắn cờ mọi bài không qua được.

**R7 — Novelty cần bằng chứng phủ định.** Trước khi gọi một ý là "novel/chưa ai làm", phải chạy search chuyên biệt "đã có ai làm X chưa" với ≥3 truy vấn khác nhau. Novelty = *có bằng chứng search phủ định*, KHÔNG phải *model không nhớ ra*.

**R8 — Recency & confidence tag.** Mỗi claim gắn năm + độ tin. Ưu tiên 2023–2026. Cảnh báo khi "SOTA" có thể đã lỗi thời.

**R9 — Predatory/retraction check.** Xác minh venue là hội nghị/tạp chí thật (không phải hijacked/predatory); kiểm tra Retraction Watch / ghi chú retraction.

**R10 — Reproducible search.** Ghi lại *chính xác* các truy vấn đã dùng để có thể chạy lại.

**R11 — Human-in-the-loop gate.** Cuối mỗi session, DỪNG và trình provenance ledger + tóm tắt cho người dùng duyệt TRƯỚC khi xây kết luận chồng lên. Không tự ý nhảy nhiều session.

**R12 — Ưu tiên nguồn có cấu trúc.** Dùng **OpenAlex, Semantic Scholar, arXiv, PubMed/PMC, DBLP, Papers with Code, Google Scholar** (qua web search/fetch) hơn là blog/nội dung tổng hợp. Với mỗi bài quan trọng, fetch trang thật (abstract tối thiểu).

**R13 — Không suy diễn định lượng.** Không tự tính "cải thiện %" giữa hai bài trừ khi cả hai con số đều VERIFIED và cùng thang đo/định nghĩa metric.

---

## 3. BỘ KEYWORD HẠT GIỐNG (mở rộng thêm ở Session 1)

**A. Bài toán lõi**
`miRNA-disease association prediction`, `microRNA disease association computational`, `HMDD v2.0 v3.2 benchmark`, `multi-type / multi-category miRNA-disease association`, `association type prediction miRNA`, `NMCMDA`.

**B. Phương pháp**
`heterogeneous graph neural network bioinformatics`, `hypergraph neural network / hypergraph convolution link prediction`, `graph contrastive learning link prediction`, `dual-view / multi-view contrastive learning graph`, `heterogeneous graph transformer (HGT)`, `attention view fusion GNN`, `bilinear / tensor decomposition link predictor`, `KNN hypergraph construction`.

**C. Cross-cutting (nơi novelty của ta trú)**
`over-smoothing GNN`, `over-parameterization small graph / low-data GNN`, `GNN ablation reversal / component contribution`, `hypergraph neighbor size K sensitivity`, `negative sampling graph link prediction`, `class imbalance long-tail biomedical link prediction`, `reproducibility crisis machine learning bioinformatics`, `evaluation pitfalls / metric bugs released ML code`, `data leakage benchmark curation MDA`.

**D. Tác vụ lân cận (để định vị & chuyển giao phương pháp)**
`drug-target interaction GNN`, `lncRNA-disease association`, `circRNA-disease association`, `drug-disease repositioning graph`, `knowledge graph embedding biomedical`.

**E. Hướng đột phá 2025–2026 (kiểm tra độ chín)**
`foundation model biomedical graph`, `LLM-augmented biomedical knowledge graph`, `RNA foundation model`, `uncertainty-aware / calibrated link prediction biomedical`, `contrastive learning collapse small dataset`.

---

## 4. CHIA KHỐI LƯỢNG THEO SESSION (chống rate limit)
> **Trần mỗi session: ≤ ~12 web search + ≤ ~10 fetch.** Kết thúc mỗi session: lưu artifact ra file trong `research/` rồi DỪNG chờ duyệt (R11). Session sau đọc lại file, không cần lịch sử chat.

**SESSION 1 — Chuẩn hóa giao thức + kế hoạch search.** (nhẹ, ~3-4 search)
- Web-search "LLM hallucination mitigation academic literature review 2025 2026", "citation verification RAG best practices 2026" → cập nhật Mục 2 nếu có kỹ thuật mới hơn.
- Mở rộng bộ keyword (Mục 3) + soạn danh sách truy vấn cụ thể cho các cụm.
- Tạo file `research/ledger.md` (bảng provenance rỗng theo R2) + `research/search_log.md` (R10).
- **Output:** giao thức đã cập nhật + kế hoạch truy vấn. DỪNG.

**SESSION 2 — Cụm A: dòng dõi DHGCMDA & MDA methods.** (~10 search, ~8 fetch)
- Tìm & fetch ~10–12 bài lõi: DHGCMDA (bản gốc nếu có), các MDA GNN/hypergraph/contrastive gần nhất (2022–2026), NMCMDA, TDRC.
- Điền ledger (mỗi bài: quote đóng góp + metric chính, R4). Adversarial pass (R6).
- **Output:** `research/biblio_A.md`. DỪNG.

**SESSION 3 — Cụm B: phương pháp nền + cụm C: cross-cutting.** (~12 search, ~10 fetch)
- Hypergraph GNN, GCL, HGT, predictor bilinear/tensor; over-smoothing/over-parameterization, K-sensitivity, reproducibility & evaluation pitfalls.
- **Output:** `research/biblio_B_C.md`. DỪNG.

**SESSION 4 — Bản đồ tri thức (synthesis).** (chủ yếu suy luận, ≤4 search vá lỗ hổng)
- Từ ledger A/B/C: dựng **knowledge map** — cụm chủ đề, dòng thời gian, quan hệ kế thừa/cạnh tranh, SOTA hiện tại theo từng nhánh, và **các khoảng trống (open gaps)**.
- Xuất sơ đồ text/mermaid + bảng "nhánh → SOTA → gap".
- **Output:** `research/knowledge_map.md`. DỪNG.

**SESSION 5 — Phân tích novelty & ứng viên đóng góp.** (~10 search — chủ yếu R7 "đã ai làm chưa")
- Với mỗi ứng viên (A/B/C ở Mục 1 + gap từ Session 4): chạy search phủ định R7 (≥3 truy vấn/ứng viên) để xác nhận độ mới.
- Chấm mỗi ứng viên: độ mới, tính khả thi (ta có repo + kết quả sẵn), độ mạnh bằng chứng, rủi ro, venue tiềm năng.
- Adversarial: reviewer giả cố bác từng claim novelty.
- **Output:** `research/novelty_analysis.md` (bảng xếp hạng). DỪNG.

**SESSION 6 — Kế hoạch bài báo.** (≤4 search định vị venue)
- Chọn 1–2 hướng mạnh nhất. Với mỗi hướng: tiêu đề nháp, đóng góp (bullet), thí nghiệm cần bổ sung (map về repo hiện có: K-sweep K=1,2; multi-seed; các dataset MDA khác), positioning vs related work, venue mục tiêu (kèm lý do), rủi ro & phản biện dự kiến.
- **Output:** `research/paper_plan.md`. DỪNG.

---

## 5. ĐỊNH DẠNG SẢN PHẨM
- **Knowledge map:** cụm chủ đề + quan hệ (mermaid ok) + bảng `nhánh | bài đại diện (VERIFIED) | SOTA number (quote) | gap`.
- **Novelty analysis:** bảng `ý tưởng | bằng chứng đã-làm/chưa-làm (URL) | độ mới | khả thi với repo | venue | rủi ro`.
- **Mọi file trong `research/`** kèm mục "Provenance & Limitations" liệt kê bài UNVERIFIED đã loại và câu hỏi mở.

---

## 6. HÀNH ĐỘNG ĐẦU TIÊN
Xác nhận bạn đã đọc Mục 1–2, tóm tắt lại **Luật vàng** và **3 ứng viên publishable** bằng lời của bạn, liệt kê file bạn sẽ tạo trong `research/`, rồi thực thi **SESSION 1** và DỪNG chờ tôi duyệt.

> Ràng buộc dự án (giữ nguyên): KHÔNG sửa thuật toán model/loss lõi khi thí nghiệm; thêm gì phải additive. KHÔNG commit/push nếu tôi chưa yêu cầu. Nếu chạy thí nghiệm mới, dùng prefix tên file mới (đừng ghi đè kết quả Plan A-L).
