# Biblio B+C — Phương pháp nền (B) & Cross-cutting (C) (SESSION 3, 2026-07-05)

> B = nền kiến trúc DHGCMDA. C = nơi novelty của ta trú (over-smoothing, over-param, K-sensitivity, reproducibility, metric bug).
> Chỉ ghi VERIFIED (đã fetch/đọc). Claim chưa fetch được → SEARCH-HIT-only, KHÔNG dùng làm citation cứng.

---

## PHẦN B — PHƯƠNG PHÁP NỀN (kiến trúc DHGCMDA kế thừa)

### B1. HGNN — Hypergraph Neural Networks — **VERIFIED-fetched (arXiv abs)**
- **Tác giả:** Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao
- **Năm/Venue:** AAAI 2019 · **arXiv:** 1809.09401
- **Quote đóng góp:** *"a hyperedge convolution operation is designed to handle the data correlation during representation learning"*; học "considering the high-order data structure".
- **Vai trò với ta:** đây là gốc của **HGCN** trong DHGCMDA. Khi ta ablate `no_hgcn` (thay G bằng identity) là degrade chính operation này. Over-smoothing của hyperedge convolution ↔ phát hiện K_neigs.

### B2. HGT — Heterogeneous Graph Transformer — **VERIFIED-fetched (arXiv abs)**
- **Tác giả:** Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun
- **Năm/Venue:** WWW 2020 · **arXiv:** 2003.01332
- **Quote đóng góp:** *"We design node- and edge-type dependent parameters to characterize the heterogeneous attention over each edge"*.
- **Số liệu:** *"consistently outperforms all the state-of-the-art GNN baselines by 9%–21% on various downstream tasks"* (Open Academic Graph, 179M nodes / 2B edges).
- **Vai trò với ta:** HGT được thiết kế cho **web-scale** (179M nodes). DHGCMDA áp lên HMDD v2.0 (~495×383, ~1.5K assoc) — **lệch 5–6 bậc độ lớn**. Trực tiếp chống lưng giả thuyết "HGT là over-kill/nhiễu cho đồ thị MDA nhỏ" (ablation `no_hgt` +8.2% của ta).

### B3. SimplE — full expressiveness của link predictor — **VERIFIED-fetched (arXiv abs)**
- **Tác giả:** Seyed Mehran Kazemi, David Poole
- **Năm/Venue:** NeurIPS 2018 · **arXiv:** 1802.04868
- **Quote:** *"SimplE is fully expressive"* (motivation: nhiều bilinear model KHÔNG fully expressive).
- **Vai trò với ta:** nền lý thuyết cho phát hiện **full bilinear > diagonal bilinear (+6.4%)**. Predictor đường chéo (kiểu DistMult) hạn chế biểu diễn; full-matrix biểu diễn được quan hệ bất đối xứng của *type*. (Chi tiết DistMult-diagonal-symmetric ở SEARCH-HIT S3-x — cần anchor mạnh hơn nếu đưa vào bài.)

---

## PHẦN C — CROSS-CUTTING (chống lưng novelty của ta)

### C1. Survey on Oversmoothing in GNNs — **VERIFIED-fetched (arXiv abs)**
- **Tác giả:** T. Konstantin Rusch, Michael M. Bronstein, Siddhartha Mishra
- **Năm:** 2023 · **arXiv:** 2303.10993
- **Quote định nghĩa:** over-smoothing = *"the exponential convergence of suitable similarity measures on the node features"*; *"Node features of GNNs tend to become more similar with the increase of the network depth."*
- **Vai trò với ta:** khung lý thuyết cho **K_neigs=3**. Hypergraph dày (K lớn) ↑ kết nối ↑ khuếch tán ↑ over-smoothing → embedding đồng nhất → mất phân biệt type. K nhỏ = giảm over-smoothing trên đồ thị thưa v2.0. (Lưu ý: survey không nói thẳng quan hệ với *sparsity* — ta chỉ dùng nguyên lý depth/connectivity → over-smoothing.)

### C2. Leakage & the Reproducibility Crisis in ML-based Science — **VERIFIED-fetched (arXiv abs)** ⭐ anchor angle A
- **Tác giả:** Sayash Kapoor, Arvind Narayanan
- **Năm/Venue:** arXiv 2022 (2207.07048); công bố **Patterns (Cell Press) 2023** (cell.com bản full chặn bot, nhưng arXiv xác nhận cùng bài).
- **Quote thesis:** *"Data leakage is indeed a widespread problem and has led to severe reproducibility failures."*
- **Số liệu:** *"329 papers across 17 fields"*; *"a fine-grained taxonomy of 8 types of leakage that range from textbook errors to open research problems"*.
- **Vai trò với ta:** citation nền cho **toàn bộ angle A** (reproducibility/critique). Đặt phát hiện của ta (metric bug bỏ mẫu v3.2, số baseline không nhất quán, data curation ẩn) vào bối cảnh khủng hoảng tái lập ML rộng hơn — nâng tầm từ "bug 1 repo" → "instance của vấn đề hệ thống đã được ghi nhận".

---

## PHẦN D — SEARCH-HIT-ONLY (đồng thuận đa nguồn, CHƯA fetch anchor — không dùng làm citation cứng)

| Chủ đề | Đồng thuận từ search | Chống lưng finding | Cần làm |
|---|---|---|---|
| **DistMult diagonal = symmetric-only, kém biểu diễn hơn RESCAL full-matrix** | ≥3 nguồn: "DistMult forces relations to be symmetric", "RESCAL full-rank matrix", "diagonal reduces params nhưng mất expressiveness" | full_bilinear > diagonal (+6.4%) | Fetch anchor: RESCAL (Nickel 2011) hoặc ComplEx (Trouillon 2016) |
| **KNN K-sensitivity: K nhỏ tối ưu cho link prediction (K≈2–4), K lớn thêm nhiễu/bias** | ≥3 nguồn search hội tụ: "optimal around 4 neighbors", "K range 2 to 4 generally good for link prediction", "large k … bias covering important characteristics" | K_neigs=3 tốt nhất | Fetch 1 anchor sạch (mdpi 2227-7390/9/8/830 bị 403; thử nguồn khác) |
| **Over-parameterization/low-data: bỏ component có thể GIẢM overfitting, TĂNG hiệu năng** | GRENADE (nghi "bỏ CL cải thiện link prediction") — ❌ **KHÔNG verify được, đã LOẠI** | ablation reversal (no_hgcn/hgt/cl +8–11%) | Tìm anchor khác (over-param GNN low-data) — Session 5 |
| **Data leakage benchmark biomedical link prediction (KGE) làm phồng metric** | bioRxiv 2025.01.23.634511 (Benchmarking Impact of Data Leakage on Biomedical Link Prediction) — biorxiv chặn bot | data curation / benchmark MDA thiếu chuẩn | Fetch qua mirror hoặc chấp nhận metadata-only |

---

## PHẦN E — ADVERSARIAL PASS (R6/R9/R15)

| Bài | Precision (tồn tại?) | Relevance | Temporality | Retraction/predatory | Verdict |
|---|---|---|---|---|---|
| B1 HGNN | ✅ arXiv+GitHub iMoonLab+AAAI | ✅ nền HGCN | 2019 (canonical) | AAAI (top-tier) | GIỮ |
| B2 HGT | ✅ arXiv+ACM DL+GitHub pyHGT | ✅ nền HGT | 2020 (canonical) | WWW (top-tier) | GIỮ |
| B3 SimplE | ✅ arXiv | ✅ nền expressiveness | 2018 | NeurIPS (top-tier) | GIỮ (dùng cho luận điểm, không cho số) |
| C1 Oversmoothing survey | ✅ arXiv, tác giả Bronstein (uy tín) | ✅ cơ chế K | 2023 | survey (chưa venue cụ thể — arXiv) | GIỮ |
| C2 Kapoor leakage | ✅ arXiv+Patterns | ✅ anchor angle A | 2022/23 | Patterns/Cell Press (uy tín) | GIỮ |
| GRENADE (ablation reversal) | ❌ fetch trả SAI title ("Graph Neural Collapse…") — không đọc được PDF | — | — | — | **LOẠI (unverifiable)** |

**Bài học R1/R6 session này:** summary của WebSearch gán cho GRENADE claim "bỏ CL cải thiện link prediction" — khi fetch để xác minh, model đọc PDF hỏng và **bịa một title khác hoàn toàn**. Đúng kịch bản LUẬT VÀNG cảnh báo → kiên quyết loại, KHÔNG đưa vào bất kỳ kết luận nào.

---

## PHẦN F — SELF-AUDIT (R17) SESSION 3
- Web search: **8**. Fetch thử: **~13** (nhiều bị 403/PDF-binary).
- **VERIFIED-fetched: 5** (HGNN, HGT, SimplE, Oversmoothing survey, Kapoor-leakage).
- **SEARCH-HIT consensus (chưa anchor): 4 chủ đề** (DistMult expressiveness, K-sensitivity, over-param ablation, biomedical leakage).
- **LOẠI vì không verify được: 1** (GRENADE).
- Nguồn chặn bot: cell.com, biorxiv.org, mdpi.com → dùng arXiv abs thay thế khi có.

## PHẦN G — Provenance & Limitations
- **Trang PDF (arXiv/pdf) hay trả binary-corrupt cho fetch** → luôn ưu tiên trang `arxiv.org/abs/` (HTML) để lấy metadata + abstract. Không đọc được PDF thì KHÔNG bịa nội dung (đã tránh ở GRENADE, HGT-pdf).
- 4 chủ đề SEARCH-HIT cần anchor fetched trước khi vào bài chính thức → xử lý ở Session 5 (novelty) hoặc Session 6.
- Chưa đọc phần Method/Data của PDF DHGCMDA (K_neigs mặc định, số types, định nghĩa CV_type/CV_triplet) — có sẵn cục bộ `_pdf_text/`, để Session 4 đối chiếu khi dựng knowledge map.
- Bản đồ tri thức (Session 4) chỉ được dùng **5 bài VERIFIED session này + 6 bài Cụm A** = **11 bài** làm xương sống; các SEARCH-HIT ghi rõ nhãn.
