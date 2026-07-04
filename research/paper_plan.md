# Paper Plan — Bài A+B kết hợp (SESSION 6, 2026-07-05)

> Hướng đã chốt với user: **A+B kết hợp** — một bài *reproducibility/critique* (A) đồng thời rút ra *bài học phương pháp chuyển giao được* (B: right-sizing hypergraph GNN cho MDA thưa).
> Mọi số/claim dựa trên ledger VERIFIED + kết quả repo Plan A–L. Prior art định vị đã qua R7 (Session 5) + verify (Session 6).

---

## 1. TIÊU ĐỀ NHÁP (chọn 1)
1. **"DHGCMDA Revisited: A Broken Metric, an Over-Parameterized Model, and the Case for Right-Sizing Hypergraph GNNs in Sparse miRNA–Disease Type Prediction"**
2. "Revisiting Dual-View Hypergraph Contrastive Learning for miRNA–Disease Type Prediction: Faithful Reimplementation, Metric Pitfalls, and Capacity Right-Sizing"
3. (ReScience-style) "[Re] DHGCMDA: a dual-view heterogeneous graph contrastive learning framework for miRNA–disease association type prediction"

→ Khuyến nghị #1 (nêu bật cả A critique lẫn B insight).

## 2. LUẬN ĐIỂM TRUNG TÂM (thesis)
Một reimplementation **trung thực** của DHGCMDA (i) phát lộ một **lỗi metric trong code phát hành** làm sập đánh giá v3.2, (ii) cho thấy **predictor đúng như mô tả paper (full bilinear) VƯỢT cả bản gốc**, và (iii) chứng minh mô hình **over-parameterized cho HMDD v2.0 nhỏ/thưa** — bỏ bớt thành phần & làm hypergraph THƯA hơn lại tốt hơn. Từ đó rút ra nguyên tắc **right-sizing capacity cho đồ thị sinh học thưa**.

## 3. ĐÓNG GÓP (bullets — mỗi cái map tới bằng chứng)

| # | Đóng góp | Bằng chứng repo | Nền literature (VERIFIED) |
|---|---|---|---|
| C1 (A) | **Metric bug**: `compute_top1_metrics` hardcode 4 types → bỏ 100% mẫu v3.2 (5 types) → Top-1 F1=0.0 GIẢ. Chứng minh 3 cách (synthetic-perfect; tương đương 4-type; đo lại 0.27–0.33) | Plan K, `run_v32_correct_metric.py` | MassSpecGym-in-the-Wild (metric divergence, 17/26 papers); Kapoor taxonomy |
| C2 (A) | **Predictor faithfulness**: code dùng bilinear ĐƯỜNG CHÉO (degenerate); full bilinear mᵀWₜd → v2.0 Top-1 F1 **+6.4%**, VƯỢT paper | Plan J-1, `--predictor_mode full_bilinear` | SimplE ("fully expressive"); DistMult diagonal=symmetric [SH] |
| C3 (A+B) | **K-sparsity right-sizing**: K_neigs=3 → **0.688±0.011** (+15.3% vs paper); K=13 (default) là K tệ nhất; monotone K13<K7<K3 | Plan L, multi-seed, `run_council_matrix.sh` | Oversmoothing survey; K-sens link-pred [SH] |
| C4 (A+B) | **Ablation reversal / over-param**: bỏ HGCN/HGT/CL đều +8–11% v2.0; verify 5 cách (additive + rebuild + 4-seed + 2 loss + full_bilinear) | Plan E/F/H/L | Untrained-MP ("superior…"); Oversmoothing |
| C5 (A) | **v3.2 0.86 không tái hiện** (public best ~0.33; curated 411×271 ẩn) + **baseline lệch có hệ thống** (own-paper ≠ bảng DHGCMDA) | Plan K + knowledge_map O2 | Kapoor; consistent-eval-2020 [SH]; TDRC/KBLTDARD own numbers |
| C6 (B) | **Nguyên tắc chuyển giao**: "right-sizing hypergraph capacity cho MDA thưa — khi nào CL/transformer là NHIỄU" | tổng hợp C2–C4 | PROXI, GAE-reconsidered, Untrained-MP |

**Headline number:** v2.0 Top-1 F1 **0.5970 (paper) → 0.688±0.011 (ta)**, vượt mọi public baseline (KBLTDARD 0.5869, TFLP 0.5996, TDRC 0.5286).

## 4. THÍ NGHIỆM CẦN BỔ SUNG (map về repo — additive, KHÔNG sửa model/loss lõi)

| Ưu tiên | Thí nghiệm | Vì sao | Chi phí | Map repo |
|---|---|---|---|---|
| **P0 (bắt buộc cho B)** | K-sweep **K=1,2** dưới full_bilinear (chưa thử — CLAUDE.md ghi) | Hoàn tất đường cong K, xác nhận đáy | ~2 fold × vài run | `run_council_matrix.sh` mở rộng grid |
| **P0 (nâng B từ critique→insight)** | Replicate right-sizing trên **≥1 dataset MDA/analog khác** (HGCLAMIR MDAv2.0 split, hoặc lncRNA/circRNA-disease) | Cho thấy finding TỔNG QUÁT, không riêng HMDD v2.0 | 1 dataset × K-sweep + ablation | cần adapter data mới (additive) |
| **P1 (cơ chế C3/C4)** | Diagnostic over-smoothing: đo similarity embedding theo K & #layers | Bằng chứng CƠ CHẾ, không chỉ tương quan | rẻ (post-hoc trên checkpoint) | script mới `measure_oversmoothing.py` |
| **P2 (mạnh C5)** | Cố định split + recompute similarity chỉ từ train (chống leakage kiểu consistent-eval-2020) | Chặn phản biện "số của anh cũng leak" | vừa | audit pipeline hiện có |
| P3 (tùy) | C-uq mở màn: conformal post-hoc trên score type → prediction set + coverage | Hạt giống bài 2 / future work | vừa | post-hoc, additive |

**Ràng buộc giữ nguyên:** prefix tên file mới, KHÔNG ghi đè Plan A–L, KHÔNG commit/push khi user chưa yêu cầu.

## 5. POSITIONING vs RELATED WORK (đã VERIFIED trừ [SH])

```
Đối tượng phê bình:  DHGCMDA (Sun+ 2026, BMC Bioinf)  ← ta reimplement trung thực
Parallel art cùng lab: MHNNMDA (Sun+ 2026, JCAMD)     ← thừa nhận, không trùng đóng góp
Baseline định vị:     TDRC(2020), NMCMDA(2021), KBLTDARD(2024)  ← số VERIFIED
Prior art phê bình MDA: consistent-eval-2020 [SH]  → họ BINARY+similarity-leakage; TA type+metric-bug+faithfulness+K
Precedent metric-audit: MassSpecGym-in-the-Wild(2026, VERIFIED)  → domain molecule; TA cung cấp instance MDA
Khung reproducibility:  Kapoor-Narayanan(2022/23, VERIFIED)      → 8-type leakage taxonomy
Precedent "simplification wins": PROXI[SH], GAE-reconsidered[SH], Untrained-MP(VERIFIED)
Nền lý thuyết:          SimplE(expressiveness), HGNN, HGT, Oversmoothing (đều VERIFIED)
```

**Câu phân biệt then chốt** (chống reviewer "consistent-eval-2020 làm rồi"):
> *"Prior reproducibility work on MDA (consistent-eval-2020) targets **binary** association and **feature-similarity leakage**. We are the first to audit **multi-type** MDA prediction, exposing a **code-level metric bug**, a **predictor-faithfulness gap**, and **capacity over-parameterization** — orthogonal axes that leakage-focused audits do not touch."*

## 6. VENUE (đã verify scope)

| Venue | Khớp A+B? | Rủi ro | Ghi chú (VERIFIED scope) |
|---|---|---|---|
| **Briefings in Bioinformatics** (PRIMARY) | ✅ cao | vừa | Khuyến khích *"independent evaluation of software tools… by authors who are not originators of the software"* + methodological insight → khớp cả A lẫn B |
| **ReScience C** (FALLBACK/song song) | ✅ phần A | thấp | Scope: reimplement methods + document obstacles + partial reproduction. Khớp HOÀN HẢO lõi A; nhưng ít trọng "methods insight" → dùng nếu tách riêng B |
| GigaScience | ✅ | vừa | data/reproducibility-friendly |
| BMC Bioinformatics (Matters Arising/comment tới DHGCMDA) | ✅ phần A | vừa-cao | cùng nơi DHGCMDA đăng; đối đầu trực diện |

**Khuyến nghị:** nhắm **Briefings in Bioinformatics** (A+B đầy đủ). Nếu reviewer đòi thu hẹp, tách lõi replication về **ReScience C**.

## 7. RỦI RO & PHẢN BIỆN DỰ KIẾN (từ pha đối kháng Session 5 + mới)

| Rủi ro | Giảm thiểu |
|---|---|
| "Metric bug lặt vặt" | Precedent MassSpecGym (cả 1 bài về đúng loại lỗi này); chứng minh nó **sập hoàn toàn** kết luận v3.2 |
| "K=3 chỉ là tuning" | Cơ chế over-smoothing + K=13 là K tệ nhất + kiểm định thống kê + diagnostic P1 |
| "Ablation = noise" | 5 cách verify độc lập, 8/8 delta dương |
| "Anh reproduce sai" | TDRC ~98% cùng pipeline; không dòng nào đạt gần 0.86 |
| "consistent-eval-2020 đã làm" | Câu phân biệt Mục 5 (trục khác hẳn) |
| **"DHGCMDA in-press unedited — số sẽ đổi"** | Nêu rõ dùng bản in-press (accessed 2026-07-05); cam kết re-verify khi final; số v2.0 0.5970 + v3.2 0.86 khớp bản PDF ta có |
| **Số baseline ta trích từ own-paper ≠ bảng DHGCMDA** | Đây CHÍNH là finding C5 (không nhất quán) — trình bày minh bạch cả hai nguồn |

## 8. CHECKLIST BẮT BUỘC TRƯỚC KHI NỘP (từ Provenance & Limitations toàn dự án)
- [ ] **Verify consistent-eval-2020** (bioRxiv 403 × 4 lần) — tìm published version / mirror để phân biệt chính xác. **Rủi ro novelty #1.**
- [ ] Fetch số Top-1 của **TFLP & SPLDHyperAWNTF** nếu đưa vào bảng SOTA đầy đủ (hiện [SH]).
- [ ] Re-verify **DHGCMDA final published** (hiện "Article in Press, unedited") — số/bảng có thể đổi.
- [ ] Verify **PROXI (2410.01802) & GAE-reconsidered (2411.03845)** abstract nếu cite làm precedent B.
- [ ] Lấy số **MHNNMDA** (paywall) nếu cần so sánh định lượng — hoặc giữ metadata-only.
- [x] MassSpecGym-in-the-Wild — VERIFIED (2606.19624).
- [x] Kapoor, SimplE, HGNN, HGT, Oversmoothing, KBLTDARD, Untrained-MP — VERIFIED.

---

## 9. KHUNG CẤU TRÚC BÀI (draft outline)
1. **Introduction** — MDA type prediction; DHGCMDA & dòng dõi; khủng hoảng tái lập (Kapoor); đóng góp C1–C6.
2. **Background** — hypergraph conv (HGNN), HGT, bilinear predictor & expressiveness (SimplE), HMDD v2.0/v3.2, CV_type/CV_triplet.
3. **Faithful reproduction setup** — môi trường (Linux port), data, reimplementation trung thực, đo lại baseline.
4. **Finding 1 — the metric bug** (C1) + cách đo đúng.
5. **Finding 2 — predictor faithfulness** (C2).
6. **Finding 3 — right-sizing: K-sparsity & ablation reversal** (C3+C4) + over-smoothing diagnostic.
7. **Finding 4 — the v3.2 gap & benchmark inconsistency** (C5).
8. **Discussion** — right-sizing lesson (C6/B); lời kêu gọi benchmark multi-type MDA chuẩn hoá; giới hạn.
9. **Conclusion.**

---

## 10. Provenance & Limitations (bài & toàn cung nghiên cứu 6-session)
- Kế hoạch dựa trên **13 bài VERIFIED** + kết quả repo Plan A–L (đã kiểm định đối kháng nội bộ). Prior art [SH] (consistent-eval-2020, TFLP/SPLD numbers, PROXI/GAE) **phải verify trước nộp** — đã liệt kê Mục 8.
- consistent-eval-2020 là **giới hạn novelty lớn nhất chưa gỡ hoàn toàn** (biorxiv chặn bot); phân biệt hiện dựa trên nhiều snippet search nhất quán (binary + similarity-leakage), CHƯA đọc full — không được tuyên bố "ta đầu tiên" tuyệt đối cho tới khi đọc full.
- Điểm mạnh nhất & rẻ nhất để publish = **lõi A** (đã có sẵn mọi thứ). Nâng lên A+B cần P0 (K=1,2 + 1 dataset khác) — vẫn additive, khả thi.
- Không dùng bất kỳ nguồn LOẠI (GRENADE) nào trong kế hoạch.

---

# ✅ KẾT THÚC CUNG 6-SESSION
File `research/`: `protocol_v1.md`, `ledger.md`, `search_log.md`, `biblio_A.md`, `biblio_B_C.md`, `knowledge_map.md`, `novelty_analysis.md`, `paper_plan.md`.
Tổng: **13 bài VERIFIED · 6 SEARCH-HIT cần verify · 1 LOẠI (GRENADE)**. Luật vàng giữ nguyên xuyên suốt: 0 citation từ trí nhớ.

---

## CẬP NHẬT 2026-07-05 — Bài 2 (C-uq conformal): PROOF-OF-CONCEPT XONG trên v2.0

Branch `breakthrough-conformal`. Uncertainty-aware type prediction bằng split-conformal (APS) — post-hoc, không train lại. **APS @90%: coverage 0.924, set-size 2.29/4; @95%: 0.966, 2.91/4.** Negative-control set-size → 3.57≈C xác nhận model info thật. Chi tiết `novelty_analysis.md` (cập nhật cùng ngày). C-uq từ "future work" → **có proof-of-concept**. Còn lại để thành bài đầy đủ: (i) v3.2 5-type (chờ user), (ii) so calibration methods (post-hoc temperature), (iii) coverage plots đa-α, (iv) positioning conformal-in-biomedicine.
