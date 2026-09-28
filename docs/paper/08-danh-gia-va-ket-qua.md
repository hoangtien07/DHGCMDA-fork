# Chương 8 — Đánh giá & kết quả (paper §3)

> Paper §3.1–3.6 + Conclusion. Chương này giải thích *cách* paper chứng minh model tốt, và kết quả tới đâu.

---

## 8.1. Hai chế độ cross-validation — đo hai khả năng khác nhau (§3.2)

Paper dùng 5-fold CV với **hai cách chia khác nhau**, tương ứng hai mục tiêu:

| Setting | Chia gì | Test hỏi | Metrics | Đo khả năng |
|---------|---------|----------|---------|-------------|
| **CVtriplet** | Chia toàn bộ **triplet** (miRNA, disease, type) thành 5 phần; test = 1 phần + số unknown tương đương | "Cặp (m,d) này có liên kết type t không?" | AUPR, AUC, F1 | **Discovery** — tìm liên kết mới trong biển unknown |
| **CVtype** | Chia các *cặp có ≥1 type đã biết* thành 5 phần | "Biết là có liên kết — đoán đúng type không?" (argmax trên C loại) | Top-1 P / R / F1 | **Mechanism identification** — nhận diện đúng cơ chế |

> Điểm tinh tế: CVtype chỉ hỏi trên cặp *đã biết có liên kết* → đo trực tiếp phần mới của paper (type discrimination). CVtriplet đo bài toán discovery quen thuộc hơn.

## 8.2. Kết quả chính (Table 3–4)

So với 6 SOTA (TDRC, SPLDHyperAWNTF, TFLP, NMCMDA, MRFGMDA, KBLTDARD):

**CVtype** (Table 3):
- v3.2: DHGCMDA **Top-1 F1 = 0.8600** — vượt method tốt nhất tiếp theo (MRFGMDA 0.6335) **+35.8%**. Precision 0.7915, recall **0.9421**.
- v2.0: F1 = 0.5970 — đứng đầu về recall (0.6341) nhưng precision (0.5842) thua vài method → tổng F1 chỉ ngang nhóm đầu, không vượt trội như v3.2. Paper giải thích đây là trade-off có chủ đích của class weighting (ưu tiên recall cho bài toán biomedical).

**CVtriplet** (Table 4):
- v2.0: AUPR 0.9738, AUC 0.9669, F1 0.9278 — cao nhất toàn bảng.
- v3.2: AUPR 0.9271, F1 0.8674 — đầu bảng; AUC 0.9181 (KBLTDARD 0.9263 cao hơn nhẹ).

Hyperparam sensitivity (Fig. 2–3): $t{=}0.5$, $\lambda_2{=}0.3$ tối ưu; inverse-proportional class weighting tốt nhất; Top-1 đỉnh tại K=13.

## 8.3. Ablation (Fig. 4) — component nào quan trọng?

5 variant bỏ/thay từng khối, kết quả paper (pattern, không có số cụ thể trong text):

| Variant | Bỏ gì | Kết quả paper |
|---------|-------|---------------|
| w/o DV | dual-view → single-view | **Sụt nặng nhất** — 2 view bổ sung thông tin, mất 1 mất cả nhánh |
| w/o HGT | HGT → dùng fused embedding luôn | Sụt mạnh — type-aware message passing quan trọng |
| w/o HGCN | HGCN → GCN thường | Sụt rõ — high-order relation thắng pairwise |
| w/o CL | Bỏ cả intra+inter contrastive | Sụt — CL tăng discriminative power |
| w/o AVF | attention fusion → concat | Sụt nhẹ hơn các cái trên |

Thứ tự paper suy ra: DV > HGT > HGCN > CL > AVF về mức độ ảnh hưởng.

> ⚠️ **Reproduce note** (Chương 9): khi fork này chạy lại 5 ablation trên v2.0, thứ tự **ngược paper** — bỏ CL hay HGT lại *tăng* F1. Đã kiểm chứng 4 cách (additive switch, true rebuild, multi-seed, 2 loss mode) → finding thật, không phải bug implement.

## 8.4. Case study — chứng minh trên bệnh thật (§3.6)

- Train trên v1 full v2.0 rồi predict top-15 miRNA cho **breast neoplasms** và **hepatocellular carcinoma**, verify bằng PubMed.
- Breast: **13/15** (86.7%) được văn liệu xác nhận đúng type. HCC: **12/15** (80%).
- 5 "Unconfirmed" — paper lập luận đây có thể là **triplet mới** chưa ai khám phá (ứng viên cho wet-lab), chứ không hẳn lỗi.
- Enrichment analysis trên target genes của miRNA dự đoán cho breast cancer: các module EGFR/ERBB2/ERBB3 + TGFβ/SMAD + cell cycle — khớp với cơ chế đã biết của breast cancer.

## 8.5. Limitations paper tự thừa nhận (§4)

1. Phụ thuộc nhiều nguồn similarity — miRNA/disease mới mà data thưa → dual-view thoái hóa gần như single-view.
2. Chi phí scale theo số node + hyperedge — cần mini-batch/attention hiệu quả hơn cho đồ thị lớn.
3. Interpretability của embedding còn hạn chế.
4. Dynamic update dùng θ cố định + interval cố định — chưa chắc tối ưu mọi giai đoạn training.

Paper đề xuất mở rộng: drug–drug interaction type, lncRNA–disease, PU-learning cho negative samples, adaptive thresholding.

## 8.6. Số liệu tính toán (tham chiếu)

- ~2.55M parameters; RTX 4060 Ti 16GB; ~3.16 phút/fold trên v2.0; ~15.8 phút cả 5-fold; peak VRAM 3.68GB → model khá nhẹ.

---

**Chương tiếp**: [09 — Paper vs code trong repo này](09-paper-vs-code.md) — map từng công thức về file, các chỗ code ≠ paper, và kết quả reproduce thực tế.
