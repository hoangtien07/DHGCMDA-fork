# Chương 1 — Bối cảnh sinh học & phát biểu bài toán

> Paper §1 (Introduction) + §2.2 (Problem Formulation). Đọc xong chương này bạn phải trả lời được: DHGCMDA dự đoán *cái gì*, và input/output nhìn như thế nào.

---

## 1.1. miRNA và bệnh — tối thiểu cần biết

**miRNA** (microRNA) là các đoạn RNA ngắn (~22 nucleotide) không mã hóa protein, nhưng điều tiết gen — chủ yếu bằng cách bám vào mRNA để *tắt* nó. Khi miRNA bị rối loạn (quá nhiều / quá ít), cơ thể dễ mắc bệnh — đặc biệt ung thư.

**Vấn đề thực tế**: có hàng nghìn miRNA × hàng nghìn bệnh. Kiểm chứng bằng thí nghiệm ướt (wet-lab) cho từng cặp rất tốn kém → cần phương pháp tính toán để *gợi ý* cặp nào đáng thí nghiệm.

## 1.2. Điểm mới của paper: dự đoán LOẠI liên kết, không chỉ có/không

Các phương pháp cũ hầu hết chỉ trả lời: "miRNA X có liên quan bệnh Y không?" — bài toán **binary link prediction**.

DHGCMDA trả lời câu khó hơn: "liên kết đó thuộc **cơ chế sinh học nào**?"

| Mã | Type | Ý nghĩa sinh học | Dataset |
|----|------|------------------|---------|
| 1 | Genetics | Biến thể di truyền/SNP ở vùng miRNA binding site | v2.0 + v3.2 |
| 2 | Epigenetics | miRNA điều tiết qua methyl hóa DNA / sửa histone | v2.0 + v3.2 |
| 3 | Target | miRNA trực tiếp bám mRNA của gen gây bệnh | v2.0 + v3.2 |
| 4 | Circulation | miRNA lưu hành trong máu/dịch cơ thể → biomarker | v2.0 + v3.2 |
| 5 | Tissue | miRNA biểu hiện đặc thù trong mô bệnh | **chỉ v3.2** |

**Tại sao quan trọng?** Ví dụ paper đưa: miR-146a vừa *thúc đẩy* ung thư dạ dày (qua target SMAD4) vừa *ức chế* nó (qua EGFR/IRAK1) — cùng 1 cặp miRNA–bệnh nhưng cơ chế khác nhau hoàn toàn. Dự đoán "có liên kết" là chưa đủ cho người làm nghiên cứu.

## 1.3. Phát biểu toán học (§2.2)

- $M = \{m_1..m_{n_m}\}$: tập miRNA; $D = \{d_1..d_{n_d}\}$: tập bệnh.
- Ma trận liên kết $A \in \mathbb{R}^{n_m \times n_d}$:
  - Binary prediction: $A_{ij} \in \{0,1\}$ — 1 = đã xác nhận liên kết.
  - Type prediction (task của DHGCMDA): $A_{ij} \in \{0,1,2,\dots,C\}$ — $C$ loại.
- **⚠️ Cạm bẫy quan trọng**: $A_{ij}=0$ **không** có nghĩa "không liên kết" — nó nghĩa "chưa được xác nhận". Đây là bài toán *positive-unlabeled*: tập "negative" thực ra là hỗn hợp negative thật + positive chưa khám phá. Hệ quả: paper sample ngẫu nhiên unknown pairs làm "negative" khi train (xem Chương 7).

→ Bài toán = **link + edge-type prediction** trên đồ thị sinh học, với label rất thưa (v2.0: 1,679 associations trên 495×383 ≈ 190 nghìn cặp — ~0.9% có nhãn).

## 1.4. Ba hạn chế của các phương pháp trước mà DHGCMDA muốn sửa

Paper lập luận các method cũ (RBM, label propagation, tensor factorization, GNN thường) mắc 3 lỗi:

1. **"Quantification bias"** — similarity giữa miRNA thường được *tính từ chính ma trận association* (vd. Gaussian kernel trên A). Tức là input feature vay từ thứ cần dự đoán → phụ thuộc vòng (circular dependency), và vô dụng với miRNA mới chưa có annotation nào.
   → *Fix của DHGCMDA*: dùng similarity sinh học thật (chuỗi, chức năng, gene, semantic) — Chương 2.
2. **Đồ thị pairwise không capture được quan hệ bậc cao** — cạnh thường chỉ nối 2 node, trong khi sinh học là nhóm miRNA cùng điều tiết.
   → *Fix*: hypergraph — Chương 3.
3. **Embedding không nhất quán giữa các view và modality** — mỗi nguồn similarity cho 1 "góc nhìn" khác; nếu không ép chúng thống nhất, thông tin bị rời rạc.
   → *Fix*: contrastive learning 2 tầng — Chương 4.

## 1.5. So sánh nhanh với các method cùng lĩnh vực

| Method | Hướng | Hạn chế chính (theo paper) |
|--------|-------|-----------------------------|
| RBMMMDA | RBM | Chỉ dùng association, không dùng similarity |
| NLPMMDA | label propagation | Mỗi type là task độc lập, bỏ qua tương quan giữa types |
| TDRC, WeightTDAIGN, SPLDHyperAWNTF, TFLP | tensor factorization | False negative, phức tạp tính toán, data sparsity |
| PDMDA, SGNNMD, NMCMDA, mDLinker, deepMDpred | GNN | Vấp 3 hạn chế ở §1.4 |

---

**Chương tiếp**: [02 — Dữ liệu & hai "view"](02-du-lieu-va-hai-view.md) — model nhìn miRNA/disease qua 4 ma trận similarity nào.
