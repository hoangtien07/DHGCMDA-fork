# Chương 2 — Dữ liệu đầu vào & khái niệm "dual view"

> Paper §2.2–2.3 (Eq. 1–3). Chương này trả lời: model được cho ăn gì, và "view" nghĩa là gì.

---

## 2.1. Hai modality, mỗi modality hai view

"Modality" = loại thực thể sinh học (miRNA hoặc disease). "View" = một nguồn similarity mô tả các entity *cùng modality* tương tự nhau thế nào. Mỗi modality có **đúng 2 view**:

| | View 1 | View 2 | File trong repo |
|--|--------|--------|-----------------|
| **miRNA** | Sequence similarity $S^{seq}_m$ — giống nhau về trình tự nucleotide | Functional similarity $S^{func}_m$ — giống nhau về vai trò điều tiết | `M_GSM.txt`, `M_FSM.txt` (495×495) |
| **Disease** | Gene-based similarity $S^{gene}_d$ — chia sẻ gen liên quan | Semantic similarity $S^{sem}_d$ — giống nhau trên cây phân loại MeSH | `D_SSM2.txt`, `D_SSM1.txt` (383×383) |

Tại sao 2 view? Vì mỗi nguồn chỉ phản ánh một mặt của thực thể: hai miRNA giống trình tự chưa chắc cùng chức năng, và ngược lại. Paper gọi đây là **dual complementary views** — hai góc nhìn bổ sung.

> **Điểm mấu chốt với hạn chế #1 ở Chương 1**: cả 4 similarity này đều *không* được tính từ ma trận association A — chúng tới từ dữ liệu sinh học độc lập → phá vỡ circular dependency, và miRNA mới (chưa có association nào) vẫn có feature vì vẫn có trình tự.

## 2.2. Augmented feature: dán similarity + association lại với nhau (Eq. 1–2)

Với mỗi view $v \in \{1,2\}$, feature của một node = similarity của nó với các node cùng loại **nối thêm** association của nó với các node loại kia:

$$X^{(v)}_m = [\,S^{(v)}_m,\; A\,] \in \mathbb{R}^{n_m \times (n_m + n_d)} \quad (1)$$
$$X^{(v)}_d = [\,S^{(v)}_d,\; A^{T}\,] \in \mathbb{R}^{n_d \times (n_d + n_m)} \quad (2)$$

Với số liệu v2.0: miRNA view có feature dài $383 + 495 = 878$; disease view cũng dài $383 + 495 = 878$. Hai bên cùng độ dài là cố ý — phần "association" chính là *cùng một ma trận* A nhìn từ hai phía.

**Trực giác**: hàng $i$ của $X^{(v)}_m$ trả lời đồng thời "miRNA $i$ giống các miRNA nào?" (intra-modality) và "miRNA $i$ đang liên kết các bệnh nào?" (cross-modality). Như vậy mỗi node mang cả *bản sắc sinh học* lẫn *mẫu tương tác chức năng*.

> Trong code: `concat_mi_tensor_view1 = cat([association_matrix, m_ss])`, `concat_mi_tensor_view2 = cat([association_matrix, mi_fun])`, tương tự disease với `d_gs`, `dis_sem` — xem `main_experiments_hetero1.py:905-926`. Lưu ý thứ tự cột trong code là `[A, S]` chứ không phải `[S, A]` như paper viết — chỉ là khác thứ tự dán, toán tương đương.

## 2.3. Cách build siêu cạnh từ feature (Eq. 3)

Từ mỗi ma trận $X^{(v)}$ (4 cái: miRNA×2 view, disease×2 view), paper dựng một **hypergraph** bằng KNN:

- Với mỗi node $v_i$: tính cosine similarity với mọi node khác trên $X^{(v)}$, lấy $K$ hàng xóm gần nhất.
- Siêu cạnh $e_i = \{v_i\} \cup KNN(v_i, K)$ — tức **mỗi node sinh ra một hyperedge chứa nó + K láng giềng**.
- Ma trận incidence $H \in \{0,1\}^{n \times |E|}$: $H_{ij}=1$ nếu node $i$ thuộc hyperedge $j$. Vì mỗi node cho 1 hyperedge nên $|E| = n$ và mỗi cột có $K+1$ số 1 (theo paper; code đếm cả node trung tâm trong K láng giềng → mỗi cột thực tế $K$ số 1).

Kết quả: 4 hypergraph — `G_mi_view1`, `G_mi_view2`, `G_dis_view1`, `G_dis_view2`. Paper chọn $K=13$ sau khi sweep (Fig. 3: Top-1 metrics tăng dần đến K=13 rồi giảm nhẹ — quá ít láng giềng thì thiếu thông tin, quá nhiều thì nhiễu).

> Trong code: `hypergraph_construct_KNN.construct_H_with_KNN` — implementation dùng **khoảng cách Euclid** thay cosine similarity và thêm nhánh `is_probH` để cho weight mềm $\exp(-d^2/(m\cdot\bar d)^2)$ (mặc định tắt, nhị phân 0/1). Chi tiết chương 3.

## 2.4. Số liệu dataset (Table 1–2 của paper)

| Dataset | miRNA | Disease | Associations | Types | Mật độ |
|---------|-------|---------|--------------|-------|--------|
| HMDD v2.0 | 495 | 383 | 1,679 | 4 | ~0.9% |
| HMDD v3.2 | 411 | 271 | 11,748 | 5 | ~10.5% |

Phân bố type **rất mất cân bằng** — v2.0: Genetics 40.6% vs Epigenetics 11.9%; v3.2: Target 34.0% vs Epigenetics 3.4%. Đây là lý do paper phải dùng class-weighted loss (Chương 7).

> ⚠️ **Cảnh báo reproduce** (chi tiết Chương 9): số "411×271, 11,748" của v3.2 **không tái lập được** từ HMDD v3.2 raw (1,049×758, 18,084 triplets). Paper chỉ nói "loại entity có <2 association" nhưng con số đó không ra đúng → curation của tác giả chưa công bố. Repo này đã tổng hợp nhiều bản preprocess khác nhau (`v3.2_wang`, `v3.2_spld_paper_pipeline`, ...) để kiểm chứng.

---

**Chương tiếp**: [03 — Hypergraph & HGCN](03-hypergraph-va-hgcn.md) — siêu đồ thị là gì và tích chập trên nó hoạt động ra sao.
