# DHGCMDA Documentation

Tài liệu hướng dẫn dự án DHGCMDA (tiếng Việt).

## 📚 Danh mục

| File | Nội dung | Khi nào đọc |
|------|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | ASCII diagram toàn bộ pipeline, file → function map | **Đọc đầu tiên** — lấy cái nhìn tổng quan |
| [NOTES_DATAFLOW.md](NOTES_DATAFLOW.md) | Data flow chi tiết từng bước, shape tensor | Đọc khi muốn hiểu dữ liệu di chuyển thế nào |
| [NOTES_MODEL.md](NOTES_MODEL.md) | Giải thích từng class + công thức toán | Đọc khi muốn hiểu sâu về kiến trúc |
| [CHEATSHEET.md](CHEATSHEET.md) | Lệnh chạy, hyperparams, troubleshooting, prompt templates | Tham chiếu nhanh khi vọc code |
| [RESULTS_PLAN_M.md](RESULTS_PLAN_M.md) | **Kết quả cải thiện**: K-sweep hoàn tất, K_neigs=2 → v2.0 Top-1 F1 **0.697±0.003** (+16.8%) | Xem kết quả mới nhất v2.0 |
| [RESULTS_CONFORMAL.md](RESULTS_CONFORMAL.md) | **Đột phá**: uncertainty-aware type prediction (conformal APS/RAPS/Mondrian) | Hướng mở rộng mới |
| [REPRODUCE.md](REPRODUCE.md) | Lệnh tái hiện đầy đủ Plan M + conformal (đường cong K, dump, coverage) | Khi cần chạy lại từ đầu |

## 🚀 Quick Start

Nếu bạn là người mới:

1. Đọc [ARCHITECTURE.md](ARCHITECTURE.md) (15 phút) để có bản đồ tổng quan
2. Xem section "Quick Commands" trong [CHEATSHEET.md](CHEATSHEET.md) để chạy thử
3. Khi gặp đoạn code khó hiểu, lookup trong [NOTES_DATAFLOW.md](NOTES_DATAFLOW.md) hoặc [NOTES_MODEL.md](NOTES_MODEL.md)

## 🔍 Tìm theo vấn đề

- **"Không biết bắt đầu từ đâu"** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **"Tensor này shape gì?"** → [NOTES_DATAFLOW.md](NOTES_DATAFLOW.md)
- **"Công thức toán của block X?"** → [NOTES_MODEL.md](NOTES_MODEL.md)
- **"Lệnh này chạy sao?"** → [CHEATSHEET.md](CHEATSHEET.md) §Quick Commands
- **"Đang bị lỗi Y"** → [CHEATSHEET.md](CHEATSHEET.md) §Troubleshooting
- **"Muốn tune hyperparameter"** → [CHEATSHEET.md](CHEATSHEET.md) §Tuning recipes
- **"Hỏi AI ở nhà"** → [CHEATSHEET.md](CHEATSHEET.md) §Prompt templates

## 🏗️ Project structure

```
DHGCMDA/
├── docs/                             ← bạn đang ở đây
│   ├── README.md                     ← index này
│   ├── ARCHITECTURE.md
│   ├── NOTES_DATAFLOW.md
│   ├── NOTES_MODEL.md
│   └── CHEATSHEET.md
├── main_experiments_hetero1.py       ← entry point + training loop
├── hetero_model.py                   ← model definitions
├── prepareData.py                    ← data loading
├── trainData.py                      ← Dataset + 5-fold CV
├── create_hetero_data.py             ← HeteroData builder
├── ConstructHW.py                    ← hypergraph wrapper
├── hypergraph_construct_KNN.py       ← KNN hypergraph
├── hypergraph_construct_kmeans.py    ← K-means hypergraph
├── param.py                          ← hyperparameters
├── Calculate_Metrics.py              ← evaluation metrics
├── wieghts.py                        ← class distribution analyzer (typo trong tên file, không phải lỗi)
├── utils.py                          ← MSE loss + L2 reg
├── layers.py                         ← attention + FFN modules
├── check_v2.0_495m383D.py           ← data validation script
└── v2.0_495m383D/                    ← dataset folder
```

## 💡 Lưu ý quan trọng

- **Encoding**: luôn dùng `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` khi chạy trên Windows
- **File CSV**: dùng **1-based index**; code tự convert sang 0-based
- **`HGCN_Attention_Mechanism`**: tên gây hiểu nhầm — thực chất là weighted sum cố định 0.6/0.4
- **`np.mat` đã bị xoá trong NumPy 2.0**: repo này đã fix sang `np.asmatrix`
- **GPU cũ (< Kepler/compute 3.5)**: bắt buộc chạy CPU vì PyTorch đã drop support

## 🔬 Kết quả nghiên cứu (2026-07-05)

Hai luồng công việc mới, **additive** (không sửa model/loss lõi):

1. **Plan M — Right-sizing hypergraph** ([RESULTS_PLAN_M.md](RESULTS_PLAN_M.md)): hoàn tất đường cong `K_neigs` dưới
   `full_bilinear` (đơn điệu K13<K7<K3<K2<K1). Best hợp lệ = **K=2 → 0.697±0.003** (so paper 0.5970: **+16.8%**).
   K=1 = no_hgcn (trùng khít) → xác nhận over-parameterization.

2. **Conformal — Uncertainty-aware type prediction** ([RESULTS_CONFORMAL.md](RESULTS_CONFORMAL.md)): prediction SET
   với đảm bảo coverage (phân phối-tự-do, post-hoc). v2.0 informative (2.29/4); v3.2 phát lộ + sửa collapse
   type Tissue bằng class-conditional Mondrian (T5 0.64→0.96).

Bối cảnh học thuật + kế hoạch bài báo: thư mục `research/` ở gốc repo.

## 📝 Version history của docs

- 2026-04-17: Khởi tạo docs set (ARCHITECTURE, NOTES_DATAFLOW, NOTES_MODEL, CHEATSHEET)
- 2026-07-05: Thêm RESULTS_PLAN_M, RESULTS_CONFORMAL, REPRODUCE (Plan M K-sweep + conformal breakthrough)
