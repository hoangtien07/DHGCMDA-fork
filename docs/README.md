# DHGCMDA Documentation

Tài liệu hướng dẫn dự án DHGCMDA (tiếng Việt).

## 📚 Danh mục

| File | Nội dung | Khi nào đọc |
|------|----------|-------------|
| [paper/](paper/README.md) | **Series 9 chương giải thích PAPER cho người mới** — từ bối cảnh sinh học đến reproduce notes | **Đọc đầu tiên nếu mục tiêu là hiểu bài báo** |
| [ARCHITECTURE.md](ARCHITECTURE.md) | ASCII diagram toàn bộ pipeline, file → function map | **Đọc đầu tiên** nếu mục tiêu là vọc code — lấy cái nhìn tổng quan |
| [NOTES_DATAFLOW.md](NOTES_DATAFLOW.md) | Data flow chi tiết từng bước, shape tensor | Đọc khi muốn hiểu dữ liệu di chuyển thế nào |
| [NOTES_MODEL.md](NOTES_MODEL.md) | Giải thích từng class + công thức toán | Đọc khi muốn hiểu sâu về kiến trúc |
| [CHEATSHEET.md](CHEATSHEET.md) | Lệnh chạy, hyperparams, troubleshooting, prompt templates | Tham chiếu nhanh khi vọc code |

## 🚀 Quick Start

Nếu bạn là người mới:

1. Muốn **hiểu paper**: đọc series [paper/](paper/README.md) theo thứ tự chương 01→09 (chỉ cần 01+03+09 cho bản rút gọn)
2. Muốn **vọc code**: đọc [ARCHITECTURE.md](ARCHITECTURE.md) (15 phút) để có bản đồ tổng quan
3. Xem section "Quick Commands" trong [CHEATSHEET.md](CHEATSHEET.md) để chạy thử
4. Khi gặp đoạn code khó hiểu, lookup trong [NOTES_DATAFLOW.md](NOTES_DATAFLOW.md) hoặc [NOTES_MODEL.md](NOTES_MODEL.md)

## 🔍 Tìm theo vấn đề

- **"Chưa đọc/hiểu paper DHGCMDA"** → [paper/](paper/README.md)
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
│   ├── paper/                        ← series chương giải thích paper (9 file)
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

## 📝 Version history của docs

- 2026-04-17: Khởi tạo docs set (ARCHITECTURE, NOTES_DATAFLOW, NOTES_MODEL, CHEATSHEET)
- 2026-09-28: Thêm `paper/` — 9 chương giải thích paper DHGCMDA cho người mới (đọc theo thứ tự 01→09)
