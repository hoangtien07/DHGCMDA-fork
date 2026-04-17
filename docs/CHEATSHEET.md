# Cheatsheet — DHGCMDA

Tham chiếu nhanh: lệnh chạy, hyperparams, troubleshooting. Không cần kết nối internet vẫn dùng được.

---

## 🚀 Quick Commands

### Setup lần đầu (đã làm)

```bash
cd e:/VSCode/DHGCMDA
py -m venv venv
./venv/Scripts/python.exe -m pip install --upgrade pip
./venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/Scripts/python.exe -m pip install numpy pandas scikit-learn scipy torch-geometric
```

### Chạy các lần sau

```bash
# Verify data loading
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 ./venv/Scripts/python.exe check_v2.0_495m383D.py

# Phân tích class distribution
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 ./venv/Scripts/python.exe wieghts.py

# Training ngắn (debug, ~45s)
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 ./venv/Scripts/python.exe main_experiments_hetero1.py \
    --epoch 3 --validation 2 --device cpu

# Training vừa phải (~30 phút trên CPU)
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 ./venv/Scripts/python.exe main_experiments_hetero1.py \
    --epoch 50 --validation 3 --device cpu

# Full training (650 epoch × 5 fold — CHỈ làm khi có GPU!)
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 ./venv/Scripts/python.exe main_experiments_hetero1.py \
    --epoch 650 --validation 5
```

**Trên Windows bắt buộc** có `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` vì code có nhiều emoji và tiếng Trung, default cp1252 không handle được.

---

## ⚙️ Hyperparameter Cheatsheet

### Training

| Param | Default | Range | Ảnh hưởng |
|-------|---------|-------|-----------|
| `--epoch` | 650 | 100–1000 | Nhiều hơn → fit tốt hơn, nhưng có thể overfit |
| `--lr` | 0.0001 | 1e-5 → 1e-3 | Cao hơn → train nhanh nhưng dễ NaN |
| `--dropout` | 0.3 | 0.1–0.5 | Cao hơn → regularize mạnh hơn |
| `--weight_decay` | 5e-5 | 0–1e-3 | L2 regularization |
| `--validation` | 5 | 2–10 | Số fold CV (5 là standard) |
| `--seed` | 1234 | any int | Reproducibility |

### Architecture

| Param | Default | Range | Ảnh hưởng |
|-------|---------|-------|-----------|
| `--nlayer` | 2 | 1–4 | Số layer HGT. 3+ có thể over-smoothing |
| `--n_head` | 8 | 4, 8, 16 | Số attention head. hidden_dim phải chia hết cho n_head |
| `--K_neigs` | `[13]` | `[5]`–`[20]` | K của KNN hypergraph. K nhỏ → local, lớn → global |
| `--clusters` | `[9]` | `[5]`–`[15]` | Số cluster K-means (dùng khi fallback) |

### Contrastive Learning

| Param | Default | Range | Ảnh hưởng |
|-------|---------|-------|-----------|
| `--cl_temperature` | 0.5 | 0.1–1.0 | Thấp → sharp, cao → soft |
| `--cl_weight` | 1.0 | 0–2 | Weight của intra-view CL |
| `--enable_inter_view_cl` | True | bool | Bật cross-modal CL |
| `--inter_view_weight` | 0.3 | 0–1 | Weight của inter-view CL |
| `--inter_view_temperature` | 0.5 | 0.1–1.0 | Temperature cho inter-view |
| `--inter_view_margin` | 0.5 | 0.1–1.0 | Margin trong ranking loss |

### Loss

| Param | Default | Range | Ảnh hưởng |
|-------|---------|-------|-----------|
| `--use_focal_loss` | True | bool | Focal vs BCE |
| `--focal_gamma` | 2.5 | 0–5 | Cao hơn → focus mạnh hơn vào hard examples |
| `--alpha` | 0.5 | 0–1 | Weight của negative samples |
| `--class_weights` | auto | list of 4 | Weight cho 4 type |

---

## 🎯 Tuning recipes — kịch bản thường gặp

### Model không học (loss flat)

```bash
--lr 0.001 --epoch 100 --validation 3
```
Tăng LR, giảm epoch để debug nhanh. Nếu vẫn không học → check gradient (code tự print mỗi 50 epoch).

### Model NaN loss sau vài epoch

```bash
--lr 0.00005 --dropout 0.4 --weight_decay 1e-4
```
Giảm LR, tăng dropout và weight decay. Code có clip gradient (`max_norm=1.0`) rồi, nếu vẫn NaN là LR quá cao.

### AUC cao nhưng Top-1 F1 thấp (type classification kém)

```bash
--focal_gamma 3.5 --inter_view_weight 0.5
```
Tăng focal gamma để focus vào minority classes. Tăng inter-view weight để học cross-modal tốt hơn.

### Overfit (train loss ↓ nhưng val AUC không ↑)

```bash
--dropout 0.5 --weight_decay 1e-3 --epoch 300
```
Regularize mạnh hơn, giảm epoch.

### Ablation — đo đóng góp từng phần

```bash
# Không có inter-view CL
--enable_inter_view_cl False

# 1 layer HGT thay vì 2
--nlayer 1

# Không dùng focal loss
--use_focal_loss False

# K=5 (local only)
--K_neigs 5

# K=20 (global)
--K_neigs 20
```

---

## 🐛 Troubleshooting

### `UnicodeEncodeError: 'charmap' codec can't encode character`

**Fix**: luôn set env vars trước khi chạy.
```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 ./venv/Scripts/python.exe ...
```

Hoặc set 1 lần cho session (bash):
```bash
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
```

### `AttributeError: np.mat was removed in the NumPy 2.0 release`

Đã fix trong repo này — thay bằng `np.asmatrix`. Xem commit history nếu muốn biết chi tiết.

### `CUDA out of memory`

```bash
--nlayer 1 --n_head 4   # giảm model size
# hoặc
--device cpu            # dùng CPU (chậm nhưng không OOM)
```

### `ImportError: torch_geometric`

```bash
./venv/Scripts/python.exe -m pip install torch-geometric
```
Nếu fail — `torch_geometric` đôi khi cần `torch-scatter`, `torch-sparse`. Chạy:
```bash
./venv/Scripts/python.exe -m pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-$(torch --version).html
```

### Training rất chậm trên CPU

Ước lượng: 650 epoch × 5 fold ≈ 2–3 giờ trên CPU hiện đại (i5/i7). Với GPU sẽ ~10 phút.

**Workaround**: giảm `--epoch 100 --validation 3` để có kết quả trong ~30 phút, vẫn đánh giá được model.

### Seed không reproduce

Code có set seed trong `seed_torch()`, nhưng một số op của PyTorch trên GPU **không deterministic**. Để force deterministic:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Tradeoff: chậm hơn ~20%.

### Gradient check báo "❌ 过小" (quá nhỏ)

Có nghĩa là gradient ≈ 0, phần đó không học. Kiểm tra:
1. Có bị `detach()` nhầm ở đâu không?
2. Activation có chết không (ReLU → 0 mãi)?
3. Learning rate cho layer đó có được scale không?

### Kết quả run 2 lần khác nhau dù cùng seed

- Seed được set ở start [main_experiments_hetero1.py:48](main_experiments_hetero1.py#L48)
- Nhưng có **negative sampling ngẫu nhiên** ở [main_experiments_hetero1.py:659](main_experiments_hetero1.py#L659) dùng `torch.randperm` trên `device` → state có thể khác
- Và HGT/torch-geometric có nondeterministic ops
- → Chấp nhận sai lệch ~±0.02 AUC giữa các run

---

## 📊 Metric reference

Hiểu output của training:

```
Epoch 1, Total Loss: 18.9850, Recover Loss: 1.1192, miRNA CL: 8.8816, ...
       ↑ tổng loss       ↑ existence+type    ↑ intra+inter CL miRNA
```

| Metric | Range tốt | Ý nghĩa |
|--------|-----------|---------|
| Total Loss | ↓ đều | Cần giảm, không nên plateau sớm |
| Recover Loss | 0.5–2.0 | Task loss chính |
| miRNA/Disease CL | 0.5–5.0 | CL loss, giảm theo epoch |
| Recon Loss | 0.1–1.0 | MSE của similarity recon |
| AUC (binary) | >0.85 là tốt | Sample-level binary classification |
| AUPR | >0.6 là tốt | Precision-recall AUC |
| Top-1 F1 | >0.5 là tốt | Multi-class type prediction |

---

## 📁 File reference quick lookup

| Cần làm gì | Mở file nào |
|------------|-------------|
| Thêm hyperparam CLI | [param.py](../param.py) |
| Thay đổi data loading | [prepareData.py](../prepareData.py) |
| Thay đổi architecture | [hetero_model.py](../hetero_model.py) |
| Thay đổi loss | [main_experiments_hetero1.py:56](../main_experiments_hetero1.py#L56) `SimplifiedMultiTypeAssociationLoss` |
| Thay đổi training loop | [main_experiments_hetero1.py:647](../main_experiments_hetero1.py#L647) `train_epoch_optimized` |
| Thay đổi KNN / K-means | [hypergraph_construct_KNN.py](../hypergraph_construct_KNN.py), [hypergraph_construct_kmeans.py](../hypergraph_construct_kmeans.py) |
| Thay đổi hetero graph edges | [create_hetero_data.py](../create_hetero_data.py) |
| Thêm metric | [Calculate_Metrics.py](../Calculate_Metrics.py) |

---

## 🧪 Prompt templates cho Claude Free / ChatGPT ở nhà

Khi bạn hỏi AI về project này mà không có context đầy đủ, paste template sau:

### Template 1: Hỏi về 1 đoạn code

```
Tôi đang đọc dự án DHGCMDA (miRNA-disease association prediction
dùng heterogeneous graph + hypergraph contrastive learning).
Cho tôi đoạn code sau từ file [tên file]:

[paste code]

Giải thích:
1. Đoạn này đang làm gì (step-by-step)
2. Shape của tensor input/output
3. Công thức toán học (nếu có)
4. Tại sao lại thiết kế như vậy
```

### Template 2: Hỏi về lỗi

```
Tôi đang chạy dự án DHGCMDA (PyTorch + torch_geometric).
Command: [command]
Lỗi nhận được:

[paste error trace]

Môi trường:
- Python 3.14
- PyTorch CPU
- NumPy 2.4

Gợi ý cho tôi root cause và cách fix.
```

### Template 3: Review idea

```
Trong dự án DHGCMDA tôi muốn thử [idea]. Project này:
- Dùng dual-view hypergraph (KNN)
- Inter-view contrastive learning
- HGT (heterogeneous graph transformer)
- Multi-type classification (4 loại association)

Review idea của tôi: [idea chi tiết]
Có khả thi không? Pros/cons? Cần sửa file nào?
```

### Template 4: Debug training

```
Dự án DHGCMDA của tôi đang train với config:
--epoch 100 --lr 0.0001 --nlayer 2

Kết quả:
Epoch 1: Loss=18.9, AUC sau fold 1 = 0.75
Epoch 50: Loss=12.3, AUC = 0.78
Epoch 100: Loss=11.8, AUC = 0.78

→ plateau sớm. Nghi ngờ [gì đó]. Gợi ý cách debug/tune.
```

---

## 🎓 Học gì khi đi xa hơn

Nếu muốn hiểu sâu, đọc theo thứ tự:

1. **Hypergraph Neural Networks** — Feng et al., AAAI 2019 (arxiv.org/abs/1809.09401)
2. **SimCLR** — Chen et al., 2020 (arxiv.org/abs/2002.05709)
3. **Heterogeneous Graph Transformer** — Hu et al., WWW 2020 (arxiv.org/abs/2003.01332)
4. **Focal Loss** — Lin et al., ICCV 2017 (arxiv.org/abs/1708.02002)
5. **Class-Balanced Loss (Effective Number)** — Cui et al., CVPR 2019 (arxiv.org/abs/1901.05555)
6. **Multi-view Contrastive Learning** — review bất kỳ về "multimodal contrastive"

Gần với domain sinh học:
- miRNA biology: Bartel, "MicroRNAs: genomics, biogenesis, mechanism, and function" (Cell 2004)
- miRNA-disease prediction review: Chen et al., "miRNA-disease association prediction: a survey" (Brief Bioinform 2019)
