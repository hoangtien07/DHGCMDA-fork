# Mâu thuẫn nội tại trong paper (đọc từ PDF đính kèm, render page 26 và 30: `lit/page26.png`, `lit/page30.png`)

Hàng "DHGCMDA" (mô hình đầy đủ) trong Fig. 4 cho v3.2 không khớp với số của chính paper ở Table 3/4. Với v2.0 thì Fig. 4 khớp Table.

| Metric | v2.0 Table | v2.0 Fig.4 | v3.2 Table 3/4 | v3.2 Fig.4 |
|---|---:|---:|---:|---:|
| Top-1 P | 0.5842 | 0.5842 | **0.7915** | **0.6005** |
| Top-1 R | 0.6341 | 0.6341 | **0.9421** | **0.5253** |
| Top-1 F1 | 0.5970 | 0.5970 | **0.8600** | **0.5678** |
| AUC | 0.9669 | 0.9669 | 0.9181 | 0.9429 |
| AUPR | 0.9738 | 0.9738 | 0.9271 | 0.9439 |
| F1 (triplet) | 0.9278 | 0.9278 | 0.8674 | 0.8760 |

- Cùng một mô hình, cùng một dataset, nhưng hai nơi trong paper báo Top-1 F1 lệch nhau 0.29. Ở bộ số Fig. 4 thì R < P, nhất quán với micro recall hoặc với các baseline.
- Không xác định được bộ số nào là đúng: paper không giải thích sự khác biệt [không đủ bằng chứng].
- Hệ quả: mục tiêu "v3.2 Top-1 F1 = 0.86" không phải mục tiêu duy nhất mà paper đưa ra. Bộ số Fig. 4 (0.5678) là một mục tiêu khác, cũng do paper công bố.
- Fig. 3: lưới K = {1,3,5,7,9,11,13,15}. Paper báo đỉnh Top-1 tại K=13 và AUC/AUPR/F1 ổn định trên toàn lưới.
- Fig. 2: λ2 ∈ {0.1,0.3,0.5}, t ∈ {0.5,1,3}, cùng 3 cách đặt class weight (inverse, uniform, sqrt-inverse).
