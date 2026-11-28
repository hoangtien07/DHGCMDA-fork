# Kết quả luận văn DHGCMDA

## 1. Phương pháp và dữ liệu

Tái chạy mã phát hành DHGCMDA trên HMDD v2.0 và port năm loại trên MDAv3.2-3. Giao thức đánh giá có kiểm soát dùng cùng các cặp fit, validation và test cho mọi phương pháp.

## 2. Tái hiện Bảng 3 và Bảng 4

Mã phát hành `06f9ba3eed92e334d9a2b8003fc06d83596e5614`. Kết luận G3 `far`.

| Run | AUC | AUPR | F1 nhị phân | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1-cpu | 0.9739 | 0.9672 | 0.9289 | 0.9045 | 0.9557 |
| c2-cpu | 0.9631 | 0.9551 | 0.9105 | 0.8826 | 0.9418 |
| c3-cpu | 0.9742 | 0.9685 | 0.9312 | 0.9006 | 0.9645 |
| v32-K_star-1234-cpu | 0.8861 | 0.8893 | 0.8097 | 0.7817 | 0.8403 |
| v32-T_star-1234-cpu | 0.8713 | 0.8755 | 0.7952 | 0.7608 | 0.8343 |

| Đối chiếu | Sai lệch có dấu |
| --- | --- |
| v2.0 Bảng 4 | auc +0.0001, aupr +0.0003, f1 +0.0011 |
| v2.0 Bảng 3 | top1_precision -0.0282, top1_recall -0.0089, top1_f1 -0.0086 |
| v3.2 Bảng 4 | auc -0.0410, aupr -0.0288, f1 -0.0577 |
| v3.2 Bảng 3 | precision -0.0307, recall -0.1078, f1 -0.0648 |

Chênh lệch C3 − C1 khi đổi seed, tính trên cùng cách chia fold của mã phát hành.

| Chỉ số | C3 − C1 |
| --- | ---: |
| auc | +0.0003 |
| aupr | +0.0013 |
| f1 | +0.0023 |
| accuracy | +0.0019 |
| recall | +0.0088 |
| specificity | -0.0051 |
| precision | -0.0039 |
| top1_precision | +0.0343 |
| top1_recall | +0.0137 |
| top1_f1 | +0.0259 |

## 3. Đánh giá lại và phân rã rò rỉ

| Run | Top-1 F1 TDRC | Top-1 mã gốc hoặc mở rộng | Macro-F1 đa nhãn |
| --- | ---: | ---: | ---: |
| c1-cpu | 0.5160 | 0.5515 | 0.5472 |
| c2-cpu | 0.5623 | 0.5884 | 0.5733 |
| c3-cpu | 0.5440 | 0.5774 | 0.5601 |
| v32-K_star-1234-cpu | 0.2827 | 0.3371 | 0.3296 |
| v32-T_star-1234-cpu | 0.3222 | 0.3787 | 0.3484 |

Macro-F1 trên các dự đoán tác giả dùng ngưỡng cố định 1/K cho mỗi loại. Phép chấm Top-1 mở rộng áp dụng cho v3.2.

### v2.0_495m383D

| Phương pháp | Top-1 F1 TDRC | SD | Chênh lệch với legacy |
| --- | ---: | ---: | ---: |
| prior | 0.5389 | 0.0174 | — |
| logistic | 0.6075 | 0.0362 | — |
| tdrc | 0.4244 | 0.0214 | — |
| spld | 0.5251 | 0.0444 | — |
| clean | 0.5007 | 0.0521 | -0.0168 |
| legacy | 0.5174 | 0.0272 | — |
| gip-only | 0.4855 | 0.0498 | -0.0320 |
| association-only | 0.4143 | 0.0414 | -0.1031 |
| validation-only | 0.4999 | 0.0594 | -0.0176 |

### v3.2_spld_paper

| Phương pháp | Top-1 F1 TDRC | SD | Chênh lệch với legacy |
| --- | ---: | ---: | ---: |
| prior | 0.5285 | 0.0078 | — |
| logistic | 0.5565 | 0.0084 | — |
| tdrc | 0.4123 | 0.0159 | — |
| spld | 0.5182 | 0.0111 | — |
| clean | 0.2729 | 0.0148 | +0.0004 |
| legacy | 0.2726 | 0.0138 | — |
| gip-only | 0.2692 | 0.0121 | -0.0033 |
| association-only | 0.2538 | 0.0324 | -0.0188 |
| validation-only | 0.2726 | 0.0138 | +0.0000 |
| multilabel-only | 0.4336 | 0.0119 | +0.1611 |

## 4. Thí nghiệm cải tiến

| Dữ liệu | Top-1 F1 TDRC trung bình | SD | Δ với DHGCMDA sạch | Δ với baseline chọn trên validation | DHGCMDA gốc theo giao thức tác giả |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2.0_495m383D | 0.5368 | 0.0304 | +0.0345 | -0.0673 | 0.5160 |
| v3.2_spld_paper | 0.4542 | 0.0091 | +0.1886 | -0.1033 | 0.2827 |

Cột giao thức tác giả dùng cùng quy ước F1 TDRC để đối chiếu. So sánh ghép cặp dùng các kết quả trên tập test chung của giao thức sạch.

| Dữ liệu | Macro-F1 đa nhãn | AUC | AUPR | Wilcoxon p so với DHGCMDA sạch | p kiểm định dấu |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2.0_495m383D | 0.3939 | 0.9188 | 0.9165 | 0.0946 | 0.3018 |
| v3.2_spld_paper | 0.2886 | 0.8249 | 0.8197 | 0.0001 | 0.0001 |

Mô hình được chọn trên validation chưa vượt cả prior và logistic trên v3.2. Trọng tâm kết luận là tái hiện và đánh giá lại.

## 5. Hạn chế

- Đặc trưng GIP trong tái hiện dùng toàn bộ ma trận liên kết. Kết quả theo giao thức tác giả cần được diễn giải theo yếu tố này.
- View bệnh thứ hai v3.2 dùng bản sao D_MESH theo dữ liệu hiện có.
- Fold lặp qua seed dùng chung dữ liệu. Kiểm định ghép cặp mô tả kết quả trên benchmark này.
