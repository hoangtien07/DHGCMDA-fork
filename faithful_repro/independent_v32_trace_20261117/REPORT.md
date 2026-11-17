# Kiểm toán độc lập v3.2: cặp–loại và nguồn đặc trưng U1

Ngày 2026-11-17. Kiểm toán chỉ đọc dữ liệu, nhãn và bundle đầu vào; không huấn luyện hoặc chấm dự đoán outer-test.

**Kết luận:** bộ reference v3.2 nội bộ và snapshot SPLD khớp **từng triplet, thứ tự tên và nhãn**, chứ không chỉ khớp tổng số. Lượt U1 650 epoch fold0 và fold1 đã kiểm tra không dùng nguyên M_MISIM đầy đủ làm đầu vào chức năng: hai khối chức năng và ngữ nghĩa được dựng lại độc lập từ các nguồn đã pin và khớp chính xác bundle. Tuy nhiên, chưa có dữ liệu v3.2 gốc từ tác giả DHGCMDA để chứng minh identity, và đặc trưng thay thế chưa được chứng minh giống đặc trưng paper.

## 1. Dữ liệu theo từng cặp–loại

- Tensor snapshot `Y_multilabel.npy`, CSV `triplets.csv`, reference NPZ và strict 4-view NPZ đều khớp: 411 miRNA × 271 bệnh × **5 loại**, 11.748 triplet trên 8.735 cặp; 2.110 cặp có hơn một loại. CSV không có triplet lặp, tập triplet bằng chính các ô dương trong tensor.
- Thứ tự kênh được xác nhận bằng tên: genetics **1.155**, epigenetics **403**, circulation **2.293**, target **3.997**, tissue **3.900**. Thứ tự và tên miRNA/bệnh khớp giữa snapshot, reference và strict 4-view. Hai miRNA `hsa-mir-1179` và `hsa-mir-1244` chỉ có một cặp/một triplet trong tensor cuối; chưa xác định điều này có trái recipe tác giả hay do đơn vị và thứ tự lọc khác nhau.
- Ledger đầy đủ: `triplet_ledger.csv` gồm tên thực thể, loại, segment fold0 và tier đối chiếu MISIM; `pair_ledger.csv` cho từng cặp. Đây là identity **giữa các nguồn nội bộ đã nêu**; snapshot đến từ dòng SPLD, không phải tensor v3.2 do nhóm DHGCMDA công bố. Bản mã DHGCMDA đã kiểm tra chỉ phát hành loader/data v2.0 bốn loại. Các số khớp paper không đủ chứng minh hai danh sách chính xác là một.

## 2. Cách chia fold và tính đúng đầu vào U1

CSV candidate chứa đủ 8.735 cặp, mỗi cặp nằm ở outer-test đúng một lần trên 5 fold. Theo group key của candidate (miRNA theo index, bệnh theo cùng MeSH UI), không có group nào bị chia giữa inner-train, inner-validation và outer-test trong bất kỳ fold nào. Group theo UI là quy tắc phòng ngừa; không khẳng định mọi tên chung UI đồng nhất sinh học. Các họ miRNA mơ hồ vẫn để riêng, nên rủi ro alias còn lại không bằng 0.

| Fold | Inner-train cặp | Inner-validation cặp | Outer-test cặp | Group inner-train | Group inner-validation | Group outer-test |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.293 | 695 | 1.747 | 6.154 | 684 | 1.712 |
| 1 | 6.287 | 701 | 1.747 | 6.151 | 684 | 1.715 |
| 2 | 6.286 | 702 | 1.747 | 6.165 | 686 | 1.699 |
| 3 | 6.284 | 704 | 1.747 | 6.156 | 684 | 1.710 |
| 4 | 6.288 | 700 | 1.747 | 6.152 | 684 | 1.714 |

Lưu ý tên trường manifest `train_groups`: nó đếm **tất cả group ngoài outer-test**, gồm cả inner-validation. Ví dụ fold0 6.838 = 6.154 + 684, không phải 6.838 group inner-train. Điều này phù hợp contract hiện tại (`outer_eval_contract_rev3.py` đếm `nontest`), nhưng tên trường dễ gây hiểu nhầm; không sửa manifest đã pin. Các fold là split candidate nội bộ, không phải fold gốc tác giả.

Bundle U1 fold0/1 được đối chiếu SHA-256 với driver, metadata và `expected.json` của từng lượt 650 epoch; tất cả hash từng array khớp. Nhãn train/val khớp Y. Cả bốn khối association trong view là train mask hoặc transpose, và mask không có ô dương của validation/outer-test. Fold0 có 6.293 train và 695 val; fold1 6.287 train và 701 val. Khối sequence/gene khớp strict NPZ; khối functional **không bằng** M_MISIM tĩnh. Dựng lại hai khối functional/semantic từ nguồn và code được pin khớp **elementwise** cả fold0 lẫn fold1 (`feature_rebuild.json`); chỉ có thể kết luận train-only theo recipe đó, không kết luận giống đặc trưng tác giả.

Split candidate fold2–4 đã kiểm tra ở mức danh sách/group; chưa có bundle U1 650 tương ứng trong các đường dẫn đã kiểm tra, vì vậy chưa thể chứng nhận đặc trưng thực thi của các fold đó. Không suy từ fold0/1 sang mọi biến thể pipeline hoặc lượt chạy khác.

## 3. Nguồn từng view và rủi ro còn lại

| Đầu vào U1 | Nguồn đã kiểm tra | Rủi ro/giới hạn |
|---|---|---|
| Association trong bốn view | train mask theo fold | Fold0/1 xác minh trực tiếp, val/test không nằm trong mask. Nhận định rò rỉ trực tiếp của mã v2.0 không được áp tự động cho U1. |
| `S_seq` | strict GFF3–miRBase + căn chỉnh sequence; khớp NPZ strict | Recipe thay thế; chưa chứng minh giống tác giả. Fold0 có 40 hàng không có phần tử ngoài đường chéo khác 0. |
| `S_gene` | CTD marker/mechanism + HumanNet, khớp NPZ strict | Recipe thay thế; chưa chứng minh giống tác giả. Có 26 hàng không có phần tử ngoài đường chéo khác 0. |
| `S_func` | Dựng từ train mask + ngữ nghĩa bệnh, code/nguồn pin; tái dựng khớp bundle fold0/1 | Train-only theo recipe; không phải M_MISIM snapshot và không chứng minh trùng feature paper. Fold0 có 6 miRNA không có cạnh train. |
| `S_sem` | Từ bảng ánh xạ disease/MeSH và cây 2019 được pin; tái dựng khớp bundle fold0/1 | 262/271 disease được phủ theo audit nguồn; 9 hàng chưa phủ. Không chứng minh nguồn/recipe giống paper. |

Tập MISIM v2.0 tĩnh trong snapshot là kênh của **reference NPZ**, không phải khối `S_func` thực tế của hai bundle U1 này. Tái sử dụng ledger `eval_vs_curated_pairs.csv` từ kiểm toán nguồn MISIM, phân loại theo **split U1 candidate** hiện tại cho thấy 7.770/8.735 cặp trùng tên đã chuẩn hóa với record nguồn được curate, thêm 60 cặp cùng MeSH UI; trong fold0 validation lần lượt là 621/695 và thêm 6. Tier same-UI là ứng viên, không bằng chứng chắc về cùng khái niệm; phép match tên có chuẩn hóa miRNA. Đây là chồng lấp **theo cặp**, không chứng minh trùng nhãn loại, không đo ảnh hưởng tới điểm, và không có nghĩa bundle U1 đã dùng M_MISIM tĩnh. Nguồn ledger/recipe curate là kiểm toán trước, được tái phân đoạn ở đây chứ không tái chạy toàn bộ pipeline curate.

## 4. Metric và điều chưa thể chốt

Paper Table 1/2 ghi năm loại cho v3.2; phần CV_type lại nói bốn loại, trong khi metric mã phát hành hard-code bốn loại. Do đó dữ liệu năm loại là phù hợp với bảng paper, còn định nghĩa Top-1 cho v3.2 chưa được xác minh. Không gán macro-F1 đang dùng hoặc biến thể mở rộng mã tác giả là công thức đã tạo 0,8600. Trên cùng dự đoán validation U1 fold0 đã kiểm tra trước đó, macro-F1 = 0,470762 và first-label hybrid mở rộng = 0,514719; đây là kết quả validation/chọn checkpoint, không phải 5-fold outer-test.

## 5. Tệp và tái kiểm tra

- `audit.py`: đối chiếu 11.748 triplet, 8.735 cặp, 5 split và hash/bundle U1 fold0/1; `summary.json` là số liệu máy đọc, `pair_ledger.csv` và `triplet_ledger.csv` là ledger chi tiết.
- `rebuild_features.py`: kiểm tra SHA mã nguồn và dữ liệu pin, dựng lại khối functional/semantic cho hai fold; `feature_rebuild.json` ghi kết quả. Không đọc model hoặc dự đoán test.
- Bằng chứng cũ về nguồn MISIM: `../report/MISIM_SOURCE_RECORD_OVERLAP_AUDIT.md`; bằng chứng metric: `../independent_reproduction_audit_20261116/REPORT.md`. Các phát biểu nguồn gốc/tương đương tác giả chưa được chứng minh giữ nguyên trạng thái chưa biết.
