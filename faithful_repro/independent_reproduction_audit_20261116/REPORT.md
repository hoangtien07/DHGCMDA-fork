# Kiểm tra độc lập khả năng tái hiện DHGCMDA

Ngày 16-11-2026. Phạm vi đã thực hiện: nguồn paper và mã công bố, danh tính dữ liệu khôi phục, metric trên dự đoán đã lưu, công thức loss thực sự được chạy. Đây là đánh giá tái hiện, không phải một lượt tối ưu mô hình.

## Kết luận để quyết định

**Chưa đủ bằng chứng coi U1 là tái hiện số học của DHGCMDA. Tuy nhiên, dữ liệu năm loại là đúng với mô tả v3.2 của paper, và thay định nghĩa F1 không tự đưa các dự đoán U1 hiện có lên 0,8600.**

Hướng nên tiếp tục là xác lập tương đương của dữ liệu, metric và objective trước khi chạy thêm mô hình. Giữ nguyên U1 như một baseline triển khai theo cách diễn giải paper. Phân biệt nó với việc chạy lại nguyên bản mã phát hành.

Các kết quả mới có thể hành động ngay:

1. Cùng dự đoán validation fold0, macro-F1 = **0,470762**, còn mở rộng công thức mã tác giả cho năm loại = **0,514719**. Trên fold1 tương ứng **0,433291** và **0,481183**. Khác cách đo có tác động thực, nhưng không giải thích toàn bộ chênh lệch số học với mục tiêu công bố.
2. Metric lấy nhãn dương đầu tiên phụ thuộc thứ tự kênh. Chỉ đổi đồng thời thứ tự nhãn và score, giữ nguyên ý nghĩa dự đoán, F1 fold0 chạy trong **0,464934–0,518448**. Đây là rủi ro của convention, không phải thay đổi chất lượng mô hình.
3. Mã tác giả công khai thực sự dùng decoder diagonal bilinear. Vì vậy, đổi sang full bilinear không tự động làm mô hình trung thành hơn với bản phát hành.
4. **U1 cuối cùng không chỉ cộng các loss trung bình độc lập.** `common_mass_sums` nhân lại số lượng phần tử để khôi phục các tổng trước khi chia cùng mẫu số. Đánh giá fidelity chỉ nhìn `reconstruction_adapter` sẽ bỏ sót bước này.
5. Tensor khôi phục có hai miRNA chỉ có một unique typed link. Điều đó chưa khớp cách hiểu đơn giản của câu “loại thực thể có ít hơn hai association”, dù mọi thống kê bảng đều khớp. Cần làm rõ đơn vị đếm lúc lọc, không tự loại hai dòng này.

## 1. Nguồn và giới hạn kiểm tra

- PDF người dùng cung cấp: `[2026] DHGCMDA a dual-view heterogeneous graph constrastive learning framework for miRNA-disaese.pdf` ở gốc repo; SHA256 có trong `input_sha256.json`. Các trang phương pháp/kết quả cần dùng được trích mới vào `sources/paper_p*.txt`; trang 23 và 30 được render để đọc bảng/hình.
- [Bài xuất bản](https://link.springer.com/article/10.1186/s12859-026-06436-w) và [mã tác giả được cố định theo commit](https://github.com/CDMBlab/DHGCMDA/tree/06f9ba3eed92e334d9a2b8003fc06d83596e5614).
- Commit đã tải độc lập: `06f9ba3eed92e334d9a2b8003fc06d83596e5614`. `sources/inventory.json` ghi URL, SHA256 và kích thước; `sources/github_tree.json` là cây main đệ quy đầy đủ, không bị truncate. Không có đường dẫn mang tên v3.2/v32 trong cây này; loader và model đều xác nhận chỉ hỗ trợ v2.0/bốn loại. Không suy rộng kết quả sang mã riêng hoặc máy thực nghiệm của tác giả.
- U1 được kiểm tra bằng **source archive của từng lượt chạy**, đối chiếu SHA256 của 21 module với manifest; không suy từ code mới nhất trong workspace rằng đó là code đã thực thi.
- Dòng dữ liệu SPLD được kiểm tra trên tệp local khôi phục. Tải lại độc lập Software Heritage chưa thành công; repo SPLD trả 404. Giới hạn này ghi trong `external_lookup_limits.json`, không coi log cũ là bằng chứng mới xác thực kho lưu trữ.

Không huấn luyện, không chạy checkpoint/inference, không đọc dự đoán outer-test, không liên hệ tác giả, không gửi lệnh cho agent khác, không sửa pipeline. Toàn bộ tensor nhãn được đọc để kiểm tra danh tính/schema; chỉ các cặp inner-train/inner-validation đã lưu được chấm điểm.

## 2. Dữ liệu: đúng năm loại, chưa chứng minh đúng tensor DHGCMDA

Đã đo lại từ tensor và CSV:

| Thuộc tính | Giá trị |
|---|---:|
| miRNA × bệnh × loại | 411 × 271 × 5 |
| Genetics | 1.155 |
| Epigenetics | 403 |
| Circulation | 2.293 |
| Target | 3.997 |
| Tissue | 3.900 |
| Tổng typed links | 11.748 |
| Cặp có ít nhất một loại | 8.735 |
| Cặp có nhiều loại | 2.110 |

Năm CSV nguồn, `Y_multilabel.npy`, reference NPZ và strict-four-view NPZ có nhãn giống từng phần tử. Tên và thứ tự miRNA/bệnh của CSV cũng trùng reference; strict NPZ giữ nguyên tên/thứ tự và loại. Bằng chứng này xác nhận tính nhất quán nội bộ mạnh hơn việc chỉ khớp số lượng. Chi tiết: `data_identity.json`.

Nó chưa chứng minh rằng DHGCMDA đã dùng chính danh sách liên kết đó. Dữ liệu vẫn đến từ dòng SPLD, còn bản phát hành DHGCMDA không cung cấp v3.2 tương ứng trong main đã kiểm tra.

Phát hiện về lọc: `hsa-mir-1179` và `hsa-mir-1244` mỗi thực thể chỉ có **một unique typed link** trong tensor cuối. Bậc tối thiểu miRNA theo pair và typed link đều là 1; bệnh theo pair tối thiểu 2. Paper p23 nói loại thực thể có ít hơn hai liên kết được xác thực. Có thể tác giả đếm nhiều evidence record cho cùng link hoặc lọc ở bước trước; cũng có thể recipe/tensor khác. Chưa phân biệt được các khả năng. Không được gọi đây là bằng chứng chắc chắn bộ khôi phục sai, cũng không âm thầm prune để ép theo câu chữ.

Thứ tự kênh U1 là genetics/epigenetics/circulation/target/tissue. Loader SPLD local dùng target/circulation/epigenetics/genetics/tissue; mã scalar DHGCMDA dùng circulation/epigenetics/target/genetics. Việc map theo tên khi xây tensor đã khớp; nhưng thứ tự khác nhau có ý nghĩa đối với metric chỉ lấy nhãn dương đầu tiên.

## 3. Chấm lại cùng dự đoán, cố định định nghĩa trước khi tính

Định nghĩa được ghi ở `PLAN.md` trước lượt tính đầu tiên. Chỉ argmax score đã lưu, không đổi checkpoint, threshold, split hoặc chọn convention gần paper.

| F1 convention | Fold0 val: 695 cặp, epoch650 | Fold1 val: 701 cặp, epoch620 |
|---|---:|---:|
| Macro-F1 argmax-onehot vs multihot | 0,470762 | 0,433291 |
| Micro-F1 trên mọi kênh | 0,503995 | 0,466626 |
| First-label hybrid: mở rộng metric mã tác giả | 0,514719 | 0,481183 |
| Any-hit hybrid | 0,516006 | 0,487784 |
| Trung bình F1 từng cặp | 0,518609 | 0,480932 |

First-label hybrid đặt nhãn đúng thành kênh dương đầu tiên, precision = accuracy, recall = trung bình recall các lớp có nhãn đúng, rồi F1 = harmonic mean. Công thức tổng quát được kiểm tra bằng **30 fixture đa nhãn ngẫu nhiên** với hàm gốc bốn loại; kết quả khớp. Hàm gốc được trích AST từ snapshot GitHub, không import torch hay sửa code tác giả. AST hàm gốc trùng bản local.

Fixture dự đoán hoàn hảo: bốn loại + existence cho F1=1; năm loại + existence bị bỏ hết mẫu và cho F1=0. Đây là lỗi hỗ trợ định dạng, không phải mô hình dự đoán kém.

Hai tệp export có hash khớp metadata; từng pair và nhãn khớp tensor; hash pair/nhãn khớp manifest; train và validation rời nhau. Macro/micro-F1 tính độc lập khớp manifest và metric archived đến sai số dưới 1e-12. Một số trường telemetry `*_true_hist` là histogram nhãn đầu tiên, không phải tổng support multihot; bảng per-class trong kết quả audit dùng support thực từ `y`.

### Độ nhạy với thứ tự kênh

Đã thử toàn bộ 120 permutation đồng thời của nhãn và score. Không có argmax tie ở hai validation export.

| Metric | Fold0 | Fold1 |
|---|---:|---:|
| First-label hybrid: min–max | 0,464934–0,518448 | 0,460244–0,501653 |
| Macro-F1/any-hit | Không đổi | Không đổi |

Không chọn permutation nào làm kết quả chính. Ngay cả convention kế thừa mã tác giả cũng cần xác nhận encoding/thứ tự và cách xử lý đa nhãn trước khi gọi là metric paper.

### Baseline theo bệnh trên đúng cặp validation

Audit dựng thêm một prior chỉ từ inner-train: đếm support từng loại theo bệnh; tie-break bằng support toàn cục train; bệnh chưa thấy dùng support toàn cục. Quy tắc cố định trong script trước khi chạy. Đây là diagnostic riêng, không giả định giống recipe prior trong báo cáo cũ.

| Chỉ số | Fold0 U1 | Fold0 prior | Fold1 U1 | Fold1 prior |
|---|---:|---:|---:|---:|
| Macro-F1 | 0,470762 | 0,353269 | 0,433291 | 0,334912 |
| Micro-F1 | 0,503995 | 0,540873 | 0,466626 | 0,524189 |
| Any-hit | 0,589928 | 0,633094 | 0,543509 | 0,610556 |

Đánh đổi macro/micro tồn tại cả khi baseline và model được chấm cùng code trên cùng cặp. Nó không chứng minh một lỗi triển khai duy nhất. Các số là validation đã chọn checkpoint sau 65 lần nhìn, không phải outer performance hay trung bình CV năm fold.

## 4. Objective cuối cùng: cần sửa cách đọc fidelity audit

Nguồn thực thi: `paper_guided/u1_horizon650_fold0_rev1/source/stages/{u1_runner,objective_units,c1_runner,reconstruction_adapter}.py` và source archive tương ứng fold1. SHA các module đã được kiểm tra theo manifest.

Đặt `T(w)` là tổng weighted multihot type CE; `I` là tổng intra; `J` là tổng InfoNCE cộng một global margin với hệ số 0,1; `R_mask` là tổng squared reconstruction trên các ô hợp lệ; `M` là tổng weighted positive mass. U1 chạy:

`L_U1 = [T(w_normalized) + 1.0 I + 0.3 J + 1.0 R_mask] / M`.

Adapter reconstruction trả mean, nhưng `combine_objectives` nhân lại valid count. Intra mean cũng được nhân lại tổng node count. Vì vậy nhận xét “U1 cuối cùng dùng reconstruction mean thay cho sum” sẽ không đúng cho đường chạy này. Mask/target choice vẫn khác một Frobenius đầy đủ và cần ghi là adaptation.

Với `w_normalized = c * w_Eq34`, đo được `c=849,662663` ở fold0 và `843,237264` ở fold1. Nhân toàn bộ loss bởi `M/c` cho công thức tương đương về tỷ lệ thành phần:

`T(w_Eq34) + (1/c) I + (0.3/c) J + (1/c) R_mask`.

Điều này chứng minh hệ số 1/0,3/1 trong config chưa bảo đảm tương đương với cách đọc **literal raw-weight sums** của Eq32-34. Nó không chứng minh tác giả thực tế dùng raw weights: **mã công bố cũng chuẩn hóa sum-C**. Vì vậy không được kết luận hệ số ~850 là nguyên nhân gap với mã tác giả, hoặc tự sửa nó chỉ vì thấy lớn. Hằng số scale tổng loss cũng không bảo đảm quỹ đạo optimizer có weight decay/epsilon giống nhau.

Kiểm tra số học từ telemetry tại checkpoint đã chọn:

| Thành phần loss sau scale | Fold0 epoch650 | Fold1 epoch620 |
|---|---:|---:|
| Type | 1,034892 | 1,147507 |
| Intra | 0,818069 | 0,822919 |
| Inter | 1,988246 | 1,988594 |
| Reconstruction | 1,724860 | 2,743218 |
| Tổng | 5,566067 | 6,702238 |

Tổng khớp loss recorded trong tolerance float32. Đây là contribution vào **giá trị scalar**, không đo gradient dominance và không xác định nguyên nhân hiệu năng.

Audit fidelity cũ gắn “MATCH” khi nhìn intermediate reductions và coi declared choice là đủ để loại vấn đề fidelity là quá mạnh. Cách phân loại đúng: có thể là code hợp lệ, đã khai báo, nhưng mức tương đương với tác giả vẫn chưa chứng minh.

## 5. Những khác biệt đã xác nhận giữa ba đối tượng

`FIDELITY_MATRIX.csv` có 18 mục, tách paper, mã công bố và U1. Những điểm cần giữ trong quyết định:

- **Metric:** U1 primary macro-F1 khác released hybrid; không được gọi một con số là bản mở rộng đơn thuần của số kia.
- **Readout:** source tác giả ghi rõ và thực hiện diagonal bilinear. Paper chỉ nói bilinear nên không xác định dense hay diagonal. Full-bilinear là capacity sensitivity, không mặc nhiên là sửa lỗi fidelity.
- **Objective:** mã phát hành dùng `.3*focal_exist + .7*type_CE`, label smoothing 0,1, type scores đã qua softmax được đưa vào `F.cross_entropy`, rồi thêm CL/reconstruction/regularizer với cách scale riêng. U1 dùng multihot log-softmax CE trên logits và không có existence objective. U1 có thể sát cách đọc phương trình hơn ở vài điểm, nhưng không phải release replay.
- **Fusion:** U1 learned gates theo phương trình; mã công bố fixed 0,6/0,4. Sửa theo phương trình thay đổi mô hình so với mã gốc; phải nêu rõ đang tái hiện đối tượng nào.
- **Horizon:** source `param.py` công bố mặc định **650 epoch**. Mốc 200 là giới hạn lựa chọn của track local, không phải bằng chứng paper cũng dùng 200. Đối chứng 200→650 vừa đo undertraining local vừa tiến gần budget của release, nhưng không chứng minh hội tụ hay tái hiện F1.
- **Optimizer:** paper nói Adam và scheduling/validation stopping; source chạy AdamW có clip gradient; U1 archive dùng Adam với fixed horizon/best validation. Recipe chưa trùng hoàn toàn.
- **Feature/split:** danh tính view/recipe/fold của tác giả chưa có. Các chế độ train-only hay grouped split có thể tốt cho một bài đánh giá nghiêm ngặt hơn, nhưng không tự đồng nhất với protocol paper.

## 6. Đọc lại số liệu paper và các cách diễn giải cần tránh

Render mới PDF p30 cho Fig4(a) v3.2: **P=0,6005, R=0,5253, F1=0,5678**. Một audit trước ghi 0,6055/0,5353; hai nhãn P/R đó bị đọc sai. F1 0,5678 và sự khác biệt với Bảng3 0,8600 vẫn được xác nhận. Fig4(a) v2.0 ghi F1 **0,5970**, không phải khoảng 0,5978. Ảnh đọc trực tiếp: `sources/figure4a_v32_detail.png`; số liệu: `published_targets.json`.

Harmonic mean của P/R đã in không đồng nhất với một số F1 đã in. Không thể dùng điều này riêng lẻ để kết luận paper sai: trung bình F1 theo fold không nhất thiết bằng F1 của P/R trung bình. Cũng không được chọn Hình4 thay Bảng3 làm target thuận lợi hơn.

Các kiểm tra Arm-T/FB2 gần đây hữu ích như sensitivity trên split đã phát triển. Chúng không xác thực tensor tác giả và không chứng minh curation là nguyên nhân chính khi giả thuyết khác cho hiệu ứng nhỏ.

Train-set overlap giữa các fold là bình thường trong CV năm fold. Điều cần công bố là test của fold nào đã tham gia quyết định nghiên cứu trước đó và có dùng test để chỉnh recipe trong lần đánh giá đó hay không. Không cần đổi sang HMDD v4 chỉ để tiếp tục tái hiện benchmark v3.2; v4 là một câu hỏi khác.

## 7. Hướng chạy tiếp đã được cụ thể hóa

1. Giữ bộ năm loại khớp bảng làm **candidate reference**, không bỏ tissue, không tự prune hai singleton. Không gọi nó là author-identical.
2. Lưu cả năm convention đã định nghĩa trên mọi export đã có, giữ macro-F1 hiện tại làm endpoint của U1, đồng thời ghi rõ first-label hybrid chỉ là diagnostic kế thừa mã. Không đổi selector sau khi thấy điểm.
3. Nếu tiếp tục tái hiện mã, dùng **release track** cố định commit trên và v2.0 thật sự phát hành; mọi sửa lỗi thực thi phải ghi thành patch riêng. Nếu tiếp tục diễn giải phương trình cho v3.2, dùng **equation-guided track** với fidelity matrix và unknowns đã khai báo. Không trộn hai track thành một baseline.
4. Trước lượt fit mới cần giải quyết một câu hỏi có thể phân biệt: xác minh recipe/feature từ tác giả, hoặc kiểm tra một khác biệt objective được chỉ định trước. Không có cơ sở từ audit này để sweep thêm epoch, beta hoặc decoder nhằm đuổi một số F1 chưa cùng định nghĩa.
5. `AUTHOR_QUESTIONS.md` là danh sách thông tin tác giả cần cung cấp, có các mâu thuẫn và ví dụ cụ thể. Chưa gửi đi vì người dùng chỉ yêu cầu chạy kiểm tra độc lập.

**Tiêu chí hoàn thành audit này:** đối chiếu nguồn được pin; kiểm tra danh tính tensor nội bộ; chấm metric cùng dự đoán với fixtures; truy công thức cuối cùng từ source đã chạy; bàn giao kết luận cùng bằng chứng có thể chạy lại. Các tiêu chí đó đã đạt. Chưa đạt và không tuyên bố đạt: tái hiện F1=0,8600, danh tính dữ liệu DHGCMDA v3.2, hoặc kết quả CV năm fold tương đương tác giả.

## Chạy lại và tệp bằng chứng

Từ gốc repo:

```powershell
uv run --python 3.12 --no-project --with numpy==1.26.4 --with pymupdf python faithful_repro/independent_reproduction_audit_20261116/audit.py
uv run --python 3.12 --no-project --with numpy==1.26.4 --with scikit-learn==1.5.2 python faithful_repro/independent_reproduction_audit_20261116/verify_audit.py
```

Script dùng snapshot nguồn đã pin khi chúng tồn tại, kiểm tra lại hash, rồi chỉ ghi vào thư mục audit. `verification.json` chứa fixture và kiểm tra export/source; `input_sha256.json` chứa hash đầu vào trước/sau; `METRIC_COMPARISON.csv` và `metrics_and_objective.json` chứa số liệu, per-class và toàn bộ permutation. `ARTIFACT_SHA256.json` là inventory bàn giao, không tự bao gồm hash của chính nó.

Kiểm tra độc lập lần cuối bằng scikit-learn 1.5.2: **4 export × 4 metric đều khớp trong 1e-12**, gồm macro, micro, samples-F1 và hybrid từ accuracy/balanced accuracy. **76 hash đầu vào được kiểm tra lại**, bảng fidelity đủ **18 dòng**. Kết quả ở `independent_verification.json`. Các con số hash/inventory là kiểm tra toàn vẹn, không làm các validation score thành ước lượng độc lập.
