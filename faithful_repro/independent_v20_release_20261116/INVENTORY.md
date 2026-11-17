# Kiểm kê v2.0 trước replay nguyên trạng

Ngày 16-11-2026. Đây là kết quả từ dữ liệu/log có sẵn và smoke, không phải kết quả huấn luyện 650 epoch mới.

## Lượt baseline cũ

- `logs/baseline_v2.0_full.log` có năm kết quả fold `.5531/.5664/.5364/.5643/.5404`; file `results/baseline_v2.0_metrics.json` ghi Top-1 F1 trung bình `.5521`, khớp log. Log được đưa vào git tại `1cf27bda69d6af981115b62840b2fd8913a559fb`; điều này không tự xác thực snapshot code được thực thi khi log tạo ra.
- Log khai báo 650 epoch, seed 1234, **4 heads**, và **đã nạp cache** cho indices. Commit công bố `06f9ba3eed92e334d9a2b8003fc06d83596e5614` mặc định **8 heads**. Giữa commit nguồn và commit chứa log có thay đổi ba file `hetero_model.py`, `main_experiments_hetero1.py`, `param.py` (170 dòng thêm/47 xóa theo `git diff --stat`). Vì vậy không thể xem `.5521` là replay nguyên cấu hình release.
- Cặp test cụ thể, predictions, danh tính cache và hash môi trường không được log đó chứng minh. Số `.5570` có trong markdown tổng kết cũ nhưng không tìm thấy log/result trực tiếp trong phạm vi `logs/` và `results/` đã tìm; chưa xác nhận provenance của số này.
- `existing_baseline_audit.json` có hash, log fold và so sánh 19 file local với release; 7/19 giống byte với bản release tại thời điểm kiểm tra.

## Bản phát hành và smoke

19 blob mã/dữ liệu từ commit release được materialize bằng `git cat-file`, xác minh SHA Git và SHA256 theo tree GitHub đã lưu. Mỗi lần chạy dùng thư mục dữ liệu/cache riêng. Smoke 1 epoch × 5 fold đã hoàn thành; số điểm smoke chỉ kiểm tra kỹ thuật, không dùng làm baseline hiệu năng.

Chạy lại offline metric trên capture smoke cho kết quả khớp evaluator nguyên bản từng fold và trung bình 5 fold; macro/micro-F1 đa nhãn được tính độc lập trên đúng cặp và score đó. Không thấy các lỗi fallback được liệt kê trong `runs/smoke/metric_audit.json`. Hook xác nhận không đổi trạng thái RNG NumPy, Torch, Python khi ghi từng fold.

## Phát hiện về thông tin test

Mỗi fold smoke có 268 cặp dương test. Ở **268/268 cặp mỗi fold**, giá trị loại scalar của cặp test trong `md_p` bằng nhãn scalar test; `train_epoch_optimized` nối toàn bộ `md_p` vào feature và dựng graph từ đó. Tức là mã release đưa **nhãn loại test**, không chỉ thông tin cặp có tồn tại, vào đầu vào. Đây là quan sát về mã công bố và kiểm chứng trên split smoke; nó không chứng minh implementation đã tạo bảng paper cũng như vậy. Xem `feature_scalar_pos` trong từng fold NPZ và `split_audit.json`.

`feature_oracle.py` cố ý đọc chính trường type bị lộ từ input rồi chọn loại tương ứng: Top-1 F1 của metric gốc bằng **1,0 trên từng fold và cả 1.340 cặp**. Đây là phép chứng minh thông tin test có sẵn, không phải kết quả model hoặc phép đo tổng quát. Với truth multihot, oracle đạt macro-F1 trung bình `.9180`, micro-F1 `.9407` và any-hit `1,0`; macro/micro vẫn dưới 1 vì một dự đoán Top-1 không thể bao phủ mọi nhãn của một cặp.

Trong 1.498 positive pair, source chia seed NumPy 0: 1.340 cặp vào năm test fold, 149 cặp vào tập independent được định nghĩa nhưng `main_optimized` không chấm, 9 cặp còn dư không thuộc hai phần đó. Mỗi fold test có 268 cặp; độc lập reconstruction đối chiếu đúng thứ tự test pairs/negative pairs đã capture ở cả 5 fold. Điều này khác câu mô tả chia toàn bộ pair thành năm phần của paper, và bản scalar loader cũng ghi đè nhiều nhãn thành một nhãn.

Từ CSV 1.679 typed links tại 1.498 cặp, có 161 cặp đa nhãn và **181 typed labels bị loại khỏi biểu diễn scalar**. Cụ thể circulation 443→367 (mất 76), epigenetics 199→157 (mất 42), target 356→293 (mất 63), genetics 681→681. Trong CSV các loại được sắp theo thứ tự tăng; last-write vì vậy ưu tiên type có mã lớn khi cùng pair có nhiều loại. `label_collapse_audit.json` xác minh điều này từ source CSV và từng cặp test smoke; CSV gốc không bị thay đổi.

## Giới hạn

Trung bình release metric và macro-F1 đa nhãn, khi có lượt 650, đều dựa trên cùng thông tin test đã lộ và split của release; chúng là **baseline hành vi bản phát hành**, không phải ước lượng CV_type sạch hay bằng chứng xác định 0,5970 trong paper. Protocol tác giả thực nghiệm, fold gốc và convention paper vẫn chưa được xác minh.
