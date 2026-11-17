# Đầu vào v3.2 và baseline inner-validation cho 5 fold

Ngày 2026-11-17. Đã build các bundle selection fold2–4 bằng builder/nguồn pin của fold0/1 và kiểm toán độc lập từng bundle. Có bốn baseline số đếm train-only trên inner-validation đủ 5 fold; đây là mốc phát triển, **không phải outer-test hay F1 paper**.

| Fold | Train pairs | Validation pairs | Global macro-F1 | Disease macro-F1 | miRNA macro-F1 | Product macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.293 | 695 | 0,1209 | 0,3533 | 0,2489 | 0,4156 |
| 1 | 6.287 | 701 | 0,1235 | 0,3349 | 0,2707 | 0,3960 |
| 2 | 6.286 | 702 | 0,1292 | 0,3800 | 0,2523 | 0,4439 |
| 3 | 6.284 | 704 | 0,1242 | 0,3661 | 0,2692 | 0,4131 |
| 4 | 6.288 | 700 | 0,1274 | 0,3429 | 0,2759 | 0,4574 |
| Trung bình | — | — | **0,1250** | **0,3554** | **0,2634** | **0,4252** |

Micro-F1 trung bình tương ứng 0,3888 / 0,5384 / 0,4035 / 0,5252; any-hit 0,4548 / 0,6299 / 0,4720 / 0,6145. Product là heuristic log count với Laplace 1 được ghi trước tại `BASELINE_PROTOCOL.md`, không phải xác suất hiệu chuẩn hay kết quả chọn siêu tham số. Thứ tự các cột không hàm ý lựa chọn một model. Macro/micro là quy ước one-argmax so với multihot đã công bố, không khẳng định là metric paper.

U1 selection 650 epoch đã có ở fold0/1: macro-F1 validation 0,4708 / 0,4333. So với product trên cùng fold là 0,4156 / 0,3960; U1 checkpoint được chọn theo chính validation, product không có bước chọn checkpoint. Không lấy trung bình hai U1 để gọi là kết quả 5-fold. Các lượt bổ sung được ghi riêng theo `U1_PROTOCOL.md`.

## Kiểm toán đầu vào mới

`audit.json` xác nhận fold2/3/4 có train 6.286/6.284/6.288 cặp, validation 702/704/700, outer-test 1.747 mỗi fold. Train mask không chứa cặp val/test và không có group candidate bị cắt qua segment. Bốn block association khớp mask/transpose; nhãn train/val khớp Y; hash mảng và bundle khớp metadata. Semantic và functional tính lại từ nguồn pin, train mask khớp elementwise. S_seq/S_gene khớp strict artifact; 262/271 bệnh có semantic coverage theo recipe, lần lượt 409/411/409 miRNA có association train. Không đọc test labels để chấm điểm.

## Exposure và diễn giải

Hồ sơ `CANDIDATE_FOLD0_OUTER_EVALUATION_2026-11-15.md` ghi một lượt outer-test fold0 đã chấm với checkpoint refit 200 epoch, sau đó có chẩn đoán phát triển. Không được coi lại nó là outer-test hoàn toàn chưa tiếp xúc. Kiểm toán chỉ số `exposure.json` cho thấy toàn bộ 1.747 outer-test của mỗi fold 2–4 nằm trong train hoặc validation của fold0: lần lượt 1.583/164, 1.565/182, 1.562/185. Đó là cấu trúc bình thường của cross-validation, không tự nó là lỗi; nhưng quan sát outer-test và thay đổi recipe sau đó làm hạn chế tuyên bố rằng các lượt tiếp theo là kiểm định hoàn toàn độc lập với quá trình phát triển.

`validation_baselines.json` chứa metrics từng lớp và hash bundle, `audit.json` chứa chứng cứ build, `exposure.json` chứa giao cắt theo index; toàn bộ là dữ liệu nội bộ. Các đặc trưng sequence/gene/semantic vẫn có giới hạn nguồn và chưa chứng minh giống tác giả; code được pin là một bản diễn giải, không chứng minh công thức F1 0,8600.
