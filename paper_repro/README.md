# Thực thi kế hoạch DHGCMDA

Nguồn kế hoạch `docs/strategy_thesis_dhgcmda.md`. Mã phát hành dùng cho tái hiện là commit `06f9ba3`. Mã kiểm soát, dữ liệu dựng lại và kết quả chạy nằm trong thư mục này.

| Giai đoạn | Entry point | Kết quả và điều kiện |
| --- | --- | --- |
| 0–1 | `audit_release.py`, `run_v20.py smoke` | SHA256 mã nguồn và dữ liệu, GIP v2.0, kiểm tra evaluator và smoke GPU |
| 2 | `continue_calibration.ps1` | C1 → G2 → C2 → C3 → `runs/calibration_lock.json` |
| 3 | `prepare_v32.py`, `continue_reproduction.ps1` | V1 và V2 theo khóa, xử lý G3 rồi chấm lại các dự đoán |
| 4–5 | `clean_experiment.py all` | Chia chung 2 dataset × 3 seed × 5 fold, phân rã Stage 4 ở seed 1234, chọn mô hình theo validation và so sánh ghép cặp |
| 6 | `report_thesis.py` | Bảng tái hiện, bảng phân rã, so sánh cải tiến và giới hạn từ kết quả đã lưu |

## Các lựa chọn thực thi

- Dùng Windows, Python 3.12 và PyTorch cu118 trên GTX 1660 SUPER 6 GB. Chạy mỗi lần một tiến trình thực nghiệm với hai luồng CPU. Bản khóa gói nằm trong `windows-cu118-lock.txt`.
- C2 giảm `exist_weight` xuống 0,1 và giữ `type_weight` ở 0,7. C3 reseed sau import với 42 để thay đổi seed thực sự, trong khi mã chia fold gốc vẫn dùng NumPy seed 0.
- G3 áp ngưỡng khớp cho cả hai hàng v3.2. Với trường hợp gần, chạy đúng hai đối chứng trên T*, 4 head và thêm GIP bệnh. Với trường hợp nằm ngoài ba vùng đã mô tả, ghi `inconclusive` và chuyển sang đánh giá có kiểm soát. V3 chạy seed 42 và 2025 trên profile khớp gần hơn khi cả hai hàng đạt ngưỡng.
- Dữ liệu v3.2 có 11.748 triplet, 8.735 cặp dương. GIP miRNA lấy từ hợp nhị phân của các triplet hiện có. View bệnh thứ hai của lượt tái hiện chuẩn là bản sao D_MESH. Ánh xạ năm loại đối chiếu với từng ma trận nguồn. `PORT_DIFF.md` ghi phần port.
- Stage 4 giữ cùng các cặp và đổi từng yếu tố so với legacy. Mã phát hành dùng epoch cuối cố định, nên đối chứng validation thay epoch cuối bằng epoch được chọn trên inner validation và thay ngưỡng nhị phân theo tập test bằng ngưỡng theo validation. Đối chứng đa nhãn dùng đích multi-hot trong loss và vẫn giữ kích thước đầu vào.
- Stage 5 dùng cặp âm cố định 1:1 và dành 20% mỗi outer-train cho inner validation. GIP hai phía và functional similarity lấy từ fit. D_SSM1 là view MeSH hiện có, có ghi nguồn và hash. Validation chọn epoch tại các mốc cách 50 epoch và chọn ứng viên trong từng fold. Trong bước xác nhận, outer test chỉ chấm ứng viên được chọn và mô hình sạch đối chứng.
- Reconstruction của mã `06f9ba3` đã dùng mean. Ứng viên đầu tiên dùng trọng số học được cho loss dự đoán, CL và reconstruction. Ứng viên tiếp theo dùng decoder full bilinear, rồi bỏ tầng HGT và trọng số CL. Đối chứng đích multi-hot riêng chạy trên v3.2.
- TDRC và SPLD dùng bản mã trong repo, với tensor fit và view fit theo giao thức chung. Điểm nhị phân của các baseline kiểu theo loại lấy điểm lớn nhất trên các loại. So sánh ghép cặp ghi mean, SD và p-value mô tả. Các fold lặp qua seed dùng lại dữ liệu benchmark.
- Kết quả từng fold giữ checkpoint và hash để đọc lại. Batch tiếp tục từ run đã hoàn tất. Nếu có thư mục chạy dở, batch giữ dữ liệu để chẩn đoán.
- Các lượt chạy mới ghi số bước optimizer theo fold. Fold dừng sớm sẽ báo lỗi trước khi ghi kết quả hoàn tất. C1 đã khởi chạy trước thay đổi này, nên số epoch của C1 được đối chiếu từ log từng fold.

## Theo dõi trên máy hiện tại

Các log ở thư mục `work` của phiên Codex, gồm `track-p5-c1.log`, `track-p5-calibration.log`, `track-p5-reproduction.log` và `track-p5-thesis.log`, cùng các file `.err.log` tương ứng. `runs` giữ `config.json`, dự đoán và `completion.json` của từng lượt. G2 không đạt thì chuỗi dừng ở hiệu chuẩn. G3 quyết định có thêm seed, chẩn đoán giới hạn, hoặc chuyển sang đánh giá có kiểm soát.

Nếu chạy lại từ PowerShell sau khi không còn tiến trình đang chạy, vào thư mục repo và gọi script của giai đoạn tương ứng:

```powershell
pwsh -NoProfile -File paper_repro/continue_calibration.ps1
pwsh -NoProfile -File paper_repro/continue_reproduction.ps1
pwsh -NoProfile -File paper_repro/continue_thesis.ps1
```
