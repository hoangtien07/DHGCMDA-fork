# U1 validation mở rộng fold 2–4 — một seed

Mục tiêu: bổ sung kết quả mô tả trên inner-validation của 3 fold đã được xây và kiểm toán, so cùng các prior từ train. Đây là nghiên cứu sau khi đã xem nhiều kết quả phát triển và một outer-test fold0; không phải confirmatory hay external holdout và không dùng để khẳng định tái hiện paper.

Dùng code U1 lưu trong `u1_horizon650_fold0_rev1/source` đã hash theo manifest, config nguyên fold0 U1 (650 epoch, eval 10, seed 1234, torch2.5.1 CPU 2 threads, PyG2.6.1, NumPy1.26.4), input selection fold2/3/4 đã kiểm tra. Mỗi fold chạy process riêng, cùng seed và cấu hình; train trên `train_pairs`, chọn best macro-F1 từ inner-validation theo runner, lưu cả best và final trajectory. 650 chọn để so điều kiện đã quan sát ở fold0/1, không giả định hội tụ hay chứng minh giống tác giả. Không dùng outer-test, không chọn seed/cấu hình, không can thiệp các tệp lịch sử; abort/nonfinite phải báo và không ghi là thành công.

Hai fold 0/1 chạy Linux/Colab, 2–4 chạy Windows CPU, do đó sự khác môi trường là giới hạn. Mỗi pair outer-test 2–4 đã thuộc tập train hoặc validation của fold0 đã phát triển; CV thông thường vốn chồng train, nhưng các kết quả test fold0 đã quan sát trước khi thiết kế các bước sau, nên không tuyên bố test còn độc lập hoàn toàn. Train/val macro, micro, any-hit là chỉ số nội bộ đã công bố; không xem nó là F1 paper 0,8600.
