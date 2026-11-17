# Retry 2: đối chứng sau khi repeat bit-for-bit không đạt

Lượt đầu dừng vì trace cùng dòng bị gọi nhiều lần. Retry1 sửa trace và đạt đến đầu epoch226 nhưng dừng vì kiểm tra hai lần forward/backward cùng đồ thị không bit-for-bit giống nhau; không có số đối chứng được chấp nhận. Sai khác chi tiết chưa được ghi lại; không giả định nguyên nhân. Quỹ đạo cũng lệch nhẹ lượt 650 gốc trong các epoch trước.

Retry2 giữ seed42, nguồn và bundle, thời điểm epoch225→226 và hai nhánh đồ thị như trước, tách namespace và lưu state model, hai đồ thị, RNG, input và targets trước phép thử để có thể kiểm tra lại. Hai nhánh dùng state/RNG bắt đầu giống nhau, không bước optimizer. Nhánh new lặp lại phải có RNG cuối giống nhau, loss sai khác tương đối ≤1e-5 và norm sai khác gradient / max(1,norm gradient) ≤1e-4; các sai khác thực tế phải được ghi. Đồng thời new phải khớp loss epoch226 do runner thực sự tính trong lượt này ≤1e-5 tương đối và grad norm ≤1e-4 tương đối. Nếu bất kỳ gate thất bại, không kết luận.

Đây là dung sai số học đặt trước khi xem giá trị lặp của retry2, không chứng minh tính tất định. So sánh old/new chỉ diễn giải khi hiệu ứng lớn hơn sai khác lặp và là hiệu ứng cục bộ trên state retry2. So sánh graph hash/quỹ đạo với lượt 650 gốc vẫn ghi riêng; không gọi tái hiện spike gốc nếu không khớp. Test toy tiếp tục yêu cầu bitwise. Không chấm outer-test hoặc chỉnh mã tác giả.
