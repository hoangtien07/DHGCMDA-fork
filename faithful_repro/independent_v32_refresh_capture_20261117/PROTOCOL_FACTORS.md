# Phân tách hậu nghiệm sự kiện spike epoch 226

Sau khi đã quan sát spike và kết quả đối chứng old/new, dùng checkpoint spike_0226.pt và cùng bốn tổ hợp G, cạnh cũ/mới để phân tách thăm dò. Kiểm tra loss ở hai nhánh old/old và new/new tái hiện kết quả checkpoint, dung sai 1e-5; lặp gradient và RNG như PROTOCOL.md. Tổ hợp trộn là can thiệp nhân tạo; kết luận chỉ tại snapshot của sự kiện này, không chứng minh tính phổ biến hay F1.
