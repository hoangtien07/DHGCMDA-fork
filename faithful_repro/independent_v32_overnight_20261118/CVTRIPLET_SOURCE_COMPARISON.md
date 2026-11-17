# Khác biệt mô tả chia tập CVtriplet giữa SPLD và DHGCMDA

Ngày 2026-11-18. Hai bài sử dụng cùng thuật ngữ nhưng mô tả đơn vị chia tập khác nhau:

- [SPLDHyperAWNTF (2022), mục Implementation details and evaluation metrics](https://academic.oup.com/bib/article/23/6/bbac390/6720405): CVtype chia các **cặp dương** thành K phần; phần mô tả CVtriplet cho biết tập train/test chia theo cùng cách với CVtype. Bài còn nêu việc lấy unknown làm negative số lượng bằng positives và lấy mẫu lặp lại để đánh giá. Điều này là mô tả của bài SPLD, chưa chứng minh từng chi tiết code hoặc split thực thi đã công bố.
- [DHGCMDA (2026), mục Evaluation settings](https://link.springer.com/article/10.1186/s12859-026-06436-w): CVtriplet được mô tả là chia các **triplet dương** thành năm phần, sau đó ghép unknown để kiểm tra. CVtype được mô tả là chia cặp dương. Code và fold v3.2 đúng experiment chưa được công bố trong nguồn đã kiểm tra.

Với tensor ứng viên local, mô phỏng chia theo triplet trong `triplet_audit.json` cho thấy trung bình 38,2% positive triplet test có cùng cặp nhưng type dương khác ở train; mô phỏng chia theo pair không cho phép điều đó. Đây là hệ quả của đơn vị chia và nhiều nhãn/cặp, không tự động là sai lầm hay chứng minh có rò rỉ ngoài task được chọn.

**Hàm ý để tái hiện:** Không so trực tiếp các điểm CVtriplet từ hai bài như cùng protocol chỉ vì nhãn metric trùng nhau. DHGCMDA có thể đã chạy lại các baseline trong cùng setting của họ, hoặc sử dụng những thực hiện khác; nguồn hiện tại không xác minh được cách họ tạo từng số trong bảng. Cần có indices fold, định nghĩa unknown và predictions của từng baseline để kết luận khả năng so sánh. Không khẳng định kết quả DHGCMDA bị sai chỉ từ khác biệt trong hai mô tả.
