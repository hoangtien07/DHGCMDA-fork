# Đối chứng tương đồng chức năng theo train

So với `independent_v20_trainweights_20261116`, giữ nguyên fold, seed, loss, class weights train-only, association/GIP, disease similarities, mô hình và 5 × 650 epoch. Thay M_FSM tĩnh bằng similarity tính lại từ tập bệnh train cho từng miRNA và disease semantic D_SSM1. Với hai tập bệnh, điểm là trung bình có trọng số theo số bệnh của tổng max similarity hai chiều; chéo đặt 1 cả khi một miRNA không có bệnh train. Đây là đặc trưng thay thế theo họ Wang, không khẳng định trùng M_FSM tác giả hay quy trình paper. D_SSM1, D_SSM2 và negative sampling vẫn là đầu vào cần provenance.

Test-first công thức bằng ví dụ tính tay, symmetry/diagonal, zero-degree và test-label invariance; kiểm tra mọi nơi nhận M_FSM và tính lại fallback IM từ feature mới. Lưu hash matrix theo fold, đối chiếu train/test và class weights; smoke 5 × 1, sau đó full 5 × 650; chấm lại cùng metric, so từng fold với trainweights. Không chọn seed theo kết quả, không gọi đây là tái hiện paper.

Tham khảo phương pháp gốc: Wang et al. 2010, https://doi.org/10.1093/bioinformatics/btq241 ; Li et al. 2019 https://pmc.ncbi.nlm.nih.gov/articles/PMC6602518/ . Công thức là một lựa chọn đối chứng; M_FSM v2.0 không được khẳng định cùng nguồn.
