# Xác minh độc lập CVtriplet baseline và phân tầng theo cặp đã thấy

Tái dựng split và feature bằng cách tính số lượng độc lập (bincount, không import script baseline), train lại logistic với đúng protocol, đối chiếu các metric từng fold với output đã pin trong sai số 1e-10. Chia riêng test theo có/không positive type khác trong train; báo số positive/unknown, AUROC/AP/F1 từng nhóm nếu đủ hai nhãn. Đây là phân tích hậu nghiệm, không kết luận nhân quả hoặc chọn/tune model.
