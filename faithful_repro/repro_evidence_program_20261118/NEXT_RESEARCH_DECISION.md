# NEXT_RESEARCH_DECISION — phân tích cho owner (REV4 Phase C)

## Hiện trạng

Track R đóng ở `R_CLOSE_AS_PARTIAL_REPRODUCTION`. Nếu owner muốn
tiếp tục nghiên cứu, hai lựa chọn thực tế:

### Lựa chọn A — Gửi author request (low cost, high potential)

`AUTHOR_ARTIFACT_REQUEST_READY.md` đã sẵn sàng (5 nhóm artifact,
kèm lập luận oracle bound 0.8263 làm động lực rõ ràng cho yêu cầu
eval code). Chi phí: 1 email/issue. Lợi ích: đóng G1–G4 trong ~1
ngày tính nếu tác giả trả lời. Xác suất phản hồi không đo được.

### Lựa chọn B — Track N: benchmark + method mới

Benchmark sạch đã tồn tại (strict tensor + pair-fold + metric
contracts + priors làm baseline). Câu hỏi khoa học khả thi từ
dữ liệu REV4:

- **PU-aware type prediction trên multi-label 5-type**: priors
  train-only đã ~90% model trên aggregate → contribution thực
  nằm ở minority types và multi-type pairs (U1 thắng priors rõ
  trên `multitype_anyhit`/`singletype_acc` strata trong
  METRIC_BRIDGE.json — arm tốt hơn phải chứng minh trên
  strata đó, không chỉ aggregate).
- **Generalization tracks** theo roadmap: entity cold-start
  (miRNA-group/disease-group split) — chưa tồn tại, cần build.
- **Baselines phải có**: count priors, multi-label logistic,
  label propagation, TDRC (đã tái hiện ở mức tensor), U1 static.
- **Không dùng** kết quả BCE F1=0.0 lịch sử — đo bằng metric k4
  hỏng; nếu có stored logits thì rescore trước khi kết luận.

Cần trước khi chạy: protocol preregistered (splits cold-start
mới, primary metric khóa trước, canary, power/target effect).

## Khuyến nghị thứ tự

A trước (rẻ, đóng câu hỏi gốc), B sau nếu tác giả không trả lời
trong ~2 tuần hoặc owner muốn pivot ngay — hai track độc lập.
