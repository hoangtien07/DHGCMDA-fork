# COUNCIL_DECISION_RECORD — REV4 Phase C

## Recommended path (exactly one): `R_CLOSE_AS_PARTIAL_REPRODUCTION`

Bằng chứng:

1. Bounded diagnostics của Phase A đã vét cạn: upstream frozen
   2026-01-16, không artifact v3.2 mới ở bất kỳ repo tác giả nào
   (MRGBMDAT là benchmark khác); draft request đã sẵn sàng nhưng
   chưa gửi.
2. Ba blocker ngoại sinh còn lại (eval convention, fold indices,
   curation filter) đều KHÔNG đóng được bằng training — chỉ đóng
   được bằng artifact từ tác giả.
3. Phase B không được biện minh: câu hỏi leakage isolation đã
   answered bằng paired diag_leak runs có sẵn (v2.0 Δ+0.049,
   v3.2 Δ≈0); Gate A3 = NO_TRAIN_NECESSITY.
4. Bằng chứng mới của REV4 làm track R đậm hơn chứ không mở:
   strict tensor khớp paper ở mọi summary statistic (density
   10.5476%, per-type proportions) → G1 thu hẹp còn "eval-input
   unverified"; và bound toán học chứng minh metric paper phải
   dùng convention khác multi-hot pooled (oracle 0.8263 < 0.9421)
   → G3 là bất khả so sánh cấu trúc, không phải do ta sai.

Track N vẫn là hướng mở hợp lệ cho một *dự án mới* — xem
NEXT_RESEARCH_DECISION.md; nó là decision riêng cần protocol
preregistered, không gộp vào record này.

## Không chọn

- `R_CONTINUE_WITH_AUTHOR_ARTIFACTS`: không có artifact nào đến.
- `N_APPROVE_NEW_RESEARCH_PROTOCOL`: benchmark sạch đã có nhưng
  chưa có hypothesis + protocol preregistered đủ chặt để phê
  duyệt trong record này; để owner quyết sau khi đọc handoff.

## Không được suy ra

- Không tuyên bố gian lận hay sai của paper: chỉ "không chứng
  minh được tương đương" + "release-eval disconnect đã chứng
  minh".
- Kết quả U1 (0.46–0.49) là retrospective trên benchmark
  reconstructable, không phải con số cùng protocol với 0.86.
