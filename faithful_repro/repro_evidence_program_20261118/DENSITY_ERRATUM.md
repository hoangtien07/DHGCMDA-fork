# DENSITY_ERRATUM — sửa lỗi mật độ "4.6%" (REV4 WP-A0, mandatory)

## Erratum

Phát biểu lặp lại trong các báo cáo của program 20261118 —
"strict density ~4.6% vs paper 10.5%" — là **SAI và RETRACTED**.

Đo trực tiếp trên `v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz`
(sha256 85bb9ea4…, 411×271×5):

| Đại lượng | Tử | Mẫu | Giá trị |
|---|---|---|---:|
| typed-link density | 11,748 links | 411×271 = 111,381 cặp | **10.5476%** |
| positive-pair density | 8,735 cặp dương | 111,381 | 7.8425% |
| typed-cell density | 11,748 | 111,381×5 | 2.1095% |

→ Strict tensor có typed-link density **khớp** con số paper công bố
(~10.5%), không phải "khác 4.6%". Nguồn gốc con số 4.6%: không truy
được trong artifact nào — có thể lẫn với tensor wang (713×447,
12,534/318,831 = 3.93%) hoặc ước lượng sai khi viết báo cáo; ghi nhận
là lỗi phát biểu không có nguồn đo, không phải số đo.

## Claims phụ thuộc vào "4.6%" — trạng thái

| Vị trí | Claim | Trạng thái |
|---|---|---|
| FINAL_RESEARCH_VERDICT.md §"Không được suy ra" | "density ~4.6% vs 10.5%" là bằng chứng tensor khác | **RETRACTED** — density khớp; tensor khác ở chỗ chưa chứng minh được byte-identity, không phải độ dày |
| THESIS_RESULTS_AND_LIMITATIONS_VI.md §Giới hạn | "bài toán khó hơn do density" | **RETRACTED** — cùng summary statistics; khác biệt còn lại là fold/feature/eval input chưa xác minh |
| ADVERSARIAL_REVIEW.md A1, R2 | phòng thủ dựa trên density gap | **RETRACTED** — thay bằng: không chứng minh được same-input |
| RESCIENCE_NOTE.md §Discussion G1 | "our strict tensor is ~4.6% density vs their 10.5%" | **RETRACTED** — G1 viết lại là "author evaluation input not proven byte-identical" |
| FINAL_HANDOFF.md §3 | "Density tensor 4.6% ≠ 10.5% paper" | **RETRACTED** |

Các file gốc được giữ nguyên lịch sử; erratum này là bản ghi chính
thức thay thế. Bản sửa đánh dấu được append vào từng file.

## Hệ quả khoa học (mạnh hơn trước)

Class proportions của strict ([9.83%, 3.43%, 19.52%, 34.02%, 33.20%])
khớp tỷ lệ paper công bố (epigenetics 3.43%, target 34.02%) →
bằng chứng mạnh strict tensor cùng lineage benchmark public với
tensor paper. Điều CHƯA chứng minh được: byte-identity của input đánh
giá tác giả (feature views đi kèm + fold indices + eval script).
