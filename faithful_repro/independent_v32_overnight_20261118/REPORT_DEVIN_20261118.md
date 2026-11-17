# Báo cáo Devin 2026-11-18 — cap1 5×3 hoàn tất trên Linux + cập nhật evidence

**Tiến độ plan:** Ưu tiên 1 (thu và xác minh cap1 5×3) — **HOÀN TẤT 15/15 PASS + VERIFY PASS** trên namespace Linux mới; ưu tiên 2–4 đã kiểm chứng bổ sung (provenance 4 view, upstream re-check, đề xuất protocol CVtriplet-U1); ưu tiên 5 là báo cáo này.

## 1. Kết quả cap1 vs cap10 — KẾT LUẬN ÂM TÍNH

Thí nghiệm paired 5 fold × 3 seed, mọi thành phần giữ nguyên, chỉ `inter_cap_mult` 10→1, 650 epoch, inner-validation (đã dùng phát triển — không phải test mù, không phải F1 paper):

| Arm | Best macro | Best micro | Best any-hit | Final macro | Final micro | Final any-hit |
|---|---:|---:|---:|---:|---:|---:|
| cap10 (baseline lịch sử, Windows) | 0.4427 | 0.4728 | 0.5532 | 0.4170 | 0.4451 | 0.5207 |
| cap1 (Linux, namespace mới) | 0.4370 | 0.4656 | 0.5448 | 0.4115 | 0.4381 | 0.5126 |

- 15/15 ô hoàn tất, **không có lượt lỗi** (`results.json` missing=[]); `verification.json` PASS: hash, pairs, truth và macro/micro/any-hit tính lại bằng sklearn khớp tuyệt đối cả hai arm.
- Theo ô: cap1 thắng 7/15 ô best-macro (lớn nhất fold4_seed42 +0.011), thua 8/15 (lớn nhất fold0_seed1234 −0.043); mean mô tả cap1 thấp hơn ~0.006. Không suy 15 ô là quan sát độc lập.
- **Kết luận:** giảm negative bank 10→1 **không cải thiện** — nhất quán với đọc protocol: cap1 đổi cả số negatives lẫn mẫu số InfoNCE/objective, không cô lập được tác động false-negative riêng. Theo quy tắc đã pin: dừng sweep cap tương tự.
- So snapshot Windows (6 ô đầu): Windows cap1 fold0_seed1234=0.4384 vs Linux 0.4282; seed42=0.4705 vs 0.4477 — chênh trong biên độ OS/BLAS đã công bố, không thay kết luận.

## 2. Bằng chứng mới kiểm chứng được trong session này

1. **Provenance bốn view — đã xác minh trên dữ liệu thật** (trước đây chỉ ghi "chưa xác thực"): bundle `candidate_fold_data.npz` dùng S_seq (NW-alignment GFF3 strict, +1/−1/−2/−1, max-arm, offdiag-minmax) ≡ `sim_m1` strict; S_gene (FunSim kiểu Cheng, CTD marker/mechanism + HumanNet-XC) ≡ `sim_d1`; S_sem rebuild từ tên bệnh qua MeSH-2019 archive (HMDDv3.xlsx/meshToID/mtrees2019) ≡ rebuild `sem.S`; S_func = `functional_similarity(A_train, sem)` MISIM-default augmented one-hot cosine trên descriptor-key — **train-only, rebuild theo từng fold**. Hai ma trận shipped `M_MISIM`/`D_MESH` (nguồn association-derived) **không** được dùng trong bundle. → Tầng feature không chứa full-association leakage; phần "không khớp paper" còn lại là chưa xác minh cùng recipe tác giả, không phải rò rỉ.
2. **Re-check nguồn sơ cấp:** upstream CDMBlab/DHGCMDA vẫn chỉ `main@06f9ba3e` (push cuối 2026-01-16), không tag/release/issue/data mới → tensor/fold/metric v3.2 của tác giả tiếp tục là nguồn đóng; không có bằng chứng mới.
3. `verify_core.py` PASS trên clone mới: 12 artifact phân tích + cross-check nguyên vẹn.
4. `bank1_linux/preflight.json` PASS 15/15 trước khi fit: mọi hash bundle/manifest/cfg/expected/bank khớp; preflight mới ghi path+hash Linux (`bank1_linux/NAMESPACE.md` mô tả quan hệ với bank1 Windows).

## 3. Bước tiếp theo đề xuất

`PROTOCOL_CVTRIPLET_U1.md` (đã soạn, chưa chạy): arm mới U1+existence để lấp khoảng trống CVtriplet — hiện chỉ có baseline count/logistic (AUROC 0.9324, F1 nhị phân 0.8565, split mô phỏng) chứ không có kết quả model cho task existence. Chi phí ước tính: ~300 dòng code mới (existence channel + focal loss + triplet-bundle builder + eval) + 5–15 lượt train (~1–3 giờ CPU). Cần quyết định trước: split theo triplet (bundle mới), hệ số existence, unknown-sampling, và chấp nhận rằng kết quả là "U1+existence vs count baseline trên split mô phỏng" — không phải tái hiện paper (fold/unknown-sampling tác giả chưa công bố).

## 4. Giới hạn và trạng thái thật

- Toàn bộ 15 ô cap1 và 15 ô cap10 đều là inner-validation đã quan sát trong phát triển; outer-test đã lộ không được dùng và không gọi là blind.
- Câu hỏi lớn chưa giải quyết (không đổi so với handoff): danh sách entity/triplet/fold và công thức metric v3.2 của tác giả; F1 0.8600 không thể đối chiếu vì task/metric chưa tương đương được chứng minh.
- Không merge/push; artifact lớn (checkpoint ~34MB×30) giữ local, không commit theo quy ước handoff.

## 5. Hiện vật session này

- `bank1_linux/`: `PROTOCOL.md` (cùng hash bank1), `NAMESPACE.md`, `preflight.py/.json`, `supervisor.py` (Linux), `train.py`, `report.py`, `verify_final.py`, `status.json` (15 PASS), `results.json`, `REPORT.md`, `verification.json` (PASS), `contracts/` ×15, `runs/` ×15 (completion/launch/progress/manifest/predictions; checkpoint không commit).
- `PROTOCOL_CVTRIPLET_U1.md` — spec + cost cho bước tiếp theo.
- `REPORT_DEVIN_20261118.md` — file này.
