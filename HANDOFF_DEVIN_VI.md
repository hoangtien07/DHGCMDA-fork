# Bàn giao DHGCMDA v3.2 cho Devin — 2026-11-18

## Mục tiêu

Tái hiện có căn cứ bài *DHGCMDA: a dual-view heterogeneous graph contrastive learning framework for miRNA-disease association type prediction*, không chỉ tối ưu F1. Tách rõ mã tác giả phát hành v2.0, pipeline nội bộ diễn giải paper cho v3.2 và baseline/đánh giá mới. Cần xác định provenance dataset, feature recipe, split, targets, model, objective và metric trước khi tuyên bố tái hiện điểm paper. Người dùng muốn agent tự chạy đầy đủ, không hỏi những lựa chọn thực hiện thông thường, mỗi báo cáo có dòng ngắn **Tiến độ plan**.

## Điểm khởi đầu

- Nhánh bàn giao này được tách từ `e8a37ae` (`devin/1794078131-faithful-round5`) và chỉ chứa các hiện vật được chọn phục vụ tái hiện. Workspace ban đầu Windows `E:/1.AppCode/1.VSCode/DHGCMDA-fork` có rất nhiều thay đổi/trial chưa commit; không reset, xóa hay overwrite.
- Đọc đầu tiên `faithful_repro/independent_v32_overnight_20261118/PLAN_REVISED.md`, `REPORT_OVERNIGHT.md`, `FIDELITY_MATRIX.md`, `EXTERNAL_LINEAGE.md`; xem `bank1/PROTOCOL.md`, `bank1/status.json`, `bank1/REPORT.md` và `bank1/results.json`.
- File `status.json` và `REPORT_OVERNIGHT.md` trong commit là **snapshot tại thời điểm bàn giao**; không coi là trạng thái live. Hàng đợi chạy ở máy Windows ban đầu có supervisor `bank1/supervisor.py`, hai train processes, independent `bank1/watcher.py` và `finalizer.py`; nếu máy còn chạy, chúng tiếp tục ghi tại workspace gốc, **không tự đẩy kết quả mới lên Git**. Phải kiểm tra process/log và thời điểm file; không khởi động supervisor lần hai lên cùng output.
- Trạng thái snapshot lúc cập nhật: 6/15 lượt cap1 hoàn thành (fold0_seed1234, fold0_seed42, fold0_seed2025, fold1_seed1234, fold1_seed42, fold1_seed2025); đang chạy fold2_seed1234, fold2_seed42. Kết quả hai lượt đầu: best macro cap10→cap1 seed1234 0,4708→0,4384, seed42 0,4474→0,4705. Chưa có kết luận, 15 paired runs đã đăng ký. Code báo cáo tự ghi `bank1/REPORT.md`, `results.json`, `verification.json` khi hoàn tất. Kiểm tra dữ liệu thực tại thời điểm tiếp quản thay vì tin con số snapshot này.
- Hàng đợi ban đầu 9 lượt cosine/lambda/horizon **đã hủy theo yêu cầu người dùng** vì quá hẹp; output dở được giữ ở máy gốc, status `CANCELLED_BY_REPLAN`. Không khôi phục đơn giản chỉ để hoàn thành sweep cũ.

## Kết luận có bằng chứng

1. [Paper DHGCMDA](https://link.springer.com/article/10.1186/s12859-026-06436-w) Table 1–2: benchmark v2.0 có 4 loại; v3.2 có 5 loại, 411 miRNA × 271 bệnh, 11.748 typed links. Mã tác giả commit `06f9ba3eed92e334d9a2b8003fc06d83596e5614` chỉ có dữ liệu v2.0, metric nhận bốn scores hoặc existence+4; vector năm scores được hiểu nhầm và sáu scores bị bỏ. Không dùng code đó nguyên vẹn cho năm loại. Văn bản paper về CVtype lại nói bốn loại, chưa rõ cách áp dụng v3.2.
2. Tensor local SPLD = reference = strict theo từng phần tử labels và tên/thứ tự, 8.735 cặp dương, 2.110 đa loại, type counts [1155,403,2293,3997,3900]. Chưa chứng minh trùng tensor tác giả DHGCMDA. Nghiên cứu [SPLD 2022](https://doi.org/10.1093/bib/bbac390) nói HMDD v3.2 có 6 nhóm tổng quát, loại `other` trong benchmark 5 loại; MDAv3.2-3 có cùng kích thước và số liên kết. SPLD nói ngưỡng 3 association, DHGCMDA nói 2; hai miRNA local còn một typed link. Không tự prune hoặc khẳng định nguồn giống authors nếu không có bằng chứng.
3. CVtype và CVtriplet là hai task. U1 gần đây chỉ dự đoán type trên pair dương, không phải full CVtriplet. Mô phỏng random triplet split: khoảng 38,2% triplet dương test có cùng pair với type khác ở train; đó là đặc tính task, không tự động là lỗi. Baseline logistic/count riêng trên CVtriplet mô phỏng cho AUROC ~0,9324 và F1 binary ~0,8565 trên balanced positive/unknown; **không phải F1 Top-1 paper**. Chọn unknown dùng danh sách positives đầy đủ nên transductive; xem protocol và audit độc lập.
4. Kiểm tra builder 5 folds: khi chỉ đảo Y validation, train/view output không đổi với functional/semantic inputs cố định; static sequence/gene còn provenance chưa xác thực. Pair split không phải cold-start entity. Code release v2 có test labels trong feature; U1 rebuild train-only association blocks/functional. 40/411 miRNA thiếu mapping chuỗi nội bộ và 33 ánh xạ union locus, `coverage.json` có kết quả mô tả.
5. Sửa audit cũ: source release dùng reconstructed similarity khi cập nhật **cạnh HGT graph**; fixture AST `upstream_graph_fixture.py` xác minh. Không nhầm với việc ma trận G HGCN có/không rebuild. Paper, release và U1 khác reconstruction normalization/reduction, fusion và dynamic policy; xem fidelity matrix.
6. 15 bank gốc cap10: 508–535 / 695–704 validation positives mỗi ô được sample là unknown negatives, dù bank chỉ dùng train mask. Đây không phải nhãn validation được đưa vào sampler. Cap1 giảm overlap nhưng cũng thay số negative và contrastive objective; thí nghiệm 15 lượt đang chạy không xác định riêng tác động false negative. Không dùng outer-test đã lộ để tune.
7. 30 outer refit đã chạy dynamic/static 5×3: macro-F1 trung bình nội bộ 0,4045/0,4403; static > dynamic 14/15 ô. Product/disease prior tốt hơn micro/any-hit. Outer-test đã lộ trong phát triển: không gọi đây là blind test hay paper ablation. Cosine-reconstruction riêng 3 seed giảm spike nhưng F1 thấp hơn raw; không đẩy tiếp sweep cosine vô ích.

## Bằng chứng và kiểm tra

- `faithful_repro/independent_v32_overnight_20261118/`: các protocol và JSON: `data_audit`, `canary`, `negative_overlap`, `bank_size`, `strata`, `coverage`, `split_audit`, `triplet_audit`, `triplet_baseline`, `triplet_verify`, `upstream_graph_fixture`, `diagnostics`; `core_verification.json` PASS. Chạy `verify_core.py` sau khi khôi phục đủ inputs, không sửa các script/protocol đã pin nếu muốn giữ hash.
- `independent_v32_5fold3seed_20261117/REPORT.md` và `results.json`: 15 validation baseline; các export nhỏ/bundle selection liên quan và source runner đã đưa vào branch.
- `independent_v32_outer_refit_20261117/REPORT.md`, `INDEPENDENT_REVIEW.md`, `evaluation/results.json` và prediction nhỏ: 30 output đã chấm. Checkpoint lớn **không đưa lên Git**; các script yêu cầu ckpt cần copy riêng từ máy cũ hoặc chạy lại.
- `independent_v20_release_20261116/REPORT.md`, `pinned_release/`, `source_manifest.json`: chạy lại code authors v2.0, bao gồm cảnh báo leak; các outputs checkpoint/log lớn không được đưa lên.
- `independent_v32_author_review_20261117/REPORT.md`, upstream source snapshots, peer-review manifest: đối chứng paper/author code. PDF peer-review và checkpoint spike lớn không được đưa lên; link nguồn và SHA có trong manifest.
- Các dữ liệu/bundles 5 selection, source U1 và package feature cần thiết được đưa lên. Dữ liệu raw bên ngoài, một số archives và toàn bộ checkpoint >30 MB không được commit; đừng suy mọi script lịch sử sẽ chạy ngay trong clone sạch.

## Lưu ý thực thi trong môi trường mới

Bank1 `preflight.json` ghi **đường dẫn tuyệt đối Windows** và file runner/supervisor/watcher ghim executable `C:/Users/HoangTien07/...` cùng uv cache. Chạy trên Devin/Linux phải chỉnh wrapper/đường dẫn trong bản mới, lập **preflight/namespace mới**, ghi hash mới và kiểm tra data/source trước khi fit; không sửa JSON, protocol hay script đã pin rồi giả vờ kết quả cũ còn nguyên. Dùng Python 3.12, torch 2.5.1, torch-geometric 2.6.1, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2, pandas 2.2.3, CPU; 2 threads mỗi process. `uv` cache có thể rebuild trong môi trường mới. Hai baseline seed1234 fold0/1 có nguồn gốc khác OS, hạn chế reproducibility.

Không merge/push kết quả thiếu; không xóa artifacts; nếu kết quả live từ máy gốc đã hoàn tất hãy đưa log, completion, manifest, predictions, reports nhỏ vào commit mới, không commit checkpoint lớn nếu Git remote không hỗ trợ. Kiểm tra toàn vẹn, tính lại macro/micro bằng scikit-learn trên đúng pairs/truth, giữ tất cả lượt lỗi. Nếu không truy cập được máy Windows, báo rõ status chưa biết và chạy lại những lượt thiếu trong namespace mới, không tuyên bố đã hoàn tất.

## Ưu tiên tiếp

1. Thu và xác minh kết quả cap1 đầy đủ 5×3, báo tất cả cặp và cả best/epoch650, theo lớp và spike; không chỉ báo mean và không dùng 15 ô như quan sát độc lập. Nếu không cải thiện, kết luận âm tính và dừng sweep tương tự.
2. Hoàn tất matrix evidence: paper/commit author/fidelity về features, graph, split, existence/type và CVtriplet; tìm bản dữ liệu/fold/tác giả gốc từ nguồn sơ cấp một cách read-only, không gửi email nếu user chưa giao.
3. Đánh giá lại chuẩn protocol và baseline CVtriplet; chỉ triển khai DHGCMDA CVtriplet/existence khi định nghĩa outputs và train/test được nêu rõ; đánh giá chi phí trước khi chạy. Không nhầm binary F1 ~0,8565 với paper CVtype 0,8600.
4. Phân tích feature coverage/association leakage và negative strategy theo train-only. Bất kỳ can thiệp mới phải có giả thuyết cụ thể, protocol ghi trước, matched control và không biến outer-test lộ thành blind validation.
5. Cuối cùng lập báo cáo tiếng Việt, có links file và **Tiến độ plan**; phân biệt đã làm/đang chạy/chưa biết, nêu bằng chứng và giới hạn.
