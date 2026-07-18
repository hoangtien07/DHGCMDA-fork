# Đề xuất cải tiến tính toán — DHGCMDA (fork)

**Căn cứ:** chỉ dựa trên [docs/review/01-baseline-audit.md](01-baseline-audit.md) và mã nguồn
trong repo. Mỗi đề xuất được gắn với các phát hiện audit cụ thể (F1–F8) và mã/flag đã có sẵn.

**Tuyên bố phạm vi:** mọi "lợi ích kỳ vọng" bên dưới là **giả thuyết tính toán** về các chỉ số
của model (AUC, AUPR, Top-1 P/R/F1) đo dưới một quy trình **không rò rỉ (leakage-controlled)** và
**thống kê trung thực**. **KHÔNG có tuyên bố nào về kết quả sinh học hay lâm sàng** — những tuyên
bố đó cần bằng chứng thực nghiệm ướt/lâm sàng nằm ngoài phạm vi dự án.

**Nguyên tắc phương pháp luận (đọc trước):** các phát hiện F1 (rò rỉ transductive), F3 (bỏ ~10%
dữ liệu), F4 (tune ngưỡng trên test), F5 (gộp protocol) và F7 (AUPR cân bằng) có nghĩa là *chỉ số
hiện tại là một ước lượng bị thổi phồng và phương sai cao*. **Không đề xuất nào bên dưới có thể
được chấp nhận một cách hợp lệ cho tới khi bản thân quy trình đánh giá được sửa (P7 + P9).** Vì vậy
thứ tự khuyến nghị là **P7 → P9 → P10 trước (thiết lập baseline trung thực, tái lập được), sau đó
P1–P6, P8, và P5-RL cuối cùng.** Mọi "lợi ích kỳ vọng" đều so với baseline đã sửa đó, KHÔNG so với
con số rò rỉ hiện tại.

---

## P1 — Kiến trúc GNN: fusion học được (thay tổng cố định 0.6/0.4)

- **Giả thuyết.** "Attention-guided fusion" thực chất là tổng cứng
  `0.6·view1 + 0.4·view2` ([hetero_model.py:282-312](../../hetero_model.py#L282)); một cơ chế
  gating/attention học được trên hai view hypergraph sẽ điều chỉnh trọng số view theo từng node và
  từng dataset, cải thiện Top-1 F1 mà không hại AUC.
- **Lợi ích kỳ vọng.** Tăng nhẹ–vừa Top-1 F1; biến một hằng số không có cơ sở thành tham số học từ
  dữ liệu; xoá bỏ một điểm lệch repo–paper đã nêu trong audit (§6, "view fusion").
- **File ảnh hưởng.** [hetero_model.py](../../hetero_model.py) (`HGCN_Attention_Mechanism` và các
  chỗ gọi `AM_mi`/`AM_dis` trong `forward` ~L844); không đổi dữ liệu.
- **Chi phí tính toán.** Thấp: thêm 2 vector trọng số nhỏ mỗi fusion; thời gian/epoch tăng không
  đáng kể; 1 lần chạy 5-fold × multi-seed để so sánh (~1× baseline mỗi seed).
- **Đối chứng thí nghiệm.** Cùng folds/seeds như baseline; ablate {cố định 0.6/0.4, gate vô hướng
  học được, softmax attention theo node}; giữ nguyên mọi hyperparameter khác; so sánh dưới P7/P9.
- **Chế độ thất bại.** Tham số dư overfit trên v2.0 nhỏ (1.498 positive) → generalize kém hơn;
  attention sụp về ~một view (thoái hoá lại thành hằng số).
- **Bằng chứng cần trước khi chấp nhận.** Cải thiện Top-1 F1 (theo cặp) dưới eval không rò rỉ trên
  ≥3 seed, CI loại trừ 0, không giảm AUC; trọng số học được khác 0.6/0.4 rõ rệt và ổn định qua các
  seed.

---

## P2 — Học biểu diễn: dùng view miRNA thực sự độc lập (F6)

- **Giả thuyết.** miRNA "View 1" là GIP (`M_GSM.txt`, dẫn xuất từ association —
  [prepareData.py:389-408](../../prepareData.py#L389)), nên mục tiêu contrastive dual-view phần nào
  đang tương phản hai bản sao association thay vì hai modality. Thay bằng similarity dẫn xuất từ
  chuỗi thật (k-mer) sẽ làm hai view thực sự bổ trợ và giảm dư thừa biểu diễn.
- **Lợi ích kỳ vọng.** Embedding giàu thông tin hơn; *đồng thời* giảm rò rỉ (một view độc lập với
  association không thể tái tiêm cạnh held-out). Có thể đánh đổi giảm nhẹ chỉ số rò rỉ lấy chỉ số
  *trung thực* cao hơn.
- **File ảnh hưởng.** [build_mirna_seq_features.py](../../build_mirna_seq_features.py) (đã có) →
  `M_SEQ.txt`; nạp qua flag sẵn có `--mirna_seq_sim_path`
  ([prepareData.py:394-399](../../prepareData.py#L394)). Không đổi model.
- **Chi phí tính toán.** Thấp: build đặc trưng k-mer một lần; chi phí train không đổi.
- **Đối chứng thí nghiệm.** A/B trên cùng folds: {GIP `M_GSM`} vs {sequence thật `M_SEQ`} vs {ghép
  cả hai}; chạy dưới cả `--leakage_free` bật và tắt để tách hiệu ứng biểu diễn khỏi hiệu ứng rò rỉ.
- **Chế độ thất bại.** k-mer similarity dự báo yếu → chỉ số giảm; thiếu chuỗi cho một số miRNA →
  buộc fallback tái tiêm nhiễu.
- **Bằng chứng cần.** Dưới eval không rò rỉ, sequence-view ≥ GIP-view về AUPR/Top-1 F1 qua các seed;
  kiểm tra dư thừa biểu diễn (vd. tương quan đặc trưng giữa hai view thấp hơn).

---

## P3 — Hàm mất mát: giữ tín hiệu multi-label (multi-label BCE) (F2)

- **Giả thuyết.** "Last-wins" bỏ 20,4% nhãn type (audit §2). Một đầu multi-label BCE theo từng type
  giữ tất cả type của một cặp sẽ cải thiện phân biệt type trên các cặp đa-type so với đầu CE
  single-label hiện tại.
- **Lợi ích kỳ vọng.** Top-1 recall cao/trung thực hơn trên cặp thiểu số và đa-type; một chỉ số
  phản ánh đúng chủ ý "type dự đoán ∈ tập type đã biết" của paper.
- **File ảnh hưởng.** `--loss_mode multilabel_bce` đã có
  ([param.py:205-213](../../param.py#L205), loss tại
  [main_experiments_hetero1.py:222-271](../../main_experiments_hetero1.py#L222)); cần target
  multi-hot dựng từ CSV *gốc* (không phải matrix đã đè) — bước dựng tại
  [prepareData.py:49-97](../../prepareData.py#L49) / `--multilabel_target_path`.
- **Chi phí tính toán.** Thấp; cùng forward, chỉ khác reduction của loss.
- **Đối chứng thí nghiệm.** So sánh {two_head single-label CE, softmax_5class, multilabel_bce} trên
  cùng folds; **chỉ số Top-1 cũng phải sửa** để chấm "predicted ∈ known types"
  ([Calculate_Metrics.py:346](../../Calculate_Metrics.py#L346)) nếu không lợi ích multi-label không
  quan sát được.
- **Chế độ thất bại.** Target multi-hot nhiễu (positive theo kênh mất cân bằng) → BCE bất ổn; ghi
  chú dự án cho thấy multilabel_bce KHÔNG mở khoá v3.2 — phải test lại riêng cho v2.0.
- **Bằng chứng cần.** Tăng recall multi-label trên tập con 161 cặp đa-type, không giảm trên cặp
  single-label, qua các seed và dưới chỉ số Top-1 đã sửa.

---

## P4 — Negative sampling: negative giàu thông tin + prevalence eval trung thực (F7)

- **Giả thuyết.** Train dùng 10× negative ngẫu nhiên đều
  ([main_experiments_hetero1.py:912-918](../../main_experiments_hetero1.py#L912)); lấy mẫu
  hard-negative / theo similarity (negative gần positive trong không gian embedding/similarity) sẽ
  làm sắc nét biên quyết định và cải thiện AUPR.
- **Lợi ích kỳ vọng.** Chỉ số xếp hạng tốt hơn ở cùng mức tính toán; và bằng cách báo cáo AUPR
  *khớp prevalence* bên cạnh bản cân bằng, có cách đọc xếp hạng đáng tin về thực tế.
- **File ảnh hưởng.** [main_experiments_hetero1.py](../../main_experiments_hetero1.py) khối
  negative-sampling (~L912) và khối dựng negative lúc eval (~L1481); tuỳ chọn một sampling policy
  trong orchestrator mới (không cần sửa source nếu làm qua config).
- **Chi phí tính toán.** Thấp–vừa: hard-negative mining thêm một lần tra nearest-neighbour mỗi
  epoch (hoặc định kỳ); eval khớp prevalence thêm ~1 lần scoring.
- **Đối chứng thí nghiệm.** {đều 10×, hard theo similarity, quét tỉ lệ 1/5/10/20} × cùng positive;
  báo cáo cả AUPR cân bằng (1:1) và prevalence (~1:125); giữ cố định tập negative eval qua các
  phương pháp.
- **Chế độ thất bại.** Hard negative là false negative (vấn đề PU paper đã nêu,
  [p36](../../_pdf_text/p36.txt)) → model bị phạt vì dự đoán đúng; bất ổn nếu tỉ lệ quá gắt.
- **Bằng chứng cần.** Tăng AUPR dưới cùng tập negative eval qua các seed; chứng minh hard negative
  không phải phần lớn là positive được xác nhận về sau (ghi chú độ nhạy).

---

## P5 — Mất cân bằng lớp: logit adjustment / LDAM vs Effective-Number reweighting

- **Giả thuyết.** Số lượng type lệch (Genetics 681 vs Epigenetics 157, audit §2). Logit adjustment
  (Menon 2020) hoặc LDAM (Cao 2019) sẽ phục hồi recall lớp thiểu số sạch hơn class weight
  Effective-Number hiện tại, vốn có thể đè lớp đa số quá mức.
- **Lợi ích kỳ vọng.** Macro Top-1 recall / cân bằng recall theo lớp cao hơn ở cùng micro precision.
- **File ảnh hưởng.** Đã hiện thực: `--type_loss {ce, logit_adjust, ldam}` với `--la_tau`,
  `--ldam_max_margin` ([param.py:224-240](../../param.py#L224); logic
  [main_experiments_hetero1.py:381-400](../../main_experiments_hetero1.py#L381)); phân tích theo
  lớp qua [analyze_perclass_recall.py](../../analyze_perclass_recall.py).
- **Chi phí tính toán.** Không đáng kể (dịch logit lúc train); một lần chạy mỗi cấu hình.
- **Đối chứng thí nghiệm.** {ce+EffNum weights, logit_adjust (quét τ), ldam (quét margin), uniform}
  trên cùng folds; chỉ số chính = recall **theo lớp** + macro-F1, không chỉ micro.
- **Chế độ thất bại.** Sửa kép (logit-adjust *và* class weight) làm sụp lớp đa số — code đã chặn
  ([main_experiments_hetero1.py:372](../../main_experiments_hetero1.py#L372)), nhưng τ quá lớn đảo
  ngược mất cân bằng; lớp thiểu số quá nhỏ cho recall phương sai cao.
- **Bằng chứng cần.** Tăng recall lớp thiểu số với mất mát lớp đa số ≤ nhỏ, cải thiện macro-F1 có ý
  nghĩa qua các seed, dưới eval đã sửa.

---

## P6 — Biểu diễn/kiến trúc: full-bilinear predictor làm mặc định

- **Giả thuyết.** Mặc định `predictor_mode='diag'` là BilinearDiag rank-*d*; scorer full-bilinear
  (`mᵀ·W_t·d`) có sức chứa cao hơn và, theo flag repo, được dự định làm mặc định v2.0. Nó nên cải
  thiện Top-1 F1 dưới eval trung thực.
- **Lợi ích kỳ vọng.** Sức chứa phân biệt type cao hơn; xoá một rewrite mà audit và comment code
  đánh dấu là thoái hoá.
- **File ảnh hưởng.** `--predictor_mode full_bilinear`
  ([param.py:198-202](../../param.py#L198); [hetero_model.py:493-596](../../hetero_model.py#L493)).
- **Chi phí tính toán.** Vừa: thêm `num_types × d × d` tham số (d=128) → nhiều bộ nhớ hơn và tăng
  nhẹ thời gian/epoch; vẫn khả thi trên CPU.
- **Đối chứng thí nghiệm.** {diag, full_bilinear} trên cùng folds/seeds; theo dõi overfit trên
  1.498 positive (giữ regularization cố định).
- **Chế độ thất bại.** Over-parameterize trên v2.0 nhỏ → phương sai tăng, chỉ số trung thực giảm dù
  chỉ số rò rỉ tăng; tương tác với P1/P3 không cộng tính.
- **Bằng chứng cần.** Tăng Top-1 F1 theo cặp dưới CV multi-seed không rò rỉ với CI loại trừ 0 và
  AUC ổn định.

---

## P7 — Chống rò rỉ dữ liệu: pipeline leakage-free hoàn chỉnh (F1, F6) **[TIÊN QUYẾT]**

- **Giả thuyết.** Pipeline mặc định là transductive: ma trận association đầy đủ nạp vào hypergraph
  KNN, cạnh hetero-graph, target inter-view CL, và GIP-derived `m_ss/IM/ID` (audit §4.1).
  `--leakage_free` chỉ mask ma trận ghép, **không** mask GIP/integrated similarity. Bản vá hoàn
  chỉnh (tính lại *mọi* artifact dẫn xuất từ association chỉ từ cạnh train, theo từng fold) sẽ hạ
  chỉ số báo cáo nhưng cho ước lượng generalize thật.
- **Lợi ích kỳ vọng.** Một baseline đáng tin; đây là sửa chữa quan trọng nhất — lợi ích của mọi đề
  xuất khác chỉ có ý nghĩa khi đo trên nền này.
- **File ảnh hưởng.** [prepareData.py](../../prepareData.py) (chuyển tính GIP `Gauss_M/Gauss_D` và
  `ID/IM` vào *trong* fold trên cạnh train — hiện global tại
  [prepareData.py:208-209](../../prepareData.py#L208)); [main_experiments_hetero1.py:932-941](../../main_experiments_hetero1.py#L932)
  (mở rộng `--leakage_free` để mask `m_ss` và target inter-view CL); [trainData.py:106](../../trainData.py#L106)
  (ngừng chia sẻ một `md_p` đầy đủ qua các fold).
- **Chi phí tính toán.** Vừa: similarity/hypergraph tính lại theo fold thay vì một lần → ~5× chi phí
  tiền xử lý (không phải train) mỗi lần chạy; cache theo fold giảm bớt.
- **Đối chứng thí nghiệm.** Báo cáo **leakage gap** = metric(rò rỉ) − metric(mask hoàn toàn) trên
  cùng folds/seeds; xác minh run đã mask không bao giờ chạm cell test (assert test positive = 0
  trong mọi tensor dẫn xuất từ association).
- **Chế độ thất bại.** Bỏ sót đường rò rỉ còn lại (vd. file similarity cache); similarity theo fold
  nhiễu hơn trên train thưa → phương sai cao; mask đổi kết nối graph đủ để train bất ổn.
- **Bằng chứng cần.** Một leakage gap định lượng kèm CI; assert tự động rằng không tensor nào dẫn
  xuất từ association chứa positive của fold test; kết quả được báo cáo làm baseline mới.

---

## P8 — Ablation study: trung thực, multi-seed, dựa rebuild, không rò rỉ

- **Giả thuyết.** Các quan sát "ablation reversal" trước đây (bỏ component lại *cải thiện*) được đo
  dưới điều kiện rò rỉ, đơn cấu hình. Chạy lại các ablation *rebuild* dưới P7/P9 sẽ cho thấy từng
  component (dual-view CL, HGCN, HGT) có thực sự giúp trên v2.0 hay không.
- **Lợi ích kỳ vọng.** Bảng đóng góp component đáng tin; nhận diện component chết/gây hại để cắt
  (giảm chi phí) hoặc xác nhận chúng.
- **File ảnh hưởng.** `--ablation {no_cl_rebuild, no_hgcn_rebuild, no_hgt_rebuild, no_dv, no_avf}`
  ([param.py:283-289](../../param.py#L283); các nhánh trong
  [hetero_model.py:689-916](../../hetero_model.py#L689)); cộng ablation các component của P1/P2/P3.
- **Chi phí tính toán.** Cao: (#component × #seed × 5 fold); mục chi phí lớn nhất — cần dự trù rõ.
- **Đối chứng thí nghiệm.** Cùng folds/seeds qua mọi ablation; bỏ đơn component (không kết hợp);
  so sánh theo cặp per-fold với full model; hiệu chỉnh đa so sánh qua các component.
- **Chế độ thất bại.** Thiếu công suất thống kê (ít seed) → nhiễu bị nhầm là hiệu ứng; ablation
  rebuild không tương đương hành vi với retrain thật của model nhỏ hơn (caveat kiểu audit).
- **Bằng chứng cần.** Δ mỗi component với kiểm định theo cặp vượt hiệu chỉnh; dấu và độ lớn ổn định
  qua các seed và qua điều kiện rò rỉ/trung thực.

---

## P9 — Đánh giá thống kê: CV toàn dữ liệu, chỉ số không ngưỡng, kiểm định (F3,F4,F5,F7) **[TIÊN QUYẾT]**

- **Giả thuyết.** Ước lượng hiện tại lệch/phương sai cao: bỏ ~10% dữ liệu (F3), ngưỡng tune trên
  test (F4), gộp hai protocol (F5), chỉ AUPR cân bằng (F7). Một quy trình đã sửa cho số ít lệch hơn,
  tương đương, kèm bất định trung thực.
- **Lợi ích kỳ vọng.** Ước lượng điểm đáng tin *và* khoảng tin cậy trung thực; phân biệt được cải
  thiện thật (P1–P6, P8) với nhiễu.
- **File ảnh hưởng.** [prepareData.py:285-304](../../prepareData.py#L285) (dùng cả 10 chunk / 5-fold
  80-20 chuẩn); [Calculate_Metrics.py:104-136](../../Calculate_Metrics.py#L104) (báo cáo AUC/AUPR
  làm chính; nếu báo cáo F1 thì cố định ngưỡng trên train, không phải test); tách chạy CVtriplet vs
  CVtype để khớp paper (audit §5); tổng hợp eval tại
  [main_experiments_hetero1.py:1465-1610](../../main_experiments_hetero1.py#L1465).
- **Chi phí tính toán.** Thấp–vừa: chủ yếu nhiều seed hơn (phương sai) và protocol CV thứ hai;
  bootstrap CI rẻ, tính post-hoc.
- **Đối chứng thí nghiệm.** Lưới seed cố định (≥5); kiểm định theo cặp per-fold/per-seed (paired t
  hoặc Wilcoxon) với Bonferroni/Holm qua các đề xuất; bootstrap CI 95%; báo cáo cả AUPR cân bằng và
  prevalence.
- **Chế độ thất bại.** Quá ít seed → CI rộng che hiệu ứng thật; hiệu chỉnh đa so sánh quá gắt →
  âm tính giả; ngưỡng-trên-train kém hơn ngưỡng-tune-test và trông như thoái lui (nó là con số
  trung thực).
- **Bằng chứng cần.** Chỉ số báo cáo kèm CI và p-value kiểm định theo cặp; một protocol chỉ số cố
  định, được ghi lại, dùng đồng nhất cho mọi đề xuất tiếp theo.

---

## P10 — Khả năng tái lập: config cố định, chạy tất định, truy vết kết quả

- **Giả thuyết.** Kết quả phụ thuộc seed, flag (leakage, loss_mode, predictor_mode, K), và môi
  trường; không có manifest thì các so sánh trôi dạt. Một manifest config+provenance mỗi lần chạy
  và audit tính tất định làm mọi con số tái chạy chính xác.
- **Lợi ích kỳ vọng.** Thí nghiệm so sánh được, kiểm toán được; tránh mập mờ "cấu hình nào tạo con
  số này" mà audit gặp liên tục.
- **File ảnh hưởng.** [param.py](../../param.py) (dump args đã resolve ra JSON mỗi run);
  [main_experiments_hetero1.py:56-67,1756](../../main_experiments_hetero1.py#L56) (xác minh
  `seed_torch` phủ mọi RNG; cân nhắc `torch.use_deterministic_algorithms`);
  [requirements_linux.txt](../../requirements_linux.txt) (pin); một manifest kết quả kèm mỗi
  `results/*.json`.
- **Chi phí tính toán.** Không đáng kể.
- **Đối chứng thí nghiệm.** Chạy lại cùng config hai lần → chỉ số y hệt (kiểm tra tất định);
  manifest ghi git SHA, args, hash dataset, flag leakage, protocol chỉ số.
- **Chế độ thất bại.** Bất định BLAS/threading trên CPU
  ([main_experiments_hetero1.py:5-9](../../main_experiments_hetero1.py#L5)) cản tái lập chính xác —
  có thể cần cố định số thread; thuật toán tất định làm chậm một số op.
- **Bằng chứng cần.** Hai lần chạy lại y hệt (hoặc envelope phương sai được ghi lại); một manifest
  gắn với mọi kết quả báo cáo.

---

## Tóm tắt ưu tiên & phụ thuộc

| Pha | Đề xuất | Lý do |
|---|---|---|
| **0 (bắt buộc trước)** | P7, P9, P10 | Baseline trung thực, phương sai thấp, tái lập được — điều kiện tiên quyết để đo bất cứ thứ gì khác |
| **1 (rẻ, ROI cao)** | P2, P3, P5 | Mức config; xử lý F2/F6 và mất cân bằng; flag đã có sẵn |
| **2 (sức chứa model)** | P1, P6 | Sức chứa kiến trúc; rủi ro overfit v2.0 nhỏ — cần Pha 0 để phán xét |
| **3 (đắt/hoãn)** | P4, P8, P5-search | Chi phí cao hơn; ablation và sampling policy |
| **4 (cuối, có rào chắn)** | P5-RL bên dưới | Search tự động; rủi ro rò rỉ/selection lớn nhất |

### Ghi chú về tối ưu hyperparameter/policy bằng RL (hạng mục được yêu cầu)

- **Giả thuyết.** Một controller bandit/population-based hoặc RL trên
  {`K_neigs`, `exist_weight`, `inter_view_weight`, `la_tau`, lr, tỉ lệ negative} có thể tìm cấu
  hình tốt hơn quét thủ công, tối ưu Top-1 F1 không rò rỉ.
- **Lợi ích kỳ vọng.** Chọn hyperparameter tự động, tái lập được; cũng có thể học một *policy lấy
  mẫu negative* (chọn negative nào để đưa vào) làm không gian hành động RL (gắn với P4).
- **File ảnh hưởng.** Một orchestrator mới (không cần sửa source — nó điều khiển flag CLI trong
  [param.py](../../param.py)); tuỳ chọn một hook policy tại chỗ lấy mẫu negative
  ([main_experiments_hetero1.py:912](../../main_experiments_hetero1.py#L912)).
- **Chi phí tính toán.** **Cao**: mỗi bước RL/search là một lần chạy đầy đủ 5-fold × multi-seed;
  tổng chi phí = (số vòng search × baseline). Trên CPU đây là đề xuất đắt nhất theo khoảng cách lớn.
- **Đối chứng thí nghiệm.** **Nested CV** — search phải tối ưu trên inner-validation và báo cáo
  trên outer-test chưa đụng tới, nếu không nó thành selection leakage (một lỗi cùng lớp F4/F5 mới);
  seed search cố định; so với random search và grid thủ công làm baseline.
- **Chế độ thất bại.** Tối ưu trực tiếp trên chỉ số CV báo cáo = overfit chính việc đánh giá (rò rỉ
  ngầm); phương sai reward từ fold nhiễu làm controller bất ổn; chi phí lớn hơn lợi ích so với grid
  đơn giản khi chỉ có ~6 hyperparameter.
- **Bằng chứng cần.** RL/search thắng *cả* random search *và* grid thủ công trên fold **outer**
  giữ riêng qua các seed, với sự tách inner/outer của search được chứng minh; lợi ích sau khi trừ
  chi phí là dương. Thiếu bằng chứng nested-CV, coi mọi "cải thiện" từ RL là chưa được chứng minh.

---

*Tài liệu đề xuất chỉ-đọc. Không sửa mã nguồn; không chạy thí nghiệm. Mọi lợi ích kỳ vọng là giả
thuyết chỉ-số-tính-toán cần bằng chứng đã nêu trước khi chấp nhận; không hàm ý cải thiện sinh học
hay lâm sàng.*
