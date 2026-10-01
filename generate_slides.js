// Slide báo cáo DHGCMDA — lý thuyết paper + kết quả tái hiện & cải thiện
// Usage: npm install pptxgenjs && node gen_slides.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const REPO = "/home/ubuntu/repos/DHGCMDA-fork";
const pptx = new pptxgen();
pptx.defineLayout({ name: "W", width: 13.33, height: 7.5 });
pptx.layout = "W";

const C = {
  navy: "1F3864", blue: "2E5FA3", lblue: "DEEAF6", gold: "BF9000",
  green: "2E7D32", red: "B71C1C", gray: "595959", white: "FFFFFF",
  dark: "262626", accent: "4A7EBB", lg: "F2F5F9",
};
const FONT = "Calibri";

function header(slide, title, sub) {
  slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 13.33, h: 1.0, fill: { color: C.navy } });
  slide.addText(title, { x: 0.4, y: 0.08, w: 12.5, h: 0.55, fontFace: FONT, fontSize: 26, bold: true, color: C.white });
  if (sub) slide.addText(sub, { x: 0.4, y: 0.6, w: 12.5, h: 0.35, fontFace: FONT, fontSize: 13, italic: true, color: C.lblue });
}
function footer(slide, n) {
  slide.addText(`DHGCMDA — Reproduce & Improve  |  ${n}`, { x: 0.4, y: 7.15, w: 12.5, h: 0.3, fontFace: FONT, fontSize: 9, color: C.gray, align: "right" });
}
function bullets(slide, items, x, y, w, h, fs = 16) {
  slide.addText(items.map(t => ({ text: t.text, options: { bullet: { code: "2022", indent: 12 }, breakLine: true, ...(t.opts||{}) } })),
    { x, y, w, h, fontFace: FONT, fontSize: fs, color: C.dark, paraSpaceAfter: 8 });
}
function box(slide, txt, x, y, w, h, fill, tc, fs = 14, bold = false) {
  slide.addText(txt, { x, y, w, h, fill: { color: fill }, color: tc, fontFace: FONT, fontSize: fs, bold, align: "center", valign: "middle", margin: 4 });
}
let N = 0;
const slide = () => { const s = pptx.addSlide(); N++; return s; };

// ---------- 1. TITLE ----------
{
  const s = slide();
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 13.33, h: 7.5, fill: { color: C.navy } });
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 4.9, w: 13.33, h: 0.06, fill: { color: C.gold } });
  s.addText("DHGCMDA", { x: 0.8, y: 1.6, w: 11.7, h: 0.9, fontFace: FONT, fontSize: 54, bold: true, color: C.white, align: "center" });
  s.addText("Dual-View Heterogeneous Graph Contrastive Learning\nfor miRNA–Disease Association Type Prediction", { x: 0.8, y: 2.6, w: 11.7, h: 1.1, fontFace: FONT, fontSize: 22, color: C.lblue, align: "center" });
  s.addText("Tái hiện & cải thiện kết quả paper (BMC Bioinformatics 2026)", { x: 0.8, y: 4.0, w: 11.7, h: 0.5, fontFace: FONT, fontSize: 17, color: C.gold, align: "center", bold: true });
  s.addText("Top-1 F1: 0.5970  →  0.7041   |   AUC: 0.9669  →  0.9899", { x: 0.8, y: 5.3, w: 11.7, h: 0.5, fontFace: FONT, fontSize: 20, bold: true, color: C.white, align: "center" });
  s.addText("Báo cáo kết quả · 2026", { x: 0.8, y: 6.5, w: 11.7, h: 0.4, fontFace: FONT, fontSize: 12, color: C.lblue, align: "center" });
}

// ---------- 2. BÀI TOÁN ----------
{
  const s = slide();
  header(s, "1. Bài toán & dữ liệu", "Dự đoán loại quan hệ miRNA–bệnh — không chỉ có/không");
  bullets(s, [
    { text: "Bài toán: cho cặp (miRNA, disease), dự đoán TỒN TẠI quan hệ + LOẠI quan hệ (4 lớp: Circulation, Epigenetics, Target, Genetics)." },
    { text: "Khác biệt với link prediction thông thường: cùng một cặp có thể mang nhiều loại quan hệ — nhãn là multilabel." },
    { text: "Dataset chính: HMDD v2.0 — 495 miRNA × 383 disease, 2 926 quan hệ đã gán loại (×2 quan hệ binary đã xác minh)." },
    { text: "Cực kỳ thưa: density ~1.5% → vấn đề cốt lõi là mất cân bằng positive/negative và tín hiệu yếu." },
    { text: "Đánh giá: 5-fold CV trên positive, negatives resample ngẫu nhiên → metric Top-1 F1 (loại đúng ở rank 1) + AUC/AUPR (tồn tại)." },
  ], 0.5, 1.25, 8.2, 5.6, 17);
  // stats box
  box(s, "495", 9.0, 1.5, 1.9, 1.0, C.lblue, C.navy, 30, true);
  box(s, "miRNA", 9.0, 2.5, 1.9, 0.45, C.lblue, C.gray, 13);
  box(s, "383", 11.0, 1.5, 1.9, 1.0, C.lblue, C.navy, 30, true);
  box(s, "disease", 11.0, 2.5, 1.9, 0.45, C.lblue, C.gray, 13);
  box(s, "4 loại", 9.0, 3.15, 1.9, 1.0, "FFF2CC", C.gold, 26, true);
  box(s, "association types", 9.0, 4.15, 1.9, 0.45, "FFF2CC", C.gray, 13);
  box(s, "~1.5%", 11.0, 3.15, 1.9, 1.0, "FFF2CC", C.gold, 26, true);
  box(s, "density (sparse)", 11.0, 4.15, 1.9, 0.45, "FFF2CC", C.gray, 13);
  box(s, "Metric chính: Top-1 F1\n(loại đúng phải đứng rank 1)", 9.0, 5.0, 3.9, 1.4, C.navy, C.white, 15, true);
  footer(s, N);
}

// ---------- 3. KIẾN TRÚC ----------
{
  const s = slide();
  header(s, "2. Kiến trúc tổng quan", "Dual-view hypergraph + contrastive learning + heterogeneous fusion");
  s.addImage({ path: path.join(REPO, "results/architecture_overview.png"), x: 1.0, y: 1.3, w: 11.3, h: 6.01 });
  footer(s, N);
}

// ---------- 4. DUAL-VIEW + HGCN ----------
{
  const s = slide();
  header(s, "3. Dual-view data & Hypergraph Convolution", "Mỗi modality có 2 'góc nhìn' similarity khác nhau");
  bullets(s, [
    { text: "miRNA — view 1: sequence similarity (M_GSM); view 2: functional similarity (M_FSM)." },
    { text: "Disease — view 1: gene-based similarity (D_SSM2); view 2: semantic similarity MeSH (D_SSM1)." },
    { text: "Mỗi view → một hypergraph riêng: node = miRNA/disease, hyperedge = nhóm các node tương tự (top-K neighbors)." },
    { text: "HGCN (hypergraph convolution) lan truyền feature theo hyperedge: node → hyperedge → node — bắt quan hệ 'nhóm' mà pairwise graph thường không thấy." },
    { text: "Output mỗi modality: 2 embedding z_v1, z_v2 — cùng entity, hai biểu diễn bổ sung." },
  ], 0.5, 1.3, 7.6, 5.4, 16);
  // diagram: 2 views → 2 HGCN
  box(s, "View 1\n(seq / gene)", 8.6, 1.6, 2.0, 0.9, C.lblue, C.navy, 14, true);
  box(s, "View 2\n(func / sem)", 10.9, 1.6, 2.0, 0.9, C.lblue, C.navy, 14, true);
  s.addText("→", { x: 8.6, y: 2.6, w: 4.3, h: 0.4, align: "center", fontSize: 20, color: C.gray });
  box(s, "HGCN view 1", 8.6, 3.0, 2.0, 0.9, "D6E4F0", C.navy, 14, true);
  box(s, "HGCN view 2", 10.9, 3.0, 2.0, 0.9, "D6E4F0", C.navy, 14, true);
  s.addText("→", { x: 8.6, y: 4.0, w: 4.3, h: 0.4, align: "center", fontSize: 20, color: C.gray });
  box(s, "z₁          z₂", 8.6, 4.4, 4.3, 0.8, C.navy, C.white, 16, true);
  s.addText("Hypergraph = cạnh nối ≥2 node\n(bắt quan hệ nhóm, không chỉ cặp)", { x: 8.6, y: 5.5, w: 4.3, h: 1.0, fontSize: 12, italic: true, color: C.gray, align: "center" });
  footer(s, N);
}

// ---------- 5. AVF + HGT ----------
{
  const s = slide();
  header(s, "4. Attention View Fusion + Heterogeneous Graph Transformer", "Hợp nhất 2 view, rồi lan truyền trên đồ thị miRNA–disease");
  bullets(s, [
    { text: "AVF — Attention View Fusion: với mỗi view v, học trọng số α_v (attention) + β_v (residual) rồi kết hợp z = α·σ(z̃) + β·Σz — tự quyết view nào đáng tin hơn theo node." },
    { text: "HGT (2 lớp, 4 heads): transformer attention trên đồ thị dị thể gồm 2 loại node (miRNA, disease) và cạnh association — tinh chỉnh embedding theo ngữ cảnh cả hai phía." },
    { text: "Bilinear predictor: P(exist, type=c) = miᵀ · W_c · dis — W_c riêng cho từng loại quan hệ." },
    { text: "Paper dùng FULL bilinear (ma trận d×d đầy đủ); code release ban đầu dùng bản diagonal suy biến (rank-d) — chi tiết ở phần kết quả." },
  ], 0.5, 1.3, 7.6, 5.4, 16);
  box(s, "z_m₁, z_m₂", 8.6, 1.7, 1.9, 0.8, C.lblue, C.navy, 13, true);
  box(s, "AVF", 10.9, 1.7, 1.9, 0.8, C.blue, C.white, 15, true);
  box(s, "Z_M", 9.7, 2.9, 1.4, 0.7, C.navy, C.white, 15, true);
  box(s, "HGT ×2 lớp\n(4 heads)", 9.0, 4.0, 2.9, 1.0, "FFE699", C.gold, 15, true);
  box(s, "P(exist) + P(type 1..4)", 8.4, 5.4, 4.2, 0.9, C.green, C.white, 15, true);
  footer(s, N);
}

// ---------- 6. CL + LOSS ----------
{
  const s = slide();
  header(s, "5. Contrastive Learning & hàm mất mát thống nhất", "4 nguồn supervision trong một loss");
  bullets(s, [
    { text: "Intra-view CL (SimCLR): kéo z_v1 ↔ z_v2 của CÙNG một entity lại gần — ép 2 view nhất quán." },
    { text: "Inter-view CL (InfoNCE + margin): kéo embedding miRNA ↔ disease có quan hệ lại gần qua ranh giới modality." },
    { text: "Reconstruction: decoder dựng lại feature/similarity đầu vào (λ = 1.0) — regularizer giữ embedding giữ thông tin." },
    { text: "Prediction loss: L = 0.3·BCE(existence) + 0.7·CE(type) + λ₁·L_CL + λ₂·L_recon — cùng tối ưu một lượt." },
    { text: "Dynamic hypergraph rebuild: hyperedge tái tạo theo embedding mỗi 5 epoch (top-K trên z hiện tại)." },
  ], 0.5, 1.3, 7.8, 5.4, 16);
  box(s, "L_total", 8.9, 1.8, 3.4, 0.8, C.navy, C.white, 20, true);
  box(s, "0.3·BCE(exist)", 8.4, 2.9, 2.1, 0.8, "D6E4F0", C.dark, 13, true);
  box(s, "0.7·CE(type)", 10.7, 2.9, 2.1, 0.8, "D6E4F0", C.dark, 13, true);
  box(s, "λ₁·L_CL\n(intra+inter)", 8.4, 3.9, 2.1, 1.0, "E2EFDA", C.green, 13, true);
  box(s, "λ₂·L_recon", 10.7, 3.9, 2.1, 1.0, "E2EFDA", C.green, 13, true);
  s.addText("Ý đồ: embedding vừa bám supervised signal,\nvừa nhất quán đa-view, vừa giữ cấu trúc gốc", { x: 8.4, y: 5.3, w: 4.4, h: 1.0, fontSize: 12.5, italic: true, color: C.gray });
  footer(s, N);
}

// ---------- 7. PROTOCOL ----------
{
  const s = slide();
  header(s, "6. Protocol đánh giá", "5-fold CV + negative resampling — chuẩn của lĩnh vực MDA");
  bullets(s, [
    { text: "Chia positive thành 5 fold; mỗi lần 1 fold làm test, 4 fold train." },
    { text: "Negatives (cặp chưa biết quan hệ): resample ngẫu nhiên mỗi fold — paper không công bố tỉ lệ; code gốc mặc định ~17:1 theo tỉ lệ dữ liệu." },
    { text: "Top-1 F1: với mỗi cặp test positive, loại quan hệ dự đoán rank 1 phải đúng; F1 tổng hợp trên precision/recall." },
    { text: "AUC / AUPR: đánh giá khả năng tách positive–negative (existence), bỏ qua loại." },
    { text: "650 epoch, AdamW, CPU-only; seed cố định cho so sánh công bằng." },
  ], 0.5, 1.3, 8.0, 5.4, 16);
  box(s, "5-fold CV", 9.0, 1.8, 3.4, 1.0, C.lblue, C.navy, 22, true);
  box(s, "neg resample\nmỗi fold", 9.0, 3.0, 3.4, 1.0, "FFF2CC", C.gold, 15, true);
  box(s, "650 epochs\nAdamW · CPU", 9.0, 4.2, 3.4, 1.0, "E2EFDA", C.green, 15, true);
  footer(s, N);
}

// ---------- 8. PAPER RESULTS ----------
{
  const s = slide();
  header(s, "7. Kết quả paper (reference)", "SOTA CV_type trên HMDD v2.0 — Sun et al., BMC Bioinformatics 2026");
  const rows = [
    ["Metric", "Paper v2.0", "Ghi chú"],
    ["Top-1 Precision", "0.5842", "loại đúng ở rank 1"],
    ["Top-1 Recall", "0.6341", ""],
    ["Top-1 F1", "0.5970", "north-star metric"],
    ["AUC", "0.9669", "existence tách pos/neg"],
    ["AUPR", "0.9738", ""],
    ["F1 (binary)", "0.9278", ""],
  ];
  s.addTable(rows, { x: 0.6, y: 1.4, w: 7.2, fontFace: FONT, fontSize: 15, border: { pt: 0.5, color: "AAAAAA" }, fill: "FFFFFF", color: C.dark,
    autoPage: false, rowH: 0.55, valign: "middle",
    colW: [2.6, 1.8, 2.8],
    });
  s.addText([
    { text: "Claim paper: dual-view + CL vượt mọi baseline MDA-type.", options: { bullet: true, breakLine: true } },
    { text: "Ablation paper (Fig. 4): bỏ CL → giảm; bỏ HGT → giảm — mọi thành phần đều 'có ích'.", options: { bullet: true, breakLine: true } },
    { text: "v3.2 (5 loại): Top-1 F1 = 0.86 trên bộ 411×271 chưa public.", options: { bullet: true, breakLine: true } },
  ], { x: 0.6, y: 5.45, w: 7.6, h: 1.6, fontFace: FONT, fontSize: 13, color: C.dark, paraSpaceAfter: 4 });
  box(s, "Top-1 F1\n0.5970", 8.8, 1.7, 3.8, 1.7, C.navy, C.white, 26, true);
  box(s, "AUC\n0.9669", 8.8, 3.7, 3.8, 1.7, C.blue, C.white, 26, true);
  footer(s, N);
}

// ---------- 9. REPRODUCE ----------
{
  const s = slide();
  header(s, "8. Tái hiện — khớp và phát hiện lệch", "Baseline khớp paper; 3 phát hiện đáng chú ý");
  bullets(s, [
    { text: "✔ Baseline v2.0 tái hiện được: Top-1 F1 ~0.59–0.60, binary metrics khớp/vượt paper." },
    { text: "⚠ Ablation NGƯỢC paper: bỏ CL hoặc HGT lại TĂNG Top-1 (+9.7pp / +8.4pp sau true-rebuild) — thành phần 'trung tâm' của paper hóa ra là noise trên v2.0. Đã verify 4 cách độc lập → finding thật, không phải artifact." },
    { text: "⚠ Predictor trong code release suy biến (bilinear diag rank-d) vs paper dùng full bilinear → sửa = +5.9pp Top-1." },
    { text: "⚠ Lỗi metric v3.2: code eval hardcode 4 loại → toàn bộ 0.0000 cho v3.2 là bug metric, không phải model collapse (đo lại đúng: ~0.27-0.36)." },
    { text: "⚠ Bug seed/dead flags: --lr, --weight_decay, --similarity_threshold trước đây không tới được optimizer/đồ thị." },
  ], 0.5, 1.3, 12.4, 5.6, 16);
  footer(s, N);
}

// ---------- 10. IMPROVE ----------
{
  const s = slide();
  header(s, "9. Cải thiện kết quả — headline", "Số liệu trung thực: tách rõ like-for-like vs system-level");
  const rows = [
    ["", "Top-1 F1", "AUC", "vs paper"],
    ["Paper v2.0", "0.5970", "0.9669", "—"],
    ["Single-model tốt nhất (neg_ratio=5)", "0.6923 (mean ±0.01)", "0.9844", "+16.0% / +1.8%"],
    ["Ensemble 6-member, split độc lập (s42)", "0.7041", "0.9899", "+18.0% / +2.4%"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.4, w: 8.2, fontFace: FONT, fontSize: 14, border: { pt: 0.5, color: "AAAAAA" }, color: C.dark, rowH: 0.6, valign: "middle", colW: [4.0, 1.9, 1.2, 1.1] });
  bullets(s, [
    { text: "Công thức thắng (pure config, không sửa kiến trúc): softmax 5-class + bỏ khối contrastive (no_cl_rebuild) + full bilinear + negative ratio 5:1." },
    { text: "Ensemble = 4 seeds × config thắng + 1 no-HGT + 1 diag-predictor — trung bình xác suất; verify trên split CV độc lập (folds_s42) để loại selection bias." },
    { text: "Min seed single-model = 0.6844 → mọi seed đều vượt paper ≥ +14.6%. Kết quả chắc, không phải may mắn một seed." },
  ], 0.6, 4.0, 8.4, 2.9, 15);
  box(s, "+18%", 9.4, 1.7, 3.1, 1.4, C.green, C.white, 34, true);
  box(s, "Top-1 F1", 9.4, 3.1, 3.1, 0.5, C.green, C.white, 14);
  box(s, "0.7041", 9.4, 4.0, 3.1, 1.4, C.navy, C.white, 30, true);
  box(s, "ensemble · split s42", 9.4, 5.4, 3.1, 0.5, C.navy, C.white, 13);
  footer(s, N);
}

// ---------- 11. INSIGHTS ----------
{
  const s = slide();
  header(s, "10. Insight chính rút ra", "Cái gì thực sự quyết định kết quả");
  bullets(s, [
    { text: "Contrastive learning là noise trên v2.0 — bỏ hẳn cả intra+inter CL tăng +9.7pp. CL của paper có thể chỉ có ích ở regime data khác (dày hơn / nhiều type hơn).", opts: {} },
    { text: "Negatives làn đường: tỉ lệ ~17:1 của code gốc nhấn chìm tín hiệu 'loại'. Sweet spot đo được: 5:1 (curve 10→0.6917 / 5→0.7014 / 3→0.6993)." },
    { text: "Faithfulness ≠ tầm thường: diag→full bilinear là fix 'đúng paper' mà +5.9pp — code release ≠ paper spec." },
    { text: "Ensemble: đa dạng cấu hình > đa dạng seed. Members 'stacked nhiều mechanism' decorrelate tốt hơn; 8 members cùng family thì triệt tiêu lẫn nhau." },
    { text: "Bugs có giá trị: 3 flag chết được nối lại; metric v3.2 được vá đúng; negative pool chứa ~3 900 true positives bị gắn nhãn sai (~2% eval set)." },
  ], 0.5, 1.3, 12.4, 5.7, 16);
  footer(s, N);
}

// ---------- 12. CEILING ----------
{
  const s = slide();
  header(s, "11. Giới hạn cải thiện & hướng mở", "Đã cận trần trong không gian config/ensemble — data là biên còn lại");
  bullets(s, [
    { text: "Đánh giá độc lập 5 lens (council review): không gian config/ensemble ~đạt trần — các delta còn lại nằm dưới noise floor." },
    { text: "Hướng data-level CHƯA test, upside ước ~+1–4pp:", opts: { bold: true } },
    { text: "· Decontaminate negative pool — bug thật: 3 932 quan hệ đã xác minh bị sample làm negatives.", opts: { indentLevel: 1 } },
    { text: "· Auxiliary supervision từ dataset v3.2_filtered (3 938 assoc typed, cùng entity indices).", opts: { indentLevel: 1 } },
    { text: "· Mục tiêu multilabel (181 cặp đa-type hiện bị gộp last-write-wins).", opts: { indentLevel: 1 } },
    { text: "v3.2 paper (0.86) còn xa: chênh lệch chủ yếu do data curation 411×271 không public — không tái lập được bằng tuning." },
  ], 0.5, 1.3, 8.4, 5.6, 16);
  box(s, "Config space\n≈ TRẦN", 9.3, 1.8, 3.2, 1.3, "F8CBAD", C.red, 22, true);
  box(s, "Data space\nCÒN MỞ +1–4pp", 9.3, 3.4, 3.2, 1.3, "E2EFDA", C.green, 18, true);
  box(s, "v3.2 gap\n= data curation", 9.3, 5.0, 3.2, 1.3, C.lblue, C.navy, 18, true);
  footer(s, N);
}

// ---------- 13. CONCLUSION ----------
{
  const s = slide();
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 13.33, h: 7.5, fill: { color: C.navy } });
  s.addText("Kết luận", { x: 0.8, y: 0.9, w: 11.7, h: 0.8, fontFace: FONT, fontSize: 40, bold: true, color: C.white });
  s.addShape(pptx.shapes.RECTANGLE, { x: 0.8, y: 1.9, w: 3.0, h: 0.05, fill: { color: C.gold } });
  s.addText([
    { text: "Tái hiện baseline thành công; phát hiện và sửa nhiều bug thật trong code release.", options: { bullet: { code: "2022" }, breakLine: true } },
    { text: "Cải thiện vượt paper bằng config + ensemble, không cần thay kiến trúc:", options: { bullet: { code: "2022" }, breakLine: true } },
    { text: "Top-1 F1  0.5970 → 0.7041   (+18.0%)", options: { bullet: false, bold: true, color: "FFD966", fontSize: 26, breakLine: true } },
    { text: "AUC         0.9669 → 0.9899   (+2.4%)", options: { bullet: false, bold: true, color: "FFD966", fontSize: 26, breakLine: true } },
    { text: "Kết quả kiểm chứng trên split độc lập; mọi seed đều vượt paper.", options: { bullet: { code: "2022" }, breakLine: true } },
    { text: "Hướng mở còn lại: sửa nhãn dữ liệu (negative pool) + auxiliary supervision — data-level, ~+1–4pp.", options: { bullet: { code: "2022" }, breakLine: true } },
  ], { x: 1.0, y: 2.3, w: 11.3, h: 4.3, fontFace: FONT, fontSize: 19, color: C.lblue, paraSpaceAfter: 14 });
  s.addText("DHGCMDA fork · experiment loop 2026", { x: 0.8, y: 6.9, w: 11.7, h: 0.4, fontFace: FONT, fontSize: 12, color: "9DC3E6", align: "center" });
}

const OUT = path.join(REPO, "Slide_BaoCao_DHGCMDA.pptx");
pptx.writeFile({ fileName: OUT }).then(() => console.log("WROTE", OUT));
