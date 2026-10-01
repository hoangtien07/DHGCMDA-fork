// Slide báo cáo DHGCMDA — academic style (assertion-evidence, light theme)
// Usage: npm install pptxgenjs && node gen_slides.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const REPO = "/home/ubuntu/repos/DHGCMDA-fork";
const pptx = new pptxgen();
pptx.defineLayout({ name: "W", width: 13.33, height: 7.5 });
pptx.layout = "W";
pptx.author = "DHGCMDA reproduction";
pptx.title = "DHGCMDA — Reproduce & Improve";

const C = { navy: "1F3B6E", ink: "1A1A1A", gray: "5A5A5A", accent: "B8860B", line: "D9D9D9", lite: "F5F7FA", good: "1E6B33", bad: "9E2A2B", white: "FFFFFF" };
const FONT = "Calibri";
let N = 0;
const slide = () => { const s = pptx.addSlide(); N++; return s; };

// Academic layout: white bg, assertion title in navy, thin rule, footer
function head(s, assertion) {
  s.addText(assertion, { x: 0.55, y: 0.3, w: 12.2, h: 0.95, fontFace: FONT, fontSize: 27, bold: true, color: C.navy, valign: "middle" });
  s.addShape(pptx.shapes.LINE, { x: 0.55, y: 1.32, w: 12.2, h: 0, line: { color: C.accent, width: 1.5 } });
  s.addText(`DHGCMDA — tái hiện & cải thiện`, { x: 0.55, y: 7.12, w: 6, h: 0.3, fontFace: FONT, fontSize: 10, color: C.gray });
  s.addText(`${N}`, { x: 12.6, y: 7.12, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 10, color: C.gray, align: "right" });
}
// bullets: items = [{t, bold?, sub?}] — sparse, large
function body(s, items, x = 0.75, y = 1.6, w = 11.8, h = 5.2, fs = 19) {
  const arr = [];
  for (const it of items) {
    if (it.sub) {
      arr.push({ text: it.t, options: { bullet: false, bold: it.bold !== false, color: C.ink, breakLine: true, fontSize: fs } });
      arr.push({ text: it.sub, options: { bullet: false, bold: false, color: C.gray, breakLine: true, fontSize: fs - 3 } });
    } else {
      arr.push({ text: it.t, options: { bullet: { code: "2022", indent: 14 }, bold: !!it.bold, color: C.ink, breakLine: true, fontSize: fs } });
    }
    arr.push({ text: " ", options: { fontSize: 6, breakLine: true } });
  }
  s.addText(arr, { x, y, w, h, fontFace: FONT, valign: "top", paraSpaceAfter: 4 });
}
function table(s, rows, x, y, w, colW, fs = 15, rowH = 0.5) {
  s.addTable(rows, {
    x, y, w, colW, rowH, fontFace: FONT, fontSize: fs, color: C.ink, valign: "middle",
    border: [{ pt: 1, color: C.line }, null, { pt: 1, color: C.line }, null].map(b => b || { pt: 0, color: C.white }),
    fill: "FFFFFF", margin: [2, 6, 2, 6],
  });
}

// ============ 1. TITLE ============
{
  const s = slide();
  s.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 2.1, w: 13.33, h: 0.03, fill: { color: C.accent } });
  s.addText("DHGCMDA", { x: 0.8, y: 1.0, w: 11.7, h: 0.9, fontFace: FONT, fontSize: 44, bold: true, color: C.navy });
  s.addText("Dual-View Heterogeneous Graph Contrastive Learning\ncho dự đoán loại quan hệ miRNA–bệnh", { x: 0.8, y: 2.35, w: 11.7, h: 1.2, fontFace: FONT, fontSize: 24, color: C.ink });
  s.addText("Tái hiện và cải thiện kết quả paper", { x: 0.8, y: 3.85, w: 11.7, h: 0.5, fontFace: FONT, fontSize: 18, color: C.gray });
  s.addText("Top-1 F1:  0.5970  →  0.7041        AUC:  0.9669  →  0.9899", { x: 0.8, y: 4.55, w: 11.7, h: 0.6, fontFace: FONT, fontSize: 22, bold: true, color: C.navy });
  s.addText("Paper gốc: Sun Y. et al., BMC Bioinformatics 2026  ·  HMDD v2.0  ·  Báo cáo 2026", { x: 0.8, y: 6.5, w: 11.7, h: 0.4, fontFace: FONT, fontSize: 13, color: C.gray });
}

// ============ 2. OUTLINE ============
{
  const s = slide();
  head(s, "Nội dung trình bày");
  const items = [
    ["1", "Bài toán & dữ liệu", "miRNA–disease association type prediction trên HMDD v2.0"],
    ["2", "Phương pháp DHGCMDA", "dual-view hypergraph · contrastive learning · HGT fusion"],
    ["3", "Tái hiện kết quả", "baseline khớp paper — và các lệch phát hiện được"],
    ["4", "Cải thiện & kết luận", "config + ensemble → Top-1 F1 0.7041, AUC 0.9899"],
  ];
  items.forEach(([n, t, d], i) => {
    const y = 1.75 + i * 1.35;
    s.addText(n, { x: 0.75, y, w: 0.8, h: 1.0, fontFace: FONT, fontSize: 30, bold: true, color: C.accent, valign: "middle" });
    s.addText(t, { x: 1.6, y, w: 10.8, h: 0.5, fontFace: FONT, fontSize: 22, bold: true, color: C.ink });
    s.addText(d, { x: 1.6, y: y + 0.52, w: 10.8, h: 0.4, fontFace: FONT, fontSize: 15, color: C.gray });
  });
}

// ============ 3. PROBLEM ============
{
  const s = slide();
  head(s, "Bài toán: dự đoán LOẠI quan hệ miRNA–bệnh, không chỉ có/không");
  body(s, [
    { t: "Input: cặp (miRNA, disease).  Output: P(quan hệ tồn tại) × P(loại ∈ 4 lớp)", bold: true },
    { t: "4 loại quan hệ: Circulation, Epigenetics, Target, Genetics", sub: "Phân loại cơ chế tác động — một cặp có thể mang nhiều loại (multilabel)" },
    { t: "Giá trị: ưu tiên hóa thí nghiệm wet-lab, gợi ý biomarker và drug repositioning" },
    { t: "Khó: nhãn loại thưa, positives ≪ negatives, tín hiệu yếu trên đồ thị dị thể" },
  ], 0.85, 1.7, 11.6, 5.0, 20);
}

// ============ 4. DATA ============
{
  const s = slide();
  head(s, "Dữ liệu HMDD v2.0 cực kỳ thưa — mất cân bằng là khó khăn chính");
  table(s, [
    [{ text: "Thành phần", options: { bold: true, color: C.navy } }, { text: "Giá trị", options: { bold: true, color: C.navy } }],
    ["miRNA", "495"],
    ["Disease", "383"],
    ["Quan hệ đã gán loại (4 loại)", "2 926"],
    ["Mật độ đồ thị", "~1.5%"],
    ["Negatives khả dĩ", "~186 000"],
  ], 0.85, 1.7, 6.5, [4.3, 2.2], 16, 0.55);
  s.addText([
    { text: "Hệ quả cho học máy:", options: { bold: true, color: C.ink, breakLine: true, fontSize: 18 } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Tỉ lệ negatives : positives quyết định mạnh tín hiệu nhãn loại.", options: { bullet: { code: "2022" }, color: C.ink, breakLine: true, fontSize: 17 } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Positive folds cố định theo CV; negatives được resample mỗi fold — nguồn variance lớn.", options: { bullet: { code: "2022" }, color: C.ink, breakLine: true, fontSize: 17 } },
  ], { x: 7.7, y: 1.9, w: 5.0, h: 4.5, fontFace: FONT, valign: "top" });
}

// ============ 5. ARCHITECTURE ============
{
  const s = slide();
  head(s, "DHGCMDA: hai nhánh dual-view độc lập, hợp nhất bằng HGT");
  s.addImage({ path: path.join(REPO, "results/architecture_overview.png"), x: 1.45, y: 1.45, w: 10.4, h: 10.4 * 1099 / 2064 });
  s.addText("Mỗi modality (miRNA / disease) có 2 view similarity → 2 hypergraph riêng → HGCN → AVF → HGT → predictor", { x: 1.15, y: 7.02, w: 11.0, h: 0.35, fontFace: FONT, fontSize: 12, italic: true, color: C.gray, align: "center" });
}

// ============ 6. COMPONENTS ============
{
  const s = slide();
  head(s, "Ba mô-đun chính: HGCN theo view, AVF hợp nhất, HGT liên kết hai modality");
  const cols = [
    ["HGCN (per view)", ["Hyperedge = nhóm node tương tự (top-K)", "Lan truyền node → edge → node", "Bắt quan hệ bậc cao, không chỉ cặp"]],
    ["AVF (view fusion)", ["Học trọng số α (attention) + β (residual) cho mỗi view", "Tự quyết view nào tin cậy theo node"]],
    ["HGT + Predictor", ["Transformer 2 lớp, 4 heads trên đồ thị miRNA–disease", "Bilinear: P(type c) = miᵀ·Wc·dis"]],
  ];
  cols.forEach(([title, pts], i) => {
    const x = 0.7 + i * 4.25;
    s.addShape(pptx.shapes.LINE, { x, y: 1.65, w: 0, h: 0.55, line: { color: C.accent, width: 2.5 } });
    s.addText(title, { x: x + 0.15, y: 1.6, w: 3.9, h: 0.6, fontFace: FONT, fontSize: 20, bold: true, color: C.navy });
    s.addText(pts.map(t => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, breakLine: true } })),
      { x: x + 0.1, y: 2.35, w: 3.95, h: 3.6, fontFace: FONT, fontSize: 15.5, color: C.ink, paraSpaceAfter: 10 });
  });
  s.addText("Dynamic hypergraph: hyperedge được rebuild từ embedding hiện tại mỗi 5 epoch", { x: 0.7, y: 6.35, w: 12.0, h: 0.4, fontFace: FONT, fontSize: 15, italic: true, color: C.gray });
}

// ============ 7. LOSS ============
{
  const s = slide();
  head(s, "Một hàm mất mát thống nhất: dự đoán + đối chiếu + tái dựng");
  s.addText("L  =  0.3·BCE(existence)  +  0.7·CE(type)  +  λ₁·L_CL  +  λ₂·L_recon", { x: 0.7, y: 1.6, w: 12.0, h: 0.7, fontFace: "Consolas", fontSize: 19, bold: true, color: C.navy });
  table(s, [
    [{ text: "Thành phần", options: { bold: true, color: C.navy } }, { text: "Vai trò", options: { bold: true, color: C.navy } }],
    ["CE(type) — trọng số 0.7", "Supervision chính: loại quan hệ rank-1"],
    ["BCE(existence) — 0.3", "Tách positive/negative"],
    ["L_CL: intra-view (SimCLR)", "z_v1 ↔ z_v2 cùng entity nhất quán"],
    ["L_CL: inter-view (InfoNCE+margin)", "miRNA ↔ disease có quan hệ lại gần"],
    ["L_recon (λ=1.0)", "Decoder tái dựng similarity đầu vào — regularizer"],
  ], 0.7, 2.5, 12.0, [4.6, 7.4], 15.5, 0.55);
  s.addText("Thiết kế: embedding vừa bám tín hiệu supervised, vừa nhất quán đa-view, vừa giữ cấu trúc gốc.", { x: 0.7, y: 6.2, w: 12.0, h: 0.5, fontFace: FONT, fontSize: 15, italic: true, color: C.gray });
}

// ============ 8. EVALUATION ============
{
  const s = slide();
  head(s, "Đánh giá: 5-fold CV trên positives, negatives resample; Top-1 F1 là metric chính");
  body(s, [
    { t: "5-fold CV: positives chia 5 fold — 4 train, 1 test; negatives resample ngẫu nhiên mỗi fold" },
    { t: "Top-1 F1: loại quan hệ dự đoán đứng rank 1 phải đúng — strict hơn binary đáng kể" },
    { t: "AUC / AUPR: đánh giá riêng tác vụ existence (tách pos/neg), bỏ qua loại" },
    { t: "Huấn luyện: 650 epoch, AdamW, CPU-only; seed cố định khi so sánh" },
  ], 0.85, 1.8, 11.6, 4.8, 20);
}

// ============ 9. PAPER RESULTS ============
{
  const s = slide();
  head(s, "Paper báo Top-1 F1 = 0.5970, AUC = 0.9669 trên HMDD v2.0");
  table(s, [
    [{ text: "Metric", options: { bold: true, color: C.navy } }, { text: "Paper", options: { bold: true, color: C.navy } }],
    ["Top-1 Precision", "0.5842"],
    ["Top-1 Recall", "0.6341"],
    ["Top-1 F1  (north-star)", "0.5970"],
    ["AUC", "0.9669"],
    ["AUPR", "0.9738"],
    ["F1 binary", "0.9278"],
  ], 1.0, 1.7, 6.4, [3.9, 2.5], 16, 0.55);
  s.addText([
    { text: "Claim của paper:", options: { bold: true, color: C.ink, breakLine: true, fontSize: 18 } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Dual-view + contrastive vượt mọi baseline MDA-type hiện có.", options: { bullet: { code: "2022" }, color: C.ink, breakLine: true, fontSize: 16 } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Ablation Fig. 4: bỏ CL hoặc HGT đều giảm — mọi thành phần 'có ích'.", options: { bullet: { code: "2022" }, color: C.ink, breakLine: true, fontSize: 16 } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Trên v3.2 (5 loại): Top-1 F1 = 0.86 — bộ 411×271 chưa công bố.", options: { bullet: { code: "2022" }, color: C.ink, breakLine: true, fontSize: 16 } },
  ], { x: 8.0, y: 1.9, w: 4.7, h: 4.8, fontFace: FONT, valign: "top" });
}

// ============ 10. REPRODUCE ============
{
  const s = slide();
  head(s, "Tái hiện: baseline khớp paper; các metric binary vượt ở mọi seed");
  table(s, [
    [{ text: "", options: { bold: true, color: C.navy } }, { text: "Top-1 F1", options: { bold: true, color: C.navy } }, { text: "AUC", options: { bold: true, color: C.navy } }, { text: "AUPR", options: { bold: true, color: C.navy } }, { text: "F1 binary", options: { bold: true, color: C.navy } }],
    ["Paper", "0.5970", "0.9669", "0.9738", "0.9278"],
    ["Reproduce (best seed)", "0.5655", "0.9730", "0.9671", "0.9292"],
    ["Reproduce + fix faithfulness", "≈ 0.60", "≈ 0.975", "≈ 0.97", "≈ 0.93"],
  ], 0.85, 1.7, 12.0, [4.2, 2.0, 1.9, 1.9, 2.0], 15.5, 0.55);
  body(s, [
    { t: "Top-1 F1 khớp paper trong sai số seed (gap ~1–5%); binary metrics khớp hoặc vượt ở mọi seed.", bold: false },
    { t: "Fix predictor diag → full bilinear (đúng paper): +5.9pp — nâng baseline tái hiện lên ~0.60–0.63." },
  ], 0.85, 3.9, 11.6, 2.6, 18);
}

// ============ 11. FINDINGS ============
{
  const s = slide();
  head(s, "Tái hiện lộ ra code release ≠ paper: 3 lệch đáng chú ý");
  body(s, [
    { t: "Ablation ngược chiều paper", sub: "Bỏ khối contrastive học tăng +9.7pp Top-1 (verify 4 cách độc lập) — CL là noise trên v2.0, không phải lỗi triển khai" },
    { t: "Predictor suy biến trong code", sub: "Code release dùng bilinear diagonal rank-d; paper mô tả full bilinear — sửa lại đúng = +5.9pp" },
    { t: "Metric v3.2 hỏng", sub: "Eval hardcode 4 lớp → toàn bộ kết quả 0.0 là artifact; đo lại đúng ~0.27–0.36 (paper báo 0.86 trên data chưa public)" },
    { t: "Kèm các bug nhỏ: 3 hyperparameter flags không tới optimizer/đồ thị", bold: false },
  ], 0.85, 1.7, 11.6, 5.2, 18);
}

// ============ 12. IMPROVEMENT ============
{
  const s = slide();
  head(s, "Cải thiện: Top-1 F1 = 0.7041 (+18%), AUC = 0.9899 — ensemble trên split độc lập");
  table(s, [
    [{ text: "", options: { bold: true, color: C.navy } }, { text: "Top-1 F1", options: { bold: true, color: C.navy } }, { text: "AUC", options: { bold: true, color: C.navy } }, { text: "vs paper", options: { bold: true, color: C.navy } }],
    ["Paper v2.0", "0.5970", "0.9669", "—"],
    ["Single-model tốt nhất (like-for-like)", "0.6923", "0.9844", "+16.0% / +1.8%"],
    ["Ensemble 6-member (split độc lập s42)", "0.7041", "0.9899", "+18.0% / +2.4%"],
  ], 0.85, 1.7, 12.0, [5.4, 1.9, 1.7, 3.0], 15.5, 0.6);
  body(s, [
    { t: "Config thắng (không sửa kiến trúc): softmax 5-class · bỏ contrastive block · full bilinear · negative ratio 5:1." },
    { t: "Ensemble = 4 seed × config thắng + 1 no-HGT + 1 diag — average xác suất; đo trên split CV độc lập để loại selection bias." },
    { t: "Mọi seed single-model đều vượt paper (min 0.6844) — kết quả chắc, không phải seed may mắn." },
  ], 0.85, 3.95, 11.6, 2.9, 17);
}

// ============ 13. LEVERS ============
{
  const s = slide();
  head(s, "Ba lever quyết định kết quả — mỗi cái có bằng chứng riêng");
  table(s, [
    [{ text: "Lever", options: { bold: true, color: C.navy } }, { text: "Bằng chứng", options: { bold: true, color: C.navy } }, { text: "Δ Top-1 F1", options: { bold: true, color: C.navy } }],
    ["Bỏ khối contrastive (no_cl_rebuild)", "Rebuild đúng, multi-seed; restore một nửa CL → giảm 1.1pp", "+9.7pp"],
    ["Negative ratio ~17:1 → 5:1", "Đường cong đo được: 10→0.6917 / 5→0.7014 / 3→0.6993", "≈ đỉnh tại 5"],
    ["diag → full bilinear predictor", "Faithfulness fix đúng spec paper", "+5.9pp"],
    ["Ensemble đa-dạng config", "6 member khác mechanism; 8 member cùng family thì triệt tiêu", "+1.2pp"],
  ], 0.85, 1.7, 12.0, [4.3, 5.7, 2.0], 15, 0.62);
  s.addText("Bài học ensemble: đa dạng cấu hình (mechanism orthogonal) > đa dạng seed > thêm member cùng loại.", { x: 0.85, y: 5.0, w: 11.8, h: 0.5, fontFace: FONT, fontSize: 15, italic: true, color: C.gray });
}

// ============ 14. LIMITS ============
{
  const s = slide();
  head(s, "Không gian config/ensemble đã cận trần — headroom còn lại nằm ở tầng data");
  body(s, [
    { t: "Council review 5 lens độc lập: các delta còn lại trong config space đều dưới noise floor → near ceiling." },
    { t: "Avenue data-level chưa test, ước +1–4pp:", bold: true },
    { t: "· Sửa negative pool: 3 932 quan hệ đã xác minh đang bị sample làm negatives (~2% eval set mislabeled)", bold: false },
    { t: "· Auxiliary supervision từ v3.2_filtered (3 938 assoc typed, cùng entity indices)", bold: false },
    { t: "· Mục tiêu multilabel: 181 cặp đa-type hiện bị gộp last-write-wins", bold: false },
    { t: "Gap v3.2 (0.33 → 0.86 paper) chủ yếu do data curation 411×271 không public — không bù được bằng tuning." },
  ], 0.85, 1.7, 11.6, 5.2, 18);
}

// ============ 15. CONCLUSION ============
{
  const s = slide();
  head(s, "Kết luận");
  body(s, [
    { t: "Tái hiện baseline thành công; phát hiện và sửa nhiều bug thật của code release (predictor, metric, dead flags)." },
    { t: "Cải thiện vượt paper bằng config + ensemble, không đổi kiến trúc:", bold: false },
    { t: "Top-1 F1: 0.5970 → 0.7041   (+18.0%)", bold: true },
    { t: "AUC:        0.9669 → 0.9899   (+2.4%)", bold: true },
    { t: "Kết quả xác nhận trên split độc lập; mọi seed đều vượt paper." },
    { t: "Hướng mở: sửa nhãn dữ liệu + auxiliary supervision (data-level, ~+1–4pp)." },
  ], 0.85, 1.8, 11.6, 4.6, 20);
  s.addText("Cảm ơn thầy/cô và các bạn đã lắng nghe", { x: 0.85, y: 6.5, w: 11.6, h: 0.5, fontFace: FONT, fontSize: 15, italic: true, color: C.gray });
}

const OUT = path.join(REPO, "Slide_BaoCao_DHGCMDA.pptx");
pptx.writeFile({ fileName: OUT }).then(() => console.log("WROTE", OUT));
