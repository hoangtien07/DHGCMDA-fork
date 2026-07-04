# Biblio A — Dòng dõi DHGCMDA & MDA methods (SESSION 2, 2026-07-05)

> Kỷ luật: chỉ ghi bài **đã fetch/đọc primary source**. Số liệu đều là **quote nguyên văn** từ nguồn (R4/R14).
> Trạng thái theo R2. Adversarial pass (R6/R9/R15) ở cuối.

---

## A. BẢNG PROVENANCE (6 bài — 5 VERIFIED đầy đủ + 1 metadata-only)

### A1. DHGCMDA (bài gốc đang reproduce) — **VERIFIED (primary PDF cục bộ)**
- **Tiêu đề:** DHGCMDA: a dual-view heterogeneous graph contrastive learning framework for miRNA-disease association type prediction
- **Tác giả:** Yan Sun, Fanyu Zhang, Shijia Yan, Xiaotong Kong, Hanxiang Wang, Junliang Shang, Jin-Xing Liu
- **Năm/Venue:** BMC Bioinformatics, 2026 (Received 5 Feb 2026, Accepted 24 Mar 2026; "Article in Press", Open Access CC BY-NC-ND)
- **DOI:** 10.1186/s12859-026-06436-w
- **Nguồn:** PDF gốc tại repo (`_pdf_text/p27.txt`, `p28.txt`) — landing Springer bị cookie-wall khi fetch, nhưng ta có bản PDF chính thức (OA) → primary source.
- **Số liệu nguyên văn (Table 3, CV type — cột HMDD v2.0 | HMDD v3.2):**
  - v2.0 Top-1: **P=0.5842, R=0.6341, F1=0.5970**
  - v3.2 Top-1: **P=0.7915, R=0.9421, F1=0.8600**
  - Quote: *"on HMDD v3.2, DHGCMDA delivers a Top-1 precision of 0.7915, recall of 0.9421 and F1-score of 0.8600 … On HMDD v2.0, DHGCMDA achieves the highest Top-1 recall among all methods. While its precision lags behind … DHGCMDA still maintains a top-tier F1-score."* (p27)
- **Số liệu nguyên văn (Table 4, CV triplet — binary):**
  - v2.0: **AUPR=0.9738, AUC=0.9669, F1=0.9278**
  - v3.2: **AUPR=0.9271, AUC=0.9181, F1=0.8674**
- **Đóng góp (quote):** dual-view hypergraph + contrastive learning capture "high-order biological relationships"; *"the moderate precision on HMDD v2.0 is a deliberate trade-off via class weighting, which prioritizes true association recall"* (p27) — **xác nhận có class-weighting, khớp phát hiện loss của ta**.
- **Baseline paper tự so:** TDRC[22], SPLDHyperAWNTF[24], TFLP[25], NMCMDA[28], MRFGMDA[33], KBLTDARD[34].
- **Đối chiếu với reproduce của ta:** paper v2.0 Top-1 F1 = 0.5970; ta đạt **0.688 ± 0.011** (full_bilinear + K_neigs=3). AUC v2.0 paper 0.9669; ta 0.9810. → khớp/vượt (đã ghi CLAUDE.md).

### A2. NMCMDA — **VERIFIED-fetched (OUP)**
- **Tiêu đề:** NMCMDA: neural multicategory MiRNA–disease association prediction
- **Tác giả:** Jingru Wang, Jin Li, Kun Yue, Li Wang, Yuyun Ma, Qing Li
- **Năm/Venue:** Briefings in Bioinformatics, Vol 22(5), Sep 2021
- **DOI:** 10.1093/bib/bbab074 · **URL:** https://academic.oup.com/bib/article/22/5/bbab074/6189772
- **Datasets:** MCD-6 (HMDD v3.2, 894 disease × 1208 miRNA, 6 types, 25,849 assoc); MCD-20 (20 types); + TDRC-v2.0 (169×324, 4 types), TDRC-v3.2 (447×713, 5 types).
- **Số liệu (Table 2, MCD-6):** NMR-RGCN **P=0.5522, R=0.3967, F1=0.4617**; TDRC trên cùng data **F1=0.3422**.
  - Quote: *"NMR-RGCN is significantly superior to the state-of-the-art method TDRC in terms of Top-1 precision, Top-1 Recall and Top-1 F1."*
- **Đóng góp (quote):** *"a novel data-driven end-to-end learning-based method of neural multiple-category miRNA–disease association prediction (NMCMDA)"* — encoder RGCN + neural multi-relational decoder.
- **Trạng thái ta:** đã reproduce một phần; blocked bởi DGL incompat torch 2.5.1 (CLAUDE.md).

### A3. TDRC — **VERIFIED-fetched (OUP)**
- **Tiêu đề:** Tensor decomposition with relational constraints for predicting multiple types of microRNA-disease associations
- **Tác giả:** Feng Huang, Xiang Yue, Zhankun Xiong, Zhouxin Yu, Shichao Liu, Wen Zhang
- **Năm/Venue:** Briefings in Bioinformatics, Vol 22(3), May 2021 (published 28 Jul 2020)
- **DOI:** 10.1093/bib/bbaa140 · **URL:** https://academic.oup.com/bib/article/22/3/bbaa140/5876601
- **Datasets:** v2.0 = 1,675 assoc, 324 miRNA, 169 disease, 4 types; v3.2 = 16,341 assoc, 713 miRNA, 447 disease, 5 types.
- **Số liệu (Table 2, Top-1):** v2.0 **P=0.5609, R=0.4999, F1=0.5286**; v3.2 **P=0.6178, R=0.4741, F1=0.5365**.
- **Số liệu (Table 3, binary):** v2.0 AUPR=0.8663/AUC=0.8379/F1=0.8014; v3.2 AUPR=0.9284/AUC=0.9201/F1=0.8643.
- **Đóng góp (quote):** *"We represent the multi-type miRNA-disease associations as a tensor and formulate the multi-type miRNA-disease association prediction as a tensor completion task."*
- **Cải thiện (quote):** *"By comparing with a recent baseline NLPMMDA, the tensor decomposition methods improve up to 38% on Top-1F1."*
- **Trạng thái ta:** reproduce ~98% (CLAUDE.md).

### A4. MHNNMDA — **VERIFIED metadata / UNVERIFIED numbers (paywall)** ⭐ CÙNG NHÓM
- **Tiêu đề:** MHNNMDA: multi-stage hypergraph neural network for predicting miRNA-disease association types
- **Tác giả:** **Yan Sun**, Xiaohan Zhang, Xiaoqi Tang, Defu Qiu, **Junliang Shang, Jin-Xing Liu**
- **Năm/Venue:** Journal of Computer-Aided Molecular Design, 2026
- **DOI:** 10.1007/s10822-026-00821-6
- **Datasets:** đề cập HMDD v3.0 & v4.0 (chi tiết dim ẩn sau paywall).
- **Đóng góp (quote):** *"MHNNMDA integrates multi-source biological data to construct similarity networks … transforms these networks into hypergraph structures to capture higher-order group interactions"* với *"node-level and hyperedge-level attention mechanisms"*.
- **Số liệu:** ❌ KHÔNG lấy được (subscription). → chưa được dùng làm số so sánh.
- **🚨 GHI CHÚ NOVELTY:** Yan Sun + Junliang Shang + Jin-Xing Liu **trùng 3 tác giả với DHGCMDA** → cùng lab, **cùng bài toán type-prediction bằng hypergraph**, xuất bản 2026. Đây là *prior/parallel art gần nhất* — bất kỳ đóng góp "hypergraph cho MDA type" nào của ta phải định vị rõ so với cả DHGCMDA lẫn MHNNMDA.

### A5. MHCLMDA — **VERIFIED-fetched (PMC/OUP)**
- **Tiêu đề:** MHCLMDA: multihypergraph contrastive learning for miRNA–disease association prediction
- **Tác giả:** Wei Peng, Zhichen He, Wei Dai, Wei Lan
- **Năm/Venue:** Briefings in Bioinformatics, 2024
- **DOI:** 10.1093/bib/bbad524 · **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10796254/
- **Dataset:** HMDD v3.2 — **757 miRNA, 435 disease, 7,694 assoc**.
- **Số liệu (Table 1, random-zeroing CV):** MHCLMDA **AUC=0.9454, AUPR=0.9455, F1=0.8749**; HGCNMDA 0.9356/0.9355/0.8602; MKGAT 0.8614/0.8912/0.8425.
- **Loại bài toán:** **BINARY** (có/không), KHÔNG phải type prediction.
- **Đóng góp (quote):** multiple hypergraphs + hypergraph convolution "capturing higher order relationships" + VAE cho đặc trưng phi tuyến.

### A6. HGCLAMIR — **VERIFIED-fetched (PLOS)**
- **Tiêu đề:** HGCLAMIR: Hypergraph contrastive learning with attention mechanism and integrated multi-view representation for predicting miRNA-disease associations
- **Tác giả:** Dong Ouyang, Yong Liang, Jinfeng Wang, Le Li, Ning Ai, Junning Feng, Shanghui Lu, Shuilin Liao, Xiaoying Liu, Shengli Xie
- **Năm/Venue:** PLOS Computational Biology, 2024
- **DOI:** 10.1371/journal.pcbi.1011927
- **Datasets:** MDAv2.0 (495 miRNA × 380 disease, 5425 assoc); MDAv3.2 (917 miRNA × 486 disease, 9732 assoc).
- **Số liệu (Table 1, 5-fold):** v2.0 AUC=0.945284/AUPR=0.933251/F1=0.869305; v3.2 AUC=0.957231/AUPR=0.942157/F1=0.889234.
- **Loại bài toán:** **BINARY**. Case study: breast 49/50, lung 48/50 top-50 confirmed.
- **Liên hệ ta:** kiến trúc cùng họ (HGCN + contrastive + view-aware attention) nhưng HGCLAMIR làm **binary**, DHGCMDA làm **type**.
- **✅ ĐÍNH CHÍNH (user xác nhận 2026-07-05):** DHGCMDA **KHÔNG** phải mở rộng trực tiếp từ HGCLAMIR. Hai mô hình **phát triển độc lập**, kiến trúc + mục tiêu dự đoán khác nhau. Tên class `HeterogenousGraphCLAMIR` KHÔNG hàm ý fork. → khép nghi vấn này; định vị novelty coi HGCLAMIR là *related art cùng họ*, không phải tiền thân trực tiếp.

---

## B. PHÁT HIỆN CHÉO (ammunition cho angle A — reproducibility/critique)

**B1. Số baseline KHÔNG nhất quán giữa các paper (data split / re-impl khác nhau).**
- DHGCMDA (Table 3) báo **TDRC**: v2.0 F1=0.4801, v3.2 F1=0.4207.
- Nhưng **TDRC paper gốc** tự báo: v2.0 F1=0.5286, v3.2 F1=0.5365.
- → DHGCMDA re-run TDRC ra **thấp hơn hẳn** (−0.05 đến −0.12). Tương tự NMCMDA: DHGCMDA báo v2.0 F1=0.5716 vs NMCMDA MCD-6 tự báo 0.4617 (khác dataset).
- **Ý nghĩa:** số baseline phụ thuộc mạnh vào split/preprocessing → củng cố critique "thiếu chuẩn hoá benchmark MDA". (Ta reproduce TDRC ~98% khớp *paper gốc*, không khớp bảng DHGCMDA.)

**B2. Kích thước HMDD "cùng version" khác nhau ở MỌI paper** (mỗi nhóm tự curate):
| Paper | HMDD v2.0 | HMDD v3.2 |
|---|---|---|
| TDRC | 324×169, 4 types | 713×447, 5 types |
| NMCMDA (MCD) | — | 1208×894, 6/20 types |
| MHCLMDA | — | 757×435 (binary) |
| HGCLAMIR | 495×380 | 917×486 (binary) |
| DHGCMDA (paper) | (495×383 công bố / 411×271 curated ẩn) | 411×271 curated ẩn |
- → **Không có "HMDD v2.0/v3.2 chuẩn"**. Trực tiếp chống lưng phát hiện của ta: "gap v3.2 do data curation chưa public 411×271". Đây là *systemic* trong lĩnh vực, không riêng DHGCMDA.

**B3. DHGCMDA thừa nhận class-weighting đánh đổi precision lấy recall** (quote p27) → khớp phát hiện loss `0.3·L_existence` + class weights của ta (Plan C/F). Không phải ta hiểu nhầm — paper cố ý.

**B4. Loại bài toán phân nhánh rõ:** MHCLMDA/HGCLAMIR = **binary**; TDRC/NMCMDA/DHGCMDA/MHNNMDA = **type/multi-category**. Novelty của ta (type prediction) cạnh tranh trực tiếp nhóm sau (4 bài), trong đó 2 bài (DHGCMDA, MHNNMDA) **cùng lab**.

---

## C. ADVERSARIAL PASS (R6/R9/R15)

| Bài | Precision (tồn tại?) | Relevance | Temporality | Retraction/predatory | Verdict |
|---|---|---|---|---|---|
| DHGCMDA | ✅ DOI+PDF+search khớp | ✅ lõi | 2026, current | Venue BMC Bioinf (legit, Springer Nature), OA hợp lệ; "Article in Press" (chưa final-edited) | GIỮ |
| NMCMDA | ✅ OUP+PubMed+DBLP | ✅ baseline trực tiếp | 2021 (còn cite) | Briefings in Bioinf (legit) | GIỮ |
| TDRC | ✅ OUP+PubMed+arXiv | ✅ baseline trực tiếp | 2020/21 | Briefings in Bioinf (legit) | GIỮ |
| MHNNMDA | ✅ Springer landing khớp title+authors+DOI | ✅ competitor gần nhất | 2026 | J Comput Aided Mol Des (legit Springer) | GIỮ (metadata); số liệu = UNVERIFIED |
| MHCLMDA | ✅ PMC+OUP | ⚠ binary, không type | 2024 | Briefings in Bioinf | GIỮ (bối cảnh) |
| HGCLAMIR | ✅ PLOS+PMC | ⚠ binary; nghi là "cha" của DHGCMDA | 2024 | PLOS Comp Biol (legit) | GIỮ + đào tiếp |

**Cảnh báo temporality:** DHGCMDA là "Article in Press / unedited" → số/bảng CÓ THỂ đổi ở final. Đã lưu bản hiện có; cần re-verify khi final publish.
**Không có bài nào bị loại vì bịa** ở session này (self-audit R17 bên dưới).

---

## D. SELF-AUDIT (R17) — SESSION 2
- Search: 5 · Fetch thử: 8 (4 thành công trực tiếp: NMCMDA, TDRC, MHCLMDA, HGCLAMIR; 1 thành công qua redirect: MHNNMDA metadata; DHGCMDA lấy từ PDF cục bộ; 2 lần Springer cookie-loop).
- **VERIFIED đầy đủ số liệu: 5** (DHGCMDA, NMCMDA, TDRC, MHCLMDA, HGCLAMIR).
- **VERIFIED metadata / số liệu UNVERIFIED: 1** (MHNNMDA).
- **LOẠI vì không tồn tại: 0.**

## E. SEARCH-HIT-ONLY (để Session 3 fetch — CHƯA được dùng)
Xuất hiện trong search nhưng chưa fetch → chưa lên ledger chính:
- **DGNMDA** (Dual Heterogeneous Graph NN encoder, PMC11591469) — dual heterogeneous graph, gần kiến trúc DHGCMDA.
- **CLHGNNMDA** (Zhu, Wang, Dai 2025, J Comput Biol, 10.1089/cmb.2024.0720) — contrastive + hypergraph.
- **AMFCL** (Interdiscip Sci 2025, 10.1007/s12539-025-00724-4) — adaptive multi-source fusion + contrastive.
- **HGSMDA** (HyperGCN + Sørensen-Dice, PMC10893088).
- Từ bảng DHGCMDA (chưa fetch bài gốc): **MRFGMDA, SPLDHyperAWNTF, TFLP, KBLTDARD** — 4 baseline type-prediction.

## F. Provenance & Limitations
- MHNNMDA số liệu chưa lấy được (paywall) → **không** dùng để so sánh định lượng cho tới khi fetch được (thử sci-hub/preprint/researchgate ở Session 3, hoặc chấp nhận metadata-only).
- DHGCMDA "unedited in press" → số có thể thay đổi.
- ~~Nghi vấn DHGCMDA↔HGCLAMIR~~ → **ĐÃ GIẢI (user, 2026-07-05): độc lập, không phải fork.**
- Câu hỏi mở: dim "công bố" của DHGCMDA v2.0 (495×383 theo README/code ta) vs "curated 411×271" — cần tìm trong PDF phần Data (chưa đọc p đó session này).
