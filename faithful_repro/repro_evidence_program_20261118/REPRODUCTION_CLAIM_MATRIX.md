# REPRODUCTION_CLAIM_MATRIX — trạng thái từng claim của paper

| # | Claim (paper) | Bằng chứng ta có | Trạng thái |
|---|---|---|---|
| P1 | v3.2 CVtype Top-1 F1 = 0.8600 (P=0.7915, R=0.9421) | Tái đo trên tensor strict: 0.4558–0.4924 (harmonic first-positive, U1 static/dynamic); tensor paper, fold, eval code chưa public | NOT_REPRODUCIBLE_AS_STATED — chưa chứng minh tương đương dataset/fold/metric |
| P2 | Pipeline tác giả chạy end-to-end | Đã chạy release thật (07–12 Nov): hỏng metric k4 trên v3.2; predictor diag; dữ liệu v3.2 họ dùng không phải bản paper | INCONSISTENT_WITH_RELEASE |
| P3 | 4 nguồn similarity (Wang + functional + gene + sequence) | Provenance verified: S_seq=NW/GFF3, S_gene=FunSim, S_sem=MeSH-2019 rebuild, S_func=train-only; cùng họ, khác khớp MeSH | PARTIAL_REPRODUCED (khác pipeline rebuild, provenance sạch) |
| P4 | Recon similarity feedback vào graph | Code release cập nhật edge_index HGT từ *_sim_recon; HGCN G-matrix prebuilt không rebuild | PARTIAL — đúng ý tưởng, khác tầng ảnh hưởng |
| P5 | CVtriplet / existence | Ta chạy nhánh riêng U1X (adaptation): Task T AUROC 0.8548 vs baseline 0.9191 | OUR_INTERPRETATION — supplementary negative |
| P6 | Numbers 0.27/0.36/0.44/0.55 lịch sử | Mỗi số khác tensor/unit/metric/exposure — không cùng một câu hỏi | NONCOMPARABLE_CROSSWALK |
