# Search Log (R10 — Reproducible Search)

> Ghi lại **chính xác** mọi truy vấn web search/fetch theo thứ tự thời gian, để chạy lại được.
> Định dạng: `[Session] | công cụ | truy vấn nguyên văn | # hit | ghi chú`.

## SESSION 1 — 2026-07-05 (cập nhật giao thức + kế hoạch search)

| # | Công cụ | Truy vấn nguyên văn | Hit | Ghi chú |
|---|---|---|---|---|
| S1-1 | WebSearch | `LLM hallucination mitigation citation verification academic literature review 2025 2026 best practices` | 8 | Lấy kỹ thuật cho Mục 2 (span-level verification, multi-layer validation). Các nguồn CHƯA fetch → SEARCH-HIT-only, KHÔNG dùng làm citation học thuật. |
| S1-2 | WebSearch | `RAG citation grounding verification techniques reduce fabricated references 2026` | 10 | Citation-grounding 3 thành phần: precision/relevance/temporality. SEARCH-HIT-only. |
| S1-3 | WebSearch | `systematic literature search reproducible protocol OpenAlex Semantic Scholar API bibliographic best practices` | 8 | Củng cố R12: chiến lược đa nguồn (OpenAlex + PubMed + Semantic Scholar cross-validate). SEARCH-HIT-only. |

**Caveat Session 1:** Cả 3 search chỉ để **rút kỹ thuật giao thức**, không để trích dẫn nội dung khoa học. Không bài nào được fetch → không bài nào được lên ledger là VERIFIED. Một số arXiv ID trả về (vd 2604.xxxx, 2606.xxxx) NGHI VẤN chưa kiểm chứng — không dùng.

## SESSION 2 — 2026-07-05 (Cụm A: dòng dõi DHGCMDA & MDA methods)

### Web search
| # | Truy vấn nguyên văn | Hit | Ghi chú |
|---|---|---|---|
| S2-1 | `DHGCMDA dual-view heterogeneous graph contrastive learning miRNA disease association BMC Bioinformatics 2026` | 7 | Tìm ra bài gốc (DOI 10.1186/s12859-026-06436-w) + họ hàng HGCLAMIR/DGNMDA/MHCLMDA. |
| S2-2 | `NMCMDA neural multi-category miRNA disease association prediction` | 9 | OUP+GitHub+DBLP khớp. |
| S2-3 | `TDRC tensor decomposition miRNA disease association type prediction` | 9 | OUP+PMC+arXiv khớp. |
| S2-4 | `hypergraph contrastive learning miRNA disease association 2024 2025 HGCLAMIR MHCLMDA` | 8 | MHCLMDA, HGCLAMIR, AMFCL, MGHSTCKW. |
| S2-5 | `multi-type miRNA disease association type prediction hypergraph neural network MHNNMDA HMDD` | 7 | MHNNMDA (cùng lab), CLHGNNMDA, HGSMDA. |

### Web fetch
| # | URL | Kết quả |
|---|---|---|
| F2-1 | link.springer.com/article/10.1186/s12859-026-06436-w | ❌ cookie-wall loop → dùng PDF cục bộ `_pdf_text/p27,p28.txt` thay thế (primary). |
| F2-2 | academic.oup.com/bib/article/22/5/bbab074/6189772 (NMCMDA) | ✅ full metadata + số Table 2. |
| F2-3 | academic.oup.com/bib/article/22/3/bbaa140/5876601 (TDRC) | ✅ full metadata + số Table 2/3. |
| F2-4 | link.springer.com/article/10.1007/s10822-026-00821-6 (MHNNMDA) | ⚠ metadata OK, số liệu paywall. |
| F2-5 | pmc.ncbi.nlm.nih.gov/articles/PMC10796254 (MHCLMDA) | ✅ full. |
| F2-6 | journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011927 (HGCLAMIR) | ✅ full. |
| L2-1 | (local) `_pdf_text/p01,p27,p28.txt` | ✅ DHGCMDA tác giả + Table 3/4 số liệu nguyên văn. |

## SESSION 3 — 2026-07-05 (Cụm B phương pháp nền + Cụm C cross-cutting)

### Web search (8)
| # | Truy vấn nguyên văn | Hit | Ghi chú |
|---|---|---|---|
| S3-1 | `over-smoothing over-parameterization graph neural network small sparse graph low-data regime` | 10 | Oversmoothing survey 2303.10993. |
| S3-2 | `GNN ablation removing component improves performance negative result contrastive learning redundant` | 10 | GRENADE (sau LOẠI), GCFormer, MTGRR. |
| S3-3 | `reproducibility crisis machine learning bioinformatics released code evaluation metric bug` | 8 | Kapoor-Narayanan Patterns 2023. |
| S3-4 | `hypergraph construction KNN neighbor number K sensitivity link prediction hyperparameter` | 10 | Đồng thuận "K≈2–4 tối ưu link prediction". |
| S3-5 | `hypergraph neural networks HGNN hypergraph convolution Feng Gao 2019 AAAI` | 10 | HGNN 1809.09401. |
| S3-6 | `heterogeneous graph transformer HGT Hu Dong 2020 WWW web conference` | 10 | HGT 2003.01332. |
| S3-7 | `bilinear DistMult diagonal versus full tensor knowledge graph embedding expressiveness link prediction` | 10 | DistMult diagonal=symmetric; RESCAL full. |
| S3-8 | `data leakage benchmark biomedical link prediction drug target miRNA disease inflated performance negative sampling` | 8 | bioRxiv 2025.01.23.634511. |

### Web fetch
| # | URL | Kết quả |
|---|---|---|
| F3-1 | cell.com/patterns/fulltext/S2666-3899(23)00159-9 | ❌ 403 → thay bằng arXiv 2207.07048. |
| F3-2 | arxiv.org/pdf/2310.15109 (GRENADE) | ❌ PDF binary; model bịa title → **LOẠI**. |
| F3-3 | biorxiv.org/content/10.1101/2025.01.23.634511v2.full | ❌ 403. |
| F3-4 | arxiv.org/pdf/2003.01332 (HGT) | ❌ PDF binary → thay bằng abs. |
| F3-5 | arxiv.org/abs/2303.10993 (Oversmoothing) | ✅ VERIFIED. |
| F3-6 | arxiv.org/abs/2003.01332 (HGT) | ✅ VERIFIED. |
| F3-7 | arxiv.org/abs/1809.09401 (HGNN) | ✅ VERIFIED. |
| F3-8 | arxiv.org/abs/2207.07048 (Kapoor leakage) | ✅ VERIFIED. |
| F3-9 | arxiv.org/abs/1802.04868 (SimplE) | ✅ VERIFIED. |
| F3-10 | mdpi.com/2227-7390/9/8/830 (K-NN) | ❌ 403. |
| F3-11 | arxiv.org/abs/2305.12578 (Self-Explainable GNN LP) | ⚠ abs không có K-sweep → K-sensitivity giữ SEARCH-HIT. |

**Bài học:** trang PDF (`arxiv.org/pdf/…`) hay trả binary-corrupt → dùng `arxiv.org/abs/…`. cell.com/biorxiv/mdpi chặn bot.

## SESSION 4 — 2026-07-05 (Knowledge map — vá lỗ hổng)

### Web search (3)
| # | Truy vấn nguyên văn | Hit | Ghi chú |
|---|---|---|---|
| S4-1 | `TFLP miRNA disease association type prediction tensor factorization label propagation` | 9 | TFLP (github nayu0419) — SEARCH-HIT. |
| S4-2 | `KBLTDARD SPLDHyperAWNTF MRFGMDA miRNA disease association type prediction tensor` | 7 | KBLTDARD (PLOS 2024), SPLDHyperAWNTF (bbac390). |
| S4-3 | `graph neural network pruning over-parameterization small dataset removing layers improves generalization link prediction` | 10 | Untrained MP 2406.16687; Pruning-before-training 2301.00335. |

### Web fetch (2 — cả hai OK)
| # | URL | Kết quả |
|---|---|---|
| F4-1 | pmc.ncbi.nlm.nih.gov/articles/PMC11257412 (KBLTDARD) | ✅ số Top-1 v2.0/v3.2 chính xác. |
| F4-2 | arxiv.org/abs/2406.16687 (Untrained MP) | ✅ quote anchor ablation-reversal. |

## SESSION 5 — 2026-07-05 (Novelty analysis — search phủ định R7)

### Web search (8 — không fetch, session suy luận)
| # | Truy vấn nguyên văn | Hit | Phát hiện R7 |
|---|---|---|---|
| S5-1 | `reproducibility replication study miRNA disease association type prediction critique negative result` | 7 | **consistent-eval-MDA-2020** (bioRxiv) — prior art nguy hiểm nhất angle A (binary+leakage). |
| S5-2 | `benchmark inconsistency baseline reproducibility miRNA disease association tensor neural HMDD critique` | 8 | benchmark-36-methods (PMC6781296, binary). |
| S5-3 | `evaluation metric implementation bug hardcoded number classes released bioinformatics code Top-1 F1` | 10 | **MassSpecGym** — "metric divergence reorders leaderboard" (precedent metric-bug). Không thấy MDA-cụ-thể. |
| S5-4 | `ReScience replication challenge graph neural network link prediction bioinformatics negative result 2024 2025` | 9 | PROXI 2410.01802, GAE-reconsidered 2411.03845 (GNN-LP critique precedent). |
| S5-5 | `hypergraph neighbor number K over-parameterization sensitivity miRNA disease association type prediction sparse` | 6 | hypergraph-KNN MDA phổ biến (HFHLMDA/MSCHLMDA) nhưng K-sensitivity-như-đóng-góp CHƯA ai làm. |
| S5-6 | `diagonal versus full bilinear tensor predictor faithfulness reimplementation miRNA disease association` | 8 | KHÔNG thấy so sánh diag-vs-full trong MDA → khe trống. |
| S5-7 | `uncertainty quantification calibration conformal prediction miRNA disease association type` | 8 | Conformal có ở biomedical (skin lesion), KHÔNG có MDA-type → khe trống (novelty cao). |
| S5-8 | `foundation model LLM large language model miRNA disease association prediction 2025 2026` | 8 | KHÔNG có LLM+MDA → trống nhưng ngoài repo. |

## SESSION 6 — 2026-07-05 (Paper plan A+B — verify prior art + venue)

### Web search (3)
| # | Truy vấn nguyên văn | Hit | Ghi chú |
|---|---|---|---|
| S6-1 | `"consistent evaluation" miRNA disease association prediction models data leakage published version journal` | 5 | consistent-eval-2020 = bioRxiv (posted 13 Jun 2020), chưa rõ published version; binary+similarity-leakage. |
| S6-2 | `MassSpecGym evaluation pitfalls metric computation leaderboard molecule discovery arXiv` | 8 | MassSpecGym-in-the-Wild 2606.19624 (VERIFIED); base benchmark NeurIPS2024 2410.23326 (riêng). |
| S6-3 | `ReScience C submission scope ... Briefings in Bioinformatics reproducibility article type` | 9 | ReScience C: reimplement+document+partial repro OK. Brief Bioinf: khuyến khích independent eval by non-originators. |

### Web fetch (2)
| # | URL | Kết quả |
|---|---|---|
| F6-1 | arxiv.org/abs/2606.19624v1 (MassSpecGym Wild) | ✅ VERIFIED — "17 of 26 papers"; 3 failures. |
| F6-2 | biorxiv.org/content/10.1101/2020.05.04.075754v2.full | ❌ 403 (lần 4) → giữ SEARCH-HIT SH3, verify trước nộp. |

---
**TỔNG KẾT SEARCH LOG:** 6 session · ~30 web search · ~24 fetch thử (13 OK + local PDF). Nguồn chặn bot dai dẳng: cell.com, biorxiv.org, mdpi.com. Mẹo: dùng `arxiv.org/abs/`, PMC, PLOS, OUP; PDF gốc cục bộ cho DHGCMDA.

