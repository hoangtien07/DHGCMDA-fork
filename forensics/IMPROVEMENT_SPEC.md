# SPEC v2 + PLAN — Cải thiện DHGCMDA trên artifact thật MDAv3.2-3 (post-review)

## 0. Trạng thái hiện tại (VERIFIED)

| Thực thể | Trạng thái |
|---|---|
| Dataset đúng paper | `HMDD_data/MDAv3.2-3/` — 411×271×11,748, 5-type multilabel, 8,735 pairs, 2,110 multi-type (24%) |
| Eval protocol | SPLD pair-fold Top-1: mask toàn bộ 5 channels của test pair; P=TP/n_pairs, macro-R=mean(TP/pos), ceiling macro-R=0.8642 |
| Honest baselines | SPLD code gốc: F1≈0.56 (P=0.622/macro-R=0.507) — **thanh đối chứng**; DHGCMDA honest: 0.31; leaky: 0.19 |
| **M0 provenance (F24)** | `mi_fun_sim_3.2_3.csv` = **MISIM 2.0 exact (diff~1e-10)** → label-derived (HMDD associations) → **LEAKY nếu static**; `DSSM3.2_3.csv` tương quan 0.90 TDRC Dis_sim (MeSH/DO ontology) → **STATIC_EXTERNAL có điều kiện** |

## 1. Similarity policy (M0 quyết định)

- `DSSM` → dùng static trong primary track (ontology-derived, không phái sinh label). Flag: verify công thức Wang/DO nếu cần cho write-up.
- `mi_fun_sim` = MISIM → **KHÔNG dùng trong primary honest track.** Options: (a) recompute fold-wise nếu reconstruct được công thức MISIM (Wang 2010: miRNA functional sim từ disease sets qua DAG semantic weights); (b) giữ 1 "legacy track" riêng ghi rõ leaky. Primary = DSSM + train-GIP + typed graph.
- `GIP` → luôn TRAIN_DERIVED, recompute per outer fold (đã làm trong R2).
- Co-occurrence prior → TRAIN_DERIVED, tính từ outer-train only.

## 2. Eval protocol (CHỐT)

- **Outer 5-fold pair-CV** (KFold seed cố định trên 8,735 pairs; mask toàn bộ Y[m,d,:] test pair).
- **Nested inner validation:** trong mỗi outer-train, một deterministic inner split (train/val) → tune λ/K/epochs/early-stop. Outer test chỉ nhìn 1 lần sau khi lock config.
- **Dual endpoints (đồng cấp):**
  - *Legacy (compatibility):* `legacy_top1_precision` (TP/n_pairs), `legacy_micro_recall` (TP/n_triplets), `legacy_macro_recall` (mean(TP_i/pos_i)), `legacy_f1`. Đặt tên rõ là SPLD-style evaluator, không gọi chung là P/R/F1.
  - *Scientific:* macro/micro-AUPR, macro/micro-F1 (multilabel, k types/pair), Hit@1, Recall@{1,3}.
- **Golden evaluator gate (M1):** port SPLD metric → `forensics/eval_top1.py`; reproduce SPLD **fold-by-fold** (fold membership, n_test, TP/fold, P/micro-R/macro-R/F1 per fold + aggregate) → `forensics/golden_spld_eval.json`. Mọi runner reuse evaluator này.

## 3. Kiến trúc TDHGCMDA-v2

```
STATIC: DSSM ──→ hypergraph/dense view
TRAIN:  A_train ──→ GIP ──→ view 2
                   └─→ typed edges (5 relations) ──→ R-GCN (basis-decomp) [HGT only if R-GCN insufficient]
embeddings ──→ pair concat ──→ Full bilinear 5-type decoder ──→ base logits z
z' = z + λ·J·sigmoid(z)   (J = smoothed PMI/co-occurrence 5×5, diag=0, train-only)
loss = BCE(z', Y_train, pos_weight)
```
- **Không existence head** trong v1 (tránh negative-sampling ambiguity; Task B sau).
- Co-occurrence dùng normalized J (Laplace shrinkage cho epigenetics 403 triplets), KHÔNG raw counts.
- R-GCN basis-decomp trước; HGT chỉ nếu cần (411×271 nhỏ, epigenetics rất ít).

## 4. Baselines (M2 tournament trước encoder)

| Baseline | Mục đích |
|---|---|
| Co-occurrence-only (J prior) | đo label-dependency thuần |
| Full bilinear (embeddings học thuần, không GNN) | decoder tự giải quyết? |
| DistMult/RESCAL typed factorization | relational baseline mạnh đơn giản |
| Similarity + MLP (DSSM/GIP features) | graph encoder có cần? |
| SPLDHyperAWNTF (golden) | historical tensor baseline F1≈0.56 |
| DHGCMDA-honest | corrected released |

## 5. Plan M0→M6

- **M0 ✅** Provenance gate (xong — F24).
- **M1** Golden evaluator: port SPLD Top-1 metric, reproduce fold-by-fold vào `golden_spld_eval.json`.
- **M2** Simple ceiling tournament: bilinear, +J, DistMult/RESCAL, sim+MLP, co-occur-only — cùng golden eval + inner-val tuning. Nếu simple model đã vượt 0.56 → kết quả khoa học lớn, không cần encoder nặng.
- **M3** TDHGCMDA encoder: dual-view (DSSM static + train-GIP) + typed R-GCN → typed HGT nếu R-GCN không đủ.
- **M4** Label-interaction: fixed J → learned interaction, ablation raw vs normalized.
- **M5** Locked outer eval: 3 seeds {0,1,2}, legacy + multilabel metrics, bootstrap CI trên shared test pairs. Claim = Δ + 95% CI vượt SPLD, KHÔNG phải ngưỡng 0.60.
- **M6** Write-up: `forensics/IMPROVEMENT_RESULTS.md` + ledger + PR.

## 6. Guardrails

- Leakage audit trong mỗi runner: in ra sources (STATIC_EXTERNAL/TRAIN_DERIVED) cho từng input.
- Không bao giờ tune trên outer-test. Early stop theo inner-val.
- Báo Δ with CI, không claim "F1≥0.60=improved".
- Ghi chú công khai: R3 trước đây dùng MISIM static → chứa partial leakage → cần rerun M2/M3 không-MISIM làm primary.
