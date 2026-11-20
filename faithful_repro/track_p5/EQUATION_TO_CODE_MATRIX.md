# EQUATION_TO_CODE_MATRIX — paper Eq.1–34 vs code

Sources: paper PDF (sha 24b989d1…), `PAPER_SPEC.md` (sha 43c18d91…),
`code/faithful.py` (sha efdaa691…), author release main@06f9ba3 (pinned).
Equation numbering follows PAPER_SPEC groupings; where the paper's own
numbering is ambiguous between §2.x paragraphs, the grouping is marked
`[INDEPENDENT_ASSUMPTION]`. Test IDs refer to `track_p5/tests/`.

Legend: src = `PAPER_EXPLICIT` / `RELEASE_DERIVED` / `PREDECESSOR_DERIVED` /
`INDEPENDENT_ASSUMPTION`.

| Eq | Meaning (paper) | Shapes | Units/normalization | Code: file:line | Release source | Unresolved assumption | Test |
|---|---|---|---|---|---|---|---|
| 1 | miRNA feature X_m^(v) = [S_m^(v), A] | nm×(nm+nd) | sim rows + association rows | faithful.py:337 (m1,m2) | release feeds view-1 to both branches | whether release's shared-feature bug is intended [INDEPENDENT_ASSUMPTION: per-branch features per paper] | test_eq_hypergraph::test_feature_stacking_dims |
| 2 | disease feature X_d^(v) = [S_d^(v), Aᵀ] | nd×(nd+nm) | sim + Aᵀ | faithful.py:338 (d1,d2) | same as Eq.1 | same | test_eq_hypergraph::test_feature_stacking_dims |
| 3 | KNN hyperedge e_i = {v_i} ∪ KNN(v_i,K), K=13, cosine | H∈{0,1}^{n×n} | cosine similarity, self always included | knn_G L81-90 | release uses Euclidean | tie-break policy unspecified [INDEPENDENT_ASSUMPTION: argsort deterministic] | test_eq_hypergraph::test_knn_incidence_includes_self_and_cosine, ::test_knn_tie_deterministic |
| 4 | G = Dv^-1/2 H W De^-1 Hᵀ Dv^-1/2, W=I | G∈R^{n×n} | symmetric normalized | knn_G L91-93 | released HGNN_conv formula | none | test_eq_hypergraph::test_G_symmetric_finite, ::test_G_matches_oracle |
| 5 | X^(l+1)=σ(G X^(l) Θ^(l)) per view branch | d→d | one HGCN layer per branch + LeakyReLU | HGNNConv L109-115; CLHGCN.forward L134-140 | released CL_HGCN same | activation choice [RELEASE_DERIVED: LeakyReLU 0.25] | test_eq_hypergraph::test_hgcn_layer_math |
| 6 | intra pos pair (u_i,v_i) | — | cosine of g(u) | l_cl L128-132 | released GRACE | — | test_cl_losses::test_intra_positive_pair |
| 7 | numerator exp(sim(g(u_i),g(v_i))/t) | — | t=0.5 | l_cl L129,131 | released | t from hyperparam table [PAPER_EXPLICIT range {0.5,1,3}, chosen 0.5] | test_cl_losses::test_infonce_oracle |
| 8 | denominator = intra-view + inter-view negatives | — | refl.sum + betw.sum − refl.diag | l_cl L132 | released identical | denominator composition matches released GRACE [RELEASE_DERIVED] | test_cl_losses::test_infonce_denominator_composition |
| 9 | L_intra = α·l(u,v) + (1−α)·l(v,u), α=0.5 | scalar | symmetric combination | forward L140 | released | α unspecified → 0.5 [RELEASE_DERIVED] | test_cl_losses::test_intra_symmetric_combination |
| 10 | cross-view similarity S_cross = norm(Z_M)·norm(Z_D)ᵀ / t | nm×nd | cosine/t | inter_loss L145 | released | — | test_cl_losses::test_cross_similarity_scale |
| 11 | P_i = {j | A_ij = 1} positive set | per-miRNA | train A only | inter_loss L146 (pos=A>0 on At=train) | released used full A [DEVIATION: train-only] | leakage fix is deliberate deviation, documented | test_cl_losses::test_inter_positives_train_mask |
| 12 | InfoNCE = −log(E_p/(E_p+Σ_neg)) | scalar | exp(clamped S) | inter_loss L147-150 | released vectorized | clamp ±10 [RELEASE_DERIVED] | test_cl_losses::test_inter_infonce_oracle |
| 13 | N_i sampled negatives | — | subsample ≤10×|P_i| per step | inter_loss L151-152 | released | RNG/subsample factor unspecified [INDEPENDENT_ASSUMPTION: 10×, per-step randperm] | test_cl_losses::test_negative_subsample_semantics |
| 14 | margin loss max(0, m − (mean S_P − mean S_N)) | scalar | m=0.5 | inter_loss L153 | released margin 0.5 | margin value unspecified [RELEASE_DERIVED] | test_cl_losses::test_margin_term |
| 15 | L_inter = InfoNCE + λ_margin·Margin | scalar | margin_w=0.1 | inter_loss L153 | released 0.1 | weight unspecified [RELEASE_DERIVED] | test_cl_losses::test_inter_total_composition |
| 16 | α_v = sigmoid(W₂·ReLU(W₁·GAP(Z_v))) | d→d/r→d | per-channel gate | AVF L159-164 | release fixed 0.6/0.4 mix [DEVIATION] | channel-wise gate vs scalar [INDEPENDENT_ASSUMPTION: channel-wise]; reduction r=4 [INDEPENDENT_ASSUMPTION] | test_fusion_recon::test_avf_gate_shapes, ::test_avf_view_specific_output |
| 17 | Z̃_v = ReLU(α_v · Z_v) | n×d | gated, nonneg | AVF L164 | — | — | test_fusion_recon::test_avf_output_nonneg |
| 18 | Z_fused = Σ_v β_v Z̃_v | n×d | softmax(β), learnable | AVF L163-164 | release fixed weights [DEVIATION] | softmax normalization of β [INDEPENDENT_ASSUMPTION] | test_fusion_recon::test_avf_beta_learnable |
| 19 | Z_proj = W₂·ReLU(W₁·Z_fused) | d→d/2 | projection | Recon L169-172 | released cosine decoder [DEVIATION] | PAPER_SPEC lists d→d/2→d for S^: faithful uses output width d/2 for Z_proj then Gram gives n×n — width is a CHOICE [INDEPENDENT_ASSUMPTION, documented] | test_fusion_recon::test_recon_projection_width |
| 20 | Ŝ = Z_proj·Z_projᵀ | n×n | Gram, unnormalized | Recon L173 | released: cosine decoder | raw Gram (no sigmoid/normalize) [RELEASE_DERIVED via faithful deviations] | test_fusion_recon::test_gram_semantics_symmetric |
| 21 | L_recon = ‖S_M−Ŝ_M‖²_F + ‖S_D−Ŝ_D‖²_F | scalar | Frobenius SUM | run_fold L403 (recon_reduction='sum') | released 0.15·MSE [DEVIATION] | sum vs mean — sum is paper-literal; mean kept as sensitivity flag ONLY [PAPER_EXPLICIT sum] | test_fusion_recon::test_recon_loss_sum_vs_mean_scale |
| 22 | heterogeneous graph construction: m–m, d–d, m–d, d–m edges | 4 edge types | m-m/d-d weighted by Ŝ, m-d by A | edges() L249-253, META L214-215 | released create_hetero_data | edge directionality [RELEASE_DERIVED: md flipped for d-m] | test_dynamic_refresh::test_edge_types_and_directions |
| 23 | m–m edge threshold Ŝ_ij > θ, i≠j | — | θ=0.5 after min-max | edges L251, mm01 L256-257 | released | θ unspecified → release default 0.5 [RELEASE_DERIVED] | test_dynamic_refresh::test_threshold_and_self_exclusion |
| 24 | HGT type-specific projections | per relation | HGTConv internals | HGTBlock L206-210 (torch_geometric HGTConv) | released HGT | library semantics [INDEPENDENT_ASSUMPTION: PyG HGTConv = paper's HGT] | test_dynamic_refresh::test_hgt_shapes_and_gradients |
| 25 | multi-head attention aggregation | 4 heads | concat/mean per PyG | heads=4 (a.heads) | released default 8 [DEVIATION to paper's 4] | head-merge scheme [RELEASE_DERIVED] | test_dynamic_refresh::test_hgt_heads |
| 26 | HGT layer 1 | — | — | hgt ModuleList L227 | — | — | covered by 24 |
| 27 | HGT layer 2 | — | 2 layers | nlayer=2 | released [PAPER_EXPLICIT 2 layers] | — | test_dynamic_refresh::test_hgt_two_layers |
| 28 | residual + dropout + LayerNorm per layer | — | x = LN(x + Drop(HGT(x))) | HGTBlock.forward L210-211 | released | order of LN/dropout [RELEASE_DERIVED] | test_dynamic_refresh::test_hgt_residual_structure |
| 29 | HGT output embeddings per node type | — | — | forward L242-244 | — | — | test_dynamic_refresh::test_hgt_shapes_and_gradients |
| 30 | predictor scores s_ij^c = m_iᵀ·W_c·d_j (bilinear) + existence | (nm,nd,C+1) | diag or full W per type | Predictor L176-202 | released SimplifiedTypePredictor (diag) | diag vs full bilinear [RELEASE_DERIVED diag default; J-1 measured full > diag on v2.0 — declared choice in register] | test_type_predictor::test_diag_bilinear_math, ::test_full_bilinear_math |
| 31 | type probabilities = softmax over 5 types at temperature T | (nm,nd,5) | softmax(logits/T), T learnable clamp [0.5,5] | Predictor L201-202 | released identical | temperature policy [RELEASE_DERIVED]; **softmax over multi-hot rows is NOT independent-Bernoulli multi-label** — locked as `TYPE_SOFTMAX_MULTIHOT_WEIGHTED_CE` [INDEPENDENT_ASSUMPTION vs BCE branch] | test_type_predictor::test_softmax_temperature, ::test_type_prob_sums_one |
| 32 | L = L_type + λ₁L_intra + λ₂L_inter + λ₃L_recon | scalar | λ1=1, λ2=0.3, λ3=1 | run_fold L404 | released plus focal existence (paper has NO existence term) [ambiguity: existence head exists but loss undefined in paper — L399 exist_w=0.3 is RELEASE_DERIVED] | existence-loss presence and weight [RELEASE_DERIVED, flagged] | test_type_predictor::test_total_loss_composition |
| 33 | L_type = −Σ_(i,j)∈F Σ_k w_k·y_ij^k·log p_ij^k | scalar | mean over positive pairs | run_fold L401 | released scalar-label CE (cannot represent multi-hot) [DEVIATION to multi-hot version] | multi-hot generalization: sum over k per row, mean over pairs — denominator = pair count not label count [INDEPENDENT_ASSUMPTION, fixture-tested] | test_type_predictor::test_weighted_multihot_ce_oracle, ::test_multihot_vs_bce_distinct |
| 34 | w_k = (1−β)/(1−β^{n_k}), β→1, train counts only | length-C | normalized to sum C | run_fold L357 | released identical formula | β=0.99999 [RELEASE_DERIVED]; hyperparam table conflict (inverse-proportional) noted | test_type_predictor::test_en_weights_train_only |

## Cross-cutting rows

| Item | Finding | Test |
|---|---|---|
| Dynamic update §2.8 (after Eq.34) | every 5 epochs: Ŝ min-max→[0,1], edges >0.5, hypergraphs rebuilt by KNN on [Ŝ,A]. **T1 finding**: faithful.py L410-412 assigns the SAME rebuilt `Hm` to `m1` and `m2` (and `Hd` to `d1`,`d2`) — post-refresh the two views share one operator. Two readings: (a) §2.8 rebuilds one reconstructed similarity per modality → shared G is consistent; (b) views should keep distinct operators → potential information loss. NO conclusion before G4 policy; both retained as documented designs. | test_dynamic_refresh::test_post_refresh_view_sharing, ::test_refresh_frequency |
| top1 evaluator | `top1` (L260-270) = any-hit precision + mean per-type recall harmonic — an INDEPENDENT convention `TOP1_ANYHIT_HARMONIC_K5`, not author-equivalent | test_metrics_contract::* |
| CV_triplet split | make_splits 'triplet': same pair may sit in train/test under different types — allowed for CV_TRIPLET, forbidden under CV_PAIR; reported transductive | test_split_and_provenance::test_triplet_overlap_allowed_pair_disjoint_forbidden |
| Existence negatives | neg_pool excludes test/val pairs by pair coordinates (L347-356) | test_split_and_provenance::test_negative_pool_excludes_heldout |
| Bundled sim views | run_fold L335-336 uses D['sim_*'] directly (UNKNOWN_PROVENANCE) — diag_leak=0 masks A only; does NOT certify views | G2 verdict + test_split_and_provenance::test_views_not_certified_clean |
