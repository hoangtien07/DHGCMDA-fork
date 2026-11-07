# DHGCMDA paper spec checklist (from PDF; for faithful re-implementation)
User directive (2026-11-06): "Cho phép sửa code, dùng đúng logic và thuật toán của paper, cấm sửa code theo hướng làm đẹp kết quả. Data v3.2 thì tải về đúng bản tại http://www.cuilab.cn/hmdd và xử lý."
Rule: paper-specified -> implement paper; unspecified -> keep author code, label [paper không ghi].

## Data (3.1)
- Same preprocessing for v2.0 & v3.2: drop miRNAs/diseases with <2 verified associations.
- Table1: v2.0 495m x 383d, 1,679 assoc, 4 types; v3.2 411m x 271d, 11,748 assoc, 5 types.
- Table2 v2.0: Genetics 681, Epigenetics 199, Circulation 443, Target 356 (sum 1679)
  -> shipped code data counts Circ367/Epi157/Target293/Gen681 = 1498 (MISMATCH, check).
- Table2 v3.2: Gen 1155, Epi 403, Circ 2293, Target 3997, Tissue 3900.
- Views: miRNA S_seq (sequence) + S_func (functional); disease S_gene (gene-based) + S_sem (semantic).
## Features (Eq1-2): X_m^(v) = [S_m^(v), A] (nm x (nm+nd)); X_d^(v) = [S_d^(v), A^T]
## Hypergraph (2.3): per each of 4 views, KNN by COSINE similarity in augmented feature space, e_i = {v_i} ∪ KNN(v_i,K), K=13; W=I;
  G = Dv^-1/2 H W De^-1 H^T Dv^-1/2 (Eq4)
## HGCN (Eq5): X^(l+1)=σ(G X^(l) Θ^(l)); two parallel branches (view1, view2) -> Z1, Z2; projection head g(.)
## Intra CL (Eq6-9): GRACE-style: pos (u_i,v_i); neg: intra-view u_k + inter-view v_k; cosine of g(u), temp t=0.5; L_intra = α Σ l(u,v) + (1-α) Σ l(v,u), α=0.5
## Inter CL (Eq10-15): S_cross = norm(Z_M) norm(Z_D)^T / t ; InfoNCE over P_i={j|A_ij=1}, N_i sampled negatives; margin loss max(0, m-(mean S_P - mean S_N)); L_inter = InfoNCE + λ·Margin (λ, m unspecified -> code 0.5 margin)
## AVF (Eq16-18): α_v = sigmoid(W2 ReLU(W1 GAP(Z_v))); Z~_v = ReLU(α_v Z_v); Z_fused = Σ β_v Z~_v (learnable β)
## Recon (Eq19-21): Z_proj = W2 ReLU(W1 Z_fused) (d->d/2->d); S^ = Z_proj Z_proj^T; L_recon = ||S_M-S^_M||_F^2 + ||S_D-S^_D||_F^2
## HGT (Eq22-29): hetero graph, 4 edge types (m-m weighted S^_M, d-d weighted S^_D, m-d, d-m); type-specific linear proj; 2 layers, 4 heads; residual+dropout+LayerNorm
## Predictor: bilinear; existence + type.
## Dynamic update (2.8): every 5 epochs, regenerate m-m/d-d edges: S^_ij > θ=0.5, i≠j, S^ normalized to [0,1]; association edges unchanged; hypergraphs H rebuilt with KNN from updated S^.
## Loss (Eq32-34): L = L_type + λ1 L_intra + λ2 L_inter + λ3 L_recon; λ1=λ3=1.0, λ2=0.3, t=0.5
  L_type = -Σ_{(i,j)∈F} Σ_k w_k y_ij^k log p_ij^k  (F = all miRNA-disease association pairs) ; w_k = (1-β)/(1-β^{n_k}), β close to 1
  -> paper has NO explicit existence loss term in Eq32 (but predictor predicts existence; CV_triplet needs existence score) [ambiguity]
- Optimizer Adam; LR scheduling + early stopping on validation set (unspecified details).
## Hyperparams (3.3): t∈{0.5,1,3}, λ2∈{0.1,0.3,0.5}, best t=0.5, λ2=0.3; class weight inverse-proportional best (Fig2b) [conflicts with Eq34 effective-number]; K=13.
## Eval (3.2): 5-fold.
- CV_triplet: known (m,d,type) triplets split 5; test = fold + equal number random unknown triplets; train = other 4 folds + other unknown; AUPR/AUC/F1.
- CV_type: pairs with >=1 type split 5; train 4/test 1; per test pair, argmax over types; Top-1 P/R/F1.
## Ablations (3.5): w/o CL (remove intra+inter); w/o HGCN (replace with standard GCN); w/o AVF (concatenation); w/o HGT (fused emb direct to predictor); w/o DV (single-view hypergraph).
## Reported: v2.0 CV_type P/R/F1 .5842/.6341/.5970; CV_triplet AUPR/AUC/F1 .9738/.9669/.9278; v3.2 CV_type .7915/.9421/.8600; CV_triplet .9271/.9181/.8674
