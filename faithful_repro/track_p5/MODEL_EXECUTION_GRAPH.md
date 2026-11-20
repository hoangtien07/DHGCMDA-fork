# MODEL_EXECUTION_GRAPH — P5 interpretation (paper Eq.1–34 → faithful.py)

Static graph. Numbers are node ids; `→` shows dataflow; sources tiered
per claim label. Shapes: nm=411, nd=271, C=5, d=256 (hidden), dp=64.

## Inputs

```
Y (nm,nd,C)                [MEASURED tensor, sha 85bb…]
sim_m1, sim_m2 (nm,nm)     [AUTHOR_RELEASE bundle; provenance UNKNOWN]
sim_d1, sim_d2 (nd,nd)     [AUTHOR_RELEASE bundle; provenance UNKNOWN]
fold manifests             [INDEPENDENT locked pair folds seed1234]
```

## Stage S0 — train masking (every fold)

```
Y ──► Ytr: only fold.train pairs keep labels        faithful.py L327-332
Ytr ──► A = (Ytr.sum(2)>0)                          L332
if diag_leak: Af = full-A else A                    L333 (diagnostic flag)
```

## Stage S1 — per-view features (dual view)

```
sim_m1 ──► X_m1 = [sim_m1 ; A]       (nm × (nm+nd))  L335-338
sim_m2 ──► X_m2 = [sim_m2 ; A]
sim_d1 ──► X_d1 = [sim_d1 ; Aᵀ]      (nd × (nd+nm))
sim_d2 ──► X_d2 = [sim_d2 ; Aᵀ]
```

## Stage S2 — hypergraph operators (Eq.1–4)

```
X_v ──► cosine KNN incidence H_v (K=13, self-first, deterministic)
     ──► G_v = Dv^-1/2 H De^-1 Hᵀ Dv^-1/2   (Eq.4)   knn_G L81-93
  (ablation no_hgcn: gcn_G pairwise graph instead)   gcn_G L96-106
```

## Stage S3 — intra-view encoders + CL (Eq.5–9)

```
per modality: CLHGCN (h1 on view1, h2 on view2)
  z_v = LeakyReLU( G_v @ (X_v W + b) )               HGNNConv L109-115
  h_v = g(z_v) = fc2(ELU(fc1(z_v)))                  projection L125-126
  l_intra = 0.5·l_cl(h1,h2) + 0.5·l_cl(h2,h1)        GRACE L128-140
```

## Stage S4 — inter-view contrastive (Eq.10–15)

```
S = cos(zm, zd)/t  (t=0.5)
l_info = mean over positives of -log(E/(E+Σ_neg E))   inter_loss L143-153
margin term = 0.1·relu(0.5 - (mean P - mean N))  [negs ≤10× subsampled]
```

## Stage S5 — AVF fusion (Eq.16–18)

```
α_v = sigmoid(W2·ReLU(W1·GAP(z_v)))   channel gate, r=4   AVF L156-164
Z~_v = ReLU(α_v ⊙ z_v);  Z = Σ_v softmax(β)_v Z~_v
→ zm (nm,d), zd (nd,d)
```

## Stage S6 — HGT dynamic graph (Eq.22–29)

```
ei = {miRNA-associates-disease (A edges),
      disease-associates-miRNA,
      miRNA-similar-miRNA  (S_m edges, θ=0.5, i≠j),
      disease-similar-disease}
2× HGTBlock: x' = LN(x + Dropout(HGTConv(x, ei))), heads=4   HGTBlock L205-211
```

## Stage S7 — refresh (Sec 2.8, every 5 epochs)

```
S_m ← mm01(recon_m from stage S8);  S_d ← mm01(recon_d)
ei rebuilt;  H_m = KNN([S_m ; A]);  H_d = KNN([S_d ; Aᵀ])
G = {m1: H_m, m2: H_m, d1: H_d, d2: H_d}    L406-412
  └─ T1: shared post-refresh operator (reading A, primary)
     reading B (view-specific) documented, NOT executed
```

## Stage S8 — reconstruction (Eq.19–21)

```
Zp = W2·ReLU(W1·z)  (d → d/2);  S = Zp·Zpᵀ            Recon L167-173
l_rec = ||sim_m2 - S_m||²_F + ||sim_d2 - S_d||²_F (sum)  L403
```

## Stage S9 — predictor (Eq.30–34)

```
m = LN(Linear(zm)), d' = LN(Linear(zd))
existence: ex = sigmoid(m·diag(re)·d'ᵀ)
types:     logits_ijk = m_i·rt_k·d'_j;  p = softmax(logits/T), T∈[0.5,5]
l_type  = -mean_i Σ_k w_k y_ik log p_ik      (EN weights, β=0.99999)
l_exist = focal γ=2 on sampled unobserved     L398-401
total   = l_type + λ1·l_intra + λ2·l_inter + λ3·l_rec + 0.3·l_ex   L404
```

## Evaluation path

```
inner-val only → τ_k thresholds (multi-hot) / best-F1 τ (triplet)
primary metric: MULTIHOT_MACRO_F1
auxiliary: TOP1_ANYHIT / TOP1_FIRSTPOS harmonics; triplet AUC/AP/AUPR-trapz
```
