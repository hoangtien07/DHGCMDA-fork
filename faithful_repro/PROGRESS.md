# Faithful-repro progress (session c8a035d1)
Spec: PAPER_SPEC.md (same dir). Phase-1 (unmodified code) report: ~/dhg_runs/report/REPRODUCE_DHGCMDA.md
Prior forensic work (fork repo ~/repos/DHGCMDA-fork/forensics/FORENSICS_V32_DATASET.md): paper's "deg>=2" filter does NOT give 411x271 (gives ~722x612x17615);
 lineage approx = miRBase∩MISIM2.0 × MeSH-C + iterative typed deg mir>=4,dis>=7 -> 406x266x11970 (v3.2_lineage_approx/ in fork).
Plan:
 1. v3.2 data: fresh download (hmdd_v32_raw/, SHA256SUMS) -> build per paper text (deg>=2) = PRIMARY (paper-literal); report counts vs Table1.
    4 views: miRNA seq sim (miRBase mature), miRNA functional sim (MISIM2.0 / Wang), disease semantic (MeSH Wang), disease gene sim (source unspecified in paper).
 2. Faithful code: new dir in fork branch; implement Eq1-34 + CV_triplet/CV_type separately + 5 ablations per paper definitions.
 3. Runs: multi-seed, v2.0 + v3.2; report vs paper, no cherry-picking.
Env: venv ~/dhg_runs/venv_np1 (numpy 1.26); launch with PYTHONUNBUFFERED=1, OMP_NUM_THREADS=2, setsid nohup ... & disown (timeout 5000).
## Audit findings (upstream code vs paper)
- Intra CL (CL_HGCN.sim): MATCHES Eq6-9 (tau 0.5, alpha 0.5, ELU proj head). keep.
- Inter CL: InfoNCE uses ALL non-assoc as negatives (paper: sampled N_i, ratio unspecified) -> keep [paper không ghi]; margin m=0.5, λ=0.1 (paper unspecified) keep.
- AVF (HGCN_Attention_Mechanism): FIXED 0.6/0.4 weighted sum -> CONTRADICTS Eq16-18 -> FIX (SE gate + learnable β).
- Recon decoder (SimpleHypergraphDecoder): d->d->d/2 w/ dropout, L2-normalized cosine -> differs Eq19-20 (Z_proj = W2 ReLU(W1 Z); S^=Z Z^T) -> FIX.
- HGT: PyG HGTConv (has internal gated skip) + LayerNorm + dropout; m-m edges unweighted (HGTConv no edge weights) -> keep, note.
- Predictor SimplifiedTypePredictor: BilinearDiag + LayerNorm proj, exist head separate sigmoid; num_types=4 HARDCODED + type_distribution buffer hardcoded -> generalize to C types (paper v3.2 5 types).
- Eval mode: recon returns eye; forward catches all exceptions and returns constant scores (dangerous: hides errors) -> remove silent fallback.
- Dynamic update in model: threshold 0.5 on decoder output, every 5 calls to _update (check caller).
- Loss (main L842): tol = recover(focal exist + type CE weighted exist_weight/type_weight) + mi_cl + dis_cl (each = intra + 0.3*inter => inter counted 2x) + 0.15*(MSE recon) + 1e-4 L2
  paper Eq32: L_type + 1.0*L_intra + 0.3*L_inter + 1.0*L_recon(Frobenius^2). -> FIX weights; recon targets: code uses mi_fun / dis_sem only (paper: S_M, S_D unspecified which view) 
- Optimizer: AdamW lr1e-4 wd1e-5 + ReduceLROnPlateau (paper: Adam + LR schedule + early stopping on VALIDATION set) -> FIX Adam; early-stopping needs inner validation split (unspecified) [decide]
- KNN K=13 hardcoded (paper K=13) -> wire to --K_neigs default 13 (no behavior change).
- !! Model forward feeds concat_mi_tensor_view1 to BOTH HGCN branches (view2 features [S_func,A] never used as features, only via G) -> contradicts Eq1-2/2.4 -> FIX.
- !! Dynamic update in main: rebuilds hetero m-m/d-d edges EVERY epoch when MSE change>0.01 (not every 5 epochs); hypergraphs G never rebuilt; recon Ŝ is cosine (not [0,1]) -> FIX per 2.8: every 5 epochs; Ŝ minmax->[0,1]; θ=0.5; rebuild H by KNN on [Ŝ,A] (interpretation: both views use Ŝ -> same G; features keep raw views) [diễn giải].
- Type loss: label_smoothing 0.1 (not in paper) -> remove; class counts hardcoded [367,157,293,681] -> compute from training fold; β=0.99999 keep; multi-label Eq33 (y^k per type) check data encoding.
- Existence loss: NOT in Eq32 but needed for existence prediction/CV_triplet -> keep author focal exist loss w/ author weight 0.3 [paper không ghi]; L_type weight 1.0 (Eq32, author had 0.7).
- L2 reg 1e-4 term not in Eq32 -> remove (weight decay in Adam kept = author 1e-5).
- Recon: Frobenius^2 (sum) λ3=1.0 vs author 0.15*MSE(mean) -> FIX literal; targets S_M=mi_fun, S_D=dis_sem (author choice; paper unspecified which view).
- !! v2.0 csv: 1679 triplets / 1498 pairs / 324 mir x 174 dis typed; 161 pairs multi-type. Table2 counts (Circ443/Epi199/Tgt356/Gen681) = csv triplets exactly (csv code 1=Circ,2=Epi,3=Target,4=Gen).
  Upstream read_association_csv overwrites -> single type per pair (181 triplets lost) -> contradicts Eq33 multi-label y^k & CV_triplet (triplets) -> FIX multi-hot Y[nm,nd,C].
  495x383 = full HMDD v2.0 dims (binary 5430 assoc in all_mirna_disease_pairs.csv); deg>=2 filter evidently NOT applied to v2.0 (324 typed mirnas < 495). -> paper's "fewer than two" statement inconsistent w/ its own v2.0 Table1.
- Upstream CV: shuffle pairs (seed 0), split 10, DROP 10th part, 5-fold on remaining 90%; test negatives = ALL unknown pairs in fold (not equal number); one split used for both binary & Top-1 -> contradicts 3.2 CV_triplet/CV_type -> FIX: two separate protocols.
- Train negatives: sampled 10x positives per run (randperm once).
- !!! LEAKAGE (verified): train_data[4]=md_p = FULL association matrix (incl. test pairs) -> used in X=[A,S] features, KNN hypergraphs, HGT m-d edges, inter-CL positives, and train fn. Test positives visible in training. -> FIX: mask test pairs (A_train) everywhere.
- v2.0 view files: miRNA view1 "sequence" = M_GSM.txt (GIP kernel, NOT sequence; and computed from full A => leakage), view2=M_FSM; disease view1 "gene" = D_SSM2 (semantic v2, NOT gene), view2=D_SSM1. Paper: seq/func, gene/semantic.
- Eval binary: exist score (pair-level) on test pos + equal # sampled test neg (5 resamples avg); F1 at threshold maximizing F1 on TEST (optimistic, author) keep+note.
- Top-1 (author): P = micro accuracy, R = macro mean per-type recall, F1 = 2PR/(P+R); single label; hardcoded 4 types. Paper v3.2 R>P => consistent with this def (TDRC def gives R<=P). -> keep def, generalize C types; multi-label: hit if argmax ∈ true set; per-type recall over pairs containing type k.
- VERIFIED: v2.0 M_GSM.txt == GIP(full binary HMDD v2.0 495x383, 5430 assoc) to 4.6e-16; D_GSM == GIP(A^T). typed 1498 pairs ⊂ 5430. => "sequence" view is association-derived GIP using test pairs. FIX: recompute GIP per fold from A_train (typed pairs, binary) [author's similarity type kept, leak removed].
- Feature A in X: author uses type codes 1..4 as values; use binary A_train [giả định; Eq1 A ambiguous].
- Fork has ~/repos/DHGCMDA-fork/HMDD_data/MDAv3.2-3 (lab tensor?) - NOT from cuilab; user requires cuilab download -> use only as reference if at all.
- External: MISIM site http://www.lirmed.com/misim/Download ; MeSH ascii d2025.bin URL failed (HTML) -> find right URL.
- !!! Fork commit eaa2500 has EXACT paper v3.2 artifact HMDD_data/MDAv3.2-3 (SPLD lab tensor): 411x271, per-type gen1155/epi403/circ2293/tgt3997/tis3900 = 11748 = Table2 exactly; 8735 pairs; min mir deg = 1 (contradicts "<2 excluded"); only 2 sims (mi_fun 411x411, DSSM 271x271) -> 2 views, not 4.
  Plan: v3.2 PRIMARY = cuilab download + paper-stated processing (user request); REFERENCE = MDAv3.2-3 (paper's actual data).
- MISIM 2.0 downloaded: ext/misim/{miRNA_name.txt (1044), similarity.txt} from http://www.lirmed.com/misim/similarity.zip
- MeSH 2025 ascii: ext/d2025.bin from https://nlmpubs.nlm.nih.gov/projects/mesh/2025/asciimesh/d2025.bin
- DONE: code/build_datasets.py -> data/{v2.0,v3.2_ref_MDAv3.2-3,v3.2_cuilab_deg2}.npz (+json stats). v2.0 & ref built OK (v2.0 1679 trip/1498 pairs; ref 411x271x11748).
  v3.2_cuilab: 5 types from v3_{genetics,epigenetics,circulation,target,tissue}.txt; deg>=2 single-pass pair degree; disease views SS2(Xuan)/SS1(Wang) from MeSH 2025 via HMDD mesh UI; miRNA view2 MISIM2.0; view1 GIP per fold.
NEXT: write code/faithful.py (model+train+CV protocols), smoke test, launch runs.
- v3.2_cuilab literal deg>=2: 721x609x17607 (pairs 13744), mesh 496/609, misim 720/721; variants (pair/triplet, iter) 720-722 x 606-612 -> paper 411x271x11748 NOT reproducible from cuilab current files
- Author KNN = Euclidean distance (Eu_dis) vs paper cosine -> FIX cosine.
- Author initial hetero graph: m-m/d-d edges = raw mi_fun/dis_sem > 0.5 (i!=j); m-d edges from A. param defaults alpha=0.5 (neg weight in exist loss), dropout=0.3, nlayer=2.
- Writing code/faithful.py (single-file implementation). Design decisions logged in its header docstring.
- WROTE code/faithful.py (full impl; DEVIATIONS dict in header). Smoke: `faithful.py --data v2.0 --protocol type --epochs 20 --eval_every 5 --only_fold 0 --out /tmp/smoke_type.json` (running).
NEXT: smoke triplet + v3.2 datasets; then launch full runs: {v2.0, v3.2_cuilab_deg2, v3.2_ref_MDAv3.2-3} x {type, triplet} x seeds; ablations v2.0 type.
- Smoke v2.0 type OK (~1.1 s/epoch, 2 threads). Note: literal Frobenius^2 recon ~13.8k at ep1 dominates loss (paper-literal, not tuned).
- Smoke OK: v2.0 triplet; v3.2_cuilab type with all 6 ablation modes run (5 ep).
- Tests code/test_faithful.py ALL PASSED (splits disjoint, triplet negs unknown, multi-label kept, top1 == released metric on 4 types exactly (0.2768), 5-type perfect=1.0, shapes).
- LAUNCHED full queue (23 jobs, xargs -P 8, 1 thread each): jobs.txt, run_one.sh, logs/<name>.log, logs/_done.txt, results/*.json
- FINDING (first results): literal Eq.21 Frobenius^2 sum (~4000-28000) dominates Eq.32 under clip_grad_norm 1.0 (author also clips 1.0); type loss 1.27->1.19 in 300 ep; v2.0 Top-1 F1 ~0.34-0.40 (vs upstream leaky 0.557). Early stop best_epoch 10-300.
- Added --recon_reduction {sum(default, paper), mean(sensitivity only, = author MSE scale)}; jobs2.txt (6 sensitivity runs seed 1234) queued after main queue via waiter (logs/_queue2.out). Primary results remain literal 'sum'.
- Added --diag_leak (DIAGNOSTIC: full A in features/GIP/hypergraph/HGT edges, losses on train); 4 jobs appended to jobs2.txt (waiter reads file only after queue1 ends).
- 12:03 done: 9/23 queue1
- 12:22 done 18/23
- 12:42 waiter bug (pgrep matched itself) -> killed; jobs2 (10 jobs) launched directly with -P 6 alongside last 2 queue1 jobs
- report/STATIC_PART.md drafted (sources, checksums, data processing)
- 13:01 queue1 done 23/23? 24 total done
- report/METHOD_PART.md drafted
- KEY RESULTS (faithful, recon sum, 3 seeds): v2.0 type F1 0.3878±0.0151; v2.0 triplet AUPR 0.9218 AUC 0.9237 F1 0.8600; cuilab type 0.2567; cuilab triplet AUPR .8964 AUC .9025; ref type .2727; ref triplet AUPR .8701 AUC .8798.
  Sensitivity s1234: rmean v2.0 type .5412, triplet AUPR .9332; LEAK v2.0 type .4583 triplet AUPR .9433; ref rmean type .3885. Ablations v2.0 (1 seed): no_avf .4759 > full; no_dv .4202, no_cl .4141, no_hgcn .3968, no_hgt .3728.
  report/results_table.md = aggregate output. Remaining 3 jobs (cuilab type rmean, ref LEAK x2).
- report/HEAD_PART.md drafted (verdict + claims). Waiting last job cuilab type rmean.
- FINAL report report/REPRODUCE_DHGCMDA_FAITHFUL.md + ~/dhg_runs/dhgcmda_faithful.zip built

## ROUND 3 (user: "Tìm kiếm các phương pháp từ các paper liên quan để bổ sung các gap trong paper và code")
- Delivered round-2 report (REPRODUCE_DHGCMDA_FAITHFUL.md + dhgcmda_faithful.zip).
- Task: literature search per gap -> doc ~/dhg_runs/faithful/report/GAP_LITERATURE.md (gap -> related-paper method -> citation/URL -> whether adoptable a priori).
- Gaps: G1 v3.2 curation 411x271; G2 sequence/gene similarity sources; G3 inter-CL negative sampling ratio; G4 margin m / lambda; G5 existence loss; G6 AVF gate design; G7 early stopping/scheduler; G8 recon loss scale (F^2 sum vs mean); G9 CV_type multi-label metric def; G10 triplet score composition; G11 bilinear predictor form; G12 GIP vs sequence view; G13 dynamic update normalization.
- Lit refs found: SPLDHyperAWNTF doi 10.1093/bib/bbac390 (Ouyang 2022; datasets MDAv2.0, MDAv2.0-2, MDAv3.2?, Top-1 metrics); TFLP (Comput Biol Med 2022, S001048252200350X; v3.2 977x543x21370); TDRC github BioMedicalBigDataMiningLab/TDRC, arXiv 1911.05584, bib bbaa140 (CV_type/CV_triplet origin); HGCLAMIR PLOS CB 10.1371/journal.pcbi.1011927 (Ouyang 2024, HGCN+CL+view-aware attention; code name HeterogenousGraphCLAMIR origin); MHCLMDA bib bbad524; CLHGNNMDA; WeightTDAIGN Front Bioeng 2022 fbioe.2022.911769; MLHN multi-layer embedding biomedicines 9/9/1152; DHGCMDA springer s12859-026-06436-w.
- HGCLAMIR code: https://github.com/Ouyang-Dong/HGCLAMIR (DHGCMDA class HeterogenousGraphCLAMIR derived from it -> use as reference for AVF/CL/recon gaps). CLAUDE.md: TDRC data_v32 = 713x447x12,534; TDRC repro on v3.2 Top-1 F1 0.4378 (paper 0.4207); TFLP code github.com/nayu0419/TFLP.
- Next: fetch HGCLAMIR repo README/code, SPLDHyperAWNTF repo (Ouyang-Dong) dataset desc for MDAv3.2-3, TDRC paper CV defs; write GAP_LITERATURE.md.
- TDRC (commit 1de1a87, experiments.py L13-135) = origin of CV_type/CV_triplet: CV_type splits known PAIRS (any type) 5-fold seed0, test pair types zeroed in train tensor; Top-1: argmax type per test pair; P=TP/#test pairs, R=TP/sum(true types of test pairs) (multi-label); miRNA functional sim recomputed from train matrix per fold (no leakage). CV_triplet: 5-fold over positive triplets; negatives = random zero triplets, equal count, 10 negative draws averaged; F1 = max over 999 thresholds (get_metrics). No early-stopping/validation (tensor model).
- TFLP HMDD3.2_processed = 713x447 (same as TDRC v3.2), one Dis_sim. HGCLAMIR/SPLDHyperAWNTF_Model git clone blocked by proxy -> use raw.githubusercontent/api.
- CV_type recall def: TDRC R=TP/sum(true types) => R<=P always (each pair>=1 type). Paper reports R>P (v2.0 .6341>.5842; v3.2 .9421>.7915) => paper NOT using TDRC micro recall; consistent with released macro per-type recall (our top1). [inference, solid]
- HGCLAMIR repo: GitHub API says HGCLAMIR "Not Found" (deleted/renamed?); SPLDHyperAWNTF_Model API rate-limited.
- WeightTDAIGN (Front Bioeng 2022, doi 10.3389/fbioe.2022.911769, same group as SPLDHyperAWNTF?) splits HMDD v3.2/v2.0 into 4 datasets by sparsity (MDA v2.0-1..4 / v3.2-1..4?) -> likely origin of MDAv3.2-3 (411x271). Uses MeSH + miRBase sequence. Exclude 'other'. Fetch full text for curation rule.
- HGCLAMIR (PMC11037542, europepmc text at ~/dhg_runs/lit/hgclamir.txt): AVF Eq.20 alpha=FNN(GAP(Z)), 2-layer FNN ReLU then Sigmoid (=our AVF); KNN by EUCLIDEAN distance + K-means hyperedges; k searched {1..15}; CL = GRACE-style (positive = same node other view; negatives intra-view+inter-view, cosine critic on nonlinear projection g, temperature tau); lambda=gamma=1; Adam lr beta; recon T^=X Y^T; weighted loss with bias alpha (Eq.28) for imbalance; protocol 9/10 CV + 1/10 independent test.
- WeightTDAIGN (PMC9335924, text ~/dhg_runs/lit/weighttdaign.txt): datasets MDA v2.0-2 211x59 (<2 assoc removed... per type?), v2.0-3 69x25, v2.0-4 40x20, v3.2-5 125x65x4785; only keep miRNAs with miRBase sequence, diseases in MeSH; miRNA sequence sim = Biostrings pairwiseAlignment + min-max (Liang 2019); DSS=(DSS1+DSS2)/2; GIP miRNA & disease; MFS Wang 2010. NOT 411x271.
- DHGCMDA full text (europepmc PMC13169815 -> ~/dhg_runs/lit/dhgcmda.txt): data avail = cuilab.cn/hmdd only; preprocessing only "fewer than two"; optimizer Adam + "suitable lr scheduling and early stopping by validation set"; cites WeightTDAIGN[23], SPLDHyperAWNTF[24] (Ouyang). Baselines Table3: TDRC, SPLDHyperAWNTF, TFLP, NMCMDA, MRFGMDA, KBLTDARD.
- cuilab /static/hmdd3/data/ has OLD release files: old_alldata/old_circulation/old_epigenetics/old_genetics/old_target (+tissue_bak210505, alldata_bak210505), plus disease_MSW.txt, mir_DSW.txt, benchmark2019*.txt -> test old_* with deg>=2 (G1) and MSW/DSW as similarity sources (G2).
- !! DHGCMDA Table 4 TDRC v3.2 CV_triplet = 0.9059/0.8962/0.8309 = identical to TDRC's own paper numbers (TDRC dataset 713x447x12,534, per CLAUDE.md) -> baselines likely copied from original papers on DIFFERENT datasets (not 411x271). Verify CV_type rows.
- CONFIRMED: DHGCMDA Table3 TDRC v3.2 CV_type 0.4926/0.3671/0.4207 & Table4 0.9059/0.8962/0.8309 = TDRC paper numbers (TDRC dataset 713x447). Baseline rows copied from other papers/datasets. TDRC recall = micro (R<=P); DHGCMDA R>P (macro per-type) -> metric def mismatch vs baselines.
- cuilab old_* = HMDD v2.0 (old_alldata 10381 lines). Unprefixed alldata.txt 35548 lines (=v3.2 35,547 entries); compare vs v3_ files. disease_MSW/mir_DSW = spectrum width, not similarity.
- cuilab unprefixed alldata.txt (sha ef1a0c70..., 35548 lines): raw 1151x802x20041; iterative deg2 -> 783x638x19508. v3_alldata iterative deg2 -> 720x606x17603. Neither = 411x271.
- DHGCMDA text: KNN by cosine (paper) vs HGCLAMIR Euclidean (upstream code Euclidean); intra-CL = GRACE (all other nodes both views negatives); predictor bilinear for existence+type (existence task mentioned, no loss eq).
- faithful CV_triplet: equal # unknown triplets single draw; TDRC: equal # random negatives, 10 draws averaged.
- To check: KBLTDARD (PLOS CB 10.1371/journal.pcbi.1012287) & MRGBMDAT (Bioinformatics 2026 btag534) dataset dims / disease gene sim / miRNA seq sim.
- MRGBMDAT (Bioinformatics 2026 btag534, PMC13447018, SAME GROUP Sun/Shang/Liu, code github.com/CDMBlab/MRGBMDAT, text ~/dhg_runs/lit/mrgbmdat.txt): sims = miRNA functional (Xiao 2018: miRNA-gene + HumanNet), miRNA sequence (Needleman-Wunsch on miRBase), disease semantic, target-based disease sim, GIP. Data HMDD v3.2 cuilab.
- KBLTDARD (PLOS CB 2024 PMC11257412, code+data github.com/Mayingjun20179/KBLTDARD): v3.2 = 447 diseases (TDRC set); miRNA functional from miRTarBase 8.0 + HumanNet LLS min-max (Eq.1); 20 negative seeds averaged (after Huang/TDRC).
- MRGBMDAT cloned ~/dhg_runs/lit/MRGBMDAT commit 133b3e312e7b12a82509ea87393c30bf0b07ccc2, has dataset.rar (check dims, sims). Disease target-based sim = Cheng 2014 (shared genes); seq sim NW on seed regions (Rashed 2021); KBLTDARD clone blocked.
- MRGBMDAT dataset.rar (sha e796067e...) = 853 miRNA x 591 disease, 12,445 pairs, sims m_fs,m_gs,m_ss,d_gs,d_ts,d_ss; labels up/down/other (NOT 5 types). Not 411x271. Writing report/GAP_LITERATURE.md now.

## ROUND 4 (user chose: do BOTH (a) and (b))
- (a) eval-only: add micro Top-1 recall (TDRC def R=TP/sum true types) + micro F1 alongside macro; CV_triplet: average binary metrics over 10 random equal-size negative draws (TDRC L67-135). Rerun affected primary runs (no model change).
- (b) build m_ss (miRNA sequence sim: Needleman-Wunsch on miRBase mature seq; MRGBMDAT says seed region, WeightTDAIGN pairwiseAlignment+minmax) and d_gs (disease gene-based sim Cheng 2014 / DisGeNET or HumanNet LLS per KBLTDARD Eq.1) for v3.2_cuilab; replace GIP view-1; report coverage; rerun v3.2 cuilab both protocols 3 seeds.
- Literature doc delivered: report/GAP_LITERATURE.md.
- (b) downloads done in ~/dhg_runs/faithful/ext/seqgene: mature.fa.gz (actually plain text fa, miRBase), hairpin.fa, CTD_curated_genes_diseases.tsv.gz (cols: symbol, geneid, disease name, MESH:Dxxxx, evidence, , pmids). Plan: m_ss = NW global alignment of mature seqs (hsa-, map precursor names hsa-mir-x -> mature hsa-miR-x-5p/3p via hairpin? use max over mature arms) min-max normalized; d_gs = Cheng2014-style shared-gene sim (Jaccard or cosine of CTD curated gene sets per MeSH ID). New dataset name v3.2_cuilab_deg2_4view (sim_m1=m_ss, sim_m2=MISIM func, sim_d1=d_gs, sim_d2=MeSH sem); missing rows -> fall back GIP? [document].
- (a) code plan in faithful.py: top1() add top1_recall_micro=hit.sum()/Ytrue.sum() & f1_micro; make_splits triplet add test_negs = 10 draws (rng seeds) from unk; exclude all from training neg pool; final eval average binary over 10 draws (keep draw0 too).
- (a) IMPLEMENTED in code/faithful.py (backup of round-2 code: code/faithful_round2.py): top1 adds top1_recall_micro/top1_f1_micro; triplet adds test_negs (n_neg_draws=10, rng 1000*f+i), excluded from train negs, final metrics = mean over draws (+ *_draw0). Smoke OK (v2.0 3 epochs both protocols). TODO: SUMMARY in main() only lists old keys -> add micro keys; aggregate.py add micro columns; then build (b) dataset; then launch reruns (v2.0, cuilab, ref, 3 seeds each protocol; cuilab_4view 3 seeds x2).
- (a) summary keys + aggregate.py micro/draw0 columns patched.
- (b) HumanNet-XC.tsv downloaded (inetbio.org/humannetv3, 1,125,494 lines, cols entrezA entrezB LLS). Coverage cuilab 721 mirs: mature-base 567, hairpin 580, union 628. Plan d_gs = Cheng2014 FunSim (CTD curated disease genes by MeSH UI + HumanNet LLS minmax, KBLTDARD Eq.1); m_ss = NW (Biopython PairwiseAligner) mature seqs, max over arms, minmax; uncovered -> identity row [gia dinh]. Script code/build_4view.py -> data/v3.2_cuilab_deg2_4view.npz. venv pip: use `python -m pip`.
- (b) BUILT data/v3.2_cuilab_deg2_4view.npz (code/build_4view.py, 7s): m_ss covered 628/721 mirs, d_gs 439/609 diseases (CTD marker/mechanism, 6001 genes, 206,102 HumanNet edges); offdiag mean mss 0.289, dgs 0.051.
- LAUNCHED round 4: run_r4.sh + jobs_r4.txt (24 jobs: 4 datasets x type/triplet x seeds 1234,1,2), out results_r4/, logs logs_r4/, xargs PID in logs_r4/_xargs.pid (wait with kill -0). Then aggregate (point aggregate.py R at results_r4), compare r4 vs round2 (should be identical on old keys for type if deterministic; triplet changes by averaging), write report/ROUND4.md + update zip.
- ROUND 4 DONE: 24/24 ok. report/ROUND4.md, results_table_r4.md, compare_r4.txt. CV_type macro identical to r3; micro/10-draw change <=0.004; 4view vs 2view: top1 F1 -0.005 (noise), triplet AUPR -0.004 (all 3 seeds). Zip updated.
