# G5.2 Checkpoint Attribution Feasibility

Inspected `runs/g5_fold0_s1234/checkpoint.pt` (structure only — no forward/backward, no optimizer step).

Contents: `state_dict` (89 params: clm/cld, avf_m/avf_d, rec_m/rec_d, inp, hgt.0/1 conv+ln, pred.pm/pd), `best_val_macro_f1`, `thresholds`, `threshold_source`.

## Finding: EXACT_CHECKPOINT_GRAPH_STATE_UNAVAILABLE

The checkpoint does NOT store the epoch-60 HGCN operators `G = {m1,m2,d1,d2}` nor the HGT edge index dict `ei`. Those lived in `best_state` in process memory and were re-loaded but never persisted. Reconstruction from saved artifacts is impossible: `G` derives from `mm01(S_m/S_d)` at ep60 (recon outputs, not saved) plus `knn_G`/`edges` under θ=0.5 — none of the intermediate S matrices, KNN graphs, or edge sets exist on disk.

Per directive: no approximate graph was substituted and no "exact" attribution was claimed. Exact epoch-60 forward/backward attribution is therefore **not executable** from existing artifacts; it would require a future run that persists `G`/`ei` inside best_state snapshots.
