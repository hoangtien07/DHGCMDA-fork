# FROZEN LEADERBOARD — MDAv3.2-3, golden SPLD pair-fold protocol

Frozen at final-validation-suite completion. Seeds 0/1/2 for learned
models; deterministic methods report seed0 only (rep-CV seeds for
vote_gip_dssm shown as secondary robustness).

| Model | Track | P | micro-R | macro-R | **legacy-F1** | mAUPR | seeds |
|---|---|---:|---:|---:|---:|---:|---|
| vote_gip_only | honest | 0.6558 | 0.4877 | 0.5380 | **0.5910** | 0.4108 | seed0:0.591 |
| simknn_misim | LEAKY | 0.6456 | 0.4800 | 0.5264 | **0.5799** | 0.4379 | seed0:0.580 |
| vote_misim_dssm | LEAKY | 0.6456 | 0.4800 | 0.5264 | **0.5799** | 0.4379 | seed0:0.580 |
| vote_fmisim_dssm | honest | 0.6390 | 0.4752 | 0.5199 | **0.5733** | 0.4383 | seed0:0.573 |
| vote_gip_gip | honest | 0.6305 | 0.4688 | 0.5126 | **0.5655** | 0.4274 | seed0:0.565 |
| simknn_j | honest | 0.6275 | 0.4666 | 0.5094 | **0.5623** | 0.4254 | seed0:0.562 |
| vote_gip_dssm_j | honest | 0.6262 | 0.4657 | 0.5081 | **0.5610** | 0.4348 | seed0:0.561 |
| simknn_gip | honest | 0.6253 | 0.4650 | 0.5070 | **0.5599** | 0.4536 | seed0:0.560 |
| vote_gip_dssm | honest | 0.6253 | 0.4650 | 0.5070 | **0.5599** | 0.4536 | seed0:0.560/seed1:0.562/seed2:0.563 |
| SPLD_noMISIM | LEGACY(corrected) | 0.6227 | 0.4631 | 0.5078 | **0.5594** | nan | seed0:0.559 |
| SPLD_staticMISIM | LEGACY(leaky-lite) | 0.6219 | 0.4624 | 0.5069 | **0.5585** | nan | seed0:0.559 |
| SPLD_foldMISIM | LEGACY(corrected) | 0.6203 | 0.4613 | 0.5054 | **0.5570** | nan | seed0:0.557 |
| DHGCMDA_encoder | honest | 0.5094 | 0.3789 | 0.4000 | **0.4481** | 0.3531 | seed0:0.448 |
| vote_dssm_only | honest | 0.4781 | 0.3555 | 0.3702 | **0.4173** | 0.3977 | seed0:0.417 |
| cooccur_prior | honest | 0.4576 | 0.3403 | 0.3461 | **0.3941** | 0.2690 | seed0:0.394 |
| bilinear | honest | 0.3328 | 0.2476 | 0.2577 | **0.2904** | 0.3217 | seed0:0.290/seed1:0.282/seed2:0.275 |
| distmult | honest | 0.2835 | 0.2107 | 0.2122 | **0.2427** | 0.2792 | seed0:0.243/seed1:0.260/seed2:0.255 |
