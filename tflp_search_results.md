# A24 — TFLP baseline code search (Plan I, 2026-06-03)

## Result: CODE FOUND ✅

**TFLP public repository**: https://github.com/nayu0419/TFLP
- Method: "Predicting Multiple Types of MicroRNA-disease Associations based on Tensor Factorization and Label Propagation"
- Author: Yu et al. (ref [25] trong paper DHGCMDA)
- Approach: Tensor robust PCA (low-rank tensor) + label propagation

## Quyết định: OUT OF CURRENT SCOPE

Theo adversarial ceiling-check (Plan I roadmap), implement TFLP:
- **Cost**: 40-60h work + 3-5h CPU (adapt + multi-seed)
- **Gain**: +16.7% baseline coverage = chỉ ~1.7% overall reproduce (baseline là 1/6 categories × 1/6 weight)
- **Verdict**: LOW PRIORITY — gain quá nhỏ so với effort.

→ Code có sẵn, để dành **future work** nếu cần đẩy baseline coverage từ 1/6 (TDRC) lên 2/6 (TDRC+TFLP). Hiện tại user đã chốt STOP sau diagnostics.

## Trạng thái 6 baselines paper Table 3-4

| Baseline | Public code | Status |
|---|---|---|
| TDRC | ✅ github.com/BioMedicalBigDataMiningLab/TDRC | **Reproduced** (Plan C, +4%) |
| TFLP | ✅ github.com/nayu0419/TFLP | **Found** (out of scope, future work) |
| NMCMDA | ✅ github.com/ljatynu/NMCMDA | Cloned, DGL incompat (skipped) |
| MRFGMDA | ❌ no public code | Blocked |
| KBLTDARD | ❌ no public code | Blocked |
| SPLDHyperAWNTF | ❌ no public code | Blocked |

→ Baseline coverage ceiling thực tế: **3/6 = 50%** nếu implement TFLP + fix NMCMDA DGL. Hiện 1/6 = 17%.
