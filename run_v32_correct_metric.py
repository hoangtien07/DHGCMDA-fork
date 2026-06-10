"""
run_v32_correct_metric.py — RE-MEASURE v3.2 Top-1 F1 with a CORRECT 5-type metric.

WHY THIS EXISTS (smoking gun, 2026-06-05):
  The official Calculate_Metrics.compute_top1_metrics HARD-CODES 4 types:
    - line 309-318: ground-truth mapping handles real_value 1,2,3,4 -> else: continue
      => drops EVERY type-5 (Tissue) sample.
    - line 323-331: prediction vector length 5 (existence+4) or 4 -> else: continue
      => drops EVERY v3.2 prediction (length 6 = existence + 5 types).
  Net effect for v3.2: valid_samples == 0 -> Top-1 F1 == 0.0 GUARANTEED,
  even for a PERFECT predictor (proven with synthetic test).

  => "v3.2 class collapse Top-1 F1=0.0" across Plans C..J is a METRIC ARTIFACT,
     not a model collapse. The model was never measured on v3.2.

WHAT THIS DOES:
  Monkey-patches Metric_fun.compute_top1_metrics with a GENERAL-K correct version
  (works for any number of types), then runs the UNMODIFIED training pipeline on
  v3.2. NO repo source file is edited. The fix lives only in this standalone script.

USAGE:
  python run_v32_correct_metric.py --device cpu --dataset v3.2_wang \
      --loss_mode softmax_5class --epoch 120 --validation 2
"""
import sys
import numpy as np
from collections import defaultdict

# ---- CORRECT general-K Top-1 metric (drop-in replacement) ----------------
def correct_compute_top1_metrics(self, real_scores_list, predict_scores_list):
    """General Top-1 metric: works for ANY number of types K.
    real_score: scalar true type in 1..K  (or multi-hot vector).
    pred_score: length 1+K [existence, t1..tK]  (existence channel dropped).
    Mirrors the official precision / macro-recall / F1 definitions, but without
    the hard-coded '==4 types' filters that silently skipped all v3.2 samples.
    """
    try:
        if len(real_scores_list) == 0 or len(predict_scores_list) == 0:
            return {'top1_precision': 0.0, 'top1_recall': 0.0, 'top1_f1': 0.0}

        correct_predictions = 0
        total_predictions = 0
        type_correct = defaultdict(int)
        type_total = defaultdict(int)
        valid_samples = 0
        yt, yp = [], []   # collected true/pred type labels for full-suite metrics

        for real_score, pred_score in zip(real_scores_list, predict_scores_list):
            try:
                real_array = np.array(real_score).flatten()
                pred_array = np.array(pred_score).flatten()
                if len(real_array) == 0 or len(pred_array) == 0:
                    continue

                real_value = real_array[0] if len(real_array) == 1 else real_array

                # --- resolve TRUE type (general, no 4-type cap) ---
                if isinstance(real_value, np.ndarray):
                    if real_value.sum() == 0:
                        continue
                    idx = np.where(real_value > 0)[0]
                    if len(idx) == 0:
                        continue
                    true_type = int(idx[0])
                else:
                    if real_value == 0:
                        continue
                    true_type = int(round(float(real_value))) - 1  # 1..K -> 0..K-1
                    if true_type < 0:
                        continue

                # --- resolve PREDICTED type (general: drop existence channel) ---
                if len(pred_array) >= 2:
                    type_scores = pred_array[1:]      # [existence, t1..tK] -> t1..tK
                else:
                    continue
                predicted_type = int(np.argmax(type_scores))

                valid_samples += 1
                total_predictions += 1
                type_total[true_type] += 1
                yt.append(true_type)
                yp.append(predicted_type)
                if predicted_type == true_type:
                    correct_predictions += 1
                    type_correct[true_type] += 1
            except Exception:
                continue

        # --- PRIMARY definition = same hybrid our code uses (micro-precision + macro-recall),
        #     which is calibrated against v2.0 (our 0.5996 ~ paper 0.5970). ---
        top1_precision = correct_predictions / total_predictions if total_predictions > 0 else 0.0   # micro / accuracy
        recalls = [type_correct[t] / type_total[t] for t in type_total if type_total[t] > 0]
        top1_recall = float(np.mean(recalls)) if recalls else 0.0                                     # macro recall
        if top1_precision + top1_recall > 0:
            top1_f1 = 2 * top1_precision * top1_recall / (top1_precision + top1_recall)
        else:
            top1_f1 = 0.0

        # --- FULL metric suite for paper comparison (paper v3.2: P=0.7915 R=0.9421 F1=0.86, macro) ---
        suite = ""
        try:
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            if len(yt) > 0:
                acc = accuracy_score(yt, yp)
                mp, mr, mf, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
                wp, wr, wf, _ = precision_recall_fscore_support(yt, yp, average='weighted', zero_division=0)
                suite = (f" | acc={acc:.4f} macroP={mp:.4f} macroR={mr:.4f} macroF1={mf:.4f} "
                         f"weightedF1={wf:.4f}")
        except Exception:
            suite = ""

        # loud marker so we can grep the corrected number out of the log
        print(f"[CORRECT-METRIC] valid_samples={valid_samples} "
              f"P={top1_precision:.4f} R={top1_recall:.4f} F1={top1_f1:.4f}{suite} "
              f"per_type_total={dict(type_total)}")
        return {'top1_precision': top1_precision,
                'top1_recall': top1_recall,
                'top1_f1': top1_f1}
    except Exception as e:
        print(f"[CORRECT-METRIC] ERROR: {e}")
        return {'top1_precision': 0.0, 'top1_recall': 0.0, 'top1_f1': 0.0}


def main():
    import Calculate_Metrics
    # Patch the CLASS method BEFORE any Metric_fun() instance is created in training.
    Calculate_Metrics.Metric_fun.compute_top1_metrics = correct_compute_top1_metrics
    print("=" * 70)
    print("[PATCH] Metric_fun.compute_top1_metrics -> general-K CORRECT version")
    print("[PATCH] NO repo source file modified. Fix lives in this script only.")
    print("=" * 70)

    from param import parameter_parser
    args = parameter_parser()  # parses sys.argv (the flags you passed on the CLI)
    print(f"[RUN] dataset={getattr(args,'dataset',None)} "
          f"loss_mode={getattr(args,'loss_mode',None)} "
          f"epoch={args.epoch} validation={args.validation} "
          f"predictor_mode={getattr(args,'predictor_mode','diag')}")

    import main_experiments_hetero1 as M
    M.main_optimized(args)


if __name__ == '__main__':
    main()
