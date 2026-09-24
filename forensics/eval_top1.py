"""Golden evaluator for MDAv3.2-3 — exact port of SPLDHyperAWNTF/expriment.py::CV_type
plus multilabel metrics. All runners reuse this; no re-implementing metrics per runner.

SPLD semantics (VERIFIED from recovered source):
- pair-fold: index_matrix = np.where(A_binary > 0) over ALL pairs (a pair's whole
  5-channel row is masked during training: train_tensor[test_index] = 0).
- folds: np.random.seed(0); np.random.shuffle(index_matrix.T); folds 0-3 take
  sample_num_per_fold = pair_num // 5 pairs each, fold 4 takes the remainder.
- per test pair: top1 = argmax(5 scores); TP += 1 iff top1 is a true type;
  recall_i += tp / positive_num(pair).
- metrics per fold: P = TP / n_pairs; micro_R = TP / total_pos; macro_R =
  mean(tp_i / pos_i). aggregate = mean over folds; F1 = harmonic(P, macro_R).
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

TYPE_ORDER = ['genetics', 'epigenetics', 'circulation', 'target', 'tissue']


def build_pair_folds(A_bin, k_folds=5, seed=0):
    """Return list of k fold tuples (m_idx, d_idx) exactly replicating SPLD's
    seed+shuffle+slice construction."""
    index_matrix = np.array(np.where(A_bin > 0))
    pair_num = index_matrix.shape[1]
    per = int(pair_num / k_folds)
    rng_state = np.random.get_state()  # guard caller's rng
    np.random.seed(seed)
    np.random.shuffle(index_matrix.T)
    np.random.set_state(rng_state)
    folds = []
    for k in range(k_folds):
        if k != k_folds - 1:
            folds.append(tuple(index_matrix[:, k * per:(k + 1) * per]))
        else:
            folds.append(tuple(index_matrix[:, k * per:]))
    return folds


def mask_pair(Y, m_idx, d_idx):
    """Y copy with all channels of given pairs zeroed (SPLD train_tensor[test_index]=0)."""
    Yt = np.array(Y, copy=True)
    Yt[m_idx, d_idx, :] = 0
    return Yt


def top1_metrics(scores, Y_true, test_pairs):
    """scores[M,D,K]; Y_true[M,D,K]; test_pairs=(m_idx,d_idx).
    Returns dict with TP, n_pairs, n_pos, per-fold P/micro_R/macro_R and
    per-pair recall vector for bootstrap."""
    m_idx, d_idx = test_pairs
    n = m_idx.shape[0]
    tp = 0.0
    rec_sum = 0.0
    pos_sum = 0.0
    rec_vec = np.zeros(n)
    hit_vec = np.zeros(n)
    for i in range(n):
        s = scores[m_idx[i], d_idx[i]]
        y = Y_true[m_idx[i], d_idx[i]]
        pos = y.sum()
        pos_sum += pos
        top = int(np.argmax(s))
        h = 1.0 if y[top] > 0 else 0.0
        tp += h
        hit_vec[i] = h
        r = h / pos if pos > 0 else 0.0
        rec_vec[i] = r
        rec_sum += r
    P = tp / n
    miR = tp / pos_sum if pos_sum else 0.0
    maR = rec_sum / n
    return {
        'n_pairs': n, 'n_pos': int(pos_sum), 'TP': tp,
        'legacy_top1_precision': P,
        'legacy_micro_recall': miR,
        'legacy_macro_recall': maR,
        'legacy_f1': 0.0 if P + maR == 0 else 2 * P * maR / (P + maR),
        '_hit_vec': hit_vec, '_rec_vec': rec_vec,
    }


def multilabel_metrics(scores, Y_true, test_pairs):
    """Multi-label metrics over test pairs: micro/macro AUPR + AUC,
    micro/macro-F1 (best threshold swept on scores — caller controls tuning),
    Hit@1, Recall@k."""
    m_idx, d_idx = test_pairs
    S = scores[m_idx, d_idx]  # [n,K]
    Yt = Y_true[m_idx, d_idx]  # [n,K]
    K = Yt.shape[1]
    out = {}
    aupr_per, auc_per, f1_per = [], [], []
    for k in range(K):
        if Yt[:, k].sum() > 0 and len(np.unique(Yt[:, k])) > 1:
            aupr_per.append(average_precision_score(Yt[:, k], S[:, k]))
            auc_per.append(roc_auc_score(Yt[:, k], S[:, k]))
        else:
            aupr_per.append(np.nan)
            auc_per.append(np.nan)
    out['macro_AUPR'] = float(np.nanmean(aupr_per))
    out['macro_AUC'] = float(np.nanmean(auc_per))
    try:
        out['micro_AUPR'] = float(average_precision_score(Yt.ravel(), S.ravel()))
        out['micro_AUC'] = float(roc_auc_score(Yt.ravel(), S.ravel()))
    except ValueError:
        out['micro_AUPR'] = out['micro_AUC'] = np.nan
    # fixed-threshold 0.5 F1 (deterministic; thresholds tuned on inner-val elsewhere)
    yhat = (S > 0.5).astype(int)
    out['micro_F1@0.5'] = float(f1_score(Yt.ravel(), yhat.ravel(), zero_division=0))
    out['macro_F1@0.5'] = float(np.mean([
        f1_score(Yt[:, k], yhat[:, k], zero_division=0) for k in range(K)]))
    # Hit@1 and Recall@k (any-label retrieval)
    n = S.shape[0]
    hit1 = 0.0
    for i in range(n):
        top = int(np.argmax(S[i]))
        if Yt[i, top] > 0:
            hit1 += 1
    out['Hit@1'] = hit1 / n
    for kk in (1, 3):
        rec = 0.0
        for i in range(n):
            topk = np.argsort(S[i])[-kk:]
            pos = Yt[i].sum()
            if pos > 0:
                rec += len(set(topk) & set(np.where(Yt[i] > 0)[0])) / pos
        out[f'Recall@{kk}'] = rec / n
    return out


def evaluate_split(scores, Y_true, test_pairs):
    d = top1_metrics(scores, Y_true, test_pairs)
    d.update(multilabel_metrics(scores, Y_true, test_pairs))
    return d
