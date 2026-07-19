#!/usr/bin/env python3
"""B4-features (branch breakthrough-imbalance-features): feature miRNA từ SEQUENCE THẬT.

Bối cảnh: paper claim đóng góp #1 = "avoid excessive reliance on association-derived similarity",
nhưng code dùng M-GSM = GIP (association-derived) gán nhãn sai là "miRNA-sequence" (xem 介绍.txt).
Script này thay bằng feature sequence THẬT: đọc miRBase mature.fa → k-mer frequency cho từng miRNA →
similarity 495×495 (cosine). Xuất M_SEQ.txt cùng định dạng M_GSM.txt để nạp thay thế (additive).

Không train, không đụng model. Chỉ numpy/pandas.
"""
import argparse
import os
import re
import numpy as np
import pandas as pd


def parse_mature(fa_path, species="hsa"):
    """Trả dict {name_lower: RNA_seq} cho loài species."""
    seqs = {}
    name = None
    with open(fa_path) as fh:
        for line in fh:
            if line.startswith(">"):
                h = line[1:].split()[0]
                name = h.lower() if h.lower().startswith(species + "-") else None
            elif name:
                seqs[name] = line.strip().upper().replace("T", "U")
    return seqs


def norm_precursor(nm):
    nm = str(nm).strip().lower()
    if not nm.startswith("hsa-"):
        nm = "hsa-" + nm
    return nm


def match_matures(prec, seqs):
    """Tìm mọi mature khớp precursor prec (prefix + ranh giới ký tự, tránh 1→100).

    prec 'hsa-mir-125a' khớp 'hsa-mir-125a-5p','hsa-mir-125a-3p'.
    prec 'hsa-mir-26'   khớp 'hsa-mir-26a-...','hsa-mir-26b-...' (gộp).
    """
    # ranh giới: sau prefix phải là hết chuỗi, '-', hoặc chữ cái (a/b/c) — KHÔNG phải chữ số
    pat = re.compile("^" + re.escape(prec) + r"([a-z]|-|$)")
    hits = [s for k, s in seqs.items() if pat.match(k)]
    # ưu tiên khớp chính xác precursor + 5p/3p nếu có
    return hits


def kmer_vector(seq, k):
    from itertools import product
    kmers = ["".join(p) for p in product("ACGU", repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    v = np.zeros(len(kmers), dtype=np.float64)
    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]
        if km in idx:
            v[idx[km]] += 1
    s = v.sum()
    return v / s if s > 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="mature.fa")
    ap.add_argument("--names_xls", default="v2.0_495m383D/miRNA name.xls")
    ap.add_argument("--k", type=int, default=4, help="k-mer size (mặc định 4 → 256 chiều)")
    ap.add_argument("--out", default="v2.0_495m383D/M_SEQ.txt")
    ap.add_argument("--out_feat", default="results/b4/mirna_kmer_feat.npy")
    args = ap.parse_args()

    df = pd.read_excel(args.names_xls, header=None)
    # cột tên = cột chứa 'mir'/'let'
    name_col = 1 if df.shape[1] > 1 else 0
    for c in range(df.shape[1]):
        if df.iloc[:, c].astype(str).str.contains("mir|let", case=False, regex=True).mean() > 0.5:
            name_col = c
            break
    names = df.iloc[:, name_col].astype(str).tolist()
    n = len(names)

    seqs = parse_mature(args.fasta)
    print(f"miRNA names: {n} | human mature seqs: {len(seqs)} | k={args.k}")

    dim = 4 ** args.k
    feat = np.zeros((n, dim), dtype=np.float64)
    matched, unmatched = 0, []
    for i, nm in enumerate(names):
        prec = norm_precursor(nm)
        hits = match_matures(prec, seqs)
        if hits:
            feat[i] = np.mean([kmer_vector(s, args.k) for s in hits], axis=0)
            matched += 1
        else:
            unmatched.append(nm)
    # unmatched → mean vector (giữ neutral, không leak)
    if matched:
        mean_vec = feat[np.any(feat != 0, axis=1)].mean(axis=0)
        for i, nm in enumerate(names):
            if not np.any(feat[i] != 0):
                feat[i] = mean_vec
    print(f"MATCHED {matched}/{n} ({100 * matched / n:.1f}%); unmatched→mean: {len(unmatched)}")
    print("unmatched:", unmatched)

    # cosine similarity 495x495
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = feat / norm
    sim = unit @ unit.T
    sim = np.clip(sim, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)

    os.makedirs(os.path.dirname(args.out_feat), exist_ok=True)
    np.save(args.out_feat, feat.astype(np.float32))
    np.savetxt(args.out, sim, fmt="%.6f")
    print(f"✅ saved sequence similarity → {args.out}  shape={sim.shape}  "
          f"mean_offdiag={sim[~np.eye(n, dtype=bool)].mean():.4f}")

    # so nhanh với M_GSM (GIP) để chứng minh KHÁC association-derived
    gip_path = os.path.join(os.path.dirname(args.names_xls), "M_GSM.txt")
    if os.path.exists(gip_path):
        gip = np.loadtxt(gip_path)
        if gip.shape == sim.shape:
            off = ~np.eye(n, dtype=bool)
            corr = np.corrcoef(sim[off].ravel(), gip[off].ravel())[0, 1]
            print(f"corr(M_SEQ, M_GSM=GIP) off-diagonal = {corr:.4f} "
                  f"(thấp = feature sequence ĐỘC LẬP với association-derived GIP)")


if __name__ == "__main__":
    main()
