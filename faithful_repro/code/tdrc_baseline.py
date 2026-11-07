import argparse
import inspect
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

import faithful

TDRC_DIR = os.path.expanduser('~/dhg_runs/lit/TDRC')
sys.path.insert(0, TDRC_DIR)
from data import GetData
from method import Model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--protocol', choices=['type', 'triplet'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max_iter', type=int, default=500)
    p.add_argument('--data_dir', default=os.path.expanduser('~/dhg_runs/faithful/data'))
    p.add_argument('--only_fold', type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args()


def disease_similarity(data_name, data, nd):
    name = data_name.lower()
    if name == 'v2.0':
        key = 'sim_d2'
        description = 'D_SSM1, MeSH semantic disease similarity'
    elif 'cuilab' in name:
        key = 'sim_d2'
        description = 'SS1 (Wang), MeSH-derived semantic disease similarity'
    elif 'ref' in name:
        key = 'sim_d2'
        description = 'DSSM'
    else:
        raise ValueError(f'No TDRC disease-similarity mapping for dataset {data_name!r}')
    similarity = np.asarray(data[key], dtype=np.float64)
    if similarity.shape != (nd, nd):
        raise ValueError(f'{data_name}:{key} has shape {similarity.shape}, expected {(nd, nd)}')
    if not np.isfinite(similarity).all():
        raise ValueError(f'{data_name}:{key} contains non-finite values')
    return key, description, np.mat(similarity)


def predict_with_iteration_stats(train_tensor, S_d, S_m, max_iter):
    import tensorly as tl

    params = dict(r=4, alpha=0.125, beta=0.25, lam=0.001, tol=1e-6, max_iter=max_iter)
    model = Model('TDRC')
    tdrc_method = model.TDRC
    target_code = tdrc_method.__func__.__code__
    source, first_line = inspect.getsourcelines(tdrc_method)
    convergence_line = max(
        first_line + offset for offset, line in enumerate(source)
        if line.strip() == 'if err < tol:'
    )
    stats = dict(iterations=0, converged=False, last_relative_error=None)
    original_fold = tl.fold
    original_khatri_rao = tl.tenalg.khatri_rao

    def compatible_khatri_rao(matrices, *args, **kwargs):
        return original_khatri_rao(
            [np.asarray(matrix) for matrix in matrices], *args, **kwargs
        )

    def counted_fold(*args, **kwargs):
        mode = kwargs.get('mode', args[1] if len(args) > 1 else None)
        if mode == 2:
            stats['iterations'] += 1
            if stats['iterations'] % 25 == 0:
                print(f'  TDRC iteration {stats["iterations"]}/{max_iter}', flush=True)
        return original_fold(*args, **kwargs)

    def trace(frame, event, arg):
        if frame.f_code is target_code:
            if event == 'line' and frame.f_lineno == convergence_line:
                error = float(frame.f_locals['err'])
                stats['last_relative_error'] = error
                stats['converged'] = error < params['tol']
            return trace
        return None

    previous_trace = sys.gettrace()
    if previous_trace is not None:
        raise RuntimeError('Unexpected active Python trace; cannot instrument TDRC iterations safely')
    tl.fold = counted_fold
    tl.tenalg.khatri_rao = compatible_khatri_rao
    sys.settrace(trace)
    try:
        prediction = model()(train_tensor, S_d, S_m, **params)
    finally:
        sys.settrace(None)
        tl.fold = original_fold
        tl.tenalg.khatri_rao = original_khatri_rao
    return prediction, stats, params


def main():
    args = parse_args()
    if args.max_iter < 1:
        raise ValueError('--max_iter must be positive')
    data_path = os.path.join(args.data_dir, args.data + '.npz')
    with np.load(data_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    Y = np.asarray(data['Y'])
    nm, nd, C = Y.shape
    S_d_key, S_d_description, S_d = disease_similarity(args.data, data, nd)

    split_args = SimpleNamespace(
        protocol=args.protocol,
        folds=5,
        early_stop=1,
        val_frac=0.1,
        n_neg_draws=10,
    )
    faithful.seed_all(args.seed)
    folds = faithful.make_splits(Y, split_args, np.random.default_rng(args.seed))
    selected_folds = [
        (fold_id, fold) for fold_id, fold in enumerate(folds)
        if args.only_fold < 0 or fold_id == args.only_fold
    ]
    if not selected_folds:
        raise ValueError(f'--only_fold {args.only_fold} selects no fold')

    results = []
    for fold_id, fold in selected_folds:
        fold_start = time.time()
        train_tensor = np.array(Y, dtype=np.float64, copy=True)
        if args.protocol == 'type':
            train_tensor[fold['test'][:, 0], fold['test'][:, 1], :] = 0
        else:
            train_tensor[tuple(fold['test'].T)] = 0
        train_matrix = train_tensor.sum(axis=2)
        train_matrix[train_matrix > 0] = 1

        print(
            f'FOLD {fold_id} start: protocol={args.protocol}, '
            f'test={len(fold["test"])}, train_tensor={train_tensor.shape}',
            flush=True,
        )
        get_data = object.__new__(GetData)
        get_data.miRNA_num = nm
        get_data.dis_num = nd
        get_data.dis_sim = S_d
        S_m = np.mat(get_data.get_functional_sim(train_matrix))
        print(f'  S_m computed in {time.time() - fold_start:.1f}s', flush=True)

        prediction, iteration_stats, tdrc_params = predict_with_iteration_stats(
            train_tensor, S_d, S_m, args.max_iter
        )
        if args.protocol == 'type':
            test = fold['test']
            y_true = Y[test[:, 0], test[:, 1]]
            scores = prediction[test[:, 0], test[:, 1]]
            metrics = faithful.top1(y_true, scores)
        else:
            score = lambda triplets: prediction[tuple(np.asarray(triplets).T)]
            draw0 = faithful.evaluate(fold['test'], fold['test_neg'], score)
            draws = [
                faithful.evaluate(fold['test'], negatives, score)
                for negatives in fold['test_negs']
            ]
            metrics = dict(draw0)
            metrics.update({
                f'{key}_draw0': draw0[key] for key in ('auc', 'aupr', 'f1')
            })
            metrics.update({
                key: float(np.mean([draw[key] for draw in draws]))
                for key in ('auc', 'aupr', 'f1')
            })
            metrics['n_neg_draws'] = len(draws)

        metrics.update(
            fold=fold_id,
            train_time_s=round(time.time() - fold_start, 1),
            iterations=iteration_stats['iterations'],
            converged=iteration_stats['converged'],
            last_relative_error=iteration_stats['last_relative_error'],
        )
        results.append(metrics)
        print('FOLD ' + str(fold_id) + ' ' + json.dumps(metrics), flush=True)

    keys = (
        ['top1_precision', 'top1_recall', 'top1_f1', 'top1_recall_micro', 'top1_f1_micro']
        if args.protocol == 'type'
        else ['auc', 'aupr', 'f1', 'auc_draw0', 'aupr_draw0', 'f1_draw0']
    )
    summary = {
        key: [float(np.mean([result[key] for result in results])),
              float(np.std([result[key] for result in results]))]
        for key in keys
    }
    output_args = vars(args).copy()
    output_args.update(
        folds=5,
        early_stop=1,
        val_frac=0.1,
        n_neg_draws=10,
        ablation='TDRC',
        recon_reduction='sum',
        diag_leak=0,
    )
    output = dict(
        args=output_args,
        deviations={
            'validation': 'validation entries remain in the TDRC training tensor; no early stopping',
            'prediction': 'TDRC predict_tensor is scored directly',
        },
        folds=results,
        summary=summary,
        model='TDRC',
        tdrc_params=tdrc_params,
        similarity_sources={
            'S_d_key': S_d_key,
            'S_d': S_d_description,
            'S_m': 'TDRC GetData.get_functional_sim(train_matrix), with dis_sim=S_d',
        },
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as output_file:
        json.dump(output, output_file, indent=1)
    print('SUMMARY ' + json.dumps(summary), flush=True)
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
