# -*- coding: utf-8 -*-
"""CVtriplet-U1 REV2 job runner — one locked contract (foldN_seedS).

Gates: only keys present in preflight.json contracts; never overwrites
an existing out/ (occupied namespace refused unless it is the sanity
fold0 dir, where TRAIN is kept and only eval artifacts are appended).
Writes status.json with decomposed TRAIN_PASS / EVAL_PASS /
VERIFY_PASS (RUN_PASS = all three)."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch_geometric

HERE = Path(__file__).resolve().parent
OVER = HERE.parent
PKG = OVER.parents[1] / 'faithful_repro' / \
    'colab_package_f0_selection_20261115'
STRICT = PKG / 'inputs/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz'
SRC = OVER.parents[1] / 'faithful_repro' / 'paper_guided' / \
    'u1_horizon650_fold0_rev1' / 'source'
for _p in (str(SRC), str(SRC / 'stages'), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import runner as s1r                    # noqa: E402
import train_u1x                        # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
KEYS = ['fold%d_seed%d' % (f, s) for f in range(5)
        for s in (1234, 42, 2025)]


def write(p, x):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write('\n')


def write_allow(p, x):
    with Path(p).open('w', encoding='utf-8') as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write('\n')


def status_of(run_dir):
    d = {'key': run_dir.name, 'train_pass': False, 'eval_pass': False,
         'verify_pass': False, 'run_pass': False,
         'train_pass_source': 'completion.json',
         'eval_pass_source': 'eval_full.json',
         'verify_pass_source': 'verification_full.json'}
    try:
        d['train_pass'] = bool(
            (run_dir / 'completion.json').exists()
            and read(run_dir / 'completion.json')['status'] == 'PASS')
    except Exception:
        pass
    try:
        d['eval_pass'] = bool(
            (run_dir / 'eval_full.json').exists()
            and read(run_dir / 'eval_full.json')['status'] == 'PASS'
            and (run_dir / 'eval_arrays.npz').exists())
    except Exception:
        pass
    try:
        d['verify_pass'] = bool(
            (run_dir / 'verification_full.json').exists()
            and read(run_dir / 'verification_full.json')['status']
            == 'PASS')
    except Exception:
        pass
    d['run_pass'] = (d['train_pass'] and d['eval_pass']
                     and d['verify_pass'])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True, choices=KEYS)
    ap.add_argument('--skip-train-if-done', action='store_true',
                    default=False)
    a = ap.parse_args()
    key = a.key
    fold = int(key.split('_')[0].replace('fold', ''))

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.__version__.split('+')[0] == '2.5.1' \
        and torch_geometric.__version__ == '2.6.1' \
        and np.__version__ == '1.26.4' \
        and not torch.cuda.is_available()
    pf = read(HERE / 'preflight.json')
    assert pf['status'] == 'PASS'
    assert pf['protocol_sha256'] == sha(
        OVER / 'PROTOCOL_CVTRIPLET_U1_REV2.md')
    assert pf['preflight_script_sha256'] == sha(HERE / 'preflight.py')
    for n, h in pf['source_hashes'].items():
        assert sha(HERE / n) == h, ('source drift', n)
    e = next(x for x in pf['entries'] if x['key'] == key)
    bdir = Path(e['bundle'])
    cdir = HERE / 'contracts' / key
    for name, path in [('bundle', bdir / 'candidate_fold_data.npz'),
                       ('cfg', cdir / 'cfg.json'),
                       ('expected', cdir / 'expected.json')]:
        assert sha(path) == e[name + '_sha256'], name
    neg_idx = np.load(cdir / 'neg_exist_idx.npy')
    assert hashlib.sha256(np.ascontiguousarray(neg_idx)
                          .tobytes()).hexdigest() == \
        e['neg_exist_sha256']
    cfg = read(cdir / 'cfg.json')
    assert cfg['seed'] == cfg['inter_seed'] == int(key.split('seed')[1])
    assert cfg['max_epochs'] == 650 and cfg['lam_exist'] == 0.3 \
        and cfg['inter_cap_mult'] == 10
    exp = read(cdir / 'expected.json')
    data = s1r.load_data_npz(str(bdir / 'candidate_fold_data.npz'))

    out = HERE / 'runs' / key
    trained_here = False
    if out.exists():
        st = status_of(out)
        if not (a.skip_train_if_done and st['train_pass']):
            raise SystemExit(
                'BLOCKED: %s exists and train_pass=%s — no overwrite'
                % (out, st['train_pass']))
        comp = read(out / 'completion.json')
        assert sha(out / 'out' / 'manifest.json') == \
            comp['manifest_sha256']
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out / 'launch.json',
              {'key': key,
               'preflight_sha256': sha(HERE / 'preflight.json'),
               'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                          time.gmtime())})
        try:
            with (out / 'progress.log').open('x', encoding='utf-8',
                                            buffering=1) as f:
                def log(x):
                    print(x, file=f, flush=True)
                    if x.startswith('epoch ') \
                            and int(x.split()[1]) % 100 == 0:
                        print(key, x, flush=True)
                res = train_u1x.train_u1x(
                    cfg, data, str(out / 'out'), neg_idx, log=log,
                    expected_manifest=exp,
                    meta_path=str(bdir / 'recon_meta.json'),
                    audit_path=str(bdir / 'recon_audits.json'),
                    data_file_sha=sha(
                        bdir / 'candidate_fold_data.npz'))
            d = {'status': 'PASS' if res['aborted'] is None
                 and res['final_epoch'] == 650 else 'FAIL',
                 'key': key, 'final_epoch': res['final_epoch'],
                 'aborted': res['aborted'],
                 'manifest_sha256': sha(out / 'out' / 'manifest.json'),
                 'end_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                          time.gmtime())}
            write(out / 'completion.json', d)
            print('TRAIN DONE', key, d['status'], flush=True)
            if d['status'] != 'PASS':
                write_allow(out / 'status.json', status_of(out))
                raise RuntimeError('aborted training')
            trained_here = True
        except BaseException:
            (out / 'exception.txt').write_text(
                traceback.format_exc(), encoding='utf-8')
            write_allow(out / 'status.json', status_of(out))
            raise
    # eval (never overwrites: new filenames eval_full/eval_arrays)
    try:
        import run_eval
        ev = run_eval.run_eval(
            str(bdir), str(out / 'out' / 'ckpt_best.pt'),
            str(STRICT), fold, out / 'eval_full.json',
            arrays_path=out / 'eval_arrays.npz')
        print('EVAL DONE auroc_T %.4f auroc_P %.4f base_T %.4f'
              % (ev['task_T']['mean']['auroc'],
                 ev['task_P']['mean']['auroc'],
                 ev['baseline_logistic']['mask_zero_pairs']['mean']
                 ['auroc']), flush=True)
    except BaseException:
        (out / 'eval_exception.txt').write_text(
            traceback.format_exc(), encoding='utf-8')
        write_allow(out / 'status.json', status_of(out))
        raise
    # verify (artifact-level, independent recompute)
    import verify
    verify.main(key)
    st = status_of(out)
    st['trained_here'] = trained_here
    write_allow(out / 'status.json', st)
    print('RUN', key, 'run_pass=', st['run_pass'], flush=True)
    if not st['run_pass']:
        raise RuntimeError('verification failed')


if __name__ == '__main__':
    main()
