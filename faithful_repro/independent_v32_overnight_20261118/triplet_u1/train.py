# -*- coding: utf-8 -*-
"""CVtriplet-U1 REV2 SANITY driver — fold0 / seed1234 / 650 epochs only.

Runs the pinned U1X contract once, then scores the pinned eval sets
(Task T + Task P) and the matched logistic baseline, writes artifacts
under runs/fold0_seed1234/. NOT a result — design validation only
(PROTOCOL_CVTRIPLET_U1_REV2 §7)."""
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


def write(p, x):
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0, choices=range(5))
    ap.add_argument('--seed', type=int, default=1234,
                    choices=(1234, 42, 2025))
    ap.add_argument('--sanity', action='store_true', default=False)
    a = ap.parse_args()
    # REV2 scope gate: sanity only — anything else requires a new
    # approved protocol amendment.
    if not a.sanity or not (a.fold == 0 and a.seed == 1234):
        raise SystemExit('BLOCKED: only --sanity fold0 seed1234 is '
                         'allowed under REV2; full 5x3 needs approval')
    key = f'fold{a.fold}_seed{a.seed}'
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
    assert pf['source_hashes']['train.py'] == sha(__file__)
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
    assert cfg['seed'] == cfg['inter_seed'] == a.seed \
        and cfg['max_epochs'] == 650 and cfg['lam_exist'] == 0.3 \
        and cfg['inter_cap_mult'] == 10
    exp = read(cdir / 'expected.json')
    data = s1r.load_data_npz(str(bdir / 'candidate_fold_data.npz'))

    out = HERE / 'runs' / key
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'launch.json',
          {'key': key, 'sanity': True,
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
                data_file_sha=sha(bdir / 'candidate_fold_data.npz'))
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
            raise RuntimeError('aborted training')
        import run_eval
        ev = run_eval.run_eval(
            str(bdir), str(out / 'out' / 'ckpt_best.pt'),
            str(STRICT), a.fold, out / 'eval.json')
        print('EVAL DONE auroc_T %.4f auroc_P %.4f base_T %.4f'
              % (ev['task_T']['mean']['auroc'],
                 ev['task_P']['mean']['auroc'],
                 ev['baseline_logistic']['mean']['auroc']), flush=True)
    except BaseException:
        (out / 'exception.txt').write_text(traceback.format_exc(),
                                          encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
