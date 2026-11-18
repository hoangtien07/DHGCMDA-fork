# -*- coding: utf-8 -*-
"""CVtriplet-U1 REV2 full launcher — 5 folds x 3 seeds, max 2 workers.

Runs train.py --key K --skip-train-if-done per locked contract.
Never overwrites: existing run dirs keep their artifacts; train.py
refuses to replace a finished out/. On any provenance drift or
verification failure the job is recorded FAILED and the supervisor
keeps going (it does NOT self-fix code, per the council verdict).
Aggregates runs/*/status.json into supervisor_status.json."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
KEYS = ['fold%d_seed%d' % (f, s) for f in range(5)
        for s in (1234, 42, 2025)]
MAXW = 2


def read(p):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    except Exception:
        return None


def main():
    pending = []
    done = {}
    for key in KEYS:
        st = read(HERE / 'runs' / key / 'status.json')
        if st and st.get('run_pass'):
            st['rc'] = 0
            done[key] = st
            print('SKIP (already run_pass)', key, flush=True)
        else:
            pending.append(key)
    running = {}
    (HERE / 'job_logs').mkdir(exist_ok=True)
    while pending or running:
        while pending and len(running) < MAXW:
            key = pending.pop(0)
            f = open(HERE / 'job_logs' / f'{key}.log', 'a')
            p = subprocess.Popen(
                [PY, '-X', 'utf8', str(HERE / 'train.py'),
                 '--key', key, '--skip-train-if-done'],
                stdout=f, stderr=subprocess.STDOUT,
                cwd=str(HERE))
            running[key] = (p, f, time.time())
            print('START', key, flush=True)
        time.sleep(10)
        for key, (p, f, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            f.close()
            st = read(HERE / 'runs' / key / 'status.json') or {}
            st['rc'] = rc
            st['key'] = key
            done[key] = st
            del running[key]
            print('DONE', key, 'rc=%d run_pass=%s'
                  % (rc, st.get('run_pass')), flush=True)
            if rc != 0 and not st.get('verify_pass'):
                print('  FAILED JOB (no self-fix):', key, flush=True)
    sup = {'status': 'PASS' if all(
        d.get('run_pass') for d in done.values()) else 'PARTIAL',
           'jobs': done,
           'n_pass': sum(1 for d in done.values() if d.get('run_pass')),
           'end_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                    time.gmtime())}
    with open(HERE / 'supervisor_status.json', 'w',
              encoding='utf-8') as f:
        json.dump(sup, f, indent=1)
        f.write('\n')
    print('SUPERVISOR', sup['status'], sup['n_pass'], '/15',
          flush=True)


if __name__ == '__main__':
    main()
