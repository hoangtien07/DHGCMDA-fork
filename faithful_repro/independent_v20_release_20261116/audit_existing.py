"""Inventory the prior v2.0 baseline without trusting its retrospective label."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
REV='06f9ba3eed92e334d9a2b8003fc06d83596e5614'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    log=ROOT/'logs/baseline_v2.0_full.log';met=ROOT/'results/baseline_v2.0_metrics.json'
    raw=log.read_bytes()
    text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8',errors='replace')
    data=json.loads(met.read_text())
    folds=[(int(a),float(b),float(c)) for a,b,c in re.findall(r'Fold (\d+) completed - AUC: ([0-9.]+), Top-1 F1: ([0-9.]+)',text)]
    assert len(folds)==5 and [x[0] for x in folds]==[1,2,3,4,5]
    assert abs(sum(x[2] for x in folds)/5-data['top1_f1']) < 0.00006
    assert 'Attention Heads: 4' in text and 'Epochs: 650' in text and 'Random Seed: 1234' in text
    assert 'Loaded indices from cache' in text
    release=json.loads((HERE/'source_manifest.json').read_text())
    local=[]
    for item in release['files']:
        p=ROOT/item['path']
        local.append({'path':item['path'],'local_exists':p.exists(),
                      'local_equal_release':p.exists() and sha(p)==item['sha256']})
    cmd=['git','log','-1','--format=%H %aI %s','--','logs/baseline_v2.0_full.log']
    commit=subprocess.check_output(cmd,cwd=ROOT,text=True).strip()
    stat=subprocess.check_output(['git','diff','--stat',REV,'1cf27bda69d6af981115b62840b2fd8913a559fb','--',
           'main_experiments_hetero1.py','param.py','prepareData.py','hetero_model.py','Calculate_Metrics.py'],cwd=ROOT,text=True).strip()
    out={'log_sha256':sha(log),'metrics_json_sha256':sha(met),'log_last_git_commit':commit,
      'folds':[{'fold':a,'auc_displayed':b,'top1_f1_displayed':c} for a,b,c in folds],
      'metric_json':data,'log_evidence':{'epochs':650,'heads':4,'reported_seed':1234,'folds':5,'cache_used':True},
      'pinned_release_default_heads':8,'version_mismatch':'historical log used 4 heads; upstream commit defaults to 8',
      'historical_commit_diff_from_release':stat,
      'provenance_limit':'Commit containing the log is traceable, but log does not cryptographically bind its executing source, exact input data, cache content, fold membership, per-pair scores or environment.',
      'reported_0_5570':'appears in retrospective markdown, not located in searched local logs/results; source not established',
      'local_files_vs_release':local}
    (HERE/'existing_baseline_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','historical_log_commit':commit,'folds':len(folds),'f1':data['top1_f1'],
          'log_heads':4,'release_heads':8,'cache_used':True,'local_files_equal_release':sum(x['local_equal_release'] for x in local),
          'total_files':len(local)},indent=2))


if __name__=='__main__':main()
