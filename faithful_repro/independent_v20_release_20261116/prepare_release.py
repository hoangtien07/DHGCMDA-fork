"""Materialize selected upstream blobs directly from the pinned git object."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
REV='06f9ba3eed92e334d9a2b8003fc06d83596e5614'
TREE=ROOT/'faithful_repro/independent_reproduction_audit_20261116/sources/github_tree.json'
REQUIRED_DATA={'D_SSM1.txt','D_SSM2.txt','M_FSM.txt','M_GSM.txt',
               'multi_all_mirna_disease_pairs_without_negative.csv'}


def main():
    tree=json.loads(TREE.read_text(encoding='utf-8'))
    assert not tree['truncated'] and tree['sha']==REV
    objects=[x for x in tree['tree'] if x['type']=='blob' and
             (('/' not in x['path'] and x['path'].endswith('.py')) or
              x['path'].startswith('v2.0_495m383D/') and x['path'].split('/')[-1] in REQUIRED_DATA)]
    assert {x['path'].split('/')[-1] for x in objects if '/' in x['path']}==REQUIRED_DATA
    target=HERE/'pinned_release';target.mkdir(exist_ok=False)
    entries=[]
    for x in objects:
        b=subprocess.check_output(['git','cat-file','blob',x['sha']],cwd=ROOT)
        gitsha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        assert gitsha==x['sha'] and len(b)==x['size']
        p=target/x['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        entries.append({'path':x['path'],'git_blob_sha1':gitsha,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
    result={'source_commit':REV,'source_tree':str(TREE.relative_to(ROOT)).replace('\\','/'),
            'git_object_command':'git cat-file blob <sha>; independently verified Git blob hashes',
            'files':entries}
    with (HERE/'source_manifest.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS','commit':REV,'blobs_verified':len(entries),'data_files':len(REQUIRED_DATA)},indent=2))


if __name__=='__main__':main()
