import hashlib,json,urllib.request,datetime
from pathlib import Path
P=Path('faithful_repro/independent_v32_author_review_20261117');base='https://api.github.com/repos/CDMBlab/DHGCMDA/';out={}
for endpoint in ('branches?per_page=100','tags?per_page=100','releases?per_page=100','issues?state=all&per_page=100'):
 req=urllib.request.Request(base+endpoint,headers={'User-Agent':'DHGCMDA-read-only-provenance-audit','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(req,timeout=30) as r:raw=r.read();obj=json.loads(raw);assert isinstance(obj,list);out[endpoint]={'count':len(obj),'response_sha256':hashlib.sha256(raw).hexdigest(),'items':[{'name':x.get('name'),'sha':x.get('commit',{}).get('sha'),'title':x.get('title'),'url':x.get('html_url')} for x in obj]}
record={'status':'PASS','retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'api':out}
with (P/'repository_inventory.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
