import json,time
from pathlib import Path
P=Path(__file__).resolve().parent
syskeys=[f'fold{f}_seed{s}_{a}' for f in range(5) for s in (1234,42,2025) for a in ('dynamic','static')];done=[];pending=[]
for k in syskeys:
 d=P/'runs'/k
 if (d/'completion.json').exists():done.append(k)
 elif d.exists():
  q=d/'progress.log';tail=q.read_text(encoding='utf-8').splitlines()[-1] if q.exists() and q.stat().st_size else 'no progress log';pending.append((k,tail))
lines=['# Trạng thái outer-refit','',f'**Tiến độ plan:** {len(done)}/30 refit hoàn tất; test chưa được chấm.' if not (P/'evaluation/results.json').exists() else f'**Tiến độ plan:** {len(done)}/30 refit hoàn tất; đã có kết quả outer-test được kiểm tra.','',time.strftime('Cập nhật %Y-%m-%d %H:%M:%S UTC',time.gmtime()),'','Các thư mục đã tạo nhưng chưa có completion (không suy đoán tiến trình còn sống):']
lines += ['- '+k+': '+tail for k,tail in pending] or ['- Không có.']
lines+=['', 'Dừng lỗi: '+('có' if (P/'STOP_ON_FAILURE').exists() else 'chưa thấy'), 'Seal test: '+('đã có' if (P/'seal.json').exists() else 'chưa có'), 'Supervisor kết quả: '+('hoàn tất' if (P/'SUPERVISOR_COMPLETE.json').exists() else 'chưa hoàn tất'), '', 'Protocol và provenance: `PROTOCOL.md`, `preflight.json`, `bundle_audit.json`, `prior_manifest.json`.']
(P/'STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(lines[2]);print('\n'.join(lines[6:6+len(pending)]))
