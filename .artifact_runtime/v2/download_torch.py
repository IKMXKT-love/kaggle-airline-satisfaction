import urllib.request,concurrent.futures,hashlib,time,json
from pathlib import Path
p=Path('.artifact_runtime/v2/torch-2.11.0+cu128-cp313-cp313-win_amd64.whl');n=2753196039;step=16*1024*1024
starts=list(range(0,n,step));missing=[]
with p.open('rb') as f:
 for start in starts:
  f.seek(start)
  if f.read(16)==bytes(16):missing.append(start)
print('RESUME missing',len(missing),flush=True)
def chunk(start):
 end=min(n,start+step)-1
 for attempt in range(12):
  host='download.pytorch.org' if attempt%2==0 else 'download-r2.pytorch.org'
  u=f'https://{host}/whl/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl'
  try:
   t=time.time();req=urllib.request.Request(u,headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=25) as r:
    assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{n}'
    payload=r.read()
   assert len(payload)==end-start+1
   with p.open('r+b') as f:f.seek(start);f.write(payload)
   return start,time.time()-t
  except Exception as exc:
   if attempt==11:raise
   if attempt in [3,7]:print('RETRY',start//step,attempt,type(exc).__name__,flush=True)
with concurrent.futures.ThreadPoolExecutor(12) as ex:
 futures=[ex.submit(chunk,start) for start in missing]
 for i,f in enumerate(concurrent.futures.as_completed(futures),1):
  start,elapsed=f.result()
  if i%5==0:print('Completed',len(starts)-len(missing)+i,'/',len(starts),'last_seconds',round(elapsed,1),flush=True)
h=hashlib.sha256()
with p.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
print('SHA256',h.hexdigest(),flush=True)
assert h.hexdigest()=='6f367e62fd81b75cdf23ca4b75ced834d2db2cf98d1588ac935bde345de9de23'
print('VERIFIED',flush=True)
