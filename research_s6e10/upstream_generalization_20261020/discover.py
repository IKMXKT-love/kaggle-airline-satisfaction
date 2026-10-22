"""Anonymous, bounded metadata discovery. Never executes downloaded code."""
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')


def fetch(tag, url, payload=None, limit=16_000_000, deadline_s=50):
    start = time.monotonic()
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={
        'User-Agent': 'kaggle-api/v1.7.0', 'Content-Type': 'application/json'})
    try:
        chunks, size = [], 0
        with urllib.request.urlopen(request, timeout=12) as response:
            while True:
                if time.monotonic()-start > deadline_s:
                    raise TimeoutError('whole-file deadline exceeded')
                chunk = response.read1(256*1024)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > limit:
                    raise ValueError('size limit exceeded')
        raw = b''.join(chunks)
        path = ROOT / (tag+'.json')
        path.write_bytes(raw)
        return {'tag': tag, 'url': url, 'bytes': size,
                'sha256': hashlib.sha256(raw).hexdigest(), 'elapsed': time.monotonic()-start}
    except Exception as e:
        return {'tag': tag, 'url': url, 'error': type(e).__name__+': '+str(e),
                'elapsed': time.monotonic()-start}


def discover():
    jobs = []
    for page in [2, 3, 4]:
        jobs.append((f's6e10_page{page}', 'https://www.kaggle.com/api/v1/datasets/list?'+urllib.parse.urlencode(
            {'search': 's6e10', 'sortBy': 'updated', 'page': page})))
    for term in ['airline oof', 'ps6e10', 's6 e10', 'airline satisfaction predictions']:
        jobs.append((term.replace(' ', '_'), 'https://www.kaggle.com/api/v1/datasets/list?'+urllib.parse.urlencode(
            {'search': term, 'sortBy': 'updated', 'page': 1})))
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        receipts = list(pool.map(lambda x: fetch(*x), jobs))
    (ROOT/'discovery_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    for r in receipts:
        print(json.dumps(r,ensure_ascii=False))
        if 'error' not in r:
            rows=json.loads((ROOT/(r['tag']+'.json')).read_text('utf-8'))
            if isinstance(rows,list):
                for x in rows:
                    print(x.get('ref'), x.get('lastUpdated'), x.get('totalBytes'), x.get('title'))


if __name__ == '__main__':
    discover()
