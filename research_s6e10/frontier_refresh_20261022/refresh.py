"""Read-only metadata/source discovery. No training or remote code execution."""
import concurrent.futures
import importlib.util
import json
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('discovery', ROOT.parent/'upstream_generalization_20261020/discover.py')
discovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discovery)
discovery.ROOT = ROOT
sys.stdout.reconfigure(encoding='utf-8')

def main():
    jobs = []
    for page in range(1, 5):
        jobs.append((f'datasets_page{page}', 'https://www.kaggle.com/api/v1/datasets/list?' + urllib.parse.urlencode({'search':'s6e10','sortBy':'updated','page':page})))
    for term in ['Q-FT', 'Qwen airline', 'airline oof']:
        jobs.append(('search_'+term.replace(' ','_'), 'https://www.kaggle.com/api/v1/datasets/list?' + urllib.parse.urlencode({'search':term,'sortBy':'updated'})))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts=list(pool.map(lambda j:discovery.fetch(*j),jobs))
    (ROOT/'receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    unique={}
    for r in receipts:
        if 'error' in r:
            print(r)
            continue
        data=json.loads((ROOT/(r['tag']+'.json')).read_text('utf-8'))
        if not isinstance(data,list):
            print(r['tag'],str(data)[:200]); continue
        print(r['tag'],'count',len(data))
        for row in data:
            unique[row['ref']]={k:row.get(k) for k in ['ref','currentVersionNumber','lastUpdated','totalBytes','title','subtitle']}
    manifest=json.loads((ROOT.parents[1]/'.artifact_runtime/public_pool_trial_20261020/matrix_manifest.json').read_text('utf-8'))
    retained={x['source_ref']:x.get('captured_source_version') for x in manifest['columns'] if 'source_ref' in x}
    for row in unique.values():
        row['retained_version']=retained.get(row['ref'])
        print(json.dumps(row,ensure_ascii=False))
    (ROOT/'catalog.json').write_text(json.dumps(list(unique.values()),indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
