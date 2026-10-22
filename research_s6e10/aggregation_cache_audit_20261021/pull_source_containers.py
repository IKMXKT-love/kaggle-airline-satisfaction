"""Bounded source-container audit. Downloads data only; never executes remote code."""
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
spec = importlib.util.spec_from_file_location('bounded_fetch',ROOT.parent/'upstream_generalization_20261020/discover.py')
discovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discovery)
discovery.ROOT = ROOT
sys.stdout.reconfigure(encoding='utf-8')


def get(name):
    url = 'https://www.kaggle.com/api/v1/datasets/download/sachith7/s6e10-what-each-step-was-worth/' + name
    # Resolve actual dataset from captured metadata, rather than the notebook slug.
    meta=json.loads((PROJECT/'.artifact_runtime/public_pool_trial_20261020/sachith_version_metadata.json').read_text('utf-8'))
    url='https://www.kaggle.com/api/v1/datasets/download/'+meta['ref']+'/'+name
    tag='source_'+name.replace('.','_')
    receipt=discovery.fetch(tag,url,limit=12_000_000,deadline_s=50)
    if 'error' in receipt:
        return receipt
    raw=(ROOT/(tag+'.json')).read_bytes()
    # Kaggle can return a ZIP wrapper. NPZ is itself ZIP, so unwrap only an
    # exact basename match to the requested container, never its .npy entries.
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            matching=[x for x in archive.namelist() if Path(x).name==name]
            if len(matching)==1:
                info=archive.getinfo(matching[0])
                if info.file_size>12_000_000:
                    raise ValueError('Unwrapped file exceeds limit.')
                raw=archive.read(matching[0])
    destination=ROOT/name
    if destination.exists() and destination.read_bytes()!=raw:
        raise FileExistsError('Preserve previous source snapshot.')
    destination.write_bytes(raw)
    receipt.update(normalized_file=name,normalized_sha256=hashlib.sha256(raw).hexdigest(),normalized_bytes=len(raw))
    return receipt


if __name__=='__main__':
    (ROOT/'scope.json').write_text(json.dumps({
        'date':'2026-10-21',
        'purpose':'check whether existing/public paired members contain repeated held-out predictions or only OOF/test aggregates',
        'source_code_execution':False,'new_fits':0,'new_base_inferences':0,'new_submissions':0,
        'rule':'no model fitting unless aggregation-matched validation boundary and complete bounded cost can be established',
    },indent=2)+'\n',encoding='utf-8')
    names=['members.json','rmlp_v4aux_k10.npz','rmlp_v4aux_k10s2.npz']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipts=list(pool.map(get,names))
    (ROOT/'source_download_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    for item in receipts:
        print(json.dumps(item,ensure_ascii=False))
