"""Read public arrays; never import or execute their producer's code."""
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.stdout.reconfigure(encoding='utf-8')
START = time.monotonic()
DEADLINE = START + 900


def save(name, data):
    (HERE/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def download(item):
    name = item['fileName']
    dest = HERE/'kumo_parts'/Path(name).name
    dest.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    if not dest.exists():
        req = urllib.request.Request(item['url'], headers={'User-Agent': 'kaggle-api/v1.7.0'})
        with urllib.request.urlopen(req, timeout=12) as response:
            chunks, size = [], 0
            while True:
                if time.monotonic() > min(DEADLINE, started+50):
                    raise TimeoutError('download deadline exceeded')
                chunk = response.read1(256*1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 8_000_000:
                    raise ValueError('unexpected array size')
                chunks.append(chunk)
        partial = dest.with_suffix('.part')
        partial.write_bytes(b''.join(chunks))
        partial.replace(dest)
    return {'file': name, 'bytes': dest.stat().st_size,
            'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
            'seconds': time.monotonic()-started}


def main():
    protocol = json.loads((HERE/'kumo_protocol.json').read_text('utf-8'))
    output = json.loads((HERE/'ewtherka__s6e10-tabfm-trying-to-be-better__outputs_v14.json').read_text('utf-8'))
    tag = 'kumolarge_fsE_c30000_e4_o0'
    wanted = {f'tfm_parts/{tag}_b{b}_{part}.npy' for b in range(3)
              for part in ['f0','f1','f2','f3','f4','test']}
    items = [f for f in output['files'] if f['fileName'] in wanted]
    assert {f['fileName'] for f in items} == wanted
    save('kumo_screen_start.json', {'started_epoch': time.time(), 'protocol_sha256': hashlib.sha256((HERE/'kumo_protocol.json').read_bytes()).hexdigest()})
    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(download, items):
            receipts.append(r)
    save('kumo_download_receipts.json', receipts)
    logs = output.get('log', '[]')
    logs = json.loads(logs) if isinstance(logs,str) else logs
    (HERE/'kumo_v14_log.txt').write_text(''.join(x.get('data','') for x in logs),encoding='utf-8')

    train = pd.read_csv(ROOT/'train.csv', usecols=['id','satisfaction'])
    test_ids = pd.read_csv(ROOT/'test.csv', usecols=['id']).id.to_numpy()
    y = train.satisfaction.to_numpy()
    if y.dtype == object:
        y = train.satisfaction.astype(str).str.lower().isin(['true','1','satisfied']).astype(int).to_numpy()
    else:
        y = y.astype(int)
    folds = list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    coverage = np.zeros(len(y), dtype=np.int8)
    for _, v in folds:
        coverage[v] += 1
    assert np.all(coverage==1)
    bags, tests = [], []
    for b in range(3):
        oof = np.full(len(y), np.nan)
        for k, (_, v) in enumerate(folds):
            part = np.load(HERE/'kumo_parts'/f'{tag}_b{b}_f{k}.npy', allow_pickle=False)
            assert part.shape == (len(v),)
            oof[v] = part
        test = np.load(HERE/'kumo_parts'/f'{tag}_b{b}_test.npy',allow_pickle=False)
        assert test.shape == (len(test_ids),)
        for a in [oof,test]:
            assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
        bags.append(oof); tests.append(test)
    candidate = np.mean(bags,axis=0)
    candidate_test = np.mean(tests,axis=0)
    np.savez(HERE/'kumo_threebag_predictions.npz',id=train.id.to_numpy(),oof=candidate,test_id=test_ids,test=candidate_test)
    cache = ROOT/'.artifact_runtime/public_pool_trial_20261020'
    with np.load(cache/'selected_predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'],train.id)
        np.testing.assert_array_equal(z['test_id'],test_ids)
        base = z['oof_logit']
    manifest = json.loads((cache/'matrix_manifest.json').read_text('utf-8'))
    cols = manifest['columns'][:246]
    X = np.load(cache/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    raw_idx = next(i for i,x in enumerate(cols) if x['member']=='tfm_kumolarge')
    raw = X[:,raw_idx]
    best = roc_auc_score(y,base)
    base_fold = [roc_auc_score(y[v],base[v]) for _,v in folds]
    results = []
    for name, score in [('kumo_engineered_3bag', logit(np.clip(candidate,*protocol['probability_clip']))),('kumo_raw_existing',raw)]:
        for w in [0.05,0.2]:
            pred = (1-w)*base+w*score
            delta = [roc_auc_score(y[v],pred[v])-base_fold[k] for k,(_,v) in enumerate(folds)]
            results.append({'candidate':name,'weight':w,'auc':roc_auc_score(y,pred),
                            'delta':roc_auc_score(y,pred)-best,'fold_deltas':delta,
                            'folds_positive':sum(x>0 for x in delta),
                            'mean_fold_delta':float(np.mean(delta)),
                            'paired_fold_se':float(np.std(delta,ddof=1)/np.sqrt(5))})
    correlations = []
    # Fixed ID-free deterministic subsample for redundancy description; no outcome selection.
    idx = np.random.default_rng(12345).choice(len(y),size=50000,replace=False)
    for j,col in enumerate(cols):
        correlations.append({'member':col['member'],'spearman':float(spearmanr(candidate[idx],X[idx,j]).statistic)})
    correlations.sort(key=lambda x:x['spearman'],reverse=True)
    report = {'baseline_auc':best,'standalone_candidate_auc':roc_auc_score(y,candidate),
              'standalone_raw_auc':roc_auc_score(y,raw),
              'bag_aucs':[roc_auc_score(y,p) for p in bags],
              'candidate_fold_aucs':[roc_auc_score(y[v],candidate[v]) for _,v in folds],
              'contrasts':results,'top_correlations_50k_fixed_sample':correlations[:10],
              'source_alignment':'audited producer original-row-order fold assembly; no independent ID sidecars',
              'seconds_including_download':time.monotonic()-START,'new_model_fits':0,'new_submission_generated':False}
    save('kumo_contrasts.json',report)
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
