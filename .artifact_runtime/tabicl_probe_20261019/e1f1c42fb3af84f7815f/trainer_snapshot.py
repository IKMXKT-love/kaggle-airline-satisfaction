"""Isolated TabICLv2 download/resource probe and gated 8K/32K context screen."""
import argparse
import gc
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from report_neural import paired
from experiment_realmlp_factorial import ROOT,sha,read
from v2_training import write_json

RUNTIME=ROOT/'.artifact_runtime/tabicl_runtime'
OUT=ROOT/'.artifact_runtime/tabicl_probe_20261019'
MODEL=OUT/'tabicl-classifier-v2-20260212.ckpt'
MODEL_SHA='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
PROTOCOL=dict(context_sizes=[8192,32768],n_estimators=4,batch_size=1,kv_cache=True,
    query_batch=1024,warmup_rows=1024,probe_rows=1024,max_fit_seconds=300,max_projected_screen_seconds=1200,
    blend_weight=.2,screen_gate=.0002,seed=42,raw_features_only=True,
    base_is_current_fixed20_lean42=True,checkpoint_sha256=MODEL_SHA,
    official_repo='https://huggingface.co/jingang/TabICL',tabicl_version='2.2.0',
    reduced_context_not_full_context=True)

def download():
    OUT.mkdir(parents=True,exist_ok=True)
    if MODEL.exists():assert sha(MODEL)==MODEL_SHA;return
    url='https://huggingface.co/jingang/TabICL/resolve/main/'+MODEL.name
    print('DOWNLOAD',url,flush=True)
    tmp=MODEL.with_suffix('.partial')
    with urllib.request.urlopen(url,timeout=30) as src,tmp.open('wb') as target:
        total=0
        while block:=src.read(1024*1024):
            target.write(block);total+=len(block)
            if total%(20*1024*1024)==0:print('DOWNLOADED_MB',total//1024//1024,flush=True)
    assert sha(tmp)==MODEL_SHA,'Official LFS SHA mismatch'
    tmp.replace(MODEL)
    write_json(OUT/'checkpoint.json',dict(url=url,sha256=MODEL_SHA,bytes=MODEL.stat().st_size))
    print('DOWNLOAD_VERIFIED',MODEL.stat().st_size,flush=True)

def main():
    download();sys.path.insert(0,str(RUNTIME))
    import torch
    import psutil
    import tabicl
    from tabicl import TabICLClassifier
    import importlib.metadata
    protected=read(ROOT/'.artifact_runtime/v4/preserved.json')
    protected['submission_v4_lean_realmlp_trial.csv']=sha(ROOT/'submission_v4_lean_realmlp_trial.csv')
    hashes={n:sha(ROOT/n) for n in ['train.csv','test.csv','experiment_tabicl_probe.py']}
    hashes['checkpoint']=MODEL_SHA
    for n in ['indices','baseline']:
        hashes['reference/'+n]=sha(ROOT/'.artifact_runtime/transfer/74f731e0c4b9330ecfac/screen'/(n+'.npz'))
    hashes['lean_prediction']=sha(ROOT/'.artifact_runtime/realmlp_factorial/daf3c37e9d0734bbfbb5/lean_reference.npz')
    versions={p:importlib.metadata.version(p) for p in ['tabicl','torch','scikit-learn','numpy']}
    signature=hashlib.sha256(json.dumps([PROTOCOL,hashes,versions],sort_keys=True).encode()).hexdigest()[:20]
    directory=OUT/signature;directory.mkdir(exist_ok=True)
    write_json(directory/'protocol.json',dict(**PROTOCOL,hashes=hashes,versions=versions,protected=protected))
    (directory/'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(OUT/'latest.json',dict(directory=str(directory)))
    raw=pd.read_csv(ROOT/'train.csv');y=raw.pop('satisfaction').to_numpy();x=raw.drop(columns='id')
    reference=ROOT/'.artifact_runtime/transfer/74f731e0c4b9330ecfac/screen'
    z=np.load(reference/'indices.npz');fitidx,scored=z['fit'],z['scored']
    base=np.load(reference/'baseline.npz')['prediction'].astype('float64')
    lean=np.load(ROOT/'.artifact_runtime/realmlp_factorial/daf3c37e9d0734bbfbb5/lean_reference.npz')['prediction']
    current=.8*base+.2*lean;ys=y[scored]
    context32,_=train_test_split(fitidx,train_size=32768,stratify=y[fitidx],random_state=42)
    context8,_=train_test_split(context32,train_size=8192,stratify=y[context32],random_state=42)
    np.savez_compressed(directory/'indices.npz',context8192=context8,context32768=context32,scored=scored)
    rows=[]
    for size,idx in [(8192,context8),(32768,context32)]:
        assert not set(idx)&set(scored)
        resultpath=directory/(str(size)+'.json')
        if resultpath.exists():rows.append(read(resultpath));continue
        print('CONTEXT_START',size,flush=True)
        t=time.time();torch.cuda.reset_peak_memory_stats()
        model=TabICLClassifier(n_estimators=4,batch_size=1,kv_cache=True,model_path=str(MODEL),
            allow_auto_download=False,device='cuda',use_amp=True,use_fa3=False,offload_mode='auto',
            random_state=42,n_jobs=4,verbose=False)
        try:
            model.fit(x.iloc[idx],y[idx]);fit_seconds=time.time()-t
            warm_start=time.time();warm=model.predict_proba(x.iloc[scored[:1024]])[:,1]
            warm_seconds=time.time()-warm_start
            t=time.time();measured=model.predict_proba(x.iloc[scored[1024:2048]])[:,1];query_seconds=time.time()-t
            probe=np.concatenate([warm,measured])
            projected=query_seconds*len(scored)/1024
            d=dict(context=size,fit_seconds=fit_seconds,warmup_seconds=warm_seconds,probe_seconds=query_seconds,
                projected_query_seconds=projected,peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,
                process_rss_gb=psutil.Process().memory_info().rss/1024**3)
            print('RESOURCE_RESULT',json.dumps(d),flush=True)
            if fit_seconds>300 or projected>1200:
                d['status']='stopped_resource_budget';write_json(resultpath,d);rows.append(d);continue
            predpath=directory/(str(size)+'.npz')
            if predpath.exists():
                with np.load(predpath) as saved:
                    count=len(saved['prediction']);p=list(saved['prediction'])
                    np.testing.assert_array_equal(saved['index'],scored[:count])
            else:p=list(probe);count=2048
            query_start=time.time();runtime_stopped=False
            for start in range(count,len(scored),1024):
                values=model.predict_proba(x.iloc[scored[start:start+1024]])[:,1]
                p.extend(values)
                if len(p)%16384==0 or len(p)==len(scored):
                    np.savez_compressed(predpath,index=scored[:len(p)],prediction=np.asarray(p))
                    print('QUERY_PROGRESS',size,len(p),len(scored),flush=True)
                if time.time()-query_start>1200 and len(p)<len(scored):
                    np.savez_compressed(predpath,index=scored[:len(p)],prediction=np.asarray(p))
                    d.update(status='stopped_runtime_budget',rows_processed=len(p))
                    write_json(resultpath,d);rows.append(d);runtime_stopped=True;break
            if runtime_stopped:continue
            p=np.asarray(p);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            np.savez_compressed(predpath,index=scored,prediction=p)
            new=.8*current+.2*p
            d.update(status='screen_complete',auc=float(roc_auc_score(ys,p)),
                current_auc=float(roc_auc_score(ys,current)),blend_auc=float(roc_auc_score(ys,new)),
                paired=paired(ys,new,current),query_total_seconds=time.time()-t,
                peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3)
            write_json(resultpath,d);rows.append(d)
            print('MODEL_RESULT',json.dumps(d),flush=True)
        except (torch.OutOfMemoryError,MemoryError) as error:
            d=dict(context=size,status='stopped_memory',error=str(error));write_json(resultpath,d);rows.append(d)
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
    for n,h in protected.items():assert sha(ROOT/n)==h,n
    passing=[r for r in rows if r['status']=='screen_complete' and r['paired']['delta']>=.0002]
    write_json(directory/'summary.json',dict(rows=rows,passing=passing,protected_unchanged=True,
        status='ready_for_confirmation' if passing else 'stopped_screen_or_resource'))
    print('TABICL_DONE',bool(passing),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['download','all'],nargs='?',default='all')
    if parser.parse_args().stage=='download':download()
    else:main()
