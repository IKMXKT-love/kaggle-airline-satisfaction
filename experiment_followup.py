"""Frozen RealMLP replication and six fixed blend comparisons, 2026-10-19."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from experiment_realmlp_factorial import ROOT,LeanFeatures,REFERENCE_PARAMS,sha,read
from v2_training import load_data,write_json
from v3_features import auxiliary,frames
from v3_training import fit_model
from optimized_training import fit_component
from experiment_transfer import fit_xgb,SPECS
from report_neural import paired

REFERENCE=ROOT/'.artifact_runtime/transfer/74f731e0c4b9330ecfac'
FACTORIAL=ROOT/'.artifact_runtime/realmlp_factorial/daf3c37e9d0734bbfbb5'
OUT=ROOT/'.artifact_runtime/followup_20261019'
WEIGHTS=dict(old_xgb=.07,old_cat=.21,control=.28,realmlp=.14,lgb=.30)
PROTOCOL=dict(seeds=[42,20261019],params=REFERENCE_PARAMS,weights=WEIGHTS,
    candidate_weights=[.10,.20,.35],forms=['probability','logit'],max_fusion_trials=6,
    confirmation_min_vs_base=.0001,incremental_min_vs_current=.00005,
    historical_partitions_seen=True,no_external_score_weight_search=True,
    baseline_retrained_on_matched_fit_stop_rows=True,fit_stop_split_seed=42)

def blend(base,p,w,form):
    if form=='probability':return (1-w)*base+w*p
    assert form=='logit'
    return expit((1-w)*logit(np.clip(base,1e-6,1-1e-6))+w*logit(np.clip(p,1e-6,1-1e-6)))

def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=read(ROOT/'.artifact_runtime/v4/preserved.json')
    for n in ['submission_v4_lean_realmlp_trial.csv','submission_v4_lean_realmlp_trial.json',
              'submission_v4_lean_realmlp_trial_public_result.json']:
        protected[n]=sha(ROOT/n)
    for n,h in protected.items():assert sha(ROOT/n)==h,n
    files=['experiment_followup.py','experiment_realmlp_factorial.py','v2_training.py','v3_features.py',
        'v3_training.py','optimized_training.py','optimized_features.py','experiment_transfer.py',
        'train.csv','test.csv','external_data/train.csv','external_data/test.csv','optimized_recipe.json',
        '.artifact_runtime/writeup_search/split_indices.npz','.artifact_runtime/v3/auxiliary.json',
        '.artifact_runtime/v3/aux_train.npy','.artifact_runtime/v3/aux_test.npy',
        '.artifact_runtime/optimized_cache/train_priors.npy','.artifact_runtime/optimized_cache/test_priors.npy']
    hashes={n:sha(ROOT/n) for n in files}
    for prefix,folder,names in [('reference',REFERENCE/'screen',list(WEIGHTS)+['baseline','indices']),
                               ('factorial',FACTORIAL,['lean_reference','indices'])]:
        for n in names:hashes[prefix+'/'+n+'.npz']=sha(folder/(n+'.npz'))
    import importlib.metadata
    versions={p:importlib.metadata.version(p) for p in ['torch','pytabkit','numpy','pandas','scikit-learn','xgboost','catboost','lightgbm']}
    signature=hashlib.sha256(json.dumps([PROTOCOL,hashes,versions],sort_keys=True).encode()).hexdigest()[:20]
    directory=OUT/signature;directory.mkdir(exist_ok=True)
    write_json(directory/'protocol.json',dict(**PROTOCOL,hashes=hashes,protected=protected,versions=versions))
    (directory/'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(OUT/'latest.json',dict(directory=str(directory)))
    return directory

def save(folder,name,idx,p,d):
    assert len(idx)==len(p) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    np.savez_compressed(folder/(name+'.npz'),index=idx,prediction=p)
    write_json(folder/(name+'.json'),d)

def pred(folder,name,idx):
    with np.load(folder/(name+'.npz')) as z:
        np.testing.assert_array_equal(idx,z['index']);return z['prediction'].copy()

def fit_lean(a,ya,b,yb,c,cats,seed):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    start=time.time();torch.cuda.reset_peak_memory_stats()
    model=RealMLP_TD_Classifier(device='cuda',random_state=seed,n_cv=1,n_refit=0,
        n_threads=4,verbosity=1,val_metric_name='1-auc_ovr',**REFERENCE_PARAMS)
    model.fit(a,ya,cat_col_names=cats,X_val=b,y_val=yb)
    p=model.predict_proba(c)[:,1]
    d=dict(seconds=time.time()-start,seed=seed,features=list(a),fixed_stop_epoch=3,
           peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3)
    del model;gc.collect();torch.cuda.empty_cache()
    return p,d

def baseline(folder,stage,x,y,prior,aux,stats,fitidx,stop,scored):
    a,b,c=frames(x,x.iloc[scored],y,fitidx,stop,prior,prior[scored],aux,aux[scored])
    old=read(ROOT/'optimized_recipe.json')['components']
    for name in WEIGHTS:
        if (folder/(name+'.npz')).exists():continue
        start=time.time();print('BASE_START',stage,name,flush=True)
        if name=='control':p,d=fit_xgb(a,y[fitidx],b,y[stop],c,SPECS[0])
        elif name in ['old_xgb','old_cat']:
            config=old[['old_xgb','old_cat'].index(name)]
            _,p,d=fit_component(x.iloc[fitidx],y[fitidx],x.iloc[stop],y[stop],x.iloc[scored],config,stats,
                prior[fitidx,None],prior[stop,None],prior[scored,None],'cuda',4)
        elif name=='realmlp':_,p,d=fit_model(a,y[fitidx],b,y[stop],c,family='realmlp',seed=42)
        else:
            import lightgbm as lgb
            model=lgb.LGBMClassifier(n_estimators=4000,learning_rate=.025,num_leaves=63,min_child_samples=100,
                colsample_bytree=.85,subsample=.9,subsample_freq=1,reg_lambda=10,reg_alpha=.1,max_bin=255,
                verbosity=-1,n_jobs=4,random_state=42,deterministic=True,force_col_wise=True)
            model.fit(a,y[fitidx],eval_set=[(b,y[stop])],eval_metric='auc',callbacks=[lgb.early_stopping(150,verbose=False)])
            p=model.predict_proba(c)[:,1];d=dict(rounds=model.best_iteration_)
            del model;gc.collect()
        d.update(auc=float(roc_auc_score(y[scored],p)),seconds=time.time()-start)
        save(folder,name,scored,p,d);print('BASE_RESULT',stage,name,d['auc'],d['seconds'],flush=True)
    base=sum(w*pred(folder,n,scored).astype('float64') for n,w in WEIGHTS.items())
    save(folder,'baseline',scored,base,dict(auc=float(roc_auc_score(y[scored],base))))
    return base

def run(stage,choice=None):
    directory=setup();folder=directory/stage;folder.mkdir(exist_ok=True)
    x,xt,y,ox,oy,prior,pt,stats,meta=load_data();aux,auxt=auxiliary(x,xt,ox)
    split=np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz')
    if stage=='screen':
        z=np.load(REFERENCE/'screen/indices.npz');fitidx,stop,scored=[z[k].copy() for k in ['fit','stopping','scored']]
        for n in list(WEIGHTS)+['baseline']:
            if not (folder/(n+'.npz')).exists():
                p=pred(REFERENCE/'screen',n,scored)
                save(folder,n,scored,p,dict(reused_from=str(REFERENCE/'screen'/(n+'.npz'))))
        if not (folder/'lean_42.npz').exists():
            p=pred(FACTORIAL,'lean_reference',scored)
            save(folder,'lean_42',scored,p,dict(reused_from=str(FACTORIAL/'lean_reference.npz')))
        base=pred(folder,'baseline',scored).astype('float64')
    else:
        scored=split['audit'];fitidx,stop=train_test_split(split['development'],test_size=.08,stratify=y[split['development']],random_state=42)
        base=None
    assert not(set(fitidx)&set(stop) or set(scored)&(set(fitidx)|set(stop)))
    np.savez_compressed(folder/'indices.npz',fit=fitidx,stopping=stop,scored=scored)
    print('STAGE',stage,len(fitidx),len(stop),len(scored),flush=True)
    tf=LeanFeatures();a=tf.fit_transform(x.iloc[fitidx],y[fitidx]);b=tf.transform(x.iloc[stop]);c=tf.transform(x.iloc[scored])
    for seed in PROTOCOL['seeds']:
        name='lean_'+str(seed)
        if (folder/(name+'.npz')).exists():continue
        print('LEAN_START',stage,seed,flush=True)
        p,d=fit_lean(a,y[fitidx],b,y[stop],c,tf.cats,seed)
        d['auc']=float(roc_auc_score(y[scored],p));save(folder,name,scored,p,d)
        print('LEAN_RESULT',stage,seed,d['auc'],d['seconds'],flush=True)
    del a,b,c;gc.collect()
    if base is None:base=baseline(folder,stage,x,y,prior,aux,stats,fitidx,stop,scored)
    p1=pred(folder,'lean_42',scored).astype('float64');p2=pred(folder,'lean_20261019',scored).astype('float64')
    avg=(p1+p2)/2;save(folder,'lean_mean',scored,avg,dict(seeds=PROTOCOL['seeds']))
    anchor=blend(base,p1,.2,'probability');ys=y[scored]
    trials=[]
    specs=[dict(weight=w,form=f) for w in PROTOCOL['candidate_weights'] for f in PROTOCOL['forms']] if stage=='screen' else [choice]
    for spec in specs:
        p=blend(base,avg,spec['weight'],spec['form'])
        trials.append(dict(**spec,auc=float(roc_auc_score(ys,p)),vs_base=paired(ys,p,base),vs_current=paired(ys,p,anchor)))
    best=max(trials,key=lambda r:r['auc'])
    result=dict(stage=stage,baseline_auc=float(roc_auc_score(ys,base)),current_auc=float(roc_auc_score(ys,anchor)),
        seeds={str(seed):dict(auc=float(roc_auc_score(ys,p)),fixed20_vs_base=paired(ys,blend(base,p,.2,'probability'),base))
               for seed,p in [(42,p1),(20261019,p2)]},
        mean_fixed20_vs_single=paired(ys,blend(base,avg,.2,'probability'),anchor),trials=trials,best=best)
    write_json(folder/'summary.json',result)
    for n,h in read(directory/'protocol.json')['protected'].items():assert sha(ROOT/n)==h,n
    print('STAGE_RESULT',stage,'base',result['baseline_auc'],'anchor',result['current_auc'],'best',best,flush=True)
    return directory,result

def main(stage):
    if stage=='screen':run('screen');return
    if stage=='confirmation':
        directory=setup();s=read(directory/'screen/summary.json');run('confirmation',{k:s['best'][k] for k in ['weight','form']});return
    directory,s=run('screen');choice={k:s['best'][k] for k in ['weight','form']}
    write_json(directory/'frozen_choice.json',choice)
    _,c=run('confirmation',choice)
    stable=all(c['seeds'][str(seed)]['fixed20_vs_base']['delta']>0 for seed in PROTOCOL['seeds'])
    stable=stable and c['seeds']['42']['fixed20_vs_base']['delta']>=.0001
    improve=(s['best']['vs_current']['delta']>=.00005 and c['best']['vs_current']['delta']>=.00005
             and c['best']['vs_base']['delta']>=.0001)
    decision=dict(current_recipe_stable=stable,frozen_fusion_improves_current=improve,
        full_cv_warranted=stable,chosen_fusion=choice if improve else dict(weight=.2,form='probability'),
        seed_average_incremental_confirmation=c['mean_fixed20_vs_single'],
        note='Historical split already seen; not a pristine holdout. Full CV is a separate next phase.')
    write_json(directory/'decision.json',decision)
    print('DECISION',json.dumps(decision),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen','confirmation','all'],nargs='?',default='all')
    main(p.parse_args().stage)
