"""Finite source-informed experiments with internal stopping and frozen gates."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from v2_training import ROOT, load_data, write_json
from v3_features import auxiliary, frames
from v3_training import fit_model
from optimized_training import fit_component
from transfer_features import TransferFeatures

OUT = ROOT/'.artifact_runtime/transfer'
SPECS = [dict(name='control',mode=None,offset=False,bins=256),
         dict(name='frequency',mode='frequency',offset=False,bins=256),
         dict(name='profiles',mode='profiles',offset=False,bins=256),
         dict(name='combined',mode='combined',offset=False,bins=256),
         dict(name='teacher_offset',mode=None,offset=True,bins=256),
         dict(name='high_resolution',mode=None,offset=False,bins=4096)]
PROTOCOL = dict(candidates=SPECS, internal_stop_fraction=.08, seed=42,
                selection_gate=.0002, confirmation_gate=.0001,
                replacement_weight=.28, weights=dict(old_xgb=.07,old_cat=.21,xgb=.28,realmlp=.14,lgb=.30),
                no_outer_early_stopping=True, all_base_branches_retrained=True,
                historical_selection_and_audit_seen=True, max_candidates=5,
                fixed_blend_weights=True, no_external_predictions=True,
                source_profile_shrinkage=20, frequency_pseudocount=1,
                full_cv_only_after_confirmation=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    OUT.mkdir(exist_ok=True)
    for n,h in read(ROOT/'.artifact_runtime/v4/preserved.json').items():
        assert sha(ROOT/n)==h,n
    protected = ['train.csv','test.csv','main.py','submission_v4.csv','submission_v4_oof.csv',
                 'v4_recipe.json','submission_v3.csv','v3_recipe.json']
    files = protected+['experiment_transfer.py','transfer_features.py','v3_features.py','v3_training.py',
                      'optimized_features.py','optimized_training.py','v2_training.py']
    hashes = {n:sha(ROOT/n) for n in files}
    signature = hashlib.sha256(json.dumps(dict(protocol=PROTOCOL,hashes=hashes),sort_keys=True).encode()).hexdigest()
    directory = OUT/signature[:20]
    directory.mkdir(exist_ok=True)
    write_json(directory/'protocol.json',dict(**PROTOCOL,signature=signature,hashes=hashes,protected=protected))
    write_json(OUT/'latest.json',dict(directory=str(directory),signature=signature))
    return directory


def fit_xgb(a,ya,b,yb,c,spec,seed=42):
    assert not {'id','satisfaction'}&set(a)
    margin = spec['offset']
    model = XGBClassifier(n_estimators=5000,learning_rate=.03,max_depth=6,min_child_weight=20,
        subsample=.9,colsample_bytree=.8,reg_lambda=10,tree_method='hist',device='cuda',
        enable_categorical=True,n_jobs=4,random_state=seed,eval_metric='auc',early_stopping_rounds=150,
        max_bin=spec['bins'])
    kwargs = dict(base_margin=a.teacher_logit.to_numpy(),
                  base_margin_eval_set=[b.teacher_logit.to_numpy()]) if margin else {}
    model.fit(a,ya,eval_set=[(b,yb)],verbose=False,**kwargs)
    model.set_params(device='cpu')
    p = model.predict_proba(c,**(dict(base_margin=c.teacher_logit.to_numpy()) if margin else {}))[:,1]
    details=dict(rounds=int(model.best_iteration+1),features=list(a),inner_stop_auc=float(model.best_score),
                 spec=spec,curves=model.evals_result())
    del model
    gc.collect()
    return p,details


def save(directory,name,index,pred,detail):
    assert len(index)==len(pred) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    np.savez_compressed(directory/(name+'.npz'),index=index,prediction=pred)
    write_json(directory/(name+'.json'),detail)


def prediction(directory,name,index):
    with np.load(directory/(name+'.npz')) as z:
        np.testing.assert_array_equal(z['index'],index)
        return z['prediction'].copy()


def run_stage(stage,chosen=None):
    root = setup();directory=root/stage;directory.mkdir(exist_ok=True)
    x,xt,y,ox,oy,prior,pt,stats,meta=load_data()
    aux,auxt=auxiliary(x,xt,ox)
    split=np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz')
    development=split['fit'] if stage=='screen' else split['development']
    scored=split['selection'] if stage=='screen' else split['audit']
    fit,stop=train_test_split(development,test_size=.08,stratify=y[development],random_state=42)
    assert not set(scored)&(set(fit)|set(stop)) and not set(fit)&set(stop)
    np.savez_compressed(directory/'indices.npz',fit=fit,stopping=stop,scored=scored)
    print('STAGE',stage,'fit',len(fit),'stop',len(stop),'scored',len(scored),flush=True)
    # All trained transformations exclude both the scored rows and stopping rows.
    a,b,c=frames(x,x.iloc[scored],y,fit,stop,prior,prior[scored],aux,aux[scored])
    transformer=TransferFeatures().fit(x.iloc[fit],ox)
    specs=SPECS if stage=='screen' else [SPECS[0],next(s for s in SPECS if s['name']==chosen)]
    for spec in specs:
        name=spec['name']
        if (directory/(name+'.npz')).exists() and (directory/(name+'.json')).exists():continue
        start=time.time();print('START',stage,name,flush=True)
        aa,bb,cc=a,b,c
        if spec['mode']:
            aa,bb,cc=[pd.concat([f,transformer.transform(raw,spec['mode'])],axis=1)
                      for f,raw in [(a,x.iloc[fit]),(b,x.iloc[stop]),(c,x.iloc[scored])]]
        p,d=fit_xgb(aa,y[fit],bb,y[stop],cc,spec)
        d.update(auc=float(roc_auc_score(y[scored],p)),seconds=time.time()-start)
        save(directory,name,scored,p,d)
        print('RESULT',stage,name,d['auc'],d['rounds'],d['seconds'],flush=True)
        del aa,bb,cc;gc.collect()
    # Rebuild all other v4 branches with the identical fit/stopping/scored partition.
    old=read(ROOT/'optimized_recipe.json')['components']
    for name,config in zip(['old_xgb','old_cat'],old):
        if (directory/(name+'.npz')).exists() and (directory/(name+'.json')).exists():continue
        start=time.time();print('START',stage,name,flush=True)
        _,p,d=fit_component(x.iloc[fit],y[fit],x.iloc[stop],y[stop],x.iloc[scored],config,stats,
                           prior[fit,None],prior[stop,None],prior[scored,None],'cuda',4)
        d.update(auc=float(roc_auc_score(y[scored],p)),seconds=time.time()-start)
        save(directory,name,scored,p,d);print('RESULT',stage,name,d['auc'],flush=True)
    if not ((directory/'lgb.npz').exists() and (directory/'lgb.json').exists()):
        import lightgbm as lgb
        print('START',stage,'lgb',flush=True);start=time.time()
        model=lgb.LGBMClassifier(n_estimators=4000,learning_rate=.025,num_leaves=63,min_child_samples=100,
            colsample_bytree=.85,subsample=.9,subsample_freq=1,reg_lambda=10,reg_alpha=.1,max_bin=255,
            verbosity=-1,n_jobs=4,random_state=42,deterministic=True,force_col_wise=True)
        model.fit(a,y[fit],eval_set=[(b,y[stop])],eval_metric='auc',callbacks=[lgb.early_stopping(150,verbose=False)])
        p=model.predict_proba(c)[:,1]
        save(directory,'lgb',scored,p,dict(auc=float(roc_auc_score(y[scored],p)),rounds=model.best_iteration_,
                                            seconds=time.time()-start))
        del model;gc.collect()
    if not ((directory/'realmlp.npz').exists() and (directory/'realmlp.json').exists()):
        print('START',stage,'realmlp',flush=True);start=time.time()
        _,p,d=fit_model(a,y[fit],b,y[stop],c,family='realmlp',seed=42)
        d.update(auc=float(roc_auc_score(y[scored],p)),seconds=time.time()-start)
        save(directory,'realmlp',scored,p,d);print('RESULT',stage,'realmlp',d['auc'],flush=True)
    control=prediction(directory,'control',scored)
    base=.28*control
    for name,w in [('old_xgb',.07),('old_cat',.21),('realmlp',.14),('lgb',.30)]:
        base+=w*prediction(directory,name,scored)
    bs=float(roc_auc_score(y[scored],base));trials=[]
    for spec in specs[1:]:
        candidate=prediction(directory,spec['name'],scored)
        blend=base+.28*(candidate-control)
        d=dict(name=spec['name'],standalone_auc=float(roc_auc_score(y[scored],candidate)),
               blend_auc=float(roc_auc_score(y[scored],blend)),gain=float(roc_auc_score(y[scored],blend))-bs)
        trials.append(d)
    save(directory,'baseline',scored,base,dict(auc=bs))
    result=dict(stage=stage,baseline_auc=bs,control_auc=float(roc_auc_score(y[scored],control)),
                trials=trials,best=max(trials,key=lambda d:d['gain']),rows=dict(fit=len(fit),stop=len(stop),scored=len(scored)),
                source=meta,scored_labels_used_for_early_stopping=False,historical_split_seen=True)
    for n in read(root/'protocol.json')['protected']:
        assert sha(ROOT/n)==read(root/'protocol.json')['hashes'][n],n
    write_json(directory/'summary.json',result)
    print('SUMMARY',json.dumps(result,ensure_ascii=False),flush=True)
    return root,result


def main(stage):
    if stage=='screen':run_stage('screen');return
    if stage=='confirmation':
        root=setup();choice=read(root/'screen/summary.json')['best']
        assert choice['gain']>=PROTOCOL['selection_gate'],'Screening gate not met'
        run_stage('confirmation',choice['name']);return
    root,result=run_stage('screen')
    if result['best']['gain']<PROTOCOL['selection_gate']:
        write_json(root/'decision.json',dict(status='stopped_screen',reason='Below fixed 0.0002 blend gain gate',best=result['best']))
        print('STOP screen gate; preserve v4',flush=True);return
    _,confirmation=run_stage('confirmation',result['best']['name'])
    passed=confirmation['best']['gain']>=PROTOCOL['confirmation_gate']
    write_json(root/'decision.json',dict(status='ready_for_cv' if passed else 'stopped_confirmation',
                                       frozen_choice=result['best'],confirmation=confirmation['best']))
    print('DECISION', 'ready_for_cv' if passed else 'stopped_confirmation',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirmation','all'],default='all',nargs='?')
    main(parser.parse_args().stage)
