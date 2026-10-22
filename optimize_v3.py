"""Frozen source-informed blend with checkpointed, matching 10-fold validation."""
import argparse
import hashlib
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from v2_training import ROOT,load_data,write_json
from v3_features import OUT,auxiliary,frames
from v3_training import fit_model


def select():
    y=pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy(dtype=int)
    old=ROOT/'.artifact_runtime/writeup_search';v=np.load(old/'split_indices.npz')['selection']
    base=.25*np.load(old/'te_original_d7_selection.npy')+.75*np.load(old/'catboost_allcat_original_d8_selection.npy')
    xgb=np.load(OUT/'xgb_selection.npy');nn=np.load(OUT/'realmlp_selection.npy')
    best=float(roc_auc_score(y[v],base));weights=[1.,0.,0.];trials=[]
    for wx in [0.,.2,.4,.6,.8]:
        for wn in [0.,.2,.4,.6,.8]:
            if wx+wn>.800001:continue
            wb=1-wx-wn;score=float(roc_auc_score(y[v],wb*base+wx*xgb+wn*nn))
            trials.append(dict(weights=[wb,wx,wn],auc=score))
            if score>best:best=score;weights=[wb,wx,wn]
    recipe=dict(weights=weights,selection_auc=best,baseline_selection_auc=float(roc_auc_score(y[v],base)),trials=trials,
        candidates={name:json.loads((OUT/(name+'_screen.json')).read_text(encoding='utf-8')) for name in ['xgb','realmlp']},
        protected=json.loads((ROOT/'.artifact_runtime/v2/snapshot/manifest.json').read_text()),
        source_url='https://www.kaggle.com/code/sachith7/s6e10-what-each-step-was-worth',
        no_external_predictions_used=True,route_profiles='fit_partition_only',target_encoding='inner5_cross_fit',
        source_auxiliary='13 public-original rating regressors excluding each predicted rating; no satisfaction labels',
        audit_not_pristine=True,no_further_tuning_after_confirmation=True)
    write_json(ROOT/'v3_recipe.json',recipe)
    print('FROZEN',weights,best,flush=True)


def setup(stage):
    recipe=json.loads((ROOT/'v3_recipe.json').read_text(encoding='utf-8'))
    for name,h in recipe['protected'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    signature=hashlib.sha256((ROOT/'v3_recipe.json').read_bytes()+b''.join((ROOT/p).read_bytes()
        for p in ['v3_features.py','v3_training.py','optimize_v3.py','optimized_features.py'])).hexdigest()
    directory=OUT/stage/signature[:20];directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'signature.json',dict(sha256=signature))
    x,xt,y,ox,oy,prior,pt,stats,meta=load_data();aux,auxt=auxiliary(x,xt,ox)
    return recipe,directory,signature,x,xt,y,prior,pt,aux,auxt,meta


def confirmation():
    recipe,directory,sig,x,xt,y,prior,pt,aux,auxt,meta=setup('confirmation')
    split=np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz');fit=split['development'];valid=split['audit']
    # Identical baseline refit was already evaluated during v2; reuse predictions, never its fitted model for these candidates.
    audit=json.loads((ROOT/'v2_confirmation.json').read_text(encoding='utf-8'));old_dir=Path(audit['checkpoint_directory'])
    old_recipe=json.loads((ROOT/'optimized_recipe.json').read_text(encoding='utf-8'))
    base=sum(cfg['weight']*np.load(old_dir/('confirm_base_'+cfg['name']+'.npy')) for cfg in old_recipe['components'])
    wb,wx,wn=recipe['weights'];pred=wb*base
    a,b=frames(x,None,y,fit,valid,prior,pt,aux,auxt)
    for family,w in [('xgb',wx),('realmlp',wn)]:
        if not w:continue
        path=directory/(family+'.npy')
        if not path.exists():
            print('CONFIRM START',family,flush=True)
            rounds=int(np.ceil(recipe['candidates'][family]['rounds']*1.1)) if family=='xgb' else None
            p,_,d=fit_model(a,y[fit],b,y[valid],family=family,rounds=rounds,audit=True)
            np.save(path,p);write_json(path.with_suffix('.json'),d)
        pred+=w*np.load(path)
    np.save(directory/'baseline.npy',base);np.save(directory/'blend.npy',pred)
    score=float(roc_auc_score(y[valid],pred));bs=float(roc_auc_score(y[valid],base))
    write_json(ROOT/'v3_confirmation.json',dict(auc=score,baseline_auc=bs,change=score-bs,directory=str(directory),signature=sig,
        previously_seen_split=True,labels_used_for_early_stopping=False,nn_internal_development_holdout_fraction=.05))
    print('CONFIRM RESULT',score,bs,score-bs,flush=True)


def cv():
    recipe,directory,sig,x,xt,y,prior,pt,aux,auxt,meta=setup('cv')
    old=pd.read_csv(ROOT/'submission_optimized_oof.csv');oldp=pd.read_csv(ROOT/'submission_optimized.csv')
    wb,wx,wn=recipe['weights'];oof=wb*old.prediction.to_numpy();pred=wb*oldp.satisfaction.to_numpy();folds=np.zeros(len(y),int)
    scores=[];details=[]
    for fold,(fit,valid) in enumerate(StratifiedKFold(10,shuffle=True,random_state=42).split(x,y),1):
        np.testing.assert_array_equal(valid,np.flatnonzero(old.fold.to_numpy()==fold));folds[valid]=fold
        a=b=c=None
        for family,w in [('xgb',wx),('realmlp',wn)]:
            if not w:continue
            path=directory/f'fold_{fold:02d}_{family}.npz';js=path.with_suffix('.json')
            if not path.exists() or not js.exists():
                if a is None:a,b,c=frames(x,xt,y,fit,valid,prior,pt,aux,auxt)
                print('CV START',fold,family,flush=True)
                pv,ps,d=fit_model(a,y[fit],b,y[valid],c,family,seed=42+fold)
                np.savez_compressed(path,valid_index=valid,validation_prediction=pv,test_prediction=ps)
                write_json(js,d);print('CV RESULT',fold,family,d['auc'],flush=True)
            with np.load(path) as z:
                np.testing.assert_array_equal(z['valid_index'],valid)
                oof[valid]+=w*z['validation_prediction'];pred+=w*z['test_prediction']/10
            details.append(dict(fold=fold,**json.loads(js.read_text(encoding='utf-8'))))
        del a,b,c;gc.collect()
        score=float(roc_auc_score(y[valid],oof[valid]));scores.append(score);print('CV BLEND',fold,score,flush=True)
    train=pd.read_csv(ROOT/'train.csv',usecols=['id']);test=pd.read_csv(ROOT/'test.csv',usecols=['id'])
    pd.DataFrame(dict(id=test.id,satisfaction=pred)).to_csv(ROOT/'submission_v3.csv',index=False)
    pd.DataFrame(dict(id=train.id,target=y,fold=folds,prediction=oof)).to_csv(ROOT/'submission_v3_oof.csv',index=False)
    old_scores=[float(roc_auc_score(y[folds==i],old.prediction[folds==i])) for i in range(1,11)]
    result=dict(n_splits=10,mean_auc=float(np.mean(scores)),std_auc=float(np.std(scores)),oof_auc=float(roc_auc_score(y,oof)),
        baseline_oof_auc=float(roc_auc_score(y,old.prediction)),fold_scores=scores,baseline_fold_scores=old_scores,
        improved_folds=int((np.array(scores)>old_scores).sum()),directory=str(directory),signature=sig,components=details,
        original_data=meta,kaggle_score=None,submitted=False,validation_note='Same folds, overlapping selection rows; early stopping on outer CV; old audit previously seen.')
    write_json(ROOT/'submission_v3_metrics.json',result);print('COMPLETE',result['oof_auc'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','confirmation','cv']);args=p.parse_args();globals()[args.stage]()
