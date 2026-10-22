"""Bounded selection, frozen confirmation, and resumable 10-fold second-round training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from optimized_training import fit_component
from v2_training import (ROOT,OUT,write_json,load_data,make_cat_frames,fit_cat,source_prior)


def select():
    y=pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy(dtype=int)
    valid=np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz')['selection']
    old=ROOT/'.artifact_runtime/writeup_search'
    base=.25*np.load(old/'te_original_d7_selection.npy')+.75*np.load(old/'catboost_allcat_original_d8_selection.npy')
    cats=json.loads((OUT/'cat_screen.json').read_text(encoding='utf-8'))
    best=max(cats,key=lambda r:r['auc'])
    cat=np.load(OUT/(best['config']['name']+'_selection.npy'))
    nn_records=[json.loads((OUT/(n+'.json')).read_text(encoding='utf-8')) for n in ['tabm_ple','tabm_compact']]
    best_nn=max(nn_records,key=lambda d:d['auc'])
    nn=np.load(OUT/(best_nn['config']['name']+'_selection.npy'))
    trials=[];best_score=roc_auc_score(y[valid],base);weights=[1.,0.,0.]
    # Small, predeclared convex grid. Existing blend always retains >=20% weight.
    for wc in [0.,.2,.4,.6,.8]:
        for wn in [0.,.2,.4,.6,.8]:
            if wc+wn>.800001:continue
            wb=1-wc-wn;score=float(roc_auc_score(y[valid],wb*base+wc*cat+wn*nn))
            trials.append(dict(weights=[wb,wc,wn],auc=score))
            if score>best_score:
                best_score=score;weights=[wb,wc,wn]
    recipe=dict(weights=weights,selection_auc=best_score,baseline_selection_auc=float(roc_auc_score(y[valid],base)),
        cat_config=best['config'],cat_rounds=best['rounds'],nn=best_nn,nn_candidates=nn_records,
        source='selection_only',trials=trials,confirmation_not_pristine=True,
        data_hashes=json.loads((OUT/'snapshot/manifest.json').read_text()),
        protocol=json.loads((OUT/'protocol.json').read_text()))
    write_json(ROOT/'v2_recipe.json',recipe)
    print('FROZEN',weights,best_score,'baseline',recipe['baseline_selection_auc'],flush=True)


def load_recipe():
    recipe=json.loads((ROOT/'v2_recipe.json').read_text(encoding='utf-8'))
    for name,digest in recipe['data_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return recipe


def confirmation():
    recipe=load_recipe();x,xt,y,ox,oy,prior,pt,stats,meta=load_data()
    confirm_signature=hashlib.sha256((ROOT/'v2_recipe.json').read_bytes()+b''.join(
        (ROOT/p).read_bytes() for p in ['v2_training.py','v2_tabm.py','optimized_training.py','optimized_features.py'])).hexdigest()
    confirm_dir=OUT/'confirmation'/confirm_signature[:20]
    confirm_dir.mkdir(parents=True,exist_ok=True)
    split=np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz')
    fit,valid=split['development'],split['audit']
    wb,wc,wn=recipe['weights'];base=np.zeros(len(valid));old=json.loads((ROOT/'optimized_recipe.json').read_text(encoding='utf-8'))
    prior2=np.load(ROOT/'.artifact_runtime/optimized_cache/train_priors.npy')
    for cfg in old['components']:
        path=confirm_dir/('confirm_base_'+cfg['name']+'.npy')
        if not path.exists():
            print('CONFIRM BASE',cfg['name'],flush=True)
            p,_,d=fit_component(x.iloc[fit],y[fit],x.iloc[valid],y[valid],None,cfg,stats,
                prior2[fit],prior2[valid],None,'cuda',8,max_rounds=int(np.ceil(cfg['best_rounds']*1.1)),early_stopping=False)
            np.save(path,p)
        base+=cfg['weight']*np.load(path)
    prediction=wb*base
    for weight,name in [(wc,'cat'),(wn,'nn')]:
        if weight==0:continue
        path=confirm_dir/('confirm_'+name+'.npy')
        if not path.exists():
            print('CONFIRM',name,flush=True)
            if name=='cat':
                src=source_prior(x,xt,ox,oy) if recipe['cat_config'].get('source_prior') else None
                a,b,c,cats=make_cat_frames(x,None,y,fit,valid,stats,prior,pt,src)
                p,_,d=fit_cat(a,y[fit],b,y[valid],None,cats,recipe['cat_config'],rounds=int(np.ceil(recipe['cat_rounds']*1.1)))
            else:
                from v2_tabm import frames,fit_tabm
                a,b,c=frames(x,None,y,fit,valid,stats,prior,pt,raw_only=recipe['nn']['config'].get('raw_only',False))
                p,_,d=fit_tabm(a,y[fit],b,y[valid],config=recipe['nn']['config'],epochs=recipe['nn']['epochs'])
            np.save(path,p);write_json(confirm_dir/('confirm_'+name+'.json'),d)
        prediction+=weight*np.load(path)
    np.save(confirm_dir/'confirmation_blend.npy',prediction)
    score=float(roc_auc_score(y[valid],prediction));baseline=float(roc_auc_score(y[valid],base))
    write_json(ROOT/'v2_confirmation.json',dict(auc=score,baseline_auc=baseline,change=score-baseline,
        weights=recipe['weights'],labels_used_for_training_or_early_stopping=False,
        split_previously_seen=True,no_tuning_after_confirmation=True,checkpoint_directory=str(confirm_dir),signature=confirm_signature))
    print('CONFIRMATION',score,baseline,score-baseline,flush=True)


def cv():
    recipe=load_recipe();x,xt,y,ox,oy,prior,pt,stats,meta=load_data()
    wb,wc,wn=recipe['weights'];old_oof=pd.read_csv(ROOT/'submission_optimized_oof.csv')
    old_pred=pd.read_csv(ROOT/'submission_optimized.csv').satisfaction.to_numpy()
    cat_source=source_prior(x,xt,ox,oy) if wc and recipe['cat_config'].get('source_prior') else None
    # Code+recipe fingerprint prevents mixing incompatible checkpoints.
    signature=hashlib.sha256((ROOT/'v2_recipe.json').read_bytes()+
        b''.join((ROOT/p).read_bytes() for p in ['v2_training.py','v2_tabm.py','optimize_v2.py','optimized_features.py'])).hexdigest()
    directory=OUT/'cv'/signature[:20];directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'signature.json',dict(sha256=signature))
    blended=wb*old_oof.prediction.to_numpy();test_pred=wb*old_pred
    folds=np.zeros(len(y),dtype=int);details=[];scores=[]
    for fold,(fit,valid) in enumerate(StratifiedKFold(10,shuffle=True,random_state=42).split(x,y),1):
        folds[valid]=fold
        np.testing.assert_array_equal(np.flatnonzero(old_oof.fold.to_numpy()==fold),valid)
        for weight,name in [(wc,'cat'),(wn,'nn')]:
            if weight==0:continue
            path=directory/f'fold_{fold:02d}_{name}.npz';meta_path=path.with_suffix('.json')
            if not path.exists() or not meta_path.exists():
                print('CV START',fold,name,flush=True)
                if name=='cat':
                    a,b,c,cats=make_cat_frames(x,xt,y,fit,valid,stats,prior,pt,cat_source)
                    pv,ps,d=fit_cat(a,y[fit],b,y[valid],c,cats,recipe['cat_config'],seed=42+fold)
                else:
                    from v2_tabm import frames,fit_tabm
                    a,b,c=frames(x,xt,y,fit,valid,stats,prior,pt,raw_only=recipe['nn']['config'].get('raw_only',False))
                    pv,ps,d=fit_tabm(a,y[fit],b,y[valid],c,recipe['nn']['config'],seed=42+fold)
                np.savez_compressed(path,valid_index=valid,validation_prediction=pv,test_prediction=ps)
                write_json(meta_path,d)
                del a,b,c
                print('CV RESULT',fold,name,d['auc'],flush=True)
            with np.load(path) as saved:
                np.testing.assert_array_equal(valid,saved['valid_index'])
                blended[valid]+=weight*saved['validation_prediction']
                test_pred+=weight*saved['test_prediction']/10
            details.append(dict(fold=fold,family=name,**json.loads(meta_path.read_text(encoding='utf-8'))))
        score=float(roc_auc_score(y[valid],blended[valid]));scores.append(score)
        print('CV BLEND',fold,score,flush=True)
    # Keep new output names even if gain is small; report evidence, no leaderboard claims.
    train=pd.read_csv(ROOT/'train.csv',usecols=['id']);test=pd.read_csv(ROOT/'test.csv',usecols=['id'])
    pd.DataFrame(dict(id=test.id,satisfaction=test_pred)).to_csv(ROOT/'submission_v2.csv',index=False)
    pd.DataFrame(dict(id=train.id,target=y,fold=folds,prediction=blended)).to_csv(ROOT/'submission_v2_oof.csv',index=False)
    baseline_scores=[float(roc_auc_score(y[folds==i],old_oof.prediction[folds==i])) for i in range(1,11)]
    report=dict(n_splits=10,fold_scores=scores,mean_auc=float(np.mean(scores)),std_auc=float(np.std(scores)),
        oof_auc=float(roc_auc_score(y,blended)),baseline_oof_auc=float(roc_auc_score(y,old_oof.prediction)),
        baseline_fold_scores=baseline_scores,improved_folds=int(np.sum(np.array(scores)>baseline_scores)),
        weights=recipe['weights'],components=details,checkpoint_directory=str(directory),signature=signature,
        target_auc=.962,kaggle_score=None,kaggle_submission_status='not_submitted',
        cv_used_early_stopping=True,selection_bias_note='Recipe selected on overlapping development rows. Prior audit already seen.',
        original_data=meta)
    write_json(ROOT/'submission_v2_metrics.json',report)
    print('COMPLETE',report['oof_auc'],report['mean_auc'],report['improved_folds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','confirmation','cv'])
    args=parser.parse_args()
    globals()[args.stage]()
