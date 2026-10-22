"""Gated fixed-epoch 10-fold RealMLP prediction; all supervised features nested."""
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from experiment_realmlp_factorial import ROOT,LeanFeatures,REFERENCE_PARAMS,sha,read
from v2_training import write_json

def main():
    follow=Path(read(ROOT/'.artifact_runtime/followup_20261019/latest.json')['directory'])
    decision=read(follow/'decision.json');verification=read(follow/'verification.json')
    assert verification['verified'] and decision['current_recipe_stable'],'Confirmation gate not passed'
    screen=read(follow/'screen/summary.json');confirm=read(follow/'confirmation/summary.json')
    # Additional seed retained only if it independently clears the same 0.00005 rule.
    keep_average=(decision['frozen_fusion_improves_current'] or
        (screen['mean_fixed20_vs_single']['delta']>=.00005 and confirm['mean_fixed20_vs_single']['delta']>=.00005))
    seeds=[42,20261019] if keep_average else [42]
    choice=decision['chosen_fusion']
    protected=read(follow/'protocol.json')['protected']
    hashes={n:sha(ROOT/n) for n in ['train.csv','test.csv','sample_submission.csv',
        'submission_v4.csv','submission_v4_oof.csv','experiment_realmlp_factorial.py','train_lean_cv_followup.py']}
    config=dict(params=REFERENCE_PARAMS,fold_assignment='existing_v4_oof_fold',fold_label_offset=1,network_seeds=seeds,
        seed_formula='base_seed + fold',fusion=choice,fixed_epoch=3,outer_labels_used_for_stopping=False,
        legacy_base_oof_used_outer_early_stopping=True,legacy_oof_not_pristine=True,
        no_weight_tuning_on_full_oof=True,followup=str(follow),followup_decision=decision,
        followup_decision_hash=sha(follow/'decision.json'))
    import importlib.metadata
    versions={p:importlib.metadata.version(p) for p in ['torch','pytabkit','numpy','pandas','scikit-learn']}
    signature=hashlib.sha256(json.dumps([config,hashes,versions],sort_keys=True).encode()).hexdigest()[:20]
    directory=ROOT/'.artifact_runtime/lean_cv_followup'/signature;directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'protocol.json',dict(**config,hashes=hashes,protected=protected,versions=versions))
    (directory/'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(ROOT/'.artifact_runtime/lean_cv_followup/latest.json',dict(directory=str(directory)))
    for n,h in protected.items():assert sha(ROOT/n)==h,n
    train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv')
    base_oof=pd.read_csv(ROOT/'submission_v4_oof.csv');base_test=pd.read_csv(ROOT/'submission_v4.csv')
    np.testing.assert_array_equal(base_oof.id,train.id);np.testing.assert_array_equal(base_oof.target,train.satisfaction)
    np.testing.assert_array_equal(base_test.id,test.id)
    x=train.drop(columns=['id','satisfaction']);xt=test.drop(columns='id');y=train.satisfaction.to_numpy()
    folds=base_oof.fold.to_numpy()-1;assert set(folds)==set(range(10))
    from pytabkit import RealMLP_TD_Classifier
    import torch
    rows=[];oof=np.full(len(x),np.nan);test_sum=np.zeros(len(xt));coverage=np.zeros(len(x),dtype=int)
    for fold in range(10):
        scored=np.flatnonzero(folds==fold);fitidx=np.flatnonzero(folds!=fold)
        print('FOLD_START',fold,len(fitidx),len(scored),'seeds',seeds,flush=True)
        predfile=directory/f'fold_{fold}.npz';detailfile=directory/f'fold_{fold}.json'
        if not(predfile.exists() and detailfile.exists()):
            begin=time.time();tf=LeanFeatures();a=tf.fit_transform(x.iloc[fitidx],y[fitidx])
            b=tf.transform(x.iloc[scored]);c=tf.transform(xt)
            validation=[];testing=[];resource=[]
            for seed in seeds:
                name=f'fold_{fold}_seed_{seed}.npz';path=directory/name
                if path.exists():
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z['index'],scored);validation.append(z['prediction']);testing.append(z['test'])
                    continue
                model=RealMLP_TD_Classifier(device='cuda',random_state=seed+fold,n_cv=1,n_refit=0,val_fraction=0.,
                    n_threads=4,verbosity=0,val_metric_name='1-auc_ovr',**REFERENCE_PARAMS)
                start=time.time();torch.cuda.reset_peak_memory_stats()
                # Outer validation labels never enter fit, preprocessing, or early stopping.
                model.fit(a,y[fitidx],cat_col_names=tf.cats)
                p=model.predict_proba(b)[:,1];pt=model.predict_proba(c)[:,1]
                assert np.isfinite(p).all() and np.isfinite(pt).all()
                assert ((p>=0)&(p<=1)).all() and ((pt>=0)&(pt<=1)).all()
                np.savez_compressed(path,index=scored,prediction=p,test=pt)
                validation.append(p);testing.append(pt)
                resource.append(dict(seed=seed+fold,seconds=time.time()-start,peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3))
                print('SEED_RESULT',fold,seed+fold,resource[-1]['seconds'],float(roc_auc_score(y[scored],p)),flush=True)
                del model;gc.collect();torch.cuda.empty_cache()
            p=np.mean(np.asarray(validation,dtype='float64'),axis=0);pt=np.mean(np.asarray(testing,dtype='float64'),axis=0)
            np.savez_compressed(predfile,index=scored,prediction=p,test=pt)
            d=dict(fold=fold,auc=float(roc_auc_score(y[scored],p)),seconds=time.time()-begin,
                resources=resource,feature_count=a.shape[1],fit_rows=len(a),scored_rows=len(b))
            write_json(detailfile,d);del a,b,c;gc.collect()
        with np.load(predfile) as z:
            np.testing.assert_array_equal(z['index'],scored)
            oof[scored]=z['prediction'];test_sum+=z['test'];coverage[scored]+=1
        rows.append(read(detailfile));write_json(directory/'progress.json',dict(completed_folds=len(rows),rows=rows))
        print('FOLD_RESULT',fold,rows[-1]['auc'],rows[-1]['seconds'],flush=True)
    assert (coverage==1).all() and np.isfinite(oof).all()
    test_prediction=test_sum/10;w=choice['weight']
    def combine(base,new):
        if choice['form']=='probability':return (1-w)*base+w*new
        return expit((1-w)*logit(np.clip(base,1e-6,1-1e-6))+w*logit(np.clip(new,1e-6,1-1e-6)))
    blended_oof=combine(base_oof.prediction.to_numpy(),oof)
    blended_test=combine(base_test.satisfaction.to_numpy(),test_prediction)
    output=ROOT/'submission_lean_realmlp_10fold.csv'
    sample=pd.read_csv(ROOT/'sample_submission.csv');np.testing.assert_array_equal(sample.id,test.id)
    sample.satisfaction=blended_test
    if output.exists():np.testing.assert_allclose(pd.read_csv(output).satisfaction,blended_test,atol=5.1e-13,rtol=0)
    else:sample.to_csv(output,index=False,float_format='%.12f')
    written=pd.read_csv(output);assert written.id.is_unique and len(written)==299844 and list(written)==['id','satisfaction']
    np.testing.assert_array_equal(written.id,test.id);np.testing.assert_allclose(written.satisfaction,blended_test,atol=5.1e-13,rtol=0)
    np.savez_compressed(directory/'predictions.npz',id=train.id,oof=oof,blended_oof=blended_oof,
        test_id=test.id,test_prediction=test_prediction,blended_test=blended_test,coverage=coverage)
    for n,h in protected.items():assert sha(ROOT/n)==h,n
    result=dict(complete=True,folds=rows,seeds=seeds,blend=choice,standalone_oof_auc=float(roc_auc_score(y,oof)),
        legacy_baseline_oof_auc=float(roc_auc_score(y,base_oof.prediction)),
        blended_oof_auc=float(roc_auc_score(y,blended_oof)),
        output=str(output),output_sha256=sha(output),rows=len(written),protected_unchanged=True,
        legacy_oof_early_stopping_caveat=True,public_score_unknown=True)
    write_json(directory/'summary.json',result)
    write_json(ROOT/'submission_lean_realmlp_10fold.json',result)
    print('CV_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='folds'}),flush=True)

if __name__=='__main__':main()
