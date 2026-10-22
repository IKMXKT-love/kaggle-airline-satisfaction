"""Reconstruct all CV predictions and the exported CSV from seed-level caches."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiment_realmlp_factorial import ROOT,sha,read
from v2_training import write_json

def main():
    directory=Path(read(ROOT/'.artifact_runtime/lean_cv_followup/latest.json')['directory'])
    protocol=read(directory/'protocol.json');s=read(directory/'summary.json')
    assert s['complete'] and len(s['folds'])==10
    for n,h in protocol['hashes'].items():assert sha(ROOT/n)==h,n
    for n,h in protocol['protected'].items():assert sha(ROOT/n)==h,n
    assert sha(directory/'trainer_snapshot.py')==protocol['hashes']['train_lean_cv_followup.py']
    assert sha(Path(protocol['followup'])/'decision.json')==protocol['followup_decision_hash']
    train=pd.read_csv(ROOT/'train.csv',usecols=['id','satisfaction']);test=pd.read_csv(ROOT/'test.csv',usecols=['id'])
    base_oof=pd.read_csv(ROOT/'submission_v4_oof.csv');base_test=pd.read_csv(ROOT/'submission_v4.csv')
    np.testing.assert_array_equal(train.id,base_oof.id);np.testing.assert_array_equal(train.satisfaction,base_oof.target)
    np.testing.assert_array_equal(test.id,base_test.id)
    folds=base_oof.fold.to_numpy()-protocol['fold_label_offset'];oof=np.full(len(train),np.nan)
    coverage=np.zeros(len(train),dtype=int);test_predictions=[];fold_results=[]
    def valid(p):assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    for fold in range(10):
        idx=np.flatnonzero(folds==fold);v=[];t=[]
        for seed in protocol['network_seeds']:
            with np.load(directory/f'fold_{fold}_seed_{seed}.npz') as z:
                np.testing.assert_array_equal(z['index'],idx)
                valid(z['prediction']);valid(z['test']);assert len(z['test'])==len(test)
                v.append(z['prediction'].astype(float));t.append(z['test'].astype(float))
        vp=np.mean(v,axis=0);tp=np.mean(t,axis=0)
        with np.load(directory/f'fold_{fold}.npz') as z:
            np.testing.assert_array_equal(z['index'],idx)
            np.testing.assert_array_equal(vp,z['prediction']);np.testing.assert_array_equal(tp,z['test'])
        oof[idx]=vp;coverage[idx]+=1;test_predictions.append(tp)
        fold_results.append(dict(fold=fold+1,new_model_auc=float(roc_auc_score(train.satisfaction.to_numpy()[idx],vp))))
    assert (coverage==1).all();valid(oof)
    # Follow the fixed fold ordering used during export.
    tp=sum(test_predictions)/10;w=protocol['fusion']['weight']
    def combine(base,new):
        if protocol['fusion']['form']=='probability':return (1-w)*base+w*new
        b,p=np.clip(base,1e-6,1-1e-6),np.clip(new,1e-6,1-1e-6)
        odds=(b/(1-b))**(1-w)*(p/(1-p))**w
        return odds/(1+odds)
    expected=combine(base_test.satisfaction.to_numpy(),tp)
    expected_oof=combine(base_oof.prediction.to_numpy(),oof)
    output=Path(s['output']);written=pd.read_csv(output)
    assert sha(output)==s['output_sha256'] and len(written)==299844 and written.id.is_unique
    assert list(written)==['id','satisfaction'];np.testing.assert_array_equal(written.id,test.id)
    np.testing.assert_allclose(written.satisfaction,expected,atol=5.1e-13,rtol=0);valid(written.satisfaction.to_numpy())
    with np.load(directory/'predictions.npz') as z:
        np.testing.assert_array_equal(z['id'],train.id);np.testing.assert_array_equal(z['test_id'],test.id)
        np.testing.assert_array_equal(z['coverage'],coverage);np.testing.assert_array_equal(z['oof'],oof)
        np.testing.assert_allclose(z['blended_oof'],expected_oof,atol=1e-14,rtol=0)
        np.testing.assert_allclose(z['blended_test'],expected,atol=1e-14,rtol=0)
    assert abs(roc_auc_score(train.satisfaction,oof)-s['standalone_oof_auc'])<1e-12
    assert abs(roc_auc_score(train.satisfaction,expected_oof)-s['blended_oof_auc'])<2e-11
    result=dict(verified=True,seed_caches_rebuilt=True,folds=10,all_rows_scored_once=True,
        rows=len(written),ids_verified=True,blend_verified=True,protected_unchanged=True,
        output_sha256=s['output_sha256'],folds_auc=fold_results,
        fixed_cv_probability20_auc=float(roc_auc_score(train.satisfaction,.8*base_oof.prediction.to_numpy()+.2*oof)))
    write_json(directory/'verification.json',result)
    print({k:v for k,v in result.items() if k!='folds_auc'})

if __name__=='__main__':main()
