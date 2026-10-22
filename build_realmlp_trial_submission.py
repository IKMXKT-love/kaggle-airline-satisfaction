"""Full-data fixed-epoch candidate for an explicitly requested leaderboard trial."""
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_realmlp_factorial import ROOT,LeanFeatures,REFERENCE_PARAMS,sha,read
from v2_training import write_json

OUTPUT=ROOT/'submission_v4_lean_realmlp_trial.csv'

def main():
    protected=read(ROOT/'.artifact_runtime/v4/preserved.json')
    for name,h in protected.items():assert sha(ROOT/name)==h,name
    screen=Path(read(ROOT/'.artifact_runtime/realmlp_factorial/latest.json')['directory'])
    verified=read(screen/'verification.json');summary=read(screen/'summary.json')
    assert verified['verified'] and summary['best']=='lean_reference'
    protocol=read(screen/'protocol.json')
    assert sha(ROOT/'experiment_realmlp_factorial.py')==protocol['hashes']['experiment_realmlp_factorial.py']
    params=dict(REFERENCE_PARAMS)
    files=['train.csv','test.csv','sample_submission.csv','submission_v4.csv',
           'experiment_realmlp_factorial.py','build_realmlp_trial_submission.py']
    hashes={name:sha(ROOT/name) for name in files}
    versions={p:importlib.metadata.version(p) for p in ['torch','pytabkit','pandas','scikit-learn','numpy']}
    config=dict(params=params,seed=42,blend=dict(v4=.8,lean_realmlp=.2),
        full_training_rows=True,epochs=3,outer_cv_performed=False,
        public_trial_requested=True,screen_status=summary['status'],
        screen_candidate_blend_gain=summary['best_gain'],
        fullfit_not_equivalent_to_screen_training_size=True,
        original_v4_uses_existing_10fold_predictions=True)
    signature=hashlib.sha256(json.dumps([config,hashes,versions],sort_keys=True).encode()).hexdigest()[:20]
    directory=ROOT/'.artifact_runtime/realmlp_trial_submission'/signature
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'protocol.json',dict(**config,hashes=hashes,versions=versions,screen=str(screen)))
    (directory/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv')
    sample=pd.read_csv(ROOT/'sample_submission.csv');base=pd.read_csv(ROOT/'submission_v4.csv')
    assert list(sample)==list(base)==['id','satisfaction']
    np.testing.assert_array_equal(sample.id,test.id)
    np.testing.assert_array_equal(base.id,test.id)
    assert test.id.is_unique and not set(train.id)&set(test.id)
    y=train.satisfaction.to_numpy(dtype=np.int64)
    x=train.drop(columns=['id','satisfaction']);xt=test.drop(columns='id')
    assert list(x)==list(xt)
    prediction_path=directory/'test_prediction.npz'
    if not prediction_path.exists():
        from pytabkit import RealMLP_TD_Classifier
        import torch
        transform=LeanFeatures()
        a=transform.fit_transform(x,y);c=transform.transform(xt)
        assert len(a)==699635 and len(c)==299844 and a.shape[1]==47
        print('START_FULLFIT',len(a),len(c),a.shape[1],flush=True)
        begin=time.time();torch.cuda.reset_peak_memory_stats()
        model=RealMLP_TD_Classifier(device='cuda',random_state=42,n_cv=1,n_refit=0,
            val_fraction=0.,n_threads=4,verbosity=1,val_metric_name='1-auc_ovr',**params)
        # No split: all training labels, fold-cross-fitted TE, fixed stop_epoch=3.
        # Test labels do not exist and test features never fit the transformer.
        model.fit(a,y,cat_col_names=transform.cats)
        assert model.cv_alg_interface_.config['stop_epoch']==3
        p=model.predict_proba(c)[:,1]
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        np.savez_compressed(prediction_path,id=test.id.to_numpy(),prediction=p)
        write_json(directory/'training.json',dict(seconds=time.time()-begin,
            rows=len(a),test_rows=len(c),features=list(a),cats=transform.cats,
            peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,
            fixed_stop_epoch=3,validation_fraction=0.,n_ens=8))
        print('TRAINING_COMPLETE',time.time()-begin,flush=True)
    with np.load(prediction_path) as z:
        np.testing.assert_array_equal(test.id,z['id']);p=z['prediction'].copy()
    blended=.8*base.satisfaction.to_numpy(dtype=np.float64)+.2*p.astype(np.float64)
    assert np.isfinite(blended).all() and ((blended>=0)&(blended<=1)).all()
    output=sample.copy();output['satisfaction']=blended
    # Existing different output is never silently overwritten.
    if OUTPUT.exists():
        old=pd.read_csv(OUTPUT);pd.testing.assert_frame_equal(old,output,check_exact=False,atol=1e-14,rtol=0)
    else:
        tmp=OUTPUT.with_suffix('.tmp');output.to_csv(tmp,index=False,float_format='%.12f');tmp.replace(OUTPUT)
    written=pd.read_csv(OUTPUT)
    np.testing.assert_array_equal(written.id,test.id)
    assert list(written)==['id','satisfaction'] and len(written)==299844 and written.id.is_unique
    np.testing.assert_allclose(written.satisfaction,blended,atol=5.1e-13,rtol=0)
    for name,h in protected.items():assert sha(ROOT/name)==h,name
    for name,h in hashes.items():assert sha(ROOT/name)==h,name
    detail=dict(output=str(OUTPUT),sha256=sha(OUTPUT),rows=len(written),columns=list(written),
        min_probability=float(written.satisfaction.min()),max_probability=float(written.satisfaction.max()),
        ids_and_order_verified=True,blend_formula_verified=True,protected_files_unchanged=True,
        fullfit_candidate_only=True,new_candidate_ten_fold_cv=False,
        mean_absolute_change_vs_v4=float(np.mean(np.abs(blended-base.satisfaction.to_numpy()))),
        protocol=str(directory/'protocol.json'))
    write_json(directory/'verification.json',detail)
    write_json(ROOT/'submission_v4_lean_realmlp_trial.json',detail)
    print('VERIFIED_OUTPUT',json.dumps(detail,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
