"""Independently rebuild second-round predictions and verify preserved inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parent


def verify():
    root=ROOT;runtime=root/'.artifact_runtime/v2'
    manifest=json.loads((runtime/'snapshot/manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    recipe=json.loads((root/'v2_recipe.json').read_text(encoding='utf-8'))
    metrics=json.loads((root/'submission_v2_metrics.json').read_text(encoding='utf-8'))
    train=pd.read_csv(root/'train.csv');test=pd.read_csv(root/'test.csv');sample=pd.read_csv(root/'sample_submission.csv')
    pred=pd.read_csv(root/'submission_v2.csv');oof=pd.read_csv(root/'submission_v2_oof.csv')
    old=pd.read_csv(root/'submission_optimized_oof.csv');old_pred=pd.read_csv(root/'submission_optimized.csv')
    assert pred.shape==sample.shape==(len(test),2)
    assert pred.columns.tolist()==sample.columns.tolist()==['id','satisfaction']
    assert pred.id.equals(test.id) and pred.id.equals(sample.id) and pred.id.is_unique
    assert oof.id.equals(train.id) and np.array_equal(oof.target,train.satisfaction.astype(int))
    assert oof.fold.equals(old.fold)
    for values in [pred.satisfaction,oof.prediction]:
        assert np.isfinite(values).all() and values.between(0,1).all()
    wb,wc,wn=recipe['weights'];assert min(wb,wc,wn)>=0 and np.isclose(wb+wc+wn,1)
    rebuilt=wb*old.prediction.to_numpy();rebuilt_test=wb*old_pred.satisfaction.to_numpy()
    directory=Path(metrics['checkpoint_directory']);coverage=np.zeros(len(train),dtype=int)
    scores=[];base_scores=[]
    for fold,(fit,valid) in enumerate(StratifiedKFold(10,shuffle=True,random_state=42).split(train,train.satisfaction),1):
        assert not np.intersect1d(fit,valid).size
        coverage[valid]+=1
        np.testing.assert_array_equal(valid,np.flatnonzero(oof.fold==fold))
        for name,weight in [('cat',wc),('nn',wn)]:
            if not weight:continue
            path=directory/f'fold_{fold:02d}_{name}.npz'
            detail=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            assert not {'id','satisfaction'}.intersection(detail['features'])
            with np.load(path) as checkpoint:
                np.testing.assert_array_equal(valid,checkpoint['valid_index'])
                pv,pt=checkpoint['validation_prediction'],checkpoint['test_prediction']
                assert pv.shape==(len(valid),) and pt.shape==(len(test),)
                assert np.isfinite(pv).all() and np.isfinite(pt).all()
                rebuilt[valid]+=weight*pv;rebuilt_test+=weight*pt/10
        scores.append(float(roc_auc_score(train.satisfaction.iloc[valid],rebuilt[valid])))
        base_scores.append(float(roc_auc_score(train.satisfaction.iloc[valid],old.prediction.iloc[valid])))
    assert (coverage==1).all()
    np.testing.assert_allclose(rebuilt,oof.prediction,atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(rebuilt_test,pred.satisfaction,atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(scores,metrics['fold_scores'],atol=1e-12)
    assert abs(roc_auc_score(train.satisfaction,rebuilt)-metrics['oof_auc'])<1e-12
    signature=hashlib.sha256((root/'v2_recipe.json').read_bytes()+b''.join((root/p).read_bytes()
        for p in ['v2_training.py','v2_tabm.py','optimize_v2.py','optimized_features.py'])).hexdigest()
    assert signature==metrics['signature']
    confirmation=json.loads((root/'v2_confirmation.json').read_text(encoding='utf-8'))
    audit=np.load(runtime.parent/'writeup_search/split_indices.npz')['audit']
    confirm_dir=Path(confirmation['checkpoint_directory'])
    old_recipe=json.loads((root/'optimized_recipe.json').read_text(encoding='utf-8'))
    base_audit=sum(cfg['weight']*np.load(confirm_dir/('confirm_base_'+cfg['name']+'.npy')) for cfg in old_recipe['components'])
    new_audit=wb*base_audit
    for name,weight in [('cat',wc),('nn',wn)]:
        if weight:new_audit+=weight*np.load(confirm_dir/('confirm_'+name+'.npy'))
    np.testing.assert_allclose(new_audit,np.load(confirm_dir/'confirmation_blend.npy'))
    assert abs(roc_auc_score(train.satisfaction.iloc[audit],new_audit)-confirmation['auc'])<1e-12
    assert abs(roc_auc_score(train.satisfaction.iloc[audit],base_audit)-confirmation['baseline_auc'])<1e-12
    changes=np.array(scores)-base_scores
    receipt=dict(preserved_hashes_verified=True,ids_and_probabilities_valid=True,ten_folds_verified=True,
        predictions_rebuilt_from_checkpoints=True,confirmation_rebuilt_from_checkpoints=True,train_rows=len(train),test_rows=len(test),
        mean_auc=float(np.mean(scores)),oof_auc=float(roc_auc_score(train.satisfaction,rebuilt)),
        baseline_oof_auc=float(roc_auc_score(train.satisfaction,old.prediction)),
        oof_gain=float(roc_auc_score(train.satisfaction,rebuilt)-roc_auc_score(train.satisfaction,old.prediction)),
        improved_folds=int((changes>0).sum()),fold_gains=changes.tolist(),
        target_local_auc_reached=metrics['oof_auc']>=.962,kaggle_score_unknown=True,
        submission_sha256=hashlib.sha256((root/'submission_v2.csv').read_bytes()).hexdigest())
    (root/'v2_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':verify()
