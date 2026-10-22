"""Independent reconstruction and input-preservation checks for the final source-informed recipe."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parent


def verify():
    r=ROOT;recipe=json.loads((r/'v3_recipe.json').read_text(encoding='utf-8'))
    for name,h in recipe['protected'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
    m=json.loads((r/'submission_v3_metrics.json').read_text(encoding='utf-8'))
    y=pd.read_csv(r/'train.csv');test=pd.read_csv(r/'test.csv');sample=pd.read_csv(r/'sample_submission.csv')
    oof=pd.read_csv(r/'submission_v3_oof.csv');pred=pd.read_csv(r/'submission_v3.csv')
    old=pd.read_csv(r/'submission_optimized_oof.csv');oldp=pd.read_csv(r/'submission_optimized.csv')
    assert pred.shape==sample.shape==(len(test),2)
    assert list(pred)==list(sample)==['id','satisfaction']
    assert pred.id.equals(test.id) and pred.id.equals(sample.id) and pred.id.is_unique
    assert oof.id.equals(y.id) and np.array_equal(oof.target,y.satisfaction.astype(int))
    assert oof.fold.equals(old.fold)
    for v in [pred.satisfaction,oof.prediction]:assert np.isfinite(v).all() and v.between(0,1).all()
    weights=recipe['weights'];assert min(weights)>=0 and np.isclose(sum(weights),1)
    wb,wx,wn=weights;pv=wb*old.prediction.to_numpy();pt=wb*oldp.satisfaction.to_numpy();coverage=np.zeros(len(y),int)
    directory=Path(m['directory']);scores=[];baseline=[]
    for fold,(fit,valid) in enumerate(StratifiedKFold(10,shuffle=True,random_state=42).split(y,y.satisfaction),1):
        coverage[valid]+=1;np.testing.assert_array_equal(valid,np.flatnonzero(oof.fold==fold));assert not np.intersect1d(fit,valid).size
        for family,w in [('xgb',wx),('realmlp',wn)]:
            if not w:continue
            path=directory/f'fold_{fold:02d}_{family}.npz'
            detail=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            assert not set(['id','satisfaction']).intersection(detail['features'])
            with np.load(path) as z:
                np.testing.assert_array_equal(z['valid_index'],valid)
                assert z['validation_prediction'].shape==(len(valid),) and z['test_prediction'].shape==(len(test),)
                pv[valid]+=w*z['validation_prediction'];pt+=w*z['test_prediction']/10
        scores.append(float(roc_auc_score(y.satisfaction.iloc[valid],pv[valid])))
        baseline.append(float(roc_auc_score(y.satisfaction.iloc[valid],old.prediction.iloc[valid])))
    assert (coverage==1).all()
    np.testing.assert_allclose(pv,oof.prediction,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(pt,pred.satisfaction,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(scores,m['fold_scores'],rtol=1e-10,atol=1e-12)
    sig=hashlib.sha256((r/'v3_recipe.json').read_bytes()+b''.join((r/p).read_bytes()
        for p in ['v3_features.py','v3_training.py','optimize_v3.py','optimized_features.py'])).hexdigest()
    assert sig==m['signature']
    a=json.loads((r/'v3_confirmation.json').read_text(encoding='utf-8'));ad=Path(a['directory'])
    valid=np.load(r/'.artifact_runtime/writeup_search/split_indices.npz')['audit'];av=wb*np.load(ad/'baseline.npy')
    for family,w in [('xgb',wx),('realmlp',wn)]:
        if w:av+=w*np.load(ad/(family+'.npy'))
    np.testing.assert_allclose(av,np.load(ad/'blend.npy'),rtol=1e-10,atol=1e-12)
    assert abs(roc_auc_score(y.satisfaction.iloc[valid],av)-a['auc'])<1e-12
    gain=np.array(scores)-baseline;score=float(roc_auc_score(y.satisfaction,pv));bs=float(roc_auc_score(y.satisfaction,old.prediction))
    assert abs(score-m['oof_auc'])<1e-12
    receipt=dict(preserved_files=len(recipe['protected']),ten_folds_same_as_previous=True,every_row_oof_once=True,
        probabilities_ids_and_schema_valid=True,predictions_reconstructed=True,confirmation_reconstructed=True,
        rows=len(pred),train_rows=len(oof),oof_auc=score,baseline_oof_auc=bs,oof_gain=score-bs,
        mean_auc=float(np.mean(scores)),improved_folds=int((gain>0).sum()),fold_gains=gain.tolist(),
        confirmation_auc=a['auc'],confirmation_gain=a['change'],target_local_auc_reached=score>=.962,
        kaggle_score_unverified=True,submission_sha256=hashlib.sha256((r/'submission_v3.csv').read_bytes()).hexdigest())
    (r/'v3_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))


if __name__=='__main__':verify()
