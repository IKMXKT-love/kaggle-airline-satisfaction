"""Reconstruct v4 independently from checkpoints and preserve the reported v3 baseline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def verify():
    recipe = read(ROOT/'v4_recipe.json')
    directory = ROOT/'.artifact_runtime/v4'/recipe['signature']
    preserved = read(ROOT/'.artifact_runtime/v4/preserved.json')
    for name, expected in preserved.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    protocol = read(directory/'protocol.json')
    files = ['optimize_v4.py','v3_features.py','v3_training.py','v2_training.py',
             'optimized_features.py','optimized_training.py']
    code = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    signature = hashlib.sha256(json.dumps(code,sort_keys=True).encode() +
                              (ROOT/'.artifact_runtime/v4/preserved.json').read_bytes()).hexdigest()
    assert signature == protocol['signature'] and signature[:20] == recipe['signature']
    train = pd.read_csv(ROOT/'train.csv',usecols=['id','satisfaction'])
    y = train.satisfaction.to_numpy(dtype=int)
    w = recipe['weight'];replace = recipe['mode']=='replace_xgb'
    receipt = dict(previous_files_unchanged=len(preserved),code_signature_verified=True,
                   kaggle_score=None,independently_queried_kaggle=False)
    if (directory/'confirmation.npz').exists():
        with np.load(directory/'confirmation.npz') as z:
            valid = z['valid_index']
            old_directory = Path(read(ROOT/'v3_confirmation.json')['directory'])
            baseline = np.load(old_directory/'blend.npy')
            if replace:
                predicted = baseline-w*np.load(old_directory/'xgb.npy')+w*z['candidate']
            else:
                predicted = (1-w)*baseline+w*z['candidate']
            np.testing.assert_allclose(predicted,z['prediction'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(baseline,z['baseline'],rtol=0,atol=0)
            conf = read(directory/'confirmation.json')
            assert abs(roc_auc_score(y[valid],predicted)-conf['auc'])<1e-12
            receipt['confirmation_reconstructed']=True
    metrics_path=ROOT/'submission_v4_metrics.json'
    if metrics_path.exists() and read(metrics_path)['directory']==str(directory):
        metrics=read(metrics_path)
        base=pd.read_csv(ROOT/'submission_v3_oof.csv');out=pd.read_csv(ROOT/'submission_v4_oof.csv')
        old_test=pd.read_csv(ROOT/'submission_v3.csv');test=pd.read_csv(ROOT/'submission_v4.csv')
        template=pd.read_csv(ROOT/'sample_submission.csv')
        actual_test=pd.read_csv(ROOT/'test.csv',usecols=['id'])
        assert list(test)==list(template)==['id','satisfaction'] and test.shape==template.shape
        assert test.id.equals(template.id) and test.id.equals(actual_test.id) and test.id.is_unique
        assert out.id.equals(train.id) and np.array_equal(out.target,y) and out.fold.equals(base.fold)
        pv=base.prediction.to_numpy().copy()*(1 if replace else 1-w)
        pt=old_test.satisfaction.to_numpy().copy()*(1 if replace else 1-w)
        coverage=np.zeros(len(y),int)
        olddir=Path(read(ROOT/'submission_v3_metrics.json')['directory'])
        for fold in range(1,11):
            with np.load(directory/f'fold_{fold:02d}.npz') as z:
                idx=z['valid_index'];np.testing.assert_array_equal(idx,np.flatnonzero(out.fold==fold))
                coverage[idx]+=1
                pv[idx]+=w*z['validation_prediction'];pt+=w*z['test_prediction']/10
                if replace:
                    with np.load(olddir/f'fold_{fold:02d}_xgb.npz') as bx:
                        np.testing.assert_array_equal(bx['valid_index'],idx)
                        pv[idx]-=w*bx['validation_prediction'];pt-=w*bx['test_prediction']/10
            d=read(directory/f'fold_{fold:02d}.json')
            assert not {'id','satisfaction'}.intersection(d['features'])
        assert (coverage==1).all()
        np.testing.assert_allclose(pv,out.prediction,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(pt,test.satisfaction,rtol=1e-12,atol=1e-12)
        for p in [pv,pt]:assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        assert abs(roc_auc_score(y,pv)-metrics['oof_auc'])<1e-12
        receipt.update(ten_folds_reconstructed=True,rows=len(test),oof_auc=metrics['oof_auc'],
                       submission_sha256=hashlib.sha256((ROOT/'submission_v4.csv').read_bytes()).hexdigest())
    (ROOT/'v4_independent_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    verify()
