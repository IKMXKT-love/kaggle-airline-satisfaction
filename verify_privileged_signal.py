"""Rebuild source partitions, placebo columns, encoders and every held-out prediction."""
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from experiment_domain_transfer import ROOT, read, write, sha
from experiment_privileged_signal import RUN, ARMS, EXTRA
from optimized_features import normalize_raw, NativeCategoryEncoder
from multitask_support import paired

sys.stdout.reconfigure(encoding='utf-8')


def main():
    started=time.time(); protocol=read(RUN/'protocol.json'); evaluation=read(RUN/'evaluation.json')
    for f,h in protocol['input_sha256'].items(): assert sha(ROOT/f)==h,f
    for f,h in protocol['protected'].items(): assert sha(ROOT/f)==h,f
    features=list(pd.read_csv(ROOT/'test.csv',nrows=0).drop(columns='id'))
    comp=pd.concat([normalize_raw(pd.read_csv(ROOT/f,usecols=features)[features]) for f in ['train.csv','test.csv']],ignore_index=True)
    original=pd.concat([pd.read_csv(ROOT/'external_data'/f) for f in ['train.csv','test.csv']],ignore_index=True)
    ox=normalize_raw(original[features+[EXTRA]])
    fh=lambda z:pd.util.hash_pandas_object(z.astype('string').fillna('__MISSING__'),index=False)
    keep=~fh(ox[features]).isin(set(fh(comp)))
    x=ox.loc[keep].reset_index(drop=True); y=original.loc[keep,'satisfaction'].eq('satisfied').to_numpy(dtype='int64')
    with np.load(RUN/'indices.npz',allow_pickle=False) as z: idx={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(idx['original_rows'],np.flatnonzero(keep))
    np.testing.assert_array_equal(y,np.load(RUN/'labels.npy',allow_pickle=False))
    assert len(y)==protocol['original_rows'] and int((~keep).sum())==protocol['removed_overlap']
    train,score=train_test_split(np.arange(len(y)),test_size=.25,random_state=protocol['seed'],stratify=y)
    fit,stop=train_test_split(train,test_size=.1,random_state=protocol['seed']+1,stratify=y[train])
    for name,rows in [('fit',fit),('stopping',stop),('outer_train',train),('scored',score)]:
        np.testing.assert_array_equal(np.sort(rows),idx[name])
    np.testing.assert_array_equal(np.sort(np.r_[fit,stop,score]),np.arange(len(y)))
    results={}; scored_predictions={}
    for phase in ['selection','refit']:
        for arm in ARMS:
            fit_rows=idx['fit'] if phase=='selection' else idx['outer_train']
            score_rows=idx['stopping'] if phase=='selection' else idx['scored']
            a,b=x.iloc[fit_rows].copy(),x.iloc[score_rows].copy()
            if arm=='shared21': a,b=a.drop(columns=EXTRA),b.drop(columns=EXTRA)
            elif arm=='shuffled22':
                rng=np.random.default_rng(protocol['seed']+(0 if phase=='selection' else 1000))
                for frame in [a,b]:
                    original_values=frame[EXTRA].to_numpy(copy=True); frame[EXTRA]=rng.permutation(original_values)
                    np.testing.assert_array_equal(np.sort(original_values),np.sort(frame[EXTRA].to_numpy()))
            assert 'satisfaction' not in a and 'id' not in a and not any(c.startswith('Unnamed') for c in a)
            assert len(a.columns)==(21 if arm=='shared21' else 22)
            encoder=NativeCategoryEncoder().fit(a)
            assert dict(columns=encoder.columns_,categories=encoder.categories_)==read(RUN/phase/arm/'encoder.json')
            model=XGBClassifier(enable_categorical=True,n_jobs=4)
            model.load_model(RUN/phase/arm/'model.ubj'); model.set_params(device='cpu',n_jobs=4)
            p=model.predict_proba(encoder.transform(b))[:,1]
            with np.load(RUN/phase/arm/'prediction.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['index'],score_rows)
                err=float(np.max(np.abs(p-z['probability'])))
            assert err==0,(phase,arm,err)
            saved=read(RUN/phase/arm/'complete.json')
            assert abs(roc_auc_score(y[score_rows],p)-saved['auc'])<1e-12
            assert abs(log_loss(y[score_rows],p)-saved['logloss'])<1e-12
            if phase=='selection':
                curve=read(RUN/phase/arm/'curves.json')['validation_0']['auc']
                assert saved['trees']==int(np.argmax(curve))+1
            else:
                assert saved['trees']==read(RUN/'selection'/arm/'complete.json')['trees']
                scored_predictions[arm]=p
            results[f'{phase}/{arm}']=dict(prediction_max_error=err,auc=saved['auc'],logloss=saved['logloss'],trees=saved['trees'])
    sy=y[idx['scored']]
    comparisons={a:paired(sy,scored_predictions['privileged22'],scored_predictions[a]) for a in ['shared21','shuffled22']}
    assert comparisons==evaluation['privileged_vs_controls']
    accepted=all(z['delta']>=.0005 and z['interval95'][0]>0 for z in comparisons.values()) and all(log_loss(sy,scored_predictions['privileged22'])<=log_loss(sy,scored_predictions[a]) for a in comparisons)
    assert accepted==evaluation['passes_source_information_gate']
    assert protocol['competition_target_used'] is False
    for f,h in protocol['protected'].items(): assert sha(ROOT/f)==h
    clock=read(RUN/'clock.json')
    result=dict(verified=True,partition_encoder_and_placebo_rebuilt=True,full_six_model_prediction_rebuild=True,
        results=results,passes_source_information_gate=accepted,protected_unchanged=True,
        competition_fit_or_scoring=False,verification_seconds=time.time()-started,
        cumulative_seconds=time.time()-clock['started_epoch'],deadline_met=time.time()<=clock['deadline_epoch'])
    write(RUN/'verification.json',result); print(result,flush=True)


if __name__=='__main__': main()
