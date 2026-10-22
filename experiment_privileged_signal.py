"""Cheap original-only extra-feature gate before any competition transfer fit."""
import argparse
import gc
import math
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from experiment_domain_transfer import ROOT, PROTECTED, read, write, sha
from multitask_support import paired
from optimized_features import normalize_raw, NativeCategoryEncoder

sys.stdout.reconfigure(encoding='utf-8')
RUN=ROOT/'.artifact_runtime/privileged_signal_20261021'
ARMS=['shared21','privileged22','shuffled22']
EXTRA='Inflight service'
SEED=20261022
PARAMS=dict(max_depth=6,learning_rate=.04,min_child_weight=5,subsample=.9,colsample_bytree=1.,
    reg_lambda=5,tree_method='hist',device='cuda',enable_categorical=True,n_jobs=4,
    random_state=SEED,eval_metric='auc')


def guard(reserve=45):
    if time.time()>read(RUN/'clock.json')['deadline_epoch']-reserve: raise TimeoutError('Original cumulative budget reached')


def load_data():
    features=list(pd.read_csv(ROOT/'test.csv',nrows=0).drop(columns='id'))
    competition=pd.concat([normalize_raw(pd.read_csv(ROOT/f,usecols=features)[features]) for f in ['train.csv','test.csv']],ignore_index=True)
    original=pd.concat([pd.read_csv(ROOT/'external_data'/f) for f in ['train.csv','test.csv']],ignore_index=True)
    x=normalize_raw(original[features+[EXTRA]])
    a=pd.util.hash_pandas_object(competition.astype('string').fillna('__MISSING__'),index=False)
    b=pd.util.hash_pandas_object(x[features].astype('string').fillna('__MISSING__'),index=False)
    keep=~b.isin(set(a))
    return x.loc[keep].reset_index(drop=True),original.loc[keep,'satisfaction'].eq('satisfied').to_numpy(dtype='int64'),np.flatnonzero(keep)


def transformed(x, indices, phase, arm):
    rows=indices['fit'] if phase=='selection' else indices['outer_train']
    valid=indices['stopping'] if phase=='selection' else indices['scored']
    a,b=x.iloc[rows].copy(),x.iloc[valid].copy()
    if arm=='shared21': a,b=a.drop(columns=EXTRA),b.drop(columns=EXTRA)
    elif arm=='shuffled22':
        generator=np.random.default_rng(SEED+(0 if phase=='selection' else 1000))
        a[EXTRA]=generator.permutation(a[EXTRA].to_numpy())
        b[EXTRA]=generator.permutation(b[EXTRA].to_numpy())
    assert not {'id','satisfaction'}&set(a) and not any(c.startswith('Unnamed') for c in a)
    encoder=NativeCategoryEncoder().fit(a)
    return encoder.transform(a),encoder.transform(b),encoder


def prepare():
    if RUN.exists(): raise FileExistsError('Keep prior artifacts and clock')
    RUN.mkdir(parents=True); started=time.time()
    write(RUN/'clock.json',dict(started_epoch=started,deadline_epoch=started+900,limit=900))
    protected={f:sha(ROOT/f) for f in PROTECTED}
    files=['train.csv','test.csv','external_data/train.csv','external_data/test.csv',
        'experiment_privileged_signal.py','verify_privileged_signal.py','PRIVILEGED_SIGNAL_PLAN_20261021.md',
        'optimized_features.py','multitask_support.py','experiment_domain_transfer.py']
    hashes={f:sha(ROOT/f) for f in files}
    x,y,original_rows=load_data(); assert x[EXTRA].notna().all() and x[EXTRA].isin(range(6)).all()
    outer,scored=train_test_split(np.arange(len(y)),test_size=.25,random_state=SEED,stratify=y)
    fit,stop=train_test_split(outer,test_size=.10,random_state=SEED+1,stratify=y[outer])
    idx=dict(fit=np.sort(fit),stopping=np.sort(stop),outer_train=np.sort(outer),scored=np.sort(scored),original_rows=original_rows)
    np.savez(RUN/'indices.npz',**idx); np.save(RUN/'labels.npy',y,allow_pickle=False)
    guard(); a,_,_=transformed(x,idx,'selection','privileged22')
    # Separate short and long probes estimate per-tree scaling; both use fit rows only.
    timings=[]
    for rounds in [20,60]:
        t=time.perf_counter(); m=XGBClassifier(n_estimators=rounds,**PARAMS); m.fit(a,y[idx['fit']],verbose=False)
        timings.append(time.perf_counter()-t)
        del m; gc.collect()
    seconds_per_tree=max((timings[1]-timings[0])/40,timings[1]/60)
    elapsed=time.time()-started
    conservative=elapsed+1.5*6*1200*seconds_per_tree*(len(outer)/len(fit))+90
    profile=dict(extra_counts={str(int(k)):int(v) for k,v in x[EXTRA].value_counts().items()},
        equality_rates={c:float((x[EXTRA]==x[c]).mean()) for c in x if c not in ['Gender','Customer Type','Class','Type of Travel',EXTRA]})
    protocol=dict(seed=SEED,arms=ARMS,primary='privileged22',parameters=PARAMS,max_trees=1200,early_stopping=100,
        minimum_source_auc_gain=.0005,require_paired_lower_above_zero=True,require_logloss_not_worse=True,
        original_rows=len(y),removed_overlap=129880-len(y),row_counts={k:len(v) for k,v in idx.items() if k!='original_rows'},
        competition_target_used=False,source_field_is_not_available_for_competition_inference=True,
        input_sha256=hashes,protected=protected,profile=profile,
        costs=dict(preparation_probe_seconds=elapsed,probe_seconds_20_60_trees=timings,
            conservative_seconds_per_tree=seconds_per_tree,projected_complete_seconds=conservative),
        role='Source information gate only; does not demonstrate transfer or competition improvement')
    write(RUN/'protocol.json',protocol)
    for f in files:
        if f.endswith('.py') or f.endswith('.md'): (RUN/('snapshot_'+f.replace('/','_'))).write_bytes((ROOT/f).read_bytes())
    print('PROTOCOL',protocol,flush=True)
    if conservative>900: raise TimeoutError('Six-fit diagnostic cannot fit budget')


def run():
    protocol=read(RUN/'protocol.json'); x,y,original_rows=load_data()
    with np.load(RUN/'indices.npz',allow_pickle=False) as z: idx={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(idx['original_rows'],original_rows)
    for phase in ['selection','refit']:
        for arm in ARMS:
            guard(); path=RUN/phase/arm
            if path.exists(): raise FileExistsError(str(path))
            path.mkdir(parents=True); t=time.time()
            a,b,encoder=transformed(x,idx,phase,arm)
            rows=idx['fit'] if phase=='selection' else idx['outer_train']
            score_rows=idx['stopping'] if phase=='selection' else idx['scored']
            rounds=1200 if phase=='selection' else read(RUN/'selection'/arm/'complete.json')['trees']
            m=XGBClassifier(n_estimators=rounds,early_stopping_rounds=100 if phase=='selection' else None,**PARAMS)
            m.fit(a,y[rows],eval_set=[(b,y[score_rows])] if phase=='selection' else None,verbose=False)
            guard(); trees=int(m.best_iteration+1) if phase=='selection' else rounds
            # CPU inference matches stored model checks and avoids device fallback warnings.
            m.set_params(device='cpu'); prediction=m.predict_proba(b)[:,1]
            m.save_model(path/'model.ubj')
            np.savez(path/'prediction.npz',index=score_rows,probability=prediction)
            write(path/'encoder.json',dict(columns=encoder.columns_,categories=encoder.categories_))
            result=dict(phase=phase,arm=arm,trees=trees,seconds=time.time()-t,
                auc=float(roc_auc_score(y[score_rows],prediction)),logloss=float(log_loss(y[score_rows],prediction)),
                fitted_rows=len(rows),scored_rows=len(score_rows))
            if phase=='selection': write(path/'curves.json',m.evals_result())
            write(path/'complete.json',result); print('FIT',result,flush=True)
            del m,a,b; gc.collect()
    evaluate(protocol,y,idx)


def evaluate(protocol,y,idx):
    guard(20); score_y=y[idx['scored']]; predictions={}; arms={}
    for arm in ARMS:
        with np.load(RUN/'refit'/arm/'prediction.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['index'],idx['scored']); predictions[arm]=z['probability'].copy()
        arms[arm]=read(RUN/'refit'/arm/'complete.json')
    comparisons={a:paired(score_y,predictions['privileged22'],predictions[a]) for a in ['shared21','shuffled22']}
    accepted=all(v['delta']>=.0005 and v['interval95'][0]>0 for v in comparisons.values()) and all(arms['privileged22']['logloss']<=arms[a]['logloss'] for a in comparisons)
    result=dict(arms=arms,privileged_vs_controls=comparisons,passes_source_information_gate=accepted,
        cumulative_seconds=time.time()-read(RUN/'clock.json')['started_epoch'],
        competition_predictions_created=False,competition_score_measured=False)
    write(RUN/'evaluation.json',result); print('EVALUATION',result,flush=True)
    assert protocol['protected']=={f:sha(ROOT/f) for f in protocol['protected']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run']); args=parser.parse_args()
    try: prepare() if args.action=='prepare' else run()
    except BaseException as exc:
        if RUN.exists(): write(RUN/'stopped.json',dict(error=type(exc).__name__,detail=str(exc),epoch=time.time()))
        raise
