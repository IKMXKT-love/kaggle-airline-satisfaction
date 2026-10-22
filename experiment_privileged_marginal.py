"""Deployable source-only marginalization; inherits the original diagnostic clock."""
import argparse
import gc
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier

from experiment_domain_transfer import ROOT, read, write, sha
from experiment_privileged_signal import RUN, EXTRA, PARAMS, load_data
from optimized_features import NativeCategoryEncoder
from multitask_support import paired

sys.stdout.reconfigure(encoding='utf-8')
OUT=RUN/'marginal'


def guard(reserve=45):
    if time.time()>=read(RUN/'clock.json')['deadline_epoch']-reserve: raise TimeoutError('Original clock exhausted')


def conditional_predictions(shared, auxiliary, teacher, encoder):
    assert EXTRA not in shared and 'satisfaction' not in shared and shared.shape[1]==21
    conditional=[]
    for level in range(6):
        inputs=shared.copy(); inputs[EXTRA]=np.float32(level)
        conditional.append(teacher.predict_proba(encoder.transform(inputs))[:,1])
    conditional=np.column_stack(conditional)
    assert auxiliary.shape==conditional.shape
    return np.sum(auxiliary.astype('float64')*conditional.astype('float64'),axis=1),conditional


def model_for(rounds, stopping=None):
    params=dict(PARAMS); params['eval_metric']='mlogloss'
    return XGBClassifier(n_estimators=rounds,objective='multi:softprob',num_class=6,
        early_stopping_rounds=stopping,**params)


def prepare():
    assert read(RUN/'verification.json')['verified']
    assert read(RUN/'evaluation.json')['passes_source_information_gate']
    if OUT.exists(): raise FileExistsError('Preserve existing marginal stage')
    OUT.mkdir(); guard()
    source=read(RUN/'protocol.json')
    files=['experiment_privileged_marginal.py','verify_privileged_marginal.py','PRIVILEGED_MARGINAL_PLAN_20261021.md']
    hashes={f:sha(ROOT/f) for f in files}
    x,y,original_rows=load_data()
    with np.load(RUN/'indices.npz',allow_pickle=False) as z: rows=z['fit']
    a=x.iloc[rows].drop(columns=EXTRA); encoder=NativeCategoryEncoder().fit(a)
    t=time.perf_counter(); model=model_for(30); model.fit(encoder.transform(a),x.iloc[rows][EXTRA].to_numpy(dtype='int64'),verbose=False)
    seconds=time.perf_counter()-t; del model; gc.collect()
    elapsed=time.time()-read(RUN/'clock.json')['started_epoch']
    projection=elapsed+1.5*2*800*(seconds/30)*(len(y)*.75/len(rows))+90
    protocol=dict(primary='conditional_integral',max_trees=800,early_stopping=80,
        minimum_source_auc_gain=.0003,require_logloss_not_worse=True,require_lower_positive=True,
        require_at_least_global_mixture_auc=True,source_protocol_sha256=sha(RUN/'protocol.json'),
        input_sha256=hashes,clock=str(RUN/'clock.json'),resource_probe_seconds=seconds,
        projected_cumulative_seconds=projection,source_extra_true_value_in_query_input=False,
        scored_rows_already_observed_in_previous_stage=True,competition_labels_used=False,
        caveat='Source generalization and deployability gate only; no competition gain measured.')
    write(OUT/'protocol.json',protocol)
    for f in files: (OUT/('snapshot_'+f)).write_bytes((ROOT/f).read_bytes())
    print('MARGINAL_PROTOCOL',protocol,flush=True)
    if projection>900: raise TimeoutError('Complete marginal stage does not fit original budget')


def run():
    protocol=read(OUT/'protocol.json'); x,y,original_rows=load_data()
    with np.load(RUN/'indices.npz',allow_pickle=False) as z: idx={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(original_rows,idx['original_rows'])
    for phase in ['selection','refit']:
        guard(); dest=OUT/phase
        if dest.exists(): raise FileExistsError(str(dest))
        dest.mkdir(); started=time.time()
        fit_rows=idx['fit'] if phase=='selection' else idx['outer_train']
        score_rows=idx['stopping'] if phase=='selection' else idx['scored']
        a,b=x.iloc[fit_rows].drop(columns=EXTRA),x.iloc[score_rows].drop(columns=EXTRA)
        az,bz=[x.iloc[z][EXTRA].to_numpy(dtype='int64') for z in [fit_rows,score_rows]]
        assert set(az)==set(range(6))
        encoder=NativeCategoryEncoder().fit(a)
        rounds=800 if phase=='selection' else read(OUT/'selection/complete.json')['trees']
        model=model_for(rounds,80 if phase=='selection' else None)
        model.fit(encoder.transform(a),az,eval_set=[(encoder.transform(b),bz)] if phase=='selection' else None,verbose=False)
        model.set_params(device='cpu'); pz=model.predict_proba(encoder.transform(b))
        model.save_model(dest/'rating_model.ubj')
        teacher=XGBClassifier(enable_categorical=True,n_jobs=4); teacher.load_model(RUN/phase/'privileged22/model.ubj'); teacher.set_params(device='cpu',n_jobs=4)
        teacher_encoder=NativeCategoryEncoder().fit(x.iloc[fit_rows])
        integrated,conditional=conditional_predictions(b,pz,teacher,teacher_encoder)
        prior=np.bincount(az,minlength=6)/len(az)
        global_mix=conditional.astype('float64')@prior
        point=conditional[np.arange(len(conditional)),pz.argmax(1)]
        np.savez(dest/'prediction.npz',index=score_rows,rating_probability=pz,teacher_conditional=conditional,
            integrated=integrated,point=point,global_mixture=global_mix,rating_prior=prior)
        result=dict(phase=phase,trees=int(model.best_iteration+1) if phase=='selection' else rounds,
            rating_accuracy=float(accuracy_score(bz,pz.argmax(1))),rating_logloss=float(log_loss(bz,pz,labels=np.arange(6))),
            integrated_auc=float(roc_auc_score(y[score_rows],integrated)),integrated_logloss=float(log_loss(y[score_rows],integrated)),
            point_auc=float(roc_auc_score(y[score_rows],point)),global_mixture_auc=float(roc_auc_score(y[score_rows],global_mix)),
            seconds=time.time()-started)
        if phase=='selection': write(dest/'curves.json',model.evals_result())
        write(dest/'complete.json',result); print('MARGINAL_FIT',result,flush=True)
        del model,teacher,a,b,pz,conditional; gc.collect()
    evaluate(protocol,y,idx)


def evaluate(protocol,y,idx):
    guard(20); score_y=y[idx['scored']]
    with np.load(OUT/'refit/prediction.npz',allow_pickle=False) as z: pred={k:z[k].copy() for k in z.files}
    with np.load(RUN/'refit/shared21/prediction.npz',allow_pickle=False) as z: control=z['probability'].copy()
    delta=paired(score_y,pred['integrated'],control)
    matched=read(RUN/'refit/shared21/complete.json'); metrics=read(OUT/'refit/complete.json')
    accepted=delta['delta']>=protocol['minimum_source_auc_gain'] and delta['interval95'][0]>0 and metrics['integrated_logloss']<=matched['logloss'] and metrics['integrated_auc']>=metrics['global_mixture_auc']
    result=dict(comparison_to_shared21=delta,metrics=metrics,control=matched,
        passes_deployable_source_gate=accepted,competition_score_measured=False,
        cumulative_seconds=time.time()-read(RUN/'clock.json')['started_epoch'])
    write(OUT/'evaluation.json',result); print('MARGINAL_EVALUATION',result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run']); args=parser.parse_args()
    try: prepare() if args.action=='prepare' else run()
    except BaseException as exc:
        if OUT.exists(): write(OUT/'stopped.json',dict(error=type(exc).__name__,detail=str(exc),epoch=time.time()))
        raise
