"""Independently integrate all six counterfactual source ratings and audit inputs."""
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from xgboost import XGBClassifier
from experiment_domain_transfer import ROOT, read, write, sha
from experiment_privileged_signal import RUN, EXTRA, load_data
from experiment_privileged_marginal import OUT
from optimized_features import NativeCategoryEncoder
from multitask_support import paired

sys.stdout.reconfigure(encoding='utf-8')


def main():
    started=time.time(); protocol=read(OUT/'protocol.json'); evaluation=read(OUT/'evaluation.json'); source=read(RUN/'protocol.json')
    assert sha(RUN/'protocol.json')==protocol['source_protocol_sha256']
    for manifest in [source,protocol]:
        for f,h in manifest['input_sha256'].items(): assert sha(ROOT/f)==h,f
    for f,h in source['protected'].items(): assert sha(ROOT/f)==h
    x,y,original_rows=load_data()
    with np.load(RUN/'indices.npz',allow_pickle=False) as z: idx={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(original_rows,idx['original_rows'])
    evidence={}; final=None
    for phase in ['selection','refit']:
        rows=idx['fit'] if phase=='selection' else idx['outer_train']
        score=idx['stopping'] if phase=='selection' else idx['scored']
        zfit=x.iloc[rows][EXTRA].to_numpy(dtype='int64'); zs=x.iloc[score][EXTRA].to_numpy(dtype='int64')
        encoder=NativeCategoryEncoder().fit(x.iloc[rows].drop(columns=EXTRA))
        model=XGBClassifier(enable_categorical=True,n_jobs=4); model.load_model(OUT/phase/'rating_model.ubj'); model.set_params(device='cpu',n_jobs=4)
        common=x.iloc[score].drop(columns=EXTRA)
        assert len(common.columns)==21 and 'satisfaction' not in common and EXTRA not in common
        pz=model.predict_proba(encoder.transform(common))
        teacher=XGBClassifier(enable_categorical=True,n_jobs=4); teacher.load_model(RUN/phase/'privileged22/model.ubj'); teacher.set_params(device='cpu',n_jobs=4)
        enc22=NativeCategoryEncoder().fit(x.iloc[rows]); conditional=np.empty_like(pz)
        for k in range(6):
            frame=common.copy(); frame[EXTRA]=np.float32(k)
            conditional[:,k]=teacher.predict_proba(enc22.transform(frame))[:,1]
        # Float64 accumulation is explicit; no query's observed extra rating enters this calculation.
        integrated=np.einsum('ij,ij->i',pz.astype('float64'),conditional.astype('float64'))
        prior=np.bincount(zfit,minlength=6)/len(zfit)
        mixture=conditional.astype('float64')@prior
        point=conditional[np.arange(len(conditional)),pz.argmax(1)]
        with np.load(OUT/phase/'prediction.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['index'],score)
            for name,value in [('rating_probability',pz),('teacher_conditional',conditional),('integrated',integrated),('rating_prior',prior),('global_mixture',mixture),('point',point)]:
                np.testing.assert_allclose(value,saved[name],rtol=0,atol=1e-15)
        metrics=read(OUT/phase/'complete.json')
        actual_auc=float(roc_auc_score(y[score],integrated)); actual_loss=float(log_loss(y[score],integrated))
        assert abs(actual_auc-metrics['integrated_auc'])<1e-12 and abs(actual_loss-metrics['integrated_logloss'])<1e-12
        assert abs(accuracy_score(zs,pz.argmax(1))-metrics['rating_accuracy'])<1e-12
        if phase=='selection':
            curve=read(OUT/phase/'curves.json')['validation_0']['mlogloss']; assert int(np.argmin(curve))+1==metrics['trees']
        else:
            assert metrics['trees']==read(OUT/'selection/complete.json')['trees']; final=integrated
        evidence[phase]=dict(integrated_auc=actual_auc,integrated_logloss=actual_loss,all_predictions_rebuilt=True)
    with np.load(RUN/'refit/shared21/prediction.npz',allow_pickle=False) as z: base=z['probability'].copy()
    comparison=paired(y[idx['scored']],final,base)
    assert comparison==evaluation['comparison_to_shared21']
    m=evaluation['metrics']; control=evaluation['control']
    gate=comparison['delta']>=.0003 and comparison['interval95'][0]>0 and m['integrated_logloss']<=control['logloss'] and m['integrated_auc']>=m['global_mixture_auc']
    assert gate==evaluation['passes_deployable_source_gate']
    for f,h in source['protected'].items(): assert sha(ROOT/f)==h
    clock=read(RUN/'clock.json')
    result=dict(verified=True,all_counterfactual_source_predictions_rebuilt=True,query_true_extra_and_target_not_inputs=True,
        phase_evidence=evidence,passes_deployable_source_gate=gate,protected_unchanged=True,
        verification_seconds=time.time()-started,cumulative_seconds=time.time()-clock['started_epoch'],
        original_deadline_met=time.time()<=clock['deadline_epoch'])
    write(OUT/'verification.json',result); print(result,flush=True)


if __name__=='__main__': main()
