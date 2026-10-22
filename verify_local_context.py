"""Full index/score audit plus two complete anchor prediction rebuilds per arm."""
import gc
import json
import sys
import time
from pathlib import Path

from local_context_support import RetrievalEncoder,TaskSampler,inference_contexts,predict_context
from experiment_local_context import nearest_pools,load_net,sha
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from tabicl._sklearn.preprocessing import TransformToNumerical
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'.artifact_runtime/local_context_20261021'
ARMS=['random_frozen','local_frozen','random_tuned','local_tuned']
sys.stdout.reconfigure(encoding='utf-8')


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    started=time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    p,e=read(RUN/'protocol.json'),read(RUN/'evaluation.json')
    deadline=read(RUN/'clock.json')['deadline_epoch']
    def guard():
        if time.time()>=deadline:raise TimeoutError('Verification exceeded cumulative deadline')
    guard()
    for f,h in p['input_sha256'].items():assert sha(ROOT/f)==h,f
    for f,h in p['protected'].items():assert sha(ROOT/f)==h,f
    checkpoint=ROOT/p['checkpoint']
    assert sha(checkpoint)==p['checkpoint_sha256']
    tr=pd.read_csv(ROOT/'train.csv')
    raw=tr.drop(columns=['id','satisfaction'])
    y=tr.satisfaction.to_numpy(dtype='int64')
    with np.load(RUN/'indices.npz',allow_pickle=False) as z:fit,outer=z['fitting'],z['outer_eval']
    fold=np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy',allow_pickle=False)
    np.testing.assert_array_equal(fit,np.flatnonzero(fold!=1))
    np.testing.assert_array_equal(outer,np.flatnonzero(fold==1))
    np.testing.assert_array_equal(np.sort(np.r_[fit,outer]),np.arange(len(tr)))
    with np.load(RUN/'labels.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['reference_y'],y[fit]);np.testing.assert_array_equal(z['scored_y'],y[outer])
        np.testing.assert_array_equal(z['reference_id'],tr.id.to_numpy()[fit]);np.testing.assert_array_equal(z['scored_id'],tr.id.to_numpy()[outer])
    encoder=TransformToNumerical()
    rebuilt=np.asarray(encoder.fit_transform(raw.iloc[fit]),dtype='float64')
    queries=np.asarray(encoder.transform(raw.iloc[outer]),dtype='float64')
    x=np.load(RUN/'reference.npy',allow_pickle=False)
    q=np.load(RUN/'queries.npy',allow_pickle=False)
    np.testing.assert_array_equal(rebuilt,x);np.testing.assert_array_equal(queries,q)
    metric_encoder=RetrievalEncoder().fit(raw.iloc[fit])
    assert metric_encoder.state()==read(RUN/'retrieval_encoder.json')
    metric=metric_encoder.transform(raw.iloc[fit]);query_metric=metric_encoder.transform(raw.iloc[outer])
    np.testing.assert_array_equal(metric,np.load(RUN/'reference_metric.npy',allow_pickle=False))
    with np.load(RUN/'retrieval.npz',allow_pickle=False) as z:
        centers,assignment,qa,pools=z['centers'],z['assignment'],z['query_assignment'],z['pools']
        sample=z['fit_sample']
    expected_sample=np.random.default_rng(p['seed']).choice(len(fit),65536,replace=False)
    np.testing.assert_array_equal(sample,expected_sample)
    cluster=joblib.load(RUN/'cluster.joblib')
    np.testing.assert_array_equal(cluster.cluster_centers_,centers)
    with threadpool_limits(limits=4):
        np.testing.assert_array_equal(cluster.predict(metric),assignment)
        np.testing.assert_array_equal(cluster.predict(query_metric),qa)
    np.testing.assert_array_equal(nearest_pools(metric,centers,p['retrieval_pool']),pools)
    contexts=inference_contexts(assignment,pools,p['context_size'])
    with np.load(RUN/'contexts.npz',allow_pickle=False) as z:
        for k in contexts:np.testing.assert_array_equal(z[k],contexts[k])
    sampler=TaskSampler(assignment,pools,p['context_size'],p['query_size'],seed=p['seed'])
    with np.load(RUN/'tasks.npz',allow_pickle=False) as z:
        for step in range(p['steps']):
            task=sampler.sample(step);n=int(z['query_count'][step])
            np.testing.assert_array_equal(z['query'][step,:n],task['query'])
            assert (z['query'][step,n:]==-1).all()
            for k in ['random','local']:np.testing.assert_array_equal(z[k][step],task[k])
    positions=np.load(RUN/'scored_positions.npy',allow_pickle=False)
    assert len(np.unique(positions))==len(positions) and positions.min()>=0 and positions.max()<len(outer)
    labels=y[outer][positions]
    base=np.load(RUN/'baseline_logit.npy',allow_pickle=False)[positions]
    with np.load(ROOT/'.artifact_runtime/public_pool_trial_20261020/public_only_fold1.npz',allow_pickle=False) as z:w=z['weights']
    matrix=np.load(ROOT/'.artifact_runtime/public_pool_trial_20261020/train_logits.npy',mmap_mode='r',allow_pickle=False)
    base_rebuilt=np.empty(len(positions))
    with threadpool_limits(limits=4):
        for start in range(0,len(positions),8192):
            rows=outer[positions[start:start+8192]]
            base_rebuilt[start:start+len(rows)]=matrix[rows,:246]@w[:-1]+w[-1]
    assert np.max(abs(base_rebuilt-base))<1e-10
    assert abs(roc_auc_score(labels,base)-e['baseline_auc'])<1e-12
    evidence={}
    selected_anchors=[k for k in [0,32] if (qa[positions]==k).any()]
    if len(selected_anchors)<2:selected_anchors=list(np.unique(qa[positions])[:2])
    guard()
    for arm in ARMS:
        kind=arm.split('_')[0]
        path=RUN/arm/'weights.ckpt' if arm.endswith('tuned') else checkpoint
        meta=read(RUN/arm/'inference.json');assert sha(path)==meta['checkpoint_sha256']
        with np.load(RUN/arm/'prediction.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['scored_positions'],positions);prediction=z['probability']
        assert np.isfinite(prediction).all() and ((prediction>=0)&(prediction<=1)).all()
        assert abs(roc_auc_score(labels,prediction)-e['arms'][arm]['single_auc'])<1e-12
        margin=logit(np.clip(prediction,1e-6,1-1e-6))
        assert abs(roc_auc_score(labels,.9*base+.1*margin)-e['arms'][arm]['primary_blend_auc'])<1e-12
        for weight in p['diagnostic_weights']:
            delta=roc_auc_score(labels,(1-weight)*base+weight*margin)-e['baseline_auc']
            assert abs(delta-e['arms'][arm]['diagnostic_deltas'][str(weight)])<1e-12
        net,config=load_net(path)
        checks=[]
        for anchor in selected_anchors:
            guard()
            rows=np.flatnonzero(qa[positions]==anchor)
            rebuilt=predict_context(net,config,path,x,y[fit],contexts[kind][anchor],q[positions[rows]],p['prediction_batch'])
            error=float(np.max(abs(rebuilt-prediction[rows])))
            assert error<1e-7,(arm,anchor,error)
            checks.append(dict(anchor=anchor,rows=len(rows),maximum_error=error))
        if arm.endswith('tuned'):
            summary=read(RUN/arm/'complete.json');assert summary['steps']==p['steps']
            assert summary['changed_tensors']>0 and summary['checkpoint_sha256']==sha(path)
            history=read(RUN/arm/'history.json')
            assert [row['step'] for row in history]==list(range(1,p['steps']+1))
            assert all(np.isfinite(row['gradient_norm']) and row['gradient_scale']>0 for row in history)
            assert sum(row['overflow_retries'] for row in history)==summary['overflow_retries']
        evidence[arm]=dict(all_rows_auc_and_ids_checked=True,gpu_rebuilt_groups=checks)
        del net
        gc.collect();torch.cuda.empty_cache()
    primary=e['arms']['local_tuned']['primary_delta']
    gate=primary['delta']>=p['expansion_gate'] and primary['interval95'][0]>0 and e['primary_blend_delta_vs_random_tuned']>0
    assert gate==e['passes_expansion_gate']
    for f,h in p['protected'].items():assert sha(ROOT/f)==h,f
    result=dict(verified=True,all_input_and_incumbent_hashes_verified=True,
        verification_source_sha256=sha(Path(__file__)),
        metric_encoder_pools_and_training_tasks_rebuilt=True,numerical_encodings_and_row_ids_rebuilt=True,
        baseline_rebuild_max_error=float(np.max(abs(base_rebuilt-base))),
        prediction_checks=evidence,verification_seconds=time.time()-started,
        cumulative_seconds=time.time()-read(RUN/'clock.json')['started_epoch'],
        cumulative_deadline_met=time.time()<=deadline,passes_expansion_gate=gate,
        gpu_rebuild_scope='Two complete anchor groups per arm; not all GPU predictions rerun',
        validation=p['validation'])
    (RUN/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('VERIFICATION',json.dumps(result),flush=True)


if __name__=='__main__':main()
