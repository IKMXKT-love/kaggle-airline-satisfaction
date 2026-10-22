"""Bounded random/local context x frozen/tuned TabICL screen."""
import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from local_context_support import (RetrievalEncoder, TaskSampler, explicit_meta_batch,
                                   inference_contexts, predict_context)
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from tabicl import TabICLClassifier
from tabicl._model.tabicl import TabICL
from tabicl._sklearn.preprocessing import TransformToNumerical
from tabicl._finetune.data import move_meta_batch
from tabicl_finetune_support import inference_parameters, release

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
RUN = ROOT/'.artifact_runtime/local_context_20261021'
BASE = ROOT/'.artifact_runtime/public_pool_trial_20261020'
CHECKPOINT = ROOT/'.artifact_runtime/tabicl_probe_20261019/tabicl-classifier-v2-20260212.ckpt'
CHECKPOINT_SHA = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
PROTECTED = ['submission_public_pool_20261020.csv','CURRENT_BEST_RESULT.md',
             'CURRENT_BEST_SUBMISSION.json','CURRENT_RECOMMENDED_CANDIDATE.json']
ARMS = ['random_frozen','local_frozen','random_tuned','local_tuned']
SEED = 20261022


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.part')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def setup():
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def guard(reserve=60):
    if time.time() >= read(RUN/'clock.json')['deadline_epoch']-reserve:
        raise TimeoutError('Cumulative stage budget reached; preserve all artifacts')


def protected_check(expected):
    assert {k:sha(ROOT/k) for k in expected}==expected


def load_net(path=CHECKPOINT,training=False):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    net = TabICL(**saved['config'])
    net.load_state_dict(saved['state_dict'],strict=True)
    net = net.cuda()
    net.train(training)
    return net,saved['config']


def nearest_pools(metric,centers,count=8192):
    norms = (metric*metric).sum(1)
    result=[]
    with threadpool_limits(limits=4):
        for center in centers:
            distance = norms+float(center@center)-2*(metric@center)
            positions = np.argpartition(distance,count-1)[:count]
            # Stable index tie-break is deterministic and label independent.
            result.append(positions[np.lexsort((positions,distance[positions]))])
    return np.stack(result)


def prepare():
    if RUN.exists(): raise FileExistsError('Preserve existing stage; never reset its budget')
    RUN.mkdir(parents=True)
    started=time.time()
    write(RUN/'clock.json',dict(started_epoch=started,deadline_epoch=started+900,seconds=900))
    setup()
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    protected = {f:sha(ROOT/f) for f in PROTECTED}
    hashes = {f:sha(ROOT/f) for f in ['experiment_local_context.py','local_context_support.py',
        'tabicl_finetune_support.py','tests/test_local_context_contract.py','train.csv','test.csv',
        'LOCAL_CONTEXT_PLAN_20261021.md']}
    tr=pd.read_csv(ROOT/'train.csv')
    raw=tr.drop(columns=['id','satisfaction'])
    labels=tr.satisfaction.to_numpy(dtype='int64')
    fold=np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy',allow_pickle=False)
    expected=np.empty(len(tr),dtype='int8')
    for k,(_,e) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(raw,labels)):expected[e]=k
    np.testing.assert_array_equal(expected,fold)
    fitting,scored=np.flatnonzero(fold!=1),np.flatnonzero(fold==1)
    assert len(np.intersect1d(fitting,scored))==0
    np.savez(RUN/'indices.npz',fitting=fitting,outer_eval=scored)
    encoder=TransformToNumerical()
    x=np.asarray(encoder.fit_transform(raw.iloc[fitting]),dtype='float64')
    q=np.asarray(encoder.transform(raw.iloc[scored]),dtype='float64')
    joblib.dump(encoder,RUN/'numeric_encoder.joblib')
    np.save(RUN/'reference.npy',x,allow_pickle=False)
    np.save(RUN/'queries.npy',q,allow_pickle=False)
    np.savez(RUN/'labels.npz',reference_y=labels[fitting],scored_y=labels[scored],
             reference_id=tr.id.to_numpy()[fitting],scored_id=tr.id.to_numpy()[scored])
    metric_encoder=RetrievalEncoder().fit(raw.iloc[fitting])
    metric=metric_encoder.transform(raw.iloc[fitting])
    metric_q=metric_encoder.transform(raw.iloc[scored])
    write(RUN/'retrieval_encoder.json',metric_encoder.state())
    np.save(RUN/'reference_metric.npy',metric,allow_pickle=False)
    rng=np.random.default_rng(SEED)
    sample=rng.choice(len(fitting),65536,replace=False)
    with threadpool_limits(limits=4):
        cluster=MiniBatchKMeans(n_clusters=64,random_state=SEED,n_init=3,max_iter=80,batch_size=2048)
        cluster.fit(metric[sample])
        assignment=cluster.predict(metric)
        query_assignment=cluster.predict(metric_q)
    pools=nearest_pools(metric,cluster.cluster_centers_)
    np.savez(RUN/'retrieval.npz',centers=cluster.cluster_centers_,fit_sample=sample,
             assignment=assignment,query_assignment=query_assignment,pools=pools)
    joblib.dump(cluster,RUN/'cluster.joblib')
    with np.load(BASE/'public_only_fold1_validation.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['index'],scored)
        np.save(RUN/'baseline_logit.npy',z['logit'],allow_pickle=False)
    for name in ['public_only_fold1_validation.npz','public_only_fold1.npz','matrix_manifest.json']:
        hashes[str((BASE/name).relative_to(ROOT))]=sha(BASE/name)
    guard()
    checks=[];choice=None
    for context_size,query_size in [(2048,512),(1024,256)]:
        try:
            probe=resource_probe(x,labels[fitting],assignment,pools,context_size,query_size)
        except torch.OutOfMemoryError as error:
            checks.append(dict(context_size=context_size,error='CUDA memory',detail=str(error)[:300]))
            error.__traceback__=None
            gc.collect();torch.cuda.empty_cache();continue
        checks.append(probe)
        for size in [len(scored),40000]:
            positions=np.arange(len(scored))
            if size<len(scored):
                positions,_=train_test_split(positions,train_size=size,stratify=labels[scored],random_state=SEED)
                positions=np.sort(positions)
            counts=np.bincount(query_assignment[positions],minlength=64)
            chunks=sum(math.ceil(n/1024) for n in counts if n)
            for steps in [256,128]:
                train_cost=2*steps*probe['worst_complete_training_step_seconds']
                prediction_cost=4*chunks*probe['worst_context_and_1024_query_seconds']
                estimate=time.time()-started+1.3*(train_cost+prediction_cost)+90
                if estimate<900:
                    choice=dict(context_size=context_size,query_size=query_size,steps=steps,
                                scored_positions=positions,projection_seconds=estimate,
                                projected_train_seconds=train_cost,projected_prediction_seconds=prediction_cost)
                    break
            if choice:break
        if choice:break
    write(RUN/'resource_probes.json',checks)
    if choice is None: raise TimeoutError('No complete four-arm experiment fits the cumulative budget')
    positions=choice.pop('scored_positions')
    contexts=inference_contexts(assignment,pools,choice['context_size'])
    sampler=TaskSampler(assignment,pools,choice['context_size'],choice['query_size'],seed=SEED)
    tasks=[sampler.sample(k) for k in range(choice['steps'])]
    queries=np.full((choice['steps'],choice['query_size']),-1,dtype='int64')
    for k,t in enumerate(tasks):queries[k,:len(t['query'])]=t['query']
    np.savez(RUN/'tasks.npz',random=np.stack([t['random'] for t in tasks]),
        local=np.stack([t['local'] for t in tasks]),query=queries,
        query_count=np.array([len(t['query']) for t in tasks]),anchor=np.array([t['anchor'] for t in tasks]))
    np.savez(RUN/'contexts.npz',**contexts)
    np.save(RUN/'scored_positions.npy',positions,allow_pickle=False)
    runtime={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'.artifact_runtime/tabicl_runtime/tabicl').rglob('*.py')}
    runtime_hash=hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()
    protocol=dict(**choice,seed=SEED,outer_fold=1,reference_rows=len(fitting),outer_eval_rows=len(scored),
        scored_rows=len(positions),anchors=64,retrieval_pool=8192,background_rows=128,
        norm_methods=['none','power'],training_views=2,inference_views=2,prediction_batch=1024,
        learning_rate=1e-5,weight_decay=.01,warmup_fraction=.1,grad_clip=1.,amp='float16',
        fixed_update_count_no_score_selection=True,query_tasks_identical_across_training_arms=True,
        input='21 raw fields, numerical encoder fitted on outer train; no original teachers or target encodings',
        approximation='Queries assigned to nearest feature-only anchor; related queries share retrieved context',
        arms=ARMS,primary_candidate='local_tuned',primary_weight=.1,diagnostic_weights=[.05,.2],
        expansion_gate=.0003,require_paired_lower_above_zero=True,
        require_primary_candidate_better_than_random_tuned=True,
        checkpoint=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=CHECKPOINT_SHA,
        protected=protected,input_sha256=hashes,runtime_source_sha256=runtime_hash,
        budget=dict(preparation_probe_and_plan_seconds=time.time()-started,deadline_epoch=started+900,
                    verification_reserve_seconds=60),
        validation='New branch outer fold isolated; fusion with fixed public OOF remains conditional, not end-to-end nested.',
        historical_folds_and_recipes_seen=True,no_submission=True)
    write(RUN/'protocol.json',protocol)
    for f in ['experiment_local_context.py','local_context_support.py','tabicl_finetune_support.py']:
        (RUN/('snapshot_'+f)).write_bytes((ROOT/f).read_bytes())
    protected_check(protected)
    print('FROZEN_PROTOCOL',json.dumps({k:protocol[k] for k in ['context_size','query_size','steps','scored_rows','projection_seconds','budget']}),flush=True)


def training_step(net,opt,scaler,task,x,y,kind,seed):
    batch=explicit_meta_batch(x,y,task[kind],task['query'],seed)
    batch=move_meta_batch(batch,torch.device('cuda'))
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        logits=net(batch.X,batch.y_train.float())[...,:2]
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,2),batch.y_query.reshape(-1))
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    scaler.scale(loss).backward();scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
    scaler.step(opt);scaler.update()
    return float(loss.detach()),float(norm)


def resource_probe(x,y,assignment,pools,context_size,query_size):
    guard()
    sampler=TaskSampler(assignment,pools,context_size,query_size,seed=SEED)
    net,config=load_net(training=True)
    before=next(net.parameters()).detach().cpu().clone()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=.01)
    scaler=torch.GradScaler('cuda')
    timed=[]
    torch.cuda.reset_peak_memory_stats()
    for kind in ['random','local']:
        for step in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            task=sampler.sample(91000+step)
            loss,norm=training_step(net,opt,scaler,task,x,y,kind,SEED+step)
            torch.cuda.synchronize()
            if step:timed.append(time.perf_counter()-start)
    changed=not torch.equal(before,next(net.parameters()).detach().cpu())
    assert changed
    peak=torch.cuda.max_memory_allocated()/1024**3
    del net,opt,scaler,before
    gc.collect();torch.cuda.empty_cache()
    net,config=load_net()
    contexts=inference_contexts(assignment,pools,context_size)
    durations=[]
    # All probe queries are legal training rows; no heldout outcomes are scored.
    for kind in ['random','local']:
        for anchor in [0,int(np.argmax(np.bincount(assignment,minlength=64)))]:
            context=contexts[kind][anchor]
            query=np.setdiff1d(np.arange(len(x)),context)[:1024]
            torch.cuda.synchronize();start=time.perf_counter()
            p=predict_context(net,config,CHECKPOINT,x,y,context,x[query])
            torch.cuda.synchronize();durations.append(time.perf_counter()-start)
            assert np.isfinite(p).all()
    # Reuse-only adapter versus official prediction, matched batch and inputs.
    params=inference_parameters(CHECKPOINT,2)
    official=TabICLClassifier(**params)
    official.fit(x[:512],y[:512])
    expected=official.predict_proba(x[512:640])[:,1]
    actual=predict_context(net,config,CHECKPOINT,x,y,np.arange(512),x[512:640],query_batch=128)
    error=float(np.max(abs(expected-actual)))
    assert error<1e-7,error
    release(official);del official,net
    gc.collect();torch.cuda.empty_cache()
    result=dict(context_size=context_size,query_size=query_size,
        complete_training_step_seconds=timed,worst_complete_training_step_seconds=max(timed),
        context_and_1024_query_seconds=durations,worst_context_and_1024_query_seconds=max(durations),
        peak_training_gpu_gb=peak,real_gradient_update_verified=changed,
        reuse_adapter_max_difference_vs_official=error,probe_weights_discarded=True)
    print('RESOURCE_PROBE',json.dumps(result),flush=True)
    return result


def fine_tune(kind,protocol,x,y,tasks):
    guard()
    folder=RUN/(kind+'_tuned');folder.mkdir()
    torch.manual_seed(SEED);np.random.seed(SEED)
    net,config=load_net(training=True)
    opt=torch.optim.AdamW(net.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    scaler=torch.GradScaler('cuda')
    before=next(net.parameters()).detach().cpu().clone()
    start=time.time();history=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(protocol['steps']):
        guard()
        warm=max(1,round(protocol['steps']*protocol['warmup_fraction']))
        factor=(step+1)/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,protocol['steps']-warm-1)))
        for group in opt.param_groups:group['lr']=protocol['learning_rate']*factor
        count=int(tasks['query_count'][step])
        task={kind:tasks[kind][step], 'query':tasks['query'][step,:count]}
        loss,norm=training_step(net,opt,scaler,task,x,y,kind,SEED+step*7919)
        history.append(dict(step=step+1,loss=loss,gradient_norm=norm,lr=opt.param_groups[0]['lr'],query_rows=count))
        if step==0 or (step+1)%32==0:
            write(folder/'history.json',history)
            print('TRAIN',kind,step+1,'loss',loss,'seconds',round(time.time()-start,2),flush=True)
    assert not torch.equal(before,next(net.parameters()).detach().cpu())
    state={n:v.detach().cpu() for n,v in net.state_dict().items()}
    original_state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)['state_dict']
    changed_tensors=sum(not torch.equal(v,original_state[n]) for n,v in state.items())
    assert changed_tensors>0
    torch.save(dict(config=config,state_dict=state),folder/'weights.ckpt')
    write(folder/'history.json',history)
    write(folder/'complete.json',dict(steps=len(history),seconds=time.time()-start,
        query_supervised_rows=sum(r['query_rows'] for r in history),
        distinct_query_rows=len(np.unique(np.concatenate([tasks['query'][k,:int(tasks['query_count'][k])] for k in range(len(history))]))),
        changed_tensors=changed_tensors,
        first_parameter_changed=True,checkpoint_sha256=sha(folder/'weights.ckpt'),
        peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3))
    del net,opt,scaler,before,state,original_state
    gc.collect();torch.cuda.empty_cache()


def infer_arm(arm,protocol,x,y,q,assignments,contexts,scored_positions):
    guard()
    folder=RUN/arm;folder.mkdir(exist_ok=True)
    kind=arm.split('_')[0]
    checkpoint=RUN/arm/'weights.ckpt' if arm.endswith('tuned') else CHECKPOINT
    net,config=load_net(checkpoint)
    output=np.full(len(scored_positions),np.nan)
    counts=np.zeros(len(output),dtype='int8')
    grouped=assignments[scored_positions]
    records=[];start=time.time()
    for anchor in range(protocol['anchors']):
        guard()
        rows=np.flatnonzero(grouped==anchor)
        if not len(rows):continue
        began=time.time()
        output[rows]=predict_context(net,config,checkpoint,x,y,contexts[kind][anchor],q[scored_positions[rows]],protocol['prediction_batch'])
        counts[rows]+=1
        records.append(dict(anchor=anchor,rows=len(rows),seconds=time.time()-began))
        if (anchor+1)%16==0:print('PREDICT',arm,anchor+1,'seconds',round(time.time()-start,2),flush=True)
    assert (counts==1).all() and np.isfinite(output).all()
    np.savez(folder/'prediction.npz',scored_positions=scored_positions,probability=output)
    write(folder/'inference.json',dict(seconds=time.time()-start,groups=records,checkpoint_sha256=sha(checkpoint)))
    del net
    gc.collect();torch.cuda.empty_cache()


def auc_pair(y,p,q):
    def placements(s):
        pos,neg=s[y==1],s[y==0];sp,sn=np.sort(pos),np.sort(neg)
        return ((np.searchsorted(sn,pos,'left')+np.searchsorted(sn,pos,'right'))/(2*len(neg)),
                1-(np.searchsorted(sp,neg,'left')+np.searchsorted(sp,neg,'right'))/(2*len(pos)))
    a,b=placements(p);c,d=placements(q)
    delta=float((a-c).mean());se=float(np.sqrt((a-c).var(ddof=1)/len(a)+(b-d).var(ddof=1)/len(b)))
    return dict(delta=delta,se=se,interval95=[delta-1.96*se,delta+1.96*se])


def evaluate(protocol):
    guard(reserve=30)
    positions=np.load(RUN/'scored_positions.npy',allow_pickle=False)
    with np.load(RUN/'labels.npz',allow_pickle=False) as z:y=z['scored_y'][positions]
    base=np.load(RUN/'baseline_logit.npy',allow_pickle=False)[positions]
    baseline_auc=float(roc_auc_score(y,base));scores={};predictions={}
    for arm in ARMS:
        with np.load(RUN/arm/'prediction.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['scored_positions'],positions);p=z['probability']
        predictions[arm]=p
        margin=logit(np.clip(p,1e-6,1-1e-6));blend=.9*base+.1*margin
        scores[arm]=dict(single_auc=float(roc_auc_score(y,p)),primary_blend_auc=float(roc_auc_score(y,blend)),
            primary_delta=auc_pair(y,blend,base),diagnostic_deltas={str(w):float(roc_auc_score(y,(1-w)*base+w*margin))-baseline_auc for w in protocol['diagnostic_weights']})
    primary=scores['local_tuned']['primary_delta']
    comparison=scores['local_tuned']['primary_blend_auc']-scores['random_tuned']['primary_blend_auc']
    accepted=primary['delta']>=protocol['expansion_gate'] and primary['interval95'][0]>0 and comparison>0
    result=dict(completed=True,baseline_auc=baseline_auc,arms=scores,passes_expansion_gate=accepted,
        local_tuned_vs_random_tuned_single=auc_pair(y,predictions['local_tuned'],predictions['random_tuned']),
        local_tuned_vs_local_frozen_single=auc_pair(y,predictions['local_tuned'],predictions['local_frozen']),
        local_frozen_vs_random_frozen_single=auc_pair(y,predictions['local_frozen'],predictions['random_frozen']),
        primary_blend_delta_vs_random_tuned=comparison,scored_rows=len(y),new_public_score=None,
        elapsed_seconds=time.time()-read(RUN/'clock.json')['started_epoch'],validation=protocol['validation'])
    write(RUN/'evaluation.json',result);print('EVALUATION',json.dumps(result),flush=True)


def run():
    setup();protocol=read(RUN/'protocol.json');protected_check(protocol['protected'])
    for f in ['experiment_local_context.py','local_context_support.py','tabicl_finetune_support.py']:
        assert sha(ROOT/f)==protocol['input_sha256'][f]
    x=np.load(RUN/'reference.npy',allow_pickle=False);q=np.load(RUN/'queries.npy',allow_pickle=False)
    with np.load(RUN/'labels.npz',allow_pickle=False) as z:y=z['reference_y']
    with np.load(RUN/'tasks.npz',allow_pickle=False) as z:tasks={k:z[k] for k in z.files}
    with np.load(RUN/'contexts.npz',allow_pickle=False) as z:contexts={k:z[k] for k in z.files}
    with np.load(RUN/'retrieval.npz',allow_pickle=False) as z:assignments=z['query_assignment']
    positions=np.load(RUN/'scored_positions.npy',allow_pickle=False)
    for kind in ['random','local']:fine_tune(kind,protocol,x,y,tasks)
    for arm in ARMS:infer_arm(arm,protocol,x,y,q,assignments,contexts,positions)
    evaluate(protocol);protected_check(protocol['protected'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    try:prepare() if args.action=='prepare' else run()
    except BaseException as exc:
        if RUN.exists():write(RUN/'stopped.json',dict(error=type(exc).__name__,detail=str(exc),time_epoch=time.time()))
        raise
