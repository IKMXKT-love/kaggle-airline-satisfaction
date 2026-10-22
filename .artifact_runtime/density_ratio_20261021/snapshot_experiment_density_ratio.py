"""Bounded class-conditional domain ratio screen; no submission generation."""
import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from density_ratio_support import ARMS, RatioMLP, draw_task, loss_for_arm, predict
from domain_transfer_support import RawEncoder
from experiment_domain_transfer import ROOT, BASE, FOLDS, MODEL, PROTECTED, read, write, sha, setup, state_hash
from multitask_support import cpu_state, paired
from optimized_features import normalize_raw

sys.stdout.reconfigure(encoding='utf-8')
RUN = ROOT/'.artifact_runtime/density_ratio_20261021'
CACHE = ROOT/'.artifact_runtime/optimized_cache'
TRAIN = dict(seed=20261021, group_size=512, epoch_divisor=2048, learning_rate=.0015,
             min_lr_fraction=.08, weight_decay=.01, gradient_clip=1., prediction_batch=4096)


def guard(reserve=45):
    if time.time() >= read(RUN/'clock.json')['deadline_epoch']-reserve:
        raise TimeoutError('Cumulative 900s budget reached; preserve all artifacts')


def load_values(phase, name):
    return {k:torch.as_tensor(np.load(RUN/phase/f'{name}_{k}.npy',allow_pickle=False),device='cuda') for k in ['num','missing','cat']}


def save_values(phase, name, encoder, frame):
    values = encoder.transform(frame)
    for k in ['num','missing','cat']: np.save(RUN/phase/f'{name}_{k}.npy',values[k],allow_pickle=False)


def labels_indices():
    with np.load(RUN/'labels.npz',allow_pickle=False) as z: labels = {k:z[k].copy() for k in z.files}
    with np.load(RUN/'indices.npz',allow_pickle=False) as z: indices = {k:z[k].copy() for k in z.files}
    return labels,indices


def teacher_for(phase, indices, labels):
    rows = indices['fit'] if phase == 'selection' else indices['outer_train']
    pi = float(labels['y'][rows].mean())
    shift = float(logit(pi)-logit(labels['original_y'].mean()))
    q = np.load(RUN/'teacher_margin.npy',allow_pickle=False)
    return q+shift,pi,rows


def task_state(phase, labels, indices):
    q,pi,rows = teacher_for(phase,indices,labels)
    c,o = load_values(phase,'competition'),load_values(phase,'original')
    yc = torch.as_tensor(labels['y'][rows],device='cuda',dtype=torch.float32)
    yo = torch.as_tensor(labels['original_y'],device='cuda',dtype=torch.float32)
    pools = [torch.nonzero(z==v).flatten() for z in [yc,yo] for v in [0,1]]
    teacher = torch.tensor(q[rows],device='cuda',dtype=torch.float32)
    return c,o,yc,yo,pools,teacher,pi


def update(model, optimizer, task, arm, pi):
    inputs,y,d,q = task
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss = loss_for_arm(model(inputs),y,d,q,arm,pi)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),TRAIN['gradient_clip'],foreach=True,error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(),lr=TRAIN['learning_rate'],weight_decay=TRAIN['weight_decay'],fused=True)


def prepare():
    if RUN.exists(): raise FileExistsError('Preserve existing experiment and clock')
    RUN.mkdir(parents=True)
    started = time.time()
    write(RUN/'clock.json',dict(started_epoch=started,deadline_epoch=started+900,seconds=900,
        includes='all data preparation, fingerprint checks, probe, six fits, predictions and independent verification'))
    setup()
    protected = {f:sha(ROOT/f) for f in PROTECTED}
    files = ['train.csv','test.csv','external_data/train.csv','external_data/test.csv',
        'experiment_density_ratio.py','density_ratio_support.py','domain_transfer_support.py','multitask_support.py',
        'experiment_domain_transfer.py','optimized_features.py','verify_density_ratio.py',
        'tests/test_density_ratio_contract.py','DENSITY_RATIO_PLAN_20261021.md']
    files += [str(FOLDS.relative_to(ROOT)),str((BASE/'public_only_fold2_validation.npz').relative_to(ROOT)),
        str((BASE/'public_only_fold2.npz').relative_to(ROOT)),str((CACHE/'metadata.json').relative_to(ROOT)),
        str((CACHE/'train_priors.npy').relative_to(ROOT))]
    files += [str((CACHE/f'original_prior_depth{d}.ubj').relative_to(ROOT)) for d in [6,3,9]]
    hashes = {f:sha(ROOT/f) for f in files}
    train,test = pd.read_csv(ROOT/'train.csv'),pd.read_csv(ROOT/'test.csv')
    features = [c for c in test if c!='id']
    x,y,ids = train[features],train.satisfaction.to_numpy(dtype='int64'),train.id.to_numpy()
    original = pd.concat([pd.read_csv(ROOT/'external_data'/f) for f in ['train.csv','test.csv']],ignore_index=True)
    nx,nt,no = [normalize_raw(f) for f in [x,test[features],original[features]]]
    sig = read(CACHE/'metadata.json')['signature']
    import hashlib
    for name,frame in [('competition_train_features',nx),('competition_test_features',nt)]:
        assert hashlib.sha256(pd.util.hash_pandas_object(frame.astype('string'),index=False).to_numpy().tobytes()).hexdigest()==sig[name]
    assert sig['features']==features and sig['prior_specs']==[[6,1200],[3,1800],[9,1000]]
    assert sig['original_file_hashes']==[hashes[f'external_data/{f}.csv'] for f in ['train','test']]
    assert sig['xgboost_version']==__import__('xgboost').__version__
    cf = pd.util.hash_pandas_object(pd.concat([nx,nt],ignore_index=True).astype('string').fillna('__MISSING__'),index=False)
    of = pd.util.hash_pandas_object(no.astype('string').fillna('__MISSING__'),index=False)
    keep = ~of.isin(set(cf))
    ox = original.loc[keep,features].reset_index(drop=True)
    oy = original.loc[keep,'satisfaction'].eq('satisfied').to_numpy(dtype='int64')
    priors = np.load(CACHE/'train_priors.npy',allow_pickle=False)
    assert priors.shape==(len(y),3) and np.isfinite(priors).all()
    q = logit(np.clip(priors.astype('float64'),1e-6,1-1e-6)).mean(1)
    np.save(RUN/'teacher_margin.npy',q,allow_pickle=False)
    folded = np.load(FOLDS,allow_pickle=False)
    expected = np.empty(len(y),dtype='int8')
    for f,(_,v) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y)): expected[v]=f
    np.testing.assert_array_equal(folded,expected)
    outer,scored = np.flatnonzero(folded!=2),np.flatnonzero(folded==2)
    fit,stop = train_test_split(outer,test_size=.08,random_state=TRAIN['seed'],stratify=y[outer])
    fit,stop = np.sort(fit),np.sort(stop)
    np.savez(RUN/'indices.npz',fit=fit,stopping=stop,outer_train=outer,scored=scored,original_rows=np.flatnonzero(keep))
    np.savez(RUN/'labels.npz',y=y,original_y=oy,id=ids)
    with np.load(BASE/'public_only_fold2_validation.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['index'],scored)
        np.save(RUN/'baseline_logit.npy',z['logit'],allow_pickle=False)
    for phase,rows in [('selection',fit),('refit',outer)]:
        guard(); (RUN/phase).mkdir()
        encoder = RawEncoder().fit(pd.concat([x.iloc[rows],ox],ignore_index=True))
        write(RUN/phase/'encoder.json',encoder.state())
        save_values(phase,'competition',encoder,x.iloc[rows])
        save_values(phase,'original',encoder,ox)
        save_values(phase,'scored',encoder,x.iloc[scored])
        if phase=='selection': save_values(phase,'stopping',encoder,x.iloc[stop])
    labels,indices = labels_indices()
    resources = probe(labels,indices)
    elapsed = time.time()-started
    projections = {}
    for epochs in [24,16,12]:
        steps = 3*epochs*(math.ceil(len(fit)/2048)+math.ceil(len(outer)/2048))
        train_seconds = steps*resources['worst_seconds_per_step']
        infer_seconds = 3*((epochs+1)*len(stop)+2*len(scored))*resources['prediction_seconds_per_row']
        projections[str(epochs)] = elapsed+1.5*(train_seconds+infer_seconds)+90
    permitted = [e for e in [24,16,12] if projections[str(e)]<=900]
    if not permitted:
        write(RUN/'resource_stop.json',dict(projections=projections,elapsed=elapsed))
        raise TimeoutError('No complete three-arm trial fits budget')
    protocol = dict(arms=ARMS,primary_candidate='conditional_ratio',outer_fold=2,model=MODEL,training=TRAIN,max_epochs=permitted[0],
        primary_blend_weight=.1,diagnostic_weights=[.05,.2],expansion_gate=.0003,
        protected=protected,input_sha256=hashes,original_rows=len(oy),removed_original_overlap=int((~keep).sum()),
        fit_rows=len(fit),stopping_rows=len(stop),refit_rows=len(outer),scored_rows=len(scored),
        teacher='mean clipped source-only XGBoost logits; fixed no competition labels; per-phase class-prior shift',
        source_teacher_as_domain_input=False,extra_source_field_used=False,contrast_limit=8.,
        selection='inner AUC including epoch0; refit same schedule prefix at chosen epoch, never score-based weights',
        encoder_scope='unlabeled C fitting rows plus retained O rows; no stopping/scored statistics',
        task='balanced C0,C1,O0,O1 512 each; C objectives weighted to natural pi_C; all arms identical task RNG',
        epoch_definition='ceil(C rows/2048) steps with replacement; actual C rows per step1024, O1024',
        public_validation_caveat='Conditional screen: public upstream OOF inherits early-stopping and indirect label dependence, not nested CV; historical folds seen.',
        costs=dict(preparation_and_probe_seconds=elapsed,conservative_projections=projections),
        no_submission=True,no_weight_search=True)
    write(RUN/'protocol.json',protocol)
    for f in files:
        if f.endswith('.py') or f.endswith('.md'): (RUN/('snapshot_'+Path(f).name)).write_bytes((ROOT/f).read_bytes())
    assert protected=={f:sha(ROOT/f) for f in protected}
    print('PROTOCOL',protocol,flush=True)


def amend_encoder():
    """Pre-score correction: avoid encoding unknown-category status as domain membership."""
    setup(); protocol=read(RUN/'protocol.json')
    assert not (RUN/'evaluation.json').exists()
    assert all(not (RUN/'selection'/a).exists() for a in ARMS)
    archive=RUN/'resource_only_pre_encoder_amendment'; archive.mkdir()
    # Verified absolute paths stay inside this experiment; no deletion or cross-shell operation.
    for phase in ['selection','refit']:
        src=(RUN/phase).resolve(); dst=(archive/phase).resolve()
        assert src.is_relative_to(RUN.resolve()) and dst.is_relative_to(RUN.resolve())
        src.rename(dst); (RUN/phase).mkdir()
    for name in ['protocol.json','probe.json']:
        (archive/name).write_bytes((RUN/name).read_bytes())
    for f in RUN.glob('snapshot_*'): (archive/f.name).write_bytes(f.read_bytes())
    train=pd.read_csv(ROOT/'train.csv'); features=read(archive/'selection/encoder.json')['columns']; x=train[features]
    original=pd.concat([pd.read_csv(ROOT/'external_data'/f) for f in ['train.csv','test.csv']],ignore_index=True)
    labels,indices=labels_indices(); ox=original.iloc[indices['original_rows']][features].reset_index(drop=True)
    for phase,rows in [('selection',indices['fit']),('refit',indices['outer_train'])]:
        guard(); encoder=RawEncoder().fit(pd.concat([x.iloc[rows],ox],ignore_index=True))
        write(RUN/phase/'encoder.json',encoder.state())
        save_values(phase,'competition',encoder,x.iloc[rows]); save_values(phase,'original',encoder,ox)
        save_values(phase,'scored',encoder,x.iloc[indices['scored']])
        if phase=='selection': save_values(phase,'stopping',encoder,x.iloc[indices['stopping']])
    resources=probe(labels,indices)
    elapsed=time.time()-read(RUN/'clock.json')['started_epoch']; projections={}
    for epochs in [24,16,12]:
        steps=3*epochs*(math.ceil(len(indices['fit'])/2048)+math.ceil(len(indices['outer_train'])/2048))
        inf=3*((epochs+1)*len(indices['stopping'])+2*len(indices['scored']))*resources['prediction_seconds_per_row']
        projections[str(epochs)]=elapsed+1.5*(steps*resources['worst_seconds_per_step']+inf)+90
    allowed=[e for e in [24,16,12] if projections[str(e)]<=900]
    if not allowed: raise TimeoutError('Amended complete trial does not fit original clock')
    protocol['max_epochs']=allowed[0]
    protocol['encoder_scope']='unlabeled C fitting rows plus retained O rows; no stopping/scored statistics'
    protocol['pre_score_amendment']='Joint vocabulary/scales prevent asymmetric source unknown-token artifacts; no stopping or scored outcomes observed; original clock retained and first encoders/probe preserved.'
    protocol['costs']=dict(preparation_and_probe_seconds=elapsed,conservative_projections=projections)
    for f in protocol['input_sha256']:
        if f.endswith('.py') or f.endswith('.md'):
            protocol['input_sha256'][f]=sha(ROOT/f)
            (RUN/('snapshot_'+Path(f).name)).write_bytes((ROOT/f).read_bytes())
    write(RUN/'protocol.json',protocol); print('AMENDED_PROTOCOL',protocol,flush=True)


def probe(labels,indices):
    guard()
    c,o,yc,yo,pools,q,pi = task_state('selection',labels,indices)
    generator = torch.Generator(device='cuda').manual_seed(TRAIN['seed']+100)
    state = read(RUN/'selection/encoder.json')
    times = {}
    torch.cuda.reset_peak_memory_stats()
    for arm in ARMS:
        torch.manual_seed(TRAIN['seed']); model = RatioMLP(state,MODEL).cuda(); optimizer = optimizer_for(model)
        first = model.layers[0].weight.detach().clone()
        for step in range(42):
            if step==10: torch.cuda.synchronize(); t=time.perf_counter()
            task = draw_task(c,o,yc,yo,pools,q,generator,TRAIN['group_size'])
            update(model,optimizer,task,arm,pi)
        torch.cuda.synchronize(); times[arm]=(time.perf_counter()-t)/32
        assert (model.layers[0].weight.detach()-first).abs().max()>0
        del optimizer,first
    small={k:v[:8192] for k,v in c.items()}
    predict(model,small,q[:8192],'conditional_ratio',float(logit(pi)))
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(4): predict(model,small,q[:8192],'conditional_ratio',float(logit(pi)))
    torch.cuda.synchronize()
    result=dict(step_seconds=times,worst_seconds_per_step=max(times.values()),
        prediction_seconds_per_row=(time.perf_counter()-t)/(4*8192),probe_discarded=True,
        peak_gb=torch.cuda.max_memory_allocated()/1024**3,device=torch.cuda.get_device_name(0))
    write(RUN/'probe.json',result); print('PROBE',result,flush=True)
    del c,o,yc,yo,pools,q,task,model
    gc.collect(); torch.cuda.empty_cache()
    return result


def train_arm(phase,arm,protocol,labels,indices):
    guard(); path=RUN/phase/arm
    if path.exists(): raise FileExistsError(str(path))
    path.mkdir(); started=time.time()
    c,o,yc,yo,pools,q,pi=task_state(phase,labels,indices)
    allq,_,rows=teacher_for(phase,indices,labels)
    state=read(RUN/phase/'encoder.json')
    torch.manual_seed(TRAIN['seed']); model=RatioMLP(state,MODEL).cuda()
    initial=state_hash(model); optimizer=optimizer_for(model)
    generator=torch.Generator(device='cuda').manual_seed(TRAIN['seed']+100)
    epochs=protocol['max_epochs'] if phase=='selection' else read(RUN/'selection'/arm/'complete.json')['best_epoch']
    steps=math.ceil(len(rows)/TRAIN['epoch_divisor']); updates=0; history=[]
    best,best_epoch=-1.,0
    stopping=load_values(phase,'stopping') if phase=='selection' else None
    qs=torch.tensor(allq[indices['stopping']],device='cuda',dtype=torch.float32) if phase=='selection' else None
    if phase=='selection':
        p,_=predict(model,stopping,qs,arm,float(logit(pi)))
        best=float(roc_auc_score(labels['y'][indices['stopping']],p))
        history.append(dict(epoch=0,stopping_auc=best)); torch.save(cpu_state(model),path/'best_weights.pt')
    for epoch in range(1,epochs+1):
        guard(); t=time.time(); total=0.
        lr=TRAIN['learning_rate']*(TRAIN['min_lr_fraction']+(1-TRAIN['min_lr_fraction'])*.5*(1+math.cos(math.pi*(epoch-1)/max(1,protocol['max_epochs']-1))))
        for group in optimizer.param_groups: group['lr']=lr
        model.train()
        for step in range(steps):
            if step%32==0: guard()
            task=draw_task(c,o,yc,yo,pools,q,generator,TRAIN['group_size'])
            total+=float(update(model,optimizer,task,arm,pi)); updates+=1
        row=dict(epoch=epoch,lr=lr,objective_bce=total/steps,seconds=time.time()-t)
        if phase=='selection':
            p,_=predict(model,stopping,qs,arm,float(logit(pi)))
            row['stopping_auc']=float(roc_auc_score(labels['y'][indices['stopping']],p))
            if row['stopping_auc']>best:
                best,best_epoch=row['stopping_auc'],epoch; torch.save(cpu_state(model),path/'best_weights.pt')
        history.append(row); write(path/'history.json',history)
        if epoch==1 or epoch%4==0 or epoch==epochs: print('EPOCH',phase,arm,row,flush=True)
    if phase=='refit':
        torch.save(cpu_state(model),path/'weights.pt')
        values=load_values(phase,'scored'); qq=torch.tensor(allq[indices['scored']],device='cuda',dtype=torch.float32)
        p,b=predict(model,values,qq,arm,float(logit(pi)))
        np.savez(path/'scored_prediction.npz',index=indices['scored'],id=labels['id'][indices['scored']],probability=p,mean_contrast=b)
    result=dict(phase=phase,arm=arm,initial_state_hash=initial,epochs_completed=epochs,updates=updates,
        best_epoch=best_epoch if phase=='selection' else epochs,best_stopping_auc=best if phase=='selection' else None,
        competition_draws=updates*1024,original_draws=updates*1024,
        original_labels_used=arm=='conditional_ratio',prior=pi,seconds=time.time()-started)
    write(path/'complete.json',result); print('FIT_COMPLETE',result,flush=True)
    del model,optimizer,c,o,yc,yo,pools,q,stopping,qs
    gc.collect(); torch.cuda.empty_cache()


def evaluate(protocol,labels,indices):
    guard(25); y=labels['y'][indices['scored']]; base=np.load(RUN/'baseline_logit.npy',allow_pickle=False)
    base_auc=float(roc_auc_score(y,base)); scores={}; result={}
    teacher=np.load(RUN/'teacher_margin.npy',allow_pickle=False)[indices['scored']]
    for arm in ARMS:
        with np.load(RUN/'refit'/arm/'scored_prediction.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['index'],indices['scored']); p=z['probability']; b=z['mean_contrast']
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        margin=logit(np.clip(p,1e-6,1-1e-6)); scores[arm]=.9*base+.1*margin
        result[arm]=dict(single_auc=float(roc_auc_score(y,p)),primary_blend_auc=float(roc_auc_score(y,scores[arm])),
            primary_blend_delta=paired(y,scores[arm],base),contrast_quantiles=np.quantile(b,[0,.01,.5,.99,1]).tolist(),
            diagnostic_weights={str(w):float(roc_auc_score(y,(1-w)*base+w*margin))-base_auc for w in [.05,.2]})
    comparisons={arm:paired(y,scores['conditional_ratio'],scores[arm]) for arm in ['direct','teacher_offset']}
    delta=result['conditional_ratio']['primary_blend_delta']
    passed=delta['delta']>=protocol['expansion_gate'] and delta['interval95'][0]>0 and all(v['delta']>0 for v in comparisons.values())
    evaluation=dict(baseline_auc=base_auc,arms=result,ratio_vs_controls=comparisons,
        teacher_alone_auc=float(roc_auc_score(y,teacher)),teacher_10pct_blend_auc=float(roc_auc_score(y,.9*base+.1*teacher)),
        passes_expansion_gate=passed,cumulative_seconds=time.time()-read(RUN/'clock.json')['started_epoch'],
        caveat=protocol['public_validation_caveat'])
    write(RUN/'evaluation.json',evaluation); print('EVALUATION',evaluation,flush=True)
    assert protocol['protected']=={f:sha(ROOT/f) for f in protocol['protected']}


def run():
    setup(); protocol=read(RUN/'protocol.json'); labels,indices=labels_indices()
    for phase in ['selection','refit']:
        for arm in ARMS: train_arm(phase,arm,protocol,labels,indices)
    evaluate(protocol,labels,indices)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','amend_encoder','run']); args=parser.parse_args()
    try: {'prepare':prepare,'amend_encoder':amend_encoder,'run':run}[args.action]()
    except BaseException as exc:
        if RUN.exists(): write(RUN/'stopped.json',dict(error=type(exc).__name__,detail=str(exc),time_epoch=time.time()))
        raise
