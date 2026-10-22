"""Bounded four-arm meta-model screen on frozen public prediction arrays."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'.artifact_runtime/joint_upstream_20261021'
BASE = ROOT/'.artifact_runtime/public_pool_trial_20261020'
sys.stdout.reconfigure(encoding='utf-8')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, data):
    temp = p.with_name(p.name+'.part')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class Matrix:
    """Expose only the frozen public246 columns and this arm's two possible extras."""
    def __init__(self, base, extra, columns):
        self.base, self.extra, self.columns = base, extra, columns
        self.shape = (len(base),246+len(columns))

    def __getitem__(self, item):
        rows, cols = item
        first = np.asarray(self.base[rows,:246])
        if self.columns:
            first = np.column_stack([first,self.extra[rows][:,self.columns]])
        return first[:,cols]


def main():
    protocol = read(RUN/'protocol.json')
    deadline = protocol['deadline_epoch']
    fit_deadline = deadline-85
    if (RUN/'evaluation.json').exists():raise FileExistsError('Preserve completed screen.')
    started = time.time()

    def guard():
        if time.time() >= fit_deadline:raise TimeoutError('Fit/preparation budget reached; verification time reserved.')
        if psutil.virtual_memory().available < 550*1024**2:raise MemoryError('Below 550 MiB available memory.')

    guard()
    spec = importlib.util.spec_from_file_location('joint_reused_solver',ROOT/'fit_public_pool.py')
    solver = importlib.util.module_from_spec(spec);spec.loader.exec_module(solver)
    assert solver.C == protocol['C']
    tr = pd.read_csv(ROOT/'train.csv',usecols=['id','satisfaction'])
    test_ids = pd.read_csv(ROOT/'test.csv',usecols=['id']).id.to_numpy()
    ids, y = tr.id.to_numpy(), tr.satisfaction.to_numpy(dtype='int64')
    fold = np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy',allow_pickle=False)
    assert set(np.unique(fold))==set(range(5))
    x = np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    xt = np.load(BASE/'test_logits.npy',mmap_mode='r',allow_pickle=False)
    assert x.shape==(len(y),253) and xt.shape==(len(test_ids),253) and x.dtype==np.float64
    with np.load(BASE/'selected_predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'],ids);np.testing.assert_array_equal(z['test_id'],test_ids)
        incumbent_oof = z['oof_logit'].copy()
        incumbent_test = z['test_logit'].copy()
    kumo_path = ROOT/'research_s6e10/upstream_generalization_20261020/kumo_threebag_predictions.npz'
    with np.load(kumo_path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'],ids);np.testing.assert_array_equal(z['test_id'],test_ids)
        kumo, kumo_test = z['oof'].copy(),z['test'].copy()
    den_dir = ROOT/'research_s6e10/frontier_after_096187_20261020/predictions'
    den = np.full(len(y),np.nan);den_test=np.zeros(len(test_ids));coverage=np.zeros(len(y),dtype='int8')
    index = pd.Index(ids)
    inputs = [kumo_path,ROOT/'fit_public_pool.py',ROOT/'experiment_joint_upstream.py',BASE/'matrix_manifest.json',
              BASE/'train_logits.npy',BASE/'test_logits.npy',ROOT/'train.csv',ROOT/'test.csv']
    for k in range(5):
        guard();p=den_dir/f'den_fold{k}.npz';inputs.append(p)
        with np.load(p,allow_pickle=False) as z:
            idx=index.get_indexer(z['valid_ids']);assert (idx>=0).all() and len(np.unique(idx))==len(idx)
            np.testing.assert_array_equal(z['test_ids'],test_ids)
            den[idx]=z['valid'];den_test+=z['test']/5;coverage[idx]+=1
    assert (coverage==1).all()
    for a in [kumo,kumo_test,den,den_test]:
        assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    extra=np.column_stack([logit(np.clip(kumo,1e-6,1-1e-6)),logit(np.clip(den,1e-6,1-1e-6))])
    extra_test=np.column_stack([logit(np.clip(kumo_test,1e-6,1-1e-6)),logit(np.clip(den_test,1e-6,1-1e-6))])
    np.savez(RUN/'aligned_extra_logits.npz',id=ids,test_id=test_ids,train=extra,test=extra_test)
    guard()
    manifest=[]
    for p in inputs:
        guard();manifest.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    save(RUN/'input_manifest.json',manifest)
    arms={'baseline246':[],'plus_kumo':[0],'plus_neighborhood':[1],'plus_both':[0,1]}
    save(RUN/'execution_details.json',{'started_epoch':started,'fit_deadline_epoch':fit_deadline,
         'initialization':'baseline zero; extended arms warm-start from same-fold newly fitted baseline with zero extra coefficients; same convex objective',
         'data_ready_epoch':time.time(),'available_memory_bytes':psutil.virtual_memory().available})
    print('DATA_READY','remaining_seconds',round(deadline-time.time(),1),flush=True)
    results=[];oofs={arm:np.full(len(y),np.nan) for arm in arms};weights={a:[] for a in arms}
    for f in range(5):
        valid=np.flatnonzero(fold==f);training=fold!=f
        with np.load(BASE/f'public_only_fold{f}_validation.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['index'],valid)
        for arm,columns in arms.items():
            guard();provider=Matrix(x,extra,columns);tag=f'{arm}_fold{f}';t=time.time()
            initial=None
            if columns:
                w0=weights['baseline246'][f]
                initial=np.r_[w0[:-1],np.zeros(len(columns)),w0[-1]]
            with threadpool_limits(limits=4):
                w,fit=solver.newton(provider,y,training,provider.shape[1],fit_deadline,start=initial,
                    callback=lambda w,tag=tag:np.savez(RUN/(tag+'_checkpoint.npz'),weights=w))
                pred=np.empty(len(valid))
                for begin in range(0,len(valid),8192):
                    guard();block=valid[begin:begin+8192]
                    pred[begin:begin+len(block)]=provider[block,slice(None)]@w[:-1]+w[-1]
            score=float(roc_auc_score(y[valid],pred))
            meta={'fold':f,'arm':arm,'columns':provider.shape[1],'fit_rows':int(training.sum()),
                  'auc':score,'fit_and_validation_seconds':time.time()-t,**fit}
            np.savez(RUN/(tag+'.npz'),weights=w,index=valid,logit=pred)
            save(RUN/(tag+'.json'),meta)
            weights[arm].append(w);oofs[arm][valid]=pred;results.append(meta)
            save(RUN/'progress.json',{'completed_fits':len(results),'rows':results,'epoch':time.time()})
            print('ARM_COMPLETE',json.dumps(meta),flush=True)
    test_scores={}
    for arm,columns in arms.items():
        guard();mean_w=np.mean(weights[arm],axis=0);provider=Matrix(xt,extra_test,columns)
        pred=np.empty(len(test_ids))
        with threadpool_limits(limits=4):
            for b in range(0,len(test_ids),8192):
                pred[b:b+8192]=provider[slice(b,b+8192),slice(None)]@mean_w[:-1]+mean_w[-1]
        test_scores[arm]=pred
    np.savez(RUN/'screen_predictions.npz',id=ids,test_id=test_ids,fold=fold,
             **{a+'_oof':p for a,p in oofs.items()},**{a+'_test':p for a,p in test_scores.items()})
    base=oofs['baseline246'];base_auc=float(roc_auc_score(y,base))
    base_folds=np.array([roc_auc_score(y[fold==f],base[fold==f]) for f in range(5)])
    rank_base=rankdata(base)/len(y);rank_test_base=rankdata(test_scores['baseline246'])/len(test_ids)
    comparisons=[]
    for arm,columns in arms.items():
        p=oofs[arm];t=test_scores[arm]
        fa=np.array([roc_auc_score(y[fold==f],p[fold==f]) for f in range(5)]);delta=fa-base_folds
        se=float(delta.std(ddof=1)/np.sqrt(5));gate=protocol['meaningful_expansion_reference']
        oof_rank_std=float(np.std(rankdata(p)/len(y)-rank_base))
        test_rank_std=float(np.std(rankdata(t)/len(test_ids)-rank_test_base))
        comparisons.append({'arm':arm,'auc':float(roc_auc_score(y,p)),'pooled_delta':float(roc_auc_score(y,p)-base_auc),
          'fold_aucs':fa.tolist(),'fold_deltas':delta.tolist(),'mean_fold_delta':float(delta.mean()),'paired_fold_se':se,
          'positive_folds':int((delta>0).sum()),'passes_expansion_reference':bool(delta.mean()>=gate['mean_paired_fold_auc_delta'] and (delta>0).sum()>=gate['minimum_positive_folds'] and delta.mean()>2*se),
          'extra_coefficients_per_fold':np.array([w[246:-1] for w in weights[arm]]).tolist(),
          'oof_logit_difference_std':float(np.std(p-base)),'test_logit_difference_std':float(np.std(t-test_scores['baseline246'])),
          'oof_percentile_rank_difference_std':oof_rank_std,'test_percentile_rank_difference_std':test_rank_std,
          'rank_difference_test_to_oof_ratio':test_rank_std/oof_rank_std if oof_rank_std>0 else None,
          'test_mean_probability':float(expit(t).mean())})
    for a,p in oofs.items():assert np.isfinite(p).all()
    out={'complete':True,'rows_train':len(y),'rows_test':len(test_ids),'completed_meta_fits':len(results),
         'baseline_auc':base_auc,'baseline_max_abs_difference_vs_saved_oof':float(np.max(np.abs(base-incumbent_oof))),
         'baseline_test_meta_average_max_difference_vs_incumbent_full_refit':float(np.max(np.abs(test_scores['baseline246']-incumbent_test))),
         'comparisons':comparisons,'data_preparation_seconds':read(RUN/'execution_details.json')['data_ready_epoch']-started,
         'fit_and_validation_seconds_sum':sum(r['fit_and_validation_seconds'] for r in results),
         'execution_elapsed_seconds':time.time()-started,'elapsed_including_setup_allowance':time.time()-protocol['started_epoch'],
         'remaining_budget_seconds_for_verification':deadline-time.time(),'new_base_fits':0,'new_base_inferences':0,
         'new_submission_generated':False,'validation_status':'Exploratory fixed-public-base-OOF meta-CV; not end-to-end independent validation.'}
    save(RUN/'evaluation.json',out)
    print('SCREEN_COMPLETE',json.dumps(out),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(RUN/'stopped.json',{'error':type(e).__name__+': '+str(e),'epoch':time.time()})
        raise
