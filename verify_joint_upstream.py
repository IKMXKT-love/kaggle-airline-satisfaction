"""Independent source alignment, rank-sum scoring, gradients and protected-file checks."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'.artifact_runtime/joint_upstream_20261021'
BASE=ROOT/'.artifact_runtime/public_pool_trial_20261020'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda:f.read(1024*1024),b''):h.update(a)
    return h.hexdigest()


def auc(y,p):
    n=int(y.sum());r=rankdata(p,method='average')
    return float((r[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))


def main():
    started=time.time();protocol=read(RUN/'protocol.json');end=protocol['deadline_epoch']
    def guard():
        if time.time()>=end:raise TimeoutError('Total verification deadline reached')
    evaluation=read(RUN/'evaluation.json');assert evaluation['complete']
    assert evaluation['completed_meta_fits']==20
    tr=pd.read_csv(ROOT/'train.csv',usecols=['id','satisfaction']);y=tr.satisfaction.astype(int).to_numpy()
    test_ids=pd.read_csv(ROOT/'test.csv',usecols=['id']).id.to_numpy()
    with np.load(RUN/'aligned_extra_logits.npz',allow_pickle=False) as z:
        extra=z['train'];extra_t=z['test'];np.testing.assert_array_equal(z['id'],tr.id);np.testing.assert_array_equal(z['test_id'],test_ids)
    with np.load(ROOT/'research_s6e10/upstream_generalization_20261020/kumo_threebag_predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'],tr.id);np.testing.assert_array_equal(z['test_id'],test_ids)
        np.testing.assert_array_equal(extra[:,0],logit(np.clip(z['oof'],1e-6,1-1e-6)))
        np.testing.assert_array_equal(extra_t[:,0],logit(np.clip(z['test'],1e-6,1-1e-6)))
    index=pd.Index(tr.id);d=np.full(len(y),np.nan);dt=np.zeros(len(test_ids));coverage=np.zeros(len(y),dtype='int8')
    for f in range(5):
        with np.load(ROOT/f'research_s6e10/frontier_after_096187_20261020/predictions/den_fold{f}.npz',allow_pickle=False) as z:
            idx=index.get_indexer(z['valid_ids']);assert (idx>=0).all();assert len(np.unique(idx))==len(idx)
            np.testing.assert_array_equal(z['test_ids'],test_ids);d[idx]=z['valid'];dt+=z['test']/5;coverage[idx]+=1
    assert (coverage==1).all()
    np.testing.assert_array_equal(extra[:,1],logit(np.clip(d,1e-6,1-1e-6)))
    np.testing.assert_array_equal(extra_t[:,1],logit(np.clip(dt,1e-6,1-1e-6)))
    for row in read(RUN/'input_manifest.json'):
        guard();assert digest(ROOT/row['path'])==row['sha256'],row['path']
    x=np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    xt=np.load(BASE/'test_logits.npy',mmap_mode='r',allow_pickle=False)
    arrays=np.load(RUN/'screen_predictions.npz',allow_pickle=False)
    np.testing.assert_array_equal(arrays['id'],tr.id);np.testing.assert_array_equal(arrays['test_id'],test_ids)
    fold=arrays['fold'];native=np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy',allow_pickle=False)
    np.testing.assert_array_equal(fold,native)
    arms={'baseline246':[],'plus_kumo':[0],'plus_neighborhood':[1],'plus_both':[0,1]}
    checks=[];max_pred_error=0.;max_gradient=0.
    with threadpool_limits(limits=4):
        for arm,cols in arms.items():
            weights=[];counts=np.zeros(len(y),dtype='int8');expected=arrays[arm+'_oof']
            for f in range(5):
                guard();meta=read(RUN/f'{arm}_fold{f}.json');assert meta['converged'] and meta['max_gradient']<=1e-8
                with np.load(RUN/f'{arm}_fold{f}.npz',allow_pickle=False) as z:
                    w,idx,pred=z['weights'],z['index'],z['logit']
                assert len(w)==247+len(cols);np.testing.assert_array_equal(idx,np.flatnonzero(fold==f))
                np.testing.assert_array_equal(pred,expected[idx]);counts[idx]+=1;weights.append(w)
                g=np.zeros_like(w);n=int((fold!=f).sum())
                for begin in range(0,len(y),10007):
                    guard();stop=min(begin+10007,len(y));keep=fold[begin:stop]!=f
                    a=np.asarray(x[begin:stop,:246])
                    if cols:a=np.concatenate([a,extra[begin:stop][:,cols]],axis=1)
                    score=np.dot(a,w[:-1])+w[-1]
                    valid=~keep
                    if valid.any():max_pred_error=max(max_pred_error,float(np.max(np.abs(score[valid]-expected[begin:stop][valid]))))
                    residual=(expit(score[keep])-y[begin:stop][keep])/n
                    g[:-1]+=a[keep].T@residual;g[-1]+=residual.sum()
                g[:-1]+=w[:-1]/(protocol['C']*n)
                error=float(np.abs(g).max());assert error<1.01e-8,(arm,f,error)
                max_gradient=max(max_gradient,error)
            assert (counts==1).all()
            mean=np.mean(weights,axis=0);test_error=0.
            for begin in range(0,len(test_ids),10007):
                guard();stop=min(begin+10007,len(test_ids));a=np.asarray(xt[begin:stop,:246])
                if cols:a=np.concatenate([a,extra_t[begin:stop][:,cols]],axis=1)
                pred=a@mean[:-1]+mean[-1]
                test_error=max(test_error,float(np.max(np.abs(pred-arrays[arm+'_test'][begin:stop]))))
            assert test_error<2e-11
            row=next(x for x in evaluation['comparisons'] if x['arm']==arm)
            assert abs(auc(y,expected)-row['auc'])<2e-15
            base=arrays['baseline246_oof']
            deltas=[auc(y[fold==f],expected[fold==f])-auc(y[fold==f],base[fold==f]) for f in range(5)]
            np.testing.assert_allclose(deltas,row['fold_deltas'],rtol=0,atol=2e-15)
            checks.append({'arm':arm,'rank_sum_auc':auc(y,expected),'mean_test_logit_reconstruction_error':test_error})
    assert max_pred_error<2e-11
    protected={}
    for name,before in read(RUN/'protected_before.json').items():
        after=digest(ROOT/name);assert after==before,name;protected[name]=after
    assert evaluation['baseline_max_abs_difference_vs_saved_oof']<1e-6
    out={'all_passed':True,'completed_meta_fits_verified':20,'new_base_fits':0,'new_base_inferences':0,
         'source_alignment_checked':True,'input_hashes_checked':True,'independent_auc':'Mann-Whitney rank sum with average ties',
         'all_training_gradients_recomputed':True,'maximum_independent_gradient':max_gradient,
         'maximum_validation_logit_reconstruction_error':max_pred_error,'checks':checks,'protected_sha256':protected,
         'verification_seconds':time.time()-started,'total_elapsed_including_setup_allowance':time.time()-protocol['started_epoch'],
         'within_total_900_second_budget':time.time()<=end,
         'validation_caveat':'Independent array/numerical verification does not establish independent end-to-end model validation.'}
    (RUN/'independent_verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
