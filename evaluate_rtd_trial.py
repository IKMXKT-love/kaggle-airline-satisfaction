"""Frozen conditional stack comparison; score fold labels used only here."""
import json,time,hashlib,sys,gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from fit_public_pool import newton
from multitask_support import paired

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'.artifact_runtime/rtd_trial_20261022'

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=json.loads((RUN/'protocol.json').read_text('utf-8'));deadline=p['deadline_epoch']
    if time.time()>deadline:raise TimeoutError('Original two-hour budget expired')
    assert (RUN/'rtd_predictions.npz').exists() and (RUN/'clean_predictions.npz').exists()
    ix=np.load(RUN/'indices.npz');stop=ix['stop'];score=ix['score']
    y=pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy(dtype='int64')
    assert not np.intersect1d(stop,score).size
    matrix=np.load(ROOT/'.artifact_runtime/public_pool_trial_20261020/train_logits.npy',mmap_mode='r')
    fitx=np.array(matrix[stop,:246]);scorex=np.array(matrix[score,:246])
    zs={}
    scores={}
    meta={}
    for arm in ['matched_baseline','clean','rtd']:
        x=fitx
        if arm!='matched_baseline':
            a=np.load(RUN/(arm+'_predictions.npz'))
            assert len(a['stop_logit'])==len(stop) and len(a['score_logit'])==len(score)
            x=np.column_stack([fitx,a['stop_logit']]);testx=np.column_stack([scorex,a['score_logit']])
        else:testx=scorex
        with threadpool_limits(limits=4):
            w,m=newton(x,y[stop],np.ones(len(stop),dtype=bool),x.shape[1],deadline)
            zz=testx@w[:-1]+w[-1]
        zs[arm]=zz;meta[arm]=m
        np.savez(RUN/(arm+'_meta.npz'),weights=w)
        scores[arm]={'score_auc':float(roc_auc_score(y[score],zz)),'new_member_coefficient':float(w[-2]) if arm!='matched_baseline' else None}
        print(arm,scores[arm],flush=True)
    incumbent=np.load(ROOT/'.artifact_runtime/public_pool_trial_20261020/public_only_fold0.npz')['weights']
    with threadpool_limits(limits=4):zs['incumbent']=scorex@incumbent[:-1]+incumbent[-1]
    scores['incumbent']={'score_auc':float(roc_auc_score(y[score],zs['incumbent']))}
    for arm in ['clean','rtd']:
        a=np.load(RUN/(arm+'_predictions.npz'))
        scores[arm]['single_score_auc']=float(roc_auc_score(y[score],a['score_logit']))
        scores[arm]['vs_matched_baseline']=paired(y[score],zs[arm],zs['matched_baseline'])
        scores[arm]['vs_incumbent']=paired(y[score],zs[arm],zs['incumbent'])
    delta_vs_clean=paired(y[score],zs['rtd'],zs['clean'])
    accept=(scores['rtd']['vs_incumbent']['delta']>=.00005 and delta_vs_clean['delta']>=.00001 and scores['rtd']['vs_incumbent']['interval95'][0]>0)
    result={'scores':scores,'rtd_vs_clean':delta_vs_clean,'admission_passed':bool(accept),'meta_fits':meta,
        'scored_rows':len(score),'meta_fit_rows':len(stop),'elapsed_seconds':time.time()-p['start_epoch'],
        'limitations':'Historical fold0, public upstream indirect dependence. Candidate epoch selected on meta fitting fold4. Conditional meta comparison; no end-to-end independent guarantee. Fixed-prediction intervals do not cover training or selection uncertainty.',
        'new_test_predictions':False}
    np.savez(RUN/'scored_meta_predictions.npz',index=score,**zs)
    (RUN/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
