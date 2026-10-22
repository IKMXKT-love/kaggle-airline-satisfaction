"""A single frozen conditional correction, with a matched raw-only control."""
import gc
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from audit_pool_repairability import ROOT,RUN,BASE,FOLDS,read,write,check
from fit_public_pool import newton,predict

class Deadline(xgb.callback.TrainingCallback):
    def __init__(self,protocol): self.protocol=protocol
    def after_iteration(self,model,epoch,evals_log):
        check(self.protocol)
        return False

def raw_fit(frame):
    state={'columns':list(frame),'categories':{}}
    for c in frame:
        if not pd.api.types.is_numeric_dtype(frame[c]):
            state['categories'][c]=sorted(frame[c].fillna('__MISSING__').astype(str).unique().tolist())
    return state

def features(frame,matrix,idx,margin,state,disagreements):
    n=len(idx);k=22+(246 if disagreements else 0)
    out=np.empty((n,k),dtype='float32')
    types=[]
    for j,c in enumerate(state['columns']):
        if c in state['categories']:
            mapping={v:i for i,v in enumerate(state['categories'][c])}
            out[:,j]=frame.iloc[idx][c].fillna('__MISSING__').astype(str).map(mapping).to_numpy(dtype='float32');types.append('c')
        else:
            out[:,j]=pd.to_numeric(frame.iloc[idx][c],errors='coerce').to_numpy(dtype='float32');types.append('q')
    out[:,21]=margin[idx];types.append('q')
    if disagreements:
        for start in range(0,n,8192):
            rows=idx[start:start+8192]
            out[start:start+len(rows),22:]=matrix[rows,:246]-margin[rows,None]
        types+=['q']*246
    assert out.shape==(n,k)
    return out,types

def main():
    p=read(RUN/'protocol.json');check(p)
    assert (RUN/'pair_audit.json').exists()
    frame=pd.read_csv(ROOT/'train.csv');y=frame.pop('satisfaction').to_numpy(dtype='int64');frame=frame.drop(columns='id');assert frame.shape[1]==21
    folds=np.load(FOLDS,allow_pickle=False);matrix=np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    params=dict(p['xgb']);max_rounds=params.pop('n_estimators');patience=params.pop('early_stopping_rounds');seed=params.pop('random_state');njobs=params.pop('n_jobs')
    params['seed']=seed;params['nthread']=njobs
    results=[]
    def baseline(tag,fit_rows,existing=None):
        if existing is None:
            fit_mask=np.zeros(len(y),dtype=bool);fit_mask[fit_rows]=True
            with threadpool_limits(limits=4): w,meta=newton(matrix,y,fit_mask,246,p['started_epoch']+p['screen_budget_seconds'])
        else:
            w=np.load(existing,allow_pickle=False)['weights'].copy();meta={'cached_native_fold_fit':str(existing)}
        np.savez(RUN/f'{tag}_baseline.npz',weights=w,fit_index=fit_rows)
        write(RUN/f'{tag}_baseline.json',meta)
        return predict(matrix,w)
    def fit_arm(tag,arm,fit_idx,query_idx,margin,state,rounds=None):
        check(p);t=time.time()
        prefix=RUN/f'{tag}_{arm}';write(prefix.with_suffix('.encoder.json'),state)
        with_disagreement=arm=='conditional'
        train_x,types=features(frame,matrix,fit_idx,margin,state,with_disagreement)
        query_x,_=features(frame,matrix,query_idx,margin,state,with_disagreement)
        dtrain=xgb.QuantileDMatrix(train_x,label=y[fit_idx],base_margin=margin[fit_idx].astype('float32'),feature_types=types,max_bin=params['max_bin'],nthread=4)
        dquery=xgb.QuantileDMatrix(query_x,label=y[query_idx],base_margin=margin[query_idx].astype('float32'),feature_types=types,max_bin=params['max_bin'],ref=dtrain,nthread=4)
        history={}
        model=xgb.train(params,dtrain,num_boost_round=rounds or max_rounds,evals=[(dquery,'stopping')] if rounds is None else [],
                        early_stopping_rounds=patience if rounds is None else None,evals_result=history,verbose_eval=False,callbacks=[Deadline(p)])
        selected=int(model.best_iteration)+1 if rounds is None else rounds
        model.save_model(prefix.with_suffix('.ubj'))
        # Query prediction never reads labels. Zero margin returns the additive
        # tree correction; add original float64 LR score once, outside XGBoost.
        dquery.set_base_margin(np.zeros(len(query_idx),dtype='float32'))
        correction=model.predict(dquery,output_margin=True,iteration_range=(0,selected))
        q=margin[query_idx]+correction.astype('float64')
        np.savez(prefix.with_suffix('.prediction.npz'),index=query_idx,baseline=margin[query_idx],correction=correction,margin=q)
        meta={'tag':tag,'arm':arm,'selected_rounds':selected,'actual_rounds':model.num_boosted_rounds(),'seconds':time.time()-t,'fit_rows':len(fit_idx),'query_rows':len(query_idx),'features':train_x.shape[1],'types':types,'internal_selection':rounds is None,'query_auc':float(roc_auc_score(y[query_idx],q))}
        write(prefix.with_suffix('.fit.json'),meta);write(prefix.with_suffix('.history.json'),history)
        print('FIT',json.dumps(meta),flush=True)
        del train_x,query_x,dtrain,dquery,model;gc.collect()
        return q,meta
    def compare(tag,idx,margin,outputs):
        ba=float(roc_auc_score(y[idx],margin[idx]));r={'tag':tag,'rows':len(idx),'baseline_auc':ba}
        for arm in ['raw_control','conditional']:
            q,meta=outputs[arm];auc=float(roc_auc_score(y[idx],q));r[arm+'_auc']=auc;r[arm+'_delta']=auc-ba;r[arm+'_rounds']=meta['selected_rounds']
        r['conditional_vs_control']=r['conditional_auc']-r['raw_control_auc']
        results.append(r);write(RUN/'conditional_results.json',results);print('COMPARE',json.dumps(r),flush=True)
        return r
    # The internal scoring rows do not enter the LR fit or feature vocabulary.
    r_idx=np.flatnonzero(np.isin(folds,[1,2,3]));s_idx=np.flatnonzero(folds==4)
    margin=baseline('inner',r_idx);state=raw_fit(frame.iloc[r_idx]);np.savez(RUN/'inner_indices.npz',fit=r_idx,query=s_idx)
    outputs={a:fit_arm('inner',a,r_idx,s_idx,margin,state) for a in ['raw_control','conditional']}
    inner=compare('inner',s_idx,margin,outputs)
    def passed(r,phase):
        return r['conditional_delta']>=p[phase+'_gate_delta_vs_baseline'] and r['conditional_vs_control']>=p[phase+'_gate_delta_vs_control']
    if not passed(inner,'inner'):status='stopped_inner_gate'
    else:
        status='outer_screen_pending'
        for f in range(5):
            check(p);start=time.time()
            if f==0:
                selected={a:outputs[a][1]['selected_rounds'] for a in outputs}
            else:
                s_fold=(f+4)%5;fit_idx=np.flatnonzero((folds!=f)&(folds!=s_fold));stop_idx=np.flatnonzero(folds==s_fold)
                m=baseline(f'outer{f}_selection',fit_idx);encoder=raw_fit(frame.iloc[fit_idx]);np.savez(RUN/f'outer{f}_selection_indices.npz',fit=fit_idx,query=stop_idx)
                selected={a:fit_arm(f'outer{f}_selection',a,fit_idx,stop_idx,m,encoder)[1]['selected_rounds'] for a in ['raw_control','conditional']}
            t_idx=np.flatnonzero(folds!=f);e_idx=np.flatnonzero(folds==f)
            m=baseline(f'fold{f}',t_idx,BASE/f'public_only_fold{f}.npz');encoder=raw_fit(frame.iloc[t_idx]);np.savez(RUN/f'fold{f}_indices.npz',fit=t_idx,query=e_idx)
            scored={a:fit_arm(f'fold{f}',a,t_idx,e_idx,m,encoder,selected[a]) for a in ['raw_control','conditional']}
            result=compare(f'fold{f}',e_idx,m,scored)
            if f==0 and not passed(result,'outer'):
                status='stopped_outer_screen_gate';break
            if f==0 and time.time()+4*(time.time()-start+40)+60>p['started_epoch']+p['screen_budget_seconds']:
                status='stopped_projected_confirmation_cost';break
        else:
            ds=np.array([r['conditional_delta'] for r in results if r['tag'].startswith('fold')])
            status='confirmation_passed' if ds.mean()>=p['confirmation_mean_gain'] and (ds>0).sum()>=p['confirmation_min_positive_folds'] and ds.mean()-2*ds.std(ddof=1)/np.sqrt(5)>0 else 'stopped_confirmation_gate'
    write(RUN/'completion.json',{'status':status,'elapsed_seconds':time.time()-p['started_epoch'],'results':results,'submission_generated':False})
    print('COMPLETE',status,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(RUN/'failure.json',{'type':type(e).__name__,'error':str(e),'epoch':time.time()});raise
