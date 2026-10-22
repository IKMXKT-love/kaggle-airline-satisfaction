"""Independent full scored-row rebuild plus a fixed source-cache prediction audit."""
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from density_ratio_support import ARMS, RatioMLP, predict
from domain_transfer_support import RawEncoder
from experiment_domain_transfer import ROOT, BASE, FOLDS, read, write, sha, setup
from experiment_density_ratio import RUN, CACHE
from multitask_support import to_device, paired
from optimized_features import normalize_raw, NativeCategoryEncoder

sys.stdout.reconfigure(encoding='utf-8')


def main():
    started=time.time(); setup()
    protocol,evaluation=read(RUN/'protocol.json'),read(RUN/'evaluation.json')
    for f,h in protocol['input_sha256'].items(): assert sha(ROOT/f)==h,f
    for f,h in protocol['protected'].items(): assert sha(ROOT/f)==h,f
    train,test=pd.read_csv(ROOT/'train.csv'),pd.read_csv(ROOT/'test.csv')
    features=[c for c in test if c!='id']; x=train[features]; y=train.satisfaction.to_numpy(dtype='int64'); ids=train.id.to_numpy()
    with np.load(RUN/'indices.npz',allow_pickle=False) as z: idx={k:z[k].copy() for k in z.files}
    with np.load(RUN/'labels.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(y,z['y']); np.testing.assert_array_equal(ids,z['id']); oy=z['original_y'].copy()
    assert np.array_equal(np.sort(np.r_[idx['fit'],idx['stopping'],idx['scored']]),np.arange(len(y)))
    folded=np.load(FOLDS,allow_pickle=False)
    np.testing.assert_array_equal(idx['scored'],np.flatnonzero(folded==2))
    np.testing.assert_array_equal(idx['outer_train'],np.flatnonzero(folded!=2))
    fit,stop=train_test_split(idx['outer_train'],test_size=.08,random_state=protocol['training']['seed'],stratify=y[idx['outer_train']])
    np.testing.assert_array_equal(np.sort(fit),idx['fit']); np.testing.assert_array_equal(np.sort(stop),idx['stopping'])
    original=pd.concat([pd.read_csv(ROOT/'external_data'/f) for f in ['train.csv','test.csv']],ignore_index=True)
    nx,nt,no=[normalize_raw(f) for f in [x,test[features],original[features]]]
    original_fp=pd.util.hash_pandas_object(no.astype('string').fillna('__MISSING__'),index=False)
    competition_fp=pd.util.hash_pandas_object(pd.concat([nx,nt],ignore_index=True).astype('string').fillna('__MISSING__'),index=False)
    keep=~original_fp.isin(set(competition_fp))
    np.testing.assert_array_equal(np.flatnonzero(keep),idx['original_rows'])
    np.testing.assert_array_equal(original.loc[keep,'satisfaction'].eq('satisfied').to_numpy(dtype='int64'),oy)
    assert len(oy)==protocol['original_rows'] and int((~keep).sum())==protocol['removed_original_overlap']
    q=logit(np.clip(np.load(CACHE/'train_priors.npy',allow_pickle=False).astype('float64'),1e-6,1-1e-6)).mean(1)
    np.testing.assert_array_equal(q,np.load(RUN/'teacher_margin.npy',allow_pickle=False))
    # Prediction audit is fixed by row positions before inspecting any residuals.
    sample=np.linspace(0,len(x)-1,4096,dtype='int64')
    source_encoder=NativeCategoryEncoder().fit(no.loc[keep].reset_index(drop=True))
    audit_x=source_encoder.transform(nx.iloc[sample])
    cached=np.load(CACHE/'train_priors.npy',mmap_mode='r',allow_pickle=False)
    teacher_errors={}
    for j,depth in enumerate([6,3,9]):
        model=XGBClassifier(n_jobs=4); model.load_model(CACHE/f'original_prior_depth{depth}.ubj'); model.set_params(device='cpu',n_jobs=4)
        actual=model.predict_proba(audit_x)[:,1]
        err=float(np.max(np.abs(actual-cached[sample,j])))
        assert err<1e-7,(depth,err); teacher_errors[str(depth)]=err
    del model,audit_x
    for phase,rows in [('selection',idx['fit']),('refit',idx['outer_train'])]:
        encoder=RawEncoder().fit(x.iloc[rows]); assert encoder.state()==read(RUN/phase/'encoder.json')
        for name,frame in [('competition',x.iloc[rows]),('original',original.loc[keep,features].reset_index(drop=True)),('scored',x.iloc[idx['scored']])]+([('stopping',x.iloc[idx['stopping']])] if phase=='selection' else []):
            encoded=encoder.transform(frame)
            for k in ['num','missing','cat']: np.testing.assert_array_equal(encoded[k],np.load(RUN/phase/f'{name}_{k}.npy',allow_pickle=False))
        assert len({read(RUN/phase/a/'complete.json')['initial_state_hash'] for a in ARMS})==1
    del encoded,no,nx,nt,original
    base=np.load(RUN/'baseline_logit.npy',allow_pickle=False)
    matrix=np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    with np.load(BASE/'public_only_fold2.npz',allow_pickle=False) as z: weights=z['weights']
    assert len(weights)==247
    rebuild=np.empty(len(idx['scored']))
    with threadpool_limits(limits=4):
        for start in range(0,len(rebuild),8192):
            rows=idx['scored'][start:start+8192]; rebuild[start:start+len(rows)]=matrix[rows,:246]@weights[:-1]+weights[-1]
    base_error=float(np.max(np.abs(base-rebuild))); assert base_error<1e-10
    sy=y[idx['scored']]; assert abs(roc_auc_score(sy,rebuild)-evaluation['baseline_auc'])<1e-12
    encoder=RawEncoder().fit(x.iloc[idx['outer_train']]); values=to_device(encoder.transform(x.iloc[idx['scored']]),'cuda')
    pi=float(y[idx['outer_train']].mean()); qq=q+float(logit(pi)-logit(oy.mean()))
    teacher=torch.tensor(qq[idx['scored']],device='cuda',dtype=torch.float32)
    evidence={}; blends={}
    for arm in ARMS:
        selection=read(RUN/'selection'/arm/'complete.json'); refit=read(RUN/'refit'/arm/'complete.json')
        history=read(RUN/'selection'/arm/'history.json'); best=max(history,key=lambda v:v['stopping_auc'])
        assert selection['best_epoch']==best['epoch']==refit['epochs_completed']
        assert selection['epochs_completed']==protocol['max_epochs']
        assert refit['updates']==int(np.ceil(len(idx['outer_train'])/2048))*refit['epochs_completed']
        assert refit['competition_draws']==refit['original_draws']==refit['updates']*1024
        torch.manual_seed(protocol['training']['seed']); model=RatioMLP(encoder.state(),protocol['model']).cuda()
        model.load_state_dict(torch.load(RUN/'refit'/arm/'weights.pt',map_location='cuda',weights_only=True))
        p,b=predict(model,values,teacher,arm,float(logit(pi)),protocol['training']['prediction_batch'])
        with np.load(RUN/'refit'/arm/'scored_prediction.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['id'],ids[idx['scored']]); np.testing.assert_array_equal(z['index'],idx['scored'])
            err=float(np.max(np.abs(p-z['probability']))); contrast_error=float(np.max(np.abs(b-z['mean_contrast'])))
        assert err<1e-7 and contrast_error<1e-7,(arm,err,contrast_error)
        blends[arm]=.9*base+.1*logit(np.clip(p,1e-6,1-1e-6))
        auc=float(roc_auc_score(sy,p)); blend_auc=float(roc_auc_score(sy,blends[arm]))
        assert abs(auc-evaluation['arms'][arm]['single_auc'])<1e-12
        assert abs(blend_auc-evaluation['arms'][arm]['primary_blend_auc'])<1e-12
        assert paired(sy,blends[arm],base)==evaluation['arms'][arm]['primary_blend_delta']
        for w in [.05,.2]:
            actual=float(roc_auc_score(sy,(1-w)*base+w*logit(np.clip(p,1e-6,1-1e-6))))-evaluation['baseline_auc']
            assert abs(actual-evaluation['arms'][arm]['diagnostic_weights'][str(w)])<1e-12
        evidence[arm]=dict(full_prediction_max_error=err,full_contrast_max_error=contrast_error,selected_epoch=best['epoch'],single_auc=auc,primary_blend_auc=blend_auc)
        del model
    delta=evaluation['arms']['conditional_ratio']['primary_blend_delta']
    comparisons={a:paired(sy,blends['conditional_ratio'],blends[a]) for a in ['direct','teacher_offset']}
    assert comparisons==evaluation['ratio_vs_controls']
    passed=delta['delta']>=.0003 and delta['interval95'][0]>0 and all(v['delta']>0 for v in comparisons.values())
    assert passed==evaluation['passes_expansion_gate']
    for f,h in protocol['protected'].items(): assert sha(ROOT/f)==h
    clock=read(RUN/'clock.json'); cumulative=time.time()-clock['started_epoch']
    result=dict(verified=True,full_scored_prediction_rebuild=True,all_encodings_and_overlap_filter_rebuilt=True,
        independent_baseline_max_error=base_error,teacher_4096_row_prediction_max_errors=teacher_errors,
        source_teacher_full_cache_recomputed=False,initial_states_match=True,evidence=evidence,
        protected_unchanged=True,passes_expansion_gate=passed,verification_seconds=time.time()-started,
        cumulative_seconds=cumulative,cumulative_deadline_met=time.time()<=clock['deadline_epoch'])
    write(RUN/'verification.json',result); print(result,flush=True)


if __name__=='__main__': main()
