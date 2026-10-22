"""Independent row-block pair audit and CPU tree/prediction reconstruction."""
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from audit_pool_repairability import ROOT,RUN,BASE,FOLDS,read,write,sha,check
from multitask_support import paired
from fit_public_pool import objective

def independent_features(frame,idx,x,b,state,conditional):
    parts=[];types=[]
    for c in frame:
        if c in state['categories']:
            codes=pd.Categorical(frame.iloc[idx][c].fillna('__MISSING__').astype(str),categories=state['categories'][c]).codes.astype('float32')
            codes[codes<0]=np.nan;parts.append(codes);types.append('c')
        else:
            parts.append(pd.to_numeric(frame.iloc[idx][c],errors='coerce').to_numpy(dtype='float32'));types.append('q')
    a=np.column_stack(parts+[b.astype('float32')])
    types.append('q')
    if conditional:
        a=np.column_stack([a,(x[idx,:246]-b[:,None]).astype('float32')]);types+=['q']*246
    return a,types

def main():
    p=read(RUN/'protocol.json');check(p)
    frame=pd.read_csv(ROOT/'train.csv');ids=frame.pop('id').to_numpy();y=frame.pop('satisfaction').to_numpy(dtype='int64')
    folds=np.load(FOLDS,allow_pickle=False);x=np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    recreated=np.empty(len(y),dtype='int8')
    for f,(_,idx) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y)):recreated[idx]=f
    np.testing.assert_array_equal(folds,recreated)
    inputs=read(RUN/'inputs.json')
    for k,path in [('train.csv',ROOT/'train.csv'),('matrix',BASE/'train_logits.npy'),('manifest',BASE/'matrix_manifest.json'),('folds',FOLDS)]:assert sha(path)==inputs[k],k
    z=np.load(BASE/'selected_predictions.npz',allow_pickle=False);np.testing.assert_array_equal(z['id'],ids);b=z['oof_logit']
    audit=read(RUN/'pair_audit.json');pair_checks=[]
    for f in range(5):
        check(p);z=np.load(RUN/f'fold{f}_pairs.npz',allow_pickle=False);a=z['positive'];c=z['negative']
        assert len(a)==p['pairs_per_native_fold'] and (y[a]==1).all() and (y[c]==0).all() and (folds[a]==f).all() and (folds[c]==f).all()
        idx=np.flatnonzero(folds==f);rng=np.random.default_rng(p['pair_seed']+f)
        np.testing.assert_array_equal(a,rng.choice(idx[y[idx]==1],len(a)));np.testing.assert_array_equal(c,rng.choice(idx[y[idx]==0],len(c)))
        saved=np.load(RUN/f'fold{f}_audit_arrays.npz',allow_pickle=False)
        rec=np.zeros(246,dtype='int64');dmg=rec.copy()
        for start in range(0,len(a),4096):
            stop=min(start+4096,len(a));aa=a[start:stop];cc=c[start:stop]
            d=x[aa,:246]-x[cc,:246];ok=d>0;bad=d<0;wrong=b[aa]<b[cc];right=b[aa]>b[cc]
            np.testing.assert_array_equal(wrong,saved['baseline_wrong'][start:stop])
            np.testing.assert_array_equal(np.any(ok,axis=1),saved['any_member_correct'][start:stop])
            np.testing.assert_array_equal(np.any(bad,axis=1),saved['any_member_wrong'][start:stop])
            np.testing.assert_array_equal(np.all(bad,axis=1)&wrong,saved['all_members_wrong'][start:stop])
            rec+=np.sum(ok&wrong[:,None],axis=0);dmg+=np.sum(bad&right[:,None],axis=0)
        np.testing.assert_array_equal(rec,saved['recovered_by_member']);np.testing.assert_array_equal(dmg,saved['damaged_by_member'])
        weights=np.load(BASE/f'public_only_fold{f}.npz',allow_pickle=False)['weights'];rebuilt=np.empty(len(idx))
        with threadpool_limits(limits=4):
            for start in range(0,len(idx),4096):
                rows=idx[start:start+4096];rebuilt[start:start+len(rows)]=x[rows,:246]@weights[:-1]+weights[-1]
        err=float(np.max(abs(rebuilt-b[idx])));assert err<1e-10
        assert abs(roc_auc_score(y[idx],rebuilt)-audit['folds'][f]['exact_baseline_auc'])<1e-12
        pair_checks.append({'fold':f,'all_sampled_pair_arrays_rebuilt':True,'cached_baseline_max_error':err})
    fitted=[]
    # Includes every internal selection stage and outer scored stage actually run.
    for path in sorted(RUN.glob('*_indices.npz')):
        check(p);tag=path.name[:-len('_indices.npz')];ix=np.load(path,allow_pickle=False);fit_idx=ix['fit'];idx=ix['query']
        assert not np.intersect1d(fit_idx,idx).size
        wz=np.load(RUN/f'{tag}_baseline.npz',allow_pickle=False);weights=wz['weights'];np.testing.assert_array_equal(wz['fit_index'],fit_idx)
        rebuilt=np.empty(len(idx))
        with threadpool_limits(limits=4):
            for start in range(0,len(idx),4096):
                rows=idx[start:start+4096];rebuilt[start:start+len(rows)]=x[rows,:246]@weights[:-1]+weights[-1]
        fit_mask=np.zeros(len(y),dtype=bool);fit_mask[fit_idx]=True
        with threadpool_limits(limits=4):_,gradient=objective(weights,x,y,fit_mask,246)
        assert np.max(abs(gradient))<1e-7
        rebuilt_arms={}
        for arm in ['raw_control','conditional']:
            prefix=RUN/f'{tag}_{arm}';state=read(prefix.with_suffix('.encoder.json'));meta=read(prefix.with_suffix('.fit.json'))
            assert state['columns']==list(frame)
            for c,values in state['categories'].items():assert values==sorted(frame.iloc[fit_idx][c].fillna('__MISSING__').astype(str).unique().tolist())
            model=xgb.Booster();model.load_model(prefix.with_suffix('.ubj'));model.set_param({'device':'cpu','nthread':4})
            pred=np.empty(len(idx))
            for start in range(0,len(idx),4096):
                rows=idx[start:start+4096];a,types=independent_features(frame,rows,x,rebuilt[start:start+len(rows)],state,arm=='conditional')
                dm=xgb.DMatrix(a,feature_types=types,base_margin=np.zeros(len(rows),dtype='float32'),nthread=4)
                delta=model.predict(dm,output_margin=True,iteration_range=(0,meta['selected_rounds']))
                pred[start:start+len(rows)]=rebuilt[start:start+len(rows)]+delta.astype('float64')
            saved=np.load(prefix.with_suffix('.prediction.npz'),allow_pickle=False);np.testing.assert_array_equal(saved['index'],idx)
            berror=float(np.max(abs(rebuilt-saved['baseline'])));error=float(np.max(abs(pred-saved['margin'])));assert berror<1e-10 and error<2e-6,(tag,arm,error)
            auc=float(roc_auc_score(y[idx],pred));auc_error=auc-meta['query_auc'];assert abs(auc_error)<1e-7
            diff=paired(y[idx],pred,rebuilt)
            fitted.append({'tag':tag,'arm':arm,'prediction_max_error':error,'baseline_max_error':berror,'auc':auc,'auc_rebuild_error':auc_error,'delta_and_fixed_prediction_ci':diff,'baseline_fit_gradient_max':float(np.max(abs(gradient)))})
            rebuilt_arms[arm]=pred
        contrast=paired(y[idx],rebuilt_arms['conditional'],rebuilt_arms['raw_control'])
        fitted.append({'tag':tag,'contrast':'conditional_vs_raw_control','paired':contrast})
    for name,digest in p['protected'].items():assert sha(ROOT/name)==digest,name
    out={'verified':True,'all10million_pairs_rebuilt':True,'pair_checks':pair_checks,'fitted_prediction_checks':fitted,'protected_files_unchanged':True,'elapsed_seconds_including_verification':time.time()-p['started_epoch'],'limit':'Fixed-prediction CIs omit training, model-selection, upstream-dependence and historical-selection uncertainty.'}
    write(RUN/'verification.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
