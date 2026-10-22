"""Read-only model/data diagnosis. No fitting or submission generation."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def hashes(f):return pd.util.hash_pandas_object(f,index=False).to_numpy()
def main():
    train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv')
    official=pd.read_csv(OUT/'official_data.csv')
    old=pd.concat([pd.read_csv(ROOT/'external_data/train.csv'),pd.read_csv(ROOT/'external_data/test.csv')],ignore_index=True)
    old['satisfaction']=old.satisfaction.eq('satisfied')
    old=old[list(official)]
    for col in official:
        old[col]=old[col].astype(official[col].dtype)
    source_equal=bool(np.array_equal(np.sort(hashes(old)),np.sort(hashes(official))))
    if source_equal:
        pd.testing.assert_frame_equal(old.sort_values(list(old)).reset_index(drop=True),
                                      official.sort_values(list(official)).reset_index(drop=True))
    result={'source_comparison':dict(official_rows=len(official),previous_rows=len(old),
        normalized_rows_and_labels_identical=source_equal,official_sha256=hashlib.sha256((OUT/'official_data.csv').read_bytes()).hexdigest(),
        delay_max=official[['Departure Delay in Minutes','Arrival Delay in Minutes']].max().to_dict())}
    features=[c for c in train if c not in ['id','satisfaction']]
    y=train.satisfaction.to_numpy(dtype=int)
    drift=[]
    for c in features:
        a=train[c];b=test[c]
        if a.nunique()>30 and pd.api.types.is_numeric_dtype(a):
            stat=float(ks_2samp(a.dropna(),b.dropna()).statistic);kind='KS'
        else:
            av=a.fillna('__NA__').value_counts(normalize=True);bv=b.fillna('__NA__').value_counts(normalize=True)
            stat=float(av.subtract(bv,fill_value=0).abs().sum()/2);kind='total_variation'
        drift.append(dict(feature=c,measure=kind,distance=stat,train_missing=float(a.isna().mean()),test_missing=float(b.isna().mean())))
    result['marginal_drift']=sorted(drift,key=lambda r:-r['distance'])
    # Hash equal feature tuples only, not IDs or labels; hash collisions are negligible, not mathematically impossible.
    h=hashes(train[features]);ht=hashes(test[features])
    group=pd.DataFrame({'h':h,'y':y}).groupby('h').y.agg(['size','sum'])
    conflict=group[(group['sum']>0)&(group['sum']<group['size'])]
    result['duplicates']=dict(train_duplicate_rows=int(train[features].duplicated(keep=False).sum()),
        conflicting_groups=len(conflict),conflicting_rows=int(conflict['size'].sum()),
        test_rows_exactly_matching_train=int(np.isin(ht,h).sum()))
    v3=read(ROOT/'submission_v3_metrics.json');v4=read(ROOT/'submission_v4_metrics.json')
    base=pd.read_csv(ROOT/'submission_optimized_oof.csv');current=pd.read_csv(ROOT/'submission_v4_oof.csv')
    assert current.id.equals(train.id) and np.array_equal(current.target,y)
    model={'old_blend':base.prediction.to_numpy(),'xgb':np.zeros(len(y)),'realmlp':np.zeros(len(y)),
           'lightgbm':np.zeros(len(y)),'v4':current.prediction.to_numpy()}
    for fold in range(1,11):
        for name,directory,filename in [('xgb',v3['directory'],f'fold_{fold:02d}_xgb.npz'),
                ('realmlp',v3['directory'],f'fold_{fold:02d}_realmlp.npz'),
                ('lightgbm',v4['directory'],f'fold_{fold:02d}.npz')]:
            with np.load(Path(directory)/filename) as z:
                idx=z['valid_index'];np.testing.assert_array_equal(idx,np.flatnonzero(current.fold==fold))
                model[name][idx]=z['validation_prediction']
    result['model_auc']={k:float(roc_auc_score(y,p)) for k,p in model.items()}
    result['residual_correlations']=pd.DataFrame({k:y-p for k,p in model.items()}).corr().to_dict()
    conditional={};hard_overlap={}
    for label in [0,1]:
        idx=np.flatnonzero(y==label)
        conditional[str(label)]=pd.DataFrame({k:p[idx] for k,p in model.items()}).corr().to_dict()
        k=max(1,len(idx)//10)
        hard={name:set(np.argsort(p[idx])[-k:] if label==0 else np.argsort(p[idx])[:k]) for name,p in model.items()}
        hard_overlap[str(label)]={name:len(hard['xgb']&h)/k for name,h in hard.items()}
    result['within_label_prediction_correlations']=conditional
    result['hardest_decile_overlap_with_xgb']=hard_overlap
    result['flight_distance_unique']=int(train['Flight Distance'].nunique())
    p=model['v4'];positive=np.sort(p[y==1]);negative=np.sort(p[y==0]);np_,nn=len(positive),len(negative)
    # Per-row discordant-pair burden, half charged to each endpoint, sums exactly to 1-AUC.
    vp=(np.searchsorted(negative,p[y==1],side='left')+np.searchsorted(negative,p[y==1],side='right'))/(2*nn)
    vn=1-(np.searchsorted(positive,p[y==0],side='left')+np.searchsorted(positive,p[y==0],side='right'))/(2*np_)
    debt=np.zeros(len(y));debt[y==1]=(1-vp)/(2*np_);debt[y==0]=(1-vn)/(2*nn)
    assert abs(debt.sum()-(1-result['model_auc']['v4']))<1e-12
    groups=train['Customer Type'].astype(str)+' | '+train['Type of Travel'].astype(str)+' | '+train['Class'].astype(str)
    rows=[]
    for name in sorted(groups.unique()):
        mask=(groups==name).to_numpy();n=int(mask.sum())
        if n<100:continue
        auc={k:float(roc_auc_score(y[mask],pred[mask])) for k,pred in model.items()} if len(np.unique(y[mask]))==2 else {}
        rows.append(dict(group=name,n=n,row_share=n/len(y),positive_rate=float(y[mask].mean()),
            global_error_pair_share=float(debt[mask].sum()/debt.sum()),auc=auc))
    result['customer_groups']=sorted(rows,key=lambda z:-z['global_error_pair_share'])
    # Approximate sampling-only SE for a public-size set drawn from this OOF distribution.
    size=round(.2*len(test));npos=size*y.mean();nneg=size-npos
    se=float(np.sqrt(np.var(vp,ddof=1)/npos+np.var(vn,ddof=1)/nneg))
    result['sampling_scale']=dict(approx_public_rows=size,approx_auc_standard_error=se,
        observed_oof_lb_gap=result['model_auc']['v4']-.96082,
        caveat='OOF-based asymptotic sampling approximation, not actual public/private uncertainty; fold dependence and shift excluded')
    ratings=read(ROOT/'.artifact_runtime/v3/auxiliary.json')['ratings']
    zeros=(train[ratings]==0).sum(axis=1)
    result['unrated_diagnosis']=[]
    for name,mask in [('no_zero_rating',zeros==0),('one_or_more_zero_rating',zeros>0)]:
        mask=mask.to_numpy();result['unrated_diagnosis'].append(dict(group=name,n=int(mask.sum()),
            row_share=float(mask.mean()),global_error_pair_share=float(debt[mask].sum()/debt.sum()),
            auc=float(roc_auc_score(y[mask],p[mask]))))
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['source_comparison','duplicates','model_auc','sampling_scale']},indent=2))
    print('DRIFT',result['marginal_drift'][:5]);print('GROUPS',result['customer_groups'][:3])

if __name__=='__main__':main()
