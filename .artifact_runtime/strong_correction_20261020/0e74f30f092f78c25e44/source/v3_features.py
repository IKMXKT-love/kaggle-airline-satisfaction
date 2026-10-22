"""Distance profiles, cross-fitted target statistics and public-source auxiliary ratings."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from optimized_features import NativeCategoryEncoder
from v2_training import ROOT,load_data,write_json

OUT=ROOT/'.artifact_runtime/v3';OUT.mkdir(parents=True,exist_ok=True)
FD='Flight Distance'
CONT=['Age',FD,'Departure Delay in Minutes','Arrival Delay in Minutes']
CONTEXT=['Gender','Customer Type','Type of Travel','Class']


def auxiliary(x,xt,ox):
    signature=hashlib.sha256(b''.join(pd.util.hash_pandas_object(v.astype('string'),index=False).to_numpy().tobytes() for v in [x,xt,ox])).hexdigest()
    meta=OUT/'auxiliary.json'
    if meta.exists() and json.loads(meta.read_text())['signature']==signature:
        return np.load(OUT/'aux_train.npy'),np.load(OUT/'aux_test.npy')
    ratings=[c for c in ox.select_dtypes(include=['number']) if c not in CONT]
    train_pred=[];test_pred=[]
    for col in ratings:
        a=ox.drop(columns=col);p=NativeCategoryEncoder().fit(a)
        file=OUT/('aux_'+str(ratings.index(col))+'.npz')
        if file.exists():
            with np.load(file) as saved:
                if str(saved['signature'])==signature:
                    train_pred.append(saved['train']);test_pred.append(saved['test']);continue
        print('AUX START',col,flush=True)
        model=XGBRegressor(n_estimators=300,max_depth=6,learning_rate=.05,min_child_weight=10,
            subsample=.9,colsample_bytree=.9,reg_lambda=5,tree_method='hist',device='cpu',n_jobs=4,random_state=42)
        model.fit(p.transform(a),ox[col],verbose=False)
        tr=model.predict(p.transform(x.drop(columns=col)));ts=model.predict(p.transform(xt.drop(columns=col)))
        train_pred.append(tr);test_pred.append(ts)
        np.savez_compressed(file,signature=signature,train=tr,test=ts)
        print('AUX DONE',col,flush=True)
    tr=np.column_stack(train_pred);ts=np.column_stack(test_pred)
    np.save(OUT/'aux_train.npy',tr);np.save(OUT/'aux_test.npy',ts)
    write_json(meta,dict(signature=signature,ratings=ratings,source_labels='public_original_predictor_ratings_only',
        satisfaction_labels_used=False,each_rating_excluded_from_own_predictors=True))
    return tr,ts


def keys(x):
    result={}
    for group in [[FD],['Age'],[FD,'Age'],[FD,'Class'],[FD,'Type of Travel'],['Age','Class','Type of Travel','Customer Type']]:
        s=pd.Series('',index=x.index,dtype='string')
        for col in group:
            v=x[col].astype('string').fillna('__MISSING__');s=s+v.str.len().astype(str)+':'+v
        result['te__'+'__'.join(group)]=s
    return pd.DataFrame(result)


class RouteFeatures:
    def fit(self,x):
        self.categories={c:sorted(x[c].astype(str).unique()) for c in CONTEXT}
        z=self.numeric(x);self.means=z.drop(columns=FD).mean()
        self.route=z.groupby(FD,dropna=False).mean();self.route['route_count']=z.groupby(FD,dropna=False).size()
        self.counts={c:x[c].value_counts(dropna=False) for c in CONT}
        return self
    def numeric(self,x):
        z=x.copy()
        for c in CONTEXT:z[c]=pd.Categorical(z[c].astype(str),categories=self.categories[c]).codes.astype('float32')
        return z
    def transform(self,x):
        d={}
        for c in self.route:
            d['route_mean__'+c]=x[FD].map(self.route[c]).fillna(0 if c=='route_count' else self.means[c]).to_numpy(dtype='float32')
        for c,t in self.counts.items():d['frequency__'+c]=np.log1p(x[c].map(t).fillna(0)).to_numpy(dtype='float32')
        return pd.DataFrame(d,index=x.index)


def frames(x,xt,y,fit,valid,prior,pt,aux,auxt):
    route=RouteFeatures().fit(x.iloc[fit])
    encoder=TargetEncoder(target_type='binary',smooth=20,cv=StratifiedKFold(5,shuffle=True,random_state=42))
    ka=keys(x.iloc[fit]);a_te=encoder.fit_transform(ka,y[fit]);cols=list(ka)
    parts=[]
    for raw,teacher,auxiliary_values,te in [
        (x.iloc[fit],prior[fit],aux[fit],a_te),
        (x.iloc[valid],prior[valid],aux[valid],encoder.transform(keys(x.iloc[valid]))),
    ]+([(xt,pt,auxt,encoder.transform(keys(xt)))] if xt is not None else []):
        f=pd.concat([raw.copy(),route.transform(raw),pd.DataFrame(te.astype('float32'),columns=cols,index=raw.index)],axis=1)
        f['teacher_logit']=np.log(np.clip(teacher,1e-6,1-1e-6)/np.clip(1-teacher,1e-6,1-1e-6))
        for i in range(auxiliary_values.shape[1]):f['aux_expected_rating_'+str(i)]=auxiliary_values[:,i]
        parts.append(f)
    native=NativeCategoryEncoder().fit(parts[0])
    return tuple(native.transform(p) for p in parts)


if __name__=='__main__':
    x,xt,y,ox,oy,prior,pt,stats,meta=load_data()
    a,b=auxiliary(x,xt,ox)
    print('AUXILIARY COMPLETE',a.shape,b.shape,flush=True)
