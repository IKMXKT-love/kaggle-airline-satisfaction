import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc,json,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from optimized_features import normalize_raw,OriginalStatistics,CompetitionFeaturePipeline,NativeCategoryEncoder
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.artifact_runtime'/'writeup_search'
train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv');cols=[c for c in test if c!='id']
x=normalize_raw(train[cols]);xt=normalize_raw(test[cols]);y=train.satisfaction.to_numpy(dtype=int)
original=pd.concat([pd.read_csv(ROOT/'external_data'/'train.csv'),pd.read_csv(ROOT/'external_data'/'test.csv')],ignore_index=True)
ox=normalize_raw(original[cols]);oy=original.satisfaction.eq('satisfied').to_numpy(dtype=int)
hashes=pd.util.hash_pandas_object(pd.concat([x,xt],ignore_index=True).astype('string').fillna('__MISSING__'),index=False)
keep=~pd.util.hash_pandas_object(ox.astype('string').fillna('__MISSING__'),index=False).isin(set(hashes))
ox=ox.loc[keep].reset_index(drop=True);oy=oy[keep.to_numpy()]
split=np.load(OUT/'split_indices.npz');fit=split['fit'];validation=split['selection']
prior_list=[np.load(OUT/'original_train_prediction.npy')];test_prior_list=[np.load(OUT/'original_test_prediction.npy')]
native=NativeCategoryEncoder().fit(ox)
for depth,rounds in [(3,1800),(9,1000)]:
 print('SOURCE PRIOR',depth,rounds,flush=True)
 m=XGBClassifier(n_estimators=rounds,learning_rate=.04,max_depth=depth,min_child_weight=5,subsample=.9,
                  colsample_bytree=1.,reg_lambda=5,tree_method='hist',device='cuda',enable_categorical=True,n_jobs=8,random_state=42)
 m.fit(native.transform(ox),oy,verbose=False);m.set_params(device='cpu')
 prior_list.append(m.predict_proba(native.transform(x))[:,1]);test_prior_list.append(m.predict_proba(native.transform(xt))[:,1])
 m.save_model(OUT/f'original_prior_depth{depth}.ubj')
 del m;gc.collect()
prior=np.column_stack(prior_list);test_prior=np.column_stack(test_prior_list)
np.save(OUT/'multiple_original_train_prediction.npy',prior);np.save(OUT/'multiple_original_test_prediction.npy',test_prior)
results=[]
for name,extended,depth in [('multi_prior_te_d5',False,5),('interaction_multi_prior_d5',True,5),('interaction_multi_prior_d7',True,7)]:
 begin=time.time();print('START',name,flush=True)
 stats=OriginalStatistics(extended=extended).fit(ox,oy)
 p=CompetitionFeaturePipeline(use_te=True,original_statistics=stats,extended_keys=extended)
 a=p.fit_transform(x.iloc[fit],y[fit],prior[fit]);b=p.transform(x.iloc[validation],prior[validation])
 m=XGBClassifier(n_estimators=5000,learning_rate=.035,max_depth=depth,min_child_weight=10,subsample=.9,
                  colsample_bytree=1.,reg_lambda=5,tree_method='hist',device='cuda',enable_categorical=True,
                  n_jobs=8,random_state=42,eval_metric='auc',early_stopping_rounds=150)
 m.fit(a,y[fit],eval_set=[(b,y[validation])],verbose=False);m.set_params(device='cpu');pred=m.predict_proba(b)[:,1]
 np.save(OUT/(name+'_selection.npy'),pred)
 item={'name':name,'family':'xgboost','use_te':True,'original':True,'multiple_priors':True,'extended_keys':extended,
       'depth':depth,'child':10,'lr':.035,'selection_auc':float(roc_auc_score(y[validation],pred)),
       'best_rounds':int(m.best_iteration+1),'features':len(a.columns),'seconds':time.time()-begin}
 results.append(item);print('RESULT',json.dumps(item),flush=True)
 (OUT/'prior_interaction_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 del a,b,p,m;gc.collect()
print('DONE',flush=True)
