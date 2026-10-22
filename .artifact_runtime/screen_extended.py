import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc,json,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from optimized_features import normalize_raw,OriginalStatistics,CompetitionFeaturePipeline
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.artifact_runtime'/'writeup_search'
train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv');cols=[c for c in test if c!='id']
x=normalize_raw(train[cols]);y=train.satisfaction.to_numpy(dtype=int)
original=pd.concat([pd.read_csv(ROOT/'external_data'/'train.csv'),pd.read_csv(ROOT/'external_data'/'test.csv')],ignore_index=True)
ox=normalize_raw(original[cols]);oy=original.satisfaction.eq('satisfied').to_numpy(dtype=int)
hashes=pd.util.hash_pandas_object(pd.concat([x,normalize_raw(test[cols])],ignore_index=True).astype('string').fillna('__MISSING__'),index=False)
keep=~pd.util.hash_pandas_object(ox.astype('string').fillna('__MISSING__'),index=False).isin(set(hashes))
ox=ox.loc[keep].reset_index(drop=True);oy=oy[keep.to_numpy()]
stats=OriginalStatistics().fit(ox,oy)
prior=np.load(OUT/'original_train_prediction.npy');split=np.load(OUT/'split_indices.npz');fit=split['fit'];validation=split['selection']
configs=[
 {'name':'allcat_original_d5','use_te':False,'original':True,'numeric_categories':True,'digits':False,'depth':5,'child':10},
 {'name':'digits_original_d5','use_te':False,'original':True,'numeric_categories':False,'digits':True,'depth':5,'child':10},
 {'name':'te_digits_original_d4','use_te':True,'original':True,'numeric_categories':False,'digits':True,'depth':4,'child':20},
 {'name':'raw_original_rows025','use_te':False,'original':False,'rows_weight':.25,'depth':5,'child':10},
 {'name':'raw_original_rows1','use_te':False,'original':False,'rows_weight':1.,'depth':5,'child':10},
]
results=[]
for config in configs:
 begin=time.time();print('START',config,flush=True)
 p=CompetitionFeaturePipeline(use_te=config['use_te'],original_statistics=stats if config['original'] else None,
                              numeric_categories=config.get('numeric_categories',False),digits=config.get('digits',False))
 a_raw=x.iloc[fit].copy();fit_y=y[fit];weights=None
 if config.get('rows_weight'):
  a_raw['is_original_source']=0.;other=ox.copy();other['is_original_source']=1.
  a_raw=pd.concat([a_raw,other],ignore_index=True);fit_y=np.r_[fit_y,oy]
  weights=np.r_[np.ones(len(fit)),np.full(len(oy),config['rows_weight'])]
 b_raw=x.iloc[validation].copy()
 if config.get('rows_weight'):b_raw['is_original_source']=0.
 a=p.fit_transform(a_raw,fit_y,prior[fit] if config['original'] else None)
 b=p.transform(b_raw,prior[validation] if config['original'] else None)
 model=XGBClassifier(n_estimators=5000,learning_rate=.035,max_depth=config['depth'],min_child_weight=config['child'],
                     subsample=.9,colsample_bytree=1.,reg_lambda=5,tree_method='hist',device='cuda',enable_categorical=True,
                     n_jobs=8,random_state=42,eval_metric='auc',early_stopping_rounds=150,max_cat_threshold=64)
 model.fit(a,fit_y,sample_weight=weights,eval_set=[(b,y[validation])],verbose=False)
 model.set_params(device='cpu');pred=model.predict_proba(b)[:,1];np.save(OUT/(config['name']+'_selection.npy'),pred)
 item={**config,'lr':.035,'selection_auc':float(roc_auc_score(y[validation],pred)),'best_rounds':int(model.best_iteration+1),
       'features':len(a.columns),'seconds':time.time()-begin}
 results.append(item);print('RESULT',json.dumps(item),flush=True)
 (OUT/'extended_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 del a,b,p,model;gc.collect()
print('DONE',flush=True)
