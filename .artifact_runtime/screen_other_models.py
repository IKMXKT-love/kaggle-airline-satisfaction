import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc,json,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from optimized_features import normalize_raw,OriginalStatistics,CompetitionFeaturePipeline

family=sys.argv[1]
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
results=[]
configs=([{'name':'catboost_original','family':'catboost','use_te':False,'numeric_categories':False,'depth':6},
          {'name':'catboost_allcat_original','family':'catboost','use_te':False,'numeric_categories':True,'depth':6}]
         if family=='catboost' else
         [{'name':'lightgbm_te_original','family':'lightgbm','use_te':True,'numeric_categories':False,'depth':-1}])
if len(sys.argv) > 2 and sys.argv[2] == 'deep':
 configs=[{'name':'catboost_allcat_original_d8','family':'catboost','use_te':False,'numeric_categories':True,'depth':8}]
for config in configs:
 begin=time.time();print('START',config,flush=True)
 p=CompetitionFeaturePipeline(use_te=config['use_te'],original_statistics=stats,numeric_categories=config['numeric_categories'])
 a=p.fit_transform(x.iloc[fit],y[fit],prior[fit]);b=p.transform(x.iloc[validation],prior[validation])
 if family=='catboost':
  from catboost import CatBoostClassifier
  categorical=a.select_dtypes(include=['category']).columns.tolist()
  for f in [a,b]:
   for col in categorical:f[col]=f[col].astype('string').fillna('__MISSING__').astype(str)
  model=CatBoostClassifier(iterations=4000,learning_rate=.04,depth=config['depth'],l2_leaf_reg=5,
                            loss_function='Logloss',eval_metric='AUC',task_type='GPU',devices='0',
                            cat_features=categorical,random_seed=42,thread_count=8,
                            od_type='Iter',od_wait=150,max_ctr_complexity=2,allow_writing_files=False,
                            metric_period=50,verbose=False)
  model.fit(a,y[fit],eval_set=(b,y[validation]),verbose=False)
  rounds=int(model.best_iteration_+1)
 else:
  import lightgbm as lgb
  model=lgb.LGBMClassifier(n_estimators=4000,learning_rate=.03,num_leaves=31,min_child_samples=100,
                           colsample_bytree=1.,subsample=.9,subsample_freq=1,reg_lambda=5,
                           n_jobs=16,random_state=42,verbosity=-1)
  model.fit(a,y[fit],eval_set=[(b,y[validation])],eval_metric='auc',callbacks=[lgb.early_stopping(150,verbose=False)])
  rounds=int(model.best_iteration_)
 pred=model.predict_proba(b)[:,1];np.save(OUT/(config['name']+'_selection.npy'),pred)
 item={**config,'selection_auc':float(roc_auc_score(y[validation],pred)),'best_rounds':rounds,'features':len(a.columns),'seconds':time.time()-begin}
 results.append(item);print('RESULT',json.dumps(item),flush=True)
 (OUT/(family+('_deep' if len(sys.argv)>2 else '')+'_results.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
 del a,b,p,model;gc.collect()
print('DONE',flush=True)
