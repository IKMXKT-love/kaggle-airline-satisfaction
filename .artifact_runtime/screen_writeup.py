"""固定开发集筛选；20% 审计集不会参与这里的选择。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gc
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from optimized_features import normalize_raw, NativeCategoryEncoder, OriginalStatistics, CompetitionFeaturePipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.artifact_runtime' / 'writeup_search'
OUT.mkdir(exist_ok=True)
train = pd.read_csv(ROOT / 'train.csv')
test = pd.read_csv(ROOT / 'test.csv')
features = [c for c in test if c != 'id']
x = normalize_raw(train[features]); xt = normalize_raw(test[features])
y = train.satisfaction.to_numpy(dtype=int)
original = pd.concat([pd.read_csv(ROOT/'external_data'/'train.csv'), pd.read_csv(ROOT/'external_data'/'test.csv')], ignore_index=True)
original_y = original.satisfaction.eq('satisfied').to_numpy(dtype=int)
original_x = normalize_raw(original[features])
# 删除与比赛任何一行特征完全一致的原始数据，避免重复标签查表影响本地评估。
all_hashes = pd.util.hash_pandas_object(pd.concat([x, xt], ignore_index=True).astype('string').fillna('__MISSING__'), index=False)
original_hashes = pd.util.hash_pandas_object(original_x.astype('string').fillna('__MISSING__'), index=False)
keep = ~original_hashes.isin(set(all_hashes))
original_x = original_x.loc[keep].reset_index(drop=True); original_y=original_y[keep.to_numpy()]
print('Original rows retained:', len(original_x), 'removed overlapping:', int((~keep).sum()),flush=True)
native = NativeCategoryEncoder().fit(original_x)
orig_model = XGBClassifier(n_estimators=1200, learning_rate=.04, max_depth=6, min_child_weight=5,
                          subsample=.9, colsample_bytree=1.0, reg_lambda=5,
                          tree_method='hist', device='cuda', enable_categorical=True, n_jobs=8, random_state=42)
orig_model.fit(native.transform(original_x),original_y,verbose=False)
orig_model.set_params(device='cpu')
orig_train_prediction=orig_model.predict_proba(native.transform(x))[:,1]
orig_test_prediction=orig_model.predict_proba(native.transform(xt))[:,1]
print('Original-only competition AUC:',roc_auc_score(y,orig_train_prediction),flush=True)
np.save(OUT/'original_train_prediction.npy',orig_train_prediction)
np.save(OUT/'original_test_prediction.npy',orig_test_prediction)
orig_model.save_model(OUT/'original_prior.ubj')
metadata={'original_rows':len(original_x),'removed_exact_overlap':int((~keep).sum()),'features':features}
(OUT/'original_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
stats=OriginalStatistics().fit(original_x,original_y)
del orig_model; gc.collect()

development, audit = train_test_split(np.arange(len(x)),test_size=.2,stratify=y,random_state=20261017)
fit, selection = train_test_split(development,test_size=.2,stratify=y[development],random_state=42)
np.savez(OUT/'split_indices.npz',fit=fit,selection=selection,development=development,audit=audit)
results=[]
configs=[
    {'name':'raw_native_d5','use_te':False,'original':False,'depth':5,'child':10,'lr':.04},
    {'name':'raw_native_d7','use_te':False,'original':False,'depth':7,'child':20,'lr':.04},
    {'name':'original_d5','use_te':False,'original':True,'depth':5,'child':10,'lr':.04},
    {'name':'te_original_d5','use_te':True,'original':True,'depth':5,'child':10,'lr':.04},
    {'name':'te_original_d7','use_te':True,'original':True,'depth':7,'child':20,'lr':.04},
]
for config in configs:
    begin=time.time(); print('START',config,flush=True)
    p=CompetitionFeaturePipeline(use_te=config['use_te'],original_statistics=stats if config['original'] else None)
    a=p.fit_transform(x.iloc[fit],y[fit],orig_train_prediction[fit] if config['original'] else None)
    b=p.transform(x.iloc[selection],orig_train_prediction[selection] if config['original'] else None)
    model=XGBClassifier(n_estimators=5000,learning_rate=config['lr'],max_depth=config['depth'],
                        min_child_weight=config['child'],subsample=.9,colsample_bytree=1.,reg_lambda=5,
                        tree_method='hist',device='cuda',enable_categorical=True,n_jobs=8,
                        random_state=42,eval_metric='auc',early_stopping_rounds=150)
    model.fit(a,y[fit],eval_set=[(b,y[selection])],verbose=False)
    model.set_params(device='cpu'); pred=model.predict_proba(b)[:,1]
    np.save(OUT/(config['name']+'_selection.npy'),pred)
    item={**config,'selection_auc':float(roc_auc_score(y[selection],pred)),
          'best_rounds':int(model.best_iteration+1),'features':len(a.columns),'seconds':time.time()-begin}
    results.append(item); print('RESULT',json.dumps(item),flush=True)
    (OUT/'screen_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    del model,p,a,b;gc.collect()
print('DONE',flush=True)
