"""One source-informed XGBoost/RealMLP screen on the existing selection partition."""
import gc
import time
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from v2_training import ROOT,load_data,write_json
from v3_features import OUT,auxiliary,frames,CONTEXT


def fit_model(a,ya,b,yb,c=None,family='xgb',seed=42,device='cuda',rounds=None,audit=False):
    start=time.time()
    if family=='xgb':
        model=XGBClassifier(n_estimators=rounds or 5000,learning_rate=.03,max_depth=6,min_child_weight=20,
            subsample=.9,colsample_bytree=.8,reg_lambda=10,tree_method='hist',device=device,
            enable_categorical=True,n_jobs=4,random_state=seed,eval_metric='auc',
            early_stopping_rounds=None if rounds else 150)
        model.fit(a,ya,eval_set=None if rounds else [(b,yb)],verbose=False)
        best=int(rounds or model.best_iteration+1);curves=model.evals_result() if not rounds else {}
        model.set_params(device='cpu')
    else:
        from pytabkit import RealMLP_TD_Classifier
        numeric=a.select_dtypes(include=['number']).columns
        medians=a[numeric].median().fillna(0)
        a=a.copy();b=b.copy();c=c.copy() if c is not None else None
        for f in [a,b]+([c] if c is not None else []):
            f[numeric]=f[numeric].fillna(medians)
            for col in CONTEXT:f[col]=f[col].astype('string').fillna('__MISSING__').astype(str)
        model=RealMLP_TD_Classifier(device=device,random_state=seed,n_cv=1,n_refit=0,
            n_epochs=128,batch_size=1024,n_threads=4,verbosity=2,use_ls=False,val_metric_name='1-auc_ovr',val_fraction=.05)
        model.fit(a,ya,cat_col_names=CONTEXT,**({} if audit else dict(X_val=b,y_val=yb)))
        best=128;curves={}
    pv=model.predict_proba(b)[:,1];pt=model.predict_proba(c)[:,1] if c is not None else None
    details=dict(family=family,seed=seed,rounds=best,auc=float(roc_auc_score(yb,pv)),
        train_auc=float(roc_auc_score(ya,model.predict_proba(a)[:,1])),seconds=time.time()-start,
        features=list(a.columns),curves=curves)
    if family!='xgb':details['fit_params']=str(getattr(model,'fit_params_',None))
    del model
    gc.collect()
    if family!='xgb':
        import torch
        torch.cuda.empty_cache()
    return pv,pt,details


def screen(family,device):
    x,xt,y,ox,oy,prior,pt,stats,meta=load_data()
    aux,auxt=auxiliary(x,xt,ox)
    s=np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz');fit=s['fit'];valid=s['selection']
    a,b=frames(x,None,y,fit,valid,prior,pt,aux,auxt)
    print('SCREEN START',family,a.shape,flush=True)
    pv,_,d=fit_model(a,y[fit],b,y[valid],family=family,device=device)
    np.save(OUT/(family+'_selection.npy'),pv);write_json(OUT/(family+'_screen.json'),d)
    print('SCREEN RESULT',family,d['auc'],d['seconds'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('family',choices=['xgb','realmlp']);p.add_argument('--device',default='cuda')
    args=p.parse_args();screen(args.family,args.device)
