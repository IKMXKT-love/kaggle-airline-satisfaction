"""Official TabM model with fold-local preprocessing and independent-member loss."""
import copy
import gc
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.metrics import roc_auc_score

from optimized_features import CompetitionFeaturePipeline
from v2_training import OUT, ROOT, load_data, write_json

DEFAULT = dict(name='tabm_ple', k=16, width=256, blocks=2, embedding=16,
    bins=32, dropout=.1, lr=.002, weight_decay=.0003, batch_size=1024,
    max_epochs=60, patience=10)
COMPACT = dict(DEFAULT, name='tabm_compact', width=512, k=32,
    bins=48, lr=.001, raw_only=True)


def prepare_frames(a, b, c=None):
    """Every learned transformation is fitted using only the training partition."""
    cats = a.select_dtypes(include=['category','object','string']).columns.tolist()
    nums = [col for col in a if col not in cats and a[col].nunique() > 1]
    impute = SimpleImputer(strategy='median')
    na = impute.fit_transform(a[nums]).astype('float32')
    # Same small tie-breaking noise as the official example, in fitting data only.
    noise = np.random.default_rng(42).normal(0,1e-5,na.shape).astype('float32')
    scaler = QuantileTransformer(n_quantiles=min(1000,len(na)),output_distribution='normal',
        subsample=200000,random_state=42).fit(na+noise)
    num = [scaler.transform(na).astype('float32')]
    for frame in [b]+([c] if c is not None else []):
        num.append(scaler.transform(impute.transform(frame[nums])).astype('float32'))
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    cat = [encoder.fit_transform(a[cats].astype(str)).astype('int64')+1]
    for frame in [b]+([c] if c is not None else []):
        cat.append(encoder.transform(frame[cats].astype(str)).astype('int64')+1)
    cardinalities = [len(values)+1 for values in encoder.categories_]
    return num, cat, cardinalities, dict(imputer=impute,scaler=scaler,encoder=encoder,numeric=nums,categorical=cats)


def fit_tabm(a, ya, b, yb, c=None, config=None, seed=42, epochs=None, checkpoint=None):
    import torch
    import torch.nn.functional as F
    import tabm
    import rtdl_num_embeddings as embeddings
    import joblib
    config = DEFAULT if config is None else config
    assert torch.cuda.is_available(), 'GPU PyTorch is required for this experiment.'
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    begin = time.time()
    num,cat,cards,preprocessor = prepare_frames(a,b,c)
    num = [torch.as_tensor(v,device=device) for v in num]
    cat = [torch.as_tensor(v,device=device) for v in cat]
    target = torch.as_tensor(ya,dtype=torch.float32,device=device)
    bins = embeddings.compute_bins(num[0],n_bins=config['bins'])
    emb = embeddings.PiecewiseLinearEmbeddings(bins,d_embedding=config['embedding'],activation=False,version='B')
    model = tabm.TabM.make(n_num_features=num[0].shape[1],cat_cardinalities=cards,d_out=1,
        num_embeddings=emb,k=config['k'],d_block=config['width'],n_blocks=config['blocks'],dropout=config['dropout']).to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    def forward(part, idx):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(num[part][idx],cat[part][idx]).squeeze(-1).float()
    @torch.inference_mode()
    def predict(part, limit=None):
        model.eval()
        length = len(num[part]) if limit is None else min(limit,len(num[part]))
        return torch.cat([forward(part,slice(i,min(i+2048,length))).sigmoid().mean(1) for i in range(0,length,2048)]).cpu().numpy()
    best_auc=-np.inf; best_epoch=0; best_state=None; curves=[]
    for epoch in range(1,(epochs or config['max_epochs'])+1):
        model.train(); total=0
        indices=torch.randperm(len(target),device=device)
        for ix in indices.split(config['batch_size']):
            optim.zero_grad(set_to_none=True)
            logits=forward(0,ix)
            # Each ensemble member has its own loss; average probabilities at inference.
            loss=F.binary_cross_entropy_with_logits(logits,target[ix,None].expand_as(logits))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optim.step()
            total+=float(loss.detach())*len(ix)
        # Fixed-epoch confirmation never inspects validation labels during training.
        if epochs is None:
            score=float(roc_auc_score(yb,predict(1)))
            row=dict(epoch=epoch,loss=total/len(target),auc=score,seconds=time.time()-begin)
            curves.append(row)
            print('EPOCH',config['name'],epoch,round(score,7),round(row['seconds'],1),flush=True)
            if score>best_auc+1e-6:
                best_auc=score;best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch-best_epoch>=config['patience']: break
        else:
            best_epoch=epoch
            print('FIXED_EPOCH',epoch,flush=True)
    if best_state is not None: model.load_state_dict(best_state)
    pred=predict(1); pt=predict(2) if c is not None else None
    details=dict(config=config,seed=seed,epochs=best_epoch,auc=float(roc_auc_score(yb,pred)),
        train_auc_sample=float(roc_auc_score(ya[:50000],predict(0,50000))),curves=curves,
        seconds=time.time()-begin,features=list(a.columns),official_package=tabm.__version__ if hasattr(tabm,'__version__') else '0.0.3')
    if checkpoint is not None:
        torch.save(dict(state=model.state_dict(),bins=[v.cpu() for v in bins],cards=cards,config=config),str(checkpoint)+'.pt')
        joblib.dump(preprocessor,str(checkpoint)+'.preprocessing.joblib')
    del model,optim,num,cat,target
    gc.collect();torch.cuda.empty_cache()
    return pred,pt,details


def frames(x,xt,y,fit,valid,stats,prior,pt,raw_only=False):
    pipeline=CompetitionFeaturePipeline(original_statistics=None if raw_only else stats)
    a=pipeline.fit_transform(x.iloc[fit],y[fit],prior[fit])
    b=pipeline.transform(x.iloc[valid],prior[valid])
    c=pipeline.transform(xt,pt) if xt is not None else None
    # Low-cardinality service ratings are discrete categories as well as ordered scores.
    rating=[col for col in x.select_dtypes(include=['number']) if x[col].nunique()<=10]
    for frame,raw in [(a,x.iloc[fit]),(b,x.iloc[valid])]+([(c,xt)] if c is not None else []):
        for col in rating: frame['rating_cat__'+col]=raw[col].astype('string').fillna('__MISSING__')
    return a,b,c


def screen(config=None):
    config=DEFAULT if config is None else config
    x,xt,y,ox,oy,prior,pt,stats,meta=load_data()
    split=np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz')
    fit,valid=split['fit'],split['selection']
    a,b,c=frames(x,None,y,fit,valid,stats,prior,pt,raw_only=config.get('raw_only',False))
    pred,_,details=fit_tabm(a,y[fit],b,y[valid],config=config,checkpoint=OUT/(config['name']+'_selection_model'))
    np.save(OUT/(config['name']+'_selection.npy'),pred)
    write_json(OUT/(config['name']+'.json'),details)
    print('RESULT',details['auc'],flush=True)


if __name__=='__main__':
    import sys
    screen(COMPACT if '--compact' in sys.argv else DEFAULT)
