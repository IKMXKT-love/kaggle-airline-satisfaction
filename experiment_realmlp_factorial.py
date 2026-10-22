"""Frozen 2x2 RealMLP screen; independent stopping and fold-local features."""
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from v2_training import ROOT, load_data, write_json
from v3_features import frames, auxiliary, CONTEXT
from report_neural import paired

REFERENCE = ROOT/'.artifact_runtime/transfer/74f731e0c4b9330ecfac'
OUT = ROOT/'.artifact_runtime/realmlp_factorial'
COMBOS = [('Class','Type of Travel','Gender'), ('Class','Type of Travel','Age'),
          ('Flight Distance','Departure/Arrival time convenient')]
REFERENCE_PARAMS = dict(n_ens=8,n_epochs=3,batch_size=256,use_early_stopping=False,
    lr=.053,wd=.015,sq_mom=.988,lr_sched='flat_anneal',wd_sched='cos_log_15',
    first_layer_lr_factor=.25,embedding_size=5,max_one_hot_cat_size=18,
    hidden_sizes=[512,256,128],act='silu',p_drop=.05,p_drop_sched='invsqrtp1e-3',
    plr_hidden_1=16,plr_hidden_2=8,plr_act_name='gelu',plr_lr_factor=.1151,
    plr_sigma=2.33,ls_eps=.01,ls_eps_sched='sqrt_cos',add_front_scale=False,
    bias_init_mode='neg-uniform-dynamic-2',
    tfms=['one_hot','median_center','robust_scale','smooth_clip','embedding','l2_normalize'],
    stop_epoch=3)
OLD_PARAMS = dict(n_epochs=128,batch_size=1024,use_ls=False,val_fraction=.05)
SPECS = [('existing_old','existing','old'),('lean_reference','lean','reference'),
         ('existing_reference','existing','reference'),('lean_old','lean','old')]
PROTOCOL = dict(seed=42,specs=SPECS,reference_params=REFERENCE_PARAMS,old_params=OLD_PARAMS,
    blend_weight=.20,screen_gate=.0002,confirmation_gate=.0001,
    max_screen_candidates=3,no_outer_early_stopping=True,historical_partitions_seen=True,
    feature_fit_excludes_stopping=True,reference=str(REFERENCE),
    source='https://github.com/alvis1111/s6e10-airline-solution/blob/main/independent_repro/realmlp_repro_lean.py')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class LeanFeatures:
    """Numeric + categorical views; count maps and vocabularies fit locally."""
    @staticmethod
    def token(s):
        if pd.api.types.is_numeric_dtype(s):
            return s.astype('float64').astype('string').fillna('__MISSING__')
        return s.astype('string').fillna('__MISSING__')

    def raw(self,x):
        assert not {'id','satisfaction'} & set(x)
        z=x.copy()
        for col in self.numeric:
            z[col]=pd.to_numeric(z[col]).fillna(0).astype('float32')
            z[col+'__category']=self.token(x[col])
        for col in CONTEXT: z[col]=self.token(x[col])
        for col in CONTEXT+['Leg room service']:
            z[col+'__count']=self.token(x[col]).map(self.counts[col]).fillna(0).astype('float32')
        for i,combo in enumerate(COMBOS):
            value=pd.Series('',index=x.index,dtype='string')
            for col in combo:
                s=self.token(x[col]); value=value+s.str.len().astype(str)+':'+s
            z[f'combo_{i}']=value
        return z

    def fit_transform(self,x,y):
        self.numeric=[c for c in x if c not in CONTEXT]
        self.counts={c:self.token(x[c]).value_counts() for c in CONTEXT+['Leg room service']}
        a=self.raw(x)
        self.keys=[f'combo_{i}' for i in range(3)]
        self.te=TargetEncoder(target_type='binary',cv=5,smooth='auto',shuffle=True,random_state=42)
        te=self.te.fit_transform(a[self.keys].astype(str),y)
        self.cats=CONTEXT+[c+'__category' for c in self.numeric]+['combo_0']
        self.vocab={c:set(a[c].astype(str)) for c in self.cats}
        return self.finish(a,te)

    def finish(self,a,te):
        a=a.drop(columns=['combo_1','combo_2'])
        for i in range(3):a[f'combo_{i}__te']=te[:,i].astype('float32')
        for col in self.cats:
            s=a[col].astype(str);a[col]=s.where(s.isin(self.vocab[col]),'__UNKNOWN__')
        return a

    def transform(self,x):
        a=self.raw(x)
        return self.finish(a,self.te.transform(a[self.keys].astype(str)))

def prepare_existing(a,b,c):
    out=[f.copy() for f in [a,b,c]]
    nums=a.select_dtypes(include='number').columns
    med=a[nums].median().fillna(0)
    for f in out:
        f[nums]=f[nums].fillna(med)
        for col in CONTEXT:f[col]=f[col].astype('string').fillna('__MISSING__').astype(str)
    return out

def fit(a,ya,b,yb,c,cats,recipe):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    params=REFERENCE_PARAMS if recipe=='reference' else OLD_PARAMS
    model=RealMLP_TD_Classifier(device='cuda',random_state=42,n_cv=1,n_refit=0,
        n_threads=4,verbosity=1,val_metric_name='1-auc_ovr',**params)
    # b is inner stopping data only. Fixed stop_epoch overrides best-epoch selection.
    torch.cuda.reset_peak_memory_stats()
    start=time.time()
    model.fit(a,ya,cat_col_names=cats,X_val=b,y_val=yb)
    p=model.predict_proba(c)[:,1]
    detail=dict(seconds=time.time()-start,features=list(a),cats=cats,params=params,
        fit_params=str(getattr(model,'fit_params_',None)),
        peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3)
    del model
    gc.collect();torch.cuda.empty_cache()
    return p,detail

def main():
    import importlib.metadata
    OUT.mkdir(parents=True,exist_ok=True)
    refprotocol=read(REFERENCE/'protocol.json')
    for n,h in refprotocol['hashes'].items():
        # Actual transfer trainer preserved in a snapshot; its entrypoint later changed.
        if n=='experiment_transfer.py':
            assert sha(REFERENCE/'experiment_transfer_snapshot.py')==h
        else: assert sha(ROOT/n)==h,n
    protected=read(ROOT/'.artifact_runtime/v4/preserved.json')
    for n,h in protected.items():assert sha(ROOT/n)==h,n
    files=['train.csv','test.csv','external_data/train.csv','external_data/test.csv',
        'v3_training.py','v3_features.py','v2_training.py','optimized_features.py',
        '.artifact_runtime/writeup_search/split_indices.npz',
        '.artifact_runtime/v3/aux_train.npy','.artifact_runtime/v3/aux_test.npy',
        '.artifact_runtime/optimized_cache/train_priors.npy',
        '.artifact_runtime/optimized_cache/test_priors.npy']
    hashes={n:sha(ROOT/n) for n in files}
    hashes['experiment_realmlp_factorial.py']=sha(__file__)
    for n in ['indices.npz','baseline.npz','realmlp.npz']:
        hashes['reference/'+n]=sha(REFERENCE/'screen'/n)
    versions={p:importlib.metadata.version(p) for p in ['pytabkit','torch','scikit-learn','pandas','numpy']}
    signature=hashlib.sha256(json.dumps([PROTOCOL,hashes,versions],sort_keys=True).encode()).hexdigest()[:20]
    directory=OUT/signature;directory.mkdir(exist_ok=True)
    write_json(directory/'protocol.json',dict(**PROTOCOL,hashes=hashes,versions=versions,protected=protected))
    (directory/'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(OUT/'latest.json',dict(directory=str(directory)))
    x,xt,y,ox,oy,prior,pt,stats,meta=load_data()
    aux,auxt=auxiliary(x,xt,ox)
    with np.load(REFERENCE/'screen/indices.npz') as z:
        fitidx,stop,scored=[z[k].copy() for k in ['fit','stopping','scored']]
    assert not(set(fitidx)&set(stop) or set(fitidx)&set(scored) or set(stop)&set(scored))
    np.savez_compressed(directory/'indices.npz',fit=fitidx,stopping=stop,scored=scored)
    print('OUTPUT',directory,'ROWS',len(fitidx),len(stop),len(scored),flush=True)
    existing=prepare_existing(*frames(x,x.iloc[scored],y,fitidx,stop,prior,prior[scored],aux,aux[scored]))
    tf=LeanFeatures();lean=[tf.fit_transform(x.iloc[fitidx],y[fitidx]),tf.transform(x.iloc[stop]),tf.transform(x.iloc[scored])]
    basefile=np.load(REFERENCE/'screen/baseline.npz');np.testing.assert_array_equal(basefile['index'],scored)
    base=basefile['prediction'];base_auc=roc_auc_score(y[scored],base)
    rows=[]
    for name,features,recipe in SPECS:
        path=directory/(name+'.npz');js=path.with_suffix('.json')
        if name=='existing_old' and not path.exists():
            with np.load(REFERENCE/'screen/realmlp.npz') as z:
                np.testing.assert_array_equal(z['index'],scored);p=z['prediction'].copy()
            detail=read(REFERENCE/'screen/realmlp.json');detail['reused_verified_reference']=True
            np.savez_compressed(path,index=scored,prediction=p);write_json(js,detail)
        if not(path.exists() and js.exists()):
            a,b,c=existing if features=='existing' else lean
            for f,idx in zip([a,b,c],[fitidx,stop,scored]):np.testing.assert_array_equal(f.index,idx)
            print('START',name,'features',a.shape[1],flush=True)
            p,detail=fit(a,y[fitidx],b,y[stop],c,CONTEXT if features=='existing' else tf.cats,recipe)
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            detail['auc']=float(roc_auc_score(y[scored],p))
            np.savez_compressed(path,index=scored,prediction=p);write_json(js,detail)
            print('RESULT',name,detail['auc'],detail['seconds'],flush=True)
        with np.load(path) as z:
            np.testing.assert_array_equal(z['index'],scored);p=z['prediction'].copy()
        blend=.8*base+.2*p
        row=dict(name=name,auc=float(roc_auc_score(y[scored],p)),blend_auc=float(roc_auc_score(y[scored],blend)),
            paired=paired(y[scored],blend,base),detail=read(js))
        rows.append(row)
        write_json(directory/'progress.json',dict(baseline_auc=float(base_auc),rows=rows))
    best=max(rows[1:],key=lambda r:r['paired']['delta'])
    status='ready_for_confirmation' if best['paired']['delta']>=.0002 else 'stopped_screen'
    for n,h in protected.items():assert sha(ROOT/n)==h,n
    summary=dict(status=status,baseline_auc=float(base_auc),rows=rows,best=best['name'],
        best_gain=best['paired']['delta'],protected_verified=True)
    write_json(directory/'summary.json',summary)
    lines=['# RealMLP 四组对照实测','',f'状态：{status}。同协议基线 AUC：{base_auc:.9f}。',
        '', '| 输入/训练 | 单模型 AUC | 80% 基线＋20% 本模型 | 融合增益 | 成对约95%区间 |',
        '|---|---:|---:|---:|---|']
    for r in rows:
        lo,hi=r['paired']['interval95']
        lines.append(f"| {r['name']} | {r['auc']:.9f} | {r['blend_auc']:.9f} | {r['paired']['delta']:+.9f} | [{lo:+.9f}, {hi:+.9f}] |")
    lines+=['','划分：411944 拟合 / 35822 内部停止 / 111942 计分；外部计分标签不用于停止。',
        '现有输入＋原配置复用上轮相同划分的 RealMLP 预测，输入、训练实现和缓存哈希均核对。',
        '精简特征全部在拟合区学习词表与频次，三个组合编码采用内层五折交叉拟合。',
        '新配方采用 8 个内部成员、固定 3 epoch；原配方 128 epoch。预算差异是配方对照的一部分，非等算力架构比较。',
        '参考配方来源：https://github.com/alvis1111/s6e10-airline-solution/blob/main/independent_repro/realmlp_repro_lean.py',
        '门槛：筛选融合 +0.0002，冻结后确认 +0.0001；通过后才做完整10折。',
        '历史划分已看过；成对区间不覆盖训练随机性及多次筛选。公榜最佳仍为用户反馈的0.96082。',
        f'缓存：{directory}。原有受保护文件哈希检查通过。']
    (ROOT/'REALMLP_FACTORIAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('DECISION',status,'BEST',best['name'],best['paired']['delta'],flush=True)

if __name__=='__main__': main()
