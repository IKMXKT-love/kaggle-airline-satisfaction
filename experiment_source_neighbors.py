"""One source-only conditional neighbor representation; one XGB configuration."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from experiment_realmlp_factorial import ROOT,sha,read
from v2_training import load_data,write_json
from v3_features import auxiliary,frames,CONTEXT
from experiment_transfer import fit_xgb,SPECS
from experiment_followup import blend
from report_neural import paired

OUT=ROOT/'.artifact_runtime/source_neighbors_20261019'
PROTOCOL=dict(k=[16,64],shrink=20,group_columns=CONTEXT,numeric_scale='source_IQR_clip8_delay_log1p',
    neighbors_from_public_source_only=True,competition_targets_used_for_neighbors=False,
    workers=2,max_projected_feature_seconds=600,screen_gate=.0002,confirmation_gate=.0001,
    new_xgb_weight_within_v4=.28,one_feature_variant=True)

def group_key(x):
    z=pd.Series('',index=x.index,dtype='string')
    for col in CONTEXT:
        s=x[col].astype('string').fillna('__MISSING__');z=z+s.str.len().astype(str)+':'+s
    return z

class SourceNeighbors:
    def numeric(self,x):
        a=x[self.columns].to_numpy(dtype='float64',copy=True)
        for j,col in enumerate(self.columns):
            if col in ['Departure Delay in Minutes','Arrival Delay in Minutes']:a[:,j]=np.log1p(np.maximum(a[:,j],0))
        return a
    def fit(self,source,labels):
        self.columns=[c for c in source if c not in CONTEXT]
        a=self.numeric(source);self.median=np.nanmedian(a,axis=0);self.median=np.nan_to_num(self.median)
        a=np.where(np.isfinite(a),a,self.median)
        self.scale=np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0);self.scale=np.where(self.scale>0,self.scale,1.)
        a=np.clip((a-self.median)/self.scale,-8,8)
        self.global_prior=float(np.mean(labels));self.groups={}
        indices=pd.DataFrame({'key':group_key(source).to_numpy()}).groupby('key').indices
        self.fallback=(cKDTree(a),np.asarray(labels))
        for key,idx in indices.items():self.groups[str(key)]=(cKDTree(a[idx]),np.asarray(labels)[idx])
        return self
    def transform(self,x):
        assert not {'id','satisfaction'}&set(x)
        a=self.numeric(x);a=np.where(np.isfinite(a),a,self.median);a=np.clip((a-self.median)/self.scale,-8,8)
        keys=group_key(x).to_numpy();out=np.zeros((len(x),10),dtype='float32')
        for key,idx in pd.DataFrame({'key':keys}).groupby('key').indices.items():
            fallback=str(key) not in self.groups;tree,labels=self.groups.get(str(key),self.fallback)
            for start in range(0,len(idx),4096):
                rows=idx[start:start+4096];count=min(64,len(labels))
                distance,neighbor=tree.query(a[rows],k=count,workers=2)
                if count==1:distance=distance[:,None];neighbor=neighbor[:,None]
                yy=labels[neighbor];n16=min(16,count)
                p16=(yy[:,:n16].sum(1)+20*self.global_prior)/(n16+20)
                p64=(yy.sum(1)+20*self.global_prior)/(count+20)
                w=1/(distance+.1);neff=w.sum(1)**2/(w*w).sum(1)
                weighted=(w*yy).sum(1)/w.sum(1)
                wp=(weighted*neff+20*self.global_prior)/(neff+20)
                npos=yy.sum(1);nneg=count-npos
                pos=(distance*yy).sum(1)/np.maximum(npos,1)
                neg=(distance*(1-yy)).sum(1)/np.maximum(nneg,1)
                gap=np.where((npos>0)&(nneg>0),pos-neg,0.)
                out[rows]=np.column_stack([p16,p64,wp,np.log1p(distance[:,0]),
                    np.log1p(distance[:,n16-1]),np.log1p(distance[:,-1]),np.log1p(distance.mean(1)),
                    gap,np.full(len(rows),np.log1p(len(labels))),np.full(len(rows),int(fallback))])
        assert np.isfinite(out).all()
        return out

def setup():
    follow=Path(read(ROOT/'.artifact_runtime/followup_20261019/latest.json')['directory'])
    assert read(follow/'verification.json')['verified']
    files=['experiment_source_neighbors.py','experiment_transfer.py','experiment_followup.py',
        'v3_features.py','optimized_features.py','optimized_training.py','v2_training.py',
        'train.csv','test.csv','external_data/train.csv','external_data/test.csv']
    hashes={n:sha(ROOT/n) for n in files}
    hashes['follow_protocol']=sha(follow/'protocol.json');hashes['follow_choice']=sha(follow/'frozen_choice.json')
    signature=hashlib.sha256(json.dumps([PROTOCOL,hashes],sort_keys=True).encode()).hexdigest()[:20]
    directory=OUT/signature;directory.mkdir(parents=True,exist_ok=True)
    protected=read(follow/'protocol.json')['protected']
    for n,h in protected.items():assert sha(ROOT/n)==h,n
    write_json(directory/'protocol.json',dict(**PROTOCOL,hashes=hashes,follow=str(follow),protected=protected))
    (directory/'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(OUT/'latest.json',dict(directory=str(directory)))
    return directory,follow

def prepare():
    directory,follow=setup();cache=directory/'features.npz'
    if cache.exists() or (directory/'resource_stop.json').exists():return directory
    x,xt,y,ox,oy,*_=load_data()
    start=time.time();tf=SourceNeighbors().fit(ox,oy)
    probeidx=np.random.default_rng(42).choice(len(x),10000,replace=False)
    tf.transform(x.iloc[probeidx]);probe_seconds=time.time()-start
    projected=probe_seconds*(len(x)+len(xt))/10000
    print('NEIGHBOR_RESOURCE',probe_seconds,projected,flush=True)
    if projected>600:
        write_json(directory/'resource_stop.json',dict(probe_seconds=probe_seconds,projected=projected));return directory
    values=[]
    for name,frame in [('train',x),('test',xt)]:
        p=[]
        for first in range(0,len(frame),50000):
            p.append(tf.transform(frame.iloc[first:first+50000]))
            print('NEIGHBOR_FEATURES',name,min(first+50000,len(frame)),len(frame),flush=True)
        values.append(np.concatenate(p))
    np.savez_compressed(cache,train=values[0],test=values[1])
    write_json(directory/'features.json',dict(seconds=time.time()-start,source_rows=len(ox),
        features=10,sha256=sha(cache),source_only=True))
    return directory

def run_stage(stage):
    directory,follow=setup();folder=directory/stage;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    x,xt,y,ox,oy,prior,pt,stats,meta=load_data();aux,auxt=auxiliary(x,xt,ox)
    z=np.load(follow/stage/'indices.npz');fitidx,stop,scored=[z[k] for k in ['fit','stopping','scored']]
    assert not set(scored)&(set(fitidx)|set(stop))
    with np.load(directory/'features.npz') as z:features=z['train']
    assert sha(directory/'features.npz')==read(directory/'features.json')['sha256']
    a,b,c=frames(x,x.iloc[scored],y,fitidx,stop,prior,prior[scored],aux,aux[scored])
    cols=[f'source_joint_neighbor_{i}' for i in range(10)]
    matrices=[pd.concat([f,pd.DataFrame(features[idx],columns=cols,index=f.index)],axis=1)
              for f,idx in [(a,fitidx),(b,stop),(c,scored)]]
    start=time.time();print('NEIGHBOR_XGB_START',stage,flush=True)
    p,d=fit_xgb(*[matrices[0],y[fitidx],matrices[1],y[stop],matrices[2],SPECS[0]])
    np.savez_compressed(folder/'prediction.npz',index=scored,prediction=p)
    np.savez_compressed(folder/'indices.npz',fit=fitidx,stopping=stop,scored=scored)
    def get(n):
        with np.load(follow/stage/(n+'.npz')) as saved:
            np.testing.assert_array_equal(scored,saved['index']);return saved['prediction'].astype(float)
    base=get('baseline');lean=get('lean_mean');control=get('control')
    choice=read(follow/'decision.json')['chosen_fusion']
    anchor=blend(base,lean,choice['weight'],choice['form'])
    new=blend(base+.28*(p-control),lean,choice['weight'],choice['form'])
    assert np.isfinite(new).all() and ((new>=0)&(new<=1)).all()
    result=dict(stage=stage,auc=float(roc_auc_score(y[scored],p)),features=76,
        anchor_auc=float(roc_auc_score(y[scored],anchor)),blend_auc=float(roc_auc_score(y[scored],new)),
        paired=paired(y[scored],new,anchor),seconds=time.time()-start,model=d)
    write_json(folder/'summary.json',result);print('NEIGHBOR_RESULT',stage,result['paired'],flush=True)
    return result

def main(stage):
    directory=prepare()
    if stage=='prepare' or (directory/'resource_stop.json').exists():return
    screen=run_stage('screen')
    if screen['paired']['delta']<.0002:
        write_json(directory/'decision.json',dict(status='stopped_screen',screen=screen['paired']));return
    confirm=run_stage('confirmation')
    passed=confirm['paired']['delta']>=.0001
    write_json(directory/'decision.json',dict(status='ready_for_cv' if passed else 'stopped_confirmation',
        screen=screen['paired'],confirmation=confirm['paired']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','all'],nargs='?',default='all')
    main(p.parse_args().stage)
