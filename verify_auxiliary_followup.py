"""Independently verify reduced-context TabICL and source-neighbor screens."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import expit,logit
from experiment_realmlp_factorial import ROOT,read,sha
from report_neural import paired
from v2_training import write_json

def main():
    y=pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy()
    ref=ROOT/'.artifact_runtime/transfer/74f731e0c4b9330ecfac/screen'
    result={}
    td=Path(read(ROOT/'.artifact_runtime/tabicl_probe_20261019/latest.json')['directory'])
    p=read(td/'protocol.json');summary=read(td/'summary.json');idx=np.load(td/'indices.npz')
    reference_idx=np.load(ref/'indices.npz')
    np.testing.assert_array_equal(idx['scored'],reference_idx['scored'])
    assert set(idx['context8192'])<=set(idx['context32768'])<=set(reference_idx['fit'])
    assert not set(idx['context32768'])&set(idx['scored'])
    model=td.parent/'tabicl-classifier-v2-20260212.ckpt';assert sha(model)==p['checkpoint_sha256']
    for n,h in p['protected'].items():assert sha(ROOT/n)==h,n
    for n,h in p['hashes'].items():
        if n=='checkpoint':path=model
        elif n=='lean_prediction':path=ROOT/'.artifact_runtime/realmlp_factorial/daf3c37e9d0734bbfbb5/lean_reference.npz'
        elif n.startswith('reference/'):path=ref/(n.split('/')[1]+'.npz')
        else:path=ROOT/n
        assert sha(path)==h,n
    assert sha(td/'trainer_snapshot.py')==p['hashes']['experiment_tabicl_probe.py']
    base=np.load(ref/'baseline.npz')['prediction'].astype(float)
    lean=np.load(ROOT/'.artifact_runtime/realmlp_factorial/daf3c37e9d0734bbfbb5/lean_reference.npz')['prediction']
    current=.8*base+.2*lean
    for row in summary['rows']:
        assert row['status']=='screen_complete'
        z=np.load(td/(str(row['context'])+'.npz'));np.testing.assert_array_equal(z['index'],idx['scored'])
        q=z['prediction'];assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        assert abs(roc_auc_score(y[idx['scored']],q)-row['auc'])<1e-12
        assert abs(paired(y[idx['scored']],.8*current+.2*q,current)['delta']-row['paired']['delta'])<1e-12
    result['tabicl']=dict(verified=True,contexts=len(summary['rows']),scores_verified=True)
    write_json(td/'verification.json',result['tabicl'])
    nd=Path(read(ROOT/'.artifact_runtime/source_neighbors_20261019/latest.json')['directory'])
    p=read(nd/'protocol.json');follow=Path(p['follow'])
    for n,h in p['hashes'].items():
        path=follow/'protocol.json' if n=='follow_protocol' else follow/'frozen_choice.json' if n=='follow_choice' else ROOT/n
        assert sha(path)==h,n
    for n,h in p['protected'].items():assert sha(ROOT/n)==h,n
    assert sha(nd/'trainer_snapshot.py')==p['hashes']['experiment_source_neighbors.py']
    assert sha(nd/'features.npz')==read(nd/'features.json')['sha256']
    with np.load(nd/'features.npz') as z:
        assert z['train'].shape==(len(y),10) and z['test'].shape==(299844,10)
        assert np.isfinite(z['train']).all() and np.isfinite(z['test']).all()
    ns=read(nd/'screen/summary.json');idx=np.load(nd/'screen/indices.npz')['scored']
    np.testing.assert_array_equal(idx,np.load(follow/'screen/indices.npz')['scored'])
    def get(folder,name):
        with np.load(folder/(name+'.npz')) as z:
            np.testing.assert_array_equal(idx,z['index']);return z['prediction'].astype(float)
    base=get(follow/'screen','baseline');control=get(follow/'screen','control');lean=get(follow/'screen','lean_mean')
    candidate=get(nd/'screen','prediction');choice=read(follow/'decision.json')['chosen_fusion'];w=choice['weight']
    def combine(b):
        if choice['form']=='probability':return (1-w)*b+w*lean
        return expit((1-w)*logit(np.clip(b,1e-6,1-1e-6))+w*logit(np.clip(lean,1e-6,1-1e-6)))
    anchor=combine(base);new=combine(base+.28*(candidate-control))
    assert abs(roc_auc_score(y[idx],candidate)-ns['auc'])<1e-12
    assert abs(paired(y[idx],new,anchor)['delta']-ns['paired']['delta'])<1e-12
    assert read(nd/'decision.json')['status']=='stopped_screen' and ns['paired']['delta']<p['screen_gate']
    result['neighbors']=dict(verified=True,scores_verified=True,protected_unchanged=True)
    write_json(nd/'verification.json',result['neighbors'])
    print(result)

if __name__=='__main__':main()
