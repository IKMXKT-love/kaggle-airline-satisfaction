"""Reconstruct both follow-up stages independently from cached predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiment_realmlp_factorial import ROOT,read,sha
from report_neural import paired

def main():
    out=ROOT/'.artifact_runtime/followup_20261019'
    directory=Path(read(out/'latest.json')['directory']);protocol=read(directory/'protocol.json')
    refs=dict(reference=ROOT/'.artifact_runtime/transfer/74f731e0c4b9330ecfac/screen',
              factorial=ROOT/'.artifact_runtime/realmlp_factorial/daf3c37e9d0734bbfbb5')
    for n,h in protocol['hashes'].items():
        prefix=n.split('/')[0]
        path=refs[prefix]/n.split('/',1)[1] if prefix in refs else ROOT/n
        assert sha(path)==h,n
    for n,h in protocol['protected'].items():assert sha(ROOT/n)==h,n
    assert sha(directory/'trainer_snapshot.py')==protocol['hashes']['experiment_followup.py']
    y=pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy()
    results={}
    for stage in ['screen','confirmation']:
        folder=directory/stage;indices=np.load(folder/'indices.npz');idx=indices['scored'];ys=y[idx]
        assert not(set(indices['fit'])&set(idx) or set(indices['stopping'])&set(idx) or set(indices['fit'])&set(indices['stopping']))
        def get(name):
            with np.load(folder/(name+'.npz')) as z:
                np.testing.assert_array_equal(z['index'],idx);p=z['prediction'].astype(float)
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();return p
        base=get('baseline');rebuilt=sum(w*get(n) for n,w in protocol['weights'].items())
        np.testing.assert_allclose(base,rebuilt,atol=2e-7,rtol=0)
        one,two=get('lean_42'),get('lean_20261019');avg=(one+two)/2
        np.testing.assert_array_equal(avg,get('lean_mean'))
        current=.8*base+.2*one;s=read(folder/'summary.json')
        assert abs(roc_auc_score(ys,current)-s['current_auc'])<1e-12
        for r in s['trials']:
            w=r['weight']
            if r['form']=='probability':p=(1-w)*base+w*avg
            else:
                b,q=np.clip(base,1e-6,1-1e-6),np.clip(avg,1e-6,1-1e-6)
                odds=(b/(1-b))**(1-w)*(q/(1-q))**w;p=odds/(1+odds)
            assert abs(roc_auc_score(ys,p)-r['auc'])<1e-12
            assert abs(paired(ys,p,current)['delta']-r['vs_current']['delta'])<1e-12
            assert abs(paired(ys,p,base)['delta']-r['vs_base']['delta'])<1e-12
        results[stage]=dict(rows=len(idx),trials=len(s['trials']),auc_verified=True)
    choice=read(directory/'frozen_choice.json')
    confirm=read(directory/'confirmation/summary.json')
    assert confirm['best']['weight']==choice['weight'] and confirm['best']['form']==choice['form']
    screen=read(directory/'screen/summary.json');decision=read(directory/'decision.json')
    stable=(all(confirm['seeds'][str(seed)]['fixed20_vs_base']['delta']>0 for seed in protocol['seeds'])
            and confirm['seeds']['42']['fixed20_vs_base']['delta']>=protocol['confirmation_min_vs_base'])
    improves=(screen['best']['vs_current']['delta']>=protocol['incremental_min_vs_current']
              and confirm['best']['vs_current']['delta']>=protocol['incremental_min_vs_current']
              and confirm['best']['vs_base']['delta']>=protocol['confirmation_min_vs_base'])
    assert decision['current_recipe_stable']==stable
    assert decision['frozen_fusion_improves_current']==improves
    result=dict(verified=True,hashes_verified=True,protected_unchanged=True,stages=results,frozen_choice_verified=True)
    (directory/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
