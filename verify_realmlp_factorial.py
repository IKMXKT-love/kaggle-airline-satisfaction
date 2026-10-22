"""Independent cache reconstruction; never trains or changes predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiment_realmlp_factorial import ROOT,OUT,sha,read
from report_neural import paired

def main():
    d=Path(read(OUT/'latest.json')['directory']);protocol=read(d/'protocol.json')
    reference=Path(protocol['reference'])/'screen'
    for n,h in protocol['hashes'].items():
        path=reference/n.split('/',1)[1] if n.startswith('reference/') else ROOT/n
        assert sha(path)==h,n
    for n,h in protocol['protected'].items():assert sha(ROOT/n)==h,n
    assert sha(d/'trainer_snapshot.py')==protocol['hashes']['experiment_realmlp_factorial.py']
    idx=np.load(d/'indices.npz')['scored']
    y=pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy()[idx]
    def pred(folder,name):
        with np.load(folder/(name+'.npz')) as z:
            np.testing.assert_array_equal(idx,z['index'])
            p=z['prediction'].astype('float64')
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            return p
    base=sum(w*pred(reference,n) for n,w in [('control',.28),('old_xgb',.07),('old_cat',.21),('realmlp',.14),('lgb',.30)])
    np.testing.assert_allclose(base,pred(reference,'baseline'),atol=2e-7,rtol=0)
    # Use actual cached baseline to exactly reproduce stored float32 blend arithmetic.
    base=np.load(reference/'baseline.npz')['prediction']
    summary=read(d/'summary.json');p={};rows={}
    for row in summary['rows']:
        name=row['name'];p[name]=np.load(d/(name+'.npz'))['prediction']
        pred(d,name)
        assert abs(roc_auc_score(y,p[name])-row['auc'])<1e-12
        blend=.8*base+.2*p[name]
        assert abs(roc_auc_score(y,blend)-row['blend_auc'])<1e-12
        np.testing.assert_allclose(paired(y,blend,base)['delta'],row['paired']['delta'],atol=1e-12)
        rows[name]=row
    comparisons={}
    for a,b in [('lean_old','existing_old'),('existing_reference','existing_old'),
                ('lean_reference','existing_reference'),('lean_reference','lean_old')]:
        comparisons[a+' minus '+b]=dict(solo=paired(y,p[a],p[b]),
            blend=paired(y,.8*base+.2*p[a],.8*base+.2*p[b]))
    best=max(summary['rows'][1:],key=lambda r:r['paired']['delta'])
    assert summary['best']==best['name']
    assert (summary['status']=='ready_for_confirmation')==(best['paired']['delta']>=protocol['screen_gate'])
    result=dict(verified=True,rows=len(idx),hashes_verified=True,baseline_rebuilt=True,
        predictions_valid=True,score_recomputed=True,factorial_comparisons=comparisons)
    (d/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
