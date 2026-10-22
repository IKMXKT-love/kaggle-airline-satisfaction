import numpy as np
import pandas as pd
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scipy.special import expit
from xgboost import XGBClassifier
from transfer_features import TransferFeatures, key


def sample():
    return pd.DataFrame({'Flight Distance':[10.,10.,20.,20.], 'Age':[20.,20.,30.,40.],
        'Departure Delay in Minutes':[0.,0.,1.,2.], 'Arrival Delay in Minutes':[0.,1.,1.,2.],
        'Class':['Business','Eco','Eco','Eco Plus'], 'Customer Type':['Loyal']*4,
        'Type of Travel':['Business']*4,'Seat comfort':[1.,5.,3.,4.],
        'Inflight entertainment':[0.,4.,3.,5.]})


def test_fixed_counts_and_normalized_frequency_formula():
    train=sample();source=train.iloc[:2].copy()
    f=TransferFeatures().fit(train,source)
    out=f.transform(train,'frequency')
    assert out.shape==(4,24)
    # Three vocabulary buckets: 10,20,and one unseen bucket. Smoothing is on frequencies.
    expected=np.log(((2+1)/(4+3))/((2+1)/(2+3)))
    np.testing.assert_allclose(out.transfer_freq_0_logratio.iloc[0],expected,rtol=1e-6)


def test_unseen_groups_shrink_to_source_population_and_remain_finite():
    train=sample();f=TransferFeatures().fit(train,train)
    unseen=train.iloc[:1].copy();unseen['Flight Distance']=999.
    out=f.transform(unseen,'profiles')
    assert out.shape==(1,30) and np.isfinite(out.to_numpy()).all()
    assert out['source_fd__support'].iloc[0]==0
    np.testing.assert_allclose(out['source_fd__Seat comfort__sd'],train['Seat comfort'].std(ddof=0))
    assert np.isclose(sum(out['route_class_share__'+v].iloc[0] for v in ['Business','Eco','Eco Plus']),1)


def test_fit_state_does_not_change_on_transform_and_no_outer_information():
    train=sample();f=TransferFeatures().fit(train,train)
    before=f.transform(train,'combined')
    extra=pd.concat([train]*10,ignore_index=True);extra['Seat comfort']=0
    f.transform(extra,'combined')
    pd.testing.assert_frame_equal(before,f.transform(train,'combined'))
    assert before.shape[1]==54


def test_keys_canonicalize_numeric_types_and_reject_label_input():
    a=sample();b=a.copy();b['Flight Distance']=b['Flight Distance'].astype(int)
    pd.testing.assert_series_equal(key(a,['Flight Distance']),key(b,['Flight Distance']))
    with unittest.TestCase().assertRaises(AssertionError):
        TransferFeatures().fit(a.assign(satisfaction=1),a)


def test_xgb_margin_is_logit_and_must_be_provided_at_prediction():
    x=pd.DataFrame({'x':np.arange(20,dtype=float)})
    y=np.array([0,1]*10);margin=np.linspace(-2,2,20).astype('float32')
    model=XGBClassifier(n_estimators=1,learning_rate=0,max_depth=1,device='cpu',base_score=.5)
    model.fit(x,y,base_margin=margin)
    p=model.predict_proba(x,base_margin=margin)[:,1]
    np.testing.assert_allclose(p,expit(margin),atol=1e-7)
    assert not np.allclose(p,model.predict_proba(x)[:,1])


if __name__=='__main__':
    suite=unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in list(globals().items()) if k.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
