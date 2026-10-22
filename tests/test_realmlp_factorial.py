"""Leakage and unseen-value contracts; run directly with the project Python."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from experiment_realmlp_factorial import LeanFeatures, ROOT

class FeatureContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=pd.read_csv(ROOT/'train.csv',nrows=1200)
        cls.y=raw.pop('satisfaction').to_numpy()
        cls.x=raw.drop(columns='id')

    def test_held_data_cannot_change_maps(self):
        t=LeanFeatures();t.fit_transform(self.x.iloc[:1000],self.y[:1000])
        before=t.transform(self.x.iloc[:20])
        counts={c:s.copy() for c,s in t.counts.items()}
        held=self.x.iloc[1000:].copy();held['Flight Distance']=999999
        after=t.transform(held)
        self.assertTrue((after['Flight Distance__category']=='__UNKNOWN__').all())
        self.assertTrue(np.isfinite(after.select_dtypes(include='number')).all().all())
        for c,s in counts.items():pd.testing.assert_series_equal(t.counts[c],s)
        pd.testing.assert_frame_equal(before,t.transform(self.x.iloc[:20]))

    def test_count_fit_only_and_canonical_numeric(self):
        t=LeanFeatures();a=t.fit_transform(self.x.iloc[:1000],self.y[:1000])
        for c in ['Class','Leg room service']:
            expect=LeanFeatures.token(self.x.iloc[:1000][c]).map(t.counts[c]).astype('float32')
            np.testing.assert_array_equal(a[c+'__count'],expect)
        s=pd.Series([1,2,3])
        pd.testing.assert_series_equal(LeanFeatures.token(s),LeanFeatures.token(s.astype(float)))
        self.assertEqual(a.shape[1],47)

    def test_reject_target(self):
        x=self.x.copy();x['satisfaction']=self.y
        with self.assertRaises(AssertionError):LeanFeatures().fit_transform(x,self.y)

if __name__=='__main__':unittest.main()
