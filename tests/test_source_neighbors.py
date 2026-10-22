import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from experiment_source_neighbors import SourceNeighbors

class SourceNeighborContracts(unittest.TestCase):
    def setUp(self):
        self.source=pd.DataFrame({'Gender':['F']*160,'Customer Type':['Loyal']*160,
            'Type of Travel':['Business']*160,'Class':['A']*80+['B']*80,
            'Age':[30.]*160,'Flight Distance':[500.]*160,'Seat comfort':[3.]*160,
            'Departure Delay in Minutes':[0.]*160,'Arrival Delay in Minutes':[0.]*160})
        self.y=np.r_[np.zeros(80,dtype=int),np.ones(80,dtype=int)]
    def test_conditioning_and_exact_shrinkage(self):
        model=SourceNeighbors().fit(self.source,self.y)
        values=model.transform(self.source.iloc[[0,80]])
        np.testing.assert_allclose(values[:,0],[10/36,26/36],rtol=1e-6)
        np.testing.assert_allclose(values[:,1],[10/84,74/84],rtol=1e-6)
        np.testing.assert_array_equal(values[:,9],[0,0])
        self.assertEqual(values.shape,(2,10))
    def test_unseen_group_and_immutability(self):
        model=SourceNeighbors().fit(self.source,self.y)
        before=model.transform(self.source.iloc[:2]);med=model.median.copy()
        unknown=self.source.iloc[:1].copy();unknown['Class']='UNKNOWN';unknown['Age']=np.nan
        value=model.transform(unknown)
        self.assertTrue(np.isfinite(value).all());self.assertEqual(value[0,9],1)
        np.testing.assert_array_equal(model.median,med)
        np.testing.assert_array_equal(model.transform(self.source.iloc[:2]),before)
    def test_transform_rejects_competition_target(self):
        model=SourceNeighbors().fit(self.source,self.y)
        query=self.source.iloc[:1].copy();query['satisfaction']=1
        with self.assertRaises(AssertionError):model.transform(query)

if __name__=='__main__':unittest.main()
