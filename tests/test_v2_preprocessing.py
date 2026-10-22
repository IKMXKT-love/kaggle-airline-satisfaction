import unittest
import numpy as np
import pandas as pd

from v2_tabm import prepare_frames


class V2PreprocessingTests(unittest.TestCase):
    def test_validation_values_cannot_change_fitted_preprocessing(self):
        a=pd.DataFrame({'value':np.arange(40,dtype=float),
            'category':pd.Categorical(['a','b']*20),'constant':np.ones(40)})
        b=pd.DataFrame({'value':[np.nan,3.],
            'category':pd.Categorical(['new','a']),'constant':[4.,5.]})
        changed=b.copy();changed['value']=[1e10,-1e10]
        n,c,cards,prep=prepare_frames(a,b)
        n2,c2,cards2,prep2=prepare_frames(a,changed)
        np.testing.assert_array_equal(n[0],n2[0])
        np.testing.assert_array_equal(c[0],c2[0])
        np.testing.assert_array_equal(prep['imputer'].statistics_,prep2['imputer'].statistics_)
        self.assertNotIn('constant',prep['numeric'])
        self.assertEqual(cards,cards2)
        self.assertEqual(c[1][0,0],0)
        self.assertTrue(np.isfinite(n[1]).all())


if __name__=='__main__':unittest.main()
