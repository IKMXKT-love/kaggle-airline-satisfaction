import unittest
import numpy as np
import pandas as pd
from v3_features import RouteFeatures,keys


class SourceFeatureTests(unittest.TestCase):
    def test_profiles_use_only_fit_rows_and_unseen_routes_fall_back(self):
        fit=pd.DataFrame({'Flight Distance':[100.,100.,200.], 'Age':[20.,40.,60.],
            'Departure Delay in Minutes':[0.,2.,4.], 'Arrival Delay in Minutes':[0.,2.,np.nan],
            'Gender':['F','M','F'],'Customer Type':['A','A','B'],
            'Type of Travel':['T','B','B'],'Class':['E','E','B']})
        valid=fit.iloc[[0,1]].copy();valid['Flight Distance']=[100.,999.];valid['Age']=[10000.,-10000.]
        profile=RouteFeatures().fit(fit).transform(valid)
        self.assertEqual(profile['route_mean__Age'].iloc[0],30.)
        self.assertEqual(profile['route_mean__Age'].iloc[1],40.)
        self.assertEqual(profile['route_mean__route_count'].iloc[1],0.)
        self.assertNotIn('satisfaction',profile)
        self.assertEqual(keys(fit).shape,(3,6))


if __name__=='__main__':unittest.main()
