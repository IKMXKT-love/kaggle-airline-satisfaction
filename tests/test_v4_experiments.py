import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from optimize_v4 import transform, train, combine


class V4ExperimentTests(unittest.TestCase):
    def test_ablation_preserves_rows_raw_categories_and_missing_values(self):
        a = pd.DataFrame({'Age': [np.nan, 25.], 'Class': pd.Categorical(['E', 'B']),
                          'route_mean__Age': [40., 40.], 'te__Age': [.2, .8]}, index=[8, 3])
        original = a.copy(deep=True)
        out, valid, test = transform((a, a.iloc[[1]], None), 'without_route')
        self.assertEqual(list(out.index), [8, 3])
        self.assertEqual(list(valid.index), [3])
        self.assertIsNone(test)
        self.assertNotIn('route_mean__Age', out)
        self.assertTrue(pd.isna(out.loc[8, 'Age']))
        self.assertIsInstance(out.Class.dtype, pd.CategoricalDtype)
        pd.testing.assert_frame_equal(a, original)

    def test_residual_uses_each_matching_rating_without_modifying_input(self):
        a = pd.DataFrame({'wifi': [1., 5.], 'boarding': [4., 2.],
            'aux_expected_rating_0': [2., 3.], 'aux_expected_rating_1': [3., 3.]})
        with patch('optimize_v4.read', return_value={'ratings': ['wifi', 'boarding']}):
            out, _, _ = transform((a, a, None), 'residual')
        np.testing.assert_array_equal(out.rating_residual_0, [-1., 2.])
        np.testing.assert_array_equal(out.rating_residual_1, [1., -1.])
        self.assertNotIn('rating_residual_0', a)

    def test_training_rejects_label_or_id_before_fitting(self):
        for forbidden in ['id', 'satisfaction']:
            a = pd.DataFrame({'Age': [20., 40.], forbidden: [0, 1]})
            with self.assertRaises(AssertionError):
                train(dict(transform='full', family='xgb'), a, np.array([0, 1]), a, np.array([0, 1]))

    def test_xgb_replacement_retains_other_models(self):
        old_xgb = np.array([.1, .7]); other = np.array([.8, .2]); new_xgb = np.array([.5, .9])
        base = .4*old_xgb + .6*other
        result = combine(base, new_xgb, dict(mode='replace_xgb', weight=.4), old_xgb)
        np.testing.assert_allclose(result, .4*new_xgb + .6*other)


if __name__ == '__main__':
    unittest.main()
