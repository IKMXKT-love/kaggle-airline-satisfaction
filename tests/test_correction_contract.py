"""Check label isolation and CatBoost's binary baseline inference contract."""
import copy
import inspect
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from correction_support import (nested_plan, validate_plan, CorrectionFeatures,
    offset_pool_prediction, fit_branch, combined, BRANCHES, score)


class Contract(unittest.TestCase):
    def test_nested_isolation_and_detected_contamination(self):
        y = np.tile([0, 1], 500)
        p = nested_plan(y, 0)
        validate_plan(p, len(y))
        q = copy.deepcopy(p)
        q['selection'][0]['fit'][0] = q['outer_eval'][0]
        with self.assertRaises(AssertionError):
            validate_plan(q, len(y))
        self.assertNotIn('query_labels', inspect.signature(fit_branch).parameters)

    def test_features_fit_only_unknowns_missing_and_exclusion(self):
        x = pd.DataFrame(dict(Age=[22, 23, 24, 25], Gender=['a', 'b', 'a', 'b'],
                              Rating=[0., 1., 2., np.nan]))
        tf = CorrectionFeatures()
        tf.fit_transform(x.iloc[:3], [0., 1., 2.])
        saved = copy.deepcopy(tf.vocab)
        q = pd.DataFrame(dict(Age=[999], Gender=['heldout'], Rating=[np.nan]))
        a = tf.transform(q, [1.])
        self.assertEqual(a.Gender.iloc[0], '__UNKNOWN__')
        self.assertEqual(a.Age__category.iloc[0], '__UNKNOWN__')
        self.assertEqual(saved, tf.vocab)
        self.assertEqual(tf.transform(x.iloc[:1], [0.]).Rating__category.iloc[0], '0.0')
        for c in ['id', 'satisfaction']:
            with self.assertRaises(AssertionError):
                CorrectionFeatures().fit_transform(x.assign(**{c: 0}), np.zeros(4))

    def test_offset_is_added_exactly_once_and_zero_identity(self):
        from catboost import CatBoostClassifier, Pool
        rng = np.random.default_rng(21)
        x = pd.DataFrame(dict(a=rng.normal(size=200), c=np.tile(['a', 'b'], 100)))
        y = (x.a.to_numpy() + rng.normal(size=200) > 0).astype(int)
        base = .6 * x.a.to_numpy()
        model = CatBoostClassifier(iterations=8, depth=2, loss_function='Logloss',
            cat_features=['c'], allow_writing_files=False, verbose=False, thread_count=2)
        model.fit(Pool(x, label=y, cat_features=['c'], baseline=base))
        p = offset_pool_prediction(model, x, base, ['c'])
        np.testing.assert_allclose(p, expit(base + model.predict(x, prediction_type='RawFormulaVal')))
        self.assertGreater(np.max(abs(p - model.predict_proba(x)[:, 1])), .01)
        np.testing.assert_allclose(expit(score(expit(base))), expit(base))
        self.assertEqual(roc_auc_score(y, base), roc_auc_score(y, .8 * base + .3))

    def test_current_recipe_reconstruction(self):
        p = np.array([.1, .5, .8])
        np.testing.assert_allclose(combined({n: p for n in BRANCHES}), p)


if __name__ == '__main__':
    unittest.main()
