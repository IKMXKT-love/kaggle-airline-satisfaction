import unittest

import numpy as np
import pandas as pd

from optimized_features import CompetitionFeaturePipeline, NativeCategoryEncoder, OriginalStatistics, normalize_raw


class OptimizedFeatureTests(unittest.TestCase):
    def test_zero_is_not_missing_and_unknown_category_is_missing(self):
        train = normalize_raw(pd.DataFrame({"value": [0., np.nan, 2.], "kind": ["a", "b", "a"]}))
        encoder = NativeCategoryEncoder().fit(train)
        result = encoder.transform(train)
        self.assertEqual(result.loc[0, "value"], 0)
        self.assertTrue(pd.isna(result.loc[1, "value"]))
        unknown = encoder.transform(normalize_raw(pd.DataFrame({"value": [0.], "kind": ["new"]})))
        self.assertTrue(pd.isna(unknown.loc[0, "kind"]))

    def test_target_encoding_training_rows_are_cross_fitted(self):
        # 每个取值唯一，正确交叉拟合时不可能从本行取值查询到本行标签。
        frame = pd.DataFrame({"key": [f"unique_{i}" for i in range(100)]})
        y = np.arange(100) % 2
        pipeline = CompetitionFeaturePipeline(use_te=True)
        result = pipeline.fit_transform(frame, y)
        np.testing.assert_allclose(result["cv_te__key"], .5)
        self.assertFalse(np.array_equal(result["cv_te__key"], y))
        unseen = pipeline.transform(pd.DataFrame({"key": ["unseen"]}))
        self.assertAlmostEqual(float(unseen.loc[0, "cv_te__key"]), .5)

    def test_original_statistics_use_external_labels_only(self):
        external = normalize_raw(pd.DataFrame({"Class": ["Eco"] * 20 + ["Business"] * 20}))
        labels = np.r_[np.zeros(20), np.ones(20)]
        statistics = OriginalStatistics().fit(external, labels)
        frame = pd.DataFrame({"Class": ["Eco", "Business", "unknown"]})
        result = statistics.transform(frame)
        np.testing.assert_allclose(result["orig_mean__Class"], [1 / 6, 5 / 6, .5])
        np.testing.assert_allclose(result["orig_count__Class"], [np.log1p(20), np.log1p(20), 0])

    def test_combined_pipeline_preserves_index_and_column_order(self):
        frame = pd.DataFrame({"Age": np.arange(80) + 10., "Class": ["Eco", "Business"] * 40},
                             index=np.arange(80) + 1000)
        labels = np.arange(80) % 2
        external = OriginalStatistics().fit(frame, labels)
        pipeline = CompetitionFeaturePipeline(use_te=True, original_statistics=external)
        fit = pipeline.fit_transform(frame, labels, np.full(80, .4))
        test = pipeline.transform(frame.iloc[:3], np.full(3, .4))
        self.assertEqual(list(fit.columns), list(test.columns))
        self.assertEqual(list(fit.index), list(frame.index))
        self.assertTrue((fit.orig_model_probability == np.float32(.4)).all())
        with self.assertRaises(ValueError):
            pipeline.transform(frame.iloc[:3], np.full(2, .4))


if __name__ == "__main__":
    unittest.main()
