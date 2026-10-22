import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from feature_engineering import FoldFeatureEngineer, row_features, synthesize_training_rows
from main import train_predict
from sklearn.model_selection import StratifiedKFold


class FeatureEngineeringTests(unittest.TestCase):
    def test_rating_zero_is_distinct_from_low_score(self):
        frame = pd.DataFrame({"Seat comfort": [0, 1, 5], "Cleanliness": [5, 2, 4]})
        result = row_features(frame)
        self.assertEqual(result.loc[0, "fe_service_zero_count"], 1)
        self.assertEqual(result.loc[0, "fe_service_low_count"], 0)
        self.assertEqual(result.loc[1, "fe_service_low_count"], 2)
        self.assertEqual(result.loc[2, "fe_service_high_count"], 2)

    def test_frequency_uses_only_real_training_fold(self):
        training = pd.DataFrame({"Class": ["Eco", "Eco", "Business"]})
        validation = pd.DataFrame({"Class": ["Eco", "Unseen", "Business"]})
        engineer = FoldFeatureEngineer().fit(training)
        result = engineer.transform(validation)
        np.testing.assert_allclose(result["fe_frequency_Class"], [2 / 3, 0, 1 / 3])
        self.assertEqual(set(engineer.frequencies_["Class"].index), {"Eco", "Business"})

    def test_derived_features_recomputed_from_synthetic_rows(self):
        frame = pd.DataFrame({"Departure Delay in Minutes": [10.0],
                              "Arrival Delay in Minutes": [25.0], "Flight Distance": [500.0]})
        result = row_features(frame)
        self.assertEqual(result.loc[0, "fe_delay_change"], 15)
        self.assertEqual(result.loc[0, "fe_delay_total"], 35)
        self.assertEqual(result.loc[0, "fe_delay_per_1000_distance"], 70)

    def test_synthetic_parents_and_discrete_values(self):
        n = 120
        frame = pd.DataFrame({"Age": np.arange(n) % 30 + 20.0,
                              "Flight Distance": np.arange(n) * 3 + 501.0,
                              "Seat comfort": np.arange(n) % 6,
                              "Class": np.where(np.arange(n) % 3, "Eco", "Business")})
        labels = np.arange(n) % 2
        synthetic, y, parents = synthesize_training_rows(frame, labels, "classification", .5, 42)
        self.assertEqual(len(synthetic), 60)
        self.assertTrue((parents[:, 0] != parents[:, 1]).all())
        self.assertGreaterEqual(parents.min(), 0)
        self.assertLess(parents.max(), n)
        np.testing.assert_array_equal(y, labels[parents[:, 0]])
        np.testing.assert_array_equal(y, labels[parents[:, 1]])
        np.testing.assert_array_equal(synthetic["Class"], frame.iloc[parents[:, 0]]["Class"])
        np.testing.assert_array_equal(synthetic["Class"], frame.iloc[parents[:, 1]]["Class"])
        np.testing.assert_array_equal(synthetic["Seat comfort"], frame.iloc[parents[:, 0]]["Seat comfort"])
        for col in ["Age", "Flight Distance"]:
            a = frame.iloc[parents[:, 0]][col].to_numpy()
            b = frame.iloc[parents[:, 1]][col].to_numpy()
            values = synthetic[col].to_numpy()
            self.assertTrue(((values >= np.minimum(a, b)) & (values <= np.maximum(a, b))).all())
            self.assertTrue((values == np.round(values)).all())
        again, again_y, again_parents = synthesize_training_rows(frame, labels, "classification", .5, 42)
        pd.testing.assert_frame_equal(synthetic, again)
        np.testing.assert_array_equal(parents, again_parents)
        np.testing.assert_array_equal(y, again_y)

    def test_regression_label_uses_same_interpolation(self):
        x = np.arange(50) * .123
        frame = pd.DataFrame({"x": x})
        synthetic, labels, _ = synthesize_training_rows(frame, 2 * x + 5, "regression", .4, 9)
        np.testing.assert_allclose(labels, 2 * synthetic["x"] + 5, rtol=1e-6)

    def test_disabled_and_no_eligible_pairs(self):
        frame = pd.DataFrame({"x": np.arange(20, dtype=float)})
        for ratio, classes in [(0, np.zeros(20)), (.5, np.arange(20))]:
            synthetic, labels, parents = synthesize_training_rows(frame, classes, "classification", ratio, 42)
            self.assertEqual(len(synthetic), 0)
            self.assertEqual(len(labels), 0)
            self.assertEqual(parents.shape, (0, 2))

    def test_invalid_ratio(self):
        with self.assertRaises(ValueError):
            synthesize_training_rows(pd.DataFrame({"x": [1]}), [0], "classification", np.nan, 42)

    def test_ten_fold_integration_for_both_tasks(self):
        rng = np.random.default_rng(42)
        for task in ["classification", "regression"]:
            with self.subTest(task=task):
                x = rng.normal(size=120)
                labels = (x > 0).astype(int) if task == "classification" else 2 * x + rng.normal(size=120) * .1
                frame = pd.DataFrame({"id": np.arange(120), "x": x,
                                      "kind": np.where(np.arange(120) % 2, "a", "b"), "target": labels})
                frame.loc[0, "x"] = np.nan
                test = pd.DataFrame({"id": [1000, 1001], "x": [.1, np.nan], "kind": ["unseen", "a"]})
                with contextlib.redirect_stdout(io.StringIO()):
                    prediction, oof, folds, report = train_predict(
                        frame, test, "target", "id", task, 12, synthetic_ratio=.2,
                    )
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue(np.isfinite(oof).all())
                self.assertEqual(set(folds), set(range(1, 11)))
                self.assertEqual(len(report["fold_scores"]), 10)
                self.assertNotIn("id", report["feature_columns"])
                self.assertNotIn("target", report["feature_columns"])
                self.assertTrue(all(item["synthetic_rows"] == 21 for item in report["augmentation_by_fold"]))
                if task == "classification":
                    self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())

    def test_actual_fold_boundaries_and_baseline_switch(self):
        frame = pd.DataFrame({"id": np.arange(100), "x": np.arange(100) * .123,
                              "target": np.arange(100) % 2})
        test = pd.DataFrame({"id": [1000], "x": [.3]})
        expected = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(frame, frame.target))
        calls = []

        def audited_generator(raw, labels, *args):
            training, validation = expected[len(calls)]
            np.testing.assert_array_equal(raw.index, training)
            self.assertTrue(set(raw.index).isdisjoint(validation))
            self.assertNotIn("id", raw)
            self.assertNotIn("target", raw)
            np.testing.assert_array_equal(labels, frame.iloc[training].target)
            calls.append(len(raw))
            return synthesize_training_rows(raw, labels, *args)

        with patch("main.synthesize_training_rows", side_effect=audited_generator):
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, _, report = train_predict(frame, test, "target", "id", "classification", 5,
                                                 feature_engineering=False, synthetic_ratio=0)
        self.assertEqual(len(calls), 10)
        self.assertEqual(report["feature_columns"], ["x"])
        self.assertTrue(all(item["synthetic_rows"] == 0 for item in report["augmentation_by_fold"]))


if __name__ == "__main__":
    unittest.main()
