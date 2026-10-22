import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from optimized_features import normalize_raw
from optimized_training import fit_component, train_optimized, validation_fingerprint


class OptimizedTrainingTests(unittest.TestCase):
    def test_validation_fingerprint_detects_changed_labels(self):
        train = pd.DataFrame({"x": [1, 2], "satisfaction": [False, True]})
        test = pd.DataFrame({"x": [3]})
        before = validation_fingerprint(train, test, ["original"])
        changed = train.copy()
        changed.loc[0, "satisfaction"] = True
        self.assertNotEqual(before, validation_fingerprint(changed, test, ["original"]))
        self.assertNotEqual(before, validation_fingerprint(train, test, ["different-original"]))

    def test_target_in_test_or_mismatched_features_are_rejected(self):
        train = pd.DataFrame({"id": np.arange(20), "x": np.arange(20),
                              "satisfaction": (np.arange(20) % 2).astype(bool)})
        for test in [train.iloc[:2].copy(), pd.DataFrame({"id": [30], "other": [1]})]:
            with self.subTest(columns=test.columns.tolist()), self.assertRaises(ValueError):
                train_optimized(train, test)

    def test_fixed_round_predictions_do_not_depend_on_audit_labels(self):
        x = normalize_raw(pd.DataFrame({"x": np.arange(100) * .123, "kind": ["a", "b"] * 50}))
        y = np.arange(100) % 2
        config = {"name": "test_xgb", "family": "xgboost", "original": False,
                  "use_te": True, "depth": 2, "child": 1}
        predictions = []
        for valid_y in [y[80:], 1-y[80:]]:
            pv, pt, details = fit_component(x.iloc[:80], y[:80], x.iloc[80:], valid_y,
                x.iloc[80:82], config, None, np.zeros((80, 3)), np.zeros((20, 3)),
                np.zeros((2, 3)), "cpu", 2, max_rounds=8, early_stopping=False)
            predictions.append(pv)
            self.assertTrue(np.isfinite(pt).all())
            self.assertEqual(details["best_rounds"], 8)
        np.testing.assert_array_equal(predictions[0], predictions[1])

    def test_weighted_ten_fold_predictions_and_checkpoint_resume(self):
        root = Path(__file__).resolve().parents[1]
        runtime = root / ".artifact_runtime"
        runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime, prefix="test_optimized_") as folder:
            folder = Path(folder)
            recipe = folder / "recipe.json"
            recipe.write_text(json.dumps({"components": [
                {"name": "a", "weight": 1., "family": "xgboost"},
                {"name": "b", "weight": 3., "family": "catboost"}]}), encoding="utf-8")
            train = pd.DataFrame({"id": np.arange(100), "x": np.arange(100) * .123,
                                  "satisfaction": (np.arange(100) % 2).astype(bool)})
            test = pd.DataFrame({"id": [200, 201], "x": [.1, .2]})
            raw = normalize_raw(train[["x"]]); xt = normalize_raw(test[["x"]])
            bundle = (raw, xt, raw.iloc[:40], train.satisfaction.iloc[:40].to_numpy(dtype=int),
                      np.zeros((100, 3)), np.zeros((2, 3)), {"signature": {"test": True}})

            def fake_fit(fit, labels, valid, valid_y, prediction, config, *args, **kwargs):
                probability = .2 if config["name"] == "a" else .8
                return np.full(len(valid), probability), np.full(len(prediction), probability), {
                    "name": config["name"], "family": config["family"], "best_rounds": 1,
                    "feature_count": 1, "feature_columns": ["x"], "auc": .5}

            with patch("optimized_training.BASE_DIR", folder), \
                    patch("optimized_training.file_hash", return_value="test-code"), \
                    patch("optimized_training.load_original_bundle", return_value=bundle), \
                    patch("optimized_training.fit_component", side_effect=fake_fit) as trainer, \
                    contextlib.redirect_stdout(io.StringIO()):
                prediction, oof, folds, report = train_optimized(train, test, recipe_path=recipe)
                self.assertEqual(trainer.call_count, 20)
                prediction_again, _, _, _ = train_optimized(train, test, recipe_path=recipe)
                self.assertEqual(trainer.call_count, 20)
            np.testing.assert_allclose(prediction, .65)
            np.testing.assert_allclose(oof, .65)
            np.testing.assert_array_equal(prediction, prediction_again)
            self.assertEqual(set(folds), set(range(1, 11)))
            self.assertEqual(len(report["component_results"]), 20)


if __name__ == "__main__":
    unittest.main()
