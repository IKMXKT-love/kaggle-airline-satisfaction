import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

root = Path(__file__).resolve().parents[1]
expected = {
    "train.csv": "2313B65C74E6994F01C2628AAE82D2F97AA6F74A81950172FF83864837C98B36",
    "test.csv": "FF7932747F27C5274DE904194AC38B4D088EA4F9BB005AE74F0D619FB893A9B6",
    "sample_submission.csv": "6165ECC5769962253534C2D840F26AAB55DC3CC091D256BB3641BF6608194A71",
    "submission.csv": "C2DBFD066C710A11D2C94B46D64271A05C16859AF97C3890B0FD7A07B23EB920",
    "submission_oof.csv": "826F7642EAD919509F87221597BDCBD476C7253DF42E3823A8AF057FC5EBAB9A",
    "submission_metrics.json": "DB2028FFBC3EFF51FDF33A518DD3E4C2CEDA40645530F744BFFCC83D839F8E23",
}
for name, digest in expected.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest().upper() == digest, name

train = pd.read_csv(root / "train.csv")
test = pd.read_csv(root / "test.csv")
sample = pd.read_csv(root / "sample_submission.csv")
submission = pd.read_csv(root / "submission_enhanced.csv")
oof = pd.read_csv(root / "submission_enhanced_oof.csv")
baseline_oof = pd.read_csv(root / "submission_oof.csv")
baseline = json.loads((root / "submission_metrics.json").read_text(encoding="utf-8"))
enhanced = json.loads((root / "submission_enhanced_metrics.json").read_text(encoding="utf-8"))
assert submission.shape == sample.shape == (299844, 2)
assert list(submission.columns) == list(sample.columns)
assert submission.id.equals(sample.id) and submission.id.equals(test.id)
assert submission.id.is_unique and submission.satisfaction.between(0, 1).all()
assert np.isfinite(submission.satisfaction).all()
assert len(oof) == len(train) == 699635
assert oof.id.equals(train.id) and oof.target.equals(train.satisfaction)
assert oof.fold.equals(baseline_oof.fold)
assert oof.prediction.between(0, 1).all()
assert set(oof.fold) == set(range(1, 11))
assert enhanced["n_splits"] == 10 and len(enhanced["fold_scores"]) == 10
assert np.isclose(roc_auc_score(train.satisfaction, oof.prediction), enhanced["oof_score"])
for details in enhanced["augmentation_by_fold"]:
    assert details["validation_rows"] == (oof.fold == details["fold"]).sum()
    assert details["real_training_rows"] + details["validation_rows"] == len(train)
    assert details["synthetic_rows"] == int(details["real_training_rows"] * enhanced["synthetic_ratio"])
assert "id" not in enhanced["feature_columns"] and "satisfaction" not in enhanced["feature_columns"]

preview = pd.read_csv(root / "synthetic_preview.csv")
indexed = train.set_index("id")
left = indexed.loc[preview.source_train_id].reset_index(drop=True)
right = indexed.loc[preview.source_donor_id].reset_index(drop=True)
assert np.array_equal(preview.satisfaction, left.satisfaction)
assert np.array_equal(preview.satisfaction, right.satisfaction)
fold_by_id = oof.set_index("id").fold
assert (fold_by_id.loc[preview.source_train_id] != 1).all()
assert (fold_by_id.loc[preview.source_donor_id] != 1).all()
for col in ["Gender", "Customer Type", "Class", "Type of Travel"]:
    assert np.array_equal(preview[col], left[col])
    assert np.array_equal(preview[col], right[col])
for col in ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
    observed = left[col].notna() & right[col].notna()
    minimum = np.minimum(left.loc[observed, col], right.loc[observed, col])
    maximum = np.maximum(left.loc[observed, col], right.loc[observed, col])
    assert ((preview.loc[observed, col] >= minimum) & (preview.loc[observed, col] <= maximum)).all()

comparison = {
    "baseline_mean_auc": baseline["mean_score"],
    "enhanced_mean_auc": enhanced["mean_score"],
    "mean_auc_change": enhanced["mean_score"] - baseline["mean_score"],
    "baseline_oof_auc": baseline["oof_score"],
    "enhanced_oof_auc": enhanced["oof_score"],
    "same_validation_folds_verified": True,
    "baseline_training_device": baseline.get("device", "not_recorded"),
    "enhanced_training_device": enhanced["device"],
    "comparison_note": "combined_change_comparison_not_a_controlled_feature_vs_synthetic_ablation",
    "enhancement_improved_mean_auc": enhanced["mean_score"] > baseline["mean_score"],
    "recommended_submission": "submission_enhanced.csv" if enhanced["mean_score"] > baseline["mean_score"] else "submission.csv",
    "checks": {"original_data_and_baseline_hashes_unchanged": True,
               "submission_rows_columns_ids_probabilities_valid": True,
               "oof_each_real_row_once_and_ten_folds": True,
               "synthetic_preview_parents_exclude_validation_fold": True,
               "synthetic_preview_labels_and_numeric_bounds_valid": True},
}
(root / "enhancement_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
print(json.dumps(comparison, indent=2))
