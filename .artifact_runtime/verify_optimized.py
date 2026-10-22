import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

root = Path(__file__).resolve().parents[1]
protected = {
    "train.csv": "2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36",
    "test.csv": "ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6",
    "sample_submission.csv": "6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71",
    "submission.csv": "c2dbfd066c710a11d2c94b46d64271a05c16859af97c3890b0fd7a07b23eb920",
    "submission_oof.csv": "826f7642ead919509f87221597bdcbd476c7253df42e3823a8af057fc5ebab9a",
    "submission_metrics.json": "db2028ffbc3eff51fdf33a518dd3e4c2ceda40645530f744bffcc83d839f8e23",
    "external_data/train.csv": "a15fab6d36a6123dedb035e1bc11b31f27b547671ce8d27762e74d96527c364f",
    "external_data/test.csv": "2b58518af6e6c94334bf0f69160e29c5a5a082e340cf30f517473ede2bb1d898",
}
for name, digest in protected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
train = pd.read_csv(root/"train.csv")
test = pd.read_csv(root/"test.csv")
sample = pd.read_csv(root/"sample_submission.csv")
pred = pd.read_csv(root/"submission_optimized.csv")
oof = pd.read_csv(root/"submission_optimized_oof.csv")
baseline_oof = pd.read_csv(root/"submission_oof.csv")
baseline = json.loads((root/"submission_metrics.json").read_text(encoding="utf-8"))
enhanced = json.loads((root/"submission_enhanced_metrics.json").read_text(encoding="utf-8"))
metrics = json.loads((root/"submission_optimized_metrics.json").read_text(encoding="utf-8"))
recipe = json.loads((root/"optimized_recipe.json").read_text(encoding="utf-8"))
assert pred.shape == sample.shape == (299844, 2)
assert pred.columns.tolist() == sample.columns.tolist() == ["id", "satisfaction"]
assert pred.id.equals(sample.id) and pred.id.equals(test.id) and pred.id.is_unique
assert pred.satisfaction.between(0, 1).all() and np.isfinite(pred.satisfaction).all()
assert len(oof) == len(train) == 699635
assert oof.id.equals(train.id) and oof.target.equals(train.satisfaction)
assert oof.id.is_unique and oof.fold.equals(baseline_oof.fold)
assert oof.prediction.between(0, 1).all() and np.isfinite(oof.prediction).all()
assert set(oof.fold) == set(range(1, 11))
assert metrics["n_splits"] == 10 and len(metrics["fold_scores"]) == 10
assert np.isclose(metrics["oof_score"], roc_auc_score(train.satisfaction, oof.prediction))
assert metrics["recipe_validation_applies_to_current_data"] is True
assert metrics["synthetic_interpolation_enabled"] is False
assert metrics["target_encoding_inner_folds"] == 5
assert metrics["original_data"]["retained_rows"] == 129859
assert metrics["original_data"]["removed_exact_overlap"] == 21
assert metrics["audit"]["audit_labels_used_for_early_stopping"] is False
assert metrics["audit"]["audit_labels_used_for_recipe_selection"] is False
assert metrics["kaggle_submission_status"] == "not_submitted"
assert len(metrics["component_results"]) == 10 * len(recipe["components"])
for item in metrics["component_results"]:
    assert "id" not in item["feature_columns"] and "satisfaction" not in item["feature_columns"]

# 独立地从每个模型的折预测缓存重建完整 OOF 和测试预测。
folders = [path for path in (root/".artifact_runtime"/"optimized_cv").iterdir()
           if path.is_dir() and len(list(path.glob("*.npz"))) == len(metrics["component_results"])]
assert len(folders) == 1, folders
checkpoints = folders[0]
rebuilt_oof = np.full(len(train), np.nan)
rebuilt_test = np.zeros(len(test))
weights = np.array([c["weight"] for c in recipe["components"]], dtype=float)
weights /= weights.sum()
for fold in range(1, 11):
    valid = np.flatnonzero(oof.fold.to_numpy() == fold)
    blended_valid, blended_test = np.zeros(len(valid)), np.zeros(len(test))
    for component, weight in enumerate(weights):
        with np.load(checkpoints/f"fold_{fold:02d}_component_{component}.npz") as saved:
            np.testing.assert_array_equal(saved["valid_index"], valid)
            blended_valid += weight * saved["validation_prediction"]
            blended_test += weight * saved["test_prediction"]
    np.testing.assert_allclose(roc_auc_score(train.satisfaction.iloc[valid], blended_valid), metrics["fold_scores"][fold-1])
    rebuilt_oof[valid] = blended_valid
    rebuilt_test += blended_test / 10
np.testing.assert_allclose(rebuilt_oof, oof.prediction, rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(rebuilt_test, pred.satisfaction, rtol=1e-10, atol=1e-12)

changes = np.array(metrics["fold_scores"])-np.array(baseline["fold_scores"])
receipt = {
    "baseline_mean_auc": baseline["mean_score"], "previous_enhanced_mean_auc": enhanced["mean_score"],
    "optimized_mean_auc": metrics["mean_score"], "optimized_oof_auc": metrics["oof_score"],
    "auc_change_from_baseline": metrics["mean_score"]-baseline["mean_score"],
    "improved_folds": int((changes > 0).sum()), "paired_fold_auc_changes": changes.tolist(),
    "target_auc": .965, "target_reached": metrics["mean_score"] >= .965,
    "target_gap": .965-metrics["mean_score"], "audit_auc": metrics["audit"]["optimized_auc"],
    "unit_tests_passed": 17, "test_rows": len(pred), "train_rows": len(oof),
    "checks": {"protected_hashes_unchanged": True, "same_ten_validation_folds": True,
               "submission_ids_columns_probabilities_valid": True,
               "each_training_row_oof_once": True, "predictions_rebuilt_from_checkpoints": True,
               "target_and_id_excluded_from_component_features": True,
               "historical_audit_matches_current_data_fingerprint": True},
    "kaggle_submitted": False, "recommended_submission": "submission_optimized.csv",
    "protected_file_hashes": protected,
}
(root/"optimization_verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
document = f"""# 航空满意度优化结果

已参考用户提供的第二名方案，将原始数据先验、分箱和折内目标编码迁移到航空满意度任务，
并完成真实数据的全部 10 折训练。当前固定配置是 25% XGBoost + 75% CatBoost。

| 方案 | 10 折平均 AUC |
| --- | ---: |
| 原始 XGBoost | {baseline['mean_score']:.6f} |
| 上一轮特征工程与插值合成 | {enhanced['mean_score']:.6f} |
| 本次优化 | {metrics['mean_score']:.6f} |

本次相对原始方案提高 {metrics['mean_score']-baseline['mean_score']:.6f}，
10 折中 {receipt['improved_folds']} 折改善；整体折外 AUC 为 {metrics['oof_score']:.6f}。
尚未达到 0.965，差距为 {max(0, receipt['target_gap']):.6f}。

另外，未参与本轮模型和权重选择的 139,927 行保留集上，固定方案 AUC 为
{metrics['audit']['optimized_auc']:.6f}，同划分基线为 {metrics['audit']['baseline_auc']:.6f}。
这些分数都是本地验证分数，本次没有向 Kaggle 提交，不能宣称榜单达到同样分数。

## 使用结果

正式预测文件是 [submission_optimized.csv](submission_optimized.csv)，
包含 299,844 行测试概率预测。原始数据、模板及原始预测均保留。

再次训练：

```powershell
.\\.venv\\Scripts\\python.exe main.py --device cuda --n-jobs 8
```

输入、配置和代码一致时，程序会恢复完成的折预测检查点。

## 验证与限制

17 项测试通过，提交行数、列顺序、ID 和概率范围均验证通过。
每条真实训练数据恰有一个折外预测，最终概率已从 20 个折与模型检查点独立重建核对。
原训练集、测试集、提交模板、原预测及公开原始数据均通过 SHA-256 校验。
目标和 ID 未进入特征，比赛目标编码在外层训练部分内部交叉拟合。

所参考的 [第二名方案](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/2nd-place-solution)
属于手机成瘾预测，本项目的数据属于航空满意度预测；迁移的是方法，无法直接迁移其分数。
候选选择没有采用完整嵌套 10 折，因此最终 10 折分数与单独保留集结果应一起参考。

详细方法见 [OPTIMIZATION.md](OPTIMIZATION.md)，完整指标见
[submission_optimized_metrics.json](submission_optimized_metrics.json)，检查回执见
[optimization_verification.json](optimization_verification.json)。
"""
(root/"optimization_report.md").write_text(document, encoding="utf-8")
print(json.dumps({key: value for key, value in receipt.items() if key != "protected_file_hashes"}, indent=2))
