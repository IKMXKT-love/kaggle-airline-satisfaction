"""开发集选型与独立保留集评估；正式 10 折预测由 main.py 完成。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from optimized_features import OriginalStatistics
from optimized_training import BASE_DIR, file_hash, fit_component, load_original_bundle, validation_fingerprint


def select_recipe(search_dir, result_files, output_path):
    """只读取开发验证集预测选型，完全不计算保留集指标。"""
    signature_path = search_dir / "search_data_signature.json"
    if not signature_path.is_file():
        raise ValueError("缺少开发预测的数据指纹，不能安全复用选型缓存。")
    data_signature = json.loads(signature_path.read_text(encoding="utf-8"))
    for name, digest in data_signature.items():
        if file_hash(BASE_DIR / name) != digest:
            raise ValueError(f"数据已改变，不能复用开发预测：{name}")
    records = []
    for path in result_files:
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    for record in records:
        record.setdefault("family", "xgboost")
        record.setdefault("original", True)
    train = pd.read_csv(BASE_DIR / "train.csv")
    test = pd.read_csv(BASE_DIR / "test.csv")
    split = np.load(search_dir / "split_indices.npz")
    labels = train.satisfaction.to_numpy(dtype=int)[split["selection"]]
    eligible = [record for record in records if not record.get("rows_weight")]
    xgb_candidates = sorted([r for r in eligible if r["family"] == "xgboost"],
                            key=lambda r: r["selection_auc"], reverse=True)[:3]
    other_candidates = []
    for family in ["catboost", "lightgbm"]:
        items = [r for r in eligible if r["family"] == family]
        if items:
            other_candidates.append(max(items, key=lambda r: r["selection_auc"]))
    trials = []
    best_score, best_components = -np.inf, None
    for xgb in xgb_candidates:
        xgb_prediction = np.load(search_dir / (xgb["name"] + "_selection.npy"))
        candidates = [([dict(xgb, weight=1.)], xgb_prediction)]
        for other in other_candidates:
            other_prediction = np.load(search_dir / (other["name"] + "_selection.npy"))
            # 保留最初明确要求的 XGBoost，限制为少量预先给定的融合权重。
            for weight in [.25, .5, .75]:
                candidates.append(([dict(xgb, weight=weight), dict(other, weight=1-weight)],
                                   weight*xgb_prediction+(1-weight)*other_prediction))
        for components, prediction in candidates:
            score = float(roc_auc_score(labels, prediction))
            trials.append({"components": [{"name": c["name"], "weight": c["weight"]} for c in components],
                           "selection_auc": score})
            if score > best_score:
                best_score, best_components = score, components
    recipe = {"components": best_components, "selection_auc": best_score,
        "data_signature": data_signature,
        "validation_dataset": validation_fingerprint(train, test, [data_signature[name]
            for name in ["external_data/train.csv", "external_data/test.csv"]]),
        "selection_protocol": {
            "development_split_seed": 20261017, "development_fraction": .8,
            "selection_split_seed": 42, "selection_fraction_of_development": .2,
            "fit_rows": len(split["fit"]), "selection_rows": len(split["selection"]),
            "reserved_audit_rows": len(split["audit"]), "audit_used_for_recipe_selection": False,
            "minimum_xgboost_weight": .25, "blend_grid": [.25, .5, .75],
            "all_scores_are_local_auc_not_kaggle_scores": True},
        "best_single_candidate": max(eligible, key=lambda r: r["selection_auc"]),
        "target_auc": .965}
    output_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    (BASE_DIR / "optimization_search.json").write_text(json.dumps({"records": records,
        "blend_trials": trials, "selected": recipe}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(recipe, ensure_ascii=False, indent=2), flush=True)


def audit_recipe(recipe_path, device, n_jobs):
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    train, test = pd.read_csv(BASE_DIR / "train.csv"), pd.read_csv(BASE_DIR / "test.csv")
    y = train.satisfaction.to_numpy(dtype=int)
    x, xt, ox, oy, prior, prior_test, source = load_original_bundle(
        train, test, "satisfaction", "id", BASE_DIR / "external_data", device, n_jobs)
    development, audit = train_test_split(np.arange(len(train)), test_size=.2,
                                          stratify=y, random_state=20261017)
    # 先固定轮数；保留集标签不用于早停、调权或重新挑模型。
    baseline_config = {"name": "raw_native_d5", "family": "xgboost", "original": False,
                       "use_te": False, "depth": 5, "child": 10, "lr": .04}
    baseline_prediction, _, baseline_details = fit_component(
        x.iloc[development], y[development], x.iloc[audit], y[audit], None,
        baseline_config, None, prior[development], prior[audit], None, device, n_jobs,
        max_rounds=1696, early_stopping=False)
    prediction = np.zeros(len(audit))
    component_details = []
    for config in recipe["components"]:
        print("保留集评估：", config["name"], flush=True)
        statistics = OriginalStatistics(extended=config.get("extended_keys", False)).fit(ox, oy)
        rounds = int(np.ceil(config["best_rounds"] * 1.1))
        pv, _, details = fit_component(x.iloc[development], y[development], x.iloc[audit], y[audit],
            None, config, statistics, prior[development], prior[audit], None, device, n_jobs,
            max_rounds=rounds, early_stopping=False)
        prediction += config["weight"] * pv
        component_details.append(details)
        print("保留集组件结果：", details["name"], details["auc"], flush=True)
    score = float(roc_auc_score(y[audit], prediction))
    report = {"baseline_auc": baseline_details["auc"], "optimized_auc": score,
              "auc_change": score-baseline_details["auc"], "fit_rows": len(development),
              "audit_rows": len(audit), "split_seed": 20261017,
              "audit_labels_used_for_early_stopping": False,
              "audit_labels_used_for_recipe_selection": False,
              "fixed_rounds_rule": "ceil(selection_best_rounds * 1.1)",
              "target_auc_reached": score >= .965,
              "components": component_details}
    recipe["audit"] = report
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    (BASE_DIR / "optimization_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["select", "audit"], required=True)
    parser.add_argument("--search-dir", type=Path, default=BASE_DIR / ".artifact_runtime" / "writeup_search")
    parser.add_argument("--recipe", type=Path, default=BASE_DIR / "optimized_recipe.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "select":
        names = ["screen_results.json", "extended_results.json", "prior_interaction_results.json",
                 "catboost_results.json", "catboost_deep_results.json", "lightgbm_results.json"]
        paths = [args.search_dir/name for name in names if (args.search_dir/name).exists()]
        select_recipe(args.search_dir, paths, args.recipe)
    else:
        audit_recipe(args.recipe, args.device, args.n_jobs)
