"""可复现的原始数据先验、XGBoost 主模型与固定权重 10 折融合。"""

import gc
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from optimized_features import (
    CompetitionFeaturePipeline, NativeCategoryEncoder, OriginalStatistics, normalize_raw,
)


BASE_DIR = Path(__file__).resolve().parent
SOURCE_URL = "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction"
WRITEUP_URL = "https://www.kaggle.com/competitions/playground-series-s6e8/writeups/2nd-place-solution"
PRIOR_SPECS = [(6, 1200), (3, 1800), (9, 1000)]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_fingerprint(train, test, original_file_hashes):
    def digest(frame):
        rows = pd.util.hash_pandas_object(frame.astype("string").fillna("__MISSING__"), index=False)
        return {"columns": list(frame.columns), "rows_sha256": hashlib.sha256(rows.to_numpy().tobytes()).hexdigest()}
    return {"train": digest(train), "test": digest(test), "original_file_hashes": original_file_hashes}


def ensure_original_data(directory):
    directory = Path(directory)
    paths = [directory / "train.csv", directory / "test.csv"]
    if all(path.is_file() for path in paths):
        return paths
    if any(path.exists() for path in paths):
        raise ValueError("原始数据目录不完整，请检查其中 train.csv 和 test.csv，程序不会覆盖现有文件。")
    directory.mkdir(parents=True, exist_ok=True)
    url = "https://www.kaggle.com/api/v1/datasets/download/teejmahal20/airline-passenger-satisfaction"
    print("下载公开原始航空数据……", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    archive = directory / "airline_passenger_satisfaction.zip"
    archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as zipped:
        for name in ["train.csv", "test.csv"]:
            candidates = [entry for entry in zipped.namelist() if Path(entry).name == name]
            if len(candidates) != 1:
                raise ValueError(f"原始数据压缩包中无法唯一找到 {name}。")
            (directory / name).write_bytes(zipped.read(candidates[0]))
    (directory / "source.json").write_text(json.dumps({"dataset_url": SOURCE_URL,
        "download_url": url, "zip_sha256": hashlib.sha256(payload).hexdigest()}, indent=2), encoding="utf-8")
    return paths


def load_original_bundle(train, test, target, id_column, original_dir, device="cpu", n_jobs=4):
    paths = ensure_original_data(original_dir)
    features = [col for col in test if col != id_column]
    x, xt = normalize_raw(train[features]), normalize_raw(test[features])
    original = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if not set(features).issubset(original.columns):
        raise ValueError("公开原始航空数据的特征与当前比赛不兼容。")
    if set(original[target].dropna().unique()) != {"satisfied", "neutral or dissatisfied"}:
        raise ValueError("原始数据标签不是预期的航空满意度标签。")
    ox = normalize_raw(original[features])
    oy = original[target].eq("satisfied").to_numpy(dtype=int)
    # 以所有共同特征的指纹排除公开原始数据与比赛行的完全重合。
    hashes = pd.util.hash_pandas_object(pd.concat([x, xt], ignore_index=True).astype("string").fillna("__MISSING__"), index=False)
    original_hashes = pd.util.hash_pandas_object(ox.astype("string").fillna("__MISSING__"), index=False)
    keep = ~original_hashes.isin(set(hashes))
    ox, oy = ox.loc[keep].reset_index(drop=True), oy[keep.to_numpy()]
    signature = {
        "features": features,
        "competition_train_features": hashlib.sha256(pd.util.hash_pandas_object(x.astype("string"), index=False).to_numpy().tobytes()).hexdigest(),
        "competition_test_features": hashlib.sha256(pd.util.hash_pandas_object(xt.astype("string"), index=False).to_numpy().tobytes()).hexdigest(),
        "original_file_hashes": [file_hash(path) for path in paths],
        "prior_specs": [list(item) for item in PRIOR_SPECS],
        "xgboost_version": __import__("xgboost").__version__,
    }
    cache = BASE_DIR / ".artifact_runtime" / "optimized_cache"
    cache.mkdir(parents=True, exist_ok=True)
    metadata_path = cache / "metadata.json"
    cached = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if (cached.get("signature") == signature and (cache / "train_priors.npy").is_file()
            and (cache / "test_priors.npy").is_file()):
        prior_train = np.load(cache / "train_priors.npy")
        prior_test = np.load(cache / "test_priors.npy")
        if prior_train.shape != (len(x), len(PRIOR_SPECS)) or prior_test.shape != (len(xt), len(PRIOR_SPECS)):
            raise ValueError("原始数据先验缓存形状错误，请检查 optimized_cache。")
        print("已读取与当前数据指纹一致的原始数据先验缓存。", flush=True)
    else:
        native = NativeCategoryEncoder().fit(ox)
        fitted_original = native.transform(ox)
        predict_train, predict_test = native.transform(x), native.transform(xt)
        prior_train, prior_test = [], []
        for depth, rounds in PRIOR_SPECS:
            print(f"训练公开原始数据先验：depth={depth}，rounds={rounds}", flush=True)
            model = XGBClassifier(n_estimators=rounds, learning_rate=.04, max_depth=depth,
                min_child_weight=5, subsample=.9, colsample_bytree=1., reg_lambda=5,
                tree_method="hist", device=device, enable_categorical=True, n_jobs=n_jobs, random_state=42)
            model.fit(fitted_original, oy, verbose=False)
            model.set_params(device="cpu")
            prior_train.append(model.predict_proba(predict_train)[:, 1])
            prior_test.append(model.predict_proba(predict_test)[:, 1])
            model.save_model(cache / f"original_prior_depth{depth}.ubj")
            del model
        prior_train, prior_test = np.column_stack(prior_train), np.column_stack(prior_test)
        np.save(cache / "train_priors.npy", prior_train)
        np.save(cache / "test_priors.npy", prior_test)
        metadata_path.write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")
    metadata = {"source_url": SOURCE_URL, "retained_rows": len(ox),
                "removed_exact_overlap": int((~keep).sum()), "signature": signature,
                "source_labels_used": "public_original_only"}
    return x, xt, ox, oy, prior_train, prior_test, metadata


def fit_component(raw_train, y_train, raw_valid, y_valid, raw_test, config,
                  statistics, prior_train, prior_valid, prior_test, device, n_jobs,
                  max_rounds=None, early_stopping=True, seed=42):
    use_original = config.get("original", True)
    prior_indices = slice(None) if config.get("multiple_priors", False) else 0
    pipeline = CompetitionFeaturePipeline(use_te=config.get("use_te", False),
        original_statistics=statistics if use_original else None,
        numeric_categories=config.get("numeric_categories", False), digits=config.get("digits", False),
        extended_keys=config.get("extended_keys", False))
    a = pipeline.fit_transform(raw_train, y_train, prior_train[:, prior_indices] if use_original else None)
    b = pipeline.transform(raw_valid, prior_valid[:, prior_indices] if use_original else None)
    c = pipeline.transform(raw_test, prior_test[:, prior_indices] if use_original else None) if raw_test is not None else None
    family = config.get("family", "xgboost")
    rounds = max_rounds or (5000 if family == "xgboost" else 4000)
    if family == "xgboost":
        model = XGBClassifier(n_estimators=rounds, learning_rate=config.get("lr", .04),
            max_depth=config.get("depth", 5), min_child_weight=config.get("child", 10),
            subsample=.9, colsample_bytree=1., reg_lambda=5, tree_method="hist", device=device,
            enable_categorical=True, n_jobs=n_jobs, random_state=seed, eval_metric="auc",
            early_stopping_rounds=150 if early_stopping else None)
        model.fit(a, y_train, eval_set=[(b, y_valid)] if early_stopping else None, verbose=False)
        best_rounds = int(model.best_iteration + 1) if early_stopping else rounds
        model.set_params(device="cpu")
    elif family == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=.03, num_leaves=31,
            min_child_samples=100, colsample_bytree=1., subsample=.9, subsample_freq=1,
            reg_lambda=5, n_jobs=n_jobs, random_state=seed, verbosity=-1)
        kwargs = {"eval_set": [(b, y_valid)], "eval_metric": "auc",
                  "callbacks": [lgb.early_stopping(150, verbose=False)]} if early_stopping else {}
        model.fit(a, y_train, **kwargs)
        best_rounds = int(model.best_iteration_) if early_stopping else rounds
    elif family == "catboost":
        from catboost import CatBoostClassifier
        categorical = a.select_dtypes(include=["category"]).columns.tolist()
        for frame in [a, b] + ([c] if c is not None else []):
            for col in categorical:
                frame[col] = frame[col].astype("string").fillna("__MISSING__").astype(str)
        model = CatBoostClassifier(iterations=rounds, learning_rate=.04, depth=config.get("depth", 6),
            l2_leaf_reg=5, loss_function="Logloss", eval_metric="AUC", task_type="GPU" if device == "cuda" else "CPU",
            cat_features=categorical, random_seed=seed, thread_count=n_jobs, max_ctr_complexity=2,
            allow_writing_files=False, verbose=False)
        kwargs = {"eval_set": (b, y_valid), "early_stopping_rounds": 150} if early_stopping else {}
        model.fit(a, y_train, **kwargs)
        best_rounds = int(model.best_iteration_ + 1) if early_stopping else rounds
    else:
        raise ValueError(f"未知模型：{family}")
    validation_prediction = model.predict_proba(b)[:, 1]
    test_prediction = model.predict_proba(c)[:, 1] if c is not None else None
    details = {"name": config["name"], "family": family, "best_rounds": best_rounds,
               "feature_count": len(a.columns), "feature_columns": list(a.columns),
               "auc": float(roc_auc_score(y_valid, validation_prediction))}
    del model, pipeline, a, b, c
    gc.collect()
    return validation_prediction, test_prediction, details


def train_optimized(train, test, target="satisfaction", id_column="id", device="cpu", n_jobs=4,
                    original_dir=None, recipe_path=None, max_rounds=None):
    if target not in train or target in test or target == id_column:
        raise ValueError("训练集必须含目标列，测试集不得含目标列，且目标列不能是 ID。")
    features = [col for col in train if col not in (target, id_column)]
    if not features or set(features) != {col for col in test if col != id_column}:
        raise ValueError("排除 ID 和目标后，训练与测试的特征列必须一致且非空。")
    if test.empty:
        raise ValueError("测试集不能为空。")
    recipe_path = Path(recipe_path or BASE_DIR / "optimized_recipe.json")
    if not recipe_path.is_file():
        raise ValueError(f"找不到优化配置：{recipe_path}")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    components = recipe["components"]
    weights = np.array([item["weight"] for item in components], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("融合权重必须非负且总和大于 0。")
    weights /= weights.sum()
    encoder = LabelEncoder()
    y = encoder.fit_transform(train[target])
    if train[target].isna().any() or len(encoder.classes_) != 2 or np.bincount(y).min() < 10:
        raise ValueError("优化流程要求非空二分类标签且每类至少 10 行。")
    if encoder.classes_.tolist() not in [[False, True], [0, 1]]:
        raise ValueError("航空优化流程要求 satisfaction 为 False/True 或 0/1。")
    x, xt, ox, oy, prior_train, prior_test, source = load_original_bundle(
        train, test, target, id_column, original_dir or BASE_DIR / "external_data", device, n_jobs)
    assert target not in x and id_column not in x
    validation_matches = recipe.get("validation_dataset") == validation_fingerprint(
        train, test, source["signature"].get("original_file_hashes", []))
    if recipe.get("validation_dataset") and not validation_matches:
        print("当前数据与配置选型时的数据不同；本次报告不复用历史保留集评分。", flush=True)
    statistics = {
        extended: OriginalStatistics(extended=extended).fit(ox, oy)
        for extended in {item.get("extended_keys", False) for item in components}
    }
    checkpoint_signature = {
        "recipe": recipe, "device": device, "max_rounds": max_rounds,
        "source": source["signature"], "seed": 42, "folds": 10,
        "labels": hashlib.sha256(y.tobytes()).hexdigest(),
        "code": [file_hash(BASE_DIR / name) for name in ["optimized_training.py", "optimized_features.py"]],
    }
    checkpoint_key = hashlib.sha256(json.dumps(checkpoint_signature, sort_keys=True).encode()).hexdigest()[:20]
    checkpoint_dir = BASE_DIR / ".artifact_runtime" / "optimized_cv" / checkpoint_key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    oof, prediction = np.full(len(train), np.nan), np.zeros(len(test))
    folds = np.zeros(len(train), dtype=int)
    fold_scores, component_results = [], []
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (fit, valid) in enumerate(splitter.split(x, y), start=1):
        blended_valid, blended_test = np.zeros(len(valid)), np.zeros(len(test))
        for component_index, (weight, config) in enumerate(zip(weights, components)):
            print(f"第 {fold:02d}/10 折，{config['name']}：训练 {len(fit):,}，验证 {len(valid):,}", flush=True)
            checkpoint = checkpoint_dir / f"fold_{fold:02d}_component_{component_index}.npz"
            details_path = checkpoint.with_suffix(".json")
            if checkpoint.is_file() and details_path.is_file():
                with np.load(checkpoint) as saved:
                    if not np.array_equal(saved["valid_index"], valid):
                        raise ValueError("折缓存中的验证行号与当前划分不一致。")
                    pv, pt = saved["validation_prediction"], saved["test_prediction"]
                details = json.loads(details_path.read_text(encoding="utf-8"))
                print("  已恢复与数据、配置和代码指纹一致的折预测缓存。", flush=True)
            else:
                pv, pt, details = fit_component(x.iloc[fit], y[fit], x.iloc[valid], y[valid], xt,
                    config, statistics[config.get("extended_keys", False)], prior_train[fit], prior_train[valid],
                    prior_test, device, n_jobs, max_rounds=max_rounds, seed=42 + fold)
                np.savez_compressed(checkpoint, valid_index=valid, validation_prediction=pv, test_prediction=pt)
                details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
            blended_valid += weight * pv
            blended_test += weight * pt
            component_results.append({"fold": fold, "weight": float(weight), **details})
            print(f"  {config['name']} AUC={details['auc']:.6f}，最佳轮数={details['best_rounds']}", flush=True)
        oof[valid], folds[valid] = blended_valid, fold
        prediction += blended_test / 10
        fold_scores.append(float(roc_auc_score(y[valid], blended_valid)))
        print(f"第 {fold:02d}/10 折融合 AUC={fold_scores[-1]:.6f}", flush=True)
    if (not np.isfinite(oof).all() or not np.isfinite(prediction).all()
            or not ((prediction >= 0) & (prediction <= 1)).all()):
        raise ValueError("优化预测出现无效数值，未写入输出。")
    report = {"task": "classification", "target": target, "classes": encoder.classes_.tolist(),
        "metric": "auc", "n_splits": 10, "random_state": 42, "pipeline": "writeup_optimized",
        "train_rows": len(train), "test_rows": len(test), "raw_feature_columns": list(x.columns),
        "components": components, "component_results": component_results, "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)), "std_score": float(np.std(fold_scores)),
        "oof_score": float(roc_auc_score(y, oof)), "target_auc": .965,
        "target_auc_reached_in_cv": bool(np.mean(fold_scores) >= .965),
        "kaggle_submission_status": "not_submitted", "device": device,
        "synthetic_interpolation_enabled": False, "target_encoding_inner_folds": 5,
        "source_writeup": WRITEUP_URL, "source_competition": "smartphone_addiction_s6e8",
        "actual_competition": "airline_satisfaction_s6e10", "original_data": source,
        "selection_protocol": recipe.get("selection_protocol"),
        "recipe_validation_applies_to_current_data": validation_matches,
        "audit": recipe.get("audit") if validation_matches else None}
    print(f"10 折平均 AUC={report['mean_score']:.6f}，OOF={report['oof_score']:.6f}", flush=True)
    return prediction, oof, folds, report
