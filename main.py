"""航空满意度优化模型 + 10 折交叉验证；保留 legacy 通用基线。

真实数据：python main.py
模拟数据：python main.py --demo
指定路径：python main.py --train train.csv --test test.csv --target satisfaction
依赖安装：python -m pip install -r requirements.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

from feature_engineering import FoldFeatureEngineer, synthesize_training_rows


BASE_DIR = Path(__file__).resolve().parent
N_SPLITS = 10
SEED = 42


def create_demo_data(directory):
    """创建独立的模拟数据；它们不代表真实比赛数据。"""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / name for name in
             ("train.csv", "test.csv", "sample_submission.csv")]
    if all(path.exists() for path in paths):
        return
    if any(path.exists() for path in paths):
        raise ValueError("示例目录已有部分数据，请使用另一个 --demo-dir，避免覆盖。")

    rng = np.random.default_rng(SEED)
    size, train_size = 1000, 800
    service = rng.integers(1, 6, size)
    delay = rng.exponential(18, size)
    travel = rng.choice(["Business", "Personal", "Other"], size)
    logit = 1.1 * (service - 3) - delay / 45 + (travel == "Business") * 0.8
    target = (rng.random(size) < 1 / (1 + np.exp(-logit))).astype(int)
    frame = pd.DataFrame({
        "id": np.arange(size),
        "age": rng.integers(18, 80, size),
        "service_rating": service.astype(float),
        "arrival_delay": delay,
        "travel_type": travel,
    })
    frame.loc[rng.choice(size, 35, replace=False), "service_rating"] = np.nan
    frame.loc[rng.choice(size, 25, replace=False), "travel_type"] = None
    train = frame.iloc[:train_size].copy()
    train["satisfaction"] = target[:train_size]
    train.to_csv(paths[0], index=False)
    frame.iloc[train_size:].to_csv(paths[1], index=False)
    pd.DataFrame({"id": frame.iloc[train_size:]["id"], "satisfaction": 0.5}).to_csv(
        paths[2], index=False
    )


def train_predict(train, test, target, id_column, task, n_estimators,
                  feature_engineering=True, synthetic_ratio=0.15,
                  synthetic_weight=0.25, device="cpu", n_jobs=4):
    if not np.isfinite(synthetic_ratio) or not 0 <= synthetic_ratio <= 1:
        raise ValueError("synthetic_ratio 必须在 [0, 1]。")
    if not np.isfinite(synthetic_weight) or not 0 < synthetic_weight <= 1:
        raise ValueError("synthetic_weight 必须在 (0, 1]。")
    if target not in train or target in test:
        raise ValueError(f"训练集必须含目标列 {target}，测试集不能含此列。")
    if train[target].isna().any():
        raise ValueError("训练集目标列存在缺失值，请先检查标签。")
    if len(train) < N_SPLITS or test.empty:
        raise ValueError("10 折交叉验证至少需要 10 条训练数据，测试集不能为空。")

    features = [col for col in train if col not in (target, id_column)]
    test_features = [col for col in test if col != id_column]
    if not features or set(features) != set(test_features):
        raise ValueError("排除目标列和 ID 后，训练集与测试集的特征列必须一致且非空。")
    # ID 和目标都不会进入特征矩阵。
    x_train, x_test = train[features].copy(), test[features].copy()
    numeric = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in features if col not in numeric]
    for frame in (x_train, x_test):
        for col in numeric:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float32")
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
        for col in categorical:
            frame[col] = frame[col].astype("string").fillna("__MISSING__").astype(str)

    raw_y = train[target]
    if task == "auto":
        task = "classification" if raw_y.nunique() == 2 else "regression"
    classes = None
    if task == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw_y)
        if len(encoder.classes_) != 2:
            raise ValueError("当前分类流程用于二分类，目标列必须恰好有两个类别。")
        if np.bincount(y).min() < N_SPLITS:
            raise ValueError("分层 10 折要求每个类别至少有 10 条数据。")
        classes = encoder.classes_.tolist()
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        model_type, objective, metric = XGBClassifier, "binary:logistic", "auc"
        print(f"二分类；输出类别 {classes[1]!r} 的概率；标签顺序：{classes}")
    else:
        y = pd.to_numeric(raw_y, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("回归目标必须为有限数值；文本标签请使用 --task classification。")
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        model_type, objective, metric = XGBRegressor, "reg:squarederror", "rmse"
        print("回归；输出预测数值")

    def score(actual, prediction):
        return float(roc_auc_score(actual, prediction) if task == "classification"
                     else root_mean_squared_error(actual, prediction))

    oof = np.full(len(train), np.nan)
    fold_ids = np.zeros(len(train), dtype=int)
    prediction = np.zeros(len(test))
    fold_scores, best_iterations, augmentation = [], [], []
    engineered_features = features
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_train, y), start=1):
        fit_raw = x_train.iloc[train_idx].copy()
        valid_raw = x_train.iloc[valid_idx]
        synthetic, synthetic_y, parents = synthesize_training_rows(
            fit_raw, y[train_idx], task, synthetic_ratio, SEED + fold
        )
        if len(parents) and (parents.min() < 0 or parents.max() >= len(train_idx)):
            raise ValueError("合成数据父样本不属于当前训练折。")
        augmented = pd.concat([fit_raw, synthetic], ignore_index=True)
        fit_y = np.concatenate([y[train_idx], synthetic_y])
        weights = np.r_[np.ones(len(fit_raw)), np.full(len(synthetic), synthetic_weight)]
        if feature_engineering:
            # 所有统计量只从真实训练折学习；合成样本也不参与频次统计。
            engineer = FoldFeatureEngineer().fit(fit_raw)
            fit_features = engineer.transform(augmented)
            valid_features = engineer.transform(valid_raw)
            test_features_frame = engineer.transform(x_test)
        else:
            fit_features, valid_features, test_features_frame = augmented, valid_raw.copy(), x_test.copy()
        numeric = fit_features.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical = [col for col in fit_features if col not in numeric]
        engineered_features = fit_features.columns.tolist()
        assert target not in engineered_features and id_column not in engineered_features
        # 减少全量训练的内存占用；保留 NaN 供 XGBoost 处理。
        for frame in (fit_features, valid_features, test_features_frame):
            frame[numeric] = frame[numeric].astype("float32")
        # 每折独立学习类别编码，验证集不会参与预处理拟合。
        preprocessor = ColumnTransformer([
            ("numeric", "passthrough", numeric),
            ("category", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), categorical),
        ], sparse_threshold=1.0)
        x_fit = preprocessor.fit_transform(fit_features)
        x_valid = preprocessor.transform(valid_features)
        x_predict = preprocessor.transform(test_features_frame)
        print(f"第 {fold:02d}/{N_SPLITS} 折：真实训练 {len(fit_raw):,} 行 + "
              f"合成 {len(synthetic):,} 行，验证 {len(valid_idx):,} 行，"
              f"编码前特征 {len(engineered_features)} 列", flush=True)
        model = model_type(
            objective=objective,
            eval_metric=metric,
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            device=device,
            early_stopping_rounds=50,
            random_state=SEED + fold,
            n_jobs=n_jobs,
        )
        model.fit(x_fit, fit_y, sample_weight=weights,
                  eval_set=[(x_valid, y[valid_idx])], verbose=False)
        if device == "cuda":
            # 输入矩阵在 CPU；训练后切换推理设备以避免隐式设备转换和额外显存占用。
            model.set_params(device="cpu")
        # XGBoost 的预测接口会自动使用早停时的最佳迭代轮数。
        if task == "classification":
            valid_prediction = model.predict_proba(x_valid)[:, 1]
            test_prediction = model.predict_proba(x_predict)[:, 1]
        else:
            valid_prediction = model.predict(x_valid)
            test_prediction = model.predict(x_predict)
        oof[valid_idx] = valid_prediction
        fold_ids[valid_idx] = fold
        prediction += test_prediction / N_SPLITS
        fold_scores.append(score(y[valid_idx], valid_prediction))
        best_iterations.append(int(model.best_iteration + 1))
        augmentation.append({"fold": fold, "real_training_rows": len(fit_raw),
                             "synthetic_rows": len(synthetic), "validation_rows": len(valid_idx)})
        print(f"第 {fold:02d}/{N_SPLITS} 折 {metric.upper()}={fold_scores[-1]:.6f}，"
              f"最佳轮数={best_iterations[-1]}", flush=True)
        # 避免下一折拟合时仍保留上一折的全量数据和 GPU 模型。
        del model, preprocessor, x_fit, x_valid, x_predict
        del fit_features, valid_features, test_features_frame, fit_raw, augmented, synthetic

    if not np.isfinite(oof).all() or not np.isfinite(prediction).all():
        raise ValueError("预测中出现非有限数值，未保存提交文件。")
    if task == "classification" and not ((prediction >= 0) & (prediction <= 1)).all():
        raise ValueError("分类概率超出 [0, 1]，未保存提交文件。")
    report = {
        "task": task, "target": target, "metric": metric,
        "n_splits": N_SPLITS, "random_state": SEED,
        "train_rows": len(train), "test_rows": len(test),
        "feature_columns": engineered_features, "raw_feature_columns": features, "classes": classes,
        "feature_engineering": feature_engineering, "synthetic_ratio": synthetic_ratio,
        "synthetic_weight": synthetic_weight, "augmentation_by_fold": augmentation,
        "synthetic_method": "same_context_same_class_continuous_interpolation" if task == "classification" else "same_context_continuous_and_target_interpolation",
        "statistics_fitted_on": "real_training_fold_only",
        "device": device, "n_estimators": n_estimators,
        "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)),
        "std_score": float(np.std(fold_scores)),
        "oof_score": score(y, oof), "best_iterations": best_iterations,
    }
    print(f"10 折平均 {metric.upper()}={report['mean_score']:.6f}，"
          f"标准差={report['std_score']:.6f}，OOF={report['oof_score']:.6f}")
    return prediction, oof, fold_ids, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="训练集 CSV 路径")
    parser.add_argument("--test", type=Path, help="测试集 CSV 路径")
    parser.add_argument("--sample", type=Path, help="提交模板 CSV 路径")
    parser.add_argument("--target", help="目标列，默认从提交模板或数据列差异推断")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--task", choices=["auto", "classification", "regression"], default="auto",
                        help="auto：两个唯一标签按二分类处理，否则按回归处理")
    parser.add_argument("--n-estimators", type=int,
                        help="每折最多轮数：优化 XGBoost 5000、其他 4000；legacy 1500；示例 80")
    parser.add_argument("--output", type=Path, help="预测 CSV 输出路径")
    parser.add_argument("--pipeline", choices=["optimized", "legacy"], default="optimized",
                        help="默认运行竞赛方案优化流程；legacy 为之前的通用流程")
    parser.add_argument("--recipe", type=Path, default=BASE_DIR / "optimized_recipe.json",
                        help="优化流程的固定模型配置与融合权重")
    parser.add_argument("--original-dir", type=Path, default=BASE_DIR / "external_data",
                        help="公开原始航空数据目录；缺失时下载公开数据")
    parser.add_argument("--no-feature-engineering", action="store_true", help="使用 legacy 流程并关闭特征工程")
    parser.add_argument("--synthetic-ratio", type=float, default=0.15,
                        help="仅 legacy：合成行数/真实训练折行数，默认 0.15；0 表示关闭")
    parser.add_argument("--synthetic-weight", type=float, default=0.25,
                        help="仅 legacy：合成样本相对真实样本的权重，默认 0.25")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="XGBoost 训练设备")
    parser.add_argument("--n-jobs", type=int, default=4, help="CPU 线程数，默认 4")
    parser.add_argument("--demo", action="store_true", help="创建并运行模拟数据示例")
    parser.add_argument("--demo-dir", type=Path, default=BASE_DIR / "demo_data")
    args = parser.parse_args()
    data_dir = args.demo_dir if args.demo else BASE_DIR
    if args.demo:
        print("正在运行模拟数据示例；其分数和预测不代表真实比赛表现。")
        create_demo_data(data_dir)
    train_path = args.train or data_dir / "train.csv"
    test_path = args.test or data_dir / "test.csv"
    sample_path = args.sample or data_dir / "sample_submission.csv"
    use_optimized = (args.pipeline == "optimized" and not args.demo
                     and args.task != "regression" and not args.no_feature_engineering)
    output_path = args.output or data_dir / ("submission_optimized.csv" if use_optimized else "submission_enhanced.csv")
    rounds = args.n_estimators if args.n_estimators is not None else (80 if args.demo else 1500)
    if rounds < 1:
        parser.error("--n-estimators 必须是正整数。")
    if not np.isfinite(args.synthetic_ratio) or not 0 <= args.synthetic_ratio <= 1:
        parser.error("--synthetic-ratio 必须在 [0, 1]。")
    if not np.isfinite(args.synthetic_weight) or not 0 < args.synthetic_weight <= 1:
        parser.error("--synthetic-weight 必须在 (0, 1]。")
    if args.n_jobs < 1:
        parser.error("--n-jobs 必须是正整数。")
    for path in (train_path, test_path):
        if not path.is_file():
            parser.error(f"找不到 {path}。请提供真实数据，或使用 --demo 运行模拟数据。")
    if args.sample and not sample_path.is_file():
        parser.error(f"找不到指定的提交模板：{sample_path}")
    # 保护输入数据及现有提交模板。
    oof_path = output_path.with_name(output_path.stem + "_oof.csv")
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    inputs = {path.resolve() for path in (train_path, test_path, sample_path, Path(__file__),
                                         BASE_DIR / "feature_engineering.py", BASE_DIR / "optimized_features.py",
                                         BASE_DIR / "optimized_training.py", args.recipe,
                                         args.original_dir / "train.csv", args.original_dir / "test.csv")}
    if any(path.resolve() in inputs for path in (output_path, oof_path, metrics_path)):
        parser.error("输出路径不能覆盖输入数据、提交模板或脚本。")

    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    sample = pd.read_csv(sample_path) if sample_path.is_file() else None
    candidates = ([col for col in sample if col != args.id_column] if sample is not None
                  else [col for col in train if col not in test])
    target = args.target
    if target is None:
        if len(candidates) != 1:
            parser.error("无法唯一推断目标列，请用 --target 指定。")
        target = candidates[0]
    for name, frame in (("训练集", train), ("测试集", test)):
        if args.id_column not in frame or frame[args.id_column].isna().any() or frame[args.id_column].duplicated().any():
            parser.error(f"{name}必须包含非空且唯一的 {args.id_column} 列。")
    if sample is not None:
        if set(sample.columns) != {args.id_column, target}:
            parser.error("提交模板必须仅包含 ID 列和目标列。")
        if (len(sample) != len(test) or sample[args.id_column].isna().any()
                or sample[args.id_column].duplicated().any()
                or set(sample[args.id_column]) != set(test[args.id_column])):
            parser.error("提交模板的 ID 必须与测试集一一对应。")

    if use_optimized:
        from optimized_training import train_optimized
        print("运行竞赛方案优化流程：关闭人工插值增强，使用原始数据先验和折内编码。", flush=True)
        prediction, oof, folds, report = train_optimized(
            train, test, target, args.id_column, args.device, args.n_jobs,
            original_dir=args.original_dir, recipe_path=args.recipe, max_rounds=args.n_estimators)
    else:
        prediction, oof, folds, report = train_predict(
            train, test, target, args.id_column, args.task, rounds,
            feature_engineering=not args.no_feature_engineering,
            synthetic_ratio=args.synthetic_ratio, synthetic_weight=args.synthetic_weight,
            device=args.device, n_jobs=args.n_jobs,
        )
    submission = pd.DataFrame({args.id_column: test[args.id_column], target: prediction})
    if sample is not None:
        # 按模板 ID 重排，兼容模板与测试集行顺序不同的情况。
        submission = submission.set_index(args.id_column).loc[sample[args.id_column]].reset_index()
        submission = submission[sample.columns]
    oof_frame = pd.DataFrame({args.id_column: train[args.id_column], "target": train[target],
                              "prediction": oof, "fold": folds})
    report["data_source"] = "synthetic_demo" if args.demo else "user_provided"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    oof_frame.to_csv(oof_path, index=False)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"预测文件：{output_path.resolve()}")
    print(f"折外预测：{oof_path.resolve()}")
    print(f"评估结果：{metrics_path.resolve()}")


if __name__ == "__main__":
    main()
