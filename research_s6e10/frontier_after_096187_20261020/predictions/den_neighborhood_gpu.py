"""Five frozen-fold GPU neighborhood fits, followed by one fixed 202-view refit."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from frozen import file_sha256, reconstruct_verified_v1
from neighborhood_model import PARAMS, RAW, array_sha, fit_fold
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from runtime import RuntimePaths, find_data_file

FOLD_SHA = "0d2df9a59085ee4ac172f74548ccb3b02d46be5bf5386db2e2d58263ad424c08"
PARENT_SHA = "5724eb0a2e303129ab351a535d3d2e7e8bf02d675068727ec2b9345201ffd345"
SPEC = {
    "version": "new-neighborhood-view-202-v1",
    "params": PARAMS,
    "old_members": 201,
    "new_members": 202,
    "parent": "exp037",
    "parent_submission_sha256": PARENT_SHA,
    "fold_assignment_sha256": FOLD_SHA,
    "final_stack_cv": "unavailable_without_nested_external_base_refits",
    "frozen_correction": "exp036_unchanged",
    "early_stopping": False,
}


def validate_policy(config: dict[str, Any]) -> None:
    if (
        config.get("neighborhood_gpu") != SPEC
        or config["validation"]["fold_version"] != "v1"
        or config["validation"]["seed"] != 2036
        or config["runtime"]["seed"] != 2026
        or config["runtime"]["accelerator"] != "gpu"
        or config["runtime"]["internet"] is not False
    ):
        raise ValueError("The fixed neighborhood experiment policy changed")


def parent_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("neighborhood_gpu")
    result["runtime"]["accelerator"] = "cpu"
    return result


def load_raw(
    config: dict[str, Any], paths: RuntimePaths
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    loaded = {}
    for name, key in [
        ("train.csv", "train_file"),
        ("test.csv", "test_file"),
        ("sample_submission.csv", "sample_submission_file"),
    ]:
        path = find_data_file(config["data"][key], paths)
        if file_sha256(path) != config["data"]["sha256"][name]:
            raise ValueError("Raw data fingerprint mismatch: " + name)
        loaded[name] = pd.read_csv(path)
    train, test, sample = (
        loaded["train.csv"],
        loaded["test.csv"],
        loaded["sample_submission.csv"],
    )
    if (
        train.shape != (699635, 23)
        or test.shape != (299844, 22)
        or not train.id.is_unique
        or not test.id.is_unique
        or np.intersect1d(train.id, test.id).size
        or not np.array_equal(sample.id, test.id)
        or train[list(RAW)].duplicated().any()
    ):
        raise ValueError("Raw row/ID/schema mismatch")
    target = train.satisfaction.astype(np.int8).to_numpy()
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Binary target required")
    return train, test, target


def preflight(
    config: dict[str, Any], paths: RuntimePaths, *, input_root: Path | None = None
) -> dict[str, Any]:
    from global_refresh import compose
    from global_refresh import preflight as old_preflight

    validate_policy(config)
    inp, correction, checks, _ = old_preflight(
        parent_config(config), paths, input_root=input_root
    )
    train, test, target = load_raw(config, paths)
    folds = reconstruct_verified_v1(target, expected_assignment_sha256=FOLD_SHA)
    if (
        not np.array_equal(train.id, inp.data.train_ids)
        or not np.array_equal(test.id, inp.data.test_ids)
        or not np.array_equal(folds, inp.data.own_fold)
    ):
        raise ValueError("Raw data and frozen201 identities differ")
    parent = find_data_file(
        "artifacts/reference/exp037_anchor/control_exp037.csv", paths
    )
    linear = find_data_file(
        "artifacts/reference/exp037_anchor/parent_exp037_stacker.json", paths
    )
    if (
        file_sha256(parent) != PARENT_SHA
        or file_sha256(linear)
        != "b4da3591febda2610f8e79e81ad3a3554b012287ffc8c35b29d8460899e18650"
    ):
        raise ValueError("Exp037 baseline changed")
    state = json.loads(linear.read_text())
    replay, _ = compose(
        inp.data.test, np.array(state["coefficients"]), state["intercept"], correction
    )
    parent_frame = pd.read_csv(parent, float_precision="round_trip")
    if (
        not np.array_equal(parent_frame.id, test.id)
        or np.max(np.abs(replay - parent_frame.satisfaction.to_numpy())) > 1e-7
    ):
        raise ValueError("Full exp037 test baseline replay failed")
    return {
        "data": inp.data,
        "correction": correction,
        "checks": checks,
        "train": train,
        "test": test,
        "target": target,
        "fold": folds,
        "parent": parent,
    }


def probability(value: np.ndarray, rows: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if (
        value.shape != (rows,)
        or not np.isfinite(value).all()
        or ((value < 0) | (value > 1)).any()
    ):
        raise ValueError("Invalid neighborhood probability vector")
    return value


def train_base(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: np.ndarray,
    folds: np.ndarray,
    output: Path,
    *,
    runtime_name: str,
    learner: Any = None,
) -> dict[str, Any]:
    """One OOF per row, final fixed epochs, safe complete-fold resume only."""
    learner = fit_fold if learner is None else learner
    environment = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "runtime": runtime_name,
    }
    if learner is fit_fold:
        if runtime_name not in {"kaggle", "colab"}:
            raise RuntimeError("Neighborhood training is remote-only")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Select a CUDA GPU; no CPU fallback")
        environment["torch"] = importlib.metadata.version("torch")
        environment["cuda"] = torch.version.cuda
        environment["gpu"] = torch.cuda.get_device_name(0)
        environment["capability"] = list(torch.cuda.get_device_capability(0))
    output.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(train), np.nan)
    head_oof = oof.copy()
    neighbor_oof = oof.copy()
    coverage = np.zeros(len(train), dtype=np.int8)
    test_predictions = []
    records = []
    code_hashes = {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")}
    for fold in range(5):
        training = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        directory = output / f"fold{fold}"
        directory.mkdir(exist_ok=True)
        checkpoint, completion = (
            directory / "predictions.npz",
            directory / "complete.json",
        )
        signature = {
            "environment": environment,
            "params": PARAMS,
            "seed": 2026 + fold,
            "source": code_hashes,
            "train_ids": array_sha(train.id.to_numpy()[training].astype("<i8")),
            "valid_ids": array_sha(train.id.to_numpy()[valid].astype("<i8")),
            "labels": array_sha(target[training]),
            "train_features": array_sha(
                pd.util.hash_pandas_object(
                    train.iloc[training][list(RAW)], index=False
                ).to_numpy()
            ),
            "valid_features": array_sha(
                pd.util.hash_pandas_object(
                    train.iloc[valid][list(RAW)], index=False
                ).to_numpy()
            ),
            "test_features": array_sha(
                pd.util.hash_pandas_object(test[list(RAW)], index=False).to_numpy()
            ),
        }
        if completion.exists():
            saved = json.loads(completion.read_text())
            if saved["signature"] != signature:
                raise ValueError("Fold resume signature changed")
            for name, pin in saved["files"].items():
                if file_sha256(directory / name) != pin:
                    raise ValueError("Fold cache fingerprint mismatch")
            with np.load(checkpoint, allow_pickle=False) as z:
                if not np.array_equal(
                    z["valid_ids"], train.id.to_numpy()[valid]
                ) or not np.array_equal(z["test_ids"], test.id):
                    raise ValueError("Fold cached ID mismatch")
                vp, tp, hp, npred = [
                    z[k].copy()
                    for k in ["valid", "test", "head_valid", "neighbor_valid"]
                ]
            metadata = saved["training"]
        else:
            predictions, metadata = learner(
                train.iloc[training][list(RAW)],
                target[training],
                train.id.to_numpy()[training],
                [
                    (train.iloc[valid][list(RAW)], train.id.to_numpy()[valid]),
                    (test[list(RAW)], test.id.to_numpy()),
                ],
                seed=2026 + fold,
                output=directory,
                runtime_name=runtime_name,
            )
            vp, tp = predictions
            arms = metadata.pop("arms")
            hp, npred = arms[0]["head"], arms[0]["neighbor"]
            temporary = checkpoint.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    valid=probability(vp, len(valid)),
                    test=probability(tp, len(test)),
                    head_valid=hp,
                    neighbor_valid=npred,
                    valid_ids=train.id.to_numpy()[valid],
                    test_ids=test.id.to_numpy(),
                )
            temporary.replace(checkpoint)
            files = {
                name: file_sha256(directory / name)
                for name in [
                    "predictions.npz",
                    "encoder.pt",
                    "context.json",
                    "training.json",
                ]
            }
            completion.write_text(
                json.dumps(
                    {"signature": signature, "files": files, "training": metadata},
                    sort_keys=True,
                    indent=2,
                )
            )
        oof[valid] = probability(vp, len(valid))
        coverage[valid] += 1
        head_oof[valid] = probability(hp, len(valid))
        neighbor_oof[valid] = probability(npred, len(valid))
        test_predictions.append(probability(tp, len(test)))
        record = {
            "fold": fold,
            "auc": roc_auc_score(target[valid], vp),
            "head_auc": roc_auc_score(target[valid], hp),
            "neighbor_auc": roc_auc_score(target[valid], npred),
            "fit_seconds": metadata["fit_seconds"],
        }
        records.append(record)
        print(record, flush=True)
    if not np.all(coverage == 1):
        raise ValueError("OOF coverage must equal one")
    pt = probability(np.mean(test_predictions, axis=0, dtype=np.float64), len(test))
    pd.DataFrame(
        {
            "id": train.id,
            "fold": folds,
            "prediction": oof,
            "head": head_oof,
            "neighbor": neighbor_oof,
        }
    ).to_parquet(output / "oof.parquet", index=False)
    pd.DataFrame({"id": test.id, "prediction": pt}).to_parquet(
        output / "test_predictions.parquet", index=False
    )
    base_metrics = {
        "overall_oof_auc": roc_auc_score(target, oof),
        "per_fold": records,
        "head_oof_auc": roc_auc_score(target, head_oof),
        "neighbor_oof_auc": roc_auc_score(target, neighbor_oof),
        "log_loss": log_loss(target, oof),
        "brier": brier_score_loss(target, oof),
        "coverage_one": True,
        "fold_assignment_sha256": array_sha(folds),
        "params": PARAMS,
    }
    (output / "metrics.json").write_text(json.dumps(base_metrics, indent=2))
    return {
        "oof": oof,
        "test": pt,
        "metrics": base_metrics,
        "head": head_oof,
        "neighbor": neighbor_oof,
    }


def correlations(
    target: np.ndarray, oof: np.ndarray, owned: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Differentiate prediction agreement from target-aware residual correlation."""
    rows = []
    for name, old in owned.items():
        old = probability(old, len(target))
        rows.append(
            {
                "model": name,
                "prediction_spearman": pd.Series(oof).corr(
                    pd.Series(old), method="spearman"
                ),
                "residual_pearson": float(
                    np.corrcoef(target - oof, target - old)[0, 1]
                ),
                "base_auc": roc_auc_score(target, old),
                "new_auc": roc_auc_score(target, oof),
            }
        )
    return pd.DataFrame(rows)


def run_neighborhood(
    config: dict[str, Any], paths: RuntimePaths, out: Path
) -> dict[str, Any]:
    """Train one genuinely new model view; retain the parent correction exactly."""
    from global_refresh import compose
    from public_stack import fit_stacker, logits

    if paths.name not in {"kaggle", "colab"}:
        raise RuntimeError("Remote GPU training only")
    verified = preflight(config, paths)
    trained = train_base(
        verified["train"],
        verified["test"],
        verified["target"],
        verified["fold"],
        out / "neighborhood",
        runtime_name=paths.name,
    )
    data = verified["data"]
    added = replace(
        data,
        train=np.column_stack(
            [data.train, logits(trained["oof"], len(data.train), "probability")]
        ),
        test=np.column_stack(
            [data.test, logits(trained["test"], len(data.test), "probability")]
        ),
        names=[*data.names, "owned/new_neighborhood_v1"],
    )
    started = time.monotonic()
    model = fit_stacker(added)
    seconds = time.monotonic() - started
    weights = model.coef_[0].astype(float)
    intercept = float(model.intercept_[0])
    prediction, _ = compose(added.test, weights, intercept, verified["correction"])
    state = {
        "names": added.names,
        "coefficients": weights.tolist(),
        "intercept": intercept,
        "recipe": SPEC,
        "parent_checks": verified["checks"],
        "new_base_metrics": trained["metrics"],
    }
    environment = {
        "source_sha256": {p.name: file_sha256(p) for p in paths.src_dir.glob("*.py")},
        "repository_revision_at_prepare": config["tracking"][
            "repository_revision_at_prepare"
        ],
        "source_snapshot_dirty": config["tracking"]["source_snapshot_dirty"],
        "params": SPEC,
        "seed": 2026,
        "base_fold_version": "v1",
        "execution": "Kaggle/Colab one CUDA GPU, plus fixed CPU LR",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "numpy", "pandas", "scipy", "scikit-learn"]
        },
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2))
    shutil.copy2(verified["parent"], out / "control_exp037.csv")
    ledger = [
        *data.ledger,
        {
            "name": "owned/new_neighborhood_v1",
            "source": "new five-fold GPU training in this run",
            "fold_version": "v1",
            "fold_assignment_sha256": FOLD_SHA,
            "oof_artifact_sha256": file_sha256(out / "neighborhood/oof.parquet"),
            "test_artifact_sha256": file_sha256(
                out / "neighborhood/test_predictions.parquet"
            ),
        },
    ]
    pd.DataFrame(ledger).to_csv(out / "input_provenance.csv", index=False)
    (out / "stacker.json").write_text(json.dumps(state, indent=2, allow_nan=False))
    np.save(out / "frozen_correction.npy", verified["correction"], allow_pickle=False)
    restored = json.loads((out / "stacker.json").read_text())
    again, _ = compose(
        added.test,
        np.array(restored["coefficients"]),
        restored["intercept"],
        np.load(out / "frozen_correction.npy", allow_pickle=False),
    )
    if not np.array_equal(again, prediction):
        raise ValueError("202-view JSON roundtrip mismatch")
    submission = out / "submission.csv"
    pd.DataFrame(
        {
            "id": verified["test"].id,
            "satisfaction": probability(prediction, len(prediction)),
        }
    ).to_csv(submission, index=False)
    frame = pd.read_csv(submission, float_precision="round_trip")
    if not np.array_equal(frame.id, verified["test"].id) or not np.array_equal(
        frame.satisfaction, prediction
    ):
        raise ValueError("Final CSV roundtrip mismatch")
    owned_path = find_data_file(config["reference"]["file"], paths)
    with np.load(owned_path, allow_pickle=False) as source:
        owned = {
            name: source[name + "_oof"].copy()
            for name in config["public_stack"]["own_models"]
        }
    correlations(verified["target"], trained["oof"], owned).to_csv(
        out / "oof_correlations.csv", index=False
    )
    pd.DataFrame({"model": added.names, "coefficient": weights}).to_csv(
        out / "coefficients.csv", index=False
    )
    parent_prediction = pd.read_csv(
        out / "control_exp037.csv", float_precision="round_trip"
    ).satisfaction.to_numpy()
    change = np.abs(prediction - parent_prediction)
    return {
        "status": "completed_exploratory_trained_neighborhood_hybrid",
        "metric": "roc_auc",
        "direction": "maximize",
        "overall_oof_auc": None,
        "per_fold_auc": None,
        "public_score": None,
        "selected": "exploratory_not_promoted",
        "new_base": trained["metrics"],
        "params": SPEC,
        "members": 202,
        "meta_fit_seconds": seconds,
        "submission_path": str(submission),
        "submission_sha256": file_sha256(submission),
        "new_base_coefficient": float(weights[-1]),
        "test_change_vs_parent": {
            "mean_absolute": float(change.mean()),
            "maximum_absolute": float(change.max()),
            "nonidentical_rows": int(np.count_nonzero(change)),
            "interpretation": "unlabeled probability change, not an AUC gain",
        },
        "validation_status": (
            "base OOF is frozen-v1 development; final mixed-public stack CV unavailable"
        ),
    }
