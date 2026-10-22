"""Descriptive research only: no predictor fitting, CV, or submission creation."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient",
           "Ease of Online booking", "Gate location", "Food and drink",
           "Online boarding", "Seat comfort", "Inflight entertainment",
           "On-board service", "Leg room service", "Baggage handling",
           "Checkin service", "Cleanliness"]
CONTEXT = ["Gender", "Customer Type", "Type of Travel", "Class"]
CELLS = ["Customer Type", "Type of Travel", "Class",
         "Inflight wifi service", "Online boarding"]
PROTECTED = ["submission_public_pool_20261020.csv", "CURRENT_BEST_RESULT.md",
             "CURRENT_BEST_SUBMISSION.json", "CURRENT_RECOMMENDED_CANDIDATE.json"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pattern_ids(frame):
    # Base-8 encoding is injective for these 13 integer fields, values 0..5.
    values = frame[RATINGS].to_numpy(dtype=np.int64)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 5)).all()
    return (values * (8 ** np.arange(len(RATINGS), dtype=np.int64))).sum(axis=1)


def cell_table(frame, indices):
    return frame.iloc[indices].groupby(CELLS, observed=True)["_y"].agg(["sum", "count"])


def main():
    started = time.monotonic()
    before = {f: sha(ROOT / f) for f in PROTECTED}
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    original = pd.concat([pd.read_csv(ROOT / "external_data" / f)
                          for f in ["train.csv", "test.csv"]], ignore_index=True)
    features = [c for c in test if c != "id"]
    source_features = [c for c in original if c not in {"id", "satisfaction"}
                       and not c.startswith("Unnamed")]
    extra = sorted(set(source_features) - set(features))
    assert set(features).issubset(source_features)
    # Match the existing source-overlap shield, without importing training modules.
    current_fp = pd.util.hash_pandas_object(
        pd.concat([train[features], test[features]], ignore_index=True).astype("string")
        .fillna("__MISSING__"), index=False)
    source_fp = pd.util.hash_pandas_object(
        original[features].astype("string").fillna("__MISSING__"), index=False)
    keep = ~source_fp.isin(set(current_fp))
    original = original.loc[keep].reset_index(drop=True)
    train["_y"] = train["satisfaction"].astype(int)
    original["_y"] = original["satisfaction"].eq("satisfied").astype(int)

    source_patterns = set(pattern_ids(original))
    group_indices = list(train.groupby(CONTEXT, observed=True).indices.values())
    null_rates = []
    for seed in [101, 202, 303]:
        rng = np.random.default_rng(seed)
        shuffled = train[RATINGS].copy()
        for positions in group_indices:
            for j in range(len(RATINGS)):
                shuffled.iloc[positions, j] = rng.permutation(
                    train.iloc[positions][RATINGS[j]].to_numpy())
        null_rates.append(float(np.mean(np.isin(pattern_ids(shuffled), list(source_patterns)))))
    actual_support = float(np.mean(np.isin(pattern_ids(train), list(source_patterns))))

    # Split both domains independently, then describe replicated cell-level changes.
    rng = np.random.default_rng(58203)
    oi, ci = rng.permutation(len(original)), rng.permutation(len(train))
    pieces = [cell_table(original, oi[:len(oi)//2]), cell_table(original, oi[len(oi)//2:]),
              cell_table(train, ci[:len(ci)//2]), cell_table(train, ci[len(ci)//2:])]
    joint = pd.concat(pieces, axis=1, keys=["Oa", "Ob", "Ca", "Cb"]).dropna()
    eligible = (joint[[('Oa', 'count'), ('Ob', 'count')]].min(axis=1) >= 150) & (
        joint[[('Ca', 'count'), ('Cb', 'count')]].min(axis=1) >= 500)
    table = joint.loc[eligible].copy()
    logits, variances = {}, {}
    for name in ["Oa", "Ob", "Ca", "Cb"]:
        pos = table[(name, "sum")].to_numpy() + .5
        neg = table[(name, "count")].to_numpy() - table[(name, "sum")].to_numpy() + .5
        logits[name] = np.log(pos / neg)
        variances[name] = 1 / pos + 1 / neg
    delta_a, delta_b = logits["Ca"] - logits["Oa"], logits["Cb"] - logits["Ob"]
    pairs = np.triu_indices(len(table), k=1)
    signs, strong = {}, {}
    for name in logits:
        diff = logits[name][pairs[0]] - logits[name][pairs[1]]
        signs[name] = np.sign(diff)
        strong[name] = np.abs(diff) > 1.96 * np.sqrt(
            variances[name][pairs[0]] + variances[name][pairs[1]])
    replicated = (signs["Oa"] == signs["Ob"]) & (signs["Ca"] == signs["Cb"])
    reversal = replicated & (signs["Oa"] != signs["Ca"])
    well_separated = np.logical_and.reduce(list(strong.values()))
    summary = {
        "purpose": "Exploratory descriptive counts only; no fitted predictor or AUC gain evaluation.",
        "rows": {"competition_train": len(train), "competition_test": len(test),
                 "original_after_exact_overlap_shield": len(original),
                 "original_removed_exact_overlap": int((~keep).sum())},
        "shared_feature_count": len(features), "original_extra_predictors": extra,
        "rating_pattern_support": {
            "definition": "Exact injective base-8 encoding of 13 ratings; context not in match key.",
            "source_unique_patterns": len(source_patterns),
            "competition_train_matching_source_fraction": actual_support,
            "conditional_column_shuffle_null_fractions": null_rates,
            "null_preserves": "Each rating's empirical marginal within Gender/Customer Type/Travel/Class; destroys within-row rating dependence.",
            "does_not_establish": "Source-row recovery, actual generator type, residual signal, or improvement over current pool."},
        "replicated_domain_cell_changes": {
            "cell_columns": CELLS, "random_split_seed": 58203,
            "min_source_rows_per_half": 150, "min_competition_rows_per_half": 500,
            "eligible_cells": len(table), "cell_pairs": len(pairs[0]),
            "delta_log_odds_split_correlation": float(np.corrcoef(delta_a, delta_b)[0, 1]),
            "delta_log_odds_half_a_quantiles": np.quantile(delta_a, [.05,.25,.5,.75,.95]).tolist(),
            "replicated_pair_order_reversals": int(reversal.sum()),
            "well_separated_pairs_all_four_halves": int(well_separated.sum()),
            "replicated_reversals_well_separated_all_four_halves": int((reversal & well_separated).sum()),
            "limitations": "Exploratory aggregate-cell rates; not full-vector posteriors, not independent validation, not residual gain. Wald separation is descriptive and not multiple-testing-adjusted. Cell composition differences can produce reversals."},
        "extra_field_source_description": {
            c: {"missing": int(original[c].isna().sum()),
                "unique": sorted(map(float, original[c].dropna().unique())),
                "pearson_correlation_with_source_target": float(original[c].corr(original["_y"]))}
            for c in extra},
        "input_sha256": {f: sha(ROOT / f) for f in ["train.csv", "test.csv", "external_data/train.csv", "external_data/test.csv"]},
        "protected_before": before,
        "elapsed_seconds": time.monotonic() - started,
    }
    after = {f: sha(ROOT / f) for f in PROTECTED}
    assert before == after
    summary["protected_files_unchanged"] = True
    (OUT / "domain_structure_diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"input_sha256", "protected_before"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
