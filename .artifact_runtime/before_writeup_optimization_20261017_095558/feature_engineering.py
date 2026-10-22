"""逐行特征、训练折频次编码与训练折内的低权重合成样本。"""

import numpy as np
import pandas as pd


RATING_COLUMNS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]


def row_features(frame):
    """只使用当前这一行，不读取标签、其他样本或测试集统计量。"""
    result = frame.copy()
    numeric = frame.select_dtypes(include=["number", "bool"])
    result["fe_missing_count"] = frame.isna().sum(axis=1).astype("float32")
    groups = {
        "service": RATING_COLUMNS,
        "digital": ["Inflight wifi service", "Ease of Online booking", "Online boarding"],
        "cabin": ["Food and drink", "Seat comfort", "Inflight entertainment", "Cleanliness", "Leg room service"],
        "ground": ["Gate location", "Baggage handling", "Checkin service"],
    }
    for name, columns in groups.items():
        columns = [col for col in columns if col in numeric]
        if not columns:
            continue
        values = numeric[columns]
        result[f"fe_{name}_mean"] = values.mean(axis=1)
        result[f"fe_{name}_min"] = values.min(axis=1)
        result[f"fe_{name}_max"] = values.max(axis=1)
        result[f"fe_{name}_std"] = values.std(axis=1, ddof=0)
        result[f"fe_{name}_low_count"] = ((values >= 1) & (values <= 2)).sum(axis=1)
        result[f"fe_{name}_high_count"] = (values >= 4).sum(axis=1)
        # 0 可能表示未使用服务，与低评分分别记录。
        result[f"fe_{name}_zero_count"] = (values == 0).sum(axis=1)
    if "fe_digital_mean" in result and "fe_cabin_mean" in result:
        result["fe_digital_cabin_gap"] = result["fe_digital_mean"] - result["fe_cabin_mean"]

    departure, arrival = "Departure Delay in Minutes", "Arrival Delay in Minutes"
    if departure in numeric and arrival in numeric:
        delays = numeric[[departure, arrival]]
        result["fe_delay_change"] = numeric[arrival] - numeric[departure]
        result["fe_delay_total"] = delays.sum(axis=1, min_count=2)
        result["fe_delay_max"] = delays.max(axis=1)
        result["fe_any_delay"] = delays.gt(0).any(axis=1).astype("float32")
        result["fe_arrival_missing"] = numeric[arrival].isna().astype("float32")
        for col, name in [(departure, "departure"), (arrival, "arrival")]:
            result[f"fe_{name}_log"] = np.log1p(numeric[col].clip(lower=0))
        if "Flight Distance" in numeric:
            result["fe_delay_per_1000_distance"] = result["fe_delay_total"] / (numeric["Flight Distance"].clip(lower=1) / 1000)
    if "Age" in numeric:
        result["fe_age_band"] = (numeric["Age"] // 10).astype("Int64").astype("string").fillna("missing")
        result["fe_child"] = (numeric["Age"] < 18).astype("float32")
        result["fe_senior"] = (numeric["Age"] >= 60).astype("float32")
    if "Flight Distance" in numeric:
        result["fe_distance_log"] = np.log1p(numeric["Flight Distance"].clip(lower=0))
        result["fe_distance_band"] = (numeric["Flight Distance"] // 500).astype("Int64").astype("string").fillna("missing")
    for left, right in [("Customer Type", "Type of Travel"), ("Class", "Type of Travel"), ("Class", "Customer Type")]:
        if left in frame and right in frame:
            # 长度前缀避免原始文本中含分隔符时组合值碰撞。
            a = frame[left].astype("string").fillna("missing")
            b = frame[right].astype("string").fillna("missing")
            result[f"fe_cross_{left}_{right}"] = a.str.len().astype(str) + ":" + a + b
    return result


class FoldFeatureEngineer:
    """频次只从真实训练折学习；验证、测试和合成行仅做映射。"""

    def fit(self, frame):
        derived = row_features(frame)
        self.frequencies_ = {
            col: derived[col].astype("string").fillna("__MISSING__").value_counts(normalize=True)
            for col in derived.select_dtypes(exclude=["number", "bool"]).columns
        }
        return self

    def transform(self, frame):
        derived = row_features(frame)
        for col, frequencies in self.frequencies_.items():
            key = derived[col].astype("string").fillna("__MISSING__")
            derived[f"fe_frequency_{col}"] = key.map(frequencies).fillna(0).astype("float32")
        return derived


def synthesize_training_rows(frame, labels, task, ratio, seed):
    """同上下文内选两个不同父样本，仅对连续数值插值。

    二分类父样本标签相同；原始类别与离散评分保留锚点值。
    不接收验证集或测试集，也不接收 ID 或目标特征列。
    回归目标与连续数值使用同一插值系数。返回的父样本位置用于审计。
    """
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("合成比例必须在 [0, 1]。")
    labels = np.asarray(labels)
    count = int(len(frame) * ratio)
    numeric = frame.select_dtypes(include=["number"]).columns
    continuous = [col for col in numeric if col not in RATING_COLUMNS and frame[col].nunique() > 12]
    empty_parents = np.empty((0, 2), dtype=int)
    if count == 0 or not continuous:
        return frame.iloc[:0].copy(), labels[:0].copy(), empty_parents

    context = frame.select_dtypes(exclude=["number", "bool"]).astype("string").fillna("__MISSING__").copy()
    if task == "classification":
        context["_label"] = labels
    for col, width in [("Age", 10), ("Flight Distance", 500)]:
        if col in frame:
            context[f"_bin_{col}"] = (frame[col] // width).fillna(-1)
    if context.shape[1] == 0:
        context["_all"] = 0
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(context), sort=False)
    sizes = np.bincount(codes)
    eligible = np.flatnonzero(sizes[codes] > 1)
    if not len(eligible):
        return frame.iloc[:0].copy(), labels[:0].copy(), empty_parents
    order = np.argsort(codes, kind="stable")
    positions = np.empty(len(frame), dtype=int)
    positions[order] = np.arange(len(frame))
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    rng = np.random.default_rng(seed)
    anchors = rng.choice(eligible, size=count, replace=True)
    group = codes[anchors]
    offset = rng.integers(1, sizes[group])
    donors = order[starts[group] + (positions[anchors] - starts[group] + offset) % sizes[group]]
    alpha = rng.uniform(0.05, 0.25, size=count)
    synthetic = frame.iloc[anchors].reset_index(drop=True).copy()
    for col in continuous:
        left = frame[col].to_numpy(dtype=float)[anchors]
        right = frame[col].to_numpy(dtype=float)[donors]
        observed = np.isfinite(left) & np.isfinite(right)
        mixed = np.where(observed, left + alpha * (right - left), left)
        values = frame[col].dropna().to_numpy()
        if np.equal(values, np.round(values)).all():
            mixed = np.round(mixed)
        synthetic[col] = mixed.astype("float32")
    synthetic_labels = labels[anchors].copy()
    if task == "regression":
        synthetic_labels = (1 - alpha) * labels[anchors] + alpha * labels[donors]
    return synthetic, synthetic_labels, np.column_stack([anchors, donors])
