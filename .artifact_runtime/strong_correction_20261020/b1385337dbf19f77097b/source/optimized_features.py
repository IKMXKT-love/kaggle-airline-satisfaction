"""借鉴竞赛方案：原始特征、折内目标编码、原始数据统计与先验预测。"""

import numpy as np
import pandas as pd
from importlib.metadata import version
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder


CONTEXT_COLUMNS = ["Gender", "Customer Type", "Type of Travel", "Class"]
PAIR_COLUMNS = [
    ("Online boarding", "Class"), ("Online boarding", "Type of Travel"),
    ("Inflight wifi service", "Class"), ("Inflight wifi service", "Type of Travel"),
    ("Seat comfort", "Class"), ("Inflight entertainment", "Class"),
    ("Customer Type", "Type of Travel"), ("Class", "Type of Travel"),
    ("Inflight wifi service", "Online boarding"),
]


def normalize_raw(frame):
    result = frame.copy()
    for col in result:
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = pd.to_numeric(result[col]).astype("float32").replace([np.inf, -np.inf], np.nan)
        else:
            result[col] = result[col].astype("string").fillna("__MISSING__").astype(str)
    return result


def target_keys(frame, extended=False):
    """无标签的离散键；保留原始取值，加入小规模关键交互和连续变量分箱。"""
    keys = frame.astype("string").fillna("__MISSING__").copy()
    for left, right in PAIR_COLUMNS:
        if left in keys and right in keys:
            a, b = keys[left], keys[right]
            keys[f"pair__{left}__{right}"] = a.str.len().astype(str) + ":" + a + b
    for col, widths in [("Age", [5, 10]), ("Flight Distance", [100, 500]),
                         ("Departure Delay in Minutes", [5, 30]), ("Arrival Delay in Minutes", [5, 30])]:
        if col in frame:
            for width in widths:
                keys[f"bin__{col}__{width}"] = (frame[col] // width).astype("Int64").astype("string").fillna("__MISSING__")
    if extended:
        groups = [
            ["Inflight wifi service", "Online boarding", context]
            for context in ["Customer Type", "Type of Travel", "Class"]
        ] + [
            ["Customer Type", "Type of Travel", "Class"],
            ["Seat comfort", "Inflight entertainment", "Class"],
            ["Seat comfort", "Cleanliness", "Food and drink"],
            ["Leg room service", "On-board service", "Baggage handling"],
            ["Inflight entertainment", "Cleanliness"],
            ["Inflight entertainment", "Seat comfort"],
            ["Online boarding", "Seat comfort"],
        ]
        for group in groups:
            if all(col in keys for col in group):
                value = pd.Series("", index=frame.index, dtype="string")
                for col in group:
                    value = value + keys[col].str.len().astype(str) + ":" + keys[col]
                keys["multi__" + "__".join(group)] = value
    return keys


class NativeCategoryEncoder:
    """密集数值保留 0 与 NaN 的区别，低基数文本使用 XGBoost 原生类别。"""

    def fit(self, frame):
        self.columns_ = list(frame.columns)
        self.categories_ = {
            col: sorted(frame[col].astype(str).unique().tolist())
            for col in frame.select_dtypes(exclude=["number", "bool"]).columns
        }
        return self

    def transform(self, frame):
        if list(frame.columns) != self.columns_:
            raise ValueError("训练与预测特征顺序不一致。")
        result = frame.copy()
        for col in result:
            if col in self.categories_:
                values = result[col].astype("string")
                values = values.where(values.isin(self.categories_[col]))
                result[col] = pd.Categorical(values, categories=self.categories_[col])
            else:
                result[col] = result[col].astype("float32")
        return result


class OriginalStatistics:
    """仅从公开原始数据的标签学习，与比赛训练/验证标签无关。"""

    def __init__(self, extended=False):
        self.extended = extended

    def fit(self, frame, labels):
        labels = np.asarray(labels, dtype=float)
        self.prior_ = float(labels.mean())
        self.tables_ = {}
        keys = target_keys(frame, extended=self.extended)
        for col in keys:
            table = pd.DataFrame({"key": keys[col].to_numpy(), "y": labels}).groupby("key").y.agg(["sum", "count"])
            table["mean"] = (table["sum"] + 10 * self.prior_) / (table["count"] + 10)
            self.tables_[col] = table
        return self

    def transform(self, frame):
        keys = target_keys(frame, extended=self.extended)
        result = {}
        for col, table in self.tables_.items():
            result[f"orig_mean__{col}"] = keys[col].map(table["mean"]).fillna(self.prior_).to_numpy(dtype="float32")
            result[f"orig_count__{col}"] = np.log1p(keys[col].map(table["count"]).fillna(0)).to_numpy(dtype="float32")
        return pd.DataFrame(result, index=frame.index)


class CompetitionFeaturePipeline:
    """训练用内层 5 折目标编码，外层验证和测试仅 transform。"""

    def __init__(self, use_te=False, original_statistics=None, numeric_categories=False, digits=False,
                 extended_keys=False):
        self.use_te = use_te
        self.original_statistics = original_statistics
        self.numeric_categories = numeric_categories
        self.digits = digits
        self.extended_keys = extended_keys

    def _base(self, frame, original_prediction=None):
        result = normalize_raw(frame)
        raw_numeric = result.select_dtypes(include=["number"]).columns.tolist()
        if self.numeric_categories:
            for col in raw_numeric:
                result[f"raw_category__{col}"] = result[col].astype("string").fillna("__MISSING__")
        if self.digits:
            for col in ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
                if col in result:
                    for divisor in [5, 10, 100]:
                        result[f"digit__{col}__{divisor}"] = result[col] % divisor
                        result[f"round__{col}__{divisor}"] = result[col] // divisor
        if self.original_statistics is not None:
            result = pd.concat([result, self.original_statistics.transform(result)], axis=1)
        if original_prediction is not None:
            values = np.asarray(original_prediction, dtype="float32")
            if len(values) != len(result):
                raise ValueError("原始数据先验预测行数不匹配。")
            if values.ndim == 1:
                result["orig_model_probability"] = values
            elif values.ndim == 2:
                for index in range(values.shape[1]):
                    result[f"orig_model_probability_{index}"] = values[:, index]
            else:
                raise ValueError("先验预测必须为一维或二维数组。")
        return result

    def fit_transform(self, frame, labels, original_prediction=None):
        result = self._base(frame, original_prediction)
        if self.use_te:
            if tuple(int(part) for part in version("scikit-learn").split(".")[:2]) >= (1, 9):
                cv_args = {"cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42)}
            else:
                cv_args = {"cv": 5, "shuffle": True, "random_state": 42}
            self.target_encoder_ = TargetEncoder(target_type="binary", smooth=20.0, **cv_args)
            keys = target_keys(normalize_raw(frame), extended=self.extended_keys)
            # 必须使用 fit_transform 的内部交叉拟合，不能 fit 后直接 transform 训练集。
            encoded = self.target_encoder_.fit_transform(keys, np.asarray(labels))
            self.te_columns_ = [f"cv_te__{col}" for col in keys]
            result = pd.concat([result, pd.DataFrame(encoded.astype("float32"),
                                                     columns=self.te_columns_, index=frame.index)], axis=1)
        self.native_encoder_ = NativeCategoryEncoder().fit(result)
        return self.native_encoder_.transform(result)

    def transform(self, frame, original_prediction=None):
        result = self._base(frame, original_prediction)
        if self.use_te:
            keys = target_keys(normalize_raw(frame), extended=self.extended_keys)
            encoded = self.target_encoder_.transform(keys)
            result = pd.concat([result, pd.DataFrame(encoded.astype("float32"),
                                                     columns=self.te_columns_, index=frame.index)], axis=1)
        return self.native_encoder_.transform(result)
