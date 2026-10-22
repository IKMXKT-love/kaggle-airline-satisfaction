"""A fixed ModernNCA-inspired neural metric with a shared parametric head.

Original implementation: sampled training neighborhoods and exact streamed
full-training-bank inference. No external model weights or retrieval packages.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RATINGS = (
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
)
NUMERIC = (
    "Age",
    "Flight Distance",
    *RATINGS,
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
)
CATEGORICAL = ("Gender", "Customer Type", "Type of Travel", "Class")
RAW = (*NUMERIC, *CATEGORICAL)
PARAMS = {
    "version": "hybrid-neighborhood-plr-v1",
    "epochs": 12,
    "batch_size": 512,
    "context_rows": 2048,
    "frequencies": 8,
    "frequency_scale": 2.0,
    "category_dim": 8,
    "width": 256,
    "metric_dim": 64,
    "dropout": 0.1,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "temperature": 1.0,
    "nca_loss_weight": 0.5,
    "retrieval_prediction_weight": 0.5,
    "gradient_clip": 1.0,
    "encode_batch": 4096,
    "query_batch": 256,
    "bank_chunk": 32768,
    "precision": "float32_no_amp",
    "seed": 2026,
}


def array_sha(value: np.ndarray) -> str:
    """Stable ordered numeric-array identity."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_raw(frame: pd.DataFrame) -> None:
    if list(frame.columns) != list(RAW):
        raise ValueError(
            "Neighborhood inputs must be the declared 21 raw features; no ID/target"
        )
    for column in NUMERIC:
        if np.isinf(pd.to_numeric(frame[column], errors="raise")).any():
            raise ValueError("Infinite numeric feature: " + column)


def tokens(values: pd.Series) -> np.ndarray:
    """A missing-safe representation for original categories and numeric twins."""
    if pd.api.types.is_numeric_dtype(values):
        return np.array(
            ["<NA>" if pd.isna(x) else format(float(x), ".15g") for x in values]
        )
    return values.astype("string").fillna("<NA>").to_numpy(dtype=str)


def fit_context(train: pd.DataFrame) -> dict[str, Any]:
    """Fit medians, IQRs and category vocabularies on training rows only."""
    validate_raw(train)
    values = train[list(NUMERIC)].to_numpy(np.float64)
    medians = np.nanmedian(values, axis=0)
    spread = np.nanquantile(values, 0.75, axis=0) - np.nanquantile(values, 0.25, axis=0)
    if not np.isfinite(medians).all() or not np.isfinite(spread).all():
        raise ValueError("Entirely missing numeric feature")
    categories = {}
    for column in (*CATEGORICAL, *NUMERIC):
        categories[column] = sorted(set(tokens(train[column])) - {"<NA>"})
    return {
        "version": PARAMS["version"],
        "rows": len(train),
        "medians": medians.tolist(),
        "scale": np.maximum(spread, 1.0).tolist(),
        "categories": categories,
        "numeric_clip": 10.0,
    }


def transform(
    frame: pd.DataFrame, context: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    validate_raw(frame)
    value = frame[list(NUMERIC)].to_numpy(np.float64)
    missing = np.isnan(value)
    filled = np.where(missing, np.asarray(context["medians"]), value)
    numeric = np.clip((filled - context["medians"]) / context["scale"], -10.0, 10.0)
    numeric = np.concatenate([numeric, missing.astype(float)], axis=1).astype(
        np.float32
    )
    categorical = []
    for column in (*CATEGORICAL, *NUMERIC):
        vocabulary = {v: i + 1 for i, v in enumerate(context["categories"][column])}
        categorical.append(
            np.fromiter(
                (vocabulary.get(v, 0) for v in tokens(frame[column])),
                dtype=np.int64,
                count=len(frame),
            )
        )
    return numeric, np.column_stack(categorical)


def make_network(torch: Any, context: dict[str, Any]) -> Any:
    """One encoder, learned periodic numeric representations, two prediction paths."""
    nn = torch.nn

    class NeighborhoodNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.frequencies = nn.Parameter(
                torch.randn(len(NUMERIC), PARAMS["frequencies"])
                * PARAMS["frequency_scale"]
            )
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(
                        len(context["categories"][c]) + 1,
                        PARAMS["category_dim"],
                        padding_idx=0,
                    )
                    for c in (*CATEGORICAL, *NUMERIC)
                ]
            )
            inputs = (
                len(NUMERIC) * (2 * PARAMS["frequencies"] + 2)
                + len(self.embeddings) * PARAMS["category_dim"]
            )
            width = PARAMS["width"]
            self.encoder = nn.Sequential(
                nn.Linear(inputs, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(PARAMS["dropout"]),
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(PARAMS["dropout"]),
            )
            self.metric = nn.Linear(width, PARAMS["metric_dim"])
            self.head = nn.Linear(width, 1)

        def forward(self, numeric, categorical):
            angles = (
                2 * math.pi * numeric[:, : len(NUMERIC), None] * self.frequencies[None]
            )
            periodic = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)
            categories = torch.cat(
                [
                    embedding(categorical[:, i])
                    for i, embedding in enumerate(self.embeddings)
                ],
                dim=1,
            )
            hidden = self.encoder(torch.cat([periodic, numeric, categories], dim=1))
            return self.metric(hidden), self.head(hidden).squeeze(-1)

    return NeighborhoodNet()


def class_logs(
    torch: Any, query: Any, keys: Any, labels: Any, mask: Any = None
) -> tuple[Any, Any]:
    """The query norm cancels; never materialize an all-row distance matrix."""
    scores = (2 * query @ keys.T - keys.square().sum(1)[None]) / PARAMS["temperature"]
    if mask is not None:
        scores = scores.masked_fill(mask, -torch.inf)
    zero = torch.logsumexp(scores.masked_fill(labels[None] != 0, -torch.inf), dim=1)
    one = torch.logsumexp(scores.masked_fill(labels[None] != 1, -torch.inf), dim=1)
    return zero, one


def encode(torch: Any, model: Any, numeric: Any, categorical: Any) -> tuple[Any, Any]:
    model.eval()
    keys, heads = [], []
    with torch.inference_mode():
        for start in range(0, len(numeric), PARAMS["encode_batch"]):
            end = start + PARAMS["encode_batch"]
            z, h = model(numeric[start:end], categorical[start:end])
            keys.append(z)
            heads.append(h)
        return torch.cat(keys), torch.cat(heads)


def stream_predict(torch: Any, query: Any, bank: Any, labels: Any) -> np.ndarray:
    """Exact soft neighbors over every training key, with bounded memory."""
    result = []
    with torch.inference_mode():
        for start in range(0, len(query), PARAMS["query_batch"]):
            q = query[start : start + PARAMS["query_batch"]]
            zero = torch.full(
                (len(q),), -torch.inf, device=q.device, dtype=torch.float64
            )
            one = zero.clone()
            for offset in range(0, len(bank), PARAMS["bank_chunk"]):
                stop = offset + PARAMS["bank_chunk"]
                a, b = class_logs(torch, q, bank[offset:stop], labels[offset:stop])
                zero = torch.logaddexp(zero, a.double())
                one = torch.logaddexp(one, b.double())
            result.append(torch.sigmoid(one - zero).cpu().numpy())
    value = np.concatenate(result)
    if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("Invalid full-bank neighbor predictions")
    return value


def fit_fold(
    training: pd.DataFrame,
    target: np.ndarray,
    train_ids: np.ndarray,
    queries: list[tuple[pd.DataFrame, np.ndarray]],
    *,
    seed: int,
    output: Path,
    runtime_name: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Fit on Kaggle/Colab CUDA; validation/test targets are never accepted."""
    if runtime_name not in {"kaggle", "colab"}:
        raise RuntimeError(
            "Neighborhood training is remote-only; local model fitting is disabled"
        )
    if (
        len(training) != len(target)
        or len(target) != len(train_ids)
        or set(np.unique(target)) != {0, 1}
    ):
        raise ValueError("Invalid aligned training bank")
    if not pd.Index(train_ids).is_unique or np.bincount(target.astype(int)).min() < 2:
        raise ValueError(
            "Training keys must be unique with at least two rows per class"
        )
    for _, ids in queries:
        if np.intersect1d(train_ids, ids).size:
            raise ValueError(
                "Validation/test query IDs overlap the labeled training bank"
            )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU; no CPU fallback")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(0)
    context = fit_context(training)
    xn, xc = transform(training, context)
    numeric = torch.as_tensor(xn, device=device)
    categorical = torch.as_tensor(xc, device=device)
    labels = torch.as_tensor(target, dtype=torch.long, device=device)
    model = make_network(torch, context).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PARAMS["learning_rate"],
        weight_decay=PARAMS["weight_decay"],
    )
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    history = []
    for epoch in range(PARAMS["epochs"]):
        model.train()
        total = 0.0
        steps = 0
        order = rng.permutation(len(training))
        for start in range(0, len(order), PARAMS["batch_size"]):
            q = order[start : start + PARAMS["batch_size"]]
            c = rng.choice(
                len(training), min(PARAMS["context_rows"], len(training)), replace=False
            )
            qi = torch.as_tensor(q, device=device)
            ci = torch.as_tensor(c, device=device)
            z, head = model(numeric[qi], categorical[qi])
            keys, _ = model(numeric[ci], categorical[ci])
            zero, one = class_logs(torch, z, keys, labels[ci], qi[:, None] == ci[None])
            total_log = torch.logaddexp(zero, one)
            nca = (total_log - torch.where(labels[qi] == 1, one, zero)).mean()
            point = torch.nn.functional.binary_cross_entropy_with_logits(
                head, labels[qi].float()
            )
            loss = 0.5 * nca + 0.5 * point
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neighborhood loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), PARAMS["gradient_clip"], error_if_nonfinite=True
            )
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite neighborhood gradient")
            optimizer.step()
            total += float(loss.detach()) * len(q)
            steps += 1
        item = {
            "epoch": epoch + 1,
            "training_loss": total / len(training),
            "updates": steps,
        }
        history.append(item)
        print(item, flush=True)
    torch.cuda.synchronize()
    fit_seconds = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        output / "encoder.pt",
    )
    restored = make_network(torch, context).to(device)
    restored.load_state_dict(
        torch.load(output / "encoder.pt", map_location=device, weights_only=True)
    )
    model.eval()
    restored.eval()
    with torch.inference_mode():
        for a, b in zip(
            model(numeric[:8], categorical[:8]),
            restored(numeric[:8], categorical[:8]),
            strict=True,
        ):
            if not torch.equal(a, b):
                raise ValueError("Native tensor roundtrip mismatch")
    del restored
    inference_started = time.monotonic()
    bank, _ = encode(torch, model, numeric, categorical)
    predictions = []
    arms = []
    for frame, _ in queries:
        a, b = transform(frame, context)
        q, head = encode(
            torch,
            model,
            torch.as_tensor(a, device=device),
            torch.as_tensor(b, device=device),
        )
        neighbor = stream_predict(torch, q, bank, labels)
        parametric = torch.sigmoid(head).detach().cpu().numpy().astype(np.float64)
        predictions.append(0.5 * neighbor + 0.5 * parametric)
        arms.append({"head": parametric, "neighbor": neighbor})
        del q, head
    metadata = {
        "params": PARAMS,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "inference_seconds": time.monotonic() - inference_started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(0),
        "parameters": sum(p.numel() for p in model.parameters()),
        "bank_rows": len(training),
        "training_class_counts": np.bincount(target.astype(int), minlength=2).tolist(),
        "training_label_sha256": array_sha(target.astype(np.int8)),
        "bank_train_ids_sha256": array_sha(train_ids.astype("<i8")),
        "history": history,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "native_roundtrip_exact": True,
        "bank_is_training_only": True,
        "self_matches_masked_in_training": True,
    }
    (output / "context.json").write_text(
        json.dumps(context, sort_keys=True, allow_nan=False)
    )
    (output / "training.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    del model, optimizer, bank, numeric, categorical, labels
    torch.cuda.empty_cache()
    return predictions, {**metadata, "arms": arms}
