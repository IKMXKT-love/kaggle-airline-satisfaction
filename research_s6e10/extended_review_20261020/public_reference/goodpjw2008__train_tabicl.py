"""TabICL v2 (soda-inria, BSD-3) as a stack member: a second tabular foundation model.

Same use as TabPFN-3.5 in train_tabpfn.py: no training on this data, the labelled rows outside the
fold are read as context.  TabICL was built for larger tables (column-then-row attention), so it
is a natural second in-context member next to TabPFN.

Usage: python src/train_tabicl.py <name> [nfolds=1 ctx=0 seed=0 n_est=1 batch=65536 kv=1]
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from common import CATS, FEATS, TARGET, get_folds, load, save

warnings.filterwarnings("ignore")
# the GB10 shares 121 GB between GPU and CPU: cap this process at 40% so an oversized allocation
# fails here with a CUDA OOM error instead of stalling the whole machine (two freezes on 10-03)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_MEM_FRACTION", 0.4)))


def main():
    name = sys.argv[1]
    P = dict(nfolds=1, ctx=0, seed=0, n_est=1, batch=65536, kv=1)
    P.update({k: int(v) for k, v in (a.split("=") for a in sys.argv[2:])})
    from tabicl import TabICLClassifier
    tr, te, _ = load()
    y = tr[TARGET].values
    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
    for c in CATS:
        full[c] = full[c].astype("category").cat.codes
    full["Arrival Delay in Minutes"] = full["Arrival Delay in Minutes"].fillna(-1)
    X = full.values.astype(np.float32)
    Xtr, Xte = X[:len(tr)], X[len(tr):]
    oof, pred, t0 = np.zeros(len(tr)), np.zeros(len(te)), time.time()
    for f, (a, b) in enumerate(get_folds(y, 5)[:P["nfolds"]]):
        if P["ctx"]:
            a = np.random.RandomState(P["seed"]).choice(a, P["ctx"], replace=False)
        clf = TabICLClassifier(n_estimators=P["n_est"], random_state=P["seed"], device="cuda", kv_cache=bool(P["kv"]))
        clf.fit(Xtr[a], y[a])
        torch.cuda.reset_peak_memory_stats()

        def predict(Xq, tag):
            out = np.empty(len(Xq), dtype=np.float32)
            t = time.time()
            for i in range(0, len(Xq), P["batch"]):
                out[i:i + P["batch"]] = clf.predict_proba(Xq[i:i + P["batch"]])[:, 1]
                if i == 0:
                    n = min(P["batch"], len(Xq))
                    print(f"    {tag}: first {n:,} rows in {time.time() - t:.0f}s -> ETA {(time.time() - t) * len(Xq) / n / 60:.0f} min "
                          f"(peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
            return out
        oof[b] = predict(Xtr[b], "val")
        print(f"  fold {f}: context {len(a):,} | auc {roc_auc_score(y[b], oof[b]):.6f} ({time.time() - t0:.0f}s)", flush=True)
        if P["nfolds"] == 5:
            pred += predict(Xte, "test") / 5
    if P["nfolds"] == 5:
        save(name, oof, pred, y, extra=f"tabicl-v2 P={P}")


if __name__ == "__main__":
    main()
