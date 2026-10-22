"""TabPFN-3.5 (Prior Labs' tabular foundation model) as a stack member.

Idea from hemingweb's public notebook "S6E10 EXP01 TabPFN35 Full Context" and the S6E9 2nd-place
write-up it cites: an in-context model whose AUC keeps growing with the size of the labelled
context.  On this machine the whole context of a fold (~560k rows) fits in unified memory, so no
subsampling is needed; `ctx=<n>` subsamples it for quick probes.

Usage: python src/train_tabpfn.py <name> [nfolds=1 ctx=0 seed=0 batch=32768 feats=raw|aux]
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
    # batch: rows predicted per call.  Prediction, not fitting, is the memory peak with a 560k-row context
    # (10-03: a run grew past 100 GB at the first prediction batch), so keep it modest.
    # pl=<teacher prefix>: add confidently predicted TEST rows (|p - 0.5| > plthr - 0.5) to the context with their
    # 0/1 pseudo-labels, using pred/<pl><fold>.npy from a teacher that never saw the fold (teacher_folds.py).
    P = dict(nfolds=1, ctx=0, seed=0, batch=8192, feats="raw", kv="int8", ogctx=0, model="v35", n_est=1, pl="", plthr=0.95)
    # ogctx=1: add the original rows to the context | model=auto: the package's default checkpoint (TabPFN-2.5)
    # n_est: TabPFN's own ensemble size (different preprocessing / feature orders per estimator)
    P.update({k: (v if k in ("feats", "kv", "model", "pl") else float(v) if k == "plthr" else int(v))
              for k, v in (a.split("=") for a in sys.argv[2:])})
    if P["pl"] and P["nfolds"] == 5:
        # test rows inside the context would be predicted by copying their own pseudo-label, unlike the OOF rows
        raise SystemExit("pl= is for single-fold probes only (a 5-fold member needs a split-test scheme)")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from huggingface_hub import hf_hub_download
    from tabpfn import TabPFNClassifier
    # model=fast: the smaller TabPFN-3.5 "fast" checkpoint of the same repository (seen in a top-10 team's public
    # notebook): other weights, so another member rather than another seed
    ckpt_file = "tabpfn-v3.5-fast-20260909.safetensors" if P["model"] == "fast" else "tabpfn-v3.5-20260909.safetensors"
    ckpt = ("auto" if P["model"] == "auto" else
            hf_hub_download("Prior-Labs/tabpfn_3_5", ckpt_file, local_dir=os.path.join("cache", "tabpfn_ckpt")))
    tr, te, og = load()
    y = tr[TARGET].values
    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
    for c in CATS:
        full[c] = full[c].astype("category").cat.codes
    full["Arrival Delay in Minutes"] = full["Arrival Delay in Minutes"].fillna(-1)
    parts = P["feats"].split("+")              # e.g. raw, raw+og, raw+te, raw+og+te
    if "aux" in parts:
        af = pd.read_pickle(os.path.join("cache", "aux_feats.pkl"))
        full = pd.concat([full, af[[c for c in af if c.startswith("aux_ev_")]]], axis=1)
    if "og" in parts:
        # the original-data models of Part 1, as logits
        import og_features
        ogf = og_features.get()
        for c in ("og_xgb_d8", "og_realmlp"):
            q = np.clip(ogf[c].values, 1e-6, 1 - 1e-6)
            full[c] = np.log(q / (1 - q)).astype(np.float32)
    if "cnt" in parts:
        full["cnt_fd"] = full["Flight Distance"].map(full["Flight Distance"].value_counts()).astype(np.float32)
    te_keys = {}
    if "te" in parts:
        # Flight Distance as a route ID: fold-safe target encodings (computed inside each fold below)
        import features
        te_keys = {"te_fd": features.combine_keys(full, ["Flight Distance"]),
                   "te_fd_cls": features.combine_keys(full, ["Flight Distance", "Class"])}
    if "te6" in parts:
        # the six target-encoding keys of view A (Part 1): route, age, route x age / class / travel, segment
        import features
        te_keys = {"te_" + "|".join(c[:6] for c in cols): features.combine_keys(full, cols) for cols in features.TE_KEYS["nums"]}
    if "cnt4" in parts:
        for c in ("Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"):
            full["cnt_" + c[:3]] = full[c].map(full[c].value_counts()).fillna(0).astype(np.float32)
    cat_idx = [full.columns.get_loc(c) for c in CATS]
    # catfd: Flight Distance declared categorical (the route ID in TabPFN's own encoding);
    # catrat: the 13 ratings declared categorical (idea: Liam O'Brien's notebook "Should Ratings Be Categories?")
    from common import RATINGS
    if "catfd" in parts:
        cat_idx.append(full.columns.get_loc("Flight Distance"))
    if "catrat" in parts:
        cat_idx += [full.columns.get_loc(c) for c in RATINGS]
    if "catage" in parts:
        cat_idx.append(full.columns.get_loc("Age"))
    if "ragg" in parts:
        # rating summaries (as in the same top-10 notebook): mean, spread, min, max, zero count, in-flight / pre-flight means
        from common import RATINGS as _R
        r = full[_R]
        full["r_mean"], full["r_std"], full["r_min"], full["r_max"] = r.mean(1), r.std(1), r.min(1), r.max(1)
        full["r_zero"] = (r == 0).sum(1)
        full["r_inflight"] = full[["Inflight wifi service", "Food and drink", "Seat comfort", "Inflight entertainment", "On-board service"]].mean(1)
        full["r_preflight"] = full[["Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Online boarding", "Checkin service"]].mean(1)
    if "catfd10" in parts:
        # a coarser route group next to the exact route: distance // 10, categorical
        full["fd10"] = (full["Flight Distance"] // 10).astype(np.float32)
        cat_idx.append(full.columns.get_loc("fd10"))
    for prt in parts:
        # catfdqK: one more coarse route group, distance // K, categorical (catfd10 / catfd100 are the K = 10 / 100 cases)
        if prt.startswith("catfdq"):
            k = int(prt[6:])
            full[f"fdq{k}"] = (full["Flight Distance"] // k).astype(np.float32)
            cat_idx.append(full.columns.get_loc(f"fdq{k}"))
    if "catfd100" in parts:
        full["fd100"] = (full["Flight Distance"] // 100).astype(np.float32)
        cat_idx.append(full.columns.get_loc("fd100"))
    X = full.values.astype(np.float32)
    Xtr, Xte = X[:len(tr)], X[len(tr):]
    if P["ogctx"]:
        O = og[FEATS].copy()
        for c in CATS:   # same category codes as the competition rows (same sorted vocabulary)
            O[c] = pd.Categorical(O[c], categories=sorted(pd.concat([tr[c], te[c]]).unique())).codes
        O["Arrival Delay in Minutes"] = O["Arrival Delay in Minutes"].fillna(-1)
        Xog, yog = O[full.columns[:len(FEATS)]].values.astype(np.float32), og[TARGET].values
    oof, pred, t0 = np.zeros(len(tr)), np.zeros(len(te)), time.time()
    folds = get_folds(y, 5)[:P["nfolds"]]
    # each finished fold is saved, so a reboot does not lose it (the machine restarted several times on 10-03)
    part_dir = os.path.join("cache", "partial")
    os.makedirs(part_dir, exist_ok=True)
    for f, (a, b) in enumerate(folds):
        part = os.path.join(part_dir, f"{name}_f{f}.npz")
        if P["nfolds"] == 5 and os.path.exists(part):
            d = np.load(part)
            oof[b], pred = d["oof"], pred + d["test"] / 5
            print(f"  fold {f} restored from {part}: auc {roc_auc_score(y[b], oof[b]):.6f}", flush=True)
            continue
        if P["ctx"]:
            a = np.random.RandomState(P["seed"]).choice(a, P["ctx"], replace=False)
        if te_keys:
            import features
            base = pd.DataFrame(index=range(len(full)))
            Ta, (Tb, Tt) = features.add_te(base.iloc[a], y[a], [base.iloc[b], base.iloc[len(tr):]], te_keys)
            add = lambda A, T: np.column_stack([A, T.values.astype(np.float32)])
            Xa_, Xb_, Xt_ = add(Xtr[a], Ta), add(Xtr[b], Tb), add(Xte, Tt)
        else:
            Xa_, Xb_, Xt_ = Xtr[a], Xtr[b], Xte
        clf = TabPFNClassifier(model_path=ckpt, n_estimators=P["n_est"], random_state=P["seed"], device="cuda",
                               ignore_pretraining_limits=True, fit_mode="fit_with_cache",
                               kv_cache_precision=P["kv"], inference_precision="autocast",
                               categorical_features_indices=cat_idx, memory_saving_mode="auto")
        if P["ogctx"]:
            clf.fit(np.vstack([Xtr[a], Xog]), np.r_[y[a], yog])
        else:
            if P["pl"]:
                pt_teacher = np.load(os.path.join("pred", f"{P['pl']}{f}.npy"))
                keep = (pt_teacher > P["plthr"]) | (pt_teacher < 1 - P["plthr"])
                clf.fit(np.vstack([Xa_, Xt_[keep]]), np.r_[y[a], (pt_teacher[keep] > 0.5).astype(int)])
                print(f"    context + {keep.sum():,} pseudo-labelled test rows from {P['pl']}{f}", flush=True)
            else:
                clf.fit(Xa_, y[a])
        print(f"  fold {f}: context {len(a) + (len(Xog) if P['ogctx'] else 0):,} rows, fit {time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)

        def predict(Xq, tag):
            out = np.empty(len(Xq), dtype=np.float32)
            t = time.time()
            torch.cuda.reset_peak_memory_stats()
            for i in range(0, len(Xq), P["batch"]):
                out[i:i + P["batch"]] = clf.predict_proba(Xq[i:i + P["batch"]])[:, 1]
                if i == 0:
                    print(f"    {tag}: prediction peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                    print(f"    {tag}: first batch of {min(P['batch'], len(Xq)):,} rows in {time.time() - t:.0f}s "
                          f"-> ETA {(time.time() - t) * len(Xq) / min(P['batch'], len(Xq)) / 60:.0f} min", flush=True)
            return out
        oof[b] = predict(Xb_, "val")
        print(f"  fold {f} auc {roc_auc_score(y[b], oof[b]):.6f} ({time.time() - t0:.0f}s)", flush=True)
        if P["nfolds"] == 5:
            pt = predict(Xt_, "test")
            pred += pt / 5
            np.savez(part, oof=oof[b], test=pt)
    if P["nfolds"] == 5:
        save(name, oof, pred, y, extra=f"tabpfn-3.5 P={P}")


if __name__ == "__main__":
    main()
