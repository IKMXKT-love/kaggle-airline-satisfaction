# S6E10 — 18 OOF pairs the public pool was missing (17 members + our frozen 245 stack)

**One line:** out-of-fold + test predictions, in `train.csv` / `test.csv` row order, on the shared
seed-42 folds — drop-in members for any S6E10 stacking notebook, with a model card for every file.

**Who is using it:** this dataset is an input of
[Dariush Afshar's 247-pair stack (LB 0.96187)](https://www.kaggle.com/code/dariushafshar/s6e10-247-public-oofs-one-stack-lb-0-96187),
[ewtherka's family-wise L2 notebooks (LB 0.96188)](https://www.kaggle.com/code/ewtherka/s6e10-0-96188-lb-l2-family-wise-actually-helps)
and [our own series, Parts 9 – 11](https://www.kaggle.com/code/goodpjw2008/s6e10-twin-stacks-family-reg-lb-0-96187),
where the 17 members added +0.000022 nested CV to the full public pool (5/5 folds).

**Load everything in four lines:**
```python
import glob, numpy as np, os
d = "/kaggle/input/s6e10-goodpjw2008-extra-members-oof"
oof  = {os.path.basename(f)[4:-4]: np.load(f) for f in glob.glob(f"{d}/oof_*.npy")}
test = {k: np.load(f"{d}/test_{k}.npy") for k in oof}        # drop 'stack245' unless you read the warning below
```

*If this saved you a training run, an upvote on the dataset helps it reach the people still re-training
these members from scratch. Questions and corrections are welcome in the discussion tab.*

---

# S6E10 — 17 extra members from the goodpjw2008 series (OOF + test)

Out-of-fold and test predictions of 17 models trained during Parts 1–8 of the
[goodpjw2008 S6E10 series](https://www.kaggle.com/code/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160)
that were **never published** before: they did not help our own 48-model stack, but each one
lifts the 211-column public meta stack (nested CV, 4–5 of 5 folds up). Used in Part 9.

- `oof_<name>.npy`: 699,635 probabilities in `train.csv` order. `test_<name>.npy`: 299,844 in `test.csv` order.
- Outer folds: `StratifiedKFold(5, shuffle=True, random_state=42)`; names ending `_k10` use the same call with 10 folds
  (still one out-of-fold prediction per row). Names ending `_k` are the Kaggle run of the Part 6 notebook
  (same recipe as our local member, a second draw because GPU training is not deterministic).
- `members.csv`: OOF AUC and what each adds alone to the 211-column meta stack.

| member | what it is |
|---|---|
| xgb_k10, xgb_d10_k10, lgb_k10, cat_k10 | the Part 1 view-A trees (route profile + original-data model predictions + counts), 10 folds |
| cat_aux | view-A CatBoost + the 30 auxiliary-task features of Part 4 |
| gen_v1 | a *generative* classifier: P(x_j given x_<j, y) column by column, Bayes rule at the end (AUC 0.952, very different errors) |
| realmlp_pub_aux_k, realmlp_fa_aux3_k, realmlp_fa_s7_k | Part 6 RealMLPs (Kaggle run) |
| realmlp_fa_aux_n16, realmlp_fa_auxev, realmlp_v4fa_auxfull | RealMLP variants: 16-member ensemble, expected values only, full auxiliary distributions |
| realmlp_vb | RealMLP on the view-B description (digits, GPT-2 tokens, original target means) |
| tabm_v4aux | TabM on view A + auxiliary features |
| cat_ctr_k, glm6_xgb_k, lgb_v3_k | Part 3 CatBoost CTR, GLM-margin XGBoost, view-A LightGBM (Kaggle run) |

Early-stopping note: the tree members early-stop on the fold they are scored on (the habit sachith7 pointed out);
we measured the optimism at about +0.00005 of single-model OOF AUC and ~0 at the stack level.

**Added for Part 11 (2026-10-12):** `oof_stack245.npy` / `test_stack245.npy` — our full 245-member pool
stack itself as one pair: the nested seed-7 out-of-fold probabilities and the exact full-fit test
probabilities that scored public LB 0.96186 (`meta245.csv`, submitted 10-12). The code that builds the
245 columns and fits the stack is in the Part 10/11 notebook; this frozen copy exists so a Kaggle re-run
can reproduce the scored file bit for bit (platform LR fits differ in the 5th decimal otherwise).

> **Warning (2026-10-13), please read before stacking `oof_stack245` / `test_stack245`:** this pair is a
> *stack-level* OOF. It is published so the scored submission can be reproduced bit for bit and used in
> fixed-weight blends — **not as a member of another stacker.** Although its OOF is honestly nested
> (seed-7 meta folds), inside anyone else's stacker its training-row values encode other folds' labels,
> which inflates nested CV without improving the test file.
> [ewtherka measured this directly](https://www.kaggle.com/code/ewtherka/s6e10-0-96188-lb-l2-family-wise-actually-helps):
> with this pair as a member, nested CV optimism grows to +4 × 1e-5 and vanishes under a fully nested
> rebuild. This is the same caveat our side episode raised about blend-level OOFs, now proven on our own file.


## Added in version 3 (Part 12)

`oof_tabpfn_ne4.npy` / `test_tabpfn_ne4.npy` — TabPFN-3.5 with its OWN 4-estimator ensemble
(n_estimators=4), full 560k-row context, route categories (`raw+catfd+catfd10`), seed 42 folds.
OOF AUC 0.961491 (the strongest honest single model of the series; fold AUCs
0.961768 / 0.960754 / 0.962056 / 0.961684 / 0.961218), public LB 0.96110 as a single model.
Deliberately NOT in `members.csv`: adding it to the 245-member stack changes nested CV by
+-0.000000 (1 of 5 folds up) -- the pool's sixteen n_estimators=1 TabPFN columns, averaged by
the stacker, already act as this ensemble.  Part 12 of the notebook series measures this.
Command: `python src/train_tabpfn.py tabpfn_ne4f nfolds=5 ctx=0 feats=raw+catfd+catfd10 n_est=4`.


## Added in version 4

`twin_submission_1015.csv` — the series' final twin file exactly as submitted on 10-15
(public LB 0.96187): ½ rank(our frozen 245 stack) + ½ rank(Dariush Afshar's 247 stack, his
10-15 run).  Re-hosted because kernel outputs are NOT frozen: his kernel re-ran later and the
live rebuild drifted to LB 0.96186 (rank corr 0.999994 — noise-scale, but a different file).
Lesson: pin by re-hosting, not by linking.


## Added in version 5

`anchor_twin_dp30_1017.csv` — the anchor used by "What 0.96193 Is Made Of": rank space,
(1 - 0.30) * rank(twin_submission_1015) + 0.30 * rank(Den Pugovkin's file), exactly as written to
CSV on 10-17 (public LB 0.96188).  Re-hosted because a CSV round-trip moves values by ~1e-16,
enough to reorder tied ranks; the notebook loads this file so its output is byte-for-byte the
file that scored 0.96191.


## Added in version 6 (Meta Stack part 2, 10-19)

Pinned inputs of the Meta Stack notebook's 0.96197 file (kernel outputs change when their authors
re-run, so dated copies are kept here):
- `mayankraaz_surface_v2_1019.csv` — **mayankraaz's** "LB Surface Ascent v2" output as of 10-19
  (public 0.96196), his work, re-hosted with credit.
- `hanifnoerrofiq_agentic_1019.csv` — **hanifnoerrofiq's** "Agentic Ensemble" output as of 10-19
  (public 0.96198), his work, re-hosted with credit.
- `surface_u_k6.npy` — our ascent direction from the quadratic public-LB surface (his model) refit on
  46 points including 22 of our own measured submissions.
- `surface_step2_1019.csv` — his v2 file + 2 units along our direction (public 0.96196), exactly as
  written to CSV.
- `stack182_lr_lgb_test.npz` — test logits of our LR and LGB stackers on the 182-member public pool.


## Added in version 7 (Twin Stacks Part 13, 10-20)

- `series_stacks_p1_p8.npz` — the 16 stacks we submitted during Parts 1 – 8 of the series
  (`ens_v3` … `ens_v18`): for each, `oof_<name>` (699,635 out-of-fold stack predictions, train order)
  and `test_<name>` (299,844, test order, exactly the submitted file), float32, plus `names` and
  `public` (the public LB each one scored). With them, every public step of our series can be re-read
  against its CV and against the public board's own sampling noise (Part 13 does this: the rule
  *public sd ≈ 0.003 × std of the percentile-rank difference of the two files* matches the
  measured sd at a median ratio of 0.99 over these 15 steps).
