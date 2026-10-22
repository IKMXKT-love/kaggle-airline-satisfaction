# S6E10 OOF library: 21 models and a 247-pair stack, on frozen folds

**The 10-second answer.** Out-of-fold (OOF) and test predictions for 21 single models I trained for Kaggle Playground S6E10 (airline satisfaction, ROC AUC), all on one frozen 5-fold split, plus one stack member. `oof_wide.parquet` holds `id`, `fold` and one column per single model; `test_wide.parquet` holds `id` and the matching test predictions. The best single model is **q** (`q_lgbm_auxev_10fold`), OOF AUC **0.961388**. A plain logistic stack of all 21 scores **0.961860** nested on these folds, which is what the first snippet below prints; the four members added in version 6 are worth +0.000058 of that over the 17 of version 5 (SE 0.000016, 5/5 folds), a real gain.

**The stack member.** `stack247_oof.npy` and `stack247_test.npy` are the nested OOF and test predictions of my stack of 247 public OOF pairs (notebook "S6E10 | 247 Public OOFs, One Stack | LB 0.96187"): nested CV **0.962205**, public LB **0.96187**. Stack it only with models that were not among its 247 inputs. The version-6 members never were: next to them it scores 0.962198, -0.000003 over `stack247` alone (SE 0.000002, 2/5 folds), not a real gain. Re-stacked with its own inputs (the 17 version-5 members are all among them) it moves by -0.000022 (SE 0.000006, 0/5 folds positive). That is why it is kept out of `manifest.csv` and the two parquet files.

```python
import glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
find = lambda f: glob.glob(f"/kaggle/input/**/{f}", recursive=True)[0]
oof, test = pd.read_parquet(find("oof_wide.parquet")), pd.read_parquet(find("test_wide.parquet"))
y = pd.read_csv(find("playground-series-s6e10/train.csv")).set_index("id").loc[oof["id"], "satisfaction"].astype(int).to_numpy()
cols = [c for c in oof.columns if c not in ("id", "fold")]
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
X, XT, fold = lg(oof[cols].to_numpy()), lg(test[cols].to_numpy()), oof["fold"].to_numpy()
stack, sub = np.zeros(len(oof)), np.zeros(len(test))
for k in range(5):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X[fold != k], y[fold != k])
    stack[fold == k], sub = m.decision_function(X[fold == k]), sub + m.predict_proba(XT)[:, 1] / 5
print(f"nested stack AUC {roc_auc_score(y, stack):.6f}")   # 0.961860
pd.DataFrame({"id": test["id"], "satisfaction": sub}).to_csv("submission.csv", index=False)
```

```python
# stack247 next to the version-6 members only (they were never among its inputs); continues the snippet above
s247, s247t = lg(np.load(find("stack247_oof.npy"))), lg(np.load(find("stack247_test.npy")))
new = ["r_realmlp_auxev", "s_realmlp_raw_auxev_10fold", "t_xgb_d5_auxev", "u_catboost_ctr_auxev"]
Z, ZT = np.column_stack([s247, lg(oof[new].to_numpy())]), np.column_stack([s247t, lg(test[new].to_numpy())])
stack2 = np.zeros(len(oof))
for k in range(5):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[fold != k], y[fold != k])
    stack2[fold == k] = m.decision_function(Z[fold == k])
print(f"stack247 + new members {roc_auc_score(y, stack2):.6f}")   # 0.962198
```

## What version 6 adds

Four single models chosen by measurement, and the stack member. Before training anything I added each of the 247 public pairs inside `stack247` to my 17-member library on its own (0.961799 nested), to see which kinds of model the library was missing; 93 of them were real gains. My own unpublished boosters added nothing (the largest change 0.000005 in either direction), and neither did TabM members (at most +0.000004), kNN-feature members (at most +0.000006) or Sadam Torres's original-data experiments (at most +0.000005). What did add was RealMLP fed the auxiliary-task expected values, best on 10 folds (Sachith7's `sa_rmlp_raw_aux_k10`, +0.000101, SE 0.000011, 5/5), a CatBoost building categorical CTRs with those values (Sachith7's `sa_cat_ctr_auxall_k10`, +0.000067), and a shallow XGBoost (Mitudru Dutta's `mi_xgb_d5`, +0.000058). The single largest gain was AbdullahSafwan333's 15-fold TabPFN, `ab_tabpfn_catfd10` (+0.000149), which needs GPU hours I did not spend here. The library had one RealMLP without those values and no shallow XGBoost, so version 6 adds **r**, RealMLP with the auxiliary-task values on the frozen folds; **s**, RealMLP on the raw columns and those values over 10 folds; **t**, a depth-5 XGBoost; **u**, a CTR CatBoost with those values, each trained on the frozen folds with my own pipeline and the feature blocks members j to q already use.

I measured each new member three ways and wrote down what I expected before any of them ran (the list after the table). 'Adds' means the same thing everywhere: the mean of the five paired fold differences is above two standard errors and at least 4 of 5 folds are positive. Every stack is fit on four frozen folds and scored on the fifth.

* **My library**: the 17 version-5 members (logistic stack on logits, C = 1, the instrument of versions 3 to 5), 0.961799 nested, plus the new member alone.
* **Next to `stack247`**: `stack247` (0.962205) plus the new member alone, same logistic stack.
* **The 247-pair pool**: the same 247 public pairs, refit with that notebook's own submission stacker (logistic regression, C = 0.003, tolerance 1e-8, logits clipped to +-30), here 0.962205 nested, the same value the notebook's Kaggle run printed, plus the new member alone.

| member | model | own OOF AUC | adds to my library (v5) | adds next to `stack247` | adds to the 247-pair pool |
|---|---|---|---|---|---|
| **r** `r_realmlp_auxev` | RealMLP + aux EVs | 0.961349 | +0.000010 (SE 0.000004, 5/5): **yes** | -0.000002 (SE 0.000002, 1/5): **no** | +0.000000 (SE 0.000002, 3/5): **no** |
| **s** `s_realmlp_raw_auxev_10fold` | RealMLP, raw + aux EVs, 10 folds | 0.961134 | +0.000050 (SE 0.000012, 5/5): **yes** | -0.000001 (SE 0.000002, 3/5): **no** | -0.000002 (SE 0.000002, 2/5): **no** |
| **t** `t_xgb_d5_auxev` | XGBoost depth 5 + aux EVs | 0.961259 | +0.000003 (SE 0.000002, 2/5): **no** | -0.000002 (SE 0.000001, 1/5): **no** | -0.000002 (SE 0.000002, 2/5): **no** |
| **u** `u_catboost_ctr_auxev` | CatBoost CTR + aux EVs | 0.961125 | +0.000007 (SE 0.000007, 3/5): **no** | -0.000001 (SE 0.000001, 1/5): **no** | -0.000002 (SE 0.000001, 1/5): **no** |
| all four together | | | +0.000058 (SE 0.000016, 5/5): **yes** | -0.000003 (SE 0.000002, 2/5): **no** | -0.000002 (SE 0.000002, 3/5): **no** |

**The honest reading.** **r** and **s** add to my library on their own, and together the four add +0.000058 (real). Next to `stack247` none of them adds anything real on its own; the set moves it by -0.000003 (not real). Inside the full 247-pair pool none of them is a real gain on its own; the set moves it by -0.000002 (not real). The public pool already holds 58 RealMLPs, 21 of them with auxiliary-task values, and several 10- and 15-fold members, so these four are new to my library and much less new to the board. If you are building your own stack, they are worth having; if you start from `stack247`, expect little from them.

**What I predicted, and what happened** (written down before any of them ran):

* **r**: own OOF between 0.96125 and 0.96150: 0.961349, **hit**. Adds to my library, predicted real, +0.00004 to +0.00012: +0.000010 (real), **miss** (real, but below the band). Adds nothing real to the 247-pair pool (within 0.00001): +0.000000, **hit**.
* **s**: own OOF between 0.96115 and 0.96145: 0.961134, **miss**. Adds to my library, predicted real, +0.00004 to +0.00012: +0.000050 (real), **hit**. Adds nothing real to the 247-pair pool (within 0.00001): -0.000002, **hit**.
* **t**: own OOF between 0.96110 and 0.96140: 0.961259, **hit**. Adds to my library, predicted +0.00000 to +0.00003: +0.000003 (not real), **hit**. Adds nothing real to the 247-pair pool (within 0.00001): -0.000002, **hit**.
* **u**: own OOF between 0.96105 and 0.96135: 0.961125, **hit**. Adds to my library, predicted +0.00001 to +0.00006: +0.000007 (not real), **miss** (below the band). Adds nothing real to the 247-pair pool (within 0.00001): -0.000002, **hit**.
* All four together add to my library, real, +0.00008 to +0.00020: +0.000058 (real), **miss**. Next to `stack247`, between -0.00001 and +0.00002: -0.000003, **hit**. Nothing real in the 247-pair pool (within 0.00002): -0.000002, **hit**.

**What every member adds given the other 20** (drop it from the 21-member stack and refit; descriptive, not pre-registered):

| member | own OOF AUC | stack loses without it | folds + | adds? |
|---|---|---|---|---|
| a `a_lgbm_digits` | 0.959568 | -0.000000 (SE 0.000001) | 4/5 | no |
| b `b_xgb_target_enc` | 0.958727 | +0.000001 (SE 0.000005) | 3/5 | no |
| c `c_catboost` | 0.958338 | +0.000001 (SE 0.000002) | 2/5 | no |
| d `d_histgb` | 0.958341 | +0.000005 (SE 0.000005) | 4/5 | no |
| e `e_mlp` | 0.956817 | +0.000008 (SE 0.000005) | 3/5 | no |
| f `f_spline_gam` | 0.948865 | -0.000002 (SE 0.000003) | 2/5 | no |
| g `g_extratrees` | 0.954592 | +0.000008 (SE 0.000008) | 4/5 | no |
| h `h_lgbm_freq` | 0.959791 | +0.000001 (SE 0.000004) | 3/5 | no |
| i `i_xgb_freq` | 0.959651 | -0.000000 (SE 0.000002) | 3/5 | no |
| j `j_lgbm_lr02` | 0.961150 | -0.000001 (SE 0.000002) | 2/5 | no |
| k `k_lgbm_extratrees` | 0.960934 | +0.000003 (SE 0.000003) | 4/5 | no |
| l `l_catboost_d8` | 0.960956 | -0.000002 (SE 0.000002) | 2/5 | no |
| m `m_lgbm_lr05` | 0.960411 | +0.000002 (SE 0.000002) | 3/5 | no |
| n `n_realmlp` | 0.961233 | +0.000002 (SE 0.000003) | 3/5 | no |
| o `o_tabpfn35` | 0.961082 | +0.000132 (SE 0.000028) | 5/5 | yes |
| p `p_lgbm_auxev` | 0.961314 | +0.000003 (SE 0.000003) | 3/5 | no |
| q `q_lgbm_auxev_10fold` | 0.961388 | +0.000023 (SE 0.000010) | 4/5 | yes |
| r `r_realmlp_auxev` | 0.961349 | -0.000001 (SE 0.000002) | 2/5 | no |
| s `s_realmlp_raw_auxev_10fold` | 0.961134 | +0.000044 (SE 0.000009) | 5/5 | yes |
| t `t_xgb_d5_auxev` | 0.961259 | +0.000002 (SE 0.000002) | 4/5 | no |
| u `u_catboost_ctr_auxev` | 0.961125 | +0.000003 (SE 0.000003) | 4/5 | no |

Given everything else, only **o**, **q** and **s** still carry weight. For the rest, what each one knows is already covered by the others in a stack this size. They stay in because they are useful on their own: for smaller stacks, for calibration work, and as checked baselines.

## The stack member: `stack247`

`stack247_oof.npy` (699,635 rows, `train.csv` order) and `stack247_test.npy` (299,844 rows, `test.csv` order) hold P(satisfaction = 1) as float64 from my public notebook "S6E10 | 247 Public OOFs, One Stack | LB 0.96187" (https://www.kaggle.com/code/dariushafshar/s6e10-247-public-oofs-one-stack-lb-0-96187). That notebook loads 292 public OOF pairs from 18 sources, drops 45 near-copies and stacks the remaining 247 with one logistic regression on their logits (C = 0.003), fit on four of these same frozen folds and scored on the fifth.

* **Nested CV 0.962205** (folds 0.962358 / 0.961410 / 0.962804 / 0.962446 / 0.962041), from that notebook's Kaggle run. The OOF file is its nested out-of-fold prediction, so every value comes from a stacker that did not see that row's label.
* **Public LB 0.96187.** `stack247_test.npy` is the stacker refit on every training row: the same predictions as the `submission.csv` that was scored (sha256 starts `ac2f1c51d067615e`), to the precision that file prints.
* **Its inputs are listed in `stack247_inputs.csv`**: all 247, with source, author, OOF AUC and scale. All 17 members of version 5 of this library are among them; the version-6 members are not.

**How to use it.** Treat it as one strong member and stack it with models that are not among its inputs: your own, or the version-6 members here. Re-stacking it with its inputs re-fits weights the stack already fitted; on these folds that moves it by -0.000022 (SE 0.000006, 0/5 folds positive). For that reason it is not in `manifest.csv`, not in the parquet files and not named `oof_*.npy`: anything that loads the library generically (including the 247-pair notebook itself, which reads this library's manifest) never picks it up by accident.

**Credit.** Every model inside it is somebody's: 247 pairs by Chris Deotte, Kratosyan and the authors listed in `stack247_inputs.csv` and in the notebook's source table. I only checked, aligned and stacked them.

## Version 5: which of its members added to the public pool

Version 5 added 3 members (o, p, q). I measured each one against my library and against the full public pool, and wrote down what I expected before any of them ran. 'Adds' means the same thing everywhere: the mean of the five paired fold differences is above two standard errors and at least 4 of 5 folds are positive. The stacks are logistic regressions on logits (C=1), fitted on four frozen folds and scored on the fifth.

* **My library**: the 14 members a to n (stack 0.961473), plus the new member alone.
* **The full public pool**: the 135 public OOF pairs Chris Deotte packaged (the 124 that Kratosyan stacks, which already include my v4 members, plus 11 tabular foundation-model pairs), stack 0.962079, plus the new member alone.

| member | model | own OOF AUC | adds to my library (a to n) | adds to the full public pool |
|---|---|---|---|---|
| **o** `o_tabpfn35` | TabPFN-3.5 | 0.961082 | +0.000232 (SE 0.000045, 5/5): **yes** | -0.000001 (SE 0.000007, 3/5): **no** |
| **p** `p_lgbm_auxev` | LightGBM + aux EVs | 0.961314 | +0.000062 (SE 0.000028, 4/5): **yes** | +0.000000 (SE 0.000010, 2/5): **no** |
| **q** `q_lgbm_auxev_10fold` | LightGBM + aux EVs, 10 folds | 0.961388 | +0.000108 (SE 0.000042, 4/5): **yes** | +0.000002 (SE 0.000005, 3/5): **no** |
| all 3 together | | | stack 0.961473 to 0.961799 | +0.000001 (SE 0.000007, 3/5): **no** |

**The honest reading.** All 3 add to my own library, and none of them adds anything measurable to the full public pool: the largest pool change is +0.000002. The pool already holds several TabPFN members, auxiliary-task members and 10-fold members, so these are new to my library, not to the board. If you stack the public pool, these three will not move you. If you are building a smaller stack of your own, they are worth having.

Added one at a time instead (forward selection over a to n, the rule this library has used since v3), o then q are kept and p is not (+0.000007, 3/5 once the others are in): p and q are the same recipe on 5 and 10 folds, so they overlap. Version 5 publishes all 3 anyway, with these numbers next to them, so you can decide.

**What I predicted, and what happened** (written down before the runs):

* o, TabPFN own OOF between 0.9609 and 0.9612: 0.961082, **hit**.
* p beats j (the same recipe without the auxiliary-task values) by 0.0001 to 0.0002, fold by fold: +0.000167 (SE 0.000056, 4/5): **real**, inside the band, **hit**.
* q (10 folds) beats p (5 folds) by 0.00003 to 0.0001, fold by fold: +0.000068 (SE 0.000041, 4/5): **not real**. The mean is inside the band but it is not above two standard errors, so **miss**: as single models, 10 folds did not measurably beat 5 for this recipe.
* All 3 add to my library: **hit** (each alone). None adds to the public pool: **hit**.

## The library

Out-of-fold (OOF) and test predictions for 21 single models I trained for Kaggle Playground Series **S6E10** (airline passenger satisfaction), all aligned to one frozen fold partition:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, train.satisfaction)
# over train.csv in ORIGINAL FILE ROW ORDER: never sorted, never shuffled
```

If you build your own OOF arrays on this same split, they stack with this library row for row, with no refitting. `folds_seed42.npy` is included so you can check the split rather than take my word for it: regenerate the partition with the lines above and compare. I ran exactly that comparison before packaging, and it matches.

**Members j to u are frontier-grade.** I trained them later, on the same frozen folds, with a second pipeline that rebuilds the feature ideas of the strongest public notebooks on this board (member o, TabPFN, ran on a Kaggle GPU kernel; the rest on CPU kernels, and r to u on my own machine); the best of them, **q**, scores 0.961388 out of fold. Every idea they borrow is credited below to the person who published it, and their arrays passed exactly the same checks as members a to i.

**What an OOF array is, if you have not used one before.** For each of the 5 folds, the model trains on the other 4 and predicts the held-out fold. Stitch those held-out blocks back together in original row order and you get one prediction per training row, each produced by a model that never saw that row's label: an honest, leak-free feature you can fit a *second* model on. That is stacking. The matching `test_*.npy` is the mean of the fold models over test.csv, so the pair is a complete stacking unit: fit the meta-model on the OOF array, apply it to the test array.

## Row order is the key for the .npy files, and it is load-bearing

The two parquet files carry `id`, so you can join them. The `.npy` files carry no ids and there are no labels anywhere in this dataset: **for the .npy files, alignment is positional.**

* every `oof_*.npy` and `stack247_oof.npy` has **699,635** rows, in `train.csv` file order
* every `test_*.npy` and `stack247_test.npy` has **299,844** rows, in `test.csv` file order
* `folds_seed42.npy` has **699,635** entries, same order, values 0 to 4

```python
import numpy as np, pandas as pd
train = pd.read_csv(".../train.csv")          # do NOT sort or reindex
oof_a = np.load(".../oof_a.npy")              # oof_a[i] belongs to train.iloc[i]
assert len(oof_a) == len(train)
```

If you sort, filter or `sample()` the competition frame before aligning, every number you compute afterwards is silently wrong and will still look plausible. Align first, subset second.

## No competition data is redistributed here

Model predictions only (the stack member included), plus the competition's `id` column in the two parquet files as a join key. No `train.csv`, no `test.csv`, no feature values, no labels. `folds_seed42.npy` is shipped for convenience. It is a deterministic function of the label vector that reveals nothing about individual labels: fold membership under a stratified shuffle tells you the fold, not the target.

## Files

| File | Rows | What it is |
|---|---|---|
| `oof_wide.parquet` | 699,635 | `id`, `fold` (0 to 4) and one float64 column per single-model member (`a_lgbm_digits`, `b_xgb_target_enc`, `c_catboost`, ...), `train.csv` order |
| `test_wide.parquet` | 299,844 | `id` and one float64 column per member, `test.csv` order |
| `oof_{a..n}.npy` | 699,635 | OOF P(satisfaction = 1), float64, `train.csv` order (names unchanged since v4) |
| `test_{a..n}.npy` | 299,844 | test predictions, mean over the 5 fold models, `test.csv` order |
| `oof_o_tabpfn35.npy`, `oof_p_lgbm_auxev.npy`, `oof_q_lgbm_auxev_10fold.npy` | 699,635 | the same, for the v5 members |
| `test_o_tabpfn35.npy`, `test_p_lgbm_auxev.npy`, `test_q_lgbm_auxev_10fold.npy` | 299,844 | the same, for the v5 members |
| `oof_r_realmlp_auxev.npy`, `oof_s_realmlp_raw_auxev_10fold.npy`, `oof_t_xgb_d5_auxev.npy`, `oof_u_catboost_ctr_auxev.npy` | 699,635 | the same, for the version-6 members |
| `test_r_realmlp_auxev.npy`, `test_s_realmlp_raw_auxev_10fold.npy`, `test_t_xgb_d5_auxev.npy`, `test_u_catboost_ctr_auxev.npy` | 299,844 | the same, for the version-6 members (test = sigmoid of the mean fold-model logit, as for j to q) |
| `stack247_oof.npy` | 699,635 | the stack member's nested OOF P(satisfaction = 1), float64, `train.csv` order (not in manifest.csv or the parquet files, on purpose) |
| `stack247_test.npy` | 299,844 | the stack member's test predictions: the refit stacker, the same predictions as the scored submission |
| `stack247_inputs.csv` | 247 | the stack member's inputs: member, source, author, OOF AUC, scale |
| `marginal_value.csv` | 21 | what each member adds: given the other 20; for the version-6 members alone over version 5, next to `stack247` and over the 247-pair pool; for o, p and q the version-5 measurements |
| `folds_seed42.npy` | 699,635 | int8 fold index per training row, `train.csv` order |
| `manifest.csv` | 21 | one row per single-model member (`stack247` is documented above instead): recipe, honest CV, per-fold AUC, ranges |
| `rank_correlation.csv` | 21 x 21 | Spearman correlation between every pair of OOF arrays |
| `dataset-cover.png` | | the version-5 cover chart: own OOF AUC against what each of the 17 members added given the others |

## The members and what each one honestly scores

Every AUC below was recomputed from the arrays in this dataset against the `train.csv` labels at packaging time, not copied from a training log.

| | Model | Feature set | OOF AUC | fold sd |
|---|---|---|---|---|
| **a** | LightGBM LGBMClassifier | numerics raw + native categorical dtype (vocabulary from the training fold only) + quantisation-digit features floor(x/10**k) mod 10 for k in -4..3 over every numeric column | **0.959568** | 0.000777 |
| **b** | XGBoost XGBClassifier (tree_method=hist) | numerics raw + smoothed target encoding of every categorical, encoded inside an inner 5-fold split so no training row sees its own label through the encoder; prior weight m=20 rows | **0.958727** | 0.000704 |
| **c** | CatBoost CatBoostClassifier | numerics raw (NaN kept, CatBoost handles it) + every categorical passed as a string to CatBoost's native handling, missing levels as the explicit token __NA__; single-feature ordered target statistics only (max_ctr_complexity=1, no combination features), categoricals with 8 or fewer levels one-hot instead | **0.958338** | 0.000741 |
| **d** | scikit-learn HistGradientBoostingClassifier | numerics raw with NaN kept native + categoricals as pandas category dtype with the vocabulary taken from the training fold only, so a level the training fold never saw becomes missing rather than an error | **0.958341** | 0.000823 |
| **e** | scikit-learn MLPClassifier, hidden (128,64), adam, alpha 1e-4, batch 1024, 3 seeds averaged | numerics quantile-transformed to a normal marginal (fit on the training fold only) with NaN filled at the transformed median, plus one 0/1 missingness mask per numeric column that has any NaN, plus one-hot categoricals with the vocabulary from the training fold | **0.956817** | 0.000522 |
| **f** | scikit-learn SplineTransformer -> LogisticRegression (C=1.0, lbfgs) | cubic B-splines on quantile knots per numeric column (up to 12 knots), median imputation with medians from the training fold only, one missingness mask per imputed column, one-hot categoricals; no interaction terms of any kind | **0.948865** | 0.000587 |
| **g** | scikit-learn ExtraTreesClassifier (500 trees, min_samples_leaf=50) | numerics median-imputed with medians from the training fold only, one missingness mask per imputed column, categoricals as integer codes with the vocabulary from the training fold (-1 for unseen) | **0.954592** | 0.000523 |
| **h** | LightGBM LGBMClassifier (member a's parameters) + frequency encoding | member a's set - numerics raw + native categorical dtype (training-fold vocabulary) + quantisation digits floor(x/10**k) mod 10, k in -4..3, over the numeric columns - plus a frequency-encoded twin of every one of those columns: {col}_freq is the value's normalised frequency in train.csv and test.csv CONCATENATED. That frequency step is TRANSDUCTIVE (it reads test feature values, so it cannot be reproduced from the training fold alone) but LABEL-FREE (no target enters it) | **0.959791** | 0.000754 |
| **i** | XGBoost XGBClassifier (tree_method=hist, depth 7), @evgendvorkin's S6E9 single-XGB recipe | numerics raw + every categorical label-encoded (training-fold vocabulary) + quantisation digits floor(x/10**k) mod 10, k in -4..3, over ALL of those columns + a frequency-encoded twin of every one of those columns: {col}_freq is the value's normalised frequency in train.csv and test.csv CONCATENATED. That frequency step is TRANSDUCTIVE (it reads test feature values, so it cannot be reproduced from the training fold alone) but LABEL-FREE (no target enters it) | **0.959651** | 0.000735 |
| **j** | LightGBM, learning rate 0.02, 127 leaves, min_child_samples 100, feature fraction 0.5, bagging 0.8 every round, lambda_l2 1, max_bin 255 | blocks: the 21 columns, digits, value counts, route profile, original-data model predictions, original-data target means, GPT-2 token keys, fold-safe target encoding (defined below) | **0.961150** | 0.000641 |
| **k** | LightGBM with extra_trees (random split thresholds), learning rate 0.03, 255 leaves, min_child_samples 100, feature fraction 0.4, bagging 0.8 | blocks: the 21 columns, digits, value counts, route profile, original-data model predictions, original-data target means, GPT-2 token keys, fold-safe target encoding (defined below) | **0.960934** | 0.000610 |
| **l** | CatBoost, learning rate 0.06, depth 8, l2_leaf_reg 3, border_count 254, max_ctr_complexity 2, with Flight Distance, Age and the 4 text columns as categorical features (CatBoost's own ordered target statistics, computed inside the training fold) | blocks: the 21 columns, digits, value counts, route profile, original-data model predictions, original-data target means (defined below) | **0.960956** | 0.000528 |
| **m** | LightGBM (learning rate 0.05, 127 leaves, min_child_samples 100, feature fraction 0.5, bagging 0.8 every round, lambda_l2 1, max_bin 255) | blocks: the 21 columns, digits, value counts, route profile (defined below) | **0.960411** | 0.000603 |
| **n** | pytabkit RealMLP_TD (8-model internal ensemble, 4 epochs, hidden 512-256-128, Vladimir Demidov's public hyper-parameters as adapted in goodpjw2008's notebook), with a categorical twin of every numeric column | blocks: the 21 columns, value counts, route profile, original-data model predictions, original-data target means, fold-safe target encoding (defined below) | **0.961233** | 0.000581 |
| **o** | TabPFN-3.5 (tabpfn 9.0.0, checkpoint tabpfn-v3.5-20260909, by Prior Labs, non-commercial licence), one estimator, seed 0, the whole training fold (559,708 rows) as context, no fine-tuning | blocks: raw columns + route group (defined below) | **0.961082** | 0.000500 |
| **p** | LightGBM, member j's recipe (learning rate 0.02, 127 leaves, min_child_samples 100, feature fraction 0.5, bagging 0.8 every round, lambda_l2 1, max_bin 255) with 13 auxiliary-task expected values added | blocks: the 21 columns, digits, value counts, route profile, original-data model predictions, original-data target means, GPT-2 token keys, auxiliary-task expected values, fold-safe target encoding (defined below) | **0.961314** | 0.000568 |
| **q** | member p's recipe trained on 10 folds that refine the frozen 5 (each frozen fold split in half by StratifiedKFold(2, shuffle=True, random_state=42)), so each OOF row comes from a model that saw 90% of train; test = mean of the 10 models; fold AUCs are reported on the frozen 5 | blocks: the 21 columns, digits, value counts, route profile, original-data model predictions, original-data target means, GPT-2 token keys, auxiliary-task expected values, fold-safe target encoding (defined below) | **0.961388** | 0.000574 |
| **r** | pytabkit RealMLP_TD with member n's settings (8-model internal ensemble, 4 epochs, hidden 512-256-128, Vladimir Demidov's public hyper-parameters as adapted in goodpjw2008's notebook), a categorical twin of every numeric column, seed 7, 5 frozen folds | blocks: the 21 columns, value counts, route profile, original-data model predictions, original-data target means, auxiliary-task expected values, fold-safe target encoding (defined below) | **0.961349** | 0.000585 |
| **s** | pytabkit RealMLP_TD with member n's settings (8-model internal ensemble, 4 epochs, hidden 512-256-128, Vladimir Demidov's public hyper-parameters as adapted in goodpjw2008's notebook), a categorical twin of every numeric column, seed 11, trained on 10 folds that refine the frozen 5 (each frozen fold split in half by StratifiedKFold(2, shuffle=True, random_state=42)), so each OOF row comes from a model that saw 90% of train; test = mean of the 10 models' logits; fold AUCs are reported on the frozen 5 | blocks: the 21 columns, auxiliary-task expected values (defined below) | **0.961134** | 0.000592 |
| **t** | XGBoost (hist), depth 5, learning rate 0.03, subsample 0.8, colsample_bytree 0.5, min_child_weight 5, lambda 1, max_bin 256, 5 frozen folds | blocks: the 21 columns, digits, value counts, route profile, original-data model predictions, original-data target means, GPT-2 token keys, auxiliary-task expected values, fold-safe target encoding (defined below) | **0.961259** | 0.000579 |
| **u** | CatBoost, member l's settings (learning rate 0.06, depth 8, l2_leaf_reg 3, border_count 254, max_ctr_complexity 2, with Flight Distance, Age and the 4 text columns as categorical features, CatBoost's own ordered target statistics computed inside the training fold) with the 13 auxiliary-task expected values added, 5 frozen folds | blocks: the 21 columns, digits, value counts, route profile, original-data model predictions, original-data target means, auxiliary-task expected values (defined below) | **0.961125** | 0.000599 |

**Feature blocks of members j to u** (the full per-member text is in `manifest.csv`):

* **the 21 columns**: all 21 columns as they are, the 4 text columns as integer codes (vocabulary over train, test and the original data, label-free)
* **digits**: floor(x/10**k) mod 10 for k 0..3 of Age, Flight Distance and both delays (non-constant positions only), plus Flight Distance % 100, % 1000, // 10, // 100 and its digit count
* **value counts**: how often each value of Age, Flight Distance and both delays occurs in train.csv + test.csv (TRANSDUCTIVE, label-free)
* **route profile**: for every Flight Distance value, the mean of every other column and the route's row count, over train.csv + test.csv (TRANSDUCTIVE, label-free)
* **original-data model predictions**: ORIGINAL-DATA models: an XGBoost (depth 8, 600 trees) and a 3-seed LightGBM (800 rounds) fitted on the 129,880 original rows only, scored as logits on every competition row (no competition label enters them)
* **original-data target means**: ORIGINAL-DATA labels: the satisfied rate per value of every column and per (Type of Travel, Class, Customer Type) combination, from the original rows only
* **GPT-2 token keys**: first token, last token and token count of ' {value}' in the gpt2 encoding, for Flight Distance, Age and both delays
* **fold-safe target encoding**: sklearn TargetEncoder (smooth auto, cross-fitted inside the training fold) of 12 keys: distance, age, distance x age, distance x class, distance x travel type, age x class x travel type x customer type, distance // 10, distance % 1000, age x class, age x travel type, and the first and last GPT-2 token of distance
* **raw columns + route group**: the 21 columns as they are plus Flight Distance // 10, the 4 text columns as integer codes (vocabulary over train + test, label-free); the 4 text columns, Flight Distance and Flight Distance // 10 declared categorical to TabPFN; no engineered features
* **auxiliary-task expected values**: auxiliary-task expected values: for each of the 13 service ratings, E[rating | the other 20 columns] = sum of k * P(rating = k) from a multi-class XGBoost (depth 6, learning rate 0.1, 300 rounds) fitted on train.csv + test.csv rows, cross-fitted in 5 folds so no row is scored by a model that saw it (TRANSDUCTIVE, label-free: the target never enters)

Strongest here is **q** at 0.961388; weakest is **f** at 0.948865. A spread of 0.012523 across 21 members, with fold standard deviations of 0.000500 to 0.000823. Any improvement you measure that is smaller than that sd is inside fold noise.

### Early-stopping and leakage honesty, per member

Not all of these are equally clean and I would rather say so than have you find out inside a stack:

* **a**: early stopping on the held-out validation fold itself, 100 rounds - this makes member a's OOF mildly optimistic; disclosed, not corrected
* **b**: none - fixed n_estimators, the validation fold is never looked at
* **c**: none - fixed iterations, the validation fold is never passed to fit()
* **d**: none - early_stopping=False and fixed max_iter; no internal split either
* **e**: an inner 10% split carved from the TRAINING fold only (MLPClassifier validation_fraction=0.1); the validation fold is never seen
* **f**: none - all transformers and the model fit on the training fold only
* **g**: none - a bagged ensemble has nothing to early-stop
* **h**: early stopping on the held-out validation fold itself, 100 rounds, inherited from member a - h's OOF is mildly optimistic for the same reason a's is; additionally its frequency features are transductive (train+test value counts, no labels). Both disclosed, neither corrected
* **i**: early stopping on the held-out validation fold itself, 100 rounds, as the source notebook does - i's OOF is mildly optimistic for the same reason a's and h's are; additionally its frequency features are transductive (train+test value counts, no labels). Both disclosed, neither corrected
* **j, k, l, m, p**: early stopping on the held-out validation fold itself (j 200, k 200, l 200, m 100, p 200 rounds), as in the pipeline that trained them - mildly optimistic for the same reason a's is; additionally their value-count and route-profile features (and, in p, the auxiliary-task expected values) are transductive (train+test, no labels). Disclosed, not corrected
* **n**: early stopping on the held-out validation fold itself, in RealMLP's form: 4 fixed epochs, with the fold passed to pytabkit as its validation set, which it can use to pick the best epoch - treated as mildly optimistic for the same reason a's is; additionally its value-count and route-profile features are transductive (train+test, no labels). Disclosed, not corrected
* **o**: none - TabPFN has no training loop to stop; each fold's model conditions on the four training folds only and never sees the validation fold's labels
* **q**: early stopping on the held-out half of the frozen fold itself, 200 rounds, as in the modeler pipeline - this OOF is mildly optimistic for the same reason a's is; additionally its count, route-profile and auxiliary-task features are transductive (train+test, no labels). Disclosed, not corrected
* **r**: early stopping on the held-out validation fold itself, in RealMLP's form, exactly as member n: 4 fixed epochs, with the fold passed to pytabkit as its validation set, which it can use to pick the best epoch - treated as mildly optimistic for the same reason a's is; additionally its value-count, route-profile and auxiliary-task features are transductive (train+test, no labels). Disclosed, not corrected
* **s**: early stopping on the held-out half of the frozen fold itself, in RealMLP's form (4 fixed epochs, that half passed to pytabkit as its validation set) - mildly optimistic for the same reason a's is; additionally its auxiliary-task features are transductive (train+test, no labels). Disclosed, not corrected
* **t**: early stopping on the held-out validation fold itself, 400 rounds, as in the modeler pipeline - mildly optimistic for the same reason a's is; additionally its count, route-profile and auxiliary-task features are transductive (train+test, no labels). Disclosed, not corrected
* **u**: early stopping on the held-out validation fold itself, 200 rounds, as member l - mildly optimistic for the same reason a's is; additionally its count, route-profile and auxiliary-task features are transductive (train+test, no labels). Disclosed, not corrected

### Which members use borrowed ideas

Some members are built on ideas other people published first. Here is which ones, and whose ideas; the Credits section at the end names each notebook.

| member | borrowed ideas, and from whom |
|---|---|
| **a** | none |
| **b** | none |
| **c** | none |
| **d** | none |
| **e** | none |
| **f** | none |
| **g** | none |
| **h** | frequency encoding (Дворкин Евгений Владимирович (@evgendvorkin), S6E9) |
| **i** | the single-XGB recipe: hyper-parameters, label encoding, digits over every column, frequency encoding (Дворкин Евгений Владимирович (@evgendvorkin), S6E9) |
| **j** | distance digits, value counts, route profile, original-data XGBoost, original-data LightGBM, original-data target means, GPT-2 token keys, fold-safe target encoding, target-encoding keys, LightGBM schedule; from 인공저능연구소( Artificial Idiocy Lab ), Arhan canli, goodpjw2008 |
| **k** | distance digits, value counts, route profile, original-data XGBoost, original-data LightGBM, original-data target means, GPT-2 token keys, fold-safe target encoding, target-encoding keys, LightGBM leaf settings; from 인공저능연구소( Artificial Idiocy Lab ), Arhan canli, goodpjw2008 |
| **l** | distance digits, value counts, route profile, original-data XGBoost, original-data LightGBM, original-data target means; from 인공저능연구소( Artificial Idiocy Lab ), Arhan canli, goodpjw2008 |
| **m** | distance digits, value counts, route profile, LightGBM leaf settings; from 인공저능연구소( Artificial Idiocy Lab ), Arhan canli, goodpjw2008 |
| **n** | value counts, route profile, original-data XGBoost, original-data LightGBM, original-data target means, fold-safe target encoding, target-encoding keys, RealMLP recipe, RealMLP inputs and schedule; from 인공저능연구소( Artificial Idiocy Lab ), goodpjw2008, Arhan canli, Vladimir Demidov |
| **o** | TabPFN input set, TabPFN full-context worker; from goodpjw2008, Hemingweb |
| **p** | distance digits, value counts, route profile, original-data XGBoost, original-data LightGBM, original-data target means, GPT-2 token keys, auxiliary-task expected values, fold-safe target encoding, target-encoding keys, LightGBM schedule; from 인공저능연구소( Artificial Idiocy Lab ), Arhan canli, goodpjw2008 |
| **q** | distance digits, value counts, route profile, original-data XGBoost, original-data LightGBM, original-data target means, GPT-2 token keys, auxiliary-task expected values, fold-safe target encoding, target-encoding keys, LightGBM schedule; from 인공저능연구소( Artificial Idiocy Lab ), Arhan canli, goodpjw2008 |
| **r** | value counts, route profile, original-data XGBoost, original-data LightGBM, original-data target means, auxiliary-task expected values, fold-safe target encoding, target-encoding keys, RealMLP recipe, RealMLP inputs and schedule; from 인공저능연구소( Artificial Idiocy Lab ), goodpjw2008, Arhan canli, Vladimir Demidov |
| **s** | auxiliary-task expected values, RealMLP recipe, RealMLP on the raw columns plus auxiliary values, 10 folds; from goodpjw2008, Vladimir Demidov, Sachith7 |
| **t** | distance digits, value counts, route profile, original-data XGBoost, original-data LightGBM, original-data target means, GPT-2 token keys, auxiliary-task expected values, fold-safe target encoding, target-encoding keys, a depth-5 XGBoost; from 인공저능연구소( Artificial Idiocy Lab ), Arhan canli, goodpjw2008, Mitudru Dutta |
| **u** | distance digits, value counts, route profile, original-data XGBoost, original-data LightGBM, original-data target means, auxiliary-task expected values, CTR CatBoost plus auxiliary values; from 인공저능연구소( Artificial Idiocy Lab ), Arhan canli, goodpjw2008, Sachith7 |

### Where the OOF AUC and the fold-mean AUC disagree

`oof_auc` pools all rows into one ranking; `fold_mean_auc` averages the 5 fold-wise AUCs. When the fold models are calibrated differently from each other, pooling their outputs into one ranking scores worse than the folds do alone. The gap is a real property of the member, not a bug, and it matters if you plan to use the raw probabilities rather than the ranks:

| member | oof_auc | fold_mean_auc | gap |
|---|---|---|---|
| a | 0.959568 | 0.959574 | -0.000006 |
| b | 0.958727 | 0.958730 | -0.000003 |
| c | 0.958338 | 0.958347 | -0.000009 |
| d | 0.958341 | 0.958347 | -0.000006 |
| e | 0.956817 | 0.956863 | -0.000046 |
| f | 0.948865 | 0.948868 | -0.000003 |
| g | 0.954592 | 0.954604 | -0.000012 |
| h | 0.959791 | 0.959797 | -0.000006 |
| i | 0.959651 | 0.959658 | -0.000007 |
| j | 0.961150 | 0.961154 | -0.000004 |
| k | 0.960934 | 0.960939 | -0.000005 |
| l | 0.960956 | 0.960967 | -0.000011 |
| m | 0.960411 | 0.960418 | -0.000007 |
| n | 0.961233 | 0.961249 | -0.000016 |
| o | 0.961082 | 0.961086 | -0.000004 |
| p | 0.961314 | 0.961321 | -0.000007 |
| q | 0.961388 | 0.961389 | -0.000001 |
| r | 0.961349 | 0.961356 | -0.000008 |
| s | 0.961134 | 0.961134 | +0.000000 |
| t | 0.961259 | 0.961264 | -0.000005 |
| u | 0.961125 | 0.961133 | -0.000008 |

## Diversity: the correlation matrix is the map

Off-diagonal Spearman rank correlations run 0.870427 to 0.995137. The tightest pair is **n and r** at 0.995137 (near-redundant; a blend gains little from having both), the loosest is **f and k** at 0.870427. Among the version-6 members, **r** is closest to **n** (0.995137); **s** is closest to **r** (0.982527); **t** is closest to **p** (0.989931); **u** is closest to **l** (0.985926).

Rank correlation rather than Pearson, because AUC depends only on ordering. A weak, decorrelated member can be worth more *subtracted* than added: in a logistic stack on rank features, a member that additive greedy selection gives zero weight can take a large negative coefficient and earn its place as a corrector. `rank_correlation.csv` is where you look for that.

## How members j to u were chosen

Candidates come from my second pipeline, trained on the same folds. Each must pass the same array checks as the members above, then earn its place by forward selection on a nested logistic stack (logits, C=1, weights fitted on four folds and scored on the fifth) over the library as it stands: at each step the candidate with the largest paired gain joins only if the gain is real, mean of the five fold differences above two standard errors and at least four of five folds positive. A candidate rank-correlated above 0.9999 with a member already in the library would be refused as a duplicate. Published members keep their letters: each new version only tests new candidates on top of the library already out. From version 5 on, every candidate that passes the array checks is published and the rule's verdict is printed next to it, instead of the rule deciding what you get to see.

Block names: raw = the 21 columns; dig = digits; cnt = value counts; fdagg = route profile; ogp = original-data model predictions; om = original-data target means; tok = GPT-2 token keys; te_pub = fold-safe target encoding; aux = auxiliary-task expected values; pfn_in = raw columns + route group.

**Version v3**: 8 candidates, tested on top of members a to i (nested stack 0.959981).

| candidate (pipeline name) | verdict | at step | nested stack AUC with it | gain over that step's stack | folds + | max rank corr to the library |
|---|---|---|---|---|---|---|
| LightGBM lr 0.02, blocks raw+dig+cnt+fdagg+ogp+om+tok+te_pub (F3) | joins as **j** | 1 | 0.961277 | +0.001291 (SE 0.000055) | 5/5 | 0.960708 |
| XGBoost depth 8, blocks raw+dig+cnt+fdagg+ogp+om+tok+te_pub (X3) | not kept (null) | 5 | 0.961403 | +0.000011 (SE 0.000007) | 4/5 | 0.992775 |
| LightGBM lr 0.05, blocks raw+dig (L0) | not kept (null) | 5 | 0.961393 | +0.000000 (SE 0.000004) | 3/5 | 0.978492 |
| LightGBM lr 0.05, blocks raw+dig+cnt+fdagg (L1) | joins as **m** | 4 | 0.961393 | +0.000020 (SE 0.000006) | 5/5 | 0.971535 |
| LightGBM lr 0.05, blocks raw+dig+cnt+fdagg+ogp (L2) | not kept (null) | 5 | 0.961395 | +0.000002 (SE 0.000004) | 1/5 | 0.979694 |
| LightGBM lr 0.05, blocks raw+dig+cnt+fdagg+ogp+om+tok+te_pub (L3) | not kept (null) | 5 | 0.961400 | +0.000010 (SE 0.000007) | 4/5 | 0.989596 |
| CatBoost depth 8, blocks raw+dig+cnt+fdagg+ogp+om (C3) | joins as **l** | 3 | 0.961373 | +0.000037 (SE 0.000016) | 5/5 | 0.982134 |
| LightGBM extra-trees, blocks raw+dig+cnt+fdagg+ogp+om+tok+te_pub (E3) | joins as **k** | 2 | 0.961338 | +0.000061 (SE 0.000016) | 5/5 | 0.982352 |

Nested stack after version v3: 0.961393. Still running then, so not considered in that version: RealMLP, blocks raw+cnt+fdagg+ogp+om+te_pub (R3).

**Version v4**: 2 candidates, tested on top of the 13 members of the previous version (nested stack 0.961393).

| candidate (pipeline name) | verdict | at step | nested stack AUC with it | gain over that step's stack | folds + | max rank corr to the library |
|---|---|---|---|---|---|---|
| RealMLP, blocks raw+cnt+fdagg+ogp+om+te_pub (R3) | joins as **n** | 1 | 0.961473 | +0.000086 (SE 0.000008) | 5/5 | 0.984611 |
| LightGBM lr 0.02, 3 seeds, blocks raw+dig+cnt+fdagg+ogp+om+tok+te_pub (F3x3) | not kept (null) | 2 | 0.961474 | +0.000000 (SE 0.000003) | 2/5 | 0.997802 |

Nested stack after version v4: 0.961473.

**Version v5**: 3 candidates, tested on top of the 14 members of the previous version (nested stack 0.961473).

| candidate (pipeline name) | verdict | at step | nested stack AUC with it | gain over that step's stack | folds + | max rank corr to the library |
|---|---|---|---|---|---|---|
| TabPFN-3.5, blocks pfn_in (T1) | published as **o**, kept by the rule | 1 | 0.961712 | +0.000232 (SE 0.000045) | 5/5 | 0.969004 |
| LightGBM + aux EVs, blocks raw+dig+cnt+fdagg+ogp+om+tok+aux+te_pub (G1) | published as **p**, not kept by the rule | 3 | 0.961799 | +0.000007 (SE 0.000005) | 3/5 | 0.990338 |
| LightGBM + aux EVs, 10 folds, blocks raw+dig+cnt+fdagg+ogp+om+tok+aux+te_pub (G2) | published as **q**, kept by the rule | 2 | 0.961792 | +0.000078 (SE 0.000026) | 4/5 | 0.987546 |

Nested stack after version v5: 0.961799.

A candidate that is not kept may have been real at an earlier step; it stopped adding anything once the members above it had joined. The step-by-step numbers are printed by `select_external.py`. Overall: the nested logistic stack goes from 0.959981 with members a to i to 0.961799 with all 17 members.

**Version v6**: the four members above were not chosen by forward selection over candidates of mine; they were chosen before training, from a measured gap (each public pair added alone to the 17-member library, see the version-6 section at the top), and all four are published with their numbers, whatever they measured. Nested stack after version v6, all 21 single members: 0.961860.

## What you can actually do with these

1. **Blending practice with a real answer key.** Greedy vs logistic regression vs NNLS over 21 honest OOF arrays, on CPU, in minutes, with no training bill.
2. **Stacking without the retrain.** Members a to i took about 181 minutes to train on one machine's CPU, and members j to q about 676 minutes of fold training on Kaggle kernels (o alone, TabPFN with the full fold context, about 7.3 GPU hours), summed over members in both cases. Skip straight to the meta-model.
3. **Calibration exercises.** These members have visibly different probability geometry (see `oof_min`/`oof_max` in `manifest.csv`: 0.000354266 to 0.999778 across the set). Same rows, same folds, different reliability curves.
4. **Diversity-vs-strength experiments.** Several model families, an AUC spread of 0.012523, and a correlation matrix spanning 0.870 to 0.995.
5. **A null band for OOF-vs-test drift checks.** These are honestly produced arrays on a known split, so the KS distance between a member's OOF and test distributions gives you a calibrated *null* to judge someone else's arrays against.

## What is deliberately NOT in here

* **Duplicates.** `verify.py` refuses to package two members that are byte-identical or rank-correlated above 0.9999. Padding a library with re-runs of the same model is exactly what makes a public OOF pool untrustworthy.
* **Any hand-made blend.** If you want a blend, build it from these; a shipped blend would just be my opinion wearing the clothes of a measurement. The one exception, from version 6, is `stack247`: not a blend I tuned, but the nested out-of-fold prediction of a published notebook whose every step is printed, shipped in its own files so nothing stacks it by accident.
* **Original rows as extra training rows.** Members **j**, **k**, **l**, **n**, **p**, **q**, **r**, **t** and **u** use the original dataset only as features (see Provenance below); no single-model member adds its rows to the training data. The public models inside `stack247` follow their authors' own recipes.
* **A leaderboard score in the filename.** These are OOF arrays; what they are worth on the private leaderboard is not something this dataset can honestly claim.

## float64 on purpose

Single-model AUC survives float32 because AUC is rank-only. A *stacking* meta-model does not: it works on differences far below float32 resolution near p=0 and p=1, and this dataset exists to be stacked. Values are left exactly as the models produced them; nothing is clipped or rounded.

## Provenance and method

* **Data.** Kaggle Playground Series `playground-series-s6e10`, target `satisfaction`, metric ROC AUC. 699,635 training rows, 299,844 test rows, positive rate 0.443573. Target encoding as read from train.csv: `bool True -> 1, False -> 0`.
* **The original dataset is used as features, never as rows.** The competition data is described as inspired by the Airline Satisfaction Data dataset uploaded to Kaggle by Arseniy Shutko (`arseniyshutko/binary-aviation-satisfaction-129k`, CC0-1.0). Members **j**, **k**, **l**, **n**, **p**, **q**, **r**, **t** and **u** use it through models fitted on the original rows only and through per-value target means computed from the original labels; neither reads a competition label. Those CC0 rows, aligned to train.csv, are also published as `dariushafshar/s6e10-original-data-aligned`. Every single-model member is trained on competition rows only; no such member adds original rows to its training data.
* **Folds.** Frozen once, before any model was trained, and never recomputed: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over train.csv in file order. Fold sizes [139927, 139927, 139927, 139927, 139927].
* **Fitting discipline.** Every transformer (category vocabularies, imputation medians, quantile transforms, spline knots, one-hot vocabularies, target-encoding statistics) is fit on the training part of each fold and only applied to the validation fold and the test frame. Deliberate exceptions, all disclosed above and visible in `manifest.csv`: **a**, **h**, **i**, **j**, **k**, **l**, **m**, **n**, **p**, **q**, **r**, **s**, **t** and **u** early-stop on the held-out validation fold itself, which makes those members' OOF mildly optimistic; **h**, **i**, **j**, **k**, **l**, **m**, **n**, **p** and **q** use label-free statistics computed over train.csv and test.csv together (frequency encoding in h and i, value counts and the route profile in the members after them), which read test feature VALUES (never labels, of which test.csv has none), so those features are transductive rather than leaky; the auxiliary-task expected values in p, q, r, s, t and u are the same kind of statistic (r to u reuse the exact values p and q were built with) (a label-free model over train.csv and test.csv rows); **j**, **k**, **l**, **n**, **p** and **q** also use features computed from the original dataset's labels, which is external data, not a competition label. **j**, **k**, **n**, **p** and **q** target-encode inside the training fold with sklearn's TargetEncoder, which cross-fits so that no training row sees its own label.
* **Verification.** `verify.py` reloads every shipped array, recomputes all 5 fold AUCs against the labels, and refuses to package if any of them disagrees with the training log by more than 1e-06. It also rejects byte-identical or rank-correlation > 0.9999 members, so nothing here is padding, and it regenerates the fold partition from the recipe above to confirm the shipped file matches it. Members j to q were not retrained here: their arrays come from my second pipeline (Kaggle kernels, CPU, and 2x T4 GPU for o; same folds) and passed this same gate, after their fold AUCs were recomputed from the arrays and matched that pipeline's own log. The permutation leak test, which retrains, covers members a to i. Members r to u were trained by my own script with the same feature code as j to q, then checked by `measure.py`: row counts, finite values, the frozen folds regenerated from the recipe, every fold AUC recomputed from the arrays, and the near-duplicate refusal against all 17 earlier members.

## Credits

Ideas taken from other people's public notebooks on this competition, by the display name Kaggle shows for each author (checked through the Kaggle API on 2026-10-01), with the members that use each one:

* **goodpjw2008** (@goodpjw2008), "S6E10 | Route-ID FE + OG-Model Stack | LB 0.96108", https://www.kaggle.com/code/goodpjw2008/s6e10-route-id-fe-og-model-stack-lb-0-96108: the route profile (per-Flight-Distance means of the other columns, no target) (members j, k, l, m, n, p, q); an XGBoost depth-8 model fitted on the original data only, used as a feature (members j, k, l, n, p, q); the target-encoding key set, the union of the keys in the three notebooks (members j, k, n, p, q); engineered inputs and original-model logits as RealMLP inputs, and the adapted RealMLP schedule (its weight decay and the lin_cos_log_15 learning-rate schedule) (member n).
* **Arhan canli** (@arhancanli12), "S6E10 | Generator Fingerprints | LGBM CV 0.9610", https://www.kaggle.com/code/arhancanli12/s6e10-generator-fingerprints-lgbm-cv-0-9610: Flight Distance digit features (% 100, % 1000, // 10, // 100 and length) (members j, k, l, m, p, q); a LightGBM model fitted on the original data only, used as a feature (members j, k, l, n, p, q); original-data target means per value, and per travel type x class x customer type (members j, k, l, n, p, q); GPT-2 token keys of the numeric values (members j, k, p, q); the target-encoding key set, the union of the keys in the three notebooks (members j, k, n, p, q); the LightGBM schedule: learning rate 0.02, 127 leaves, min_child_samples 100, feature fraction 0.5 (members j, p, q); the LightGBM leaf settings: 127 leaves, min_child_samples 100, feature fraction 0.5 (members k, m).
* **인공저능연구소( Artificial Idiocy Lab )** (@megayak), "S6E10 Honest OOF Stack", https://www.kaggle.com/code/megayak/s6e10-honest-oof-stack: Flight Distance digit features (% 100, % 1000, // 10, // 100 and length) (members j, k, l, m, p, q); value counts over train + test (members j, k, l, m, n, p, q); original-data target means per value, and per travel type x class x customer type (members j, k, l, n, p, q); fold-safe target encoding with sklearn TargetEncoder cross-fitted inside each training fold (members j, k, n, p, q); the target-encoding key set, the union of the keys in the three notebooks (members j, k, n, p, q); the LightGBM schedule: learning rate 0.02, 127 leaves, min_child_samples 100, feature fraction 0.5 (members j, p, q); the LightGBM leaf settings: 127 leaves, min_child_samples 100, feature fraction 0.5 (members k, m).
* **Vladimir Demidov** (@yekenot), "PS|S6|E10: RealMLP · PyTabKit", https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit: the RealMLP hyper-parameters and a categorical twin of every numeric column (member n).
* **goodpjw2008** (@goodpjw2008), "S6E10 | Auxiliary-Task Features | LB 0.96129", https://www.kaggle.com/code/goodpjw2008/s6e10-auxiliary-task-features-lb-0-96129: the auxiliary-task features (per service rating, its expected value given the other 20 columns, from a label-free model over train + test) (members p, q).
* **goodpjw2008** (@goodpjw2008), "S6E10 | TabPFN + Route Categories | LB 0.96160", https://www.kaggle.com/code/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160: the TabPFN input set (raw columns plus Flight Distance // 10, with Flight Distance and the route group declared categorical) (member o).
* **Hemingweb** (@hemingweb), "S6E10 EXP01 TabPFN35 Full Context", https://www.kaggle.com/code/hemingweb/s6e10-exp01-tabpfn35-full-context: the TabPFN worker, the two-GPU fold scheduler and the chunked KV-cache patch that fits the full 559,708-row context on a 16 GB T4, used as published (member o).

Members built here whose recipe borrows from a public S6E9 notebook:

* **h**: the frequency-encoding step comes from the public S6E9 single-XGB notebook by Дворкин Евгений Владимирович (@evgendvorkin), kaggle.com/code/evgendvorkin/s6e9-single-xgb-cv-0-94608, where it was measured on S6E9. This member measures it on S6E10 (compare h with a in manifest.csv). The model, folds and code are mine.
* **i**: recipe (hyperparameters, label encoding, digit features over every column, and frequency encoding) from the public S6E9 single-XGB notebook by Дворкин Евгений Владимирович (@evgendvorkin), kaggle.com/code/evgendvorkin/s6e9-single-xgb-cv-0-94608. The notebook's eight S6E9 interaction features are left out because they named S6E9 columns. Re-run here on this library's frozen folds, so this AUC is not comparable to any number in his notebook. Folds, code and verification are mine.

The public pool I measured the v5 members against: **Chris Deotte**'s datasets `cdeotte/s6e10-124-public-oof-and-test-predictions` and `cdeotte/s6e10-tfm-oof-and-test-predictions` (used in his notebook "Stacking TFMs with Public OOF", https://www.kaggle.com/code/cdeotte/stacking-tfms-with-public-oof), which package the 124 public members that **Kratosyan** collected in "S6E10 | Stacking every public OOF library" (https://www.kaggle.com/code/kratosyan/s6e10-stacking-every-public-oof-library), plus 11 tabular foundation-model pairs. Each member of that pool is credited to its author in those two notebooks. I used the pool locally to measure; none of it is redistributed here.


Version 6 ideas, by the display name Kaggle shows for each author (checked through the Kaggle API on 2026-10-13):

* **Sachith7** (@sachith7): the idea of a RealMLP fed only the 21 raw columns and the auxiliary-task values, trained on 10 folds (their member `sa_rmlp_raw_aux_k10`; I saw its name and its predictions, not its code) (member s); and the idea of giving the auxiliary-task values to a CatBoost that builds categorical CTRs (their member `sa_cat_ctr_auxall_k10`; name and predictions only) (member u); their public members in `cdeotte/s6e10-124-public-oof-and-test-predictions` (also in their dataset `sachith7/s6e10-stack-oof-predictions`).
* **Mitudru Dutta** (@mitudru): the choice of a shallow, depth-5 XGBoost (their member `mi_xgb_d5`; name and predictions only) (member t); their public member `mi_xgb_d5` in `cdeotte/s6e10-124-public-oof-and-test-predictions` (from their notebook `mitudru/s6e10-analysis-first-equality-blocks-stack`).
* Members r to u also use ideas already credited above for members j to q (the auxiliary-task features, the RealMLP recipe and schedule, the feature blocks); the borrowed-ideas table lists them member by member.

`stack247` is built from the public pairs of: Chris Deotte (`cdeotte/s6e10-124-public-oof-and-test-predictions`, the 124 pairs Kratosyan collected, and `cdeotte/s6e10-tfm-oof-and-test-predictions`), AbdullahSafwan333, Ahmet Kağan Koral, Aman Atar, Dmitriy Kolmikov, Hermengardo, Koumei Maki, Kozykorpesh, Kratosyan, Sachith7, Sadam Torres, Vladimir Demidov, goodpjw2008, and, inside the 124, Akmal Xodarev (Busya PRIME), Arhan canli, Mitudru Dutta, Naji, Sachith7, Shelton Wang, goodpjw2008, samanyu1808 and Hemingweb, thisray, 인공저능연구소( Artificial Idiocy Lab ). `stack247_inputs.csv` names the author of each of the 247; the notebook links every source.

## Reproduce / verify

`manifest.csv` and `rank_correlation.csv` are recomputed from the shipped `.npy` files at packaging time. To check me, load `train.csv` in file order and run:

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(train.satisfaction.astype(int), np.load('oof_a.npy'))
# 0.959568
```

## Licence

CC0-1.0, public domain. Use them for anything. A link back is appreciated, not required. One caveat: member o's predictions come from TabPFN-3.5 by Prior Labs, whose model licence is non-commercial; check it before any commercial use of that column. The same applies to `stack247`, whose inputs include TabPFN members; its other inputs keep their authors' licences, and only the stacked prediction is published here.
