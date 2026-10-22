# Predicting Airline Passenger Satisfaction

Binary classification of airline passenger satisfaction for Kaggle's [Playground Series – Season 6, Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10). The project is a single, end-to-end notebook: data download, EDA, a shared preprocessing pipeline, a four-model baseline comparison, cross-validation, Optuna hyperparameter tuning, and generating the submission file.

**Result:** a tuned XGBoost model scores **0.9590 ROC-AUC** on a held-out dev set (5-fold CV ROC-AUC 0.9585 ± 0.0004 for the untuned baseline).

---

## The Problem

Each row is a passenger survey response. The task is to predict the probability that a passenger is **satisfied** (`satisfaction = True`). Submissions are scored on **ROC-AUC**.

| Split | Rows | Columns |
|---|---|---|
| Train | 699,635 | 21 features + target |
| Test | 299,844 | 21 features |

**Features**

- **Demographics / trip context (categorical):** `Gender`, `Customer Type`, `Type of Travel`, `Class`
- **Service ratings (0–5):** 14 ratings, including inflight wifi, online boarding, seat comfort, inflight entertainment, leg room, cleanliness, and others
- **Numeric:** `Age`, `Flight Distance`, `Departure Delay in Minutes`, `Arrival Delay in Minutes`

The target is fairly balanced (~44% satisfied). The only missing values are a few hundred in `Arrival Delay in Minutes` (<0.05% of rows).

## Approach

### 1. Exploratory analysis
EDA uses only the training split, so the dev set never influences it. Satisfaction rates by category already point to the strongest signals:

| Feature | Group | Satisfied |
|---|---|---|
| Class | Business / Eco Plus / Eco | 73% / 24% / 17% |
| Type of Travel | Business / Personal | 58% / 10% |
| Customer Type | Loyal / Disloyal | 50% / 20% |
| Gender | Female / Male | 44% / 45% |

Gender carries almost no signal. Travel class and trip purpose carry a lot.

### 2. Preprocessing
A single scikit-learn `ColumnTransformer` is shared by every model, so all candidates see identical inputs and nothing leaks from the dev set:

- Median imputation for `Arrival Delay in Minutes`
- One-hot encoding for the four categorical features (`handle_unknown='ignore'`)
- Pass-through for the ordinal ratings and other numerics
- `StandardScaler` added only for the linear model

### 3. Baseline model comparison
The data is split 80/20 into train and dev sets, stratified on the target. Each model is wrapped in the pipeline and scored on the dev set, with ROC curves, confusion matrices, and classification reports:

| Model | Dev ROC-AUC | Fit time |
|---|---|---|
| **XGBoost** | **0.9587** | 4.6 s |
| CatBoost | 0.9584 | 13.2 s |
| LightGBM | 0.9583 | 4.1 s |
| Logistic Regression | 0.9220 | 0.6 s |

The three gradient-boosted models score within 0.0004 of each other, and all beat the linear baseline by about 0.037 AUC. That gap suggests the target depends on nonlinear interactions between features. XGBoost was selected for its top score at low training cost.

### 4. Cross-validation
5-fold stratified CV on the training split confirms the baseline is stable:

| Metric | Mean ± Std |
|---|---|
| ROC-AUC | 0.9585 ± 0.0004 |
| Accuracy | 0.9258 ± 0.0002 |
| Precision | 0.9266 ± 0.0006 |
| Recall | 0.9043 ± 0.0009 |
| F1 | 0.9153 ± 0.0003 |

### 5. Hyperparameter tuning
**Optuna** ran a 50-trial search over 10 XGBoost hyperparameters, including tree depth, learning rate, regularization, row/column subsampling, `gamma`, and `scale_pos_weight`. Each trial was scored by 5-fold CV ROC-AUC. Folds run in parallel and each model uses one thread, which avoids CPU oversubscription.

Best trial: **CV ROC-AUC 0.9588**, using deeper trees (`max_depth=10`), a low learning rate (~0.01), ~1,460 estimators, and moderate positive-class weighting (`scale_pos_weight≈1.56`).

### 6. Final evaluation

| Model | Dev ROC-AUC | Accuracy | Macro F1 |
|---|---|---|---|
| Tuned XGBoost | **0.9590** | 0.93 | 0.92 |

Tuning traded a little precision for better recall on the satisfied class (recall 0.90 → 0.92).

### 7. Feature importance
The top drivers in the tuned model match what EDA showed:

1. `Class = Business`
2. `Online boarding` rating
3. `Type of Travel` (Business vs. Personal)
4. `Inflight wifi service` rating
5. `Customer Type` (Loyal vs. Disloyal)

Then `Class = Eco`, `Inflight entertainment`, and `Seat comfort`. The digital experience (online boarding, wifi) matters more than delays or food.

## Key Takeaways

- **The gradient-boosted models were close to interchangeable.** XGBoost, LightGBM, and CatBoost scored within 0.0004 of each other, so further gains will likely come from features or ensembling, not from switching algorithms.
- **Tuning helped only a little** (+0.0003 AUC over a sensible baseline). Strong default settings already captured most of the available signal.
- **Who is flying and why matters as much as how the flight went.** Class, trip purpose, and loyalty sit next to the top service ratings in feature importance.

## Project Structure

```
.
├── data/                     # Competition data (downloaded via kagglehub)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   └── 01_predicting_airline_satisfaction.ipynb   # Full end-to-end workflow
├── submissions/
│   └── submission.csv        # Predicted probabilities for the test set
├── pyproject.toml            # Dependencies (managed with uv)
└── uv.lock
```

## Reproducing the Results

**Requirements:** Python 3.14, [uv](https://docs.astral.sh/uv/), and a Kaggle account with API credentials configured. You must also accept the competition rules on Kaggle before downloading.

```bash
git clone <this-repo-url>
cd playground-series-s6e10
uv sync
uv run --with jupyterlab jupyter lab notebooks/01_predicting_airline_satisfaction.ipynb
```

The notebook downloads the competition data into `data/` with `kagglehub` and writes predictions to `submissions/submission.csv`. All randomness is seeded (`RANDOM_STATE = 206`). The Optuna sampler is not seeded, so tuned hyperparameters may vary slightly between runs.

## Tech Stack

`pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `Optuna` · `seaborn` / `matplotlib` · `kagglehub` · `uv`

## Possible Next Steps

- **Refit on all labeled data.** The final model is trained on the 80% training split only. Refitting on train + dev before predicting should give a small boost.
- **Ensemble.** Average or stack the XGBoost, LightGBM, and CatBoost predictions, since their errors are probably not identical.
- **Feature engineering.** Try an aggregate service score, delay ratios or flags, and age/distance bins.
- **Native categorical handling.** Use CatBoost's and LightGBM's native categorical support instead of one-hot encoding.
- **Seed the tuner.** Pass a seeded `TPESampler` to Optuna so tuning runs are fully reproducible.

## Author

**Mike Johnson**
