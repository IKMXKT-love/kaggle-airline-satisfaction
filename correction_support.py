"""Fold-local features, fixed first-stage recipe and binary offset fitting."""
import gc
import time
from pathlib import Path

import joblib
import cloudpickle
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold, train_test_split

from optimized_features import CompetitionFeaturePipeline, NativeCategoryEncoder
from v3_features import RouteFeatures, keys, CONTEXT
from experiment_realmlp_factorial import LeanFeatures, REFERENCE_PARAMS

WEIGHTS = dict(old_xgb=.07, old_cat=.21, control=.28, realmlp=.14, lgb=.30)
BRANCHES = list(WEIGHTS) + ['lean_42', 'lean_20261019']
# Frozen from the previously completed, independent-stopping screen. These
# are historical hyperparameters, not newly selected using current outer labels.
BASE_ROUNDS = dict(old_xgb=435, old_cat=907, control=623, realmlp=63, lgb=641)


def score(p):
    p = np.asarray(p, dtype=np.float64)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def combined(predictions):
    assert set(predictions) == set(BRANCHES)
    v4 = sum(w * np.asarray(predictions[n], dtype=np.float64) for n, w in WEIGHTS.items())
    lean = (np.asarray(predictions['lean_42'], dtype=np.float64) + predictions['lean_20261019']) / 2
    return expit(.65 * score(v4) + .35 * score(lean))


def nested_plan(labels, outer_fold, seed=20261020, inner_folds=5):
    y = np.asarray(labels)
    all_idx = np.arange(len(y))
    t, e = list(StratifiedKFold(10, shuffle=True, random_state=seed).split(all_idx, y))[outer_fold]
    r, s = train_test_split(t, test_size=.08, random_state=seed + outer_fold,
                            stratify=y[t])
    r, s, t, e = [np.sort(v) for v in [r, s, t, e]]
    plan = dict(outer_train=t, outer_eval=e, selection_fit=r, selection_stop=s)
    for phase, idx in [('selection', r), ('refit', t)]:
        folds = []
        for a, b in StratifiedKFold(inner_folds, shuffle=True,
                                   random_state=seed + 100 + outer_fold).split(idx, y[idx]):
            folds.append(dict(fit=idx[a], query=idx[b]))
        plan[phase] = folds
    validate_plan(plan, len(y))
    return plan


def validate_plan(p, n):
    t, e, r, s = [np.asarray(p[k]) for k in
                  ['outer_train', 'outer_eval', 'selection_fit', 'selection_stop']]
    assert len(np.unique(np.r_[t, e])) == n and len(t) + len(e) == n
    assert not np.intersect1d(r, s).size and np.array_equal(np.sort(np.r_[r, s]), t)
    for phase, allowed in [('selection', r), ('refit', t)]:
        seen = []
        for fold in p[phase]:
            a, b = fold['fit'], fold['query']
            assert not np.intersect1d(a, b).size
            assert np.array_equal(np.sort(np.r_[a, b]), allowed)
            assert not np.intersect1d(np.r_[a, b], e).size
            if phase == 'selection':
                assert not np.intersect1d(np.r_[a, b], s).size
            seen.extend(b)
        assert np.array_equal(np.sort(seen), allowed)


class RichFeatures:
    def fit_transform(self, x, y, prior, aux):
        from sklearn.preprocessing import TargetEncoder
        self.route = RouteFeatures().fit(x)
        self.te = TargetEncoder(target_type='binary', smooth=20,
                                cv=StratifiedKFold(5, shuffle=True, random_state=42))
        k = keys(x)
        self.te_names = list(k)
        a = self.build(x, prior, aux, self.te.fit_transform(k, y))
        self.native = NativeCategoryEncoder().fit(a)
        self.median = a.select_dtypes(include='number').median().fillna(0)
        return self.native.transform(a)

    def build(self, x, prior, aux, te):
        a = pd.concat([x.copy(), self.route.transform(x),
                       pd.DataFrame(np.asarray(te, dtype=np.float32),
                                    columns=self.te_names, index=x.index)], axis=1)
        a['teacher_logit'] = score(prior)
        for i in range(aux.shape[1]):
            a[f'aux_expected_rating_{i}'] = aux[:, i]
        return a

    def transform(self, x, prior, aux):
        return self.native.transform(self.build(x, prior, aux, self.te.transform(keys(x))))


def nn_frame(a, median):
    a = a.copy()
    a[median.index] = a[median.index].fillna(median)
    for c in CONTEXT:
        a[c] = a[c].astype('string').fillna('__MISSING__').astype(str)
    return a


def text_categories(a):
    a = a.copy()
    for c in a.select_dtypes(include='category'):
        a[c] = a[c].astype('string').fillna('__MISSING__').astype(str)
    return a


def fit_branch(name, x, y, query, prior, query_prior, aux, query_aux,
               stats, old_recipes, seed, model_path=None):
    """No query labels are accepted. Every transform is fitted on x only."""
    import torch
    start = time.time()
    assert not {'id', 'satisfaction'} & set(x)
    assert set(x) == set(query)
    if name in ['old_xgb', 'old_cat']:
        config = old_recipes[0 if name == 'old_xgb' else 1]
        tf = CompetitionFeaturePipeline(use_te=config.get('use_te', False),
            original_statistics=stats, numeric_categories=config.get('numeric_categories', False))
        a = tf.fit_transform(x, y, prior)
        b = tf.transform(query, query_prior)
        if name == 'old_xgb':
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=BASE_ROUNDS[name], learning_rate=.04,
                max_depth=7, min_child_weight=20, subsample=.9, colsample_bytree=1.,
                reg_lambda=5, tree_method='hist', device='cuda', enable_categorical=True,
                n_jobs=4, random_state=seed, eval_metric='auc')
        else:
            from catboost import CatBoostClassifier
            cats = list(a.select_dtypes(include='category'))
            a, b = text_categories(a), text_categories(b)
            model = CatBoostClassifier(iterations=BASE_ROUNDS[name], learning_rate=.04,
                depth=8, l2_leaf_reg=5, loss_function='Logloss', task_type='GPU',
                cat_features=cats, random_seed=seed, thread_count=4,
                max_ctr_complexity=2, allow_writing_files=False, verbose=False)
        kind = 'old'
        model.fit(a, y, verbose=False)
        if name == 'old_xgb':
            model.set_params(device='cpu')
    elif name.startswith('lean_'):
        from pytabkit import RealMLP_TD_Classifier
        tf = LeanFeatures()
        a = tf.fit_transform(x, y)
        b = tf.transform(query)
        # Two independent network seeds, held fixed across bundles.
        network_seed = int(name.split('_')[1]) + seed - 42
        model = RealMLP_TD_Classifier(device='cuda', random_state=network_seed,
            n_cv=1, n_refit=0, val_fraction=0., n_threads=4, verbosity=0,
            val_metric_name='1-auc_ovr', **REFERENCE_PARAMS)
        model.fit(a, y, cat_col_names=tf.cats)
        kind = 'lean'
    else:
        tf = RichFeatures()
        a = tf.fit_transform(x, y, prior, aux)
        b = tf.transform(query, query_prior, query_aux)
        kind = 'rich'
        if name == 'control':
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=BASE_ROUNDS[name], learning_rate=.03,
                max_depth=6, min_child_weight=20, subsample=.9, colsample_bytree=.8,
                reg_lambda=10, tree_method='hist', device='cuda', enable_categorical=True,
                n_jobs=4, random_state=seed, eval_metric='auc')
            model.fit(a, y, verbose=False)
            model.set_params(device='cpu')
        elif name == 'lgb':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=BASE_ROUNDS[name], learning_rate=.025,
                num_leaves=63, min_child_samples=100, colsample_bytree=.85,
                subsample=.9, subsample_freq=1, reg_lambda=10, reg_alpha=.1,
                max_bin=255, verbosity=-1, n_jobs=4, random_state=seed,
                deterministic=True, force_col_wise=True)
            model.fit(a, y)
        else:
            assert name == 'realmlp'
            from pytabkit import RealMLP_TD_Classifier
            a, b = nn_frame(a, tf.median), nn_frame(b, tf.median)
            model = RealMLP_TD_Classifier(device='cuda', random_state=seed,
                n_cv=1, n_refit=0, val_fraction=0., n_epochs=128, stop_epoch=BASE_ROUNDS[name],
                use_early_stopping=False, batch_size=1024, n_threads=4, verbosity=0,
                use_ls=False, val_metric_name='1-auc_ovr')
            model.fit(a, y, cat_col_names=CONTEXT)
    prediction = np.asarray(model.predict_proba(b)[:, 1], dtype=np.float64)
    assert np.isfinite(prediction).all()
    if model_path is not None:
        with Path(model_path).open('wb') as handle:
            cloudpickle.dump(dict(model=model, transform=tf, kind=kind, name=name), handle)
    details = dict(seconds=time.time()-start, train_rows=len(x), query_rows=len(query),
                   name=name, seed=seed, features=list(a), fit_params=str(getattr(model, 'fit_params_', None)),
                   n_rounds=3 if name.startswith('lean_') else BASE_ROUNDS[name])
    del model, a, b, tf
    gc.collect()
    torch.cuda.empty_cache()
    return prediction, details


def restored_predict(path, query, prior, aux):
    with Path(path).open('rb') as handle:
        saved = cloudpickle.load(handle)
    tf, name, kind, model = [saved[k] for k in ['transform', 'name', 'kind', 'model']]
    if kind == 'old':
        b = tf.transform(query, prior)
        if name == 'old_cat':
            b = text_categories(b)
    elif kind == 'lean':
        b = tf.transform(query)
    else:
        b = tf.transform(query, prior, aux)
        if name == 'realmlp':
            b = nn_frame(b, tf.median)
    p = np.asarray(model.predict_proba(b)[:, 1], dtype=np.float64)
    del saved, model, b, tf
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return p


class CorrectionFeatures:
    """21 raw fields, 17 categorical numeric copies and the fixed base score."""
    @staticmethod
    def raw(x, base):
        assert not {'id', 'satisfaction'} & set(x)
        a = x.copy()
        for c in x.select_dtypes(include='number'):
            a[c + '__category'] = x[c].astype('string').fillna('__MISSING__')
        a['current_logit'] = np.asarray(base, dtype=np.float64)
        return a

    def fit_transform(self, x, base):
        a = self.raw(x, base)
        self.cats = list(a.select_dtypes(exclude='number'))
        self.vocab = {c: set(a[c].astype('string').fillna('__MISSING__')) for c in self.cats}
        self.medians = a.select_dtypes(include='number').median().fillna(0)
        return self.transform(x, base)

    def transform(self, x, base):
        a = self.raw(x, base)
        for c in self.cats:
            v = a[c].astype('string').fillna('__MISSING__')
            a[c] = v.where(v.isin(self.vocab[c]), '__UNKNOWN__').astype(str)
        a[self.medians.index] = a[self.medians.index].fillna(self.medians)
        assert list(a.columns)[-1] == 'current_logit'
        return a


def offset_pool_prediction(model, frame, base, cats):
    from catboost import Pool
    base = np.asarray(base, dtype=np.float64)
    explicit = expit(base + model.predict(frame, prediction_type='RawFormulaVal'))
    automatic = model.predict_proba(Pool(frame, cat_features=cats, baseline=base))[:, 1]
    # Pool baselines may be stored in float32 by CatBoost.
    np.testing.assert_allclose(explicit, automatic, atol=2e-7, rtol=1e-6)
    return explicit
