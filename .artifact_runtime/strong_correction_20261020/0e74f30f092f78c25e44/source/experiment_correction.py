"""Two outer folds of strict, nested competition-ensemble correction."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import expit
from sklearn.metrics import roc_auc_score, log_loss

from correction_support import (BASE_ROUNDS, WEIGHTS, BRANCHES, RichFeatures,
    combined, score, nested_plan, CorrectionFeatures, fit_branch,
    restored_predict, offset_pool_prediction)
from v2_training import ROOT, load_data, write_json
from report_neural import paired

OUT = ROOT / '.artifact_runtime/strong_correction_20261020'
SOURCE_FILES = ['experiment_correction.py', 'correction_support.py',
    'experiment_realmlp_factorial.py', 'optimized_features.py',
    'optimized_training.py', 'v3_features.py', 'v2_training.py', 'report_neural.py',
    'experiment_neural.py', 'tests/test_correction_contract.py']
DATA_FILES = ['train.csv', 'test.csv', 'external_data/train.csv', 'external_data/test.csv',
    'optimized_recipe.json', '.artifact_runtime/optimized_cache/train_priors.npy',
    '.artifact_runtime/optimized_cache/test_priors.npy', '.artifact_runtime/optimized_cache/metadata.json',
    '.artifact_runtime/v3/aux_train.npy', '.artifact_runtime/v3/aux_test.npy',
    '.artifact_runtime/v3/auxiliary.json']
CAT_PARAMS = dict(iterations=2500, depth=6, learning_rate=.03, l2_leaf_reg=20,
    max_ctr_complexity=2, loss_function='Logloss', eval_metric='AUC',
    task_type='GPU', thread_count=4, allow_writing_files=False, verbose=False)
PROTOCOL = dict(outer_folds=[0, 1], outer_splits=10, split_seed=20261020,
    inner_folds=5, internal_stop_fraction=.08, base_rounds=BASE_ROUNDS,
    legacy_nn_original_schedule_epochs=128, legacy_nn_fixed_stop_epoch=63,
    nn_validation_fraction=0., lean_epochs=3, lean_seeds=[42, 20261019],
    weights=WEIGHTS, blend_logit_weights=[.65, .35],
    modes=['plain', 'offset'], correction_params=CAT_PARAMS, patience=150,
    no_outer_early_stopping=True, raw_features_and_numeric_copies_only=True,
    train_base_margins_are_inner_oof=True, selection_stop_excluded_from_base_oof_fit=True,
    refit_all_outer_training_after_inner_stopping=True,
    historical_data_and_hyperparameter_selection_seen=True,
    base_rounds_source='completed_transfer_screen_74f731e0c4b9330ecfac',
    screen_min_gain=.0003, both_folds_positive=True, require_interval_lower_positive=True,
    no_continuous_blend_search=True, no_kaggle_submission=True)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def setup():
    OUT.mkdir(exist_ok=True)
    hashes = {n: sha(ROOT/n) for n in SOURCE_FILES + DATA_FILES}
    versions = {n: importlib.metadata.version(n) for n in
                ['numpy', 'pandas', 'scikit-learn', 'torch', 'pytabkit', 'xgboost', 'catboost', 'lightgbm']}
    signature = hashlib.sha256(json.dumps([PROTOCOL, hashes, versions], sort_keys=True).encode()).hexdigest()[:20]
    directory = OUT/signature
    directory.mkdir(exist_ok=True)
    path = directory/'protocol.json'
    payload = dict(**PROTOCOL, hashes=hashes, versions=versions, signature=signature)
    if path.exists():
        assert read(path) == payload
    else:
        write_json(path, payload)
    preserved = OUT/'initial_preserved.json'
    if not preserved.exists():
        write_json(preserved, {p.name: sha(p) for p in ROOT.iterdir() if p.is_file()
                              and p.name not in ['experiment_correction.py', 'correction_support.py']})
    preserve_check()
    snap = directory/'source'; snap.mkdir(exist_ok=True)
    for n in SOURCE_FILES:
        dest = snap/n.replace('/', '__')
        if dest.exists():
            assert sha(dest) == hashes[n]
        else:
            dest.write_bytes((ROOT/n).read_bytes())
    write_json(OUT/'latest.json', dict(directory=str(directory), signature=signature))
    return directory


def preserve_check():
    for n, h in read(OUT/'initial_preserved.json').items():
        assert sha(ROOT/n) == h, 'Protected file changed: '+n


def data():
    x, xt, y, ox, oy, prior, pt, stats, metadata = load_data()
    aux = np.load(ROOT/'.artifact_runtime/v3/aux_train.npy')
    assert len(aux) == len(x)
    # Independent source-only auxiliary predictions have already been verified;
    # their exact arrays and metadata are frozen in DATA_FILES.
    return x, y, prior, aux, stats, read(ROOT/'optimized_recipe.json')['components']


def save_prediction(path, idx, p):
    assert len(idx) == len(p) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.savez_compressed(path, index=idx, prediction=p)


def bundle(folder, fit, query, dataset, seed, checkpoints=False):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'indices.npz').exists():
        with np.load(folder/'indices.npz') as z:
            np.testing.assert_array_equal(z['fit'], fit)
            np.testing.assert_array_equal(z['query'], query)
    else:
        np.savez_compressed(folder/'indices.npz', fit=fit, query=query)
    assert not np.intersect1d(fit, query).size
    x, y, prior, aux, stats, recipes = dataset
    values = {}
    for name in BRANCHES:
        path = folder/(name+'.npz')
        if path.exists() and (folder/(name+'.json')).exists():
            with np.load(path) as z:
                np.testing.assert_array_equal(z['index'], query)
                values[name] = z['prediction'].copy()
            if checkpoints:
                assert (folder/(name+'.joblib')).exists()
            continue
        print('BASE_START', folder.relative_to(OUT), name, len(fit), len(query), flush=True)
        p, detail = fit_branch(name, x.iloc[fit], y[fit], x.iloc[query], prior[fit],
            prior[query], aux[fit], aux[query], stats, recipes, seed,
            folder/(name+'.joblib') if checkpoints else None)
        save_prediction(path, query, p)
        write_json(folder/(name+'.json'), detail)
        values[name] = p
        print('BASE_DONE', name, round(detail['seconds'], 1), 'seconds', flush=True)
    p = combined(values)
    save_prediction(folder/'combined.npz', query, p)
    return p


def crossfit(folder, allowed, folds, dataset, seed):
    oof = np.full(len(dataset[0]), np.nan)
    counts = np.zeros(len(oof), dtype=int)
    for k, split in enumerate(folds):
        p = bundle(folder/f'inner_{k}', split['fit'], split['query'], dataset, seed)
        oof[split['query']] = p
        counts[split['query']] += 1
    assert np.all(counts[allowed] == 1) and np.sum(counts) == len(allowed)
    save_prediction(folder/'oof.npz', allowed, oof[allowed])
    return oof[allowed]


def select_correction(folder, mode, x, y, r, s, fit_pred, stop_pred, seed):
    path = folder/(mode+'_selection.json')
    if path.exists():
        return read(path)
    tf = CorrectionFeatures()
    a = tf.fit_transform(x.iloc[r], score(fit_pred))
    b = tf.transform(x.iloc[s], score(stop_pred))
    a_pool = Pool(a, label=y[r], cat_features=tf.cats,
                  baseline=score(fit_pred) if mode == 'offset' else None)
    b_pool = Pool(b, label=y[s], cat_features=tf.cats,
                  baseline=score(stop_pred) if mode == 'offset' else None)
    model = CatBoostClassifier(**CAT_PARAMS, random_seed=seed)
    begin = time.time()
    model.fit(a_pool, eval_set=b_pool, early_stopping_rounds=PROTOCOL['patience'],
              use_best_model=True)
    p = (offset_pool_prediction(model, b, score(stop_pred), tf.cats) if mode == 'offset'
         else model.predict_proba(b)[:, 1])
    base_auc = float(roc_auc_score(y[s], stop_pred))
    candidate_auc = float(roc_auc_score(y[s], p))
    rounds = int(model.tree_count_) if candidate_auc > base_auc else 0
    d = dict(mode=mode, seed=seed, selected_rounds=rounds,
        best_model_rounds=int(model.tree_count_), base_stopping_auc=base_auc,
        candidate_stopping_auc=candidate_auc, seconds=time.time()-begin,
        features=list(a), cats=tf.cats, curves=model.get_evals_result(),
        selected_by_internal_stop_only=True, fallback_to_zero=rounds == 0)
    model.save_model(str(folder/(mode+'_selection.cbm')))
    joblib.dump(tf, folder/(mode+'_selection_features.joblib'))
    save_prediction(folder/(mode+'_stopping.npz'), s, p)
    write_json(path, d)
    print('INNER_SELECTION', mode, rounds, 'gain', candidate_auc-base_auc, flush=True)
    del model, a_pool, b_pool, a, b
    gc.collect()
    return d


def final_correction(folder, mode, selected, x, y, t, e, oof, base, seed):
    path = folder/(mode+'_outer.npz')
    if path.exists():
        return
    tf = CorrectionFeatures()
    a = tf.fit_transform(x.iloc[t], score(oof))
    b = tf.transform(x.iloc[e], score(base))
    rounds = selected['selected_rounds']
    if rounds == 0:
        p = base.copy()
    else:
        model = CatBoostClassifier(**dict(CAT_PARAMS, iterations=rounds), random_seed=seed)
        model.fit(Pool(a, label=y[t], cat_features=tf.cats,
                       baseline=score(oof) if mode == 'offset' else None))
        p = (offset_pool_prediction(model, b, score(base), tf.cats) if mode == 'offset'
             else model.predict_proba(b)[:, 1])
        model.save_model(str(folder/(mode+'_final.cbm')))
        del model
    joblib.dump(tf, folder/(mode+'_final_features.joblib'))
    save_prediction(path, e, p)


def run_fold(directory, outer, dataset):
    x, y, prior, aux, stats, recipes = dataset
    p = nested_plan(y, outer, seed=PROTOCOL['split_seed'], inner_folds=PROTOCOL['inner_folds'])
    folder = directory/f'outer_{outer}'; folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'plan.npz', **{k: p[k] for k in
        ['outer_train', 'outer_eval', 'selection_fit', 'selection_stop']})
    r, s, t, e = [p[k] for k in ['selection_fit', 'selection_stop', 'outer_train', 'outer_eval']]
    seed = 42 + outer
    oof_r = crossfit(folder/'selection', r, p['selection'], dataset, seed)
    s_pred = bundle(folder/'selection/full_fit', r, s, dataset, seed)
    selected = {mode: select_correction(folder, mode, x, y, r, s, oof_r, s_pred, seed)
                for mode in PROTOCOL['modes']}
    # Fixed after internal stopping; no outer labels have been scored yet.
    write_json(folder/'frozen_selection.json', selected)
    oof_t = crossfit(folder/'refit', t, p['refit'], dataset, seed)
    base = bundle(folder/'refit/full_fit', t, e, dataset, seed, checkpoints=True)
    for mode in PROTOCOL['modes']:
        final_correction(folder, mode, selected[mode], x, y, t, e, oof_t, base, seed)
    write_json(folder/'complete.json', dict(complete=True, outer_fold=outer,
        selection_fit_rows=len(r), selection_stop_rows=len(s), refit_rows=len(t), eval_rows=len(e),
        correction_training_margin='inner_oof', base_prediction_margin='full_outer_fit',
        margin_distribution=dict(oof_mean=float(score(oof_t).mean()),
            oof_std=float(score(oof_t).std()), eval_mean=float(score(base).mean()),
            eval_std=float(score(base).std()))))
    preserve_check()
    print('OUTER_TRAINING_COMPLETE', outer, 'outer scoring deferred', flush=True)


def summarize(directory, y):
    records = []
    joined = {k: [] for k in ['index', 'base'] + PROTOCOL['modes']}
    for outer in PROTOCOL['outer_folds']:
        f = directory/f'outer_{outer}'
        assert read(f/'complete.json')['complete']
        with np.load(f/'refit/full_fit/combined.npz') as z:
            idx, base = z['index'].copy(), z['prediction'].copy()
        d = dict(outer=outer, rows=len(idx), baseline_auc=float(roc_auc_score(y[idx], base)), modes={})
        joined['index'].append(idx); joined['base'].append(base)
        for mode in PROTOCOL['modes']:
            with np.load(f/(mode+'_outer.npz')) as z:
                np.testing.assert_array_equal(z['index'], idx)
                p = z['prediction'].copy()
            joined[mode].append(p)
            d['modes'][mode] = dict(auc=float(roc_auc_score(y[idx], p)), vs_base=paired(y[idx], p, base),
                logloss=float(log_loss(y[idx], p)), selected_rounds=read(f/(mode+'_selection.json'))['selected_rounds'])
        d['offset_minus_plain'] = paired(y[idx], joined['offset'][-1], joined['plain'][-1])
        records.append(d)
    joined = {k: np.concatenate(v) for k, v in joined.items()}
    idx = joined['index']; assert len(np.unique(idx)) == len(idx)
    result = dict(folds=records, rows=len(idx), baseline_auc=float(roc_auc_score(y[idx], joined['base'])),
                  modes={}, historical_selection_limitations=True)
    for mode in PROTOCOL['modes']:
        interval = paired(y[idx], joined[mode], joined['base'])
        gate = (interval['delta'] >= PROTOCOL['screen_min_gain'] and interval['interval95'][0] > 0
                and all(r['modes'][mode]['vs_base']['delta'] > 0 for r in records))
        result['modes'][mode] = dict(auc=float(roc_auc_score(y[idx], joined[mode])),
            vs_base=interval, passed=gate)
    result['offset_minus_plain'] = paired(y[idx], joined['offset'], joined['plain'])
    result['any_passed'] = any(v['passed'] for v in result['modes'].values())
    result['decision'] = 'requires_frozen_second_seed' if result['any_passed'] else 'stopped_no_full_cv'
    np.savez_compressed(directory/'outer_predictions.npz', **joined)
    write_json(directory/'result.json', result)
    preserve_check()
    print('RESULT', json.dumps(result, ensure_ascii=False), flush=True)
    return result


def probe(directory, dataset):
    path = directory/'probe.json'
    if path.exists():
        assert read(path)['passed']
        return
    x, y, prior, aux, stats, recipes = dataset
    allowed = nested_plan(y, 0)['selection_fit']
    rng = np.random.default_rng(20261020)
    idx = rng.choice(allowed, size=2304, replace=False)
    fit, query = np.sort(idx[:2048]), np.sort(idx[2048:])
    folder = directory/'probe'; folder.mkdir(exist_ok=True)
    p = bundle(folder, fit, query, dataset, 42, checkpoints=True)
    diffs = {}
    for name in BRANCHES:
        q = restored_predict(folder/(name+'.joblib'), x.iloc[query], prior[query], aux[query])
        with np.load(folder/(name+'.npz')) as z:
            diffs[name] = float(np.max(abs(q-z['prediction'])))
        assert diffs[name] < 1e-6
    write_json(path, dict(passed=True, fit_rows=len(fit), query_rows=len(query),
        fit_and_query_within_selection_fit=True, no_query_labels_scored=True,
        checkpoint_prediction_max_differences=diffs))


def main(action):
    directory = setup()
    if action == 'summarize':
        y = pd.read_csv(ROOT/'train.csv', usecols=['satisfaction']).satisfaction.to_numpy()
        summarize(directory, y)
        return
    dataset = data()
    probe(directory, dataset)
    if action == 'probe':
        return
    for outer in PROTOCOL['outer_folds']:
        run_fold(directory, outer, dataset)
    summarize(directory, dataset[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', nargs='?', choices=['all', 'probe', 'summarize'], default='all')
    main(parser.parse_args().action)
