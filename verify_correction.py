"""Independent cache, scope and checkpoint audit for strong-model correction."""
import gc
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiment_correction import ROOT, OUT, read, sha, preserve_check, write_json, data
from correction_support import restored_predict
from report_neural import paired


def main():
    directory = Path(read(OUT/'latest.json')['directory'])
    config = read(directory/'protocol.json')
    result = read(directory/'result.json')
    for n, h in config['hashes'].items():
        assert sha(ROOT/n) == h, n
    preserve_check()
    from v2_training import load_data
    x, xt, y, ox, oy, prior, pt, stats, metadata = load_data()
    aux = np.load(ROOT/'.artifact_runtime/v3/aux_train.npy')
    auxiliary_signature = hashlib.sha256(b''.join(
        pd.util.hash_pandas_object(v.astype('string'), index=False).to_numpy().tobytes()
        for v in [x, xt, ox])).hexdigest()
    assert read(ROOT/'.artifact_runtime/v3/auxiliary.json')['signature'] == auxiliary_signature
    allidx = np.arange(len(x))
    splits = list(StratifiedKFold(10, shuffle=True, random_state=config['split_seed']).split(allidx, y))
    output = {}; joined = {k: [] for k in ['index', 'base', 'plain', 'offset']}
    for outer in config['outer_folds']:
        f = directory/f'outer_{outer}'
        t, e = map(np.sort, splits[outer])
        r, s = train_test_split(t, test_size=.08, random_state=config['split_seed']+outer, stratify=y[t])
        r, s = np.sort(r), np.sort(s)
        with np.load(f/'plan.npz') as z:
            for k, a in dict(outer_train=t, outer_eval=e, selection_fit=r, selection_stop=s).items():
                np.testing.assert_array_equal(z[k], a)
        branch_diffs = {}; cache_checks = 0
        for phase, allowed, query in [('selection', r, s), ('refit', t, e)]:
            expected = list(StratifiedKFold(config['inner_folds'], shuffle=True,
                random_state=config['split_seed']+100+outer).split(allowed, y[allowed]))
            seen = np.zeros(len(x), dtype=int); oof = np.full(len(x), np.nan)
            for k in range(config['inner_folds']+1):
                d = f/phase/(f'inner_{k}' if k < config['inner_folds'] else 'full_fit')
                if k < config['inner_folds']:
                    a, b = [allowed[v] for v in expected[k]]
                    seen[b] += 1
                else:
                    a, b = allowed, query
                assert not np.intersect1d(a, b).size
                assert not np.intersect1d(a, e).size
                if phase == 'selection':
                    assert not np.intersect1d(a, s).size
                with np.load(d/'indices.npz') as z:
                    np.testing.assert_array_equal(z['fit'], a)
                    np.testing.assert_array_equal(z['query'], b)
                branches = {}
                for name in list(config['weights'])+['lean_42', 'lean_20261019']:
                    with np.load(d/(name+'.npz')) as z:
                        np.testing.assert_array_equal(z['index'], b)
                        p = z['prediction'].astype('float64')
                    assert len(p) == len(b) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                    branches[name] = p
                    detail = read(d/(name+'.json'))
                    assert detail['train_rows'] == len(a) and detail['query_rows'] == len(b)
                    if phase == 'refit' and k == config['inner_folds']:
                        rebuilt = restored_predict(d/(name+'.joblib'), x.iloc[e], prior[e], aux[e])
                        branch_diffs[name] = float(np.max(abs(rebuilt-p)))
                        np.testing.assert_allclose(rebuilt, p, atol=1e-6, rtol=1e-6)
                    cache_checks += 1
                v4 = sum(w*branches[name] for name, w in config['weights'].items())
                nn = (branches['lean_42']+branches['lean_20261019'])/2
                p = expit(.65*logit(np.clip(v4, 1e-6, 1-1e-6)) +
                          .35*logit(np.clip(nn, 1e-6, 1-1e-6)))
                with np.load(d/'combined.npz') as z:
                    np.testing.assert_array_equal(z['index'], b)
                    np.testing.assert_allclose(z['prediction'], p, atol=1e-14, rtol=0)
                if k < config['inner_folds']:
                    oof[b] = p
            assert np.sum(seen) == len(allowed) and np.all(seen[allowed] == 1)
            with np.load(f/phase/'oof.npz') as z:
                np.testing.assert_array_equal(z['index'], allowed)
                np.testing.assert_allclose(z['prediction'], oof[allowed], atol=1e-14, rtol=0)
        with np.load(f/'refit/full_fit/combined.npz') as z:
            base = z['prediction'].copy()
        frozen = read(f/'frozen_selection.json')
        correction_diffs = {}
        for mode in config['modes']:
            detail = read(f/(mode+'_selection.json'))
            assert frozen[mode] == detail
            curve = detail['curves']['validation']['AUC']
            best_rounds = int(np.argmax(curve))+1
            # CatBoost can pick the last occurrence of a numerical tie.
            assert abs(curve[detail['best_model_rounds']-1] - max(curve)) < 1e-7
            selected = detail['best_model_rounds'] if detail['candidate_stopping_auc'] > detail['base_stopping_auc'] else 0
            assert detail['selected_rounds'] == selected
            tf = joblib.load(f/(mode+'_final_features.joblib'))
            a = tf.transform(x.iloc[e], logit(np.clip(base, 1e-6, 1-1e-6)))
            if selected == 0:
                p = base
            else:
                model = CatBoostClassifier()
                model.load_model(str(f/(mode+'_final.cbm')))
                assert model.tree_count_ == selected
                raw = model.predict(a, prediction_type='RawFormulaVal')
                p = expit(raw + logit(np.clip(base, 1e-6, 1-1e-6))) if mode == 'offset' else expit(raw)
                del model
            with np.load(f/(mode+'_outer.npz')) as z:
                np.testing.assert_array_equal(z['index'], e)
                correction_diffs[mode] = float(np.max(abs(z['prediction']-p)))
                np.testing.assert_allclose(z['prediction'], p, atol=1e-12, rtol=1e-12)
            joined[mode].append(p)
            r0 = result['folds'][config['outer_folds'].index(outer)]['modes'][mode]
            np.testing.assert_allclose(r0['auc'], roc_auc_score(y[e], p), atol=1e-12)
            np.testing.assert_allclose(r0['vs_base']['delta'], roc_auc_score(y[e], p)-roc_auc_score(y[e], base), atol=1e-12)
        joined['index'].append(e); joined['base'].append(base)
        output[str(outer)] = dict(base_cache_branches_checked=cache_checks,
            base_checkpoint_max_differences=branch_diffs, correction_checkpoint_max_differences=correction_diffs)
    joined = {k: np.concatenate(v) for k, v in joined.items()}
    idx = joined['index']
    with np.load(directory/'outer_predictions.npz') as z:
        for k, p in joined.items():
            np.testing.assert_allclose(z[k], p, atol=1e-12)
    for mode in config['modes']:
        pair = paired(y[idx], joined[mode], joined['base'])
        for k in ['delta', 'approximate_se', 'interval95']:
            np.testing.assert_allclose(pair[k], result['modes'][mode]['vs_base'][k], atol=1e-12)
        gate = (pair['delta'] >= .0003 and pair['interval95'][0] > 0 and
                all(f['modes'][mode]['vs_base']['delta'] > 0 for f in result['folds']))
        assert gate == result['modes'][mode]['passed']
    preserve_check()
    write_json(directory/'verification.json', dict(verified=True, folds=output,
        protected_root_files_unchanged=True, input_and_source_fingerprints_match=True,
        all_oof_training_scopes_rebuilt=True, gate_reconstructed=True,
        auxiliary_cache_matches_current_features_and_source=True))
    print('VERIFIED', output, flush=True)


if __name__ == '__main__':
    main()
