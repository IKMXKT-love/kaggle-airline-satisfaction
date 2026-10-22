"""Bounded neural representation screening; never overwrites existing submissions."""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from v2_training import ROOT, load_data, write_json
from v3_features import auxiliary, frames, CONTEXT

OUT = ROOT / '.artifact_runtime/neural_representation'
SPECS = [dict(name='control', category=False, sigma=.1),
         dict(name='distance_category', category=True, sigma=.1),
         dict(name='numeric_sigma32', category=False, sigma=32.)]
PROTOCOL = dict(specs=SPECS, epochs=64, seed=20261018, internal_validation_fraction=.08,
                weights=[.14, .30, .50], gate_blend_gain=.0002,
                screen_only=True, outer_selection_used_for_nn_stopping=False,
                existing_tree_predictions_have_selection_exposure=True,
                historical_selection_and_audit_already_seen=True,
                full_cv_requires_confirmation=True,
                note='64 epoch feasibility screen, not a claim about 128 epoch performance; sigma affects all numeric features.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(a, b, c, category):
    """Fit imputation and any distance vocabulary on training rows only."""
    result = [f.copy() for f in [a, b, c]]
    assert all('satisfaction' not in f and 'id' not in f for f in result)
    numeric = a.select_dtypes(include=['number']).columns
    medians = a[numeric].median().fillna(0)
    cats = list(CONTEXT)
    vocabulary = set(a['Flight Distance'].dropna().astype('float32').astype(str))
    for f in result:
        if category:
            values = f['Flight Distance'].astype('float32').astype(str)
            f['distance_category'] = values.where(values.isin(vocabulary), '__UNKNOWN__')
        f[numeric] = f[numeric].fillna(medians)
        for col in CONTEXT:
            f[col] = f[col].astype('string').fillna('__MISSING__').astype(str)
    if category:
        cats.append('distance_category')
    return (*result, cats)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = ['train.csv', 'test.csv', 'submission_v4.csv', 'submission_v3.csv',
             'v3_recipe.json', 'v4_recipe.json', 'v3_features.py', 'v3_training.py',
             'optimized_features.py', 'optimized_training.py']
    protected = {p: sha(ROOT / p) for p in files}
    signature = hashlib.sha256((json.dumps(PROTOCOL, sort_keys=True)+json.dumps(protected, sort_keys=True)
                               + Path(__file__).read_text(encoding='utf-8')).encode()).hexdigest()
    directory = OUT / signature[:20]
    directory.mkdir(exist_ok=True)
    write_json(directory/'protocol.json', dict(**PROTOCOL, protected=protected, signature=signature))
    print('OUTPUT', directory, flush=True)
    x, xt, y, ox, oy, prior, pt, stats, meta = load_data()
    aux, auxt = auxiliary(x, xt, ox)
    split = np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz')
    selection = split['selection']
    fit, stopping = train_test_split(split['fit'], test_size=.08, stratify=y[split['fit']], random_state=PROTOCOL['seed'])
    assert not set(fit) & set(stopping)
    assert not set(selection) & (set(fit) | set(stopping))
    assert not set(split['audit']) & (set(fit) | set(stopping) | set(selection))
    np.savez_compressed(directory/'indices.npz', fit=fit, stopping=stopping, selection=selection)
    both = np.concatenate([stopping, selection])
    a, held = frames(x, None, y, fit, both, prior, pt, aux, auxt)
    b, c = held.iloc[:len(stopping)], held.iloc[len(stopping):]
    assert np.array_equal(a.index, fit) and np.array_equal(c.index, selection)
    del xt, ox, oy, stats, held
    gc.collect()
    from pytabkit import RealMLP_TD_Classifier
    import torch
    results = []
    predictions = {}
    for spec in SPECS:
        path = directory/(spec['name']+'.npz')
        js = path.with_suffix('.json')
        if not (path.exists() and js.exists()):
            aa, bb, cc, cats = prepare(a, b, c, spec['category'])
            model = RealMLP_TD_Classifier(device='cuda', random_state=PROTOCOL['seed'], n_cv=1,
                n_refit=0, n_epochs=PROTOCOL['epochs'], batch_size=1024, n_threads=4, verbosity=0,
                use_ls=False, val_metric_name='1-auc_ovr', plr_sigma=spec['sigma'])
            print('START', spec['name'], len(aa), len(bb), len(cc), flush=True)
            start = time.time()
            model.fit(aa, y[fit], cat_col_names=cats, X_val=bb, y_val=y[stopping])
            pred = model.predict_proba(cc)[:, 1]
            assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
            np.savez_compressed(path, index=selection, prediction=pred)
            detail = dict(spec=spec, auc=float(roc_auc_score(y[selection], pred)),
                          seconds=time.time()-start, features=list(aa.columns), cats=cats,
                          fit_params=str(getattr(model, 'fit_params_', None)))
            write_json(js, detail)
            print('RESULT', spec['name'], detail['auc'], detail['seconds'], flush=True)
            del model, aa, bb, cc
            gc.collect()
            torch.cuda.empty_cache()
        with np.load(path) as z:
            np.testing.assert_array_equal(z['index'], selection)
            predictions[spec['name']] = z['prediction'].copy()
        results.append(json.loads(js.read_text(encoding='utf-8')))
    old = ROOT/'.artifact_runtime/writeup_search'
    v3 = ROOT/'.artifact_runtime/v3'
    frozen = json.loads((ROOT/'v4_recipe.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'.artifact_runtime/v4'/frozen['signature']/'lgb_63_selection.npz') as z:
        np.testing.assert_array_equal(z['valid_index'], selection)
        lgb = z['prediction'].copy()
    old_nn = np.load(v3/'realmlp_selection.npy')
    base = .7*(.4*(.25*np.load(old/'te_original_d7_selection.npy')+
                    .75*np.load(old/'catboost_allcat_original_d8_selection.npy'))+
               .4*np.load(v3/'xgb_selection.npy')+.2*old_nn)+.3*lgb
    ys = y[selection]
    auc = lambda p: float(roc_auc_score(ys, p))
    control = predictions['control']
    matched = base+.14*(control-old_nn)
    # Shared cached tree terms make this a screening diagnostic, not independent confirmation.
    trials = []
    for spec in SPECS[1:]:
        p = predictions[spec['name']]
        for w in PROTOCOL['weights']:
            candidate = (1-w)*(base-.14*old_nn)/.86+w*p
            comparison = (1-w)*(base-.14*old_nn)/.86+w*control
            trials.append(dict(name=spec['name'], weight=w, auc=auc(candidate),
                               gain_vs_v4=auc(candidate)-auc(base),
                               gain_vs_matched_control=auc(candidate)-auc(comparison)))
    best = max(trials, key=lambda d: d['auc'])
    gate = best['gain_vs_v4'] >= .0002 and best['gain_vs_matched_control'] >= .0002
    for p, h in protected.items():
        assert sha(ROOT/p) == h, p
    summary = dict(protocol=PROTOCOL, directory=str(directory), models=results,
                   v4_selection_auc=auc(base), matched_control_blend_auc=auc(matched),
                   trials=trials, best=best, warrants_confirmation=gate, protected_files_unchanged=True,
                   train_rows=len(fit), stopping_rows=len(stopping), selection_rows=len(selection))
    write_json(directory/'summary.json', summary)
    write_json(OUT/'latest.json', dict(directory=str(directory), signature=signature))
    print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
