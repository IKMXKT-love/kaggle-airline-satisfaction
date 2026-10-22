"""Second-round experiments. Previous pipelines and submissions remain unchanged."""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from optimized_features import CompetitionFeaturePipeline, OriginalStatistics
from optimized_training import load_original_bundle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '.artifact_runtime' / 'v2'
OUT.mkdir(parents=True, exist_ok=True)
CAT_CONFIGS = [
    dict(name='cat_control', depth=8, l2_leaf_reg=5, learning_rate=.04),
    dict(name='cat_d6_l20', depth=6, l2_leaf_reg=20, learning_rate=.04),
    dict(name='cat_d7_l10', depth=7, l2_leaf_reg=10, learning_rate=.04),
    dict(name='cat_d8_l20_slow', depth=8, l2_leaf_reg=20, learning_rate=.025),
    dict(name='cat_d8_l50', depth=8, l2_leaf_reg=50, learning_rate=.04),
    dict(name='cat_d9_l10', depth=9, l2_leaf_reg=10, learning_rate=.04),
    dict(name='cat_source_prior', depth=8, l2_leaf_reg=5, learning_rate=.04, source_prior=True),
]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def load_data():
    train, test = pd.read_csv(ROOT/'train.csv'), pd.read_csv(ROOT/'test.csv')
    bundle = load_original_bundle(train, test, 'satisfaction', 'id', ROOT/'external_data', 'cuda', 8)
    x, xt, ox, oy, prior, pt, meta = bundle
    stats = OriginalStatistics().fit(ox, oy)
    y = train.satisfaction.to_numpy(dtype=int)
    return x, xt, y, ox, oy, prior[:, 0], pt[:, 0], stats, meta


def cat_frame(frame):
    result = frame.copy()
    for col in result.select_dtypes(include=['category']).columns:
        result[col] = result[col].astype('string').fillna('__MISSING__').astype(str)
    return result


def make_cat_frames(x, xt, y, fit, valid, stats, prior, pt, source=None):
    p = CompetitionFeaturePipeline(original_statistics=stats, numeric_categories=True)
    a = p.fit_transform(x.iloc[fit], y[fit], prior[fit])
    b = p.transform(x.iloc[valid], prior[valid])
    c = p.transform(xt, pt) if xt is not None else None
    cats = a.select_dtypes(include=['category']).columns.tolist()
    a, b = cat_frame(a), cat_frame(b)
    c = cat_frame(c) if c is not None else None
    if source is not None:
        a['source_cat_probability'] = source[0][fit]
        b['source_cat_probability'] = source[0][valid]
        if c is not None:
            c['source_cat_probability'] = source[1]
    return a, b, c, cats


def fit_cat(a, ya, b, yb, c, cats, config, seed=42, rounds=None):
    begin = time.time()
    params = {k: v for k, v in config.items() if k not in ['name', 'source_prior']}
    model = CatBoostClassifier(**params, iterations=rounds or 4000,
        loss_function='Logloss', eval_metric='AUC', task_type='GPU', devices='0',
        cat_features=cats, random_seed=seed, thread_count=8, max_ctr_complexity=2,
        allow_writing_files=False, metric_period=25, verbose=False)
    kwargs = dict(eval_set=(b, yb), early_stopping_rounds=150) if rounds is None else {}
    model.fit(a, ya, **kwargs)
    pred = model.predict_proba(b)[:, 1]
    test_pred = model.predict_proba(c)[:, 1] if c is not None else None
    details = dict(config=config, seed=seed, rounds=int(model.tree_count_),
        auc=float(roc_auc_score(yb, pred)), train_auc=float(roc_auc_score(ya, model.predict_proba(a)[:,1])),
        seconds=time.time()-begin, curves=model.get_evals_result(), features=list(a.columns))
    del model
    gc.collect()
    return pred, test_pred, details


def source_prior(x, xt, ox, oy):
    paths = [OUT/'source_cat_train.npy', OUT/'source_cat_test.npy']
    signature = hashlib.sha256(pd.util.hash_pandas_object(ox.astype('string'), index=False).to_numpy().tobytes()+oy.tobytes()).hexdigest()
    if all(p.exists() for p in paths) and (OUT/'source_cat_metadata.json').exists():
        assert json.loads((OUT/'source_cat_metadata.json').read_text())['signature'] == signature
        return tuple(np.load(p) for p in paths)
    from optimized_features import NativeCategoryEncoder
    p = NativeCategoryEncoder().fit(ox)
    a, b, c = p.transform(ox), p.transform(x), p.transform(xt)
    cats = a.select_dtypes(include=['category']).columns.tolist()
    a,b,c = map(cat_frame, [a,b,c])
    model = CatBoostClassifier(iterations=1800, depth=8, learning_rate=.04, l2_leaf_reg=5,
        loss_function='Logloss', task_type='GPU', devices='0', cat_features=cats,
        random_seed=42, thread_count=8, max_ctr_complexity=2, allow_writing_files=False, verbose=False)
    model.fit(a, oy)
    values = (model.predict_proba(b)[:,1], model.predict_proba(c)[:,1])
    for path, arr in zip(paths,values): np.save(path,arr)
    model.save_model(str(OUT/'source_cat.cbm'))
    write_json(OUT/'source_cat_metadata.json',dict(signature=signature, rows=len(ox), source_labels_only=True,
        config=model.get_params()))
    del model,a,b,c
    gc.collect()
    return values


def screen_cats():
    x, xt, y, ox, oy, prior, pt, stats, meta = load_data()
    split = np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz')
    fit, valid = split['fit'], split['selection']
    frames = make_cat_frames(x, None, y, fit, valid, stats, prior, pt)
    results = []
    for config in CAT_CONFIGS:
        path = OUT/(config['name']+'.json')
        if path.exists():
            results.append(json.loads(path.read_text(encoding='utf-8')))
            continue
        print('START',config,flush=True)
        if config.get('source_prior'):
            src = source_prior(x,xt,ox,oy)
            a,b,c,cats = frames
            a = a.copy(); b = b.copy()
            a['source_cat_probability']=src[0][fit]; b['source_cat_probability']=src[0][valid]
            current=(a,b,c,cats)
        else: current=frames
        a,b,c,cats=current
        pred,_,details = fit_cat(a,y[fit],b,y[valid],None,cats,config)
        np.save(OUT/(config['name']+'_selection.npy'), pred)
        write_json(path,details)
        results.append(details)
        write_json(OUT/'cat_screen.json',results)
        print('RESULT',config['name'],{k:details[k] for k in ['auc','train_auc','rounds','seconds']},flush=True)


if __name__ == '__main__':
    screen_cats()
