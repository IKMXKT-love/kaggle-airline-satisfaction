"""Fixed-prediction research contrasts only. No model fitting or inference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main():
    target = OUT/'neighborhood_contrasts.json'
    if target.exists(): raise FileExistsError('Preserve previous research contrasts')
    protocol = json.loads((OUT/'contrast_protocol.json').read_text(encoding='utf-8'))
    assert protocol['fixed_rule'].startswith('0.8 *')
    tr = pd.read_csv(ROOT/'train.csv', usecols=['id', 'satisfaction'])
    te = pd.read_csv(ROOT/'test.csv', usecols=['id'])
    ids, y = tr.id.to_numpy(), tr.satisfaction.to_numpy(dtype='int64')
    lookup = pd.Series(np.arange(len(tr)), index=ids)
    current_file = ROOT/'.artifact_runtime/public_pool_trial_20261020/selected_predictions.npz'
    with np.load(current_file, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'], ids)
        current = z['oof_logit'].copy()
    public_fold = np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy', allow_pickle=False)
    candidates = {k: np.full(len(y), np.nan) for k in ['combined', 'head', 'neighbor']}
    source_fold = np.full(len(y), -1)
    coverage = np.zeros(len(y), dtype='int8')
    testing = np.zeros(len(te))
    for f in range(5):
        with np.load(OUT/'predictions'/f'den_fold{f}.npz', allow_pickle=False) as z:
            assert len(np.unique(z['valid_ids'])) == len(z['valid_ids'])
            idx = lookup.reindex(z['valid_ids']).to_numpy()
            assert np.isfinite(idx).all()
            idx = idx.astype(int)
            np.testing.assert_array_equal(z['test_ids'], te.id)
            for name, key in [('combined', 'valid'), ('head', 'head_valid'), ('neighbor', 'neighbor_valid')]:
                p = z[key]; assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                candidates[name][idx] = p
            t = z['test']; assert np.isfinite(t).all() and ((t >= 0) & (t <= 1)).all()
            testing += t/5
            coverage[idx] += 1; source_fold[idx] = f
    assert (coverage == 1).all() and (source_fold >= 0).all()
    current_auc = float(roc_auc_score(y, current))
    rows = []
    for name, p in candidates.items():
        assert np.isfinite(p).all()
        score = .8*current+.2*logit(np.clip(p, 1e-6, 1-1e-6))
        row = {'candidate': name, 'standalone_auc': float(roc_auc_score(y, p)),
               'fixed20_auc': float(roc_auc_score(y, score)),
               'fixed20_delta': float(roc_auc_score(y, score)-current_auc)}
        for label, fold in [('current_native_folds', public_fold), ('candidate_native_folds', source_fold)]:
            row[label+'_deltas'] = [float(roc_auc_score(y[fold == f], score[fold == f])-roc_auc_score(y[fold == f], current[fold == f])) for f in range(5)]
        rows.append(row)
    author = json.loads((OUT/'sources/den_exp042__metrics.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(rows[0]['standalone_auc'], author['new_base']['overall_oof_auc'], atol=1e-12)
    result = {'training': False, 'model_inference': False, 'submission_generated': False,
              'protocol': protocol, 'current_oof_auc': current_auc,
              'source_oof_ids_full_unique_coverage': True, 'test_ids_verified': True,
              'author_single_model_auc_reproduced': True,
              'source': 'denpugovkin/s6e10-passenger-context-foundation-ensemble version45',
              'rows': rows, 'candidate_test_probability_mean': float(testing.mean()),
              'scope': 'A fixed20 negative result rejects this fixed use, not every possible weight, sign, conditional use or the whole model family.'}
    target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
