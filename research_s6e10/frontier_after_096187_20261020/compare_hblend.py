"""One prespecified row-order fusion rule on fixed parent OOFs. No fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def pct(v):
    return (rankdata(v)-.5)/len(v)


def hblend(matrix, weights, correction, sides):
    n = len(matrix); outputs = []
    for reverse in [False, True]:
        order = np.argsort(-matrix if reverse else matrix, axis=1, kind='stable')
        score = np.zeros(n)
        for j in range(4):
            k = order[:, j]
            score += matrix[np.arange(n), k]*(weights[k]+correction[j])
        outputs.append(score)
    return sides[0]*outputs[0]+sides[1]*outputs[1]


def main():
    target = OUT/'hblend_contrasts.json'
    if target.exists(): raise FileExistsError('Preserve prior contrasts')
    protocol = json.loads((OUT/'hblend_protocol.json').read_text(encoding='utf-8'))
    tr = pd.read_csv(ROOT/'train.csv', usecols=['id', 'satisfaction']); y = tr.satisfaction.to_numpy(dtype='int64')
    test_ids = pd.read_csv(ROOT/'test.csv', usecols=['id']).id.to_numpy()
    with np.load(ROOT/'.artifact_runtime/public_pool_trial_20261020/selected_predictions.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'], tr.id)
        np.testing.assert_array_equal(z['test_id'], test_ids)
        current, current_test = z['oof_logit'].copy(), z['test_logit'].copy()
    parents, testing, standalone = [], [], []
    for name in ['atlas181', 'cdeotte135', 'gp_stack245']:
        o = np.load(OUT/'predictions'/(name+'_oof.npy'), allow_pickle=False)
        t = np.load(OUT/'predictions'/(name+'_test.npy'), allow_pickle=False)
        assert o.shape == y.shape and t.shape == test_ids.shape
        assert np.isfinite(o).all() and np.isfinite(t).all()
        assert ((o >= 0) & (o <= 1)).all() and ((t >= 0) & (t <= 1)).all()
        parents.append(pct(o)); testing.append(pct(t))
        standalone.append({'name': name, 'auc': float(roc_auc_score(y, o)), 'alignment': 'author-declared file order; no independent ID sidecar'})
    parents.append(pct(current)); testing.append(pct(current_test))
    matrix, xt = np.column_stack(parents), np.column_stack(testing)
    w, correction, sides = (np.array(protocol[k]) for k in ['main_weights', 'rank_position_corrections', 'side_mix'])
    arms = {'fixed_rank_mean': (matrix@w, xt@w),
            'fixed_row_hblend': (hblend(matrix, w, correction, sides), hblend(xt, w, correction, sides))}
    baseline_auc = float(roc_auc_score(y, current))
    folds = np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy', allow_pickle=False)
    rows = []
    for name, (o, t) in arms.items():
        rows.append({'arm': name, 'auc': float(roc_auc_score(y, o)), 'delta': float(roc_auc_score(y, o)-baseline_auc),
                     'native_fold_deltas': [float(roc_auc_score(y[folds == f], o[folds == f])-roc_auc_score(y[folds == f], current[folds == f])) for f in range(5)],
                     'test_rank_correlation_with_current': float(spearmanr(t, current_test).statistic)})
    result = {'protocol': protocol, 'training': False, 'model_inference': False, 'submission_generated': False,
              'current_auc': baseline_auc, 'parent_scores': standalone, 'arms': rows,
              'nonlinearity_delta_vs_fixed_rank_mean': rows[1]['auc']-rows[0]['auc'],
              'caveat': 'No new target-fitted stack layer, but parent early stopping, indirect OOF dependencies and source selection persist. A fixed recipe negative result does not reject all nonlinear blending.'}
    target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
