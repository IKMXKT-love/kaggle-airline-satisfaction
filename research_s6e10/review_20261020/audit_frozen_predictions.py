"""Descriptive statistics on saved predictions only. No model fitting or inference."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / '.artifact_runtime/strong_correction_20261020/b1385337dbf19f77097b'
CONTEXT = ['Class', 'Type of Travel', 'Customer Type']
RATINGS = ['Inflight wifi service', 'Departure/Arrival time convenient',
           'Ease of Online booking', 'Gate location', 'Food and drink',
           'Online boarding', 'Seat comfort', 'Inflight entertainment',
           'On-board service', 'Leg room service', 'Baggage handling',
           'Checkin service', 'Cleanliness']
GROUPS = {
    'context': CONTEXT,
    'flight_distance': ['Flight Distance'],
    'distance_context': ['Flight Distance'] + CONTEXT,
    'age_context': ['Age'] + CONTEXT,
    'wifi_context': ['Inflight wifi service'] + CONTEXT,
    'boarding_context': ['Online boarding'] + CONTEXT,
    'digital_context': ['Inflight wifi service', 'Ease of Online booking', 'Online boarding'] + CONTEXT,
    'comfort_context': ['Seat comfort', 'Inflight entertainment', 'Cleanliness'] + CONTEXT,
    'distance_wifi_context': ['Flight Distance', 'Inflight wifi service'] + CONTEXT,
    'full_ratings': RATINGS,
    'full_ratings_context': RATINGS + CONTEXT,
}
EDGES = np.array([0, .001, .01, .05, .1, .2, .35, .5, .65, .8, .9, .95, .99, .999, 1.])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_error(y, p):
    """Split each wrong positive-negative pair equally between its endpoints."""
    a, b = np.sort(p[y == 1]), np.sort(p[y == 0])
    result = np.zeros(len(y))
    m = y == 1
    left, right = np.searchsorted(b, p[m], 'left'), np.searchsorted(b, p[m], 'right')
    result[m] = (len(b) - right + .5 * (right - left)) / (2 * float(len(a)) * len(b))
    left, right = np.searchsorted(a, p[~m], 'left'), np.searchsorted(a, p[~m], 'right')
    result[~m] = (left + .5 * (right - left)) / (2 * float(len(a)) * len(b))
    np.testing.assert_allclose(result.sum(), 1 - roc_auc_score(y, p), atol=1e-13)
    return result


def aggregate(x, y, p, columns):
    frame = x[columns].copy()
    frame['residual'] = y - p
    frame['residual_sq'] = (y - p) ** 2
    return frame.groupby(columns, dropna=False, observed=True).agg(
        n=('residual', 'size'), mean=('residual', 'mean'),
        square=('residual_sq', 'mean'))


def moments(values, weights):
    mean = np.average(values, weights=weights)
    return mean, np.average((values - mean) ** 2, weights=weights)


def main():
    started = time.monotonic()
    assert len(RATINGS) == 13
    protocol = {
        'date': '2026-10-20', 'training': False, 'prediction_model_loaded': False,
        'groups': GROUPS, 'common_cell_min_support_per_fold': [5, 20, 50],
        'score_bin_edges': EDGES.tolist(),
        'interpretation': 'Exploratory fixed-prediction diagnostics. No candidate or blend selected. Previously viewed folds; shared training history; residual means concern calibration, not all ranking errors.',
        'input_hashes': {'train.csv': sha(ROOT / 'train.csv'),
                         'outer_predictions.npz': sha(BASE / 'outer_predictions.npz'),
                         'audit_frozen_predictions.py': sha(Path(__file__)),
                         'submission_lean_realmlp_10fold.csv': sha(ROOT / 'submission_lean_realmlp_10fold.csv')},
    }
    protocol_file = OUT / 'diagnostic_protocol.json'
    if protocol_file.exists():
        raise RuntimeError('Refusing to overwrite an existing diagnostic run')
    protocol_file.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    train = pd.read_csv(ROOT / 'train.csv')
    labels = train.satisfaction.to_numpy(dtype='int64')
    indices, probs, samples, folds = [], [], [], []
    component_rows = []
    for fold in [0, 1]:
        full = BASE / f'outer_{fold}/refit/full_fit'
        with np.load(full / 'combined.npz') as z:
            idx, p = z['index'].copy(), z['prediction'].copy()
        with np.load(full / 'indices.npz') as z:
            np.testing.assert_array_equal(z['query'], idx)
            assert not np.intersect1d(idx, z['fit']).size
        components = {}
        for name in ['old_xgb', 'old_cat', 'control', 'realmlp', 'lgb', 'lean_42', 'lean_20261019']:
            path = full / (name + '.npz')
            protocol['input_hashes'][str(path.relative_to(ROOT))] = sha(path)
            with np.load(path) as z:
                np.testing.assert_array_equal(z['index'], idx)
                components[name] = z['prediction'].copy()
        v4 = sum(weight * components[name] for weight, name in [(.07, 'old_xgb'), (.21, 'old_cat'), (.28, 'control'), (.14, 'realmlp'), (.30, 'lgb')])
        lean = (components['lean_42'] + components['lean_20261019']) / 2
        reconstructed = expit(.65 * logit(np.clip(v4, 1e-6, 1 - 1e-6)) + .35 * logit(np.clip(lean, 1e-6, 1 - 1e-6)))
        np.testing.assert_allclose(p, reconstructed, atol=1e-13, rtol=0)
        components.update(v4=v4, lean_mean=lean)
        component_rows.append({'fold': fold, 'baseline_auc': float(roc_auc_score(labels[idx], p)),
                               'component_auc': {k: float(roc_auc_score(labels[idx], a)) for k, a in components.items()}})
        indices.append(idx); probs.append(p); samples.append(train.iloc[idx].reset_index(drop=True)); folds.append(np.full(len(idx), fold))
    idx, p, fold_id = np.concatenate(indices), np.concatenate(probs), np.concatenate(folds)
    assert len(idx) == 139928 and len(np.unique(idx)) == len(idx)
    y = labels[idx]
    with np.load(BASE / 'outer_predictions.npz') as z:
        order = np.argsort(z['index'])
        np.testing.assert_array_equal(z['index'][order], np.sort(idx))
        np.testing.assert_allclose(z['base'][order], p[np.argsort(idx)], atol=0, rtol=0)
    x = pd.concat(samples, ignore_index=True)
    error = endpoint_error(y, p)
    score_bins = np.minimum(np.searchsorted(EDGES, p, side='right') - 1, len(EDGES) - 2)
    bins = []
    for i in range(len(EDGES) - 1):
        m = score_bins == i
        bins.append({'low': float(EDGES[i]), 'high': float(EDGES[i + 1]), 'rows': int(m.sum()),
                     'mean_prediction': float(p[m].mean()) if m.any() else None,
                     'positive_rate': float(y[m].mean()) if m.any() else None,
                     'ranking_error_endpoint_share': float(error[m].sum() / error.sum())})
    residuals = {}
    for name, columns in GROUPS.items():
        a = aggregate(samples[0], labels[indices[0]], probs[0], columns)
        b = aggregate(samples[1], labels[indices[1]], probs[1], columns)
        joined = a.join(b, how='inner', lsuffix='_a', rsuffix='_b')
        rows = []
        for minimum in [5, 20, 50]:
            common = joined[(joined.n_a >= minimum) & (joined.n_b >= minimum)]
            item = {'min_rows_each': minimum, 'common_cells': len(common),
                    'eval_rows_covered': int(common.n_a.sum() + common.n_b.sum())}
            if len(common) > 2:
                va, vb = common['mean_a'].to_numpy(), common['mean_b'].to_numpy()
                weights = (2 * common.n_a * common.n_b / (common.n_a + common.n_b)).to_numpy()
                ma, var_a = moments(va, weights); mb, var_b = moments(vb, weights)
                cov = np.average((va - ma) * (vb - mb), weights=weights)
                noise_a = np.maximum(common.square_a.to_numpy() - va ** 2, 0) / (common.n_a.to_numpy() - 1)
                noise_b = np.maximum(common.square_b.to_numpy() - vb ** 2, 0) / (common.n_b.to_numpy() - 1)
                item.update(weighted_residual_correlation=float(cov / np.sqrt(var_a * var_b)) if var_a * var_b > 0 else None,
                            noise_to_observed_variance_a=float(np.average(noise_a, weights=weights) / var_a) if var_a > 0 else None,
                            noise_to_observed_variance_b=float(np.average(noise_b, weights=weights) / var_b) if var_b > 0 else None)
            rows.append(item)
        residuals[name] = rows
    confident_wrong = ((y == 0) & (p >= .9)) | ((y == 1) & (p <= .1))
    result = {'rows': len(y), 'baseline_auc': float(roc_auc_score(y, p)), 'components': component_rows,
              'score_bins': bins, 'residual_stability': residuals,
              'confident_wrong': {'rows': int(confident_wrong.sum()), 'row_share': float(confident_wrong.mean()),
                                  'ranking_error_endpoint_share': float(error[confident_wrong].sum() / error.sum()),
                                  'note': 'Defined by observed labels; not evidence of label corruption or a deployable sample filter.'},
              'elapsed_seconds': time.monotonic() - started, 'protocol': protocol}
    (OUT / 'frozen_prediction_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Rows', result['rows'], 'AUC', result['baseline_auc'], 'seconds', result['elapsed_seconds'])
    print('Confident wrong', result['confident_wrong'])
    for name, rows in residuals.items():
        print(name, rows[1])


if __name__ == '__main__':
    main()
