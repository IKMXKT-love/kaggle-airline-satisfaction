"""Descriptive review of frozen predictions; no model/weight selection or training."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def pair_matrix(y, p, groups):
    names = sorted(set(groups))
    denom = float((y == 1).sum()) * (y == 0).sum()
    positives = {name: p[(y == 1) & (groups == name)] for name in names}
    negatives = {name: np.sort(p[(y == 0) & (groups == name)]) for name in names}
    rows = []
    for a in names:
        pos = positives[a]
        for b in names:
            neg = negatives[b]
            if not len(pos) or not len(neg):
                continue
            left = np.searchsorted(neg, pos, side='left')
            right = np.searchsorted(neg, pos, side='right')
            errors = float(np.sum(len(neg) - right + .5 * (right - left)))
            weight = len(pos) * len(neg) / denom
            rows.append(dict(positive_group=a, negative_group=b, pair_weight=weight,
                             error_contribution=errors / denom,
                             pair_auc=1 - errors / (len(pos) * len(neg))))
    return rows


def main():
    train = pd.read_csv(ROOT / 'train.csv')
    base = pd.read_csv(ROOT / 'submission_v4_oof.csv')
    path = ROOT / '.artifact_runtime/lean_cv_followup/8f29c4956de9c49f6f97/predictions.npz'
    with np.load(path) as z:
        np.testing.assert_array_equal(z['id'], train.id)
        p = z['blended_oof'].copy()
        nn = z['oof'].copy()
    np.testing.assert_array_equal(base.id, train.id)
    np.testing.assert_array_equal(base.target, train.satisfaction)
    y = train.satisfaction.to_numpy()
    total_error = 1 - roc_auc_score(y, p)
    out = dict(rows=len(y), frozen_auc=float(1-total_error),
               note='Descriptive only; legacy OOF outer stopping and historical selection caveats apply.',
               decompositions={})
    for cols in [['Class', 'Type of Travel'], ['Customer Type', 'Type of Travel'],
                 ['Class', 'Customer Type', 'Type of Travel']]:
        groups = train[cols].astype(str).agg(' | '.join, axis=1).to_numpy()
        matrix = pair_matrix(y, p, groups)
        assert abs(sum(r['error_contribution'] for r in matrix) - total_error) < 1e-12
        within = sum(r['error_contribution'] for r in matrix if r['positive_group'] == r['negative_group'])
        bygroup = []
        for name in sorted(set(groups)):
            mask = groups == name
            if len(np.unique(y[mask])) < 2:
                continue
            error = sum(.5*r['error_contribution'] *
                        ((r['positive_group'] == name) + (r['negative_group'] == name)) for r in matrix)
            halves = []
            for parity in [0, 1]:
                m = mask & (base.fold.to_numpy() % 2 == parity)
                if len(np.unique(y[m])) < 2:
                    halves.append(dict(rows=int(m.sum()), auc=None, new_minus_base=None))
                    continue
                halves.append(dict(rows=int(m.sum()), auc=float(roc_auc_score(y[m], p[m])),
                                   new_minus_base=float(roc_auc_score(y[m], p[m]) -
                                                        roc_auc_score(y[m], base.prediction.to_numpy()[m]))))
            bygroup.append(dict(group=name, rows=int(mask.sum()), row_share=float(mask.mean()),
                                global_error_endpoint_share=error/total_error,
                                touched_error_ceiling=sum(r['error_contribution'] for r in matrix
                                    if r['positive_group'] == name or r['negative_group'] == name),
                                auc=float(roc_auc_score(y[mask], p[mask])),
                                nn_auc=float(roc_auc_score(y[mask], nn[mask])),
                                base_auc=float(roc_auc_score(y[mask], base.prediction.to_numpy()[mask])),
                                halves=halves))
        out['decompositions'][' x '.join(cols)] = dict(
            within_group_error_share=within/total_error,
            across_group_error_share=1-within/total_error,
            groups=bygroup,
            largest_error_pairs=sorted(matrix, key=lambda r: r['error_contribution'], reverse=True)[:12])
    target = ROOT / 'research_s6e10/residual_review_20261019.json'
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved:', target)
    for name, data in out['decompositions'].items():
        print(name, 'across_group_error_share=', data['across_group_error_share'])
        for g in data['groups']:
            print(g['group'], g['rows'], 'auc=', round(g['auc'], 6),
                  'touched_error_ceiling=', round(g['touched_error_ceiling'], 7))


if __name__ == '__main__':
    main()
