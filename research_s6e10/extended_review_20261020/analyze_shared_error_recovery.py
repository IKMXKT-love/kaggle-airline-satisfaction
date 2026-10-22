"""Describe ranking errors using existing arrays only; never fit a model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / '.artifact_runtime/strong_correction_20261020/b1385337dbf19f77097b'
NAMES = ['old_xgb', 'old_cat', 'control', 'realmlp', 'lgb', 'lean_42', 'lean_20261019']


def main():
    target = OUT / 'shared_error_recovery.json'
    protocol_path = OUT / 'shared_error_recovery_protocol.json'
    if target.exists() or protocol_path.exists():
        raise RuntimeError('Preserve previous analyses')
    protocol = {
        'training': False, 'model_inference': False, 'own_submission_generated': False,
        'seed': 20261021, 'pairs_per_fold': 2000000, 'chunk': 100000,
        'folds': [0, 1],
        'purpose': 'Describe whether the frozen public reference changes errors unanimously shared by our seven branches; no fitted fusion or selection.',
        'caveat': 'Exploratory only. Our baseline uses strict cached outer predictions, but public OOF partitions and historical selection differ. This is not strict validation of a new ensemble. Monte Carlo uncertainty is conditional on fixed predictions.',
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    y_all = pd.read_csv(ROOT / 'train.csv', usecols=['satisfaction']).satisfaction.to_numpy(dtype='int64')
    ext = np.load(OUT / 'reference_predictions/stack247_frozen_v1/stack_oof.npy', allow_pickle=False)
    rows = []
    for fold in protocol['folds']:
        folder = CACHE / f'outer_{fold}/refit/full_fit'
        with np.load(folder / 'combined.npz', allow_pickle=False) as z:
            idx, base = z['index'].copy(), z['prediction'].copy()
        branches = []
        for name in NAMES:
            with np.load(folder / (name + '.npz'), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['index'], idx)
                branches.append(z['prediction'].copy())
        a, y, public = np.column_stack(branches), y_all[idx], ext[idx]
        positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        rng = np.random.default_rng(protocol['seed'] + fold)
        counts = dict(base_wrong=0, all_seven_wrong=0, public_corrects_unanimous=0,
                      public_corrects_base_wrong=0, public_breaks_base_correct=0,
                      public_ties=0, base_ties=0)
        for begin in range(0, protocol['pairs_per_fold'], protocol['chunk']):
            n = min(protocol['chunk'], protocol['pairs_per_fold'] - begin)
            pi, ni = rng.choice(positives, n), rng.choice(negatives, n)
            db, de = base[pi] - base[ni], public[pi] - public[ni]
            unanimous = (a[pi] - a[ni] < 0).all(axis=1)
            counts['base_wrong'] += int((db < 0).sum())
            counts['all_seven_wrong'] += int(unanimous.sum())
            counts['public_corrects_unanimous'] += int((unanimous & (de > 0)).sum())
            counts['public_corrects_base_wrong'] += int(((db < 0) & (de > 0)).sum())
            counts['public_breaks_base_correct'] += int(((db > 0) & (de < 0)).sum())
            counts['public_ties'] += int((de == 0).sum())
            counts['base_ties'] += int((db == 0).sum())
        rows.append({
            'fold': fold, 'rows': len(idx), 'sampled_pairs': protocol['pairs_per_fold'],
            'counts': counts,
            'public_recovers_fraction_of_all_seven_wrong': counts['public_corrects_unanimous'] / counts['all_seven_wrong'],
            'sampled_net_pair_gain_ignoring_ties': (counts['public_corrects_base_wrong'] - counts['public_breaks_base_correct']) / protocol['pairs_per_fold'],
            'baseline_auc': float(roc_auc_score(y, base)),
            'public_reference_auc_on_same_rows': float(roc_auc_score(y, public)),
            'exact_auc_delta_on_these_rows': float(roc_auc_score(y, public) - roc_auc_score(y, base)),
        })
    result = {'protocol': protocol, 'folds': rows}
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
