"""Monte Carlo pair-ranking diagnostics on frozen predictions; no training."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / '.artifact_runtime/strong_correction_20261020/b1385337dbf19f77097b'
NAMES = ['old_xgb', 'old_cat', 'control', 'realmlp', 'lgb', 'lean_42', 'lean_20261019']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    protocol = {'date': '2026-10-20', 'training': False, 'model_inference': False,
                'seed': 20261021, 'pairs_per_fold': 2000000, 'chunk': 100000,
                'folds': [0, 1], 'components': NAMES,
                'definition': 'Uniform positive-negative pairs sampled with replacement within each frozen outer fold. A strictly inverted pair on every component stays inverted under any nonzero global nonnegative weighted sum of strictly increasing component transforms.',
                'uncertainty': 'Monte Carlo error only, conditional on the fixed observed rows and predictions; not generalization uncertainty.'}
    protocol_file = OUT / 'pair_agreement_protocol.json'
    if protocol_file.exists():
        raise RuntimeError('Refusing to overwrite a previous diagnostic')
    protocol_file.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    y_all = pd.read_csv(ROOT / 'train.csv', usecols=['satisfaction']).satisfaction.to_numpy(dtype='int64')
    rows = []
    hashes = {}
    for fold in protocol['folds']:
        folder = BASE / f'outer_{fold}/refit/full_fit'
        with np.load(folder / 'combined.npz') as z:
            idx, p = z['index'].copy(), z['prediction'].copy()
        columns = []
        for name in NAMES:
            path = folder / (name + '.npz')
            hashes[str(path.relative_to(ROOT))] = sha(path)
            with np.load(path) as z:
                np.testing.assert_array_equal(z['index'], idx)
                columns.append(z['prediction'].copy())
        a = np.column_stack(columns)
        y = y_all[idx]
        positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        rng = np.random.default_rng(protocol['seed'] + fold)
        totals = dict(base_wrong=0, base_ties=0, all_seven_strictly_wrong=0,
                      base_wrong_with_some_component_correct=0,
                      base_correct_with_some_component_wrong=0)
        for begin in range(0, protocol['pairs_per_fold'], protocol['chunk']):
            n = min(protocol['chunk'], protocol['pairs_per_fold'] - begin)
            pi = rng.choice(positives, n); ni = rng.choice(negatives, n)
            delta = a[pi] - a[ni]
            base_delta = p[pi] - p[ni]
            wrong = base_delta < 0
            unanimous = (delta < 0).all(axis=1)
            assert not (unanimous & ~wrong).any()
            totals['base_wrong'] += int(wrong.sum())
            totals['base_ties'] += int((base_delta == 0).sum())
            totals['all_seven_strictly_wrong'] += int(unanimous.sum())
            totals['base_wrong_with_some_component_correct'] += int((wrong & (delta > 0).any(axis=1)).sum())
            totals['base_correct_with_some_component_wrong'] += int(((base_delta > 0) & (delta < 0).any(axis=1)).sum())
        denom = protocol['pairs_per_fold']
        ratio = totals['all_seven_strictly_wrong'] / totals['base_wrong']
        rate = totals['all_seven_strictly_wrong'] / denom
        rows.append({'fold': fold, 'pairs': denom, 'counts': totals,
                     'estimated_current_pair_error': (totals['base_wrong'] + .5 * totals['base_ties']) / denom,
                     'estimated_unanimous_wrong_pair_rate': rate,
                     'unanimous_wrong_fraction_of_current_strict_errors': ratio,
                     'approx_mc_se_of_conditional_fraction': float(np.sqrt(ratio * (1 - ratio) / totals['base_wrong'])),
                     'sampled_auc_upper_bound_for_global_nonnegative_blends': 1 - rate,
                     'bound_note': 'Optimistic bound, not an achievable blend, not a ceiling on new models, not a public-score forecast.'})
    result = {'protocol': protocol, 'folds': rows, 'hashes': hashes,
              'elapsed_seconds': time.monotonic() - started}
    (OUT / 'pair_agreement_diagnostics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'folds': rows, 'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
