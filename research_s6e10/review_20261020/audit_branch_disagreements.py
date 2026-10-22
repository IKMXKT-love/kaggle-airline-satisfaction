"""Descriptive conditional residual contrasts on cached branch scores only."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / '.artifact_runtime/strong_correction_20261020/b1385337dbf19f77097b'
CONTEXT = ['Class', 'Type of Travel', 'Customer Type']
NAMES = ['old_xgb', 'old_cat', 'control', 'realmlp', 'lgb', 'lean_42', 'lean_20261019']


def main():
    started = time.monotonic()
    protocol = {'training': False, 'model_inference': False, 'date': '2026-10-20',
                'score_bin_width': .02, 'minimum_rows_per_sign_per_cell': 30,
                'difference_threshold_in_logits': .25,
                'cells': [['score_bin'], CONTEXT + ['score_bin']],
                'limitation': 'Exploratory fixed-prediction calibration contrasts, not AUC gains. No model or weight selected; intervals uncorrected for multiple comparisons and training history.'}
    target = OUT / 'branch_disagreement_protocol.json'
    if target.exists():
        raise RuntimeError('Refusing to overwrite prior diagnostic')
    target.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    tr = pd.read_csv(ROOT / 'train.csv')
    rows = []
    for fold in [0, 1]:
        folder = BASE / f'outer_{fold}/refit/full_fit'
        with np.load(folder / 'combined.npz') as z:
            idx, p = z['index'].copy(), z['prediction'].copy()
        frame = tr.iloc[idx][CONTEXT].reset_index(drop=True).copy()
        frame['score_bin'] = np.minimum((p / .02).astype(int), 49)
        frame['residual'] = tr.satisfaction.to_numpy(dtype='float64')[idx] - p
        components = {}
        for name in NAMES:
            with np.load(folder / (name + '.npz')) as z:
                np.testing.assert_array_equal(z['index'], idx)
                components[name] = z['prediction'].copy()
        components['lean_mean'] = (components.pop('lean_42') + components.pop('lean_20261019')) / 2
        base_score = logit(np.clip(p, 1e-6, 1 - 1e-6))
        for name, probability in components.items():
            difference = logit(np.clip(probability, 1e-6, 1 - 1e-6)) - base_score
            frame['positive'] = difference >= .25
            valid = np.abs(difference) >= .25
            for cols in protocol['cells']:
                cells = frame[valid].groupby(cols + ['positive'], observed=True).residual.agg(['size', 'mean', 'var'])
                common = cells.xs(False, level='positive').join(cells.xs(True, level='positive'), how='inner', lsuffix='_lo', rsuffix='_hi')
                common = common[(common.size_lo >= 30) & (common.size_hi >= 30)]
                row = {'fold': fold, 'branch': name, 'cells': cols, 'common_cells': len(common),
                       'covered_rows': int(common.size_lo.sum() + common.size_hi.sum())}
                if len(common):
                    w = (2 * common.size_lo * common.size_hi / (common.size_lo + common.size_hi)).to_numpy(copy=True)
                    w /= w.sum()
                    row['higher_minus_lower_branch_residual'] = float(w @ (common.mean_hi - common.mean_lo).to_numpy())
                    row['approx_fixed_cell_se'] = float(np.sqrt(w ** 2 @ (common['var_hi'] / common.size_hi + common['var_lo'] / common.size_lo).to_numpy()))
                rows.append(row)
    result = {'protocol': protocol, 'rows': rows, 'elapsed_seconds': time.monotonic() - started}
    (OUT / 'branch_disagreement_diagnostics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for row in rows:
        if len(row['cells']) > 1:
            print(row)
    print('elapsed_seconds', result['elapsed_seconds'])


if __name__ == '__main__':
    main()
