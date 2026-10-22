"""Summarize two published equality hypotheses; no fitting or prediction."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / '.artifact_runtime/strong_correction_20261020/b1385337dbf19f77097b'
BLOCKS = {
    'digital': ['Inflight wifi service', 'Departure/Arrival time convenient', 'Gate location'],
    'comfort': ['Food and drink', 'Seat comfort', 'Inflight entertainment', 'Cleanliness'],
}
CONTEXT = ['Class', 'Type of Travel', 'Customer Type']


def main():
    start = time.monotonic()
    protocol = {'date': '2026-10-20', 'training': False,
                'source': 'https://www.kaggle.com/code/mitudru/s6e10-analysis-first-equality-blocks-stack',
                'blocks': BLOCKS, 'score_bins': 'fixed probability width 0.02',
                'conditional_cells': CONTEXT + ['score_bin'], 'min_rows_each_flag': 30,
                'meaning': 'Exploratory calibration residual contrasts; no weights, predictions, AUC gains or noise ceiling inferred.'}
    target = OUT / 'equality_protocol.json'
    if target.exists():
        assert json.loads(target.read_text(encoding='utf-8')) == protocol
    else:
        target.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    tr = pd.read_csv(ROOT / 'train.csv')
    rows = []
    for fold in [0, 1]:
        with np.load(BASE / f'outer_{fold}/refit/full_fit/combined.npz') as z:
            idx, p = z['index'], z['prediction']
        x = tr.iloc[idx].reset_index(drop=True)
        residual = x.satisfaction.to_numpy(dtype='float64') - p
        for name, block in BLOCKS.items():
            flag = x[block].nunique(axis=1).to_numpy() == 1
            raw = []
            for value in [False, True]:
                m = flag == value
                raw.append({'equal': value, 'rows': int(m.sum()),
                            'positive_rate': float(x.satisfaction.to_numpy()[m].mean()),
                            'mean_prediction': float(p[m].mean()),
                            'mean_residual': float(residual[m].mean())})
            frame = x[CONTEXT].copy()
            frame['score_bin'] = np.minimum((p / .02).astype(int), 49)
            frame['equal'] = flag
            frame['residual'] = residual
            cells = frame.groupby(CONTEXT + ['score_bin', 'equal'], observed=True).residual.agg(['size', 'mean', 'var'])
            a = cells.xs(False, level='equal')
            b = cells.xs(True, level='equal')
            common = a.join(b, lsuffix='_no', rsuffix='_yes', how='inner')
            common = common[(common.size_no >= 30) & (common.size_yes >= 30)]
            item = {'fold': fold, 'block': name, 'unconditional': raw,
                    'common_context_score_cells': len(common),
                    'covered_rows': int(common.size_no.sum() + common.size_yes.sum())}
            if len(common):
                weights = (2 * common.size_no * common.size_yes / (common.size_no + common.size_yes)).to_numpy(copy=True)
                weights /= weights.sum()
                difference = (common.mean_yes - common.mean_no).to_numpy()
                variance = (common.var_yes / common.size_yes + common.var_no / common.size_no).to_numpy()
                item['weighted_equal_minus_nonequal_residual'] = float(weights @ difference)
                item['approx_fixed_cell_standard_error'] = float(np.sqrt(weights ** 2 @ variance))
            rows.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    (OUT / 'equality_diagnostics.json').write_text(json.dumps({'protocol': protocol, 'rows': rows,
        'elapsed_seconds': time.monotonic() - start}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
