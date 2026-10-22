"""Label-free signature support and descriptive conditional dependence; no fit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RATINGS = ['Inflight wifi service', 'Departure/Arrival time convenient',
           'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
           'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
           'Baggage handling', 'Checkin service', 'Cleanliness']


def main():
    train = pd.read_csv(ROOT/'train.csv', usecols=RATINGS+['Class', 'Type of Travel', 'satisfaction'])
    test = pd.read_csv(ROOT/'test.csv', usecols=RATINGS)
    exact = train[RATINGS].value_counts(dropna=False)
    hashed = pd.util.hash_pandas_object(train[RATINGS], index=False)
    counts = hashed.value_counts()
    assert len(exact) == len(counts)
    np.testing.assert_array_equal(np.sort(exact.to_numpy()), np.sort(counts.to_numpy()))
    result = dict(note='Descriptive diagnostics only; no satisfaction predictor trained.',
                  raw_columns=RATINGS, train_rows=len(train), test_rows=len(test),
                  exact_rating_patterns=len(exact),
                  rows_in_repeated_pattern=int(exact[exact>=2].sum()),
                  rows_in_pattern_ge10=int(exact[exact>=10].sum()),
                  collision_check='Exact tuple and 64-bit hash pattern multiplicities agree')
    def support(reference_hashes, query_hashes):
        frequency = reference_hashes.value_counts()
        n = query_hashes.map(frequency).fillna(0).to_numpy()
        return {f'fit_support_ge{k}':dict(rows=int((n>=k).sum()), fraction=float((n>=k).mean()))
                for k in [1,2,5,10]}
    result['test_from_full_train_support'] = support(
        hashed, pd.util.hash_pandas_object(test[RATINGS], index=False))
    folder = ROOT/'.artifact_runtime/followup_20261019/4261d13e038c1512ffc1'
    result['matched_split_support'] = {}
    for stage in ['screen', 'confirmation']:
        with np.load(folder/stage/'indices.npz') as z:
            result['matched_split_support'][stage] = support(hashed.iloc[z['fit']], hashed.iloc[z['scored']])
    result['within_label_correlations'] = []
    for label, group in train.groupby('satisfaction'):
        result['within_label_correlations'].append(dict(label=int(label), rows=len(group),
            seat_entertainment=float(group['Seat comfort'].corr(group['Inflight entertainment'])),
            seat_cleanliness=float(group['Seat comfort'].corr(group['Cleanliness']))))
    result['within_context_and_label_correlations'] = []
    for (cls, travel, label), group in train.groupby(['Class', 'Type of Travel', 'satisfaction']):
        if len(group) >= 10000:
            result['within_context_and_label_correlations'].append(dict(
                cabin=cls, travel=travel, label=int(label), rows=len(group),
                seat_entertainment=float(group['Seat comfort'].corr(group['Inflight entertainment']))))
    result['input_sha256'] = {}
    for name in ['train.csv', 'test.csv']:
        h = hashlib.sha256()
        with (ROOT/name).open('rb') as f:
            while block := f.read(8*1024*1024):
                h.update(block)
        result['input_sha256'][name] = h.hexdigest()
    output = ROOT/'research_s6e10/joint_rating_diagnostics.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
