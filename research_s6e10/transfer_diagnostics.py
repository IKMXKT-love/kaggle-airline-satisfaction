"""Read-only feasibility checks for methods found in past competition writeups."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent/'cross_competition'
OUT.mkdir(exist_ok=True)
train = pd.read_csv(ROOT/'train.csv')
orig = pd.read_csv(ROOT/'research_s6e10/official_data.csv')
test = pd.read_csv(ROOT/'test.csv')
cols = [c for c in train if c not in ['id', 'satisfaction']]
result = dict(note='Unsupervised descriptive diagnostics only. No target model fitted; full-source comparison is not a training feature pipeline.', columns=[])
for col in cols:
    a, b = train[col], orig[col]
    pa, pb = a.fillna('__NA__').value_counts(normalize=True), b.fillna('__NA__').value_counts(normalize=True)
    r = dict(name=col, train_unique=int(a.nunique()), original_unique=int(b.nunique()),
             train_values_absent_from_original_rows=int((~a.isin(b.dropna()) & a.notna()).sum()),
             source_vs_train_total_variation=float(pa.subtract(pb,fill_value=0).abs().sum()/2))
    if pd.api.types.is_numeric_dtype(a):
        r['train_noninteger_rows'] = int(((a % 1) != 0)[a.notna()].sum())
        r['original_noninteger_rows'] = int(((b % 1) != 0)[b.notna()].sum())
    result['columns'].append(r)
ratings = [c for c in cols if pd.api.types.is_numeric_dtype(train[c]) and c not in ['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']]
result['rating_columns'] = ratings
result['rating_pairs'] = []
for i,c in enumerate(ratings):
    for d in ratings[i+1:]:
        a = float(train[c].corr(train[d]))
        b = float(orig[c].corr(orig[d]))
        result['rating_pairs'].append(dict(left=c,right=d,train_pearson=a,original_pearson=b,absolute_change=abs(a-b)))
result['rating_pairs'].sort(key=lambda x:-x['absolute_change'])
result['id_blocks'] = []
for block, indexes in enumerate(np.array_split(np.argsort(train.id.to_numpy()), 10)):
    frame=train.iloc[indexes]
    result['id_blocks'].append(dict(block=block+1, rows=len(frame), target_rate=float(frame.satisfaction.mean()),
                                    age_mean=float(frame.Age.mean()), distance_mean=float(frame['Flight Distance'].mean())))
(OUT/'local_feasibility.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(numeric=[r for r in result['columns'] if 'train_noninteger_rows' in r],
                     largest_rating_correlation_changes=result['rating_pairs'][:5],
                     id_blocks=result['id_blocks']),indent=2))
