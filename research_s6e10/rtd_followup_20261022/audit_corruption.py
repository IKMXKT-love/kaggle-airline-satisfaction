"""Unlabelled marginal audit; no model fitting or prediction scoring."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RATINGS = ['Inflight wifi service','Departure/Arrival time convenient',
           'Ease of Online booking','Gate location','Food and drink','Online boarding',
           'Seat comfort','Inflight entertainment','On-board service','Leg room service',
           'Baggage handling','Checkin service','Cleanliness']
COLS = ['Gender','Customer Type','Age','Type of Travel','Class','Flight Distance',
        *RATINGS,'Departure Delay in Minutes','Arrival Delay in Minutes']

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    paths=['train.csv','test.csv','external_data/train.csv','external_data/test.csv']
    counts={c:Counter() for c in COLS}
    receipt=[]
    for name in paths:
        n=0
        for frame in pd.read_csv(ROOT/name,usecols=COLS,chunksize=100000):
            n+=len(frame)
            for c in COLS:
                vc=frame[c].value_counts(dropna=False)
                counts[c].update({('__MISSING__' if pd.isna(k) else k):int(v) for k,v in vc.items()})
        receipt.append({'path':name,'rows':n})
    n=sum(x['rows'] for x in receipt)
    rows=[]
    for c,co in counts.items():
        # Independent receiver/donor draws with replacement from the empirical pool.
        equal=sum(v*v for v in co.values())/(n*n)
        rows.append({'column':c,'unique_including_missing':len(co),'same_value_probability':equal,
                     'selected_probability':.15,'changed_probability':.15*(1-equal)})
    rat=[x for x in rows if x['column'] in RATINGS]
    result={'protocol':'Exact empirical independent donor marginal equality; 15% independent selection per field. Missing equals missing. Descriptive only; no satisfaction labels read.',
            'inputs':receipt,'total_rows':n,'columns':rows,
            'rating_mean_same_value_probability':sum(x['same_value_probability'] for x in rat)/len(rat),
            'expected_selected_fields_per_row':.15*len(COLS),
            'expected_changed_fields_per_row':sum(x['changed_probability'] for x in rows),
            'training':False,'auc_computed':False}
    (OUT/'corruption_marginals.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='columns'},ensure_ascii=False))
    for x in rows:
        print(x['column'],round(x['same_value_probability'],4),round(x['changed_probability'],4))

if __name__=='__main__':main()
