"""Independent rank-sum recomputation and producer-log checks; no fitting."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def auc(y, score):
    positive = y == 1
    n_pos = int(positive.sum())
    ranks = rankdata(score, method='average')
    return float((ranks[positive].sum()-n_pos*(n_pos+1)/2)/(n_pos*(len(y)-n_pos)))


def main():
    train = pd.read_csv(ROOT/'train.csv',usecols=['id','satisfaction'])
    y = train.satisfaction.astype(int).to_numpy()
    test_ids = pd.read_csv(ROOT/'test.csv',usecols=['id']).id.to_numpy()
    report = json.loads((HERE/'kumo_contrasts.json').read_text('utf-8'))
    receipts = json.loads((HERE/'kumo_download_receipts.json').read_text('utf-8'))
    for row in receipts:
        p = HERE/'kumo_parts'/Path(row['file']).name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
    folds = list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y))
    with np.load(HERE/'kumo_threebag_predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'],train.id)
        np.testing.assert_array_equal(z['test_id'],test_ids)
        candidate = z['oof']
        tests = z['test']
    out, test_parts = [], []
    tag = 'kumolarge_fsE_c30000_e4_o0'
    for b in range(3):
        p = np.full(len(y), np.nan)
        for k,(_,v) in enumerate(folds):
            p[v] = np.load(HERE/'kumo_parts'/f'{tag}_b{b}_f{k}.npy',allow_pickle=False)
        out.append(p)
        test_parts.append(np.load(HERE/'kumo_parts'/f'{tag}_b{b}_test.npy',allow_pickle=False))
    np.testing.assert_array_equal(candidate,np.mean(out,axis=0))
    np.testing.assert_array_equal(tests,np.mean(test_parts,axis=0))
    assert abs(auc(y,candidate)-report['standalone_candidate_auc']) < 2e-15
    with np.load(ROOT/'.artifact_runtime/public_pool_trial_20261020/selected_predictions.npz',allow_pickle=False) as z:
        baseline = z['oof_logit']
    raw_manifest = json.loads((ROOT/'.artifact_runtime/public_pool_trial_20261020/matrix_manifest.json').read_text('utf-8'))
    raw_idx = next(i for i,c in enumerate(raw_manifest['columns']) if c['member']=='tfm_kumolarge')
    raw = np.load(ROOT/'.artifact_runtime/public_pool_trial_20261020/train_logits.npy',mmap_mode='r',allow_pickle=False)[:,raw_idx]
    for row in report['contrasts']:
        signal = raw if row['candidate']=='kumo_raw_existing' else logit(np.clip(candidate,1e-6,1-1e-6))
        score = (1-row['weight'])*baseline+row['weight']*signal
        assert abs(auc(y,score)-row['auc']) < 2e-15
        for k,(_,v) in enumerate(folds):
            delta = auc(y[v],score[v])-auc(y[v],baseline[v])
            assert abs(delta-row['fold_deltas'][k]) < 2e-15
    log = (HERE/'kumo_v14_log.txt').read_text('utf-8')
    matches = re.findall(r'^bag 2\s+(\d): AUC (0\.\d+)',log,re.M)
    assert len(matches)==5
    for k,saved in matches:
        v = folds[int(k)][1]
        assert f'{auc(y[v],out[2][v]):.5f}' == saved
    expected = 'd3029122f333be8c98844b8af4c7f2f2fae91d74487e95994a488df1684a7728'
    actual = hashlib.sha256((ROOT/'submission_public_pool_20261020.csv').read_bytes()).hexdigest()
    assert actual == expected
    receipt = {'checks_passed': True,'auc_implementation':'independent Mann-Whitney rank sum with average ties',
               'part_hashes_checked':len(receipts),'bag2_producer_fold_scores_matched':5,
               'all_fixed_contrasts_reproduced':True,'current_best_sha256':actual,
               'limitations':'Array verification does not independently prove producer cache provenance or end-to-end independence.'}
    (HERE/'kumo_independent_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
