"""Descriptive audit of existing saved fits, with no new model fitting or submissions."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT/'.artifact_runtime/public_pool_trial_20261020'


def auc(y, score):
    keep = y == 1
    n = int(keep.sum())
    ranks = rankdata(score,method='average')
    return float((ranks[keep].sum()-n*(n+1)/2)/(n*(len(y)-n)))


def main():
    started = time.monotonic()
    protocol = {
        'status':'exploratory post-selection audit of existing saved fits',
        'purpose':'Check whether existing members have useful signed corrective roles; not a new-model test.',
        'members':'All 11 members from the existing cdeotte TFM source, without selecting by removal results.',
        'contrasts':'Current OOF versus fixed deletion of the saved per-fold signed contribution; positive 5% and 20% additions to current OOF.',
        'no_coefficient_refitting':True,'no_weight_search':True,
        'limitations':['Deletion holds all other coefficients fixed; this is not a refitted leave-one-member-out estimate.',
                      'Existing OOF ancestry and historical selection remain; this is not independent validation.',
                      'Negative or large coefficients alone do not prove incremental predictive information.',
                      'The current pool already contains these members, so this diagnostic cannot determine whether a new excluded member would help.']}
    (HERE/'signed_roles_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    y = pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.astype(int).to_numpy()
    manifest = json.loads((CACHE/'matrix_manifest.json').read_text('utf-8'))['columns'][:246]
    X = np.load(CACHE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    base = np.load(CACHE/'selected_predictions.npz',allow_pickle=False)['oof_logit']
    reconstructed = np.empty(len(y))
    assignments = np.zeros(len(y),dtype=np.int8)
    weights,folds = [],[]
    for k in range(5):
        w = np.load(CACHE/f'public_only_fold{k}.npz',allow_pickle=False)['weights']
        with np.load(CACHE/f'public_only_fold{k}_validation.npz',allow_pickle=False) as z:
            idx,score = z['index'],z['logit']
            np.testing.assert_array_equal(base[idx],score)
        reconstructed[idx] = X[idx,:246]@w[:-1]+w[-1]
        assignments[idx] += 1
        weights.append(w);folds.append(idx)
    assert np.all(assignments==1)
    reconstruction_error = float(np.max(np.abs(reconstructed-base)))
    assert reconstruction_error < 1e-10
    W = np.stack(weights)
    baseline = auc(y,base)
    baseline_folds = [auc(y[i],base[i]) for i in folds]
    rows=[]
    for j,m in enumerate(manifest):
        if m['source']!='tfm':continue
        score = np.asarray(X[:,j])
        dropped = base.copy()
        for k,idx in enumerate(folds):dropped[idx] -= W[k,j]*score[idx]
        row = {'member':m['member'],'standalone_auc':auc(y,score),
               'fold_coefficients':W[:,j].tolist(),
               'fixed_deletion_auc':auc(y,dropped),'fixed_deletion_delta':auc(y,dropped)-baseline,
               'fixed_deletion_fold_deltas':[auc(y[idx],dropped[idx])-baseline_folds[k] for k,idx in enumerate(folds)]}
        for w in [0.05,0.2]:
            pred = (1-w)*base+w*score
            row[f'positive_add_{w}_delta'] = auc(y,pred)-baseline
        rows.append(row)
    all_negative = [{'member':m['member'],'coefficients':W[:,j].tolist()}
                    for j,m in enumerate(manifest) if np.all(W[:,j]<0)]
    out={'baseline_auc':baseline,'reconstruction_max_abs_error':reconstruction_error,
         'members_negative_in_all_five_saved_fits':len(all_negative),
         'always_negative_members':all_negative,'tfm_roles':rows,
         'wall_seconds':time.monotonic()-started,'new_model_fits':0,
         'current_best_sha256':hashlib.sha256((ROOT/'submission_public_pool_20261020.csv').read_bytes()).hexdigest()}
    (HERE/'signed_roles_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='always_negative_members'},indent=2),flush=True)


if __name__=='__main__':
    main()
