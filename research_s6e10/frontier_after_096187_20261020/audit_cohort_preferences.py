"""Descriptive cohort ranking diagnostics, with no fitted gate or submission."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'.artifact_runtime/public_pool_trial_20261020'


def main():
    target = OUT/'cohort_preferences.json'
    protocol_file = OUT/'cohort_preferences_protocol.json'
    if target.exists() or protocol_file.exists(): raise FileExistsError('Preserve previous diagnostics')
    protocol = {
        'frozen_epoch': time.time(), 'training': False, 'model_inference': False, 'submission_generated': False,
        'cohorts': 'Customer Type x Type of Travel x Class, plus each factor alone',
        'experts': ['ab_tabpfn_catfd10', 'km_rmlp_f_nc', 'kga_cat_allcat_v1_k5', 'mi_xgb_d5', 'gpx_gen_v1'],
        'minimum_positives_and_negatives_per_fold': 100,
        'descriptive_flag': 'mean within-cohort expert-minus-current AUC > 2*paired fold SE and at least 4/5 positive',
        'limits': 'Exploratory, many comparisons without multiplicity correction. Group AUC does not determine global AUC; changing a group also changes between-group ranking. Native meta-folds inherit public OOF dependencies.'}
    protocol_file.write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    tr = pd.read_csv(ROOT/'train.csv', usecols=['satisfaction', 'Customer Type', 'Type of Travel', 'Class'])
    y = tr.satisfaction.to_numpy(dtype='int64')
    with np.load(RUN/'selected_predictions.npz', allow_pickle=False) as z: baseline = z['oof_logit'].copy()
    fold = np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy', allow_pickle=False)
    manifest = json.loads((RUN/'matrix_manifest.json').read_text(encoding='utf-8'))
    names = [m['member'] for m in manifest['columns']]
    x = np.load(RUN/'train_logits.npy', mmap_mode='r', allow_pickle=False)
    experts = {name: np.array(x[:, names.index(name)], copy=True) for name in protocol['experts']}
    definitions = []
    for factors in [['Customer Type', 'Type of Travel', 'Class'], ['Customer Type'], ['Type of Travel'], ['Class']]:
        key = tr[factors].astype(str).agg(' | '.join, axis=1)
        for value in sorted(key.unique()):
            definitions.append(('|'.join(factors)+'='+value, (key == value).to_numpy()))
    rows, omitted = [], []
    for cohort, mask in definitions:
        if any(min(int((mask & (fold == f) & (y == label)).sum()) for label in [0, 1]) < 100 for f in range(5)):
            omitted.append(cohort); continue
        base_auc = [float(roc_auc_score(y[mask & (fold == f)], baseline[mask & (fold == f)])) for f in range(5)]
        positive, negative = int((mask & (y == 1)).sum()), int((mask & (y == 0)).sum())
        pair_weight = positive*negative/((y == 1).sum()*(y == 0).sum())
        for name, pred in experts.items():
            aucs = [float(roc_auc_score(y[mask & (fold == f)], pred[mask & (fold == f)])) for f in range(5)]
            delta = np.array(aucs)-base_auc; se = float(delta.std(ddof=1)/np.sqrt(5))
            rows.append({'cohort': cohort, 'expert': name, 'rows': int(mask.sum()),
                         'within_cohort_pair_share_of_global_pairs': float(pair_weight),
                         'current_fold_auc': base_auc, 'expert_fold_auc': aucs,
                         'deltas': delta.tolist(), 'mean_delta': float(delta.mean()), 'paired_fold_se': se,
                         'positive_folds': int((delta > 0).sum()),
                         'descriptive_flag': bool(delta.mean() > 2*se and (delta > 0).sum() >= 4),
                         'within_group_only_global_pair_contribution': float(pair_weight*delta.mean())})
    flagged = [r for r in rows if r['descriptive_flag']]
    result = {'protocol': protocol, 'rows': rows, 'omitted_cohorts': omitted,
              'flagged': flagged, 'evaluated_comparisons': len(rows)}
    target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    top = sorted(rows, key=lambda r: r['mean_delta'], reverse=True)[:8]
    print(json.dumps({'comparisons': len(rows), 'flagged': flagged, 'largest_mean_deltas': top}, indent=2))


if __name__ == '__main__':
    main()
