"""Research contrasts on fixed public OOF arrays. No fitting or model inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REF = OUT / 'reference_predictions'


def placements(y, p):
    pos, neg = p[y == 1], p[y == 0]
    a, b = np.sort(pos), np.sort(neg)
    vp = (np.searchsorted(b, pos, 'left') + np.searchsorted(b, pos, 'right')) / (2 * len(neg))
    vn = 1 - (np.searchsorted(a, neg, 'left') + np.searchsorted(a, neg, 'right')) / (2 * len(pos))
    np.testing.assert_allclose(vp.mean(), roc_auc_score(y, p), atol=1e-12)
    return vp, vn


def comparison(y, candidate, baseline):
    vp, vn = placements(y, candidate)
    bp, bn = placements(y, baseline)
    dp, dn = vp - bp, vn - bn
    delta = float(dp.mean())
    se = float(np.sqrt(dp.var(ddof=1) / len(dp) + dn.var(ddof=1) / len(dn)))
    return {'auc': float(vp.mean()), 'delta': delta, 'fixed_prediction_approx_se': se,
            'fixed_prediction_nominal_interval95': [delta - 1.96 * se, delta + 1.96 * se]}


def main():
    target = OUT / 'public_reference_comparison.json'
    if target.exists():
        raise RuntimeError('Refusing to overwrite previous contrasts')
    assert json.loads((REF / 'protocol.json').read_text(encoding='utf-8'))['fixed_tfm_use'].startswith('20 percent')
    tr = pd.read_csv(ROOT / 'train.csv', usecols=['id', 'satisfaction'])
    y = tr.satisfaction.to_numpy(dtype='int64')
    baseline_file = ROOT / '.artifact_runtime/lean_cv_followup/8f29c4956de9c49f6f97/predictions.npz'
    with np.load(baseline_file, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'], tr.id.to_numpy())
        current = z['blended_oof'].copy()
    folds = np.load(REF / 'dariushafshar__folds_seed42.npy', allow_pickle=False)
    recreated = np.full(len(y), -1)
    for f, (_, q) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y)):
        recreated[q] = f
    np.testing.assert_array_equal(folds, recreated)
    out = {'training': False, 'model_inference': False, 'submission_generated': False,
           'scope': 'Exploratory contrasts of already selected fixed prediction arrays, not end-to-end isolated validation. Own baseline has legacy outer-stopping caveats. Public stack has indirect base-OOF target dependence. Different model partitions; nominal intervals are fixed-prediction only, uncorrected for selection.',
           'rows': len(y), 'current_legacy_oof_auc': float(roc_auc_score(y, current)),
           'current_legacy_oof_folds': [float(roc_auc_score(y[folds == f], current[folds == f])) for f in range(5)], 'tfm': []}
    with np.load(ROOT / '.artifact_runtime/strong_correction_20261020/b1385337dbf19f77097b/outer_predictions.npz', allow_pickle=False) as z:
        strict_idx, strict_base = z['index'].copy(), z['base'].copy()
    out['strict_baseline_auc'] = float(roc_auc_score(y[strict_idx], strict_base))
    for name in ['tabpfn_full_raw', 'tabpfn_catfd10']:
        path = REF / ('goodpjw2008__oof_' + name + '.csv')
        df = pd.read_csv(path)
        assert df.id.is_unique and len(df) == len(tr)
        np.testing.assert_array_equal(df.id.to_numpy(), tr.id.to_numpy())
        p = df.pred.to_numpy()
        assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
        blended = expit(.8 * logit(np.clip(current, 1e-6, 1 - 1e-6)) + .2 * logit(np.clip(p, 1e-6, 1 - 1e-6)))
        row = {'name': name, 'author': 'goodpjw2008', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'standalone_auc': float(roc_auc_score(y, p)), 'fixed_20percent_logit_contrast': comparison(y, blended, current),
               'native_five_fold_deltas': [float(roc_auc_score(y[folds == f], blended[folds == f]) - roc_auc_score(y[folds == f], current[folds == f])) for f in range(5)]}
        blend_strict = expit(.8 * logit(np.clip(strict_base, 1e-6, 1 - 1e-6)) + .2 * logit(np.clip(p[strict_idx], 1e-6, 1 - 1e-6)))
        row['on_strict_baseline_rows_exploratory'] = comparison(y[strict_idx], blend_strict, strict_base)
        out['tfm'].append(row)
    external = np.load(REF / 'dariushafshar__stack247_oof.npy', allow_pickle=False)
    assert external.shape == y.shape and np.isfinite(external).all() and ((external >= 0) & (external <= 1)).all()
    out['external_stack_reference_only'] = comparison(y, external, current)
    out['external_stack_native_fold_deltas'] = [float(roc_auc_score(y[folds == f], external[folds == f]) - roc_auc_score(y[folds == f], current[folds == f])) for f in range(5)]
    out['external_stack_not_own_training_result'] = True
    out['baseline_sha256'] = hashlib.sha256(baseline_file.read_bytes()).hexdigest()
    target.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
