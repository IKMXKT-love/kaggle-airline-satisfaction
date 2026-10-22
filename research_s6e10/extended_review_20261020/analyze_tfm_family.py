"""All prespecified fixed-weight contrasts on public TFM OOF; no fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from analyze_public_references import comparison

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main():
    target = OUT / 'tfm_family_comparison.json'
    if target.exists():
        raise RuntimeError('Refusing overwrite')
    protocol = json.loads((OUT / 'tfm_references/protocol.json').read_text(encoding='utf-8'))
    y = pd.read_csv(ROOT / 'train.csv', usecols=['satisfaction']).satisfaction.to_numpy(dtype='int64')
    folds = np.load(OUT / 'reference_predictions/dariushafshar__folds_seed42.npy', allow_pickle=False)
    with np.load(ROOT / '.artifact_runtime/lean_cv_followup/8f29c4956de9c49f6f97/predictions.npz', allow_pickle=False) as z:
        base = z['blended_oof'].copy()
    s0 = logit(np.clip(base, 1e-6, 1 - 1e-6))
    rows, scores = [], []
    for name in protocol['models']:
        p = np.load(OUT / ('tfm_references/oof_' + name + '.npy'), allow_pickle=False).astype('float64')
        assert p.shape == y.shape and np.isfinite(p).all()
        if p.min() < 0 or p.max() > 1:
            s = p
            scale = 'already logits'
        else:
            s = logit(np.clip(p, 1e-6, 1 - 1e-6))
            scale = 'probabilities'
        scores.append(s)
        blended = expit(.8 * s0 + .2 * s)
        r = {'name': name, 'scale': scale, 'standalone_auc': float(roc_auc_score(y, p)),
             'fixed20': comparison(y, blended, base),
             'five_fold_deltas': [float(roc_auc_score(y[folds == f], blended[folds == f]) - roc_auc_score(y[folds == f], base[folds == f])) for f in range(5)]}
        rows.append(r)
        print(json.dumps(r), flush=True)
    avg = np.mean(scores, axis=0)
    blend = expit(.8 * s0 + .2 * avg)
    collective = {'name': 'equal_logits_all_nine', 'standalone_auc': float(roc_auc_score(y, avg)),
                  'fixed20': comparison(y, blend, base),
                  'five_fold_deltas': [float(roc_auc_score(y[folds == f], blend[folds == f]) - roc_auc_score(y[folds == f], base[folds == f])) for f in range(5)]}
    result = {'training': False, 'model_inference': False, 'submission_generated': False,
              'scope': 'Prespecified exploratory fixed-prediction comparisons only. Public label and split provenance not fully audited; legacy baseline outer stopping, historical selection, row-order reliance and multiple comparisons apply. No independent generalization claim.',
              'current_auc': float(roc_auc_score(y, base)), 'models': rows, 'collective': collective}
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(collective), flush=True)


if __name__ == '__main__':
    main()
