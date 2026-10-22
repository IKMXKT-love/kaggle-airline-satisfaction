"""Verify saved candidate, indices, convergence and protected previous best. No fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from fit_public_pool import objective

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'.artifact_runtime/public_pool_trial_20261020'


def main():
    meta = json.loads((ROOT/'submission_public_pool_20261020.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(meta['submission'], float_precision='round_trip')
    tr = pd.read_csv(ROOT/'train.csv', usecols=['id', 'satisfaction'])
    test = pd.read_csv(ROOT/'test.csv', usecols=['id'])
    sample = pd.read_csv(ROOT/'sample_submission.csv', nrows=1)
    assert list(frame.columns) == list(sample.columns) == ['id', 'satisfaction']
    assert frame.id.is_unique and len(frame) == len(test) == 299844
    np.testing.assert_array_equal(frame.id.to_numpy(), test.id.to_numpy())
    p = frame.satisfaction.to_numpy(); assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    digest = hashlib.sha256(Path(meta['submission']).read_bytes()).hexdigest()
    assert digest == meta['submission_sha256']
    with np.load(RUN/'selected_predictions.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'], tr.id.to_numpy())
        np.testing.assert_array_equal(z['test_id'], test.id.to_numpy())
        np.testing.assert_allclose(p, expit(z['test_logit']), atol=1e-15, rtol=0)
        oof, test_logit = z['oof_logit'].copy(), z['test_logit'].copy()
    auc = float(roc_auc_score(tr.satisfaction, oof))
    assert abs(auc-meta['pooled_aucs'][meta['selected_arm']]) < 1e-12
    with np.load(RUN/'selected_full_refit.npz', allow_pickle=False) as z: w = z['weights'].copy()
    xt = np.load(RUN/'test_logits.npy', mmap_mode='r', allow_pickle=False)
    ix = np.random.default_rng(20261020).choice(len(test), 2048, replace=False)
    with threadpool_limits(limits=4):
        np.testing.assert_allclose(xt[ix, :len(w)-1]@w[:-1]+w[-1], test_logit[ix], atol=2e-12, rtol=0)
        x = np.load(RUN/'train_logits.npy', mmap_mode='r', allow_pickle=False)
        value, gradient = objective(w, x, tr.satisfaction.to_numpy(dtype='int64'), np.ones(len(tr), dtype=bool), len(w)-1)
    assert np.abs(gradient).max() < 1e-8
    source = pd.read_csv(ROOT/'public_pool_sources_20261020.csv', float_precision='round_trip')
    assert len(source) == len(w)-1 == meta['public_columns']
    np.testing.assert_allclose(source.coefficient, w[:-1], atol=1e-15, rtol=0)
    previous = ROOT/'submission_lean_realmlp_10fold.csv'
    previous_digest = hashlib.sha256(previous.read_bytes()).hexdigest()
    assert previous_digest == 'b8cbfd891761efa14fac950684f03929b01da2d10a2c868fdb8228e7cb878eb8'
    ref = ROOT/'research_s6e10/extended_review_20261020/reference_predictions/stack247_frozen_v1/submission.csv'
    ref_p = pd.read_csv(ref, float_precision='round_trip').satisfaction.to_numpy()
    receipt = {'passed': True, 'training': False, 'new_public_score': None, 'rows': len(frame),
               'column_names_and_ids_verified': True, 'finite_probabilities': True,
               'csv_matches_saved_test_prediction': True, 'saved_linear_model_sample_reconstruction_verified': True,
               'full_training_objective': float(value), 'saved_full_model_max_gradient': float(np.abs(gradient).max()),
               'oof_auc_recomputed': auc, 'source_rows': len(source), 'submission_sha256': digest,
               'test_rank_correlation_with_author_frozen247_reference': float(spearmanr(p, ref_p).statistic),
               'max_abs_probability_difference_from_author_reference': float(np.abs(p-ref_p).max()),
               'reference_comparison_note': 'Prediction similarity is not a public-score or generalization guarantee.',
               'previous_best_sha256': previous_digest, 'previous_best_unchanged': True}
    (RUN/'independent_verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
