"""Descriptive stability of rating-pattern errors; no training or selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from joint_rating_diagnostics import RATINGS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.artifact_runtime/followup_20261019/4261d13e038c1512ffc1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    train = pd.read_csv(ROOT / 'train.csv', usecols=RATINGS + ['satisfaction'])
    hashes = pd.util.hash_pandas_object(train[RATINGS], index=False)
    # Verify tuple/hash group membership, not only their count distributions.
    membership = train[RATINGS].copy()
    membership['pattern'] = hashes
    assert not membership.drop_duplicates().pattern.duplicated().any()
    groups = pd.DataFrame({'pattern': hashes, 'y': train.satisfaction})
    counts = groups.groupby('pattern').y.agg(['size', 'sum'])
    composition = []
    for minimum in [2, 10, 20]:
        block = counts[counts['size'] >= minimum]
        mixed = (block['sum'] > 0) & (block['sum'] < block['size'])
        composition.append(dict(
            minimum_train_support=minimum, patterns=len(block),
            rows=int(block['size'].sum()),
            rows_in_mixed_label_patterns=int(block.loc[mixed, 'size'].sum()),
            in_sample_majority_fraction=float(
                np.maximum(block['sum'], block['size'] - block['sum']).sum()
                / block['size'].sum())))

    frames, indices, artifacts = {}, {}, {}
    for stage in ['screen', 'confirmation']:
        folder = CACHE / stage
        with np.load(folder / 'indices.npz') as saved:
            index = saved['scored'].copy()
        probabilities = []
        for name in ['baseline', 'lean_mean']:
            path = folder / (name + '.npz')
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['index'], index)
                probability = saved['prediction'].copy()
            assert np.isfinite(probability).all()
            assert ((probability >= 0) & (probability <= 1)).all()
            probabilities.append(probability)
            artifacts[str(path.relative_to(ROOT))] = sha256(path)
        score = expit(.65 * logit(np.clip(probabilities[0], 1e-6, 1 - 1e-6))
                      + .35 * logit(np.clip(probabilities[1], 1e-6, 1 - 1e-6)))
        part = groups.iloc[index].copy()
        part['error'] = part.y.to_numpy() - score
        frames[stage] = part.groupby('pattern').agg(
            rows=('y', 'size'), target_rate=('y', 'mean'),
            mean_error=('error', 'mean'))
        indices[stage] = index
        artifacts[str((folder / 'indices.npz').relative_to(ROOT))] = sha256(folder / 'indices.npz')
    assert len(np.intersect1d(indices['screen'], indices['confirmation'])) == 0
    shared = frames['screen'].join(frames['confirmation'], how='inner',
                                   lsuffix='_screen', rsuffix='_confirmation')
    stability = []
    for minimum in [3, 5, 10]:
        block = shared[(shared.rows_screen >= minimum)
                       & (shared.rows_confirmation >= minimum)]
        stability.append(dict(
            minimum_scored_support_per_stage=minimum, common_patterns=len(block),
            screen_rows=int(block.rows_screen.sum()),
            confirmation_rows=int(block.rows_confirmation.sum()),
            target_rate_pearson=float(block.target_rate_screen.corr(block.target_rate_confirmation)),
            mean_error_pearson=float(block.mean_error_screen.corr(block.mean_error_confirmation))))
    result = dict(
        note='Exploratory unweighted pattern-level correlations. No new model fit. '
             'Not an independent blind test; predictions share training history. '
             'No significance test or reliability correction. Small groups and '
             'different contexts can obscure real effects. In-sample majority '
             'fraction is not validation accuracy or a Bayes ceiling.',
        error_definition='y - expit(0.65*logit(v4)+0.35*logit(lean_mean)), clip=1e-6',
        collision_check='Each hash maps to exactly one distinct rating tuple',
        label_composition=composition, pattern_stability=stability,
        inputs_sha256={'train.csv': sha256(ROOT / 'train.csv'), **artifacts})
    path = ROOT / 'research_s6e10/rating_residual_stability.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(label_composition=composition, pattern_stability=stability), indent=2))


if __name__ == '__main__':
    main()
