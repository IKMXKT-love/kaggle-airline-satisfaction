"""Replay checkpoints and independently rebuild every reported AUC and gate."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from multitask_support import (ROOT, SPECS, RawEncoder, RatingMLP, predict, sha,
                               to_device, write_json)

FAMILY = ROOT / '.artifact_runtime/denoising_multitask_20261019'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def auc_difference(y, first, second):
    # Independent implementation of the correlated placement variance.
    def contributions(score):
        positive = score[y == 1]
        negative = score[y == 0]
        sorted_positive = np.sort(positive)
        sorted_negative = np.sort(negative)
        pos = (np.searchsorted(sorted_negative, positive, 'left')
               + np.searchsorted(sorted_negative, positive, 'right')) / (2 * len(negative))
        neg = 1 - (np.searchsorted(sorted_positive, negative, 'left')
                   + np.searchsorted(sorted_positive, negative, 'right')) / (2 * len(positive))
        return pos, neg
    ap, an = contributions(first)
    bp, bn = contributions(second)
    differences = [ap - bp, an - bn]
    standard_error = math.sqrt(sum(float(d.var(ddof=1)) / len(d) for d in differences))
    delta = float(roc_auc_score(y, first) - roc_auc_score(y, second))
    return delta, [delta - 1.96 * standard_error, delta + 1.96 * standard_error]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--skip-replay', action='store_true')
    args = parser.parse_args()
    directory = args.directory or Path(read(FAMILY / 'latest.json')['directory'])
    protocol = read(directory / 'protocol.json')
    for name, expected in protocol['hashes'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in protocol['protected'].items():
        assert sha(ROOT / name) == expected, name
    probe = read(FAMILY / 'probe.json')
    assert sha(FAMILY / 'probe.json') == protocol['probe_sha256']
    assert probe['maximum_primary_weight_change'] > 0
    assert probe['maximum_auxiliary_weight_change'] > 0
    assert probe['probe_parameters_discarded'] and probe['no_stopping_or_scored_data_used']
    frame = pd.read_csv(ROOT / 'train.csv')
    x = frame.drop(columns=['id', 'satisfaction'])
    y = frame.satisfaction.to_numpy(dtype='int64')
    config = protocol['training']
    results = {}
    if not args.skip_replay:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
    for stage in ['screen', 'confirmation']:
        folder = directory / stage
        if not (folder / 'summary.json').exists():
            continue
        summary = read(folder / 'summary.json')
        with np.load(folder / 'indices.npz') as saved:
            idx = {k: saved[k].copy() for k in ['fit', 'stopping', 'scored']}
        merged = np.concatenate(list(idx.values()))
        assert len(np.unique(merged)) == len(merged)
        reference = ROOT / '.artifact_runtime/followup_20261019/4261d13e038c1512ffc1' / stage
        with np.load(reference / 'indices.npz') as saved:
            for k, value in idx.items():
                np.testing.assert_array_equal(value, saved[k])
        if stage == 'confirmation':
            first = read(directory / 'screen/summary.json')
            assert first['passed']
            assert read(folder / 'frozen_selection.json') == first['winner']
            assert len(summary['trials']) == 1
            assert summary['winner']['spec'] == first['winner']['spec']
            assert summary['winner']['usage'] == first['winner']['usage']
        encoder = RawEncoder().fit(x.iloc[idx['fit']])
        assert encoder.state() == read(folder / 'encoder.json')
        predictions, replays, initial_hashes = {}, {}, []
        requested = SPECS if stage == 'screen' else [summary['winner']['spec']]
        for spec in requested:
            arm = folder / spec
            detail = read(arm / 'complete.json')
            history = read(arm / 'history.json')
            cap = config['warm_epochs'] + config['fine_epochs']
            assert len(history) == cap == detail['epochs_completed']
            assert [r['epoch'] for r in history] == list(range(1, cap + 1))
            assert detail['updates_completed'] == cap * math.ceil(len(idx['fit']) / config['batch_size'])
            warm = history[:config['warm_epochs']]
            fine = history[config['warm_epochs']:]
            assert all(r['phase'] == 'warm' for r in warm)
            assert all(r['phase'] == 'fine' and r['recovery_loss'] == 0 for r in fine)
            if spec == 'C_multitask':
                assert all(r['recovery_loss'] > 0 for r in warm)
            else:
                assert all(r['recovery_loss'] == 0 for r in warm)
            best = max(fine, key=lambda row: row['stopping_auc'])
            assert detail['best_epoch'] == best['epoch']
            assert detail['best_stopping_auc'] == best['stopping_auc']
            assert detail['best_checkpoint_sha256'] == sha(arm / 'best.pt')
            with np.load(arm / 'best_stopping.npz') as saved:
                assert abs(roc_auc_score(y[idx['stopping']], saved['prediction']) - best['stopping_auc']) < 1e-12
            initial_hashes.append(detail['initial_hash'])
            torch.manual_seed(config['seed'])
            model = RatingMLP(encoder.state(), protocol['model'], y[idx['fit']].mean())
            initial_recovery = model.recovery.weight.detach().clone()
            state = torch.load(arm / 'best.pt', map_location='cpu', weights_only=True)
            difference = float((state['recovery.weight'] - initial_recovery).abs().max())
            assert (difference > 0) if spec == 'C_multitask' else (difference == 0)
            model.load_state_dict(state)
            with np.load(folder / (spec + '.npz')) as saved:
                np.testing.assert_array_equal(saved['index'], idx['scored'])
                p = saved['prediction'].copy()
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            assert abs(roc_auc_score(y[idx['scored']], p) - summary['standalone_auc'][spec]) < 1e-12
            predictions[spec] = p
            if not args.skip_replay:
                values = to_device(encoder.transform(x.iloc[idx['scored']]), 'cuda')
                replayed = predict(model.cuda(), values, config['prediction_batch_size'])
                max_difference = float(np.max(np.abs(replayed - p)))
                assert max_difference < 1e-7, (spec, max_difference)
                replays[spec] = dict(rows=len(p), maximum_prediction_difference=max_difference,
                                     auxiliary_weight_difference=difference)
                del values
                model.cpu()
                torch.cuda.empty_cache()
        assert len(set(initial_hashes)) == 1
        source_predictions = []
        for name in ['baseline', 'lean_mean']:
            with np.load(reference / (name + '.npz')) as saved:
                np.testing.assert_array_equal(saved['index'], idx['scored'])
                source_predictions.append(saved['prediction'].copy())
        v4_margin, lean_margin = [logit(np.clip(p, 1e-6, 1 - 1e-6)) for p in source_predictions]
        baseline = expit(.65 * v4_margin + .35 * lean_margin)
        target = y[idx['scored']]
        assert abs(roc_auc_score(target, baseline) - summary['baseline_auc']) < 1e-12
        for trial in summary['trials']:
            candidate_margin = logit(np.clip(predictions[trial['spec']], 1e-6, 1 - 1e-6))
            if trial['usage'] == 'replace35':
                blended = expit(.65 * v4_margin + .35 * candidate_margin)
            else:
                assert trial['usage'] == 'add20'
                blended = expit(.8 * (.65 * v4_margin + .35 * lean_margin) + .2 * candidate_margin)
            delta, interval = auc_difference(target, blended, baseline)
            assert abs(delta - trial['delta']) < 1e-12
            assert abs(roc_auc_score(target, blended) - trial['auc']) < 1e-12
            np.testing.assert_allclose(interval, trial['interval95'], atol=1e-12, rtol=0)
        winner = max(summary['trials'], key=lambda row: row['delta'])
        assert winner == summary['winner']
        threshold = protocol['screen_gate'] if stage == 'screen' else protocol['confirmation_gate']
        assert summary['passed'] == (winner['delta'] >= threshold and winner['interval95'][0] > 0)
        for key, comparison in summary['arm_comparisons'].items():
            first_name, second_name = key.split('_minus_')
            delta, interval = auc_difference(target, predictions[first_name], predictions[second_name])
            assert abs(delta - comparison['delta']) < 1e-12
            np.testing.assert_allclose(interval, comparison['interval95'], atol=1e-12, rtol=0)
        results[stage] = dict(rows={k: len(v) for k, v in idx.items()}, replay=replays,
                             metrics_rebuilt=True, gates_rebuilt=True, shared_initial_weights=True,
                             warm_and_fine_budgets_verified=True,
                             recovery_head_updated_only_in_multitask=True)
    assert 'screen' in results
    screen = read(directory / 'screen/summary.json')
    if not screen['passed']:
        assert 'confirmation' not in results
    check = dict(passed=True, original_files_unchanged=True, original_files=len(protocol['protected']),
                 probe_weight_updates_verified=True, stages=results,
                 checkpoint_replay=not args.skip_replay,
                 no_new_submission_generated_by_this_experiment=True)
    write_json(directory / 'verification.json', check)
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
