"""Independent artifact/score/isolation checks; no training or model selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while block := f.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()


def pair_check(y, p, q):
    # Independently derive positive/negative AUC influence values with ties.
    def influence(prediction):
        pos = np.sort(prediction[y == 1]); neg = np.sort(prediction[y == 0])
        a = (np.searchsorted(neg, prediction[y == 1], side='left') +
             np.searchsorted(neg, prediction[y == 1], side='right'))/(2*len(neg))
        b = 1-(np.searchsorted(pos, prediction[y == 0], side='left') +
               np.searchsorted(pos, prediction[y == 0], side='right'))/(2*len(pos))
        return a, b
    ap, an = influence(p); bp, bn = influence(q)
    dp, dn = ap-bp, an-bn
    delta = roc_auc_score(y, p)-roc_auc_score(y, q)
    np.testing.assert_allclose(dp.mean(), delta, rtol=0, atol=1e-12)
    se = np.sqrt(dp.var(ddof=1)/len(dp) + dn.var(ddof=1)/len(dn))
    return float(delta), float(se)


def main():
    directory = Path(read(ROOT/'.artifact_runtime/tabicl_finetune_20261019/latest.json')['directory'])
    protocol = read(directory/'protocol.json')
    for name, expected in protocol['protected'].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in protocol['hashes'].items():
        assert sha(ROOT/name) == expected, name
    for name in ['experiment_tabicl_finetune.py', 'tabicl_finetune_support.py']:
        assert sha(directory/name) == protocol['hashes'][name]
    checkpoint = ROOT/'.artifact_runtime/tabicl_probe_20261019/tabicl-classifier-v2-20260212.ckpt'
    assert sha(checkpoint) == protocol['checkpoint_sha256']
    runtime_paths = list((ROOT/'.artifact_runtime/tabicl_runtime/tabicl').rglob('*.py'))
    runtime_hash = hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): sha(p)
        for p in sorted(runtime_paths)}, sort_keys=True).encode()).hexdigest()
    assert runtime_hash == protocol['runtime_source_sha256']
    resources = read(directory/'resource_probe.json')
    assert resources['cache_adapter_verified'] and resources['maximum_prediction_difference'] <= 1e-7
    if resources['chosen_chunk'] is not None:
        selected_probe = next(r for r in resources['checks'] if r['chunk'] == resources['chosen_chunk'])
        assert selected_probe['parameters_updated']
    y = pd.read_csv(ROOT/'train.csv', usecols=['satisfaction']).satisfaction.to_numpy()
    follow = ROOT/'.artifact_runtime/followup_20261019/4261d13e038c1512ffc1'
    verified = []
    screen_decision = read(directory/'screen/decision.json')
    for stage in ['screen', 'confirmation']:
        folder = directory/stage
        if not (folder/'decision.json').exists():
            continue
        with np.load(folder/'indices.npz') as z:
            arrays = {k:v.copy() for k,v in z.items()}
        fit, stop, scored = arrays['fit'], arrays['stopping'], arrays['scored']
        with np.load(follow/stage/'indices.npz') as z:
            for name in ['fit', 'stopping', 'scored']:
                np.testing.assert_array_equal(arrays[name], z[name])
        assert not set(fit)&set(stop) and not set(fit)&set(scored) and not set(stop)&set(scored)
        for size in [32768, 65536]:
            context = arrays[f'context{size}']; positions = arrays[f'positions{size}']
            assert len(context) == len(set(context)) == size
            assert set(context) <= set(fit)
            np.testing.assert_array_equal(fit[positions], context)
        np.testing.assert_array_equal(arrays['context65536'][:32768], arrays['context32768'])
        reference = []
        for name in ['baseline', 'lean_mean']:
            with np.load(follow/stage/(name+'.npz')) as z:
                np.testing.assert_array_equal(scored, z['index'])
                reference.append(z['prediction'].astype(np.float64))
        current = expit(.65*logit(np.clip(reference[0], 1e-6, 1-1e-6)) +
                        .35*logit(np.clip(reference[1], 1e-6, 1-1e-6)))
        with np.load(folder/'current.npz') as z:
            np.testing.assert_array_equal(scored, z['index'])
            np.testing.assert_array_equal(current, z['prediction'])
        for path in sorted(folder.glob('*_*.json')):
            if not path.stem.startswith(('frozen_', 'finetuned_')):
                continue
            row = read(path)
            if row['status'] != 'complete':
                verified.append(dict(stage=stage, name=row['name'], status=row['status']))
                continue
            assert sha(path.with_suffix('.npz')) == row['prediction_sha256']
            with np.load(path.with_suffix('.npz')) as z:
                np.testing.assert_array_equal(scored, z['index'])
                p = z['prediction'].astype(np.float64)
            assert p.shape == current.shape and np.isfinite(p).all()
            assert ((p >= 0)&(p <= 1)).all()
            assert abs(roc_auc_score(y[scored], p)-row['auc']) < 1e-12
            for trial in row['trials']:
                combined = (1-trial['weight'])*current + trial['weight']*p
                delta, se = pair_check(y[scored], combined, current)
                assert abs(delta-trial['paired']['delta']) < 1e-12
                assert abs(se-trial['paired']['approximate_se']) < 1e-12
                assert abs(roc_auc_score(y[scored], combined)-trial['auc']) < 1e-12
            verified.append(dict(stage=stage, name=row['name'], status='verified', auc=row['auc']))
        if (folder/'finetune/summary.json').exists():
            ft = read(folder/'finetune/summary.json')
            if ft['status'] == 'complete':
                assert sha(folder/'finetune/best.ckpt') == ft['checkpoint_sha256']
                trace = ft['validation_trace']
                assert len(trace) == len(ft['epochs'])+1
                assert all(r['context_rows'] == 32768 and r['rows'] == len(stop) for r in trace)
                assert ft['completed_steps'] > 0
                assert abs(max(t['auc'] for t in trace)-ft['best_inner_auc']) <= protocol['min_delta']+1e-12
                import torch
                original_weights = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
                selected_weights = torch.load(folder/'finetune/best.ckpt', map_location='cpu',
                                              weights_only=False)['state_dict']
                h = hashlib.sha256()
                for name, tensor in sorted(selected_weights.items()):
                    h.update(name.encode()); h.update(tensor.contiguous().numpy().tobytes())
                assert h.hexdigest() == ft['selected_weights_sha256']
                changes = sum(not torch.equal(original_weights[n], selected_weights[n]) for n in original_weights)
                assert changes == ft['changed_tensors'] and (changes == 0) == ft['selected_pretrained']
                del original_weights, selected_weights
    cpu_attempt = directory.parent/'93c9003c772d241f9dd5/screen/frozen_32768.npz'
    if cpu_attempt.exists():
        with np.load(cpu_attempt) as old, np.load(directory/'screen/frozen_32768.npz') as new:
            np.testing.assert_array_equal(old['index'], new['index'])
            np.testing.assert_array_equal(old['prediction'], new['prediction'])
    candidates = [t for t in screen_decision['trials'] if
                  t['paired']['delta'] >= protocol['screen_gate'] and t['paired']['interval95'][0] > 0]
    assert screen_decision['chosen'] == (max(candidates, key=lambda t:t['paired']['delta']) if candidates else None)
    if not candidates:
        assert screen_decision['status'] == 'stopped_screen_or_resource'
        assert not (directory/'confirmation/decision.json').exists()
    result = dict(verified=True, cells=verified, protected_unchanged=True,
                  cache_adapter_verified=True, matched_current_recipe_reconstructed=True,
                  large_context_cpu_vs_sequential_adapter_agreement=cpu_attempt.exists(),
                  fit_stop_scored_disjoint=True, context_indices_verified=True,
                  prediction_scores_and_paired_intervals_verified=True,
                  status=screen_decision['status'], no_submission_written=True,
                  verifier_sha256=sha(Path(__file__)))
    (directory/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
