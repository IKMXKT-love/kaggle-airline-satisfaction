"""Rebuild field-model checkpoints, fold boundaries and fixed blend results."""
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from experiment_ffm import ROOT, OUT, read, sha, write_json, restore_predict
from experiment_correction import preserve_check
from correction_support import nested_plan
from ffm_support import FieldEncoder
from report_neural import paired


def main():
    import pandas as pd
    from optimized_features import normalize_raw
    d = Path(read(OUT/'latest.json')['directory']); protocol = read(d/'protocol.json')
    result = read(d/'result.json'); reference = Path(protocol['reference'])
    assert read(reference/'verification.json')['verified']
    for n, h in protocol['hashes'].items():
        assert sha(ROOT/n) == h, n
    frame = pd.read_csv(ROOT/'train.csv'); y = frame.satisfaction.to_numpy()
    x = normalize_raw(frame.drop(columns=['id', 'satisfaction']))
    collected = {k: [] for k in ['index', 'base', 'additive', 'ffm']}; checks = {}
    for outer in protocol['outer_folds']:
        f = d/f'outer_{outer}'; plan = nested_plan(y, outer, protocol['split_seed'])
        r, s, t, e = [plan[k] for k in ['selection_fit', 'selection_stop', 'outer_train', 'outer_eval']]
        with np.load(f/'indices.npz') as z:
            for k, a in dict(fit=r, stopping=s, refit=t, scored=e).items():
                np.testing.assert_array_equal(z[k], a)
        with np.load(reference/f'outer_{outer}/refit/full_fit/combined.npz') as z:
            np.testing.assert_array_equal(z['index'], e); base = z['prediction'].copy()
        collected['index'].append(e); collected['base'].append(base)
        checks[str(outer)] = {}
        for mode in protocol['modes']:
            selection = read(f/('selection_'+mode)/'training.json')
            history = selection['history']
            best = max(history, key=lambda a: a['stopping_auc'])
            assert selection['best_epoch'] == best['epoch']
            refit = read(f/('refit_'+mode)/'training.json')
            assert refit['best_epoch'] == best['epoch'] and refit['fit_rows'] == len(t)
            for phase, allowed in [('selection', r), ('refit', t)]:
                saved = torch.load(f/(phase+'_'+mode)/'best.pt', weights_only=False, map_location='cpu')
                enc = FieldEncoder().fit(x.iloc[allowed]); other = saved['encoder']
                assert enc.vocab == other.vocab and enc.sizes == other.sizes
                np.testing.assert_array_equal(enc.offsets, other.offsets)
                np.testing.assert_allclose(enc.median, other.median, atol=0)
                np.testing.assert_allclose(enc.iqr, other.iqr, atol=0)
                assert saved['epoch'] == best['epoch']
            stop_pred = restore_predict(f/('selection_'+mode)/'best.pt', x.iloc[s])
            np.testing.assert_allclose(roc_auc_score(y[s], stop_pred), best['stopping_auc'], atol=1e-10)
            pred = restore_predict(f/('refit_'+mode)/'best.pt', x.iloc[e])
            with np.load(f/(mode+'.npz')) as z:
                np.testing.assert_array_equal(z['index'], e)
                diff = float(np.max(abs(pred-z['prediction'])))
                np.testing.assert_allclose(pred, z['prediction'], atol=1e-7, rtol=1e-6)
            blend = expit(.8*logit(np.clip(base, 1e-6, 1-1e-6))+.2*logit(np.clip(pred, 1e-6, 1-1e-6)))
            collected[mode].append(blend)
            row = result['folds'][protocol['outer_folds'].index(outer)]['modes'][mode]
            np.testing.assert_allclose(row['blend_auc'], roc_auc_score(y[e], blend), atol=1e-12)
            checks[str(outer)][mode] = dict(checkpoint_max_difference=diff, selected_epoch=best['epoch'])
    collected = {k: np.concatenate(v) for k, v in collected.items()}; idx = collected['index']
    with np.load(d/'blended_predictions.npz') as z:
        for k, a in collected.items():
            np.testing.assert_allclose(z[k], a, atol=1e-12)
    for mode in protocol['modes']:
        pair = paired(y[idx], collected[mode], collected['base'])
        for k in ['delta', 'approximate_se', 'interval95']:
            np.testing.assert_allclose(pair[k], result['modes'][mode]['gain'][k], atol=1e-12)
        gate = (pair['delta'] >= .0003 and pair['interval95'][0] > 0 and
                all(r['modes'][mode]['gain']['delta'] > 0 for r in result['folds']))
        assert gate == result['modes'][mode]['passed']
    preserve_check()
    write_json(d/'verification.json', dict(verified=True, checks=checks,
        encoder_refitted_and_identical=True, checkpoint_predictions_rebuilt=True,
        outer_scope_and_gates_rebuilt=True, protected_files_unchanged=True))
    print('FFM_VERIFIED', checks)


if __name__ == '__main__':
    main()
