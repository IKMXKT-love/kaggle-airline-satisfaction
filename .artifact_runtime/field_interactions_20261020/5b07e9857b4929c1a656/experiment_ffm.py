"""Independent additive/field-aware comparison on the correction outer folds."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from ffm_support import FieldEncoder, FieldModel, tensor_data, predict
from experiment_correction import ROOT, OUT as CORRECTION_OUT, read, sha, write_json, preserve_check
from correction_support import nested_plan
from report_neural import paired

OUT = ROOT/'.artifact_runtime/field_interactions_20261020'
CONFIG = dict(outer_folds=[0, 1], split_seed=20261020, max_epochs=24,
    patience=4, batch_size=2048, members=4, rank=8, lr=.003, weight_decay=.02,
    label_smoothing=.005, seed=42, modes=['additive', 'ffm'], fixed_blend_weight=.20,
    blend_space='logit', gate=.0003, early_stopping_only_on_internal_split=True,
    refit_full_outer_training=True, historical_data_previously_seen=True)


def setup():
    OUT.mkdir(exist_ok=True)
    reference = Path(read(CORRECTION_OUT/'latest.json')['directory'])
    hashes = {n: sha(ROOT/n) for n in ['experiment_ffm.py', 'ffm_support.py',
        'tests/test_ffm_contract.py', 'correction_support.py', 'train.csv', 'test.csv']}
    signature = hashlib.sha256(json.dumps([CONFIG, hashes, str(reference)], sort_keys=True).encode()).hexdigest()[:20]
    d = OUT/signature; d.mkdir(exist_ok=True)
    if not (d/'protocol.json').exists():
        assert not (reference/'result.json').exists(), 'New FFM recipe must be frozen before correction outer scoring'
        write_json(d/'protocol.json', dict(**CONFIG, hashes=hashes, reference=str(reference),
            frozen_before_correction_outer_scores=True))
        for n in ['experiment_ffm.py', 'ffm_support.py', 'tests/test_ffm_contract.py']:
            (d/n.replace('/', '__')).write_bytes((ROOT/n).read_bytes())
    else:
        assert read(d/'protocol.json')['hashes'] == hashes
    write_json(OUT/'latest.json', dict(directory=str(d)))
    preserve_check()
    return d, reference


def train_one(d, phase, mode, x, y, fit, stop=None, epochs=None, seed=42):
    out = d/(phase+'_'+mode); out.mkdir(exist_ok=True)
    if (out/'training.json').exists():
        return read(out/'training.json')
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    enc = FieldEncoder().fit(x.iloc[fit])
    a = tensor_data(enc.transform(x.iloc[fit]), 'cuda')
    b = tensor_data(enc.transform(x.iloc[stop]), 'cuda') if stop is not None else None
    labels = torch.as_tensor(np.asarray(y[fit], dtype='float32'), device='cuda')
    labels = labels*(1-CONFIG['label_smoothing'])+.5*CONFIG['label_smoothing']
    params = dict(total_tokens=sum(enc.sizes), fields=len(enc.columns), numeric_dim=a['numeric'].shape[1],
                  interactions=mode == 'ffm', members=CONFIG['members'], rank=CONFIG['rank'])
    model = FieldModel(**params).to('cuda')
    optim = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    generator = torch.Generator(device='cuda').manual_seed(seed+1000)
    history = []; best = -np.inf; best_epoch = 0; start = time.time(); steps = 0
    torch.cuda.reset_peak_memory_stats()
    n_epochs = CONFIG['max_epochs'] if epochs is None else epochs
    for epoch in range(1, n_epochs+1):
        model.train(); order = torch.randperm(len(fit), generator=generator, device='cuda')
        loss_sum = 0.
        for start_row in range(0, len(fit), CONFIG['batch_size']):
            idx = order[start_row:start_row+CONFIG['batch_size']]
            progress = (epoch-1+start_row/len(fit))/CONFIG['max_epochs']
            lr = CONFIG['lr']*(.05+.95*.5*(1+math.cos(math.pi*progress)))
            for group in optim.param_groups:
                group['lr'] = lr
            optim.zero_grad(set_to_none=True)
            logits = model(a['tokens'][idx], a['numeric'][idx])
            target = labels[idx][None, :].expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step(); steps += 1
            loss_sum += float(loss.detach())*len(idx)
        item = dict(epoch=epoch, loss=loss_sum/len(fit), steps=steps)
        if b is not None:
            p = predict(model, b)
            auc = float(roc_auc_score(y[stop], p)); item['stopping_auc'] = auc
            if auc > best:
                best, best_epoch = auc, epoch
                torch.save(dict(weights={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    params=params, encoder=enc, epoch=epoch), out/'best.pt')
            print('FFM_STOP', phase, mode, epoch, auc, flush=True)
        else:
            best_epoch = epoch
        history.append(item)
        if b is not None and epoch-best_epoch >= CONFIG['patience']:
            break
    if b is None:
        torch.save(dict(weights={k: v.detach().cpu() for k, v in model.state_dict().items()},
            params=params, encoder=enc, epoch=best_epoch), out/'best.pt')
    result = dict(mode=mode, phase=phase, fit_rows=len(fit), stop_rows=0 if stop is None else len(stop),
        best_epoch=best_epoch, history=history, steps=steps, seconds=time.time()-start,
        peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,
        parameters=sum(p.numel() for p in model.parameters()))
    write_json(out/'training.json', result)
    del model, optim, a, b, labels
    import gc
    gc.collect(); torch.cuda.empty_cache()
    return result


def restore_predict(path, frame):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = FieldModel(**saved['params']).to('cuda')
    model.load_state_dict(saved['weights'])
    values = tensor_data(saved['encoder'].transform(frame), 'cuda')
    p = predict(model, values).astype('float64')
    del model, values, saved
    torch.cuda.empty_cache()
    return p


def run():
    d, reference = setup()
    train = pd.read_csv(ROOT/'train.csv')
    # Same numeric normalization as the primary experiment, without needing
    # any source teacher or target encoding for this independent branch.
    from optimized_features import normalize_raw
    x = normalize_raw(train.drop(columns=['id', 'satisfaction']))
    y = train.satisfaction.to_numpy()
    for outer in CONFIG['outer_folds']:
        f = d/f'outer_{outer}'; f.mkdir(exist_ok=True)
        p = nested_plan(y, outer, CONFIG['split_seed'])
        r, s, t, e = [p[k] for k in ['selection_fit', 'selection_stop', 'outer_train', 'outer_eval']]
        np.savez_compressed(f/'indices.npz', fit=r, stopping=s, refit=t, scored=e)
        for mode in CONFIG['modes']:
            selected = train_one(f, 'selection', mode, x, y, r, s, seed=42+outer)
            train_one(f, 'refit', mode, x, y, t, epochs=selected['best_epoch'], seed=42+outer)
            path = f/(mode+'.npz')
            if not path.exists():
                pred = restore_predict(f/('refit_'+mode)/'best.pt', x.iloc[e])
                np.savez_compressed(path, index=e, prediction=pred)
    # Score only after both outer folds have finished and the primary base
    # predictions exist. There is no outer-dependent feature or epoch search.
    assert (reference/'result.json').exists(), 'Finish primary experiment before combined scoring'
    rows = []; collected = {k: [] for k in ['index', 'base']+CONFIG['modes']}
    for outer in CONFIG['outer_folds']:
        f = d/f'outer_{outer}'
        with np.load(reference/f'outer_{outer}/refit/full_fit/combined.npz') as z:
            idx, base = z['index'].copy(), z['prediction'].copy()
        collected['index'].append(idx); collected['base'].append(base)
        record = dict(outer=outer, baseline_auc=float(roc_auc_score(y[idx], base)), modes={})
        for mode in CONFIG['modes']:
            with np.load(f/(mode+'.npz')) as z:
                np.testing.assert_array_equal(z['index'], idx); pred = z['prediction'].copy()
            blended = expit(.8*logit(np.clip(base, 1e-6, 1-1e-6))+.2*logit(np.clip(pred, 1e-6, 1-1e-6)))
            collected[mode].append(blended)
            record['modes'][mode] = dict(single_auc=float(roc_auc_score(y[idx], pred)),
                blend_auc=float(roc_auc_score(y[idx], blended)), gain=paired(y[idx], blended, base))
        rows.append(record)
    collected = {k: np.concatenate(v) for k, v in collected.items()}
    idx = collected['index']; result = dict(folds=rows, modes={})
    for mode in CONFIG['modes']:
        difference = paired(y[idx], collected[mode], collected['base'])
        gate = (difference['delta'] >= CONFIG['gate'] and difference['interval95'][0] > 0
                and all(r['modes'][mode]['gain']['delta'] > 0 for r in rows))
        result['modes'][mode] = dict(gain=difference, passed=gate)
    result['ffm_minus_additive'] = paired(y[idx], collected['ffm'], collected['additive'])
    np.savez_compressed(d/'blended_predictions.npz', **collected)
    write_json(d/'result.json', result)
    preserve_check()
    print('FFM_RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'all'], nargs='?', default='all')
    if parser.parse_args().action == 'freeze':
        print(setup()[0])
    else:
        run()
