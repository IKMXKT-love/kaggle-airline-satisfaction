"""Rebuild preprocessing, prediction, split and AUC evidence independently."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from domain_transfer_support import ARMS, DomainMLP, RawEncoder, predict, to_device

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
RUN = ROOT/'.artifact_runtime/domain_transfer_20261021'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    started = time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    protocol, evaluation = read(RUN/'protocol.json'), read(RUN/'evaluation.json')
    for name, expected in protocol['input_sha256'].items(): assert sha(ROOT/name) == expected, name
    for name, expected in protocol['protected'].items(): assert sha(ROOT/name) == expected, name
    train = pd.read_csv(ROOT/'train.csv')
    y, ids = train.satisfaction.to_numpy(dtype='int64'), train.id.to_numpy()
    x = train.drop(columns=['id','satisfaction'])
    with np.load(RUN/'indices.npz', allow_pickle=False) as z: idx = {k:z[k].copy() for k in z.files}
    with np.load(RUN/'labels.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['y'], y)
        np.testing.assert_array_equal(z['id'], ids)
    assert np.array_equal(np.sort(np.r_[idx['fit'], idx['stopping'], idx['scored']]), np.arange(len(y)))
    assert np.array_equal(np.sort(np.r_[idx['fit'],idx['stopping']]),idx['outer_train'])
    folded = np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy', allow_pickle=False)
    np.testing.assert_array_equal(idx['scored'], np.flatnonzero(folded==0))
    for phase, training in [('selection', idx['fit']), ('refit',idx['outer_train'])]:
        assert RawEncoder().fit(x.iloc[training]).state() == read(RUN/phase/'encoder.json')
        initials = [read(RUN/phase/arm/'complete.json')['initial_state_hash'] for arm in ARMS]
        assert len(set(initials)) == 1, (phase, initials)
    original = pd.concat([pd.read_csv(ROOT/'external_data'/n) for n in ['train.csv','test.csv']], ignore_index=True)
    selected_original = original.iloc[idx['original_rows']]
    assert len(selected_original) == protocol['original_rows']
    assert len(original)-len(selected_original) == protocol['removed_original_overlap']
    with np.load(RUN/'labels.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['original_y'], selected_original.satisfaction.eq('satisfied').to_numpy(dtype='int64'))
    for phase, fit_rows in [('selection',idx['fit']),('refit',idx['outer_train'])]:
        encoder = RawEncoder().fit(x.iloc[fit_rows])
        original_encoding = encoder.transform(selected_original[list(x)].reset_index(drop=True))
        for key in ['cat','num','missing']:
            np.testing.assert_array_equal(original_encoding[key],np.load(RUN/phase/f'original_{key}.npy',allow_pickle=False))
    del original_encoding, original, selected_original
    base = np.load(RUN/'baseline_logit.npy',allow_pickle=False)
    base_dir = ROOT/'.artifact_runtime/public_pool_trial_20261020'
    with np.load(base_dir/'public_only_fold0.npz',allow_pickle=False) as z: weights = z['weights']
    assert len(weights) == 247
    matrix = np.load(base_dir/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    reconstructed_base = np.empty(len(idx['scored']))
    with threadpool_limits(limits=4):
        for start in range(0,len(idx['scored']),8192):
            rows = idx['scored'][start:start+8192]
            reconstructed_base[start:start+len(rows)] = matrix[rows,:246]@weights[:-1]+weights[-1]
    base_error = float(np.max(np.abs(base-reconstructed_base)))
    assert base_error < 1e-10, base_error
    score_y = y[idx['scored']]
    assert abs(roc_auc_score(score_y,base)-evaluation['baseline_auc']) < 1e-12
    encoder = RawEncoder().fit(x.iloc[idx['outer_train']])
    rebuilt_values = encoder.transform(x.iloc[idx['scored']])
    for k in ['num','missing','cat']:
        np.testing.assert_array_equal(rebuilt_values[k],np.load(RUN/'refit'/f'scored_{k}.npy',allow_pickle=False))
    values = to_device(rebuilt_values,'cuda')
    evidence = {}
    for arm in ARMS:
        selection = read(RUN/'selection'/arm/'complete.json')
        refit = read(RUN/'refit'/arm/'complete.json')
        history = read(RUN/'selection'/arm/'history.json')
        best = max(history,key=lambda v:v['stopping_auc'])
        assert selection['best_epoch'] == best['epoch'] == refit['epochs_completed']
        assert selection['epochs_completed'] == protocol['max_epochs']
        assert refit['updates'] == int(np.ceil(len(idx['outer_train'])/protocol['training']['competition_batch']))*refit['epochs_completed']
        torch.manual_seed(protocol['training']['seed'])
        model = DomainMLP(encoder.state(),protocol['model'],float(y[idx['outer_train']].mean())).cuda()
        model.load_state_dict(torch.load(RUN/'refit'/arm/'weights.pt',map_location='cuda',weights_only=True),strict=True)
        actual = predict(model,values,protocol['training']['prediction_batch'])
        with np.load(RUN/'refit'/arm/'scored_prediction.npz',allow_pickle=False) as z:
            expected = z['probability']; np.testing.assert_array_equal(z['id'],ids[idx['scored']])
        err = float(np.max(np.abs(actual-expected)))
        assert err < 1e-7, (arm,err)
        actual_auc = float(roc_auc_score(score_y,actual))
        blend_auc = float(roc_auc_score(score_y,.9*base+.1*logit(np.clip(actual,1e-6,1-1e-6))))
        assert abs(actual_auc-evaluation['arms'][arm]['single_auc']) < 1e-10
        assert abs(blend_auc-evaluation['arms'][arm]['primary_blend_auc']) < 1e-10
        evidence[arm] = dict(prediction_rebuild_max_error=err, single_auc=actual_auc,
                             primary_blend_auc=blend_auc, selected_epoch=selection['best_epoch'],
                             refit_weight_sha256=sha(RUN/'refit'/arm/'weights.pt'))
        del model
    delta = evaluation['arms']['dual_head']['primary_blend_delta']
    gate = delta['delta']>=protocol['expansion_gate'] and delta['interval95'][0]>0 and evaluation['dual_vs_competition_primary_blend']['delta']>0
    assert gate == evaluation['passes_expansion_gate']
    for name, expected in protocol['protected'].items(): assert sha(ROOT/name) == expected, name
    result = dict(verified=True, initial_states_match_across_arms=True,
        encoder_scope_rebuilt=True, original_labels_and_encodings_rebuilt=True,
        row_partitions_and_ids_verified=True, baseline_rebuild_max_error=base_error,
        prediction_evidence=evidence, protected_files_unchanged=True,
        verification_seconds=time.time()-started,
        cumulative_seconds=time.time()-read(RUN/'clock.json')['started_epoch'],
        cumulative_deadline_met=time.time() <= read(RUN/'clock.json')['deadline_epoch'],
        gate=gate, public_validation_caveat=protocol['public_validation_caveat'])
    (RUN/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == '__main__': main()
