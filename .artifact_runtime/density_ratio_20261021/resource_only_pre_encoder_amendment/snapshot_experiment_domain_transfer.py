"""Bounded three-arm source/competition representation screen.

No downloads, test-set inference, submissions, public-pool refits, or CURRENT edits.
The new branch has an isolated outer fold; public-pool comparisons remain conditional.
"""
import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from domain_transfer_support import ARMS, DomainMLP, RawEncoder, domain_loss, joint_batch, predict, to_device
from multitask_support import cpu_state, paired

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
RUN = ROOT / '.artifact_runtime/domain_transfer_20261021'
BASE = ROOT / '.artifact_runtime/public_pool_trial_20261020'
FOLDS = ROOT / 'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy'
MODEL = dict(members=8, embedding_dim=8, frequencies=8, frequency_sigma=2.33,
             hidden=[512, 256, 128], dropout=.05)
TRAIN = dict(seed=20261021, competition_batch=2048, original_batch=512, source_weight=.25,
             smoothing=.005, learning_rate=.0015, min_lr_fraction=.08,
             weight_decay=.01, gradient_clip=1., amp='bfloat16', prediction_batch=4096)
PROTECTED = ['submission_public_pool_20261020.csv', 'CURRENT_BEST_RESULT.md',
             'CURRENT_BEST_SUBMISSION.json', 'CURRENT_RECOMMENDED_CANDIDATE.json']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name+'.part')
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    part.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''): result.update(block)
    return result.hexdigest()


def setup():
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Verified CUDA + bfloat16 device required')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def guard(reserve=45):
    if time.time() >= read(RUN/'clock.json')['deadline_epoch']-reserve:
        raise TimeoutError('Cumulative stage budget reached; preserve completed artifacts')


def check_protected(expected):
    assert {f: sha(ROOT/f) for f in expected} == expected


def load_values(phase, kind, device='cuda'):
    return {k: torch.as_tensor(np.load(RUN/phase/f'{kind}_{k}.npy', allow_pickle=False), device=device)
            for k in ['num', 'missing', 'cat']}


def save_values(phase, kind, encoder, frame):
    result = encoder.transform(frame)
    for k in ['num', 'missing', 'cat']:
        np.save(RUN/phase/f'{kind}_{k}.npy', result[k], allow_pickle=False)


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare():
    if RUN.exists():
        raise FileExistsError('Existing experiment is preserved; do not reset its clock')
    RUN.mkdir(parents=True)
    started = time.time()
    write(RUN/'clock.json', dict(started_epoch=started, deadline_epoch=started+900, seconds=900,
                                includes='data preparation, discarded resource probe, selection fits, full outer refits, evaluation, verification'))
    setup()
    protected = {f: sha(ROOT/f) for f in PROTECTED}
    source_files = ['experiment_domain_transfer.py', 'domain_transfer_support.py', 'multitask_support.py',
                    'tests/test_domain_transfer_contract.py', 'train.csv', 'test.csv',
                    'external_data/train.csv', 'external_data/test.csv']
    hashes = {f: sha(ROOT/f) for f in source_files}
    hashes[str(FOLDS.relative_to(ROOT))] = sha(FOLDS)
    hashes[str((BASE/'public_only_fold0_validation.npz').relative_to(ROOT))] = sha(BASE/'public_only_fold0_validation.npz')
    train = pd.read_csv(ROOT/'train.csv')
    test = pd.read_csv(ROOT/'test.csv')
    features = [c for c in test if c != 'id']
    x, y, ids = train[features], train.satisfaction.to_numpy(dtype='int64'), train.id.to_numpy()
    original = pd.concat([pd.read_csv(ROOT/'external_data'/f) for f in ['train.csv', 'test.csv']], ignore_index=True)
    source_fp = pd.util.hash_pandas_object(original[features].astype('string').fillna('__MISSING__'), index=False)
    comp_fp = pd.util.hash_pandas_object(pd.concat([x, test[features]], ignore_index=True).astype('string').fillna('__MISSING__'), index=False)
    keep = ~source_fp.isin(set(comp_fp))
    ox = original.loc[keep, features].reset_index(drop=True)
    oy = original.loc[keep, 'satisfaction'].eq('satisfied').to_numpy(dtype='int64')
    fold = np.load(FOLDS, allow_pickle=False)
    expected = np.empty(len(y), dtype='int8')
    for f, (_, e) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y)):
        expected[e] = f
    np.testing.assert_array_equal(fold, expected)
    outer_eval, outer_train = np.flatnonzero(fold == 0), np.flatnonzero(fold != 0)
    fit, stopping = train_test_split(outer_train, test_size=.08, random_state=TRAIN['seed'], stratify=y[outer_train])
    fit, stopping = np.sort(fit), np.sort(stopping)
    assert np.array_equal(np.sort(np.r_[fit, stopping, outer_eval]), np.arange(len(y)))
    np.savez(RUN/'indices.npz', fit=fit, stopping=stopping, outer_train=outer_train, scored=outer_eval,
             original_rows=np.flatnonzero(keep.to_numpy()))
    np.savez(RUN/'labels.npz', y=y, original_y=oy, id=ids)
    with np.load(BASE/'public_only_fold0_validation.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['index'], outer_eval)
        np.save(RUN/'baseline_logit.npy', z['logit'], allow_pickle=False)
    for phase, ci in [('selection', fit), ('refit', outer_train)]:
        guard()
        (RUN/phase).mkdir()
        encoder = RawEncoder().fit(x.iloc[ci])
        write(RUN/phase/'encoder.json', encoder.state())
        save_values(phase, 'competition', encoder, x.iloc[ci])
        save_values(phase, 'original', encoder, ox)
        save_values(phase, 'scored', encoder, x.iloc[outer_eval])
        if phase == 'selection': save_values(phase, 'stopping', encoder, x.iloc[stopping])
    probe_result = probe(fit, outer_train, oy)
    elapsed = time.time()-started
    epoch_estimates = {}
    for epochs in [32, 24, 16]:
        # Worst observed arm, plus epoch scoring, source sampling, CPU AUC/checkpoints,
        # and final scored prediction. All three selection and all three refits included.
        steps = len(ARMS)*epochs*(math.ceil(len(fit)/TRAIN['competition_batch'])+
                                 math.ceil(len(outer_train)/TRAIN['competition_batch']))
        training_estimate = steps*probe_result['worst_seconds_per_step']
        inference_rows = len(ARMS)*(epochs*len(stopping)+len(outer_eval))
        inference_estimate = inference_rows*probe_result['prediction_seconds_per_row']
        epoch_estimates[str(epochs)] = dict(training_seconds=training_estimate,
            inference_seconds=inference_estimate,
            conservative_total_seconds=elapsed+1.5*(training_estimate+inference_estimate)+90)
    permitted = [e for e in [32, 24, 16] if epoch_estimates[str(e)]['conservative_total_seconds'] <= 900]
    if not permitted:
        write(RUN/'resource_stop.json', dict(elapsed=elapsed, epoch_estimates=epoch_estimates))
        raise TimeoutError('No complete meaningful three-arm screen fits the cumulative budget')
    max_epochs = permitted[0]
    protocol = dict(model=MODEL, training=TRAIN, max_epochs=max_epochs, arms=ARMS,
        phase='single outer-fold conditional screen', outer_fold=0, native_public_fold_seed=42,
        fit_rows=len(fit), stopping_rows=len(stopping), refit_rows=len(outer_train), scored_rows=len(outer_eval),
        original_rows=len(ox), removed_original_overlap=int((~keep).sum()),
        original_extra_field_used=False, source_pretraining=False, joint_training=True,
        encoder_fit_scope='selection competition fit / refit complete outer train; no source or scored fit statistics',
        labels_used='C-only original loss weight zero; other arms mean C BCE + .25 mean O BCE',
        compute_control='All arms execute both-domain forward/loss; equal competition batches/epochs and source batch RNG',
        stopping='selection inner AUC only, then reinitialize and fit complete outer train for selected epoch count',
        refit_lr='prefix of the same resource-frozen max_epochs cosine schedule, not compressed to chosen epochs',
        initializer='same full state across all arms per phase; source head starts as copy of competition head',
        primary_blend_weight=.1, diagnostic_blend_weights=[.05, .2],
        primary_candidate='dual_head', expansion_gate=.0003, require_paired_lower_above_zero=True,
        require_primary_dual_blend_better_than_matched_competition_only=True,
        no_weight_search=True, no_submission_during_screen=True,
        public_validation_caveat='Fixed public OOF baseline inherits upstream early-stopping and indirect fold label dependence; not end-to-end nested validation.',
        evidence='Mechanism and source/competition structure support a bounded test; remaining ranking gain is not established.',
        historical_partitions_and_recipes_seen=True,
        costs=dict(preparation_and_probe_seconds=elapsed, resource_projection=epoch_estimates,
                   cumulative_hard_deadline_epoch=started+900, verification_reserve_seconds=45),
        protected=protected, input_sha256=hashes, probe_sha256=sha(RUN/'probe.json'))
    write(RUN/'protocol.json', protocol)
    for f in ['experiment_domain_transfer.py', 'domain_transfer_support.py', 'multitask_support.py']:
        (RUN/('snapshot_'+f)).write_bytes((ROOT/f).read_bytes())
    check_protected(protected)
    print('PROTOCOL', json.dumps({k:protocol[k] for k in ['max_epochs', 'fit_rows', 'stopping_rows', 'refit_rows', 'scored_rows', 'original_rows', 'costs']}), flush=True)


def probe(fit, outer_train, oy):
    guard()
    state = read(RUN/'selection/encoder.json')
    c = load_values('selection', 'competition')
    o = load_values('selection', 'original')
    with np.load(RUN/'labels.npz', allow_pickle=False) as z: y = z['y']
    yc, yo = torch.tensor(y[fit], device='cuda'), torch.tensor(oy, device='cuda')
    ci = torch.arange(TRAIN['competition_batch'], device='cuda')
    oi = torch.arange(TRAIN['original_batch'], device='cuda')
    inputs = joint_batch(c, o, ci, oi)
    measurements = {}
    torch.cuda.reset_peak_memory_stats()
    for arm in ARMS:
        torch.manual_seed(TRAIN['seed'])
        model = DomainMLP(state, MODEL, y[fit].mean()).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN['learning_rate'], weight_decay=TRAIN['weight_decay'], fused=True)
        first = model.layers[0].weight.detach().clone()
        for step in range(42):
            if step == 10:
                torch.cuda.synchronize(); start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _, _ = domain_loss(model(inputs), yc[ci], yo[oi], arm, TRAIN['source_weight'], TRAIN['smoothing'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN['gradient_clip'], foreach=True, error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        measurements[arm] = (time.perf_counter()-start)/32
        assert (model.layers[0].weight.detach()-first).abs().max() > 0
        del optimizer, first
    small = {k:v[:8192] for k,v in c.items()}
    predict(model, small, TRAIN['prediction_batch'])
    torch.cuda.synchronize(); start = time.perf_counter()
    for _ in range(4): predict(model, small, TRAIN['prediction_batch'])
    torch.cuda.synchronize()
    prediction_rate = (time.perf_counter()-start)/(4*8192)
    result = dict(step_seconds_by_arm=measurements, worst_seconds_per_step=max(measurements.values()),
                  prediction_seconds_per_row=prediction_rate, peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                  device=torch.cuda.get_device_name(0), probe_parameters_discarded=True,
                  uses_only_fit_and_original_rows=True)
    write(RUN/'probe.json', result)
    print('RESOURCE_PROBE', json.dumps(result), flush=True)
    del c, o, model, inputs, yc, yo
    gc.collect(); torch.cuda.empty_cache()
    return result


def train_arm(phase, arm, protocol, labels, indices):
    guard()
    path = RUN/phase/arm
    if path.exists(): raise FileExistsError(f'Preserve existing fit: {path}')
    path.mkdir()
    started = time.time()
    state = read(RUN/phase/'encoder.json')
    ci = indices['fit'] if phase == 'selection' else indices['outer_train']
    c, o = load_values(phase, 'competition'), load_values(phase, 'original')
    stopping = load_values('selection', 'stopping') if phase == 'selection' else None
    yc = torch.as_tensor(labels['y'][ci], device='cuda', dtype=torch.float32)
    yo = torch.as_tensor(labels['original_y'], device='cuda', dtype=torch.float32)
    torch.manual_seed(TRAIN['seed'])
    model = DomainMLP(state, MODEL, float(yc.mean())).cuda()
    initial = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN['learning_rate'], weight_decay=TRAIN['weight_decay'], fused=True)
    cgen = torch.Generator(device='cuda').manual_seed(TRAIN['seed']+100)
    ogen = torch.Generator(device='cuda').manual_seed(TRAIN['seed']+200)
    max_epochs = protocol['max_epochs']
    epochs = max_epochs if phase == 'selection' else read(RUN/'selection'/arm/'complete.json')['best_epoch']
    history, best, best_epoch = [], -1., None
    updates = 0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for epoch in range(1, epochs+1):
        guard()
        lr = TRAIN['learning_rate']*(TRAIN['min_lr_fraction']+(1-TRAIN['min_lr_fraction'])*.5*(1+math.cos(math.pi*(epoch-1)/max(1,max_epochs-1))))
        for group in optimizer.param_groups: group['lr'] = lr
        order = torch.randperm(len(yc), device='cuda', generator=cgen)
        loss_sum, source_sum, steps = 0., 0., 0
        model.train()
        epoch_start = time.time()
        for start in range(0, len(order), TRAIN['competition_batch']):
            if steps % 32 == 0: guard()
            ic = order[start:start+TRAIN['competition_batch']]
            io = torch.randint(len(yo), (TRAIN['original_batch'],), device='cuda', generator=ogen)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, lc, lo = domain_loss(model(joint_batch(c, o, ic, io)), yc[ic], yo[io], arm,
                                           TRAIN['source_weight'], TRAIN['smoothing'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN['gradient_clip'], foreach=True, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(lc); source_sum += float(lo); steps += 1; updates += 1
        row = dict(epoch=epoch, lr=lr, competition_bce=loss_sum/steps,
                   original_bce=source_sum/steps, epoch_seconds=time.time()-epoch_start)
        if phase == 'selection':
            probability = predict(model, stopping, TRAIN['prediction_batch'])
            auc = float(roc_auc_score(labels['y'][indices['stopping']], probability))
            row['stopping_auc'] = auc
            if auc > best:
                best, best_epoch = auc, epoch
                torch.save(cpu_state(model), path/'best_weights.pt')
        history.append(row)
        write(path/'history.json', history)
        if epoch == 1 or epoch % 4 == 0 or epoch == epochs:
            print('EPOCH', phase, arm, json.dumps(row), flush=True)
    if phase == 'refit':
        torch.save(cpu_state(model), path/'weights.pt')
        evaluation = load_values('refit', 'scored')
        probability = predict(model, evaluation, TRAIN['prediction_batch'])
        np.savez(path/'scored_prediction.npz', index=indices['scored'], id=labels['id'][indices['scored']], probability=probability)
        del evaluation
    result = dict(arm=arm, phase=phase, initial_state_hash=initial, epochs_completed=epochs,
                  updates=updates, seconds=time.time()-started,
                  best_epoch=best_epoch if phase == 'selection' else epochs,
                  best_stopping_auc=best if phase == 'selection' else None,
                  peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                  parameter_count=sum(p.numel() for p in model.parameters()))
    write(path/'complete.json', result)
    print('FIT_COMPLETE', json.dumps(result), flush=True)
    del model, optimizer, c, o, stopping, yc, yo
    gc.collect(); torch.cuda.empty_cache()


def evaluate(protocol, labels, indices):
    guard(reserve=20)
    y = labels['y'][indices['scored']]
    base = np.load(RUN/'baseline_logit.npy', allow_pickle=False)
    base_auc = float(roc_auc_score(y, base))
    probabilities, output = {}, {}
    for arm in ARMS:
        with np.load(RUN/'refit'/arm/'scored_prediction.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['index'], indices['scored'])
            np.testing.assert_array_equal(z['id'], labels['id'][indices['scored']])
            p = z['probability']
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        probabilities[arm] = p
        blend = expit(.9*base+.1*logit(np.clip(p,1e-6,1-1e-6)))
        output[arm] = dict(single_auc=float(roc_auc_score(y,p)), primary_blend_auc=float(roc_auc_score(y,blend)),
                           primary_blend_delta=paired(y,blend,base), diagnostic_weights={})
        for weight in protocol['diagnostic_blend_weights']:
            score = (1-weight)*base+weight*logit(np.clip(p,1e-6,1-1e-6))
            output[arm]['diagnostic_weights'][str(weight)] = float(roc_auc_score(y,score))-base_auc
    dual, control, shared = [probabilities[k] for k in ['dual_head','competition_only','shared_head']]
    dual_blend = .9*base+.1*logit(np.clip(dual,1e-6,1-1e-6))
    control_blend = .9*base+.1*logit(np.clip(control,1e-6,1-1e-6))
    comparison = paired(y,dual_blend,control_blend)
    delta = output['dual_head']['primary_blend_delta']
    accepted = delta['delta'] >= protocol['expansion_gate'] and delta['interval95'][0] > 0 and comparison['delta'] > 0
    result = dict(completed=True, baseline_auc=base_auc, arms=output,
        dual_vs_competition_single=paired(y,dual,control), dual_vs_shared_single=paired(y,dual,shared),
        dual_vs_competition_primary_blend=comparison, passes_expansion_gate=accepted,
        gate_uses_only_predeclared_dual_10percent_blend=True, scored_rows=len(y), new_public_score=None,
        elapsed_from_preparation_seconds=time.time()-read(RUN/'clock.json')['started_epoch'],
        uncertainty='Paired fixed-prediction sample uncertainty only; no training randomness, multiple-selection, or public OOF dependence adjustment.',
        validation_caveat=protocol['public_validation_caveat'])
    write(RUN/'evaluation.json', result)
    print('EVALUATION', json.dumps(result), flush=True)
    check_protected(protocol['protected'])
    return result


def run():
    setup()
    protocol = read(RUN/'protocol.json')
    check_protected(protocol['protected'])
    for f in ['experiment_domain_transfer.py', 'domain_transfer_support.py', 'multitask_support.py']:
        assert sha(ROOT/f) == protocol['input_sha256'][f], f
    with np.load(RUN/'labels.npz', allow_pickle=False) as z: labels = {k:z[k].copy() for k in z.files}
    with np.load(RUN/'indices.npz', allow_pickle=False) as z: indices = {k:z[k].copy() for k in z.files}
    for phase in ['selection','refit']:
        for arm in ARMS: train_arm(phase,arm,protocol,labels,indices)
    evaluate(protocol, labels, indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','run'])
    args = parser.parse_args()
    try:
        prepare() if args.action == 'prepare' else run()
    except BaseException as exc:
        if RUN.exists(): write(RUN/'stopped.json',dict(error=type(exc).__name__, detail=str(exc), time_epoch=time.time()))
        raise
