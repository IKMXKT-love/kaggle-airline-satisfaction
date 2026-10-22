"""Bounded A/B/C rating-denoising experiment with separate stopping data."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from multitask_support import (ROOT, RATINGS, SPECS, RawEncoder, RatingMLP, batch,
    classification_loss, combine, cpu_state, paired, predict, reconstruction_metrics,
    recovery_loss, sha, to_device, write_json, write_torch)

FAMILY = ROOT / '.artifact_runtime/denoising_multitask_20261019'
REFERENCE = ROOT / '.artifact_runtime/followup_20261019/4261d13e038c1512ffc1'
MODEL = dict(members=8, embedding_dim=8, frequencies=8, frequency_sigma=2.33,
             hidden=[512, 256, 128], dropout=.05)
BASE_TRAINING = dict(seed=42, batch_size=1024, prediction_batch_size=2048,
    mask_probability=.15, auxiliary_weight=.1, smoothing=.01, weight_decay=.01,
    warm_lr=.001, fine_lr=.0003, minimum_lr_fraction=.1, gradient_clip=1.,
    amp_bfloat16=True, optimizer='AdamW', fused=True)
USAGES = ['replace35', 'add20']


def setup():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This bounded experiment requires the verified CUDA device')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.is_bf16_supported()


def data():
    train = pd.read_csv(ROOT / 'train.csv')
    y = train.satisfaction.to_numpy(dtype='int64')
    assert set(np.unique(y)) == {0, 1}
    return train.drop(columns=['id', 'satisfaction']), y, train.id.to_numpy()


def split(stage):
    with np.load(REFERENCE / stage / 'indices.npz') as saved:
        result = {key: saved[key].copy() for key in ['fit', 'stopping', 'scored']}
    merged = np.concatenate(list(result.values()))
    assert len(np.unique(merged)) == len(merged)
    assert merged.min() >= 0 and merged.max() < 699635
    return result


def check_preserved():
    receipt = json.loads((FAMILY / 'initial_preserved.json').read_text(encoding='utf-8'))
    for name, expected in receipt.items():
        assert sha(ROOT / name) == expected, name
    return receipt


def source_hashes():
    names = ['experiment_multitask.py', 'multitask_support.py', 'tests/test_multitask_contract.py']
    return {n: sha(ROOT / n) for n in names}


def probe():
    setup()
    check_preserved()
    x, y, _ = data()
    index = split('screen')['fit']
    encoder = RawEncoder().fit(x.iloc[index])
    chosen = index[:8192]
    values = to_device(encoder.transform(x.iloc[chosen]), 'cuda')
    target = torch.as_tensor(y[chosen], device='cuda', dtype=torch.float32)
    torch.manual_seed(BASE_TRAINING['seed'])
    model = RatingMLP(encoder.state(), MODEL, y[index].mean()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_TRAINING['warm_lr'],
                                weight_decay=BASE_TRAINING['weight_decay'], fused=True)
    before = model.layers[0].weight.detach().clone()
    recovery_before = model.recovery.weight.detach().clone()
    generator = torch.Generator(device='cuda').manual_seed(1938)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    warmup, timed = 10, 40
    started = None
    for step in range(warmup + timed):
        if step == warmup:
            torch.cuda.synchronize()
            started = time.perf_counter()
        idx = torch.arange((step % 8) * 1024, (step % 8 + 1) * 1024, device='cuda')
        mask = torch.rand((len(idx), len(RATINGS)), device='cuda', generator=generator) < .15
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, decoded = model(batch(values, idx), mask, recovery=True)
            classification = classification_loss(logits, target[idx], BASE_TRAINING['smoothing'])
            reconstruction = recovery_loss(decoded, values['rating'][idx], mask)
            loss = classification + .1 * reconstruction
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True)
        optimizer.step()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / timed
    changed = float((model.layers[0].weight.detach() - before).abs().max())
    auxiliary_changed = float((model.recovery.weight.detach() - recovery_before).abs().max())
    assert changed > 0 and auxiliary_changed > 0
    # Resource selection only. No stopping/scored features or labels used above.
    estimated_24 = seconds * math.ceil(len(index) / 1024) * 24
    if estimated_24 <= 1200:
        warm_epochs, fine_epochs = 12, 12
    elif seconds * math.ceil(len(index) / 1024) * 16 <= 1800:
        warm_epochs, fine_epochs = 8, 8
    else:
        raise RuntimeError(f'Probe exceeds bounded budget: {estimated_24:.1f}s per 24 epochs')
    result = dict(model=MODEL, base_training=BASE_TRAINING, source_hashes=source_hashes(),
        train_rows=len(index), probe_rows=len(chosen), warmup_steps=warmup, timed_steps=timed,
        seconds_per_step=seconds, estimated_24_epoch_seconds=estimated_24,
        warm_epochs=warm_epochs, fine_epochs=fine_epochs,
        maximum_primary_weight_change=changed, maximum_auxiliary_weight_change=auxiliary_changed,
        final_probe_classification_loss=float(classification.detach()),
        final_probe_recovery_loss=float(reconstruction.detach()),
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
        device=torch.cuda.get_device_name(0), probe_parameters_discarded=True,
        no_stopping_or_scored_data_used=True)
    write_json(FAMILY / 'probe.json', result)
    print('PROBE', json.dumps(result, ensure_ascii=False), flush=True)
    return result


def freeze_protocol():
    setup()
    protected = check_preserved()
    path = FAMILY / 'probe.json'
    measured = json.loads(path.read_text(encoding='utf-8')) if path.exists() else probe()
    assert measured['source_hashes'] == source_hashes(), 'Re-run probe after source changes'
    training = dict(BASE_TRAINING, warm_epochs=measured['warm_epochs'], fine_epochs=measured['fine_epochs'])
    hashes = dict(source_hashes(), **{'train.csv': sha(ROOT / 'train.csv'),
                                  'test.csv': sha(ROOT / 'test.csv')})
    for stage in ['screen', 'confirmation']:
        for name in ['indices.npz', 'baseline.npz', 'lean_mean.npz']:
            p = REFERENCE / stage / name
            hashes[str(p.relative_to(ROOT))] = sha(p)
    protocol = dict(model=MODEL, training=training, specs=SPECS, usages=USAGES,
        screen_gate=.0003, confirmation_gate=.0002, require_paired_lower_above_zero=True,
        architecture='standalone MLP with periodic numerical and categorical embeddings; not pytabkit RealMLP',
        input='raw 21 columns; dual views; no target encodings/counts/teachers/base predictions',
        prediction_ensemble='mean of member probabilities',
        two_stage_optimizer_reset=True, exact_full_epoch_budgets=True,
        warm_phase_no_stopping_selection=True, fine_phase_inner_auc_checkpoint=True,
        historical_partitions_seen=True, frozen_prediction_intervals_exclude_training_uncertainty=True,
        probe_sha256=sha(path), hashes=hashes, protected=protected,
        versions={p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'scikit-learn']})
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:20]
    folder = FAMILY / signature
    folder.mkdir(exist_ok=True)
    if (folder / 'protocol.json').exists():
        assert json.loads((folder / 'protocol.json').read_text(encoding='utf-8')) == protocol
    else:
        write_json(folder / 'protocol.json', protocol)
        for name in ['experiment_multitask.py', 'multitask_support.py']:
            (folder / name).write_bytes((ROOT / name).read_bytes())
    write_json(FAMILY / 'latest.json', dict(directory=str(folder), signature=signature))
    return folder, protocol


def init_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_arm(folder, spec, fit_values, ya, stop_values, yb, encoder, protocol):
    config = protocol['training']
    out = folder / spec
    out.mkdir(exist_ok=True)
    if (out / 'complete.json').exists():
        return json.loads((out / 'complete.json').read_text(encoding='utf-8'))
    torch.manual_seed(config['seed'])
    model = RatingMLP(encoder.state(), protocol['model'], float(ya.mean())).cuda()
    initial_hash = init_hash(model)
    y_gpu = torch.as_tensor(ya, device='cuda', dtype=torch.float32)
    total = config['warm_epochs'] + config['fine_epochs']
    steps_per_epoch = math.ceil(len(ya) / config['batch_size'])
    checkpoint = out / 'progress.pt'
    progress = torch.load(checkpoint, map_location='cuda', weights_only=True) if checkpoint.exists() else None
    completed = progress['epoch'] if progress else 0
    logs = progress['logs'] if progress else []
    best_auc = progress['best_auc'] if progress else -1.
    best_epoch = progress['best_epoch'] if progress else None
    if progress:
        assert progress['initial_hash'] == initial_hash
        model.load_state_dict(progress['model'])
        torch.set_rng_state(progress['rng_cpu'].cpu())
        torch.cuda.set_rng_state(progress['rng_cuda'].cpu())
    optimizer, scheduler, active_phase = None, None, None
    accumulated_seconds = float(progress['elapsed_seconds']) if progress else 0.
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if progress is None:
        initial_recovery = reconstruction_metrics(model, stop_values, config['prediction_batch_size'])
        write_json(out / 'initial_recovery.json', initial_recovery)
    for epoch in range(completed + 1, total + 1):
        phase = 0 if epoch <= config['warm_epochs'] else 1
        phase_epoch = epoch if phase == 0 else epoch - config['warm_epochs']
        phase_cap = config['warm_epochs'] if phase == 0 else config['fine_epochs']
        if active_phase != phase:
            lr = config['warm_lr'] if phase == 0 else config['fine_lr']
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                weight_decay=config['weight_decay'], fused=config['fused'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=phase_cap * steps_per_epoch, eta_min=lr * config['minimum_lr_fraction'])
            if progress and phase == progress['phase']:
                optimizer.load_state_dict(progress['optimizer'])
                scheduler.load_state_dict(progress['scheduler'])
            active_phase = phase
        order_gen = torch.Generator(device='cuda').manual_seed(config['seed'] + 10000 * phase + phase_epoch)
        mask_gen = torch.Generator(device='cuda').manual_seed(config['seed'] + 50000 * phase + phase_epoch)
        order = torch.randperm(len(ya), device='cuda', generator=order_gen)
        model.train()
        aggregate = torch.zeros(3, device='cuda')
        epoch_start = time.perf_counter()
        for step in range(steps_per_epoch):
            idx = order[step * config['batch_size']:(step + 1) * config['batch_size']]
            mask = None
            if phase == 0 and spec != 'A_clean':
                mask = torch.rand((len(idx), len(RATINGS)), device='cuda', generator=mask_gen) < config['mask_probability']
            use_auxiliary = phase == 0 and spec == 'C_multitask'
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['amp_bfloat16']):
                logits, decoded = model(batch(fit_values, idx), mask, recovery=use_auxiliary)
                cls = classification_loss(logits, y_gpu[idx], config['smoothing'])
                rec = recovery_loss(decoded, fit_values['rating'][idx], mask) if use_auxiliary else cls.new_zeros(())
                loss = cls + config['auxiliary_weight'] * rec
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'],
                                          error_if_nonfinite=True, foreach=True)
            optimizer.step()
            scheduler.step()
            aggregate += torch.stack([loss.detach(), cls.detach(), rec.detach()]) * len(idx)
        losses = (aggregate / len(ya)).cpu().tolist()
        assert np.isfinite(losses).all()
        stopping_prediction = predict(model, stop_values, config['prediction_batch_size'])
        auc = float(roc_auc_score(yb, stopping_prediction))
        row = dict(epoch=epoch, phase='warm' if phase == 0 else 'fine', phase_epoch=phase_epoch,
                   train_loss=losses[0], classification_loss=losses[1], recovery_loss=losses[2],
                   stopping_auc=auc, learning_rate=float(scheduler.get_last_lr()[0]),
                   seconds=time.perf_counter() - epoch_start)
        if phase == 0 and phase_epoch == phase_cap:
            row['masked_stopping_recovery'] = reconstruction_metrics(model, stop_values, config['prediction_batch_size'])
        if phase == 1 and auc > best_auc:
            best_auc, best_epoch = auc, epoch
            write_torch(out / 'best.pt', cpu_state(model))
            np.savez_compressed(out / 'best_stopping.npz', prediction=stopping_prediction)
        logs.append(row)
        write_json(out / 'history.json', logs)
        write_torch(checkpoint, dict(epoch=epoch, phase=phase, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(),
            logs=logs, best_auc=best_auc, best_epoch=best_epoch, initial_hash=initial_hash,
            elapsed_seconds=accumulated_seconds + time.perf_counter() - start))
        print('EPOCH', spec, json.dumps(row), flush=True)
    assert best_epoch is not None
    model.load_state_dict(torch.load(out / 'best.pt', map_location='cuda', weights_only=True))
    final_recovery = reconstruction_metrics(model, stop_values, config['prediction_batch_size'])
    result = dict(spec=spec, initial_hash=initial_hash, best_stopping_auc=best_auc,
        best_epoch=best_epoch, epochs_completed=total, updates_completed=total * steps_per_epoch,
        warm_epochs=config['warm_epochs'], fine_epochs=config['fine_epochs'],
        masked_stopping_recovery=final_recovery,
        seconds=accumulated_seconds + time.perf_counter() - start,
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
        parameter_count=sum(p.numel() for p in model.parameters()),
        best_checkpoint_sha256=sha(out / 'best.pt'),
        initial_recovery=json.loads((out / 'initial_recovery.json').read_text(encoding='utf-8')))
    write_json(out / 'complete.json', result)
    del optimizer, scheduler, model, y_gpu
    gc.collect()
    torch.cuda.empty_cache()
    return result


def stage_inputs(stage, x, y, folder):
    indices = split(stage)
    encoder = RawEncoder().fit(x.iloc[indices['fit']])
    state = encoder.state()
    path = folder / 'encoder.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == state
    else:
        write_json(path, state)
    np.savez_compressed(folder / 'indices.npz', **indices)
    fit_values = to_device(encoder.transform(x.iloc[indices['fit']]), 'cuda')
    stop_values = to_device(encoder.transform(x.iloc[indices['stopping']]), 'cuda')
    return indices, encoder, fit_values, stop_values


def predict_arm(folder, spec, x, scored, encoder, protocol):
    path = folder / (spec + '.npz')
    if path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['index'], scored)
            return saved['prediction'].copy()
    model = RatingMLP(encoder.state(), protocol['model']).cuda()
    model.load_state_dict(torch.load(folder / spec / 'best.pt', map_location='cuda', weights_only=True))
    values = to_device(encoder.transform(x.iloc[scored]), 'cuda')
    p = predict(model, values, protocol['training']['prediction_batch_size'])
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.savez_compressed(path, index=scored, prediction=p)
    del model, values
    gc.collect()
    torch.cuda.empty_cache()
    return p


def references(stage, scored):
    out = []
    for name in ['baseline', 'lean_mean']:
        with np.load(REFERENCE / stage / (name + '.npz')) as saved:
            np.testing.assert_array_equal(saved['index'], scored)
            out.append(saved['prediction'].copy())
    return out


def evaluate(stage, folder, x, y, scored, predictions, details, frozen_usage=None):
    v4, lean = references(stage, scored)
    baseline = combine(v4, lean)
    target = y[scored]
    trials = []
    for spec, p in predictions.items():
        for usage in ([frozen_usage] if frozen_usage else USAGES):
            q = combine(v4, lean, p, usage)
            comparison = paired(target, q, baseline)
            trials.append(dict(spec=spec, usage=usage, auc=float(roc_auc_score(target, q)), **comparison))
    winner = max(trials, key=lambda row: row['delta'])
    threshold = .0003 if stage == 'screen' else .0002
    passing = winner['delta'] >= threshold and winner['interval95'][0] > 0
    ratings_hash = pd.util.hash_pandas_object(x[RATINGS], index=False)
    fitidx = split(stage)['fit']
    seen = ratings_hash.iloc[scored].isin(set(ratings_hash.iloc[fitidx])).to_numpy()
    groups = {}
    for name, mask in [('seen_rating_pattern', seen), ('unseen_rating_pattern', ~seen)]:
        q = combine(v4, lean, predictions[winner['spec']], winner['usage'])
        groups[name] = dict(rows=int(mask.sum()), baseline_auc=float(roc_auc_score(target[mask], baseline[mask])),
            candidate_blend_auc=float(roc_auc_score(target[mask], q[mask])))
    arm_comparisons = {}
    for a, b in [('B_masked', 'A_clean'), ('C_multitask', 'B_masked'), ('C_multitask', 'A_clean')]:
        if a in predictions and b in predictions:
            arm_comparisons[a + '_minus_' + b] = paired(target, predictions[a], predictions[b])
    result = dict(stage=stage, baseline_auc=float(roc_auc_score(target, baseline)),
        standalone_auc={key: float(roc_auc_score(target, value)) for key, value in predictions.items()},
        trials=trials, winner=winner, gate=threshold, passed=bool(passing),
        details=details, arm_comparisons=arm_comparisons, pattern_groups=groups,
        note='Historical partitions already viewed; intervals cover fixed predictions only; '
             'screen winner is selected from six fixed trials; no multiple-selection correction.')
    write_json(folder / 'summary.json', result)
    print('STAGE_RESULT', json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run_stage(stage, directory, protocol):
    folder = directory / stage
    folder.mkdir(exist_ok=True)
    if (folder / 'summary.json').exists():
        return json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    chosen, usage = SPECS, None
    if stage == 'confirmation':
        previous = json.loads((directory / 'screen/summary.json').read_text(encoding='utf-8'))
        if not previous['passed']:
            raise RuntimeError('Screen gate failed; confirmation is not authorized by the frozen protocol')
        chosen = [previous['winner']['spec']]
        usage = previous['winner']['usage']
        write_json(folder / 'frozen_selection.json', previous['winner'])
    x, y, _ = data()
    indices, encoder, fit_values, stop_values = stage_inputs(stage, x, y, folder)
    print('STAGE', stage, {k: len(v) for k, v in indices.items()}, flush=True)
    details, predictions = {}, {}
    for spec in chosen:
        details[spec] = train_arm(folder, spec, fit_values, y[indices['fit']],
                                  stop_values, y[indices['stopping']], encoder, protocol)
        predictions[spec] = predict_arm(folder, spec, x, indices['scored'], encoder, protocol)
    assert len(set(d['initial_hash'] for d in details.values())) == 1
    result = evaluate(stage, folder, x, y, indices['scored'], predictions, details, usage)
    del fit_values, stop_values
    gc.collect()
    torch.cuda.empty_cache()
    check_preserved()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['probe', 'screen', 'confirmation', 'all'])
    args = parser.parse_args()
    FAMILY.mkdir(exist_ok=True)
    if args.command == 'probe':
        probe()
        return
    directory, protocol = freeze_protocol()
    print('OUTPUT', directory, flush=True)
    screen = run_stage('screen', directory, protocol)
    if args.command in ['all', 'confirmation'] and screen['passed']:
        confirmed = run_stage('confirmation', directory, protocol)
        print('NEXT', 'budget_control_and_second_seed_required' if confirmed['passed'] else 'stop_confirmation', flush=True)
    else:
        print('NEXT', 'confirmation_required' if screen['passed'] else 'stop_screen_no_new_submission', flush=True)
    check_preserved()


if __name__ == '__main__':
    main()
