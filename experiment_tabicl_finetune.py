"""Predeclared frozen/fine-tuned x 32K/64K experiment, matched to latest recipe."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import time
from pathlib import Path

from tabicl_finetune_support import (
    ROOT, CPUCacheClassifier, SequentialViewsClassifier, BoundedFinetuner, inference_parameters, release)
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier
from tabicl._finetune.data import _build_meta_batch, move_meta_batch
from tabicl._sklearn.preprocessing import TransformToNumerical

OUT = ROOT / '.artifact_runtime/tabicl_finetune_20261019'
FOLLOW = ROOT / '.artifact_runtime/followup_20261019/4261d13e038c1512ffc1'
OLD_TAB = ROOT / '.artifact_runtime/tabicl_probe_20261019/e1f1c42fb3af84f7815f'
CHECKPOINT = OLD_TAB.parent / 'tabicl-classifier-v2-20260212.ckpt'
CHECKPOINT_SHA = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
PROTOCOL = dict(context_sizes=[32768, 65536], epochs=3, learning_rate=1e-5,
    weight_decay=.01, grad_clip=1., amp=True, training_members=2, inference_members=4,
    query_fraction=.2, candidate_chunk_sizes=[2048, 1024, 512], freeze_modules=False,
    lr_scheduler='cosine_warmup', warmup_proportion=.1, patience=2, min_delta=.00001,
    maximum_training_seconds=1200, maximum_fit_seconds=300, maximum_query_seconds=1200,
    maximum_probe_step_seconds=15, maximum_process_rss_gb=20,
    minimum_available_ram_gb=2, seed=42, raw_features_only=True,
    preprocessing_fit_rows_only=True, early_stopping_context=32768,
    checkpoint_selected_on_inner_auc=True, fine_tune_once_then_evaluate_both_contexts=True,
    blend_weights=[.05, .20], blend_form='probability', screen_gate=.0003,
    confirmation_gate=.0002, require_positive_paired_lower_bound=True,
    maximum_fusion_comparisons=8, historical_splits_seen=True,
    base_recipe=dict(v4=.65, lean_realmlp=.35, form='logit'),
    cache_policy='same four views and logit average; one context view on GPU for complete query set',
    tabicl_version='2.2.0', no_public_score_tuning=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def placements(y, p):
    pos, neg = p[y == 1], p[y == 0]
    sp, sn = np.sort(pos), np.sort(neg)
    return ((np.searchsorted(sn, pos, 'left') + np.searchsorted(sn, pos, 'right')) / (2*len(neg)),
            1-(np.searchsorted(sp, neg, 'left') + np.searchsorted(sp, neg, 'right')) / (2*len(pos)))


def paired(y, p, baseline):
    a, b = placements(y, p)
    c, d = placements(y, baseline)
    dp, dn = a-c, b-d
    delta = float(dp.mean())
    se = float(np.sqrt(dp.var(ddof=1)/len(dp) + dn.var(ddof=1)/len(dn)))
    return dict(delta=delta, approximate_se=se, interval95=[delta-1.96*se, delta+1.96*se])


def weight_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def setup():
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    sources = ['experiment_tabicl_finetune.py', 'tabicl_finetune_support.py', 'train.csv',
               'test.csv', 'CURRENT_BEST_SUBMISSION.json']
    hashes = {name: sha(ROOT/name) for name in sources}
    for stage in ['screen', 'confirmation']:
        for name in ['indices.npz', 'baseline.npz', 'lean_mean.npz']:
            hashes[str((FOLLOW/stage/name).relative_to(ROOT))] = sha(FOLLOW/stage/name)
    runtime_paths = list((ROOT/'.artifact_runtime/tabicl_runtime/tabicl').rglob('*.py'))
    runtime_hash = hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): sha(p)
                                            for p in sorted(runtime_paths)}, sort_keys=True).encode()).hexdigest()
    versions = {p: importlib.metadata.version(p) for p in
                ['tabicl', 'torch', 'transformers', 'huggingface-hub', 'numpy', 'scikit-learn']}
    signature = hashlib.sha256(json.dumps([PROTOCOL, hashes, versions, runtime_hash],
                                         sort_keys=True).encode()).hexdigest()[:20]
    directory = OUT/signature
    directory.mkdir(parents=True, exist_ok=True)
    protected = read(ROOT/'.artifact_runtime/v4/preserved.json')
    protected.update({str(p.relative_to(ROOT)): sha(p) for p in ROOT.glob('*.py')
                      if p.name not in ['experiment_tabicl_finetune.py', 'tabicl_finetune_support.py',
                                        'verify_tabicl_finetune.py']})
    protected.update({str(p.relative_to(ROOT)): sha(p) for p in ROOT.glob('submission*.csv')})
    for name in ['CURRENT_BEST_RESULT.md', 'CURRENT_BEST_SUBMISSION.json',
                 'submission_lean_realmlp_10fold_public_result.json']:
        protected[name] = sha(ROOT/name)
    for name, expected in read(ROOT/'.artifact_runtime/v4/preserved.json').items():
        assert sha(ROOT/name) == expected, name
    protocol = dict(**PROTOCOL, hashes=hashes, versions=versions, protected=protected,
                    runtime_source_sha256=runtime_hash, checkpoint_sha256=CHECKPOINT_SHA,
                    gpu=torch.cuda.get_device_name(), signature=signature)
    if (directory/'protocol.json').exists():
        assert read(directory/'protocol.json') == protocol, 'Protocol drift'
    else:
        write(directory/'protocol.json', protocol)
        for name in ['experiment_tabicl_finetune.py', 'tabicl_finetune_support.py']:
            (directory/name).write_bytes((ROOT/name).read_bytes())
    write(OUT/'latest.json', dict(directory=str(directory)))
    return directory


def check_preserved(directory):
    for name, expected in read(directory/'protocol.json')['protected'].items():
        assert sha(ROOT/name) == expected, name


def data_for_stage(directory, stage):
    folder = directory/stage
    folder.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT/'train.csv')
    y = raw.satisfaction.to_numpy(dtype=np.int64)
    x = raw.drop(columns=['id', 'satisfaction'])
    with np.load(FOLLOW/stage/'indices.npz') as z:
        fit, stop, scored = z['fit'].copy(), z['stopping'].copy(), z['scored'].copy()
    assert not set(fit)&set(stop) and not set(fit)&set(scored) and not set(stop)&set(scored)
    if stage == 'screen':
        with np.load(OLD_TAB/'indices.npz') as z:
            c32 = z['context32768'].copy()
    else:
        c32, _ = train_test_split(fit, train_size=32768, stratify=y[fit], random_state=42)
    available = np.setdiff1d(fit, c32)
    extra, _ = train_test_split(available, train_size=32768, stratify=y[available], random_state=43)
    c64 = np.concatenate([c32, extra])
    position_lookup = pd.Series(np.arange(len(fit)), index=fit)
    p32, p64 = position_lookup.loc[c32].to_numpy(), position_lookup.loc[c64].to_numpy()
    assert set(c32) < set(c64) <= set(fit)
    np.savez_compressed(folder/'indices.npz', fit=fit, stopping=stop, scored=scored,
                        context32768=c32, context65536=c64, positions32768=p32, positions65536=p64)
    def load_prediction(name):
        with np.load(FOLLOW/stage/(name+'.npz')) as z:
            np.testing.assert_array_equal(scored, z['index'])
            return z['prediction'].astype(np.float64)
    v4, lean = load_prediction('baseline'), load_prediction('lean_mean')
    current = expit(.65*logit(np.clip(v4, 1e-6, 1-1e-6)) +
                    .35*logit(np.clip(lean, 1e-6, 1-1e-6)))
    np.savez_compressed(folder/'current.npz', index=scored, prediction=current)
    encoder = TransformToNumerical()
    a = np.asarray(encoder.fit_transform(x.iloc[fit]), dtype=np.float64)
    b = np.asarray(encoder.transform(x.iloc[stop]), dtype=np.float64)
    c = np.asarray(encoder.transform(x.iloc[scored]), dtype=np.float64)
    joblib.dump(encoder, folder/'raw_encoder.joblib')
    write(folder/'data.json', dict(fit_rows=len(fit), stopping_rows=len(stop), scored_rows=len(scored),
                                  features=list(x), encoded_features=a.shape[1],
                                  current_auc=float(roc_auc_score(y[scored], current))))
    return dict(folder=folder, fit=fit, stop=stop, scored=scored, positions={32768:p32, 65536:p64},
                a=a, b=b, c=c, ya=y[fit], yb=y[stop], yc=y[scored], current=current)


def probe(directory, data):
    path = directory/'resource_probe.json'
    if path.exists():
        return read(path)
    a, ya = data['a'], data['ya']
    # Numerical equality test isolates CPU cache placement from model behavior.
    p = inference_parameters(CHECKPOINT)
    p['n_estimators'] = 4
    torch.cuda.reset_peak_memory_stats()
    native = TabICLClassifier(**p)
    native.fit(a[:512], ya[:512])
    reference = native.predict_proba(a[512:640])[:, 1]
    reference32 = np.concatenate([native.predict_proba(a[i:i+32])[:,1] for i in range(512,640,32)])
    del native
    gc.collect(); torch.cuda.empty_cache()
    adapted = SequentialViewsClassifier(**p)
    adapted.fit(a[:512], ya[:512])
    actual = adapted.predict_all(a[512:640], query_batch=128)[:, 1]
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-7)
    actual32 = adapted.predict_all(a[512:640], query_batch=32)[:,1]
    np.testing.assert_allclose(actual32, reference32, rtol=0, atol=1e-7)
    release(adapted); del adapted
    checks = []
    chosen = None
    for chunk in PROTOCOL['candidate_chunk_sizes']:
        model = None
        try:
            torch.manual_seed(42); np.random.seed(42)
            torch.cuda.reset_peak_memory_stats()
            model = CPUCacheClassifier(**inference_parameters(CHECKPOINT, 2))
            model._resolve_device(); model._load_model()
            net = model.model_.to('cuda').train()
            before = weight_digest(net)
            batch = _build_meta_batch(a[:chunk], ya[:chunk], classification=True,
                n_estimators=2, query_size=round(chunk*.2), epoch_seed=42, chunk_idx=0,
                norm_methods=['none', 'power'], feat_shuffle_method='latin',
                class_shuffle_method='shift', outlier_threshold=4., preprocessing_seed=42)
            batch = move_meta_batch(batch, torch.device('cuda'))
            opt = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=.01)
            scaler = torch.GradScaler('cuda')
            durations = []
            for step in range(2):
                torch.cuda.synchronize(); start = time.time()
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    scores = net(batch.X, batch.y_train.float())[...,:2].reshape(-1, 2)
                    loss = torch.nn.functional.cross_entropy(scores, batch.y_query.long().reshape(-1))
                assert torch.isfinite(loss)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                assert torch.isfinite(norm), 'Non-finite gradients'
                scaler.step(opt); scaler.update()
                torch.cuda.synchronize(); durations.append(time.time()-start)
            assert weight_digest(net) != before, 'No weight updates'
            projected = max(durations)*math.ceil(len(a)/chunk)*PROTOCOL['epochs']
            row = dict(chunk=chunk, status='ok', step_seconds=durations,
                       loss=float(loss.detach()), gradient_norm=float(norm),
                       peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,
                       projected_training_seconds=projected, parameters_updated=True)
            checks.append(row)
            if max(durations) <= PROTOCOL['maximum_probe_step_seconds'] and projected <= 1200:
                chosen = chunk
            del batch, opt, scaler, scores, loss, norm, net
            if chosen is not None:
                break
        except (torch.OutOfMemoryError, MemoryError) as error:
            checks.append(dict(chunk=chunk, status='stopped_memory', error=str(error)))
        finally:
            release(model)
            del model
        print('PROBE_RESULT', checks[-1], flush=True)
    result = dict(cache_adapter_verified=True, maximum_prediction_difference=float(max(np.max(abs(actual-reference)),
                                                                                     np.max(abs(actual32-reference32)))),
                  official_query_batch_sensitivity=float(np.max(abs(reference-reference32))),
                  checks=checks, chosen_chunk=chosen, system_ram_gb=psutil.virtual_memory().total/1024**3,
                  available_ram_gb=psutil.virtual_memory().available/1024**3)
    write(path, result)
    print('PROBE_COMPLETE', result, flush=True)
    return result


class RunLogger:
    def __init__(self, folder):
        self.folder = folder
        self.epochs = []
        self.steps = 0
        self.start = time.time()

    def setup(self, config):
        return None

    def log_step(self, metrics, *, step):
        self.steps = step
        if step % 25 == 0:
            print('FINETUNE_STEP', step, metrics, 'seconds', round(time.time()-self.start, 1), flush=True)

    def log_epoch(self, metrics, *, step):
        row = dict(metrics, step=step, seconds=time.time()-self.start)
        self.epochs.append(row)
        write(self.folder/'training_progress.json', dict(epochs=self.epochs, steps=step))
        print('FINETUNE_EPOCH', row, flush=True)

    def finish(self):
        return None


def fine_tune(directory, data, chunk):
    folder = data['folder']/'finetune'
    folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        result = read(folder/'summary.json')
        if result['status'] == 'complete':
            assert sha(folder/'best.ckpt') == result['checkpoint_sha256']
        return result
    torch.manual_seed(42); np.random.seed(42)
    trainer = BoundedFinetuner(epochs=3, learning_rate=1e-5, weight_decay=.01,
        grad_clip=1., amp=True, use_lr_scheduler=True, warmup_proportion=.1,
        n_estimators_finetune=2, n_estimators_validation=4, n_estimators_inference=4,
        max_data_size=chunk, finetune_ctx_query_ratio=.2, early_stopping=True,
        patience=2, min_delta=.00001, time_limit=1200, save_interval=1,
        norm_methods=['none', 'power'], model_path=str(CHECKPOINT), allow_auto_download=False,
        freeze_col=False, freeze_row=False, freeze_icl=False, device='cuda', random_state=42,
        verbose=False, eval_metric='roc_auc')
    trainer.training_rows_ = len(data['a'])
    trainer.context_positions_ = data['positions'][32768]
    trainer.validation_trace_ = []
    trainer.run_logger_ = RunLogger(folder)
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.fit(data['a'], data['ya'], X_val=data['b'], y_val=data['yb'], output_dir=folder)
        joblib.dump(trainer._X_encoder_, folder/'numeric_encoder.joblib')
        digest = weight_digest(trainer.model_)
        state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['state_dict']
        selected = trainer.model_.state_dict()
        changed = sum(not torch.equal(state[n], selected[n].detach().cpu()) for n in state)
        del state, selected
        result = dict(status='complete', seconds=time.time()-start, chunk=chunk,
            peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,
            process_rss_gb=psutil.Process().memory_info().rss/1024**3,
            validation_trace=trainer.validation_trace_, epochs=trainer.run_logger_.epochs,
            completed_steps=trainer.run_logger_.steps,
            best_inner_auc=float(trainer._best_metric_), selected_weights_sha256=digest,
            changed_tensors=changed, selected_pretrained=(changed == 0),
            checkpoint_sha256=sha(folder/'best.ckpt'))
    except (torch.OutOfMemoryError, MemoryError) as error:
        result = dict(status='stopped_memory', seconds=time.time()-start, error=str(error))
    finally:
        release(getattr(trainer, '_final_estimator_', None))
        del trainer
        gc.collect(); torch.cuda.empty_cache()
    write(folder/'summary.json', result)
    print('FINETUNE_COMPLETE', result, flush=True)
    return result


def inference_cell(data, state, context):
    folder = data['folder']
    name = f'{state}_{context}'
    path = folder/(name+'.json')
    if path.exists():
        return read(path)
    checkpoint = CHECKPOINT if state == 'frozen' else folder/'finetune/best.ckpt'
    model = None
    start = time.time()
    try:
        torch.cuda.reset_peak_memory_stats()
        if psutil.virtual_memory().available/1024**3 < 2:
            raise MemoryError('Less than 2 GB system RAM available before inference')
        model = SequentialViewsClassifier(**inference_parameters(checkpoint))
        pos = data['positions'][context]
        a, c = data['a'], data['c']
        if state == 'finetuned':
            encoder = joblib.load(folder/'finetune/numeric_encoder.joblib')
            a, c = encoder.transform(a), encoder.transform(c)
        model.fit(a[pos], data['ya'][pos])
        fit_seconds = time.time()-start
        if fit_seconds > 300:
            result = dict(name=name, state=state, context=context, status='stopped_fit_budget',
                          fit_seconds=fit_seconds)
        else:
            query_start = time.time()
            prediction = model.predict_all(c, maximum_seconds=1200, progress=True)[:,1].astype(np.float64)
            if psutil.Process().memory_info().rss/1024**3 > 20:
                raise MemoryError('Process RSS exceeded 20 GB')
            assert len(prediction) == len(data['yc']) and np.isfinite(prediction).all()
            assert ((prediction >= 0) & (prediction <= 1)).all()
            np.savez_compressed(folder/(name+'.npz'), index=data['scored'], prediction=prediction)
            trials = []
            weights = (PROTOCOL['blend_weights'] if folder.name == 'screen' else
                       [read(folder.parent/'screen/decision.json')['chosen']['weight']])
            for w in weights:
                combined = (1-w)*data['current'] + w*prediction
                comparison = paired(data['yc'], combined, data['current'])
                trials.append(dict(weight=w, form='probability',
                                   auc=float(roc_auc_score(data['yc'], combined)), paired=comparison))
            result = dict(name=name, state=state, context=context, status='complete',
                fit_seconds=fit_seconds, query_seconds=time.time()-query_start,
                seconds=time.time()-start, auc=float(roc_auc_score(data['yc'], prediction)),
                baseline_auc=float(roc_auc_score(data['yc'], data['current'])), trials=trials,
                peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,
                process_rss_gb=psutil.Process().memory_info().rss/1024**3,
                checkpoint_sha256=sha(checkpoint), prediction_sha256=sha(folder/(name+'.npz')))
    except (torch.OutOfMemoryError, MemoryError, TimeoutError) as error:
        result = dict(name=name, state=state, context=context, status='stopped_resource',
                      seconds=time.time()-start, error=str(error))
    finally:
        release(model); del model
        gc.collect(); torch.cuda.empty_cache()
    write(path, result)
    print('CELL_RESULT', result, flush=True)
    return result


def screen(directory):
    data = data_for_stage(directory, 'screen')
    resources = probe(directory, data)
    cells = [inference_cell(data, 'frozen', size) for size in PROTOCOL['context_sizes']]
    if resources['chosen_chunk'] is not None:
        trained = fine_tune(directory, data, resources['chosen_chunk'])
        if trained['status'] == 'complete':
            cells.extend(inference_cell(data, 'finetuned', size) for size in PROTOCOL['context_sizes'])
    trials = [dict(name=cell['name'], state=cell['state'], context=cell['context'], **trial)
              for cell in cells if cell['status'] == 'complete' for trial in cell['trials']]
    passing = [t for t in trials if t['paired']['delta'] >= .0003 and t['paired']['interval95'][0] > 0]
    chosen = max(passing, key=lambda t:t['paired']['delta']) if passing else None
    result = dict(cells=cells, trials=trials, chosen=chosen,
                  status='ready_for_confirmation' if chosen else 'stopped_screen_or_resource',
                  completed_cells=sum(c['status']=='complete' for c in cells),
                  four_cell_design_completed=(len(cells)==4 and all(c['status']=='complete' for c in cells)))
    check_preserved(directory)
    write(directory/'screen/decision.json', result)
    print('SCREEN_DONE', result['status'], 'chosen', chosen, flush=True)
    return result


def confirmation(directory):
    decision = read(directory/'screen/decision.json')
    chosen = decision['chosen']
    assert chosen is not None, 'Screen gate not passed'
    data = data_for_stage(directory, 'confirmation')
    if chosen['state'] == 'finetuned':
        resources = read(directory/'resource_probe.json')
        trained = fine_tune(directory, data, resources['chosen_chunk'])
        if trained['status'] != 'complete':
            write(directory/'confirmation/decision.json', dict(status='stopped_resource', training=trained))
            return
    # Only the frozen choice is evaluated: no confirmation-time weight/context search.
    row = inference_cell(data, chosen['state'], chosen['context'])
    if row['status'] != 'complete':
        write(directory/'confirmation/decision.json', dict(status='stopped_resource', cell=row))
        return
    selected = next(t for t in row['trials'] if t['weight'] == chosen['weight'])
    passing = selected['paired']['delta'] >= .0002 and selected['paired']['interval95'][0] > 0
    result = dict(chosen=chosen, selected=selected, status='ready_for_full_cv' if passing else 'stopped_confirmation')
    check_preserved(directory)
    write(directory/'confirmation/decision.json', result)
    print('CONFIRMATION_DONE', result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirmation', 'all'], nargs='?', default='all')
    args = parser.parse_args()
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    directory = setup()
    print('DIRECTORY', directory, flush=True)
    if args.stage in ['screen', 'all']:
        result = screen(directory)
    else:
        result = read(directory/'screen/decision.json')
    if args.stage == 'confirmation' or (args.stage == 'all' and result['chosen']):
        confirmation(directory)
    check_preserved(directory)


if __name__ == '__main__':
    main()
