"""Memory-bounded L2 logistic meta-model on frozen prediction vectors."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.optimize import minimize
from scipy.linalg import solve
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RUN = ROOT / '.artifact_runtime/public_pool_trial_20261020'
C = 0.003
CHUNK = 8192


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def objective(w, x, y, training, columns):
    """Mean log loss + ||coef||^2/(2*C*n), sklearn's unweighted objective."""
    n = int(training.sum()); value = 0.; grad = np.zeros_like(w)
    for begin in range(0, len(y), CHUNK):
        stop = min(begin+CHUNK, len(y)); keep = training[begin:stop]
        a = x[begin:stop, :columns][keep]
        target = y[begin:stop][keep]
        z = a @ w[:-1] + w[-1]
        value += np.sum(np.logaddexp(0., z)-target*z)
        residual = expit(z)-target
        grad[:-1] += a.T @ residual; grad[-1] += residual.sum()
    value = value/n + np.dot(w[:-1], w[:-1])/(2*C*n)
    grad /= n; grad[:-1] += w[:-1]/(C*n)
    return value, grad


def hessian_objective(w, x, y, training, columns):
    n = int(training.sum()); value = 0.; grad = np.zeros_like(w)
    hessian = np.zeros((len(w), len(w)))
    for begin in range(0, len(y), CHUNK):
        stop = min(begin+CHUNK, len(y)); keep = training[begin:stop]
        a = x[begin:stop, :columns][keep]; target = y[begin:stop][keep]
        z = a@w[:-1]+w[-1]; probability = expit(z)
        value += np.sum(np.logaddexp(0., z)-target*z)
        residual = probability-target; variance = probability*(1-probability)
        grad[:-1] += a.T@residual; grad[-1] += residual.sum()
        hessian[:-1, :-1] += a.T@(a*variance[:, None])
        cross = a.T@variance
        hessian[:-1, -1] += cross; hessian[-1, :-1] += cross
        hessian[-1, -1] += variance.sum()
    value = value/n+np.dot(w[:-1], w[:-1])/(2*C*n)
    grad /= n; grad[:-1] += w[:-1]/(C*n)
    hessian /= n
    hessian[np.arange(columns), np.arange(columns)] += 1/(C*n)
    return value, grad, hessian


def newton(x, y, training, columns, deadline, start=None, callback=None):
    w = np.zeros(columns+1) if start is None else start.copy()
    begin = time.time()
    for iteration in range(100):
        if time.time() >= deadline: raise TimeoutError('Stage wall-clock deadline reached')
        if psutil.virtual_memory().available < 300*1024**2: raise MemoryError('Available memory below 300 MiB')
        value, grad, hessian = hessian_objective(w, x, y, training, columns)
        norm = float(np.abs(grad).max())
        print('NEWTON_PROGRESS', iteration, 'seconds', round(time.time()-begin, 1), 'gradient', norm, flush=True)
        if norm <= 1e-8:
            return w, {'iterations': iteration, 'converged': True, 'value': float(value), 'max_gradient': norm}
        direction = solve(hessian, -grad, assume_a='pos')
        directional_derivative = np.dot(grad, direction)
        assert directional_derivative < 0
        step = 1.
        for line in range(25):
            if time.time() >= deadline: raise TimeoutError('Stage deadline in line search')
            candidate = w+step*direction
            candidate_value, _ = objective(candidate, x, y, training, columns)
            if candidate_value <= value+1e-4*step*directional_derivative+1e-15:
                w = candidate; break
            step *= .5
        else: raise RuntimeError('Newton line search failed')
        if callback is not None: callback(w)
    raise RuntimeError('Newton iteration bound reached')


def fit(x, y, training, columns, tag, deadline):
    completed = RUN / (tag + '.npz')
    if completed.exists():
        meta = read(RUN / (tag + '.json'))
        assert meta['converged'] and meta['columns'] == columns
        with np.load(completed, allow_pickle=False) as z: return z['weights'].copy(), meta
    start = time.time()
    checkpoint = RUN/(tag+'_checkpoint.npz')
    initial = None
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as z: initial = z['weights'].copy()
        assert len(initial) == columns+1
    with threadpool_limits(limits=4):
        weights, result = newton(x, y, training, columns, deadline, initial,
                                callback=lambda w: np.savez(checkpoint, weights=w))
    meta = {'tag': tag, 'columns': columns, 'fit_rows': int(training.sum()), 'C': C,
            'seconds': time.time()-start, 'solver': 'streaming damped Newton, identical penalized objective',
            'resumed_same_fold_checkpoint': initial is not None, **result}
    write(RUN/(tag+'.json'), meta)
    if not result['converged']: raise RuntimeError('Meta-model did not converge')
    np.savez(completed, weights=weights)
    print('FIT_COMPLETE', meta, flush=True)
    return weights, meta


def predict(x, w):
    p = np.zeros(x.shape[0]); columns = len(w)-1
    with threadpool_limits(limits=4):
        for begin in range(0, len(p), CHUNK):
            p[begin:begin+CHUNK] = x[begin:begin+CHUNK, :columns] @ w[:-1]+w[-1]
    return p


def check_solver():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(20261020)
    x = rng.normal(size=(1703, 9)); w = rng.normal(size=9)
    y = (rng.random(len(x)) < expit(x@w+.25)).astype(int)
    a = np.arange(len(y)) % 5 != 0
    with threadpool_limits(limits=4):
        result = minimize(lambda v: objective(v, x, y, a, x.shape[1]), np.zeros(10), jac=True,
                          method='L-BFGS-B', options={'gtol': 1e-10, 'ftol': 1e-15})
        ref = LogisticRegression(C=C, tol=1e-10, max_iter=10000).fit(x[a], y[a])
    expected = np.r_[ref.coef_[0], ref.intercept_]
    error = float(np.max(np.abs(result.x-expected)))
    assert result.success and error < 2e-6, error
    with threadpool_limits(limits=4):
        nw, meta = newton(x, y, a, x.shape[1], time.time()+30)
    newton_error = float(np.abs(nw-expected).max())
    assert meta['converged'] and newton_error < 2e-6
    out = {'solver_max_coefficient_error_vs_sklearn': error, 'rows': len(x),
           'regularization_convention_matches': True, 'seed': 20261020}
    out['newton_max_coefficient_error_vs_sklearn'] = newton_error
    write(RUN/'solver_verification.json', out); print(out)


def finalize():
    protocol = read(RUN/'selection_protocol.json')
    result = read(RUN/'confirmation_result.json'); assert result['completed']
    manifest = read(RUN/'matrix_manifest.json')
    tr = pd.read_csv(ROOT/'train.csv', usecols=['id', 'satisfaction'])
    test_ids = pd.read_csv(ROOT/'test.csv', usecols=['id']).id.to_numpy()
    sample_ids = pd.read_csv(ROOT/'sample_submission.csv', usecols=['id']).id.to_numpy()
    np.testing.assert_array_equal(sample_ids, test_ids)
    y = tr.satisfaction.to_numpy(dtype='int64')
    fold = np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy', allow_pickle=False)
    predictions = {}
    for arm in ['public_only', 'public_plus_own']:
        pred = np.zeros(len(y)); coverage = np.zeros(len(y), dtype='int8')
        for f in range(5):
            with np.load(RUN/f'{arm}_fold{f}_validation.npz', allow_pickle=False) as z:
                idx = z['index']; np.testing.assert_array_equal(idx, np.flatnonzero(fold == f))
                pred[idx] = z['logit']; coverage[idx] += 1
        assert (coverage == 1).all() and np.isfinite(pred).all()
        predictions[arm] = pred
    with np.load(ROOT/'.artifact_runtime/lean_cv_followup/8f29c4956de9c49f6f97/predictions.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'], tr.id)
        predictions['current_best'] = z['blended_oof'].copy()
    scores = {k: [float(roc_auc_score(y[fold == f], v[fold == f])) for f in range(5)] for k, v in predictions.items()}
    differences = np.array(scores['public_plus_own'])-np.array(scores['public_only'])
    se = float(differences.std(ddof=1)/np.sqrt(5))
    use_own = differences.mean() > 2*se and (differences > 0).sum() >= 4
    arm = 'public_plus_own' if use_own else 'public_only'
    delta = np.array(scores[arm])-np.array(scores['current_best'])
    accepted = delta.mean() >= protocol['overall_expansion_gate_vs_current_mean_fold_delta'] and (delta > 0).sum() >= protocol['overall_minimum_positive_folds']
    report = {'fold_scores': scores, 'pooled_aucs': {k: float(roc_auc_score(y, v)) for k, v in predictions.items()},
              'own_paired_deltas': differences.tolist(), 'own_delta_mean': float(differences.mean()),
              'own_delta_se': se, 'own_positive_folds': int((differences > 0).sum()), 'own_included': bool(use_own),
              'selected_arm': arm, 'selected_deltas_vs_current': delta.tolist(), 'passes_material_gain_gate': bool(accepted),
              'public_columns': manifest['public_columns'], 'new_public_score': None,
              'validation_caveat': 'Fixed-base-OOF meta-CV, inherited early-stopping and indirect label dependencies; not end-to-end nested CV.'}
    write(RUN/'evaluation.json', report); print('EVALUATION', json.dumps(report), flush=True)
    if not accepted:
        print('NO_NEW_SUBMISSION: material-gain gate failed', flush=True); return
    deadline = read(RUN/'confirmation_protocol.json')['deadline_epoch']
    columns = manifest['public_columns']+(manifest['own_columns'] if use_own else 0)
    x = np.load(RUN/'train_logits.npy', mmap_mode='r', allow_pickle=False)
    w, meta = fit(x, y, np.ones(len(y), dtype=bool), columns, 'selected_full_refit', deadline)
    xt = np.load(RUN/'test_logits.npy', mmap_mode='r', allow_pickle=False)
    test_logits = predict(xt, w); probability = expit(test_logits)
    assert len(probability) == len(test_ids) and np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    selected = manifest['columns'][:columns]
    versions = {v['source']: v for v in read(RUN/'source_versions.json')}
    frame = pd.DataFrame({'member': [m['member'] for m in selected], 'coefficient': w[:-1],
                          'author': [m['author'] for m in selected],
                          'source_ref': [m.get('source_ref', 'local saved cache') for m in selected],
                          'alignment_evidence': [m['alignment_evidence'] for m in selected]})
    frame['source_version'] = [versions.get(m.get('source'), {}).get('version', 'local cache hash') for m in selected]
    frame['reported_source_license'] = [versions.get(m.get('source'), {}).get('license', 'see source') for m in selected]
    notebook_sources = {'yekenot', 'hermengardo', 'kratosyan', 'kagan_enc', 'aman_v5'}
    frame['source_url'] = [('https://www.kaggle.com/'+('code/' if m.get('source') in notebook_sources else 'datasets/')+m['source_ref'])
                           if m.get('source_ref') else '' for m in selected]
    frame.to_csv(ROOT/'public_pool_sources_20261020.csv', index=False, lineterminator='\n')
    out = ROOT/'submission_public_pool_20261020.csv'
    if out.exists(): raise FileExistsError('Preserve existing candidate submission')
    pd.DataFrame({'id': test_ids, 'satisfaction': probability}).to_csv(out, index=False, lineterminator='\n')
    import hashlib
    report.update({'submission': str(out), 'submission_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                   'full_refit': meta, 'test_rows': len(test_ids), 'finite_probabilities': True,
                   'test_probability_mean': float(probability.mean()), 'intercept': float(w[-1]),
                   'all_base_models_reused': True, 'public_score_unknown': True})
    write(ROOT/'submission_public_pool_20261020.json', report)
    np.savez(RUN/'selected_predictions.npz', id=tr.id.to_numpy(), oof_logit=predictions[arm], test_id=test_ids, test_logit=test_logits)
    print('SUBMISSION_READY', str(out), report['submission_sha256'], flush=True)


def run(stage):
    protocol = read(RUN/'protocol.json'); manifest = read(RUN/'matrix_manifest.json')
    y = pd.read_csv(ROOT/'train.csv', usecols=['satisfaction']).satisfaction.to_numpy(dtype='int64')
    fold = np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy', allow_pickle=False)
    x = np.load(RUN/'train_logits.npy', mmap_mode='r', allow_pickle=False)
    p = manifest['public_columns']; all_columns = p+manifest['own_columns']
    row_index = np.arange(len(y))
    if stage == 'screen':
        deadline = protocol['screen_deadline_epoch']; folds = [0]
        # Cost screen only; the later five-fold confirmation uses all training rows.
        # Freeze this size/seed before any candidate fit or score is observed.
        sample_file = RUN/'screen_sampling_protocol.json'
        if not sample_file.exists():
            write(sample_file, {'rows': 200000, 'seed': 20261020,
                                'purpose': 'Bounded CPU/memory and direction screen; not full-pool reproduction',
                                'gate_vs_current': 0.0003, 'candidate_results_observed': False})
        sample = read(sample_file)
        row_index = np.sort(np.random.default_rng(sample['seed']).choice(len(y), sample['rows'], replace=False))
        path = RUN/'screen_logits.npy'
        if not path.exists():
            sx = np.lib.format.open_memmap(path, mode='w+', dtype='float64', shape=(len(row_index), x.shape[1]))
            for begin in range(0, len(row_index), CHUNK): sx[begin:begin+CHUNK] = x[row_index[begin:begin+CHUNK]]
            sx.flush(); del sx
        x = np.load(path, mmap_mode='r', allow_pickle=False); y, fold = y[row_index], fold[row_index]
    else:
        state = RUN/'confirmation_protocol.json'
        if not state.exists():
            screen = read(RUN/'screen_result.json')
            assert screen['completed']
            write(state, {'started_epoch': time.time(), 'deadline_epoch': time.time()+3600,
                          'arms': ['public_only', 'public_plus_own'], 'C': C, 'folds': [0,1,2,3,4],
                          'no_base_model_retraining': True,
                          'reason': 'Public model pool has material prior evidence; screen verifies import, cost and own-branch contribution.'})
        deadline = read(state)['deadline_epoch']; folds = list(range(5))
    rows = []
    for f in folds:
        for arm, columns in [('public_only', p), ('public_plus_own', all_columns)]:
            tag = ('screen_' if stage == 'screen' else '')+f'{arm}_fold{f}'
            w, meta = fit(x, y, fold != f, columns, tag, deadline)
            z = predict(x, w); idx = np.flatnonzero(fold == f)
            validation_name = tag+('_screen' if stage == 'screen' else '')+'_validation.npz'
            np.savez(RUN/validation_name, index=row_index[idx], logit=z[idx])
            rows.append({'fold': f, 'arm': arm, 'auc': float(roc_auc_score(y[idx], z[idx])), 'fit': meta})
    out = {'completed': True, 'stage': stage, 'rows': rows, 'public_columns': p,
           'own_columns': all_columns-p, 'epoch': time.time(),
           'validation_status': 'Fixed-base-OOF meta-CV, exploratory; not end-to-end nested validation'}
    if stage == 'screen':
        with np.load(ROOT/'.artifact_runtime/lean_cv_followup/8f29c4956de9c49f6f97/predictions.npz', allow_pickle=False) as z:
            base = z['blended_oof'][row_index[fold == 0]]
        out['current_auc_on_same_screen_rows'] = float(roc_auc_score(y[fold == 0], base))
        out['public_gain_vs_current'] = rows[0]['auc']-out['current_auc_on_same_screen_rows']
        out['own_increment'] = rows[1]['auc']-rows[0]['auc']
        out['public_passes_expansion_gate'] = out['public_gain_vs_current'] >= 0.0003
    write(RUN/(stage+'_result.json'), out); print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['check', 'screen', 'confirmation', 'finalize'])
    args = parser.parse_args()
    try:
        if args.stage == 'check': check_solver()
        elif args.stage == 'finalize': finalize()
        else: run(args.stage)
    except Exception as e:
        write(RUN/(args.stage+'_stopped.json'), {'error': type(e).__name__+': '+str(e), 'epoch': time.time()})
        raise
