"""Frozen public base predictions + cached own branches; no base-model training."""
import argparse
import concurrent.futures
import gc
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RUN = ROOT / '.artifact_runtime/public_pool_trial_20261020'
REF = ROOT / 'research_s6e10/extended_review_20261020'
WHITELIST = pd.read_csv(REF / 'sources/stack247_inputs.csv')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def guard():
    confirmation = RUN / 'confirmation_protocol.json'
    deadline = (read_json(confirmation)['deadline_epoch'] if confirmation.exists()
                else read_json(RUN / 'protocol.json')['screen_deadline_epoch'])
    if time.time() > deadline:
        raise TimeoutError('Stage deadline: checkpoint preserved')
    if psutil.virtual_memory().available < 350 * 1024**2:
        raise MemoryError('Available memory below 350 MiB: checkpoint preserved')


def file_inventory(source):
    data = read_json(RUN / (source + '_files.json'))
    return data.get('datasetFiles', data.get('files', []))


def select_file(source, basename):
    found = [x for x in file_inventory(source)
             if x.get('name', x.get('fileName', '')).split('/')[-1] == basename]
    if len(found) != 1:
        raise FileNotFoundError(f'{source}/{basename}: {len(found)} matches')
    return found[0]


def specification(row):
    s, n = row.source, row.member
    if s == 'c124':
        return [f'oof_{n}.npy', f'test_{n}.npy'], 'npy'
    prefixes = {'tfm': 'tfm_', 'golem': 'gl_', 'forge': 'am_', 'koumeimaki': 'km_',
                'gp_extra': 'gpx_', 'kagan_abl': 'kga_'}
    if s in prefixes:
        tag = n[len(prefixes[s]):]
        return [f'oof_{tag}.npy', f'test_{tag}.npy'], 'npy'
    if s == 'abdullah':
        tag = n[3:]
        return [f'{tag}_oof.csv', f'{tag}_test.csv'], 'csv_pred'
    if s == 'sadam':
        tag = n[3:]
        return [tag + '.npz' if tag.startswith('orig_') else 'preds_' + tag + '.npz'], 'npz_orig' if tag.startswith('orig_') else 'npz'
    if s == 'sachith':
        return [n[3:] + '.npz'], 'npz'
    if s == 'dmitriy':
        return ['oof_cb_cat.npy', 'test_cb_cat.npy'], 'npy'
    if s == 'yekenot':
        return ['oof_preds.csv', 'submission.csv'], 'csv_satisfaction'
    if s == 'hermengardo':
        return ['oof_train.npy', 'oof_test.npy'], 'npy'
    if s == 'kratosyan':
        return [n[3:] + '.npz'], 'npz'
    if s == 'kagan_enc':
        return ['oof_members.csv', 'test_members.csv'], 'csv_member'
    if s == 'aman_v5':
        return ['oof_artifacts.npz'], 'npz_aman'
    if s == 'kozy':
        return ['source_transfer_oof.npy', 'source_transfer_test.npy'], 'npy'
    raise ValueError(s)


def download(job):
    source, name, url, creation = job
    guard()
    folder = RUN / 'downloads' / source
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if not path.exists():
        req = urllib.request.Request(url, headers={'User-Agent': 'kaggle-api/v1.7.0'})
        with urllib.request.urlopen(req, timeout=25) as response:
            chunks, received = [], 0
            while True:
                guard()
                chunk = response.read(1024**2)
                if not chunk:
                    break
                chunks.append(chunk); received += len(chunk)
                if received > 200*1024**2:
                    raise ValueError('Download exceeded 200 MiB per-file bound')
            raw = b''.join(chunks)
        if len(raw) > 200 * 1024**2:
            raise ValueError('Download exceeded 200 MiB per-file bound')
        if raw[:2] == b'PK':
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                hits = [x for x in z.namelist() if x.split('/')[-1] == name]
                if len(hits) == 1:
                    raw = z.read(hits[0])
                elif not name.endswith('.npz'):
                    raise ValueError('Unexpected archive members')
        partial = path.with_name(path.name+'.part')
        partial.write_bytes(raw)
        partial.replace(path)
    return {'source': source, 'filename': name, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'file_creation_date': creation, 'download_url': url}


def prepare_downloads():
    jobs, members, skipped = {}, [], []
    for row in WHITELIST.itertuples():
        try:
            names, kind = specification(row)
            for name in names:
                f = select_file(row.source, name)
                url = f.get('url') if row.source_kind == 'notebook' else (
                    'https://www.kaggle.com/api/v1/datasets/download/' + row.source_ref + '/' +
                    urllib.parse.quote(f['name'], safe='/'))
                jobs[row.source, name] = (row.source, name, url, f.get('creationDate'))
            members.append({'member': row.member, 'source': row.source, 'source_ref': row.source_ref,
                            'author': row.author, 'files': names, 'kind': kind,
                            'reference_auc': row.oof_auc, 'reference_scale': row.scale})
        except FileNotFoundError as e:
            skipped.append({'member': row.member, 'reason': str(e)})
    # Sidecar IDs independently establish the seven Koumei Maki members' order.
    for name in ['train_ids_folds.csv', 'test_ids.csv']:
        f = select_file('koumeimaki', name)
        url = 'https://www.kaggle.com/api/v1/datasets/download/koumeimaki/s6e10-route-tfm-nn-oof/' + name
        jobs['koumeimaki', name] = ('koumeimaki', name, url, f.get('creationDate'))
    write_json(RUN / 'selected_members.json', members)
    write_json(RUN / 'skipped_members.json', skipped)
    print('DOWNLOAD_PLAN', len(members), 'members', len(jobs), 'files', 'skipped', skipped, flush=True)
    receipts, failed = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(download, job): key for key, job in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                receipts.append(future.result())
            except Exception as e:
                failed.append({'file': list(futures[future]), 'error': type(e).__name__ + ': ' + str(e)})
                print('DOWNLOAD_FAILED', failed[-1], flush=True)
            if (len(receipts) + len(failed)) % 40 == 0:
                print('DOWNLOAD_PROGRESS', len(receipts), 'failed', len(failed), flush=True)
    write_json(RUN / 'download_receipts.json', receipts)
    write_json(RUN / 'download_failures.json', failed)
    print('DOWNLOAD_COMPLETE', len(receipts), 'failed', len(failed), flush=True)


def csv_by_id(path, ids, col):
    frame = pd.read_csv(path)
    assert frame.id.is_unique and len(frame) == len(ids)
    out = frame.set_index('id')[col].reindex(ids).to_numpy(dtype='float64')
    assert np.isfinite(out).all()
    return out


def load_member(m, train_ids, test_ids, y):
    files = [RUN / 'downloads' / m['source'] / n for n in m['files']]
    kind = m['kind']
    if kind == 'npy':
        return *(np.load(p, allow_pickle=False).ravel() for p in files), 'author_declared_row_order'
    if kind.startswith('csv_'):
        col = {'csv_pred': 'pred', 'csv_satisfaction': 'satisfaction', 'csv_member': m['member'][3:]}[kind]
        return csv_by_id(files[0], train_ids, col), csv_by_id(files[1], test_ids, col), 'id_reindex'
    with np.load(files[0], allow_pickle=False) as z:
        if kind == 'npz_aman':
            np.testing.assert_array_equal(z['y'], y)
            tag = m['member'][4:]
            return z['oof_' + tag], z['test_' + tag], 'author_row_order_plus_label_vector'
        if kind == 'npz_orig':
            return z['train'], z['test'], 'author_declared_row_order'
        return z['oof'], z['test'], 'author_declared_row_order'


def to_logits(o, t):
    o, t = np.asarray(o, dtype='float64').ravel(), np.asarray(t, dtype='float64').ravel()
    assert np.isfinite(o).all() and np.isfinite(t).all()
    already = o.min() < 0 or o.max() > 1 or t.min() < 0 or t.max() > 1
    if not already:
        o, t = np.clip(o, 1e-15, 1-1e-15), np.clip(t, 1e-15, 1-1e-15)
        o, t = np.log(o / (1-o)), np.log(t / (1-t))
    # Match the reference's float32 base-logit rounding, then store float64 for fitting.
    return np.clip(o, -30, 30).astype('float32').astype('float64'), np.clip(t, -30, 30).astype('float32').astype('float64'), 'logits' if already else 'probability_to_logit'


def own_branches(n, nt):
    specs = [('own_old_xgb', 'optimized_cv/ad685bf982755538eefb', 'fold_{f:02d}_component_0.npz'),
             ('own_old_cat', 'optimized_cv/ad685bf982755538eefb', 'fold_{f:02d}_component_1.npz'),
             ('own_xgb', 'v3/cv/f8070b1c8026703e7d6e', 'fold_{f:02d}_xgb.npz'),
             ('own_realmlp', 'v3/cv/f8070b1c8026703e7d6e', 'fold_{f:02d}_realmlp.npz'),
             ('own_lgb', 'v4/4891ee0e44b7ca16b6ed', 'fold_{f:02d}.npz')]
    for name, directory, pattern in specs:
        o, t, coverage, hashes = np.zeros(n), np.zeros(nt), np.zeros(n, dtype='int8'), []
        for f in range(1, 11):
            p = ROOT / '.artifact_runtime' / directory / pattern.format(f=f)
            hashes.append({'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
            with np.load(p, allow_pickle=False) as z:
                idx = z['valid_index']; o[idx] = z['validation_prediction']; t += z['test_prediction']/10; coverage[idx] += 1
        assert (coverage == 1).all()
        yield name, o, t, hashes
    for seed in [42, 20261019]:
        o, t, coverage, hashes = np.zeros(n), np.zeros(nt), np.zeros(n, dtype='int8'), []
        for f in range(10):
            p = ROOT / '.artifact_runtime/lean_cv_followup/8f29c4956de9c49f6f97' / f'fold_{f}_seed_{seed}.npz'
            hashes.append({'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
            with np.load(p, allow_pickle=False) as z:
                idx = z['index']; o[idx] = z['prediction']; t += z['test']/10; coverage[idx] += 1
        assert (coverage == 1).all()
        yield f'own_lean_{seed}', o, t, hashes


def build_matrix():
    guard()
    tr = pd.read_csv(ROOT / 'train.csv', usecols=['id', 'satisfaction'])
    ids, y = tr.id.to_numpy(), tr.satisfaction.to_numpy(dtype='int64')
    test_ids = pd.read_csv(ROOT / 'test.csv', usecols=['id']).id.to_numpy()
    side = RUN / 'downloads/koumeimaki'
    sidecars_ready = (side/'train_ids_folds.csv').exists() and (side/'test_ids.csv').exists()
    if sidecars_ready:
        np.testing.assert_array_equal(pd.read_csv(side/'train_ids_folds.csv').id, ids)
        np.testing.assert_array_equal(pd.read_csv(side/'test_ids.csv').id, test_ids)
    members = read_json(RUN / 'selected_members.json')
    # Validate one member at a time before reserving matrix columns; failed pairs are explicit.
    valid, rejected = [], []
    for m in members:
        if (m['source'] != 'koumeimaki' or sidecars_ready) and all((RUN/'downloads'/m['source']/f).is_file() for f in m['files']):
            valid.append(m)
        else:
            rejected.append({'member': m['member'], 'reason': 'missing download'})
    n_public = len(valid)
    x = np.lib.format.open_memmap(RUN/'train_logits.npy', mode='w+', dtype='float64', shape=(len(y), n_public+7))
    xt = np.lib.format.open_memmap(RUN/'test_logits.npy', mode='w+', dtype='float64', shape=(len(test_ids), n_public+7))
    columns = []
    for j, m in enumerate(valid):
        guard()
        o, t, evidence = load_member(m, ids, test_ids, y)
        if m['source'] == 'koumeimaki':
            evidence = 'sidecar_ids_exact_match'
        assert len(o) == len(y) and len(t) == len(test_ids), m['member']
        o, t, scale = to_logits(o, t)
        auc = float(roc_auc_score(y, o))
        if abs(auc-m['reference_auc']) > 2e-6:
            raise ValueError(f"Source changed: {m['member']} AUC {auc} vs {m['reference_auc']}")
        x[:, j], xt[:, j] = o, t
        columns.append(dict(m, actual_auc=auc, alignment_evidence=evidence, applied_scale=scale))
        if j % 40 == 0: print('MATRIX_PROGRESS', j, n_public, flush=True)
    for j, (name, o, t, hashes) in enumerate(own_branches(len(y), len(test_ids)), start=n_public):
        guard(); o, t, scale = to_logits(o, t); x[:, j], xt[:, j] = o, t
        columns.append({'member': name, 'author': 'this project', 'alignment_evidence': 'saved_fold_indices_full_coverage',
                        'actual_auc': float(roc_auc_score(y, o)), 'cache_hashes': hashes, 'applied_scale': scale})
    x.flush(); xt.flush(); del x, xt; gc.collect()
    write_json(RUN/'matrix_manifest.json', {'public_columns': n_public, 'own_columns': 7,
               'rows_train': len(y), 'rows_test': len(test_ids), 'columns': columns,
               'rejected': rejected, 'dtype': 'float64 after reference float32 logit rounding'})
    print('MATRIX_COMPLETE', n_public, '+7', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['download', 'matrix'])
    args = parser.parse_args()
    try:
        prepare_downloads() if args.stage == 'download' else build_matrix()
    except Exception as error:
        write_json(RUN/(args.stage+'_stopped.json'), {'error': type(error).__name__+': '+str(error), 'epoch': time.time()})
        raise
