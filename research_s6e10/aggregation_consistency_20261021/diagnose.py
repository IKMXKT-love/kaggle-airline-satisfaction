"""Label-free descriptive audit of cached OOF/test prediction geometry. No fits."""
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
BASE = PROJECT / '.artifact_runtime/public_pool_trial_20261020'


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def summary(values):
    return dict(zip(['min','p05','p25','median','p75','p95','max'], map(float, np.quantile(values,[0,.05,.25,.5,.75,.95,1]))))


def family(name):
    for label, pattern in [('realmlp','rmlp|realmlp'),('tabpfn','tabpfn'),('xgb','xgb'),('cat','cat'),('lgb','lgb')]:
        if re.search(pattern, name, re.I):
            return label
    return None


def main():
    start = time.perf_counter()
    if (ROOT / 'diagnostics.json').exists():
        raise FileExistsError('Preserve completed diagnostic.')
    protected_names = ['submission_public_pool_20261020.csv','CURRENT_BEST_SUBMISSION.json','CURRENT_BEST_RESULT.md','CURRENT_RECOMMENDED_CANDIDATE.json']
    protected = {name: sha(PROJECT / name) for name in protected_names}
    n, seed = 20000, 202610211
    protocol = {
        'date': '2026-10-21', 'kind': 'label-free descriptive audit',
        'samples': 'three disjoint OOF samples A/B/C and two disjoint test samples A/B; 20000 rows each',
        'seed': seed, 'columns': 'first 246 cached public members; assert manifest order',
        'fits': 0, 'base_inferences': 0, 'labels_read': False, 'submissions': 0,
        'pair_selection': 'same author and name-inferred family; rank correlation >.97 in OOF A and B; rank-difference SD >.002 in OOF A and B',
        'descriptive_flag': 'test/OOF rank-difference SD ratio <.8 in both A and B; not a significance test',
        'limits': ['not an AUC comparison','not a covariate-shift attribution','row sampling cannot establish retraining uncertainty','name-based family classification is approximate','no rescaling or new prediction recipe'],
        'protected_before': protected,
    }
    save('protocol.json', protocol)
    if psutil.virtual_memory().available < 700 * 2**20:
        raise MemoryError('Less than 700 MiB available.')
    manifest = json.loads((BASE / 'matrix_manifest.json').read_text('utf-8'))['columns'][:246]
    members = list(csv.DictReader((PROJECT / 'public_pool_sources_20261020.csv').open(encoding='utf-8-sig')))
    assert [x['member'] for x in members] == [x['member'] for x in manifest]
    train = np.load(BASE / 'train_logits.npy', mmap_mode='r', allow_pickle=False)
    test = np.load(BASE / 'test_logits.npy', mmap_mode='r', allow_pickle=False)
    assert train.shape == (699635,253) and test.shape == (299844,253)
    rng = np.random.default_rng(seed)
    tr_indices = rng.choice(len(train), 3*n, replace=False).reshape(3,n)
    te_indices = rng.choice(len(test), 2*n, replace=False).reshape(2,n)
    np.savez(ROOT / 'sample_indices.npz', train=tr_indices, test=te_indices)
    statistics = {}
    with threadpool_limits(limits=2):
        for name, matrix, index in [('train_a',train,tr_indices[0]),('train_b',train,tr_indices[1]),('train_c',train,tr_indices[2]),('test_a',test,te_indices[0]),('test_b',test,te_indices[1])]:
            data = np.asarray(matrix[index,:246])
            assert np.isfinite(data).all()
            raw_std = data.std(axis=0,ddof=1)
            ranks = rankdata(data,axis=0,method='average') / (n+1)
            covariance = np.cov(ranks,rowvar=False)
            sd = np.sqrt(np.diag(covariance))
            correlation = covariance / np.outer(sd,sd)
            disagreement_sd = np.sqrt(np.maximum(np.diag(covariance)[:,None]+np.diag(covariance)[None,:]-2*covariance,0))
            statistics[name] = dict(raw_std=raw_std,rank_correlation=correlation,rank_disagreement_sd=disagreement_sd)
            del data,ranks
    np.savez(ROOT/'sample_statistics.npz',**{f'{sample}__{key}':value for sample,row in statistics.items() for key,value in row.items()})
    upper = np.triu_indices(246,k=1)
    raw_ratio = ((statistics['test_a']['raw_std'] + statistics['test_b']['raw_std']) / (statistics['train_a']['raw_std'] + statistics['train_b']['raw_std']))
    raw_null = statistics['train_c']['raw_std'] / statistics['train_a']['raw_std']
    pairs = []
    for i,j in zip(*upper):
        a,b = members[i],members[j]
        fam = family(a['member'])
        if not fam or a['author'] != b['author'] or fam != family(b['member']):
            continue
        if any(statistics[s]['rank_correlation'][i,j] <= .97 or statistics[s]['rank_disagreement_sd'][i,j] <= .002 for s in ['train_a','train_b']):
            continue
        ratios = [statistics['test_'+s]['rank_disagreement_sd'][i,j]/statistics['train_'+s]['rank_disagreement_sd'][i,j] for s in ['a','b']]
        null_ratio = statistics['train_c']['rank_disagreement_sd'][i,j]/statistics['train_a']['rank_disagreement_sd'][i,j]
        pairs.append({
            'members':[a['member'],b['member']], 'author':a['author'], 'name_inferred_family':fam,
            'oof_rank_corr_a':float(statistics['train_a']['rank_correlation'][i,j]),
            'test_rank_corr_a':float(statistics['test_a']['rank_correlation'][i,j]),
            'rank_disagreement_sd_ratio_a':float(ratios[0]), 'rank_disagreement_sd_ratio_b':float(ratios[1]),
            'within_oof_null_ratio':float(null_ratio),
            'descriptive_shrinkage_flag':bool(max(ratios)<.8),
            'existing_full_fit_coefficients':[float(a['coefficient']),float(b['coefficient'])],
        })
    column_rows = [dict(member=m['member'],author=m['author'],test_oof_raw_std_ratio=float(r),within_oof_null_ratio=float(s)) for m,r,s in zip(members,raw_ratio,raw_null)]
    save('all_pair_descriptions.json',pairs)
    save('column_descriptions.json',column_rows)
    corr_shift = statistics['test_a']['rank_correlation']-statistics['train_a']['rank_correlation']
    null_shift = statistics['train_c']['rank_correlation']-statistics['train_a']['rank_correlation']
    result = {
        'new_fits':0,'base_inferences':0,'labels_read':False,'new_submissions':0,
        'sample_rows_each':n, 'public_columns':246,
        'test_oof_raw_std_ratio':summary(raw_ratio), 'within_oof_raw_std_null_ratio':summary(raw_null),
        'rank_correlation_shift_test_minus_oof':summary(corr_shift[upper]),
        'rank_correlation_shift_within_oof_null':summary(null_shift[upper]),
        'eligible_same_author_family_pairs':len(pairs),
        'descriptive_shrinkage_flags':sum(p['descriptive_shrinkage_flag'] for p in pairs),
        'strongest_descriptive_shrinkage':sorted(pairs,key=lambda p:max(p['rank_disagreement_sd_ratio_a'],p['rank_disagreement_sd_ratio_b']))[:12],
        'input_fingerprints':{name:sha(BASE/name) for name in ['train_logits.npy','test_logits.npy','matrix_manifest.json']},
        'protected_after':{name:sha(PROJECT/name) for name in protected_names},
        'elapsed_seconds':time.perf_counter()-start,
        'limitations':protocol['limits'],
    }
    assert result['protected_after'] == protected
    save('diagnostics.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_fingerprints','protected_after','strongest_descriptive_shrinkage']},ensure_ascii=False,indent=2))
    for pair in result['strongest_descriptive_shrinkage'][:6]:
        print(json.dumps(pair,ensure_ascii=False))


if __name__ == '__main__':
    main()
