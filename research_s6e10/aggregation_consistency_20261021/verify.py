"""Independent direct-rank arithmetic and score-stratum checks; no labels or fits."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
BASE = PROJECT / '.artifact_runtime/public_pool_trial_20261020'


def save(name, value):
    (ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    if (ROOT/'verification.json').exists():
        raise FileExistsError('Preserve verification.')
    pairs = [('sa_rmlp_v4aux_k10','sa_rmlp_v4aux_k10s2'),('gp_realmlp_pub_s1','gp_realmlp_pub_s2')]
    save('verification_protocol.json',{
        'pairs':pairs,
        'selection':'first pair is the strongest previously observed shrinkage; second is a separately named public-seed pair; not independent hypothesis selection',
        'method':'pandas average ranks, direct difference SD; compare cached covariance identity; 20 equal-size baseline-score percentile strata',
        'limitations':'strata are descriptive; different train/test score functions remain; this does not identify causal source or AUC gain',
        'labels_read':False,'model_fits':0,
    })
    members = list(csv.DictReader((PROJECT/'public_pool_sources_20261020.csv').open(encoding='utf-8-sig')))
    columns = {row['member']:i for i,row in enumerate(members)}
    with np.load(ROOT/'sample_indices.npz',allow_pickle=False) as z:
        ti,ei=z['train'],z['test']
    assert len(np.unique(ti))==ti.size and len(np.unique(ei))==ei.size
    train=np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    test=np.load(BASE/'test_logits.npy',mmap_mode='r',allow_pickle=False)
    with np.load(BASE/'selected_predictions.npz',allow_pickle=False) as z:
        scores={'train':z['oof_logit'],'test':z['test_logit']}
    descriptions=json.loads((ROOT/'all_pair_descriptions.json').read_text('utf-8'))
    output=[]
    for names in pairs:
        col=[columns[name] for name in names]
        diffs={};bands={}
        for sample,matrix,indices,kind in [('train_a',train,ti[0],'train'),('train_b',train,ti[1],'train'),('train_c',train,ti[2],'train'),('test_a',test,ei[0],'test'),('test_b',test,ei[1],'test')]:
            raw=np.asarray(matrix[np.ix_(indices,col)])
            ranks=pd.DataFrame(raw).rank(method='average').to_numpy()/(len(indices)+1)
            diffs[sample]=ranks[:,0]-ranks[:,1]
            score_rank=pd.Series(scores[kind][indices]).rank(method='first').to_numpy()-1
            group=(20*score_rank/len(indices)).astype(int)
            assert set(group)==set(range(20))
            bands[sample]=[float(diffs[sample][group==k].std(ddof=1)) for k in range(20)]
        ratio_a=float(diffs['test_a'].std(ddof=1)/diffs['train_a'].std(ddof=1))
        ratio_b=float(diffs['test_b'].std(ddof=1)/diffs['train_b'].std(ddof=1))
        cached=next(row for row in descriptions if row['members']==list(names))
        assert abs(ratio_a-cached['rank_disagreement_sd_ratio_a'])<2e-10
        assert abs(ratio_b-cached['rank_disagreement_sd_ratio_b'])<2e-10
        row={
            'members':names,
            'direct_rank_sd_ratio_a':ratio_a,'direct_rank_sd_ratio_b':ratio_b,
            'within_oof_null_ratio':float(diffs['train_c'].std(ddof=1)/diffs['train_a'].std(ddof=1)),
            'median_within_score_stratum_sd_ratio_a':float(np.median(np.array(bands['test_a'])/np.array(bands['train_a']))),
            'median_within_score_stratum_sd_ratio_b':float(np.median(np.array(bands['test_b'])/np.array(bands['train_b']))),
            'score_stratum_sd':bands,
            'covariance_identity_matches_direct_ranks':True,
        }
        output.append(row)
    current=json.loads((ROOT/'diagnostics.json').read_text('utf-8'))
    for name,expected in current['protected_after'].items():
        assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==expected
    result={'status':'passed_numeric_and_stratum_checks','pairs':output,'labels_read':False,'new_model_fits':0,'new_submissions':0,'protected_files_unchanged':True,'is_generalization_verification':False}
    save('verification.json',result)
    print(json.dumps({**result,'pairs':[{k:v for k,v in row.items() if k!='score_stratum_sd'} for row in output]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
