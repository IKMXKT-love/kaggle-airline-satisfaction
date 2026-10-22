"""Current246-column native-fold pair audit; no deployable oracle claims."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'.artifact_runtime/pool_repairability_20261022'
BASE=ROOT/'.artifact_runtime/public_pool_trial_20261020'
FOLDS=ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy'
sys.stdout.reconfigure(encoding='utf-8')

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as s:
        for b in iter(lambda:s.read(2**20),b''):h.update(b)
    return h.hexdigest()
def check(p):
    if time.time()>=p['started_epoch']+p['screen_budget_seconds']:raise TimeoutError('Cumulative screen budget exhausted')

def main():
    p=read(RUN/'protocol.json');check(p)
    start=time.time()
    tr=pd.read_csv(ROOT/'train.csv',usecols=['id','satisfaction']);y=tr.satisfaction.to_numpy(dtype=int)
    folds=np.load(FOLDS,allow_pickle=False);x=np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    with np.load(BASE/'selected_predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['id'],tr.id);b=z['oof_logit'].copy()
    inputs={'train.csv':sha(ROOT/'train.csv'),'matrix':sha(BASE/'train_logits.npy'),'manifest':sha(BASE/'matrix_manifest.json'),'folds':sha(FOLDS)}
    write(RUN/'inputs.json',inputs)
    members=[v['member'] for v in read(BASE/'matrix_manifest.json')['columns'][:246]]
    out=[]
    for f in range(5):
        check(p);rng=np.random.default_rng(p['pair_seed']+f);idx=np.flatnonzero(folds==f)
        a=rng.choice(idx[y[idx]==1],p['pairs_per_native_fold']);c=rng.choice(idx[y[idx]==0],p['pairs_per_native_fold'])
        np.savez(RUN/f'fold{f}_pairs.npz',positive=a,negative=c)
        bm=b[a]-b[c];wrong=bm<0;right=bm>0;tie=bm==0
        any_correct=np.zeros(len(a),dtype=bool);any_wrong=any_correct.copy();any_tied=any_correct.copy()
        recovered=np.zeros(246,dtype='int64');damaged=recovered.copy();unanimous_wrong=wrong.copy()
        for j in range(246):
            d=x[a,j]-x[c,j];ok=d>0;bad=d<0
            any_correct|=ok;any_wrong|=bad;any_tied|=d==0
            recovered[j]=np.count_nonzero(wrong&ok);damaged[j]=np.count_nonzero(right&bad)
            unanimous_wrong&=bad
        r={'fold':f,'rows':len(idx),'exact_baseline_auc':float(roc_auc_score(y[idx],b[idx])),'sampled_baseline_auc':float((right.sum()+.5*tie.sum())/len(a)),
           'sampled_pairs':len(a),'baseline_wrong_pairs':int(wrong.sum()),'baseline_tied_pairs':int(tie.sum()),
           'all246_strictly_wrong_pairs':int(unanimous_wrong.sum()),'all246_wrong_fraction_of_baseline_wrong':float(unanimous_wrong.sum()/wrong.sum()),
           'baseline_wrong_some_member_correct':int(np.count_nonzero(wrong&any_correct)),
           'baseline_wrong_no_member_correct':int(np.count_nonzero(wrong&~any_correct)),
           'baseline_correct_some_member_wrong':int(np.count_nonzero(right&any_wrong)),
           'some_member_correct_fraction_of_baseline_wrong':float(np.count_nonzero(wrong&any_correct)/wrong.sum())}
        out.append(r)
        np.savez(RUN/f'fold{f}_audit_arrays.npz',baseline_wrong=wrong,any_member_correct=any_correct,any_member_wrong=any_wrong,all_members_wrong=unanimous_wrong,recovered_by_member=recovered,damaged_by_member=damaged)
        print('PAIR_AUDIT',json.dumps(r),flush=True)
    table=pd.DataFrame({'member':members})
    for f in range(5):
        z=np.load(RUN/f'fold{f}_audit_arrays.npz',allow_pickle=False)
        table[f'corrects_wrong_pairs_fold{f}']=z['recovered_by_member'];table[f'breaks_correct_pairs_fold{f}']=z['damaged_by_member']
    table.to_csv(RUN/'per_member_pair_counts.csv',index=False,encoding='utf-8')
    write(RUN/'pair_audit.json',{'folds':out,'pair_audit_seconds':time.time()-start,'elapsed_seconds':time.time()-p['started_epoch'],'caveat':'Label-aware descriptive counts; not a deployable oracle, training gate or guarantee. Disagreements between classifiers are not new input information.'})

if __name__=='__main__':main()
