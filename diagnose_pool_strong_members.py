"""Post-screen descriptive sensitivity; never used to select a candidate."""
import numpy as np
import pandas as pd
from audit_pool_repairability import ROOT,RUN,BASE,read,write,check

def main():
    p=read(RUN/'protocol.json');check(p)
    manifest=read(BASE/'matrix_manifest.json');x=np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    out=[]
    for f in range(5):
        check(p);weights=np.load(BASE/f'public_only_fold{f}.npz',allow_pickle=False)['weights'][:-1]
        columns=np.argsort(weights)[-12:][::-1];assert (weights[columns]>0).all()
        z=np.load(RUN/f'fold{f}_pairs.npz',allow_pickle=False);a,c=z['positive'],z['negative']
        saved=np.load(RUN/f'fold{f}_audit_arrays.npz',allow_pickle=False);wrong=saved['baseline_wrong']
        any_correct=np.zeros(len(a),dtype=bool);any_wrong=any_correct.copy()
        for j in columns:
            d=x[a,j]-x[c,j];any_correct|=d>0;any_wrong|=d<0
        # Independent row-block reconstruction of these12 columns.
        wrong_correct=correct_wrong=0
        for s in range(0,len(a),8192):
            aa,cc=a[s:s+8192],c[s:s+8192]
            d=x[aa,:246][:,columns]-x[cc,:246][:,columns]
            np.testing.assert_array_equal(any_correct[s:s+len(aa)],(d>0).any(1))
            np.testing.assert_array_equal(any_wrong[s:s+len(aa)],(d<0).any(1))
            wrong_correct+=int(np.count_nonzero(wrong[s:s+len(aa)]&(d>0).any(1)))
            correct_wrong+=int(np.count_nonzero(~wrong[s:s+len(aa)]&(d<0).any(1)))
        r={'fold':f,'member_columns':columns.tolist(),'members':[manifest['columns'][j]['member'] for j in columns],
           'coefficients':weights[columns].tolist(),'wrong_pairs_corrected_by_any_top12':wrong_correct,
           'fraction_of_baseline_wrong':wrong_correct/int(wrong.sum()),'correct_pairs_broken_by_any_top12':correct_wrong,'two_computation_paths_match':True}
        out.append(r);print('STRONG_DIAGNOSTIC',f,r['fraction_of_baseline_wrong'],r['correct_pairs_broken_by_any_top12'],flush=True)
    write(RUN/'strong_member_diagnostic.json',{'folds':out,'diagnostic_defined_after_inner_result':True,'no_followup_fit_or_candidate_selection':True,'caveat':'Still a label-aware oracle; higher-weight membership does not make expert selection deployable.'})

if __name__=='__main__':main()
