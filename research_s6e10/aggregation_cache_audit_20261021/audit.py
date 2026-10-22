"""Check cache contents and model-fit/query scopes without loading labels/models."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
POOL=PROJECT/'.artifact_runtime/public_pool_trial_20261020'
LOCAL=PROJECT/'.artifact_runtime/strong_correction_20261020/b1385337dbf19f77097b'


def read(path):
    return json.loads(path.read_text('utf-8'))


def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def load_prediction(path,expected):
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={'index','prediction'}
        np.testing.assert_array_equal(z['index'],expected)
        p=z['prediction'].copy()
    assert p.shape==(len(expected),) and np.isfinite(p).all()
    assert np.all((p>=0)&(p<=1))
    return p


def main():
    if (ROOT/'audit_result.json').exists():
        raise FileExistsError('Preserve completed audit.')
    protected={name:sha(PROJECT/name) for name in ['submission_public_pool_20261020.csv','CURRENT_BEST_SUBMISSION.json','CURRENT_BEST_RESULT.md','CURRENT_RECOMMENDED_CANDIDATE.json']}
    meta=read(ROOT/'members.json')
    public=[]
    for name in ['rmlp_v4aux_k10','rmlp_v4aux_k10s2']:
        path=ROOT/(name+'.npz')
        with np.load(path,allow_pickle=False) as z:
            arrays={key:(list(z[key].shape),str(z[key].dtype)) for key in z.files}
            assert set(z.files)=={'oof','test'}
            maximum={}
            for kind in ['oof','test']:
                frozen=np.load(POOL/'downloads/c124'/f'{kind}_sa_{name}.npy',allow_pickle=False)
                np.testing.assert_array_equal(z[kind],frozen)
                maximum[kind]=float(np.max(np.abs(z[kind]-frozen)))
        public.append(dict(member=name,source_arrays=arrays,source_sha256=sha(path),
                           exact_match_to_current_pool=maximum,author_metadata=meta[name],
                           per_fold_test_matrix_present=False,repeated_heldout_predictions_present=False,
                           fold_ids_or_fit_indices_present=False))
    gp_names=['gp_realmlp_pub','gp_realmlp_pub_s1','gp_realmlp_pub_s2']
    gp=[]
    for name in gp_names:
        row={'member':name,'files':{}}
        for kind in ['oof','test']:
            path=POOL/'downloads/c124'/f'{kind}_{name}.npy'
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            assert a.ndim==1
            row['files'][kind]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':sha(path)}
        row['per_fit_or_repeat_axis_present']=False
        gp.append(row)
    protocol=read(LOCAL/'protocol.json')
    source_matches={name:sha(PROJECT/name)==protocol['hashes'][name] for name in ['experiment_correction.py','correction_support.py','experiment_realmlp_factorial.py']}
    assert all(source_matches.values())
    local=[]
    for outer in [0,1]:
        folder=LOCAL/f'outer_{outer}'
        with np.load(folder/'plan.npz',allow_pickle=False) as z:
            plan={key:z[key].copy() for key in z.files}
        assert not np.intersect1d(plan['outer_train'],plan['outer_eval']).size
        for phase,allowed_name,query_name in [('selection','selection_fit','selection_stop'),('refit','outer_train','outer_eval')]:
            allowed,full_query=plan[allowed_name],plan[query_name]
            p_oof=np.empty((len(allowed),2));coverage=np.zeros((len(allowed),2),dtype=np.int8)
            model_scopes=[]
            for k in range(5):
                bundle=folder/phase/f'inner_{k}'
                with np.load(bundle/'indices.npz',allow_pickle=False) as z:
                    fit,query=z['fit'].copy(),z['query'].copy()
                assert not np.intersect1d(fit,query).size
                assert np.array_equal(np.sort(np.r_[fit,query]),allowed)
                assert not np.intersect1d(fit,full_query).size
                assert not np.intersect1d(query,full_query).size
                positions=np.searchsorted(allowed,query)
                np.testing.assert_array_equal(allowed[positions],query)
                for j,name in enumerate(['lean_42','lean_20261019']):
                    p_oof[positions,j]=load_prediction(bundle/(name+'.npz'),query)
                    coverage[positions,j]+=1
                    detail=read(bundle/(name+'.json'))
                    assert detail['train_rows']==len(fit) and detail['query_rows']==len(query)
                model_scopes.append({'fold':k,'fit_rows':len(fit),'query_rows':len(query),'no_saved_full_query_prediction':True,'scope_file_sha256':sha(bundle/'indices.npz')})
            assert np.all(coverage==1)
            full=folder/phase/'full_fit'
            with np.load(full/'indices.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['fit'],allowed)
                np.testing.assert_array_equal(z['query'],full_query)
            p_full=np.column_stack([load_prediction(full/(name+'.npz'),full_query) for name in ['lean_42','lean_20261019']])
            rng=np.random.default_rng(202610212+outer*10+(phase=='refit'))
            oi=rng.choice(len(allowed),20000,replace=False);qi=rng.choice(len(full_query),20000,replace=False)
            ro=rankdata(p_oof[oi],axis=0,method='average')/20001
            rq=rankdata(p_full[qi],axis=0,method='average')/20001
            ratio=float((rq[:,0]-rq[:,1]).std(ddof=1)/(ro[:,0]-ro[:,1]).std(ddof=1))
            local.append({
                'outer_fold':outer,'phase':phase,'fit_rows':len(allowed),'query_rows':len(full_query),
                'oof_coverage_per_row_per_seed':[int(coverage.min()),int(coverage.max())],
                'valid_independent_seed_views_per_scored_row':2,
                'independent_inner_fold_predictions_per_training_row_per_seed':1,
                'model_fit_excludes_full_query':True,
                'full_query_predictions_are_full_fit_not_inner_fold_average':True,
                'inner_models_saved_full_query_predictions':False,
                'small_label_free_rank_disagreement_ratio_full_query_to_oof':ratio,
                'inner_model_scope_audit':model_scopes,
            })
    protected_after={name:sha(PROJECT/name) for name in protected}
    assert protected_after==protected
    result={
        'date':'2026-10-21','public_sachith_containers':public,'public_gp_cached_vectors':gp,
        'local_scope_audits':local,'local_source_fingerprints_match_frozen_protocol':source_matches,
        'validation_options':{
            'reproduce_public_5_or_10_fold_averaging_on_labeled_heldout_rows':False,
            'local_two_seed_average_sensitivity_comparison':True,
            'local_two_seed_comparison_is_same_question_as_public_fold_averaging':False,
            'cheap_strict_current_246_pool_comparison_supported':False,
        },
        'decision':'Do not fit a surrogate experiment or refit public models. Source containers lack the required repeated held-out views; local two-seed caches support a narrower question only.',
        'limits':['not an AUC test','public upstream fit boundaries remain author-declared','local scoring partitions have been historically inspected','negative cache audit does not prove aggregation correction cannot work'],
        'new_fits':0,'new_base_inferences':0,'labels_read':False,'new_submissions':0,
        'protected_files':protected_after,
    }
    save('audit_result.json',result)
    print(json.dumps({'validation_options':result['validation_options'],'local_ratios':[(x['outer_fold'],x['phase'],x['small_label_free_rank_disagreement_ratio_full_query_to_oof']) for x in local],'new_fits':0,'protected_files_unchanged':True},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
