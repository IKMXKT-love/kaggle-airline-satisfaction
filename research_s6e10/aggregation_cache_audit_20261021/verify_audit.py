"""Independent cache shape/scope checks. No labels, fits, or model loading."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
LOCAL=PROJECT/'.artifact_runtime/strong_correction_20261020/b1385337dbf19f77097b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'independent_verification.json'
    if out.exists():
        raise FileExistsError('Preserve previous verification.')
    audit=json.loads((ROOT/'audit_result.json').read_text('utf-8'))
    containers=[]
    for name in ['rmlp_v4aux_k10','rmlp_v4aux_k10s2']:
        path=ROOT/(name+'.npz')
        with zipfile.ZipFile(path) as archive:
            assert set(archive.namelist())=={'oof.npy','test.npy'}
            headers={}
            for key in archive.namelist():
                stream=io.BytesIO(archive.read(key))
                version=np.lib.format.read_magic(stream)
                if version==(1,0):
                    shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):
                    shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f'Unsupported NPY header version: {version}')
                assert len(shape)==1 and not dtype.hasobject
                headers[key]={'shape':list(shape),'dtype':str(dtype),'fortran':fortran}
        expected=next(x['source_sha256'] for x in audit['public_sachith_containers'] if x['member']==name)
        assert sha(path)==expected
        containers.append({'member':name,'zip_entries_and_npy_headers':headers,'fingerprint_matches':True})
    scopes=[]
    for outer in [0,1]:
        folder=LOCAL/f'outer_{outer}'
        with np.load(folder/'plan.npz',allow_pickle=False) as z:
            plan={key:z[key].copy() for key in z.files}
        for phase,a_name,q_name in [('selection','selection_fit','selection_stop'),('refit','outer_train','outer_eval')]:
            allowed,excluded=plan[a_name],plan[q_name]
            fit_count=np.zeros(699635,dtype=np.int8)
            heldout_count=np.zeros(699635,dtype=np.int8)
            saved_inner_models=[]
            for k in range(5):
                bundle=folder/phase/f'inner_{k}'
                with np.load(bundle/'indices.npz',allow_pickle=False) as z:
                    fit_count[z['fit']]+=1
                    heldout_count[z['query']]+=1
                for branch in ['lean_42','lean_20261019']:
                    with np.load(bundle/(branch+'.npz'),allow_pickle=False) as z:
                        assert not np.isin(z['index'],excluded).any()
                saved_inner_models.extend(str(p.relative_to(PROJECT)) for p in bundle.glob('lean*.joblib'))
            assert np.all(fit_count[allowed]==4)
            assert np.all(heldout_count[allowed]==1)
            assert not fit_count[excluded].any() and not heldout_count[excluded].any()
            scopes.append({'outer_fold':outer,'phase':phase,
                'fit_memberships_per_allowed_row':[int(fit_count[allowed].min()),int(fit_count[allowed].max())],
                'heldout_memberships_per_allowed_row':[int(heldout_count[allowed].min()),int(heldout_count[allowed].max())],
                'saved_inner_lean_checkpoints':saved_inner_models,
                'outer_or_stopping_rows_excluded_from_all_inner_fit_and_query_scopes':True})
    for name,expected in audit['protected_files'].items():
        assert sha(PROJECT/name)==expected
    result={'status':'passed','public_container_checks':containers,'local_scope_checks':scopes,
        'new_fits':0,'new_base_inferences':0,'raw_target_labels_read':False,'new_submissions':0,
        'best_and_current_records_unchanged':True,'is_generalization_test':False}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['public_container_checks','local_scope_checks']},indent=2))


if __name__=='__main__':
    main()
