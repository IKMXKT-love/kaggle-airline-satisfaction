"""Pre/post guards for the explicitly approved, fixed A-file manifest."""
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
REVIEW=ROOT/'cleanup_review_20261022'
MANIFEST=REVIEW/'proposed_deletion_manifest.json'
EXPECTED=Path(r'C:\Users\ASUS\Documents\ChatGPT\kaggle_first')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for b in iter(lambda:s.read(2**20),b''):h.update(b)
    return h.hexdigest()

def inventory():
    files={};links=[];stack=[ROOT]
    while stack:
        folder=stack.pop()
        with os.scandir(folder) as entries:
            for e in entries:
                st=e.stat(follow_symlinks=False);rel=str(Path(e.path).relative_to(ROOT))
                if e.is_symlink() or getattr(st,'st_file_attributes',0)&0x400:
                    links.append(rel);continue
                if e.is_dir(follow_symlinks=False):stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):files[rel]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    return files,links

def planned():
    assert ROOT.resolve()==EXPECTED.resolve()
    m=read(MANIFEST);assert Path(m['root']).resolve()==ROOT.resolve()
    allowed={k:v for k,v in m['groups'].items() if k.startswith('A')}
    assert set(allowed)=={'A1_installer','A2_epoch_snapshots_keep_best','A3_discarded_encoding_arrays','A4_reproducible_pair_samples','A5_python_bytecode'}
    paths={};ancestors=set()
    for group,items in allowed.items():
        assert len(items)==m['group_summaries'][group]['files']
        for item in items:
            p=Path(item['path']);rel=Path(item['relative_path'])
            assert not rel.is_absolute() and '..' not in rel.parts
            assert p==ROOT/rel and p.resolve().is_relative_to(ROOT.resolve()) and p!=ROOT
            assert str(rel) not in paths
            paths[str(rel)]={**item,'group':group}
            q=p
            while q!=ROOT.parent:ancestors.add(q);q=q.parent
    return paths,ancestors

def preflight():
    import psutil
    paths,ancestors=planned()
    for p in ancestors:
        st=p.lstat()
        assert not stat.S_ISLNK(st.st_mode) and not getattr(st,'st_file_attributes',0)&0x400,str(p)
    for rel,item in paths.items():
        p=ROOT/rel;st=p.stat()
        assert p.is_file() and st.st_size==item['expected_bytes'] and st.st_mtime_ns==item['mtime_ns_at_inventory'],rel
    processes=[]
    for proc in psutil.process_iter(['pid','name','cmdline','cwd']):
        try:
            info=proc.info
            if info['pid']==os.getpid():continue
            args=info.get('cmdline') or [];cwd=info.get('cwd') or ''
            if str(ROOT).casefold() in cwd.casefold() or any(str(ROOT).casefold() in a.casefold() for a in args):
                processes.append({'pid':info['pid'],'name':info['name']})
                assert 'python' not in info['name'].lower(),'Other project Python process active'
        except (psutil.AccessDenied,psutil.NoSuchProcess):continue
    old=read(REVIEW/'preservation_check.json')
    for rel,h in old['protected_files'].items():assert sha(ROOT/rel)==h,rel
    files,links=inventory()
    preserve={rel:meta for rel,meta in files.items() if rel not in paths and not rel.startswith('cleanup_review_20261022\\')}
    # Hash all retained project data, models, source and reports. Installed
    # package files and Git objects are checked by exact file inventory instead.
    hashes={rel:sha(ROOT/rel) for rel in preserve if not rel.startswith(('.venv\\','.git\\'))}
    before={'manifest_sha256':sha(MANIFEST),'approved_scope':'A only, user confirmed','files_to_remove':len(paths),'bytes_to_remove':sum(x['expected_bytes'] for x in paths.values()),
            'free_bytes_before':shutil.disk_usage(ROOT).free,'preserved_inventory':preserve,'preserved_sha256':hashes,'skipped_reparse_points':links,'project_process_snapshot':processes,'started_epoch':time.time()}
    write(REVIEW/'A_preflight.json',before)
    print(json.dumps({'preflight_passed':True,'files_to_remove':len(paths),'GiB_to_remove':before['bytes_to_remove']/2**30,'preserved_file_count':len(preserve),'retained_files_hashed':len(hashes),'other_project_python_processes':False}),flush=True)

def verify():
    before=read(REVIEW/'A_preflight.json');assert sha(MANIFEST)==before['manifest_sha256']
    paths,_=planned();after,links=inventory();absent=[rel for rel in paths if not (ROOT/rel).exists()];remaining=[rel for rel in paths if (ROOT/rel).exists()]
    missing=[rel for rel in before['preserved_inventory'] if rel not in after]
    changed=[rel for rel,meta in before['preserved_inventory'].items() if rel in after and after[rel]!=meta]
    hash_changes=[rel for rel,h in before['preserved_sha256'].items() if rel not in after or sha(ROOT/rel)!=h]
    assert not missing and not changed and not hash_changes,{'missing':missing,'changed':changed,'hash_changes':hash_changes}
    assert sorted(links)==sorted(before['skipped_reparse_points'])
    bgroup=read(MANIFEST)['groups']['B_optional_source_downloads']
    assert all(str(Path(i['relative_path'])) in after for i in bgroup)
    free_after=shutil.disk_usage(ROOT).free
    receipt={'completed':not remaining,'scope':'A only','deleted_files':len(absent),'deleted_logical_bytes':sum(paths[rel]['expected_bytes'] for rel in absent),'remaining_approved_files':remaining,
             'preserved_file_count':len(before['preserved_inventory']),'preserved_hashed_files':len(before['preserved_sha256']),'unexpected_missing_files':missing,'unexpected_changed_files':changed,'preserved_hash_changes':hash_changes,
             'B_download_files_preserved':len(bgroup),'reparse_points_preserved':True,'free_bytes_before':before['free_bytes_before'],'free_bytes_after':free_after,'free_bytes_delta':free_after-before['free_bytes_before'],'finished_epoch':time.time(),
             'note':'Free-space delta is drive-wide and can include concurrent system activity. Logical deleted byte count is from the authorized manifest.'}
    write(REVIEW/'A_cleanup_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
    if remaining:sys.exit(2)

if __name__=='__main__':
    if sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='verify':verify()
    else:raise ValueError(sys.argv[1])
