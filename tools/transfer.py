"""Publish and independently verify a prepared, explicitly authorized public archive."""
import argparse,json,subprocess,time,threading,hashlib,concurrent.futures,os,sys,shlex
from pathlib import Path
from prepare import ROOT,OPS,STAGE,ASSETS,REPO,TAG,save,sha,progress,inventory

BRANCH='codex/archive-20261023'
LOCK=threading.Lock()

def run(args,cwd=None,check=True):
    if args[0]=='git' and 'push' in args:
        helper='!'+shlex.join([Path(sys.executable).as_posix(),'-B',(OPS/'git_credential_helper.py').as_posix()])
        args=['git','-c','credential.helper=','-c','credential.useHttpPath=true','-c','credential.helper='+helper,*args[1:]]
    env=os.environ.copy();env['GIT_TERMINAL_PROMPT']='0';env['GCM_INTERACTIVE']='never'
    r=subprocess.run(args,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
    if check and r.returncode:raise RuntimeError(f'{args[0:3]}: {r.stderr[-1800:]}')
    return r

def api(path):return json.loads(run(['gh','api',path]).stdout)

def setup():
    assert (OPS/'prepared.json').exists()
    if not (STAGE/'.git').exists():
        run(['git','init','-b',BRANCH],STAGE)
        run(['git','config','core.autocrlf','false'],STAGE)
        run(['git','config','user.name','IKMXKT-love'],STAGE)
        run(['git','config','user.email','IKMXKT-love@users.noreply.github.com'],STAGE)
        run(['git','add','--all'],STAGE)
        run(['git','commit','-m','Archive airline satisfaction methods, source, results and recovery manifests'],STAGE)
    remote=run(['gh','api',f'repos/{REPO}'],check=False)
    if remote.returncode:
        run(['gh','repo','create',REPO,'--public','--description','Kaggle S6E10 airline satisfaction: reproducible methods, source, results and complete experiment archives'])
        save(OPS/'remote_created.json',api(f'repos/{REPO}'))
    else:
        assert (OPS/'remote_created.json').exists(),'Existing repository not created by this operation; inspect before proceeding'
    remote=api(f'repos/{REPO}');assert remote['private'] is False and remote['full_name']==REPO
    if run(['git','remote','get-url','origin'],STAGE,False).returncode:run(['git','remote','add','origin',f'https://github.com/{REPO}.git'],STAGE)
    assert run(['git','remote','get-url','origin'],STAGE).stdout.strip()==f'https://github.com/{REPO}.git'
    run(['git','push','-u','origin',BRANCH],STAGE)
    run(['gh','repo','edit',REPO,'--default-branch',BRANCH])
    info=run(['gh','release','view',TAG,'--repo',REPO,'--json','id'],check=False)
    if info.returncode:
        body=OPS/'release-notes.md'
        body.write_text('''Complete current-project snapshot, with source and readable methodology in the repository.

All volumes are independent ZIP files. Download every volume, archive_manifest.json and restore.py, then use the documented recovery command. Original datasets, submissions, checkpoints, caches, dependency files and original Git metadata are retained. External symlink targets are recorded without traversing them.

Highest user-reported public AUC: 0.96187, primarily a reproduction of public base-prediction stacking. No claim of independently originating the upstream models or achieving 0.962. See repository attribution and validation limitations.

Per-file and per-volume SHA-256 manifests accompany the archive. The local project is retained. Third-party licenses and source metadata remain applicable.
''',encoding='utf-8')
        run(['gh','release','create',TAG,'--repo',REPO,'--target',BRANCH,'--draft','--title','Complete airline satisfaction research archive — 2026-10-23','--notes-file',str(body)])
    save(OPS/'remote_repository.json',api(f'repos/{REPO}'))
    progress('repository_pushed',repository=REPO,branch=BRANCH,commit=run(['git','rev-parse','HEAD'],STAGE).stdout.strip())

def release():
    matches=[r for r in api(f'repos/{REPO}/releases?per_page=100') if r['tag_name']==TAG]
    assert len(matches)==1,('Release lookup',TAG,len(matches))
    return api(f"repos/{REPO}/releases/{matches[0]['id']}")

def upload():
    assets=sorted(p for p in ASSETS.iterdir() if p.is_file())
    plan=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in assets]
    save(OPS/'asset_plan.json',plan)
    completed={}
    if (OPS/'uploaded_assets.json').exists():completed=json.loads((OPS/'uploaded_assets.json').read_text('utf-8'))
    progress('uploading',assets=len(plan),bytes=sum(x['bytes'] for x in plan))
    def one(item):
        for attempt in range(3):
            meta=release();existing=[a for a in meta['assets'] if a['name']==item['name'] and a['state']=='uploaded']
            if existing:
                obj=existing[0];assert obj['size']==item['bytes'],(item['name'],'remote size mismatch')
                digest=obj.get('digest')
                if digest:assert digest=='sha256:'+item['sha256'],(item['name'],'server digest mismatch')
                with LOCK:
                    completed[item['name']]={**item,'id':obj['id'],'server_digest':digest,'url':obj['browser_download_url']}
                    save(OPS/'uploaded_assets.json',completed)
                    progress('uploading',completed=len(completed),total=len(plan),latest=item['name'])
                return
            with LOCK:progress('uploading',active=item['name'],attempt=attempt+1,completed=len(completed),total=len(plan))
            result=run(['gh','release','upload',TAG,str(ASSETS/item['name']),'--repo',REPO],check=False)
            if result.returncode:
                with LOCK:save(OPS/('upload-error-'+item['name']+'.json'),{'attempt':attempt+1,'stderr':result.stderr[-1600:]})
                time.sleep(2)
        # One final metadata check handles successful transfers followed by client errors.
        found=[a for a in release()['assets'] if a['name']==item['name'] and a['state']=='uploaded']
        assert found and found[0]['size']==item['bytes'],item['name']
        obj=found[0]
        if obj.get('digest'):assert obj['digest']=='sha256:'+item['sha256']
        with LOCK:
            completed[item['name']]={**item,'id':obj['id'],'server_digest':obj.get('digest'),'url':obj['browser_download_url']};save(OPS/'uploaded_assets.json',completed)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for done in pool.map(one,plan):pass
    assert len(completed)==len(plan)
    progress('all_assets_uploaded',count=len(plan))

def verify():
    plan=json.loads((OPS/'asset_plan.json').read_text('utf-8'))
    uploaded=json.loads((OPS/'uploaded_assets.json').read_text('utf-8'))
    existing={}
    if (OPS/'remote_roundtrip.json').exists():existing=json.loads((OPS/'remote_roundtrip.json').read_text('utf-8'))
    def one(item):
        if item['name'] in existing and existing[item['name']]['sha256']==item['sha256']:return
        aid=uploaded[item['name']]['id'];h=hashlib.sha256();count=0
        with LOCK:progress('remote_download_verification',active=item['name'],verified=len(existing),total=len(plan))
        proc=subprocess.Popen(['gh','api','-H','Accept: application/octet-stream',f'repos/{REPO}/releases/assets/{aid}'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        for block in iter(lambda:proc.stdout.read(4*1024**2),b''):h.update(block);count+=len(block)
        error=proc.stderr.read().decode('utf-8','replace');code=proc.wait()
        assert code==0,(item['name'],error[-1000:])
        assert count==item['bytes'] and h.hexdigest()==item['sha256'],(item['name'],'roundtrip mismatch',count)
        with LOCK:
            existing[item['name']]={'id':aid,'bytes':count,'sha256':h.hexdigest(),'remote_roundtrip_verified':True}
            save(OPS/'remote_roundtrip.json',existing);progress('remote_download_verification',verified=len(existing),total=len(plan),latest=item['name'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(one,plan):pass
    checkout=OPS/'remote-checkout'
    if not checkout.exists():run(['git','-c','core.autocrlf=false','clone','--depth','1','--single-branch','--branch',BRANCH,f'https://github.com/{REPO}.git',str(checkout)])
    expected=json.loads((OPS/'readable_manifest.json').read_text('utf-8'))
    for item in expected:
        p=checkout/item['path'];assert p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
    staged_set={x['path'] for x in expected}
    actual_set={p.relative_to(checkout).as_posix() for p in checkout.rglob('*') if p.is_file() and '.git' not in p.relative_to(checkout).parts}
    assert actual_set==staged_set,('remote file set mismatch',len(actual_set),len(staged_set))
    progress('source_snapshot_reverification',files=len(json.loads((ASSETS/'archive_manifest.json').read_text('utf-8'))['files']))
    manifest=json.loads((ASSETS/'archive_manifest.json').read_text('utf-8'))
    actual,dirs,links=inventory();assert {x['path'] for x in actual}=={x['path'] for x in manifest['files']}
    assert dirs==manifest['directories'] and links==manifest['links']
    for i,item in enumerate(manifest['files']):
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
        if i%1000==0:progress('source_snapshot_reverification',done=i+1,total=len(manifest['files']))
    run(['gh','release','edit',TAG,'--repo',REPO,'--draft=false'])
    remote=api(f'repos/{REPO}');rel=release();assert remote['private'] is False and rel['draft'] is False
    commit=run(['git','rev-parse','HEAD'],checkout).stdout.strip()
    assert commit==run(['git','rev-parse','HEAD'],STAGE).stdout.strip()
    receipt={'repository':remote['html_url'],'public':True,'branch':BRANCH,'commit':commit,'release':rel['html_url'],'published':True,'source_files':manifest['total_files'],'source_bytes':manifest['total_bytes'],'link_records':len(links),'readable_files_verified':len(expected),'assets_verified':len(existing),'remote_roundtrip_verified':True,'original_snapshot_reverified':True,'local_project_retained':True,'manifest_sha256':sha(ASSETS/'archive_manifest.json'),'completed_epoch':time.time()}
    save(OPS/'receipt.json',receipt)
    progress('complete',**receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['setup','upload','verify']);args=p.parse_args()
    {'setup':setup,'upload':upload,'verify':verify}[args.stage]()
