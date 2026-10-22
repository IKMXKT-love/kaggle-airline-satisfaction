"""Prepare readable navigation and validate links without changing original files."""
import json,re,os,hashlib,ast,shutil,importlib.metadata
from pathlib import Path
from prepare import ROOT,OPS,STAGE,save,sha

assert (OPS/'prepared.json').exists()
shutil.copy2(OPS/'polish_repository.py',STAGE/'tools/polish_repository.py')
shutil.copy2(OPS/'restore_smoke_verification.json',STAGE/'archive/restore_smoke_verification.json')
for label,folder in [('tabicl-runtime','tabicl_runtime'),('tabicl-finetune-deps','tabicl_finetune_deps')]:
    packages=sorted((d.metadata.get('Name','unknown'),d.version) for d in importlib.metadata.distributions(path=[str(ROOT/'.artifact_runtime'/folder)]))
    save(STAGE/'environment'/('packages-'+label+'.json'),packages)
manifest=json.loads((OPS/'assets/archive_manifest.json').read_text('utf-8'))
original_paths={x['path'] for x in manifest['files']}
changed=0;archived_links=0;missing=[]
for p in STAGE.rglob('*.md'):
    rel=p.relative_to(STAGE)
    text=p.read_text('utf-8-sig')
    def fix(m):
        global archived_links
        label,target=m.group(1),m.group(2).strip('<>')
        if not target or target.startswith(('#','http:','https:','mailto:','codex:','plugin:','app:')):return m.group(0)
        target=target.split('#',1)[0]
        if not target:return m.group(0)
        candidate=(p.parent/target).resolve()
        if candidate.exists() and candidate.is_relative_to(STAGE):return m.group(0)
        if rel.as_posix()=='docs/LEGACY_README.md':
            candidate=(STAGE/target).resolve()
            if candidate.exists() and candidate.is_relative_to(STAGE):
                return f'[{label}]({os.path.relpath(candidate,p.parent).replace(chr(92),chr(47))})'
        if candidate.is_relative_to(STAGE):
            archive_path=candidate.relative_to(STAGE).as_posix()
            if archive_path in original_paths:
                archived_links+=1;return f'{label}（完整归档内：`{archive_path}`）'
        missing.append({'document':rel.as_posix(),'target':target})
        return m.group(0)
    new=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',fix,text)
    if new!=text:p.write_text(new,encoding='utf-8');changed+=1
sources=[]
for p in sorted(STAGE.rglob('*.py')):
    rel=p.relative_to(STAGE).as_posix()
    if rel.startswith('research_s6e10/') or rel.startswith('.artifact_runtime/'):continue
    ast.parse(p.read_text('utf-8-sig'));sources.append(rel)
index=['# 源码文件索引','','这里列出项目训练、特征、报告、核验和归档工具源码。原始路径保持不变。机制分类见[METHODS.md](METHODS.md)。','']
for name in sources:index.append(f'- [{name}](../{name})')
(STAGE/'docs/SOURCE_FILES.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
save(STAGE/'archive/documentation_checks.json',{'python_project_files_parsed':len(sources),'markdown_documents_adjusted':changed,'archive_only_links_labeled':archived_links,'historical_unresolved_links':missing,'note':'Unresolved original historical citations are retained as evidence; not interpreted as runnable dependencies.'})
patterns={'github_token':re.compile(r'(?<![A-Za-z0-9_-])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),'openai_key':re.compile(r'(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}'),'aws_access_key':re.compile(r'(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])')}
hits=[]
for p in STAGE.rglob('*'):
    if p.is_file() and p.suffix in {'.py','.md','.json','.txt','.log','.ps1','.cmd','.toml'}:
        text=p.read_text('utf-8-sig',errors='ignore')
        for rule,pattern in patterns.items():
            if pattern.search(text):hits.append({'path':p.relative_to(STAGE).as_posix(),'rule':rule})
assert not hits,hits
save(STAGE/'archive/credential_scan.json',{'readable_candidate_patterns_checked':list(patterns),'matching_candidates':0,'values_never_logged':True,'scope':'Readable staged project files; third-party dependency binaries remain archival materials.'})
out=[]
for p in sorted(STAGE.rglob('*')):
    if p.is_file() and '.git' not in p.relative_to(STAGE).parts:out.append({'path':p.relative_to(STAGE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
save(OPS/'readable_manifest.json',out)
print(json.dumps({'readable_files':len(out),'readable_bytes':sum(x['bytes'] for x in out),'python_sources_parsed':len(sources),'unresolved_historical_links':len(missing),'credential_pattern_matches':len(hits)},indent=2))
