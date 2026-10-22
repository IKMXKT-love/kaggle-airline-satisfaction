import os, sys, json, time, hashlib, shutil, zipfile, re, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(r'C:\Users\ASUS\Documents\ChatGPT\kaggle_first')
OPS=Path(__file__).resolve().parent
STAGE=OPS/'repository'
ASSETS=OPS/'assets'
REPO='IKMXKT-love/kaggle-airline-satisfaction'
TAG='archive-20261023'
TARGET=800*1024**2

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def progress(stage,**kw):
    d={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),**kw}
    save(OPS/'progress.json',d);print(json.dumps(d,ensure_ascii=False),flush=True)

def inventory():
    files=[];dirs=[];links=[]
    for parent,sub,leaf in os.walk(ROOT,followlinks=False):
        parent=Path(parent)
        for name in list(sub):
            p=parent/name;rel=p.relative_to(ROOT).as_posix()
            if p.is_symlink() or p.is_junction():
                links.append({'path':rel,'target':os.readlink(p),'directory':True});sub.remove(name)
            else:dirs.append(rel)
        for name in leaf:
            p=parent/name;rel=p.relative_to(ROOT).as_posix()
            if p.is_symlink():links.append({'path':rel,'target':os.readlink(p),'directory':False});continue
            st=p.stat();files.append({'path':rel,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
    return sorted(files,key=lambda x:x['path']),sorted(dirs),links

def main():
    sys.stdout.reconfigure(encoding='utf-8');ASSETS.mkdir(exist_ok=True)
    assert not (OPS/'prepared.json').exists(),'Already prepared; use saved artifacts'
    files,dirs,links=inventory();total=sum(x['bytes'] for x in files)
    save(OPS/'inventory_before.json',{'files':files,'directories':dirs,'links':links,'bytes':total})
    progress('archiving',files=len(files),bytes=total,links=len(links))
    volume=None;vname=None;used=0;count=0;volumes=[]
    def close():
        if volume is not None:
            volume.close();p=ASSETS/vname
            volumes.append({'name':vname,'bytes':p.stat().st_size,'sha256':sha(p),'entries':count})
            save(OPS/'volumes-progress.json',volumes)
    archived=0
    for i,item in enumerate(files):
        if volume is None or (count and used+item['bytes']>TARGET):
            close();vname=f'kaggle-airline-satisfaction-volume-{len(volumes)+1:03d}.zip'
            volume=zipfile.ZipFile(ASSETS/vname,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True,strict_timestamps=False)
            used=0;count=0
        p=ROOT/item['path'];before=p.stat();h=hashlib.sha256()
        zi=zipfile.ZipInfo.from_file(p,item['path'],strict_timestamps=False);zi.compress_type=zipfile.ZIP_DEFLATED;zi._compresslevel=1
        with p.open('rb') as source,volume.open(zi,'w',force_zip64=True) as dest:
            for b in iter(lambda:source.read(4*1024**2),b''):h.update(b);dest.write(b)
        after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),(item['path'],'changed during archive')
        assert before.st_size==item['bytes'] and before.st_mtime_ns==item['mtime_ns'],item['path']
        item.update(sha256=h.hexdigest(),archive=vname);used+=item['bytes'];count+=1;archived+=item['bytes']
        if i%300==0:progress('archiving',done_files=i+1,total_files=len(files),done_bytes=archived,total_bytes=total,volume=vname)
    close()
    assert all(x['bytes']<1900*1024**2 for x in volumes)
    manifest={'schema':1,'source':str(ROOT),'created_utc':datetime.now(timezone.utc).isoformat(),'repository':REPO,'release':TAG,'files':files,'directories':dirs,'links':links,'total_files':len(files),'total_bytes':total,'volumes':volumes,'link_policy':'Links are recorded without traversing external targets. Restore does not recreate external links automatically.'}
    save(ASSETS/'archive_manifest.json',manifest)
    progress('readable_repository',archive_volumes=len(volumes),compressed_bytes=sum(x['bytes'] for x in volumes))
    STAGE.mkdir(exist_ok=True)
    no_git_roots={'.git','.venv','.idea','__pycache__'}
    no_git_runtime={'node_modules','tabicl_runtime','tabicl_finetune_deps'}
    extensions={'.py','.md','.txt','.json','.toml','.yml','.yaml','.ps1','.cmd','.csv','.png'}
    copied=[]
    for item in files:
        rel=Path(item['path']);parts=rel.parts
        if parts[0] in no_git_roots:continue
        if len(parts)>1 and parts[0]=='.artifact_runtime' and parts[1] in no_git_runtime:continue
        if rel.suffix.lower() not in extensions or item['bytes']>2*1024**2:continue
        if len(parts)==1 and rel.suffix=='.csv' and rel.name!='public_pool_sources_20261020.csv':continue
        dest=STAGE/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest);copied.append(item['path'])
    (STAGE/'docs').mkdir(exist_ok=True);(STAGE/'archive').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'README.md',STAGE/'docs/LEGACY_README.md')
    shutil.copy2(ASSETS/'archive_manifest.json',STAGE/'archive/manifest.json')
    save(STAGE/'archive/volumes.json',{'repository':REPO,'tag':TAG,'volumes':volumes})
    for name in ['prepare.py','transfer.py','restore.py']:
        if (OPS/name).exists():
            (STAGE/'tools').mkdir(exist_ok=True);shutil.copy2(OPS/name,STAGE/'tools'/name)
    readme='''# Kaggle Airline Satisfaction Research Archive

Playground Series S6E10 航空乘客满意度预测：方法、源码、完整实验记录与可恢复归档。

**用户反馈的最高公榜 AUC：0.96187。该结果主要来自复现公开预测池及其融合方案，并非本项目独立训练出的原创最高分。** 我们取得22位作者的246个基础预测，沿用logit＋L2逻辑回归（C=0.003）重新拟合与验证；本地七个基础分支没有稳定增益，最终未加入。

| 用户反馈公榜 AUC | 方法 |
|---:|---|
| 0.96187 | 公开246预测池＋L2逻辑回归融合 |
| 0.96114 | v4＋双种子十折精简RealMLP |
| 0.96104 | v4＋全量训练精简RealMLP |
| 0.96082 | v4：v3＋LightGBM |
| 0.96074 | v3：航程画像、辅助评分预测及树模型/RealMLP融合 |
| 0.96036 | 原始数据先验、目标编码及XGBoost/CatBoost融合 |

成绩来自用户反馈，没有独立查询Kaggle提交记录。本地OOF、源域成绩与公榜分开记录；不宣称达到0.962或保证私榜泛化。

## 从这里阅读

- [完整方法与得分登记](EXPERIMENT_SCORE_CATALOG_20261023.md)：公榜、OOF、逐项对照、消融及早期参数/权重搜索。
- [方法与源码导航](docs/METHODS.md)：各条路线对应的入口、支持模块、报告和核验程序。
- [复现与恢复](docs/REPRODUCIBILITY.md)：数据、环境、历史脚本限制及完整归档恢复。
- [数据、模型与分卷说明](docs/ARTIFACTS.md)：普通仓库和Release各保存什么。
- [来源与贡献边界](docs/ATTRIBUTION.md)：公开预测、参考方案与第三方许可范围。
- [当前最佳结果](CURRENT_BEST_RESULT.md)、[最后一次RTD实验报告](RTD_TRIAL_REPORT_20261023.md)。

## 源码布局

源码保持原始相对位置，便于保留既有导入及缓存定位：`main*.py` 为历史训练入口，`experiment_*.py` 为机制实验，`*_support.py` 为实现，`report_*.py` 为报告生成，`verify_*.py` 与 `tests/` 为核验，`research_s6e10/` 为研究和来源记录。根目录的 `main.py` 默认生成较早的 `submission_optimized.csv`，**不会直接复现最高分版本**。

## 完整归档

[Release archive-20261023](https://github.com/IKMXKT-love/kaggle-airline-satisfaction/releases/tag/archive-20261023) 保存当前项目全部普通文件的分卷ZIP，以及逐文件SHA-256清单。包括比赛/原始数据、历史预测、权重、缓存、依赖环境和原始Git元数据。符号链接记录目标但不遍历项目外的内容。各卷是独立ZIP，恢复程序会逐卷检查并恢复到新目录。

普通Git仓库提供便于阅读的源码、报告与小型元数据；不把未上传到普通Git的大文件描述为已经包含在源码克隆中。历史报告中的本机路径只作原始证据，不保证在另一台机器直接可用。

第三方代码、数据、预测和权重保留各自来源及许可；本归档未为这些材料重新授权。没有为整个混合归档附加统一开源许可证。
'''
    (STAGE/'README.md').write_text(readme,encoding='utf-8')
    methods=[
        ('XGBoost基线','main.py','submission_metrics.json'),
        ('人工特征与插值合成','feature_engineering.py','enhancement_comparison.json'),
        ('原始数据先验与交叉目标编码','optimized_features.py / optimized_training.py','OPTIMIZATION.md'),
        ('XGB/CatBoost/LightGBM组合','optimize.py / optimize_v4.py','V4_EXPERIMENT_REPORT.md'),
        ('航程画像与辅助评分预测','v3_features.py / v3_training.py','FINAL_OPTIMIZATION_REPORT.md'),
        ('RealMLP与精简配方','experiment_neural.py / experiment_realmlp_factorial.py / train_lean_cv_followup.py','REALMLP_FACTORIAL_REPORT.md'),
        ('TabM','v2_tabm.py / optimize_v2.py','FINAL_OPTIMIZATION_REPORT.md'),
        ('原始域频率与条件评分统计','transfer_features.py / experiment_transfer.py','TRANSFER_EXPERIMENT_REPORT.md'),
        ('教师logit起点修正','experiment_transfer.py','TRANSFER_EXPERIMENT_REPORT.md'),
        ('条件原始数据近邻','experiment_source_neighbors.py','FOLLOWUP_20261019_REPORT.md'),
        ('TabICL冻结/微调/局部上下文','experiment_tabicl_probe.py / experiment_tabicl_finetune.py / experiment_local_context.py','TABICL_FINETUNE_REPORT.md / LOCAL_CONTEXT_REPORT_20261021.md'),
        ('遮蔽分类与评分恢复多任务','experiment_multitask.py / multitask_support.py','DENOISING_MULTITASK_REPORT.md'),
        ('FFM字段交互与加性对照','experiment_ffm.py / ffm_support.py','FIELD_INTERACTION_REPORT.md'),
        ('强组合堆叠、offset与成员分歧纠错','experiment_correction.py / experiment_pool_repairability.py','STRONG_CORRECTION_REPORT.md / POOL_REPAIRABILITY_REPORT_20261022.md'),
        ('公开预测池融合及上游增量','public_pool_trial.py / fit_public_pool.py / experiment_joint_upstream.py','PUBLIC_POOL_TRIAL_REPORT_20261020.md / JOINT_UPSTREAM_TRIAL_REPORT_20261021.md'),
        ('双域共享表示','experiment_domain_transfer.py / domain_transfer_support.py','DOMAIN_TRANSFER_REPORT_20261021.md'),
        ('类别条件域密度比','experiment_density_ratio.py / density_ratio_support.py','DENSITY_RATIO_REPORT_20261021.md'),
        ('额外评分与条件积分','experiment_privileged_signal.py / experiment_privileged_marginal.py','PRIVILEGED_SIGNAL_REPORT_20261021.md'),
        ('排序损失融合','experiment_ranking_fusion.py','RANKING_FUSION_REPORT_20261022.md'),
        ('ModernBERT与RTD预训练','experiment_rtd_trial.py / rtd_trial_support.py / evaluate_rtd_trial.py','RTD_TRIAL_REPORT_20261023.md'),
    ]
    def links_cell(items):
        return '、'.join(f'[{s}](../{s})' for s in items.split(' / '))
    m=['# 方法与源码导航','','20条是按机制归并的路线，不是20种原创算法。每条的具体变体及分数见[完整登记](../EXPERIMENT_SCORE_CATALOG_20261023.md)。历史失败和中止均保留。','','| 序号 | 路线 | 入口/实现 | 结果证据 |','|---:|---|---|---|']
    for i,(n,c,r) in enumerate(methods,1):m.append(f'| {i} | {n} | {links_cell(c)} | {links_cell(r)} |')
    (STAGE/'docs/METHODS.md').write_text('\n'.join(m)+'\n',encoding='utf-8')
    (STAGE/'docs/REPRODUCIBILITY.md').write_text('''# 复现与恢复

## 完整文件恢复

安装GitHub CLI并登录后，在新目录下载Release资产：

```powershell
gh release download archive-20261023 --repo IKMXKT-love/kaggle-airline-satisfaction --dir .\release-assets
python .\release-assets\restore.py --assets .\release-assets --destination C:\\Restore\\kaggle-airline-satisfaction
```

恢复程序拒绝非空目标，先校验每卷SHA-256，再逐文件核对与解压。目录与文件清单覆盖当前源文件夹；已清理的历史缓存不能凭空恢复。符号链接/目录联接只保留目标记录，避免自动访问另一台电脑的外部绝对路径。

## 环境

原项目使用Windows、Python虚拟环境和NVIDIA GPU。`requirements*.txt`是各历史阶段依赖，`environment/`保存本次实际安装的包版本。`.artifact_runtime/tabicl_runtime`与`tabicl_finetune_deps`含单独依赖；完整Release保留这些文件。

恢复旧`.venv`不保证可在不同系统路径直接运行。建议另建虚拟环境并依照所选实验的依赖文件安装；需要GPU版本的PyTorch和相容驱动。这里没有重新安装环境或重新训练所有模型，因此不承诺跨平台一键复现全部分支。

## 查看已有最高分文件

恢复后，最高分文件为 `submission_public_pool_20261020.csv`，SHA-256：
`d3029122f333be8c98844b8af4c7f2f2fae91d74487e95994a488df1684a7728`。

不要把运行 `main.py` 与复现最高分混同。该入口默认运行较早的XGB/CatBoost配方。最高分相关源码为 `public_pool_trial.py`、`fit_public_pool.py`、`verify_public_pool_trial.py`，协议及权重保存在 `.artifact_runtime/public_pool_trial_20261020/`。

## 历史实验与核验

历史脚本按当时输入、缓存、原生折和固定时间协议编写；部分协议的绝对deadline现已过期，直接重跑可能停止或覆盖旧输出。应在恢复副本中新建运行目录与新协议，保留旧结果，不能仅修改分数文件或冒充原实验继续。RTD权重不含优化器状态，不能保证中断后逐步完全一致续训。

本归档只做文件、归档、远端哈希与源码语法核验，没有重新跑交叉验证或提交Kaggle。实验本身的核验范围以各报告及对应verification文件为准。公开OOF继承上游早停、间接标签依赖与历史选型限制，不能作为端到端严格嵌套验证。
''',encoding='utf-8')
    (STAGE/'docs/ARTIFACTS.md').write_text(f'''# 文件与归档范围

- 普通Git：可阅读的项目源码、方法、报告、测试、来源、配置及小型实验元数据；不包含完整依赖和大数组。
- Release：{len(files):,}个普通文件，原始合计{total:,}字节，{len(volumes)}个独立ZIP分卷；另附完整清单及恢复程序。
- 文件清单：[archive/manifest.json](../archive/manifest.json)，逐项含原始路径、长度、SHA-256及所属分卷。
- 分卷清单：[archive/volumes.json](../archive/volumes.json)。
- 符号链接：{len(links)}条，只记录目标，不遍历外部共享node_modules或Python安装目录。完整指项目文件边界内的普通文件及链接元数据，并非备份整台电脑。
- 原始README保留在[LEGACY_README.md](LEGACY_README.md)。Release中的原始文件内容没有改写；Git可阅读副本中的本机Markdown链接转换为仓库路径。只在Release中的链接目标标记为归档路径。
- 本地源项目、原最佳预测及状态记录不删除、不覆盖。原.git元数据在归档内，发布仓库使用独立整理提交，不改动用户原暂存区。

归档保留数据、预测、检查点、日志、运行环境、历史版本和已下载第三方材料。可再生成并不意味着本次已省略它们。相应第三方许可证与原始来源元数据一并保留。
''',encoding='utf-8')
    (STAGE/'docs/ATTRIBUTION.md').write_text('''# 来源与贡献

当前最高分方案主要依据[Dariush Afshar公开预测池](https://www.kaggle.com/code/dariushafshar/s6e10-247-public-oofs-one-stack-lb-0-96187)，不是独立原创基础模型。246个成员的作者、版本、来源URL与许可元数据见[完整来源表](../public_pool_sources_20261020.csv)。我们完成数据导入、重新融合、比较、验证与归档，不能将上游训练成果归为本项目原创。

比赛：[Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)。公开原始航空数据：[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)。早期迁移参考[第二名方案](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/2nd-place-solution)。RTD本地改编参考[Chris Deotte Q-FT v2](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/751334)，缺少作者完整辅助特征流水线，不能称为完整复现。

其他研究引用保存在各报告与research_s6e10目录。依赖环境、下载代码和权重保留各自许可；部分公开预测来源元数据为Unknown，公开可下载不等于本项目可以重新授权。这里不对混合归档附加统一MIT等许可证，也不宣称掌握第三方材料著作权。
''',encoding='utf-8')
    (STAGE/'.gitignore').write_text('__pycache__/\n*.pyc\n.venv/\nassets/\n',encoding='utf-8')
    # Make documentation usable on GitHub without altering the archived originals.
    for p in STAGE.rglob('*.md'):
        text=p.read_text('utf-8-sig');rel=p.relative_to(STAGE)
        def replace(m):
            label,target=m.group(1),m.group(2)
            prefix=ROOT.as_posix()+'/'
            if target.startswith(prefix):
                tail=target[len(prefix):];tail=re.sub(r':\d+$','',tail)
                if (STAGE/tail).exists():return f'[{label}]({os.path.relpath(STAGE/tail,p.parent).replace(chr(92),chr(47))})'
                return f'{label}（归档路径：`{tail}`）'
            return m.group(0)
        text=re.sub(r'\[([^\]]+)\]\((C:/Users/ASUS/Documents/ChatGPT/kaggle_first/[^)]+)\)',replace,text)
        p.write_text(text,encoding='utf-8')
    (STAGE/'environment').mkdir(exist_ok=True)
    code="import importlib.metadata,json; print(json.dumps(sorted([(d.metadata['Name'],d.version) for d in importlib.metadata.distributions()]),indent=2))"
    result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-B','-c',code],capture_output=True,text=True,check=True)
    (STAGE/'environment/packages-main.json').write_text(result.stdout,encoding='utf-8')
    save(STAGE/'environment/context.json',{'platform':sys.platform,'python':sys.version,'external_link_targets':links,'note':'No credential values or process environment variables collected.'})
    # Include the recovery program beside the release volumes as well.
    shutil.copy2(OPS/'restore.py',ASSETS/'restore.py')
    save(ASSETS/'archive_info.json',{'repository':REPO,'release':TAG,'original_files':len(files),'original_bytes':total,'volumes':volumes,'manifest_sha256':sha(ASSETS/'archive_manifest.json'),'restore_sha256':sha(ASSETS/'restore.py')})
    progress('archive_crc_check',volumes=len(volumes))
    for v in volumes:
        with zipfile.ZipFile(ASSETS/v['name']) as z:assert z.testzip() is None,v['name']
        progress('archive_crc_check',completed=v['name'])
    # Validate our Python source without importing or running any model.
    import ast
    checked=0
    for p in STAGE.rglob('*.py'):
        try:ast.parse(p.read_text('utf-8-sig'));checked+=1
        except SyntaxError:
            if p.is_relative_to(STAGE/'research_s6e10'):continue
            if '.artifact_runtime' in p.parts:continue
            raise
    staged=[]
    for p in sorted(STAGE.rglob('*')):
        if p.is_file():staged.append({'path':p.relative_to(STAGE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    save(OPS/'readable_manifest.json',staged)
    save(OPS/'prepared.json',{'source_files':len(files),'source_bytes':total,'volumes':volumes,'git_files':len(staged),'git_bytes':sum(x['bytes'] for x in staged),'python_files_parsed':checked,'source_manifest_sha256':sha(ASSETS/'archive_manifest.json')})
    progress('prepared',git_files=len(staged),zip_volumes=len(volumes),python_files_parsed=checked)

if __name__=='__main__':main()
