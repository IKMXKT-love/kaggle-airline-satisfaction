"""Read existing evidence only; never train or score models."""
import json
import re
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))

def link(name, line=None):
    path = (ROOT / name).as_posix()
    return f'[{name}]({path}' + (f':{line}' if line else '') + ')'

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] + ['| ' + ' | '.join(str(c).replace('|', '/') for c in row) + ' |' for row in rows])

public = [
    ('公开246基础预测＋L2逻辑回归融合', .96187, 'submission_public_pool_20261020.csv'),
    ('v4＋双种子10折精简RealMLP，65/35 logit融合', .96114, 'submission_lean_realmlp_10fold.csv'),
    ('v4＋全量训练精简RealMLP，80/20概率融合', .96104, 'submission_v4_lean_realmlp_trial.csv'),
    ('v4：v3＋LightGBM', .96082, 'submission_v4.csv'),
    ('v3：航程画像、目标编码、辅助评分预测＋XGB/RealMLP融合', .96074, 'submission_v3.csv'),
    ('初始优化：原始数据先验/目标编码＋XGB/CatBoost融合', .96036, 'submission_optimized.csv'),
]

reports = [
    ('初始XGBoost、合成数据及原始数据先验', 'optimization_report.md'),
    ('v3、TabM与CatBoost参数对照', 'FINAL_OPTIMIZATION_REPORT.md'),
    ('v4特征消融、残差特征与LightGBM融合', 'V4_EXPERIMENT_REPORT.md'),
    ('RealMLP航程类别及数值表示', 'NEURAL_EXPERIMENT_REPORT.md'),
    ('源域统计、教师offset与4096分箱', 'TRANSFER_EXPERIMENT_REPORT.md'),
    ('RealMLP特征/训练配方四组对照', 'REALMLP_FACTORIAL_REPORT.md'),
    ('双种子融合及初始TabICL', 'FOLLOWUP_20261019_REPORT.md'),
    ('TabICL冻结/微调与上下文容量', 'TABICL_FINETUNE_REPORT.md'),
    ('遮蔽分类与评分恢复多任务', 'DENOISING_MULTITASK_REPORT.md'),
    ('普通堆叠与固定起点纠错', 'STRONG_CORRECTION_REPORT.md'),
    ('FFM与加性模型', 'FIELD_INTERACTION_REPORT.md'),
    ('公开预测池及本地模型纳入对照', 'PUBLIC_POOL_TRIAL_REPORT_20261020.md'),
    ('Kumo及神经邻居上游联合拟合', 'JOINT_UPSTREAM_TRIAL_REPORT_20261021.md'),
    ('比赛/原始数据双域共享学习', 'DOMAIN_TRANSFER_REPORT_20261021.md'),
    ('类别条件域密度比', 'DENSITY_RATIO_REPORT_20261021.md'),
    ('TabICL局部上下文联合微调', 'LOCAL_CONTEXT_REPORT_20261021.md'),
    ('额外评分信息及条件积分：仅原始源域', 'PRIVILEGED_SIGNAL_REPORT_20261021.md'),
    ('排序损失融合：内部筛选', 'RANKING_FUSION_REPORT_20261022.md'),
    ('公开池成员分歧纠错：内部筛选', 'POOL_REPAIRABILITY_REPORT_20261022.md'),
    ('Transformer与RTD同条件比较', 'RTD_TRIAL_REPORT_20261023.md'),
]

lines = ['# 已尝试方法与得分完整登记（2026-10-23）', '',
    '本清单只读取已有文件，没有启动训练、推理、调参或提交。之前的“20类”是人为归并的方法方向，不是独立算法数、实验次数或已提交版本数。下面展开记录具体变体。', '',
    '**排序规则：公榜成绩单列降序；完整OOF单列降序；其余每张原始对照表在自身验证范围内降序。不同分区、单模型/融合、内部选型/外部计分和源域成绩不合并成一个真实优劣排行榜。** 没有线上记录的方案不补造公榜分数。', '',
    '## 一、用户反馈的公榜成绩（完整六个版本）', '',
    '成绩均来自用户反馈，未独立查询Kaggle提交记录；文件关联沿用项目当时的归档。', '',
    table(['序号', '方案', '公榜AUC', '预测文件'], [(i, n, f'{s:.5f}', link(f)) for i,(n,s,f) in enumerate(public,1)]), '',
    '## 二、完整训练集OOF结果（按数值降序）', '',
    '五折公开预测融合与十折本地训练的上游及验证协议不同；下表的数值顺序不能替代同条件比较。OOF不等于公榜。', '']

oof = []
for name, file in [('原始XGBoost','submission_metrics.json'), ('人工特征＋插值合成','submission_enhanced_metrics.json'), ('初始XGB/CatBoost优化','submission_optimized_metrics.json'), ('v3','submission_v3_metrics.json'), ('v4','submission_v4_metrics.json')]:
    d=read(file)
    oof.append((name, d.get('oof_auc',d.get('oof_score')), '本地10折', file))
d=read('submission_lean_realmlp_10fold.json')
oof += [('双种子精简RealMLP单模型',d['standalone_oof_auc'],'本地10折','submission_lean_realmlp_10fold.json'),('v4＋双种子精简RealMLP融合',d['blended_oof_auc'],'本地10折','submission_lean_realmlp_10fold.json')]
d=read('.artifact_runtime/public_pool_trial_20261020/evaluation.json')
for key,name in [('public_only','公开246预测池'),('public_plus_own','公开池＋本地七分支')]:
    oof.append((name,d['pooled_aucs'][key],'原生5折二层融合','.artifact_runtime/public_pool_trial_20261020/evaluation.json'))
d=read('.artifact_runtime/joint_upstream_20261021/evaluation.json')
joint_names={'plus_kumo':'公开池＋Kumo','plus_neighborhood':'公开池＋神经邻居','plus_both':'公开池＋Kumo＋神经邻居'}
for item in d['comparisons']:
    if item['arm'] in joint_names:
        oof.append((joint_names[item['arm']],item['auc'],'原生5折二层融合','.artifact_runtime/joint_upstream_20261021/evaluation.json'))
oof.sort(key=lambda x:x[1],reverse=True)
lines += [table(['序号','方案','整体OOF AUC','验证范围','证据'],[(i,n,f'{s:.12f}',p,link(f)) for i,(n,s,p,f) in enumerate(oof,1)]),'',
    '## 三、逐项实验、对照、消融与融合变体', '',
    '每组保留原表的全部字段与行；每表按标注的AUC列降序。逐折成绩、早停过程、分群分析只是对应实验的明细，不另算独立方法。只有差值的表按差值排序，明确不当成绝对AUC。', '']

tables=[]
def number(x):
    m=re.search(r'[-+−]?\d*\.\d+(?:[eE][-+]?\d+)?',x.replace('**','').replace(',','').replace('−','-'))
    return float(m.group()) if m else float('-inf')

for title,file in reports:
    text=(ROOT/file).read_text('utf-8-sig').splitlines()
    lines += ['### '+title, '', '来源：'+link(file), '']
    i=0; context=''
    while i<len(text):
        if text[i].startswith('#'): context=text[i].lstrip('# ').strip()
        if not text[i].startswith('|'):
            i+=1;continue
        start=i;block=[]
        while i<len(text) and text[i].startswith('|'):
            block.append([v.strip() for v in text[i].strip().strip('|').split('|')]);i+=1
        if len(block)<3:continue
        headers=block[0];rows=block[2:]
        candidates=[j for j,h in enumerate(headers) if 'AUC' in h or '增益' in h or ('差值' in h and '区间' not in h)]
        if not candidates:continue
        def priority(j):
            h=headers[j]
            if 'AUC' in h and any(w in h for w in ['差','变化']):return 0
            if 'AUC' in h and any(w in h for w in ['融合','候选','新方案','本次','整体','合并']):return 5
            if 'AUC' in h and not any(w in h for w in ['对照','基线','当前']):return 4
            return 2 if 'AUC' in h else 1
        col=max(candidates,key=priority)
        rows.sort(key=lambda row:number(row[col]) if len(row)>col else float('-inf'),reverse=True)
        lines += ['**'+context+'**；排序列：`'+headers[col]+'`。', '',table(headers,rows),'']
        tables.append({'source':file,'line':start+1,'context':context,'sort_column':headers[col],'headers':headers,'rows':rows})

lines += ['### 原始数据条件近邻（报告正文中的结果）','',
    table(['设置','单模型AUC','融合AUC','相对同条件基线','证据'],[('按客群条件检索16/64原始数据近邻', '0.960031792','0.960634072','+0.000000690',link('FOLLOWUP_20261019_REPORT.md'))]),'',
    '## 四、早期候选及参数/融合搜索逐项记录', '',
    '下列为历史开发集选型成绩，没有重新评估；同一配置也可能在上面的最终报告中出现。配置名和全部权重保留，以便逐项追溯；不把不同权重等同于新算法。', '']
search_groups=[]
for file,keys in [('optimization_search.json',['records','blend_trials']),('v2_recipe.json',['nn_candidates','trials']),('v3_recipe.json',['trials']),('v4_recipe.json',['trials','ablations'])]:
    d=read(file)
    for key in keys:
        rows=[]
        for i,item in enumerate(d[key]):
            score=item.get('selection_auc',item.get('auc'))
            if score is None:continue
            cfg={k:v for k,v in item.items() if k in ['name','family','spec','group','config','weights','components','mode','weight','features','rounds','best_rounds','seed','depth','child','lr','use_te','original','numeric_categories','digits','rows_weight','multiple_priors','extended_keys']}
            if isinstance(cfg.get('features'),list):cfg['features']=len(cfg['features'])
            rows.append((score,json.dumps(cfg,ensure_ascii=False,separators=(',',':')),i))
        rows.sort(reverse=True,key=lambda x:x[0])
        lines += ['### '+file+' / '+key,'','来源：'+link(file),'',table(['序号','配置/权重','开发集AUC','JSON原索引'],[(j,'`'+cfg+'`',f'{s:.12f}',i) for j,(s,cfg,i) in enumerate(rows,1)]),'']
        search_groups.append({'source':file,'key':key,'rows':rows})
file='.artifact_runtime/v2/cat_screen.json'
rows=[]
for i,x in enumerate(read(file)):
    rows.append((x['auc'],json.dumps(x.get('config',{}),ensure_ascii=False),i))
rows.sort(reverse=True,key=lambda x:x[0])
lines += ['### CatBoost七组参数候选','','来源：'+link(file),'',table(['序号','参数','开发集AUC'],[(i,cfg,f'{s:.12f}') for i,(s,cfg,_) in enumerate(rows,1)]),'']
search_groups.append({'source':file,'key':'all','rows':rows})

# Extract omitted endpoint scores from high-level JSON summaries, not epoch histories.
patterns=['.artifact_runtime/neural_representation/*/summary.json','.artifact_runtime/followup_20261019/*/*/summary.json','.artifact_runtime/tabicl_finetune_20261019/*/screen/*/*.json','.artifact_runtime/tabicl_finetune_20261019/*/screen/*.json','.artifact_runtime/tabicl_probe_20261019/*/summary.json','.artifact_runtime/source_neighbors_20261019/*/screen/summary.json']
extra=[]
skip={'curves','history','epochs','validation_trace','features','detail','model','protocol','config','verification','resources','source','train_auc','train_auc_sample'}
def endpoint(obj,path,source):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if key in skip:continue
            p=path+'.'+key
            if isinstance(value,(float,int)) and not isinstance(value,bool) and ('auc' in key.lower()) and not any(s in key.lower() for s in ['gain','delta','train']):
                if 0<=value<=1:extra.append({'source':source,'key':p,'auc':value})
            elif isinstance(value,(dict,list)):endpoint(value,p,source)
    elif isinstance(obj,list):
        for i,value in enumerate(obj):endpoint(value,f'{path}[{i}]',source)
seen=set()
for pattern in patterns:
    for p in ROOT.glob(pattern):
        if p.name in ['data.json','verification.json','protocol.json','resource.json','training.json','complete.json','summary.json'] and 'finetune/summary.json' in p.as_posix():continue
        name=p.relative_to(ROOT).as_posix()
        if name in seen:continue
        seen.add(name);endpoint(read(name),'$',name)
lines += ['## 五、补充端点评分索引','','包括报告中可能只展示最佳融合比例的明细；按文件分别排序。JSON路径是原始定位，不代表新增方法。','']
for source in sorted({x['source'] for x in extra}):
    rows=sorted([x for x in extra if x['source']==source],key=lambda x:x['auc'],reverse=True)
    lines += ['来源：'+link(source),'',table(['指标路径','AUC'],[(x['key'],f"{x['auc']:.12f}") for x in rows]),'']

lines += ['## 六、没有独立可排名分数的内容','','- 教师先验、目标编码、航程画像、频率、13项辅助评分预测等是组合的组成部分。其独立贡献以消融表记录，不能把完整组合分数冒充各特征自身的分数。',
    '- 原始域额外真实评分是不可部署的信息对照；0.995左右的源域分数不属于本比赛，不进入公榜或比赛OOF排名。',
    '- RTD检测损失、评分恢复准确率、训练集AUC、吞吐/资源探针以及中断且没有端点评分的配置不作为比赛成绩。',
    '- 本轮引用的外部作者成绩、只阅读未运行的方法、新51成员的作者侧刷新试验不算我们实际完成的实验。',
    '- 上述原始报告及历史JSON均保留。新清单只汇总，不修改历史实验记录与当前最佳文件。','']

summary={'public_versions':len(public),'full_oof_rows':len(oof),'original_report_sources':len(reports),'auc_or_delta_tables':len(tables),'rows_in_report_tables':sum(len(t['rows']) for t in tables),'early_search_rows':sum(len(g['rows']) for g in search_groups),'supplementary_endpoint_metrics':len(extra),'counts_are_evidence_rows_not_unique_methods':True}
lines += ['## 登记范围统计','', '这些数量是证据行数，含对照、复核、逐折记录和重复引用；不能相加当作独立实验次数。','',table(['项目','数量'],[(k,v) for k,v in summary.items()]),'']
(ROOT/'EXPERIMENT_SCORE_CATALOG_20261023.md').write_text('\n'.join(lines),encoding='utf-8')
(OUT/'catalog.json').write_text(json.dumps({'summary':summary,'public':public,'full_oof':oof,'tables':tables,'search_groups':search_groups,'supplementary':extra},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
