"""Human-readable record for the completed four-arm local-context trial."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'.artifact_runtime/local_context_20261021'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    p,e,v=[read(RUN/f) for f in ['protocol.json','evaluation.json','verification.json']]
    assert v['verified']
    names={'random_frozen':'随机上下文，冻结','local_frozen':'局部上下文，冻结',
           'random_tuned':'随机上下文，微调','local_tuned':'局部上下文，局部微调'}
    rows=[];sensitivity=[];timings=[]
    for arm,name in names.items():
        score=e['arms'][arm]
        rows.append(f"| {name} | {score['single_auc']:.9f} | {score['primary_blend_auc']:.9f} | {score['primary_delta']['delta']:+.9f} |")
        sensitivity.append(f"| {name} | {score['diagnostic_deltas']['0.05']:+.9f} | {score['primary_delta']['delta']:+.9f} | {score['diagnostic_deltas']['0.2']:+.9f} |")
        timings.append(f"| {name} | {read(RUN/arm/'inference.json')['seconds']:.2f} |")
    candidate=e['arms']['local_tuned']['primary_delta']
    decision=('通过冻结的初筛门槛，需要按固定候选设计确认；此报告不代表新公榜成绩。' if e['passes_expansion_gate'] else
              '未通过预先固定的扩大投入门槛。停止本配方，不追加更新、种子、检索参数或融合权重搜索，不生成提交。')
    train_random,train_local=[read(RUN/(kind+'_tuned')/'complete.json') for kind in ['random','local']]
    content=f'''# 局部上下文与TabICL联合微调：四组初筛

日期：2026-10-21。四组完整预测及独立核验完成。

## 结论

{decision}

主候选固定“局部上下文＋局部微调”，主要用法固定为当前公开池90%＋候选10%的logit融合。相对当前池的差值为 **{candidate['delta']:+.9f}**，固定预测配对约95%区间为 **[{candidate['interval95'][0]:+.9f}, {candidate['interval95'][1]:+.9f}]**。扩展要求+0.0003、区间下界大于0，并胜过同用法随机上下文微调组。

当前最佳公榜仍为用户反馈0.96187；本轮所有分数均来自本地条件初筛，不是公榜成绩，也不能直接加到公榜上。

## 同一计分行的结果

当前公开池基线AUC：**{e['baseline_auc']:.9f}**。计分{p['scored_rows']}行。

| 设置 | 单模型AUC | 固定10%融合AUC | 相对当前池 |
|---|---:|---:|---:|
{chr(10).join(rows)}

机制对照的单模型配对差值：

- 冻结模型的局部上下文相对随机上下文：{e['local_frozen_vs_random_frozen_single']['delta']:+.9f}。
- 局部微调相对随机上下文微调：{e['local_tuned_vs_random_tuned_single']['delta']:+.9f}。
- 局部微调相对局部冻结：{e['local_tuned_vs_local_frozen_single']['delta']:+.9f}。

单模型改善与当前强融合获得增量是不同问题；判断是否扩展只采用预先固定的主候选和主用法。

预先列出的敏感性诊断，不从中重新选择入围权重：

| 设置 | 5%差值 | 10%差值 | 20%差值 |
|---|---:|---:|---:|
{chr(10).join(sensitivity)}

## 方法与隔离

- 固定本地官方TabICLv2预训练权重，SHA256为`{p['checkpoint_sha256']}`。两种预处理视图，推理固定1024行query批次。网络复用适配器与官方接口的探针预测最大差为{read(RUN/'resource_probes.json')[0].get('reuse_adapter_max_difference_vs_official','see probes')}。
- 原始21字段，数值编码只在外层训练区拟合。没有原始满意度教师、目标编码、额外评分任务或伪标签。
- 外层使用seed42原生5折的第1折，训练{p['reference_rows']}行，原始计分区{p['outer_eval_rows']}行。资源规则在看任何外部模型分数前，选择随机分层{p['scored_rows']}行初筛。没有按错误、置信度或模型分歧挑行。
- 检索表示由17个稳健数值视图和4个类别字段one-hot组成，不使用标签。64个锚点在训练区随机65536行拟合，完整训练区为每锚点提供8192个最近候选。查询分组与context索引均保存。
- 每个预测context有{p['context_size']}行，其中128行是两组共享的全局背景。相近query按最近锚点共享context；**这是分组近似，未测试逐query精确检索**，不能称完整复现LoCalPFN。
- 两个微调从原始权重重新开始，各执行{p['steps']}次更新，共享同一批query行与随机种子，只改变context选择。训练query来自实际锚点所属训练行，按锚点训练行数加权抽取；不按标签检索。context/query互斥，query标签只进损失。
- 固定更新次数，没有使用外层计分标签早停、挑checkpoint或调学习率。随机微调累计query监督{train_random['query_supervised_rows']}行次、涉及{train_random['distinct_query_rows']}个不重复训练行；局部组采用相同query。不能称全量训练数据多轮充分训练。
- 两组分别有{train_random['changed_tensors']}和{train_local['changed_tensors']}个权重张量不同于预训练状态，确实发生微调。

正式运行初期遇到FP16缩放梯度溢出。原始检查在GradScaler处理前中断；中断目录、原协议和源码均保留。数值修复在观察计分结果前完成：溢出时GradScaler跳过参数更新并降低缩放，重新执行同一个任务与随机前向，只有有限梯度才计入成功更新。两组仍各完成相同的{p['steps']}次成功更新，没有改样本、学习率或上下文配置。两组实际溢出重试分别为{train_random.get('overflow_retries',0)}和{train_local.get('overflow_retries',0)}次。

当前公开池基线有上游早停、混合折和间接标签依赖。新分支计分区未参与训练，但**融合评价仍属于条件初筛，不是端到端嵌套验证**。该折历史上也被研究过，不能称完全未见的盲测。

## 完整成本

资源探针覆盖训练任务构造、真实反向更新以及建立context后的1024行预测；探针参数丢弃。最终选择context/query={p['context_size']}/{p['query_size']}、{p['steps']}次更新、{p['scored_rows']}行计分，选择依据只来自成本。

数据准备、探针与计划约{p['budget']['preparation_probe_and_plan_seconds']:.2f}秒。两个实际微调分别{train_random['seconds']:.2f}秒、{train_local['seconds']:.2f}秒。

| 设置 | 完整推理秒数 |
|---|---:|
{chr(10).join(timings)}

独立核验约{v['verification_seconds']:.2f}秒。从首次准备到核验完成累计 **{v['cumulative_seconds']:.2f}秒（{v['cumulative_seconds']/60:.2f}分钟）**；满足900秒上限：{v['cumulative_deadline_met']}。这个累计时间包含首次中断、修复间隔与重新运行，没有重置计时器。

## 核验范围

三项契约检查通过：query标签不能改变forward输入或context预处理；随机/局部任务共享query且context互斥；检索变换不在计分行上重新拟合。

独立核验重读CSV，重建数值编码、检索表示、近邻候选池、所有训练任务索引与context、行ID及基线预测。对全部计分行复算四组AUC和融合差值。每组另从权重重建两个完整锚点组的GPU预测，最大误差在核验文件中；**没有重新执行全部GPU预测**。

此结果评价的是当前TabICLv2、固定更新数、基于原始字段的距离、锚点共享近邻与该context规模。不能外推成所有基础模型检索/微调机制都无效。区间只反映固定预测的样本不确定性，不覆盖训练随机性、历史研究或候选选择。

## 产物和保护

证据目录：`.artifact_runtime/local_context_20261021/`。复现入口：`experiment_local_context.py`；支持实现：`local_context_support.py`；独立核验：`verify_local_context.py`；预先计划：`LOCAL_CONTEXT_PLAN_20261021.md`。

最佳CSV与三个CURRENT文件前后哈希相同。最佳CSV SHA256保持`d3029122f333be8c98844b8af4c7f2f2fae91d74487e95994a488df1684a7728`。本轮没有生成或提交新CSV。
'''
    path=ROOT/'LOCAL_CONTEXT_REPORT_20261021.md'
    if path.exists():raise FileExistsError('Preserve existing report')
    path.write_text(content,encoding='utf-8')
    print(str(path))


if __name__=='__main__':main()
