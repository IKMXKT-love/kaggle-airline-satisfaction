"""Report the frozen three-arm result without touching incumbent records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'.artifact_runtime/domain_transfer_20261021'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    p, e, v = [read(RUN/n) for n in ['protocol.json','evaluation.json','verification.json']]
    assert v['verified'] and v['protected_files_unchanged']
    names = dict(competition_only='只学习比赛标签',shared_head='两域共用预测头',dual_head='两域独立预测头')
    rows, sensitivity, costs = [], [], []
    for arm, name in names.items():
        result = e['arms'][arm]
        selection = read(RUN/'selection'/arm/'complete.json')
        refit = read(RUN/'refit'/arm/'complete.json')
        rows.append(f"| {name} | {selection['best_epoch']} | {result['single_auc']:.9f} | {result['primary_blend_auc']:.9f} | {result['primary_blend_delta']['delta']:+.9f} |")
        sensitivity.append(f"| {name} | {result['diagnostic_weights']['0.05']:+.9f} | {result['primary_blend_delta']['delta']:+.9f} | {result['diagnostic_weights']['0.2']:+.9f} |")
        costs.append(f"| {name} | {selection['seconds']:.2f} | {refit['seconds']:.2f} |")
    delta = e['arms']['dual_head']['primary_blend_delta']
    lower, upper = delta['interval95']
    decision = ('通过预先固定的初筛门槛，候选需冻结后设计确认实验；本轮不生成提交。' if e['passes_expansion_gate'] else
                '未通过预先固定的扩大投入门槛。结束此配方，不扩展十折、不叠加额外评分或继续搜索融合权重，不生成提交。')
    single_change = e['dual_vs_competition_single']['delta']
    match_change = e['dual_vs_competition_primary_blend']['delta']
    boundary = [names[a] for a in names if read(RUN/'selection'/a/'complete.json')['best_epoch']==p['max_epochs']]
    boundary_note = ('达到最大选择轮数的组：'+ '、'.join(boundary)+'。' if boundary else '三组内部最佳轮数均早于最大选择轮数。')
    counts = [(read(RUN/'refit'/a/'complete.json')['epochs_completed'], read(RUN/'refit'/a/'complete.json')['updates']) for a in names]
    update_note = (f'本轮三组均选择{counts[0][0]}轮重训，各执行{counts[0][1]}次比赛批次更新，实际耗时略有差异。' if len(set(counts))==1 else
                   '各组按相同内部选轮规则选择重训轮数，实际更新数与耗时见各组记录。')
    text = f'''# 双域共享表示：三组受控初筛

日期：2026-10-21。实际完成三次GPU选择拟合、三次完整外层训练区重训、独立预测重建及来源/索引核验。

## 结论

{decision}

主要候选固定为独立双头，当前公开池90%＋候选10%的logit融合。相对当前池的AUC差值为 **{delta['delta']:+.9f}**，固定预测配对约95%区间为 **[{lower:+.9f}, {upper:+.9f}]**。扩展门槛为+0.0003，且区间下界大于0、胜过同用法比赛单域对照。

双头相对比赛单域的单模型差值为{single_change:+.9f}；相同10%融合用法的差值为{match_change:+.9f}。这些差值帮助区分“网络本身是否改善”和“当前强组合是否得到增量”，不外推成公榜变化。

## 结果

同一计分集合的当前公开池AUC：**{e['baseline_auc']:.9f}**。本轮没有新公榜成绩；当前用户反馈最佳仍为0.96187。

| 设置 | 内部选择轮数 | 单模型AUC | 固定10%融合AUC | 相对公开池 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

预先列出的权重敏感性只用于描述，不从中重新挑选通过门槛的主候选：

| 设置 | 5%融合差值 | 10%融合差值 | 20%融合差值 |
|---|---:|---:|---:|
{chr(10).join(sensitivity)}

## 训练及隔离

- 原始共同21字段；8个独立参数成员，类别/连续/周期视图，512/256/128隐藏层。没有教师预测、目标编码、恢复任务或额外Inflight service辅助。
- 源数据保留{p['original_rows']}行，排除{p['removed_original_overlap']}个与比赛训练/测试按共同字段完全重合的行。
- seed42原生5折的第0折作计分，共{p['scored_rows']}行。外层训练{p['refit_rows']}行，再拆为拟合{p['fit_rows']}、内部停止{p['stopping_rows']}行。
- 三组在内部训练区拟合预处理、按内部AUC选轮数；随后重新初始化，在完整外层训练区重训。{boundary_note}
- 每组初始化指纹相同，训练协议及最大预算一致。{update_note}这里评价给定停止规则下的实际配方。
- 两域BCE分别取平均，源域权重0.25。C-only源损失权重0；三组均执行同样的两域前向与标签损失计算。模型无跨行统计或注意力，梯度测试验证源标签不会影响C-only。

当前公开池基线从已保存的246列二层权重重算，与缓存一致。它仍继承上游OOF早停和间接标签依赖；**组合结果是条件初筛，不是端到端嵌套验证**。新分支的计分行不参与其预处理、梯度或早停。历次方案与划分已被研究，不能把本轮称为全新盲测。

## 完整成本

资源探针用拟合/原始数据训练临时参数，随后丢弃。最慢训练步约{read(RUN/'probe.json')['worst_seconds_per_step']:.5f}秒。32轮完整流程的保守估计超出900秒，故在看分数前固定最多{p['max_epochs']}轮。

| 设置 | 内部选择训练/预测秒数 | 完整外层重训/预测秒数 |
|---|---:|---:|
{chr(10).join(costs)}

数据准备与探针约{p['costs']['preparation_and_probe_seconds']:.2f}秒；独立核验约{v['verification_seconds']:.2f}秒。从首次准备开始到核验完成累计 **{v['cumulative_seconds']:.2f}秒（{v['cumulative_seconds']/60:.2f}分钟）**，900秒累计上限满足情况：{v['cumulative_deadline_met']}。没有重训公开池上游模型。

## 核验与适用范围

三个梯度/损失权重契约检查通过。独立核验重新读取实际CSV，重建两个阶段编码器、源数据编码与标签、外层索引、相同初始化约定、选择轮数和重训更新数，并从三组保存权重重建全部计分预测，复算单模型与固定融合AUC。

此结果只评价当前共享MLP、联合训练、源权重0.25和已冻结预算。不能据此否定所有迁移学习、先原始预训练再微调或其他骨干。也没有检验额外Inflight service监督、局部检索微调、类别条件密度比。固定预测区间不覆盖训练随机性、历史筛选或公开OOF依赖。

## 产物

实验目录：`.artifact_runtime/domain_transfer_20261021/`，含协议、资源探针、完整六次拟合记录/权重、预测、评价与独立核验。代码：`experiment_domain_transfer.py`、`domain_transfer_support.py`、`verify_domain_transfer.py`。初始计划：`DOMAIN_TRANSFER_PLAN_20261021.md`。

最佳CSV与三个CURRENT文件前后哈希一致。最佳CSV SHA256仍为 `d3029122f333be8c98844b8af4c7f2f2fae91d74487e95994a488df1684a7728`。
'''
    path = ROOT/'DOMAIN_TRANSFER_REPORT_20261021.md'
    if path.exists(): raise FileExistsError('Preserve existing report')
    path.write_text(text,encoding='utf-8')
    print(str(path))


if __name__ == '__main__': main()
