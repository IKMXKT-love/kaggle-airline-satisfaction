"""Render the completed experiment facts without changing the best submission."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    directory = Path(read(ROOT/'.artifact_runtime/tabicl_finetune_20261019/latest.json')['directory'])
    protocol = read(directory/'protocol.json')
    decision = read(directory/'screen/decision.json')
    verification = read(directory/'verification.json')
    assert verification['verified']
    resources = read(directory/'resource_probe.json')
    names = dict(frozen='冻结', finetuned='微调')
    baseline_auc = read(directory/'screen/data.json')['current_auc']
    lines = []
    for cell in decision['cells']:
        if cell['status'] != 'complete':
            lines.append(f"| {names[cell['state']]} | {cell['context']:,} | {cell['status']} | — | — | — |")
            continue
        best = max(cell['trials'], key=lambda t:t['paired']['delta'])
        low, high = best['paired']['interval95']
        lines.append(f"| {names[cell['state']]} | {cell['context']:,} | {cell['auc']:.9f} | "
                     f"{best['weight']:.0%} | {best['paired']['delta']:+.9f} | [{low:+.9f}, {high:+.9f}] |")
    ft_path = directory/'screen/finetune/summary.json'
    ft = read(ft_path) if ft_path.exists() else None
    if ft and ft['status'] == 'complete':
        selected_round = next(r['validation_call'] for r in ft['validation_trace']
                              if abs(r['auc']-ft['best_inner_auc']) < 1e-12)
        trace = '\n'.join(f"| {r['validation_call']} | {r['auc']:.9f} | {r['seconds']:.1f} |"
                          for r in ft['validation_trace'])
        details = f"""
## 微调过程

微调{len(ft['epochs'])}轮，{ft['completed_steps']}次更新，每块{ft['chunk']}行，训练{ft['seconds']/60:.1f}分钟，
峰值GPU分配{ft['peak_gpu_gb']:.2f}GB。最终权重与预训练权重相比，有{ft['changed_tensors']}个张量不同。
是否退回预训练权重：{ft['selected_pretrained']}。最终采用第{selected_round}轮的权重（0表示预训练），
没有采用最后一轮权重。下面计分仅来自训练区内部的停止集。

| 内部验证次数（0为更新前） | AUC | 秒 |
|---|---:|---:|
{trace}
"""
    else:
        details = '\n## 微调状态\n\n' + json.dumps(ft, ensure_ascii=False, indent=2) + '\n'
    confirmation = (read(directory/'confirmation/decision.json')
                    if (directory/'confirmation/decision.json').exists() else None)
    if decision['chosen'] is None:
        conclusion = '四组候选均未通过预设初筛门槛，停止该配置扩展；不进入第二划分或完整十折，不生成新提交。'
    else:
        conclusion = '初筛通过，冻结方案已进入第二划分；确认结论见下方。'
    status_text = ('四组均已完成，效果未通过门槛' if decision['four_cell_design_completed'] and
                   decision['chosen'] is None else decision['status'])
    confirmation_text = json.dumps(confirmation, ensure_ascii=False) if confirmation else '未进入第二划分复核。'
    text = f"""# TabICL任务微调与上下文四组对照

日期：2026-10-19。状态：{status_text}。

{conclusion}

当前线上最佳仍为用户反馈0.96114，本轮没有新的线上成绩。

## 设计

- 411944行拟合、35822行内部停止、111942行外部筛选，三个集合互斥。
- 冻结/微调 × 32768/65536上下文。32K是64K的前半部分，全部来自拟合区。
- 同一组21原始特征；类别编码/缺失值处理只在拟合区拟合，四组共享相同编码。
- 微调分块利用拟合区，2个训练视图、最多3轮、学习率1e-5；以内部停止集32K上下文AUC选权重，然后同一权重评估两个上下文。
- 所有预测使用4个视图、相同1024行查询批次及相同logit平均。没有增添新的合成样本。
- 对照采用最新固定配方：65% v4＋35%双种子精简RealMLP的logit融合，在相同筛选行重建。
- 初筛只比较预先固定5%/20%新成员概率融合，共最多8项。门槛为+0.0003且配对区间下界大于0；确认门槛为+0.0002。

## 外部筛选结果

同协议最新组合基线AUC：{baseline_auc:.9f}。这与完整OOF或公榜评分集合不同，绝对数值不能直接比较。

表中每行显示两项固定融合中增益较大的那项；这是筛选结果，不是独立确认。

| 状态 | 上下文行数 | 单模型AUC | 新成员权重 | 相对最新组合增益 | 固定预测差值约95%区间 |
|---|---:|---:|---:|---:|---|
{chr(10).join(lines)}
{details}
## 资源与验证

先完成真实梯度探针，证实参数发生更新。适配器与官方实现，在相同查询批次下最大预测差为
{resources['maximum_prediction_difference']:.10f}。另测得官方半精度预测对查询批次大小存在差异，
因此正式四组始终固定1024行批次，不把改变批次后的数值直接当成算法收益。

最初尝试将四个视图缓存保存在CPU，32K完成但系统内存紧张，因此中止该实现。
正式结果改为逐视图处理整个查询集，保留相同模型、四个视图及平均规则，仅改变计算顺序和缓存驻留。
该适配通过官方数值对照后才进行正式筛选；旧尝试目录保留，没有当作额外候选搜分。

4项契约测试通过：上下文范围限制、实际拟合行分块覆盖、context/query互斥。
独立核验确认输入/代码哈希、原始官方权重、索引、当前基线、全部预测AUC、融合差值及区间、预设门槛和受保护文件。
完整111942行的32K预测，在CPU缓存版本与逐视图版本之间也逐行完全一致。

复现入口：`experiment_tabicl_finetune.py all`；独立核验：`verify_tabicl_finetune.py`；
契约测试：`python -m unittest discover -s tests -p test_tabicl_finetune_contract.py -v`。

## 证据限制

这些分区在历史研究中已被查看，并非全新盲测；旧v4还包含外层早停的历史限制。
差值区间只覆盖固定预测的样本不确定性，不覆盖训练随机性、候选筛选或未来分布变化。
微调本次只检验了一个训练配方、一个种子和有限训练预算；负结果不能判定整个模型家族无效。
两项融合权重门槛有意限制搜索，可能漏掉其他有益的小权重，但不能据此无止境扩展。

确认结果：{confirmation_text}

完整证据目录：`{directory}`。
官方接口：[TabICL API](https://tabicl.readthedocs.io/en/latest/api.html)。
"""
    path = ROOT/'TABICL_FINETUNE_REPORT.md'
    path.write_text(text, encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
