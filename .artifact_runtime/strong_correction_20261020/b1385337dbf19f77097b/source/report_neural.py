"""Summarize the completed, bounded neural experiment without fitting models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiment_neural import ROOT, OUT, sha


def placements(y, p):
    """AUC contributions for positives and negatives, with half-credit for ties."""
    pos, neg = p[y == 1], p[y == 0]
    sp, sn = np.sort(pos), np.sort(neg)
    vp = (np.searchsorted(sn, pos, side='left') + np.searchsorted(sn, pos, side='right')) / (2*len(neg))
    vn = 1-(np.searchsorted(sp, neg, side='left') + np.searchsorted(sp, neg, side='right')) / (2*len(pos))
    np.testing.assert_allclose(vp.mean(), roc_auc_score(y, p), atol=1e-12)
    return vp, vn


def paired(y, p, q):
    pp, pn = placements(y, p)
    qp, qn = placements(y, q)
    dp, dn = pp-qp, pn-qn
    delta = float(dp.mean())
    se = float(np.sqrt(dp.var(ddof=1)/len(dp)+dn.var(ddof=1)/len(dn)))
    return dict(delta=delta, approximate_se=se, interval95=[delta-1.96*se, delta+1.96*se])


def main():
    directory = Path(json.loads((OUT/'latest.json').read_text())['directory'])
    s = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    for name, expected in protocol['protected'].items():
        assert sha(ROOT/name) == expected, name
    with np.load(directory/'indices.npz') as z:
        index = z['selection']
    y = pd.read_csv(ROOT/'train.csv', usecols=['satisfaction']).satisfaction.to_numpy()[index]
    control = np.load(directory/'control.npz')['prediction']
    uncertainty = {}
    for name in ['distance_category', 'numeric_sigma32']:
        p = np.load(directory/(name+'.npz'))['prediction']
        uncertainty[name] = paired(y, p, control)
    (directory/'paired_uncertainty.json').write_text(json.dumps(uncertainty, indent=2), encoding='utf-8')
    names = dict(control='同预算 RealMLP 对照', distance_category='新增航程类别嵌入', numeric_sigma32='数值嵌入尺度 32')
    rows = '\n'.join(f"| {names[d['spec']['name']]} | {len(d['features'])} | {d['auc']:.9f} | {d['seconds']/60:.1f} |" for d in s['models'])
    blends = '\n'.join(f"| {names[d['name']]} | {d['weight']:.0%} | {d['auc']:.9f} | {d['gain_vs_v4']:+.9f} | {d['gain_vs_matched_control']:+.9f} |" for d in s['trials'])
    intervals = '\n'.join(f"| {names[k]} | {v['delta']:+.9f} | [{v['interval95'][0]:+.9f}, {v['interval95'][1]:+.9f}] |" for k,v in uncertainty.items())
    decision = ('通过初筛门槛，需要冻结方案后另行复核；尚不能称为最终改善。' if s['warrants_confirmation'] else
                '未通过预先固定的融合增益门槛，停止本轮扩展，保留 v4；不启动完整 10 折或生成新提交。')
    report = f'''# RealMLP 表示方式实验

日期：2026-10-18。状态：三个预先指定的设置已完成实际 GPU 训练。

## 结论

{decision}

## 固定实验设计

- 同一随机种子、同一数据划分、同一 64 epoch 预算、同一网络其余参数。
- 拟合 {s['train_rows']:,} 行，内部停止验证 {s['stopping_rows']:,} 行，外部筛选 {s['selection_rows']:,} 行。
- 拟合部分内部 5 折生成目标编码；停止验证和筛选特征只查询拟合部分的映射。缺失值中位数、航程类别词表同样仅来自拟合部分。
- 外部筛选标签不用于神经网络停止轮数选择。旧 audit 分区未用于本轮筛选训练。
- 航程类别实验保留连续航程，额外增加可学习类别表示；未知航程映射到未知类别。
- 尺度实验将 plr_sigma 从 0.1 改为 32，影响全部数值嵌入，不是仅修改航程。
- 这是 64 epoch 的方向筛选；旧版本预算为 128 epoch，不能据此断言某架构的性能上限。

## 单模型结果

| 设置 | 特征数 | 筛选 AUC | 训练及预测分钟 |
|---|---:|---:|---:|
{rows}

## 融合诊断

历史 v4 同筛选集 AUC：{s['v4_selection_auc']:.9f}。以本轮对照替换原 14% 神经网络后的 AUC：{s['matched_control_blend_auc']:.9f}。

在保留其他模型相对比例的前提下，神经网络权重预先固定比较 14%、30%、50%。同时比较历史 v4 和使用本轮对照网络的相同权重融合，避免把增加权重误认为表示方式收益。

| 设置 | 网络权重 | 融合 AUC | 相对历史 v4 | 相对同权重对照 |
|---|---:|---:|---:|---:|
{blends}

预设进入复核门槛：相对历史 v4 和相对同权重对照均至少提高 0.0002。门槛是算力投入标准，不是统计显著性结论。

这里复用了历史树模型的筛选预测；那些树模型曾用筛选标签早停。因此融合数字只用于初筛，不能称为完整外层隔离评估。若通过门槛，仍需按一致协议重新训练基线并复核。历史筛选集也已反复查看。

## 成对不确定性参考

以下仅比较固定单模型预测与本轮对照，使用 AUC placement 的成对渐近方差；未覆盖重新训练、种子差异和多重筛选偏差，不是最终泛化置信保证。

| 设置 | 单模型 AUC 差 | 约 95% 区间 |
|---|---:|---|
{intervals}

## 文件与核验

- 运行入口：[experiment_neural.py](../../experiment_neural.py)。复跑会校验指纹并复用已完成结果。
- 明细目录：`{directory}`。
- 已检查：训练/停止/筛选/audit 分区互斥；输入排除 id 和目标；预测范围及行序；训练期填补与未知类别处理；受保护文件哈希未改变。
- 未覆盖：更多随机种子、128 epoch 对照、两种改动组合、完整 10 折、未知航程专门压力测试和线上提交。因此不应把本轮否定结果推广到所有深度学习方案。
- 当前提交文件及既有模型配方未修改，没有新 Kaggle 提交。
'''
    # Report lives at project root, so its local code link is root-relative.
    report = report.replace('(../../experiment_neural.py)', '(experiment_neural.py)')
    (ROOT/'NEURAL_EXPERIMENT_REPORT.md').write_text(report, encoding='utf-8')
    print(decision)
    print(json.dumps(uncertainty, indent=2))


if __name__ == '__main__':
    main()
