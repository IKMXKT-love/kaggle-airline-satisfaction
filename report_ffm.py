from pathlib import Path
from experiment_ffm import ROOT, OUT, read


def main():
    d = Path(read(OUT/'latest.json')['directory']); r = read(d/'result.json')
    assert read(d/'verification.json')['verified']
    rows = []
    for f in r['folds']:
        for mode in ['additive', 'ffm']:
            a = f['modes'][mode]
            epoch = read(d/f"outer_{f['outer']}"/('selection_'+mode)/'training.json')['best_epoch']
            rows.append(f"| {f['outer']} | {mode} | {a['single_auc']:.9f} | {a['blend_auc']:.9f} | {a['gain']['delta']:+.9f} | {epoch} |")
    pooled = []
    for mode, a in r['modes'].items():
        p = a['gain']; lo, hi = p['interval95']
        pooled.append(f"| {mode} | {p['delta']:+.9f} | [{lo:+.9f}, {hi:+.9f}] | {a['passed']} |")
    passed = any(a['passed'] for a in r['modes'].values())
    decision = ('有候选通过两折门槛，需进一步按冻结配置复核。' if passed else
                '两组均未通过预设的实际组合收益门槛，停止当前配方，不扩展完整十折或生成新提交。')
    text = f'''# 显式字段交互实验

日期：2026-10-20。additive单字段对照与FFM两组、两个完整外层折实际训练及独立核验完成。

{decision}

当前最佳仍为用户反馈公榜0.96114的submission_lean_realmlp_10fold.csv；本轮没有新的线上成绩。

| 外层折 | 模型 | 单模型AUC | 固定融合AUC | 相对同折当前配方基线 | 内部选择轮数 |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

两折合并，固定20%新分支、80%同协议当前组合的logit融合：

| 模型 | 融合增益 | 固定预测差值约95%区间 | 通过门槛 |
|---|---:|---|---|
{chr(10).join(pooled)}

FFM融合相对additive融合的差值为{r['ffm_minus_additive']['delta']:+.9f}；不能把单模型差值直接相加到公榜上。

## 模型与协议

21原始字段均有独立类别表示，四个连续字段另有稳健尺度和单字段hinge基函数。评分0、真实缺失和未知类别分开。词表、数值中位数及IQR仅由当前拟合区学习，无目标编码、原始数据教师或测试特征预训练。

additive输出由各字段主效应相加；FFM增加字段感知的低秩二阶交互。两者各有4个独立参数成员，使用相同批次顺序；公共主效应参数采用相同初始化。FFM额外参数来自随机初始化，不是相同容量对照。它是PyTorch迁移实现，不是冠军完整LIBFFM流水线的复现。

维度8、batch2048、AdamW初始学习率0.003、权重衰减0.02、标签平滑0.005；最多24轮，内部停止耐心4轮，从整个合法训练轨迹保留最佳权重。不同组可在不同轮停止，因此比较的是给定最大预算和停止规则下的实际用途，不是严格等算力归因。

采用与纠错实验相同的十折中第0、1外层折。每折内部留8%停止，选择轮数后在完整外层训练区重拟合；学习率仍按照24轮原日程取前缀。外层标签不进入拟合、尺度、词表或早停。两组在两折完成后统一计分。

所有配置在纠错外层结果产生之前冻结；没有依据其结果改变FFM配方。当前组合基线直接复用纠错实验同折、完整外层训练区的隔离预测。历史固定基础轮数和既有数据选择的限制与纠错报告相同，不能将它与旧线上CSV视为逐行相同的拟合结果。

## 检查与解释

契约测试检查未知/缺失/零值与拟合区词表隔离，以及真实小规模XOR任务：FFM能学到纯加性对照无法表达的交互。该功能测试不是本场提分证据。

独立核验重建分区、重新拟合全部编码器、检查内部最佳轮数和重拟合轮数、从保存权重重建停止区及全部外层计分预测，复算固定融合、AUC、配对区间及门槛；旧文件指纹保持一致。

两外层训练区重叠，历史数据已被研究多次，不能称新盲测。区间仅覆盖固定预测的样本不确定性，不覆盖候选筛选、训练随机性或未来分布。单个rank/正则/优化配方的负结果不能证明所有交互模型无效。

证据目录：`{d}`。

复现：experiment_ffm.py all；核验：verify_ffm.py；报告：report_ffm.py。源文件指纹与全部模型权重保存在证据目录。
'''
    (ROOT/'FIELD_INTERACTION_REPORT.md').write_text(text, encoding='utf-8')
    print(ROOT/'FIELD_INTERACTION_REPORT.md')


if __name__ == '__main__':
    main()
