"""Write a report from independently checked correction results."""
from pathlib import Path
from experiment_correction import ROOT, OUT, read, preserve_check


def main():
    d = Path(read(OUT/'latest.json')['directory'])
    r = read(d/'result.json'); v = read(d/'verification.json')
    assert v['verified']
    preserve_check()
    names = dict(plain='普通堆叠', offset='固定起点纠错')
    rows = []
    for f in r['folds']:
        for mode in names:
            a = f['modes'][mode]; p = a['vs_base']
            rows.append(f"| {f['outer']} | {names[mode]} | {f['baseline_auc']:.9f} | {a['auc']:.9f} | {p['delta']:+.9f} | {a['selected_rounds']} |")
    pooled = []
    for mode in names:
        a = r['modes'][mode]; p = a['vs_base']; lo, hi = p['interval95']
        pooled.append(f"| {names[mode]} | {a['auc']:.9f} | {p['delta']:+.9f} | [{lo:+.9f}, {hi:+.9f}] | {a['passed']} |")
    decision = ('有候选通过首轮，需按冻结方案复核第二种子，尚不能宣称公榜改善。' if r['any_passed'] else
                '两种候选均未通过预设门槛，停止本轮纠错配方，不扩展完整十折，也不生成新提交。')
    text = f'''# 当前强组合纠错实验

日期：2026-10-20，Asia/Taipei。两个外层折训练及独立核验完成。

{decision}

当前最佳仍为用户反馈公榜0.96114的 `submission_lean_realmlp_10fold.csv`，SHA256在本轮核对不变。没有新的Kaggle成绩。

## 同折结果

| 外层折 | 候选 | 同协议基线AUC | 候选AUC | 差值 | 内部选择轮数 |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

合并{r['rows']:,}行计分，基线AUC {r['baseline_auc']:.9f}：

| 候选 | AUC | 相对基线 | 固定预测差值约95%区间 | 通过门槛 |
|---|---:|---:|---|---|
{chr(10).join(pooled)}

纠错相对普通堆叠的合并AUC差值：{r['offset_minus_plain']['delta']:+.9f}。不能仅比较纠错与基线，就把新增收益全部归因于固定起点。

## 具体做法

- 固定分层十折中的第0、1折；每折约629671行外层训练、69964行外层计分。外层训练再留8%仅用于二层停止。
- 初选阶段：拟合区内部五折重新训练全部基础分支，生成二层训练用OOF；停止区完全不进入这些基础模型。仅用拟合区重训的基础模型预测停止区，选择二层轮数。
- 重拟合阶段：在完整外层训练区再次生成内部五折OOF，按已选轮数训练二层；基础模型也在完整外层训练区重拟合，预测外层计分行。
- 每次基础组合有七个实际模型：旧XGB、CatBoost、新XGB、旧RealMLP、LightGBM及两种子精简RealMLP。前五路沿用7/21/28/14/30%的概率组合，再与精简RealMLP均值按65/35% logit组合。
- 为控制嵌套成本，基础树轮数固定为历史独立停止实验的435/907/623/641；旧RealMLP保留128轮学习率日程、固定第63轮停止；精简网络固定3轮。两个网络路径val_fraction=0，使用传入拟合区全部行。这是按既有成分重建的固定训练轮数基线，不是重新生成与线上CSV逐行相同的预测，也不是对整个基础配方重新调优。
- 新CatBoost输入为21原始字段、17数值类别副本与当前组合logit，共39列。普通堆叠和offset使用相同输入、深度6、学习率0.03、L2=20、最大2500轮和150轮耐心；仅固定baseline设置不同。
- 以内部停止AUC保留最佳权重，候选不如基线时允许回退到原预测（0轮）；不限制只从后半程选权重。外层结果在两折均完成后才计算，没有根据第一个外层折改变配方。
- offset训练和预测都使用原始logit；核对Pool自动加baseline与手工加回路径一致，避免漏加或重复叠加。

## 门槛与核验

两折均改善、合并增益至少0.0003、固定预测配对区间下界大于0，才进入第二种子及完整十折。门槛在外层计分之前固定，属于计算投入规则。

四项契约测试通过：嵌套划分与污染检测；拟合区词表/缺失/目标排除；真实CatBoost的baseline推理与全局严格递增变换保持AUC；当前组合公式重建。小数据探针进一步检查全部七路训练、保存与重新加载预测。

独立核验逐个重建两阶段内部折的拟合/查询范围与OOF覆盖，重建所有基础组合，检查源码与输入指纹及旧根文件。十四个最终基础模型从保存权重重新预测全部外层计分行，并重载二层模型，重建AUC、成对差值与门槛决定。详见verification.json。

## 解释边界

数据与历史超参数已经在先前研究中使用，重分折不能变成新盲测；本轮隔离的是新增模型的标签和特征生成路径。两个外层模型的训练区重叠，配对区间只覆盖固定预测的样本不确定性，不覆盖历史筛选、模型重训和未来分布，也未对两个候选作多重筛选校正。

既有公开原始库按所有比赛行的无标签特征排除了精确重合，再生成教师及辅助预测；这个固定的重合排除规则使用了比赛特征，没有使用比赛标签。本轮没有重新利用外层行拟合比赛侧词表、编码或分组统计。核验还重新计算了辅助预测缓存与当前比赛/原始特征的对应指纹。

二层训练的基础OOF来自较小训练区，而外层基础预测来自更大训练区，两者训练规模及预测分布不同，已记录logit分布。没有利用外层标签事后调整它。固定轮数也限制本轮结论，不能推断所有纠错模型无效。

证据目录：`{d}`。

复现：`experiment_correction.py all`；独立核验：`verify_correction.py`；报告：`report_correction.py`。已有匹配指纹的模型缓存会复用。
'''
    (ROOT/'STRONG_CORRECTION_REPORT.md').write_text(text, encoding='utf-8')
    print(ROOT/'STRONG_CORRECTION_REPORT.md')


if __name__ == '__main__':
    main()
