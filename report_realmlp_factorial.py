"""Render verified factorial results in plain Chinese; no training."""
from experiment_realmlp_factorial import ROOT,OUT,read
from pathlib import Path

def main():
    directory=Path(read(OUT/'latest.json')['directory'])
    s=read(directory/'summary.json');v=read(directory/'verification.json')
    assert v['verified']
    names={'existing_old':'现有66特征＋原配方','existing_reference':'现有66特征＋新配方',
           'lean_old':'精简47特征＋原配方','lean_reference':'精简47特征＋新配方'}
    rows={r['name']:r for r in s['rows']}
    best=rows[s['best']]
    stopped=s['status']=='stopped_screen'
    text=['# RealMLP 四组对照实测','',
        '日期：2026-10-18。三组新增训练已完成，现有输入＋原配方复用核验后的同协议预测。',
        '',f"结论：最佳候选是**{names[s['best']]}**，按预设 80% 基线＋20% 新成员融合，改善 **{s['best_gain']:+.9f}**。",
        '未达到预先固定的 +0.0002 扩展门槛，停止本轮扩展；没有启动确认、10折或生成新提交。' if stopped else
        '已达到初筛门槛，需要按冻结方案进行第二分区确认；目前尚不是最终提交改善。',
        '',f"同协议五分支基线 AUC：**{s['baseline_auc']:.9f}**。",'',
        '| 设置 | 单模型 AUC | 固定融合 AUC | 融合增益 | 样本层面约95%成对区间 |',
        '|---|---:|---:|---:|---|']
    for name in ['existing_old','existing_reference','lean_old','lean_reference']:
        r=rows[name];lo,hi=r['paired']['interval95']
        text.append(f"| {names[name]} | {r['auc']:.9f} | {r['blend_auc']:.9f} | {r['paired']['delta']:+.9f} | [{lo:+.9f}, {hi:+.9f}] |")
    text+=['','## 对照能说明什么','',
        '以上增益均以原五分支基线为参照。基线已经含14%的原RealMLP，所以再混入同一原模型只是在改变权重，其近零收益不是独立的新成员贡献。',
        '下面进一步比较四格之间的差值，检验相同配方下改变输入，以及相同输入下改变训练配方的影响。',
        '', '| 对照 | 单模型 AUC 差值 | 固定融合 AUC 差值 |', '|---|---:|---:|']
    for key,c in v['factorial_comparisons'].items():
        a,b=key.split(' minus ')
        text.append(f"| {names[a]} 相对 {names[b]} | {c['solo']['delta']:+.9f} | {c['blend']['delta']:+.9f} |")
    text+=['','单模型更高不必然使融合更好；差值只来自本次固定划分，不能推断所有种子或其他数据上的优劣。',
        '', '## 方法及验证边界','',
        '- 411944行拟合，35822行内部停止，111942行外部计分。与上轮五分支基线的行序、种子和预测一致。',
        '- 精简输入包括21个原始字段、17个数值类别副本、5个计数、1个保留的交叉类别、3个交叉目标编码，共47列。词表与计数只从拟合区学习，TE采用内部5折。',
        '- 新配方为8个内部成员、固定3轮训练及配套学习率/正则/嵌入设置；原配方为128轮预算。预算不同是配方的一部分，本实验不是等算力架构比较。',
        '- 新配方的日志确认固定3轮；库输出的fit_params诊断可为0，这不表示使用随机初始化预测。固定stop_epoch回调决定模型保留轮次。',
        '- 外部计分标签不用于训练或早停。历史划分此前已被看过，不是新的盲测。',
        '- 成对区间仅反映固定预测上的样本不确定性，不覆盖重新训练、候选筛选及未来分布变化。',
        '- 门槛在训练前固定：初筛融合+0.0002，冻结后确认+0.0001。它是预算规则，不是显著性定理；未过门槛不等于方法完全无效。',
        '- 三项特征契约测试通过；独立核验全部预测范围、行序、AUC、融合重建、输入/源码哈希及原有受保护文件。',
        '- 上次运行在最后一组中断；本次只补跑缺失组，前三组缓存已复用。',
        '', '参考配方：[同场公开源码](https://github.com/alvis1111/s6e10-airline-solution/blob/main/independent_repro/realmlp_repro_lean.py)。本地实现将特征统计限制在拟合分区。',
        '', '当前公榜最佳仍为用户反馈的 **0.96082**。本地改善不能直接加到该分数上，也不能据此声称达到0.962。',
        '', f'预测、协议及核验明细：`{directory}`。']
    path=ROOT/'REALMLP_FACTORIAL_REPORT.md'
    path.write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(path)

if __name__=='__main__':main()
