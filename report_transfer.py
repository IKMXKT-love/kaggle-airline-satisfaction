"""Report finished transfer experiments and independently reconstruct blends."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiment_transfer import ROOT, OUT, prediction, read, sha
from report_neural import paired

NAMES=dict(control='XGBoost 对照',frequency='原始/合成频率比',profiles='评分条件分布',
           combined='频率比＋评分分布',teacher_offset='教师 logit 修正',high_resolution='4096 分箱')


def main():
    directory=Path(read(OUT/'latest.json')['directory'])
    protocol=read(directory/'protocol.json')
    decision=read(directory/'decision.json')
    for name in ['experiment_transfer','transfer_features']:
        assert sha(directory/(name+'_snapshot.py'))==protocol['hashes'][name+'.py'],name
    for n in protocol['protected']:
        assert sha(ROOT/n)==protocol['hashes'][n],n
    train=pd.read_csv(ROOT/'train.csv')
    sections=[];checks={}
    for stage in ['screen','confirmation']:
        d=directory/stage
        if not (d/'summary.json').exists():continue
        summary=read(d/'summary.json');idx=np.load(d/'indices.npz')['scored'];y=train.satisfaction.to_numpy()[idx]
        control=prediction(d,'control',idx)
        independent=.28*control
        for name,w in [('old_xgb',.07),('old_cat',.21),('realmlp',.14),('lgb',.30)]:
            independent+=w*prediction(d,name,idx)
        base=prediction(d,'baseline',idx)
        np.testing.assert_array_equal(independent,base)
        np.testing.assert_allclose(roc_auc_score(y,base),summary['baseline_auc'],atol=1e-12)
        checks[stage]={}
        rows=[]
        control_score=float(roc_auc_score(y,control))
        for trial in summary['trials']:
            p=prediction(d,trial['name'],idx)
            blended=independent+.28*(p-control)
            score=float(roc_auc_score(y,blended))
            np.testing.assert_allclose(score,trial['blend_auc'],atol=1e-12)
            delta=paired(y,blended,independent)
            checks[stage][trial['name']]=delta
            detail=read(d/(trial['name']+'.json'))
            rows.append(f"| {NAMES[trial['name']]} | {len(detail['features'])} | {detail['rounds']} | {trial['standalone_auc']:.9f} | {trial['standalone_auc']-control_score:+.9f} | {score:.9f} | {trial['gain']:+.9f} |")
        intervals='\n'.join(f"| {NAMES[k]} | [{v['interval95'][0]:+.9f}, {v['interval95'][1]:+.9f}] |" for k,v in checks[stage].items())
        sections.append(f'''## {'筛选' if stage=='screen' else '冻结方案后复核'}结果

拟合 {summary['rows']['fit']:,} 行，内部停止验证 {summary['rows']['stop']:,} 行，计分 {summary['rows']['scored']:,} 行。

同协议 XGBoost 对照 AUC：{control_score:.9f}；重训的 v4 配方融合对照 AUC：{summary['baseline_auc']:.9f}。

| 设置 | 特征数 | 最佳轮数 | 单模型 AUC | 相对 XGB 对照 | 替换后融合 AUC | 相对融合对照 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

固定预测的成对 AUC 差值约 95% 区间，仅反映样本层面的渐近不确定性，不覆盖多候选筛选、训练随机性和历史数据反复查看：

| 设置 | 融合差值区间 |
|---|---|
{intervals}
''')
    status=decision['status']
    conclusion=dict(stopped_screen='五个候选均未达到预设的筛选融合增益门槛，停止本轮扩展，保留 v4。',
                    stopped_confirmation='筛选入选方案未通过冻结后的复核，停止本轮扩展，保留 v4。',
                    ready_for_cv='筛选及复核通过，当前只是进入完整 10 折的候选，尚不能宣称超过 v4 公榜。')[status]
    report=f'''# 原始数据迁移方法实测

日期：2026-10-18。结论：**{conclusion}**

## 做了什么

按照跨比赛研究，固定训练并比较五个 XGBoost 候选。其他参数不变；只在筛选阶段比较候选，不在复核阶段追加调参。已有提交未覆盖。

- 频率比：六种分组的平滑归一化频率比、原始支持数、原始未见标记、训练稀有标记，共 24 列。
- 评分分布：座椅舒适度、机上娱乐在原始航程组/原始客群组及当前训练航程组的标准差、高分比例、0 分比例、标准化偏离；加分组支持数及真实舱位比例，共 30 列。分组统计向总体收缩，强度 20。
- 两组组合：新增 54 列。该组合在训练前已指定，没有搜索其他组合。
- 教师修正：相同 66 特征，但用原始数据教师 logit 作为 base_margin；训练、停止验证、预测均传入对应 margin。
- 高分箱：仅将 XGBoost max_bin 从 256 改为 4096。

## 如何保持可比

- 所有分支使用相同拟合/内部停止/外部计分划分。监督特征、训练频率和条件统计都只使用拟合部分及已去除精确重合的公开原始数据。
- 所有外部计分标签均未用于停止轮数选择。RealMLP 使用 128 epoch 预算，与已有模型预算一致；其余树模型最多 4000/5000 轮并在内部停止集早停。
- 重训 v4 的全部五个分支，固定权重 7% 旧 XGB、21% CatBoost、28% 新 XGB、14% RealMLP、30% LightGBM。新候选只替换 28% 的新 XGB，不搜索融合权重。
- 先前 v4 的本地分数含外层早停，本次又留出了 8% 停止数据。因此新旧协议绝对分数不能直接相减，实验只与本次重训对照比较。
- 历史筛选/复核分区已被多次看过，不能称为全新的独立测试。
- 筛选进入门槛预先固定为融合 +0.0002；入选方案冻结后，复核门槛为 +0.0001。门槛是算力投入规则，不是统计显著性定义。不同阶段门槛不代表分数应固定缩放。

{chr(10).join(sections)}

## 核验与限制

五项功能测试通过，覆盖频率归一化、未知分组回退、transform 不修改统计、目标列排除及 logit margin 的正确含义。另已从各分支缓存独立重建融合，并核对行序、预测范围、AUC 和受保护文件哈希。

此次尚未测试近邻特征、更多随机种子、其他主模型、更多超参数或所有特征组合。单划分失败只限制当前配置，不能推断所有迁移学习或深度学习都无效。

模型与预测缓存：`{directory}`。
复现入口：[experiment_transfer.py](experiment_transfer.py)；特征实现：[transfer_features.py](transfer_features.py)。代码复跑按输入、配方和源码指纹复用缓存。

本次实际训练的源码快照保存在明细目录，已核对其哈希。训练结束后，入口进一步补充了外部数据、划分和辅助预测缓存的指纹；因此当前入口首次复跑会产生新目录，不会误用旧指纹结果。此项缓存检查变更未触发新的模型训练。

原始 v4 公榜成绩仍为用户反馈的 0.96082。此次未向 Kaggle 自动提交。
'''
    (ROOT/'TRANSFER_EXPERIMENT_REPORT.md').write_text(report,encoding='utf-8')
    from v2_training import write_json
    source_hashes=read(directory/'screen/summary.json')['source']['signature']['original_file_hashes']
    for name,h in zip(['external_data/train.csv','external_data/test.csv'],source_hashes):
        assert sha(ROOT/name)==h,name
    write_json(directory/'verification.json',dict(verified=True,protected_files_unchanged=True,source_files_verified=True,
               independently_reconstructed_stages=list(checks),paired_uncertainty=checks,status=status))
    print(conclusion)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
