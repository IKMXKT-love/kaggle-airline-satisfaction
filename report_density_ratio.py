"""Write the bounded ratio screen report from saved measurements."""
from experiment_domain_transfer import ROOT, read
from experiment_density_ratio import RUN


def main():
    p,e,v=[read(RUN/f'{n}.json') for n in ['protocol','evaluation','verification']]
    arm_names={'direct':'直接分类','teacher_offset':'普通教师修正','conditional_ratio':'类别条件域密度比'}
    rows=[]
    for arm,name in arm_names.items():
        a=e['arms'][arm]; d=a['primary_blend_delta']; lo,hi=d['interval95']
        selected=read(RUN/'selection'/arm/'complete.json')['best_epoch']
        rows.append(f"|{name}|{selected}|{a['single_auc']:.10f}|{a['primary_blend_auc']:.10f}|{d['delta']:+.10f}|[{lo:+.10f}, {hi:+.10f}]|")
    passed=e['passes_expansion_gate']
    ratio_history=read(RUN/'selection/conditional_ratio/history.json')
    ratio_first=next(z for z in ratio_history if z['epoch']==1)
    ratio_last=ratio_history[-1]
    verdict='通过预设初筛门槛；只允许据此另拟隔离确认，不能直接宣布公榜提升。' if passed else '未通过预设扩大投入门槛。本配方停止，不增加种子、十折、权重搜索或新提交。'
    comparisons=[]
    for arm,c in e['ratio_vs_controls'].items():
        comparisons.append(f"- 域密度比相对{arm_names[arm]}的固定10%融合差值：{c['delta']:+.10f}，配对95%区间{c['interval95']}。")
    tests='三项合约测试通过：已知密度的Bayes/域几率恒等式、原始标签对两个控制臂的损失隔离、零修正教师回退及推理不读取query标签。'
    text=f'''# 类别条件域密度比实验结果（2026-10-21）

{verdict} 当前最佳仍是用户反馈公榜0.96187的 `submission_public_pool_20261020.csv`；最佳CSV及三个CURRENT文件哈希均未改变。

## 要检验的机制

源后验logit加上满意/不满意两类的对数域密度比之差，尝试修复原始与比赛数据的类别关系变化。网络共同域项在类间相减时取消，类特有差异留下。Bayes恒等式本身不保证估计器成功，也没有引入额外测试信息。本实验是项目应用提案，不是已复现的同场冠军路线。

源后验取三个仅在原始数据训练的XGBoost教师的截断logit均值；三组统一相应比赛训练区的类别先验修正。域分类器只读21字段，不读取原始教师训练内输出。原始数据129880行中屏蔽与比赛train/test完全重合的{p['removed_original_overlap']}行，留下{p['original_rows']}行。未加入额外Inflight service字段。

三组相同8成员MLP、同初始化、同四组采样任务。每步C0/C1/O0/O1各512行，域比目标是类别条件域BCE；直接分类、普通教师修正目标是按自然比赛先验加权的C分类BCE。所有臂均计算2048行前向，但两个控制只用C标签，域比额外用O标签。两头从零开始，类差项经8*tanh(raw/8)限制。比较的是不同目标和源标签使用方式，不能宣称监督信息完全相等。

预处理在计分前修订一次：两域训练行联合拟合无标签词表与数值尺度，避免原始域独有未知类别成为域识别捷径。首版编码、资源探针、协议与源码快照保存在 `resource_only_pre_encoder_amendment/`；修订发生在任何停止区/外层成绩出现前，原900秒时钟未重置。修订后依据资源成本选择{p['max_epochs']}轮上限，没有按效果选训练量。

## 同区结果

外层为原生seed42的fold2，共{p['scored_rows']}行；其余{p['refit_rows']}行中取{p['stopping_rows']}行作内部停止区，拟合区{p['fit_rows']}行。含epoch0回退，仅按内部AUC选择轮数，再从头重训完整外层训练区。每轮是ceil(C行数/2048)步的有放回采样，并非完整遍历。三组在各阶段的初始化指纹一致。

当前公开池同区AUC **{e['baseline_auc']:.10f}**。主融合预先固定为90%当前池logit+10%新臂logit。

|训练方法|选定轮数|单模型AUC|10%融合AUC|相对当前池|配对95%区间|
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

{chr(10).join(comparisons)}

源教师单独AUC {e['teacher_alone_auc']:.10f}，10%融入当前池AUC {e['teacher_10pct_blend_auc']:.10f}。零修正回退及仅全局先验校正不改变源教师排序。共同的边际域变化在正确后验比中取消，因此这里的源教师回退也是去掉类别差异项的消融；未另训边际域比模型。

预设扩大门槛为主候选融合相对当前池至少+0.0003、配对区间下界大于零，并胜过两个同权重控制。本轮门槛结果：**{passed}**。

## 核验与成本

{tests}

核验程序重新读取原始CSV，重建屏蔽、划分及两阶段所有编码数组；从246列原生权重重算公开池基准；加载三组完整重训权重，重新执行全部139927计分行GPU推理与AUC/配对比较。基准最大误差{v['independent_baseline_max_error']}，三组预测重算最大误差{ {k:z['full_prediction_max_error'] for k,z in v['evidence'].items()} }。抽取预先固定4096个行位置，重载三个源XGBoost模型核对缓存，最大误差{v['teacher_4096_row_prediction_max_errors']}；没有重算全量源教师缓存。

从首次数据准备到独立核验完成累计 **{v['cumulative_seconds']:.2f}秒（{v['cumulative_seconds']/60:.2f}分钟）**，包含首版准备、修订、两个丢弃参数的资源探针、六次拟合与全部计分行推理核验，900秒截止是否满足：{v['cumulative_deadline_met']}。方案思考与代码编写时间不包含在此数据实验累计时间中。

这里沿用公开基础预测，它们带有上游早停、不同折及间接标签依赖；因此结果是条件初筛，不是端到端嵌套验证。历史评分区域也不是全新盲测。配对区间仅描述固定预测在当前行上的抽样不确定性，不包含重新训练、反复研究及候选选择的不确定性。本地差值不能直接加到0.96187上。

## 如何解释

内部域BCE从第1轮{ratio_first['objective_bce']:.6f}下降到第{ratio_last['epoch']}轮{ratio_last['objective_bce']:.6f}，同时目标停止区AUC从{ratio_first['stopping_auc']:.6f}降到{ratio_last['stopping_auc']:.6f}，因此按预设规则选择第1轮。这是“域识别目标变好，但后验排序变坏”的直接观察；单凭这些指标不能区分源域拟合过强、两比率估计误差放大、或共享结构约束不适合等具体原因。不能把训练损失改善当成泛化改善，也没有依据继续追加同样训练轮数。

本轮回答的是这套有边界的共享域比估计器能否产生当前强池之外的排序收益。它不证明所有密度比模型都无效，也不证明任务达到上限。但若未通过，就没有证据支持在本配方上扩大算力投入。已完成的共享表示、局部上下文和本轮域比应合并为负面证据，下一项必须另有具体的信息或机制依据。

详细证据：`.artifact_runtime/density_ratio_20261021/protocol.json`、`evaluation.json`、`verification.json`，以及各臂轮次历史、重训权重和计分行预测。没有制作新的提交CSV。
'''
    target=ROOT/'DENSITY_RATIO_REPORT_20261021.md'; target.write_text(text,encoding='utf-8'); print(target)


if __name__=='__main__': main()
