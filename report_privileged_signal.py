"""Report the original-only signal and deployability gates without claiming competition gain."""
from experiment_domain_transfer import ROOT,read
from experiment_privileged_signal import RUN
from experiment_privileged_marginal import OUT


def main():
    p,e,v=map(lambda n:read(RUN/f'{n}.json'),['protocol','evaluation','verification'])
    mp,me,mv=map(lambda n:read(OUT/f'{n}.json'),['protocol','evaluation','verification'])
    baseline=e['arms']['shared21']; oracle=e['arms']['privileged22']; placebo=e['arms']['shuffled22']
    metrics=me['metrics']; delta=e['privileged_vs_controls']['shared21']; transferred=me['comparison_to_shared21']
    deployable=me['passes_deployable_source_gate']
    verdict='条件积分通过源域可部署性参考门槛；这仍不是比赛提分，需要单独检查对当前公开池的增量。' if deployable else '真实额外评分提供明确的源域信息，但当前不读取真实评分的条件积分配方未通过准入门槛；不进入比赛训练。'
    report=f'''# 原始独有评分：信号与可部署性实测

日期：2026-10-21，Asia/Taipei。

{verdict}

**本报告所有AUC均是原始航空数据的留出成绩，不是比赛验证AUC，也不是Kaggle公榜分数。** 当前最佳仍为用户反馈公榜0.96187，原CSV及三个CURRENT文件哈希保持不变。没有生成新比赛提交。

## 新信息与预设问题

比赛提供共同21字段；原始数据多出一个Inflight service评分。现有本地v3辅助目标来自已裁为21字段的表，没有使用这一独有监督。本轮先问额外评分对源域满意度是否有条件预测价值，再问不读取query真实评分的模型能否保留足够收益。

同场作者amanatar的本地保存代码把这一额外列丢弃，其“生成器未见过该列”是作者注释，未据此声称已独立恢复官方生成机制。真实额外评分不能直接放进比赛行。

## 第一阶段：真实评分是否有价值

合并原始两个CSV并屏蔽与比赛train/test共同21字段完全重合的{p['removed_overlap']}行，保留{p['original_rows']}行。比赛标签从未读入本轮模型。固定seed{p['seed']}，原始域外层留出{p['row_counts']['scored']}行，外层训练{p['row_counts']['outer_train']}行，内部拟合{p['row_counts']['fit']}、停止{p['row_counts']['stopping']}行。预处理只拟合相应训练区，移除ID与Unnamed索引列。

三组同XGBoost深6、lr.04、min_child5、subsample.9、列采样1、lambda5，最多1200树、内部AUC早停100树，按各自选定树数重训外层训练区。打乱列在拟合/停止/计分区分别固定随机置换，不使用标签；完整重训重新固定置换，以控制增加一列及边际分布本身的影响。

|源域教师|选定树数|源留出AUC|logloss|比赛查询是否具备全部输入|
|---|---:|---:|---:|---|
|共同21字段|{baseline['trees']}|{baseline['auc']:.10f}|{baseline['logloss']:.8f}|具备|
|21字段+真实额外评分|{oracle['trees']}|{oracle['auc']:.10f}|{oracle['logloss']:.8f}|不具备，仅作信息对照|
|21字段+打乱额外评分|{placebo['trees']}|{placebo['auc']:.10f}|{placebo['logloss']:.8f}|没有可用信息内容，仅作安慰剂对照|

真实22字段相对21字段AUC差 **{delta['delta']:+.10f}**，固定预测配对95%区间 **{delta['interval95']}**；相对打乱22字段差 **{e['privileged_vs_controls']['shuffled22']['delta']:+.10f}**，区间 **{e['privileged_vs_controls']['shuffled22']['interval95']}**。同时改善logloss，通过预定+0.0005及两个配对区间下界>0的源信息准入规则。这是给定模型/划分的额外预测价值证据，不是互信息的无偏估计。

## 第二阶段：不读取真实评分能否保留收益

推理仅输入21字段，先训练六分类器给出p(z|x)，再逐个假设z=0至5，让固定22字段教师得到q(y=1|x,z)，计算sum_z p(z|x)*q(y=1|x,z)。推理接口拒绝额外真实评分和满意度标签，没有将预测最可能评分当成真实观测。

额外评分模型仅用源训练行的21字段与评分标签，最多800树，按内部mlogloss早停80树，选{metrics['trees']}树后完整重训。它不使用满意度标签；教师复用第一阶段同分区模型。该阶段协议在任何本阶段成绩前保存，沿用第一阶段时钟。源外层已在第一阶段观察，因此是分阶段探索，不是另一套独立盲测。

|未知评分处理规则|源留出AUC|角色|
|---|---:|---|
|直接21字段教师|{baseline['auc']:.10f}|主对照|
|按每行六类概率作条件积分|{metrics['integrated_auc']:.10f}|唯一主候选|
|直接填预测最可能评分|{metrics['point_auc']:.10f}|固定消融|
|按源训练区全局评分分布积分|{metrics['global_mixture_auc']:.10f}|固定消融|
|使用真实额外评分|{oracle['auc']:.10f}|不可部署的信息对照|

主候选相对21字段差 **{transferred['delta']:+.10f}**，配对95%区间 **{transferred['interval95']}**。条件积分logloss {metrics['integrated_logloss']:.8f}；额外评分预测准确率{metrics['rating_accuracy']:.2%}、六分类logloss {metrics['rating_logloss']:.8f}。预定源可部署性门槛是+0.0003、配对区间下界>0、logloss不差且AUC至少不低于全局积分，结果：**{deployable}**。未按外层结果改选MAP、混合权重或增加学生网络。

## 如何解释

额外评分真实存在时可以改变排序，但比赛查询没有它。全概率公式在真实条件分布下返回P(y|x)；额外字段并没有让只观察x的最终模型突破其可用信息边界。有限样本下的好处只能来自更有效的估计/表示，且两支估计误差可能抵消或放大。因此教师更准和可部署模型更准必须分别验证。

本轮只检验这一明确分解和原始域留出，不把结果推广为所有特权蒸馏或辅助监督无效；也没有检验当前池的比赛残差增量。若当前准入未通过，则没有依据直接投入更昂贵的比赛迁移训练。真实源字段有价值本身不能兑现成0.962公榜承诺。

## 文献核对

- [Yang等，NeurIPS2022：Toward Understanding Privileged Features Distillation in Learning-to-Rank](https://papers.nips.cc/paper_files/paper/2022/file/aa31dc84098add7dd2ffdd20646f2043-Paper-Conference.pdf)：核对§3.1/3.2的全字段教师和同字段蒸馏对照，以及§4关于教师信息强度与学生收益的消融。其主要任务为NDCG、部分标签人为生成，Table2按测试NDCG选checkpoint；不能把论文相对收益当成本场AUC预期。本轮条件积分不是该算法复现。
- [Provodin等：Rethinking Knowledge Transfer in Learning Using Privileged Information](https://arxiv.org/abs/2408.14319)：已核对作者摘要及研究机构出版信息，其质疑要求区分额外信息和模型设计效应。正式OpenReview版PDF遇浏览验证，未完整获取该版正文，报告没有将摘要当完整论文审计。

## 核验与成本

第一阶段独立重建CSV屏蔽、分层划分、安慰剂置换和词表，重载全部六个保存模型重算所有对应停止/外层预测，最大误差均为0。第二阶段重载两个六分类器及22字段教师，独立重建六种假设评分下的全部预测，用另一种float64积和实现重算条件积分，并检查真实query评分/标签未进入推理。所有原始文件与冻结代码哈希、配对AUC和最终准入结论核对通过。

第一阶段六次正式拟合及对应推理合计{sum(z['seconds'] for z in [read(RUN/ph/ar/'complete.json') for ph in ['selection','refit'] for ar in ['shared21','privileged22','shuffled22']]):.2f}秒；条件积分两次拟合及三种固定规则推理合计{sum(read(OUT/ph/'complete.json')['seconds'] for ph in ['selection','refit']):.2f}秒。数据准备、指纹检查、两个丢弃参数的探针、代码准备间隔和两阶段独立核验均计入原900秒时钟，至最后核验累计 **{mv['cumulative_seconds']:.2f}秒（{mv['cumulative_seconds']/60:.2f}分钟）**，是否满足原截止：{mv['original_deadline_met']}。初始方案研究与第一阶段代码编写发生在首次本轮数据读取前，不包含在该数据实验累计时间中。

没有下载新的预测、使用比赛标签训练、创建比赛预测或提交、修改最佳文件。证据目录：`.artifact_runtime/privileged_signal_20261021/`，第二阶段在其`marginal/`子目录。
'''
    target=ROOT/'PRIVILEGED_SIGNAL_REPORT_20261021.md'; target.write_text(report,encoding='utf-8'); print(target)


if __name__=='__main__': main()
