"""Render verified descriptive and conditional-learning outcomes."""
import time
import numpy as np
from audit_pool_repairability import ROOT,RUN,read,write,sha

def main():
    p=read(RUN/'protocol.json');v=read(RUN/'verification.json');assert v['verified']
    a=read(RUN/'pair_audit.json');s=read(RUN/'strong_member_diagnostic.json');c=read(RUN/'completion.json')
    total_wrong=sum(f['baseline_wrong_pairs'] for f in a['folds'])
    any_correct=sum(f['baseline_wrong_some_member_correct'] for f in a['folds'])
    all_wrong=sum(f['all246_strictly_wrong_pairs'] for f in a['folds'])
    damaged=sum(f['baseline_correct_some_member_wrong'] for f in a['folds'])
    top_correct=sum(f['wrong_pairs_corrected_by_any_top12'] for f in s['folds'])
    top_damage=sum(f['correct_pairs_broken_by_any_top12'] for f in s['folds'])
    summary={'sampled_pairs':10000000,'current_wrong':total_wrong,'some_of246_correct':any_correct,'some_of246_correct_fraction':any_correct/total_wrong,'all246_strictly_wrong':all_wrong,'all246_strictly_wrong_fraction':all_wrong/total_wrong,'current_correct_some_member_wrong':damaged,'top12_correct_fraction':top_correct/total_wrong,'top12_correct_pairs_broken':top_damage,'conditional_result':c['results'],'status':c['status'],'total_elapsed_seconds':time.time()-p['started_epoch'],'protected_files_unchanged':True,'new_submission':None}
    for n,h in p['protected'].items():assert sha(ROOT/n)==h,n
    write(RUN/'final_summary.json',summary)
    lines=['# 当前246列错排与可学习修复：结果','',
      f"已完成五个原生折各200万对、合计1000万对的描述性审计，以及一个固定条件纠错候选和必要对照。结束状态 `{c['status']}`。至全部核验与报告约{summary['total_elapsed_seconds']:.1f}秒。没有新提交。用户反馈最佳仍为0.96187。",'',
      '## 排错是否意味着整个模型池都错','',
      '| 原生折 | 当前排错对数 | 246列全部严格排错 | 至少一列排对占比 | 权重最高12列至少一列排对占比 |',
      '|---|---:|---:|---:|---:|']
    for f,t in zip(a['folds'],s['folds']):
        lines.append(f"|{f['fold']}|{f['baseline_wrong_pairs']}|{f['all246_wrong_fraction_of_baseline_wrong']:.2%}|{f['some_member_correct_fraction_of_baseline_wrong']:.2%}|{t['fraction_of_baseline_wrong']:.2%}|")
    lines+=['',f"合计当前错排{total_wrong:,}对，其中{any_correct:,}对至少一列排对（{any_correct/total_wrong:.2%}），{all_wrong:,}对全部严格排错（{all_wrong/total_wrong:.2%}）。另有少量单列并列，故两项不必精确相加为100%。当前已排对但至少一列排错的有{damaged:,}对，比池内可修复错排更多。",'',
      '上述70%是读取标签后选“恰好正确”的模型得到的oracle统计，不是可部署的模型、不是收益预测，也不是整个模型家族的可达AUC上界。当前LR本身含有带符号权重，按行条件融合也不受固定非负融合限制，因此“全部列排错”不能叫作不可修复误差。', '',
      '为避免宽池oracle被低权重成员的分歧误导，内部结果出来后追加了一个明确标为事后描述的敏感性检查：每折只取该折训练区LR的12个最大正系数成员。在相同配对上，至少一个排对的比例为41.69%—44.38%，总体约'+f"{top_correct/total_wrong:.2%}"+'。此检查没有用于更换候选、继续拟合或改变门槛，也仍然需要标签选择专家。', '',
      '## 学习这些分歧是否有用','',
      '固定候选为深度3、强正则XGBoost offset纠错器，读取21个原始字段、当前LR分数及全部246个成员减该LR分数的差值，共268列。必要对照只有原始字段和LR分数，共22列。两个模型除输入外参数相同，各自在内部停止区选择树数。无融合权重、种子或结构搜索。', '',
      '原生fold1/2/3的419781行拟合LR及纠错器，fold4的139927行用于内部早停和准入筛选。此阶段LR重新拟合，没有使用包含fold4标签的旧fold0权重。拟合区margin来自同区LR，属于正常逐阶段拟合；停止区margin来自该训练区LR。原生fold0没有进入本次候选拟合或轮数选择，但历史研究以及描述性配对分析已观察过它，不能称盲测。', '',
      '| 模型 | 内部AUC | 相对LR差值 | 内部选择树数 |',
      '|---|---:|---:|---:|']
    r=c['results'][0]
    lines.append(f"|当前LR|{r['baseline_auc']:.10f}|—|—|")
    for arm,label in [('raw_control','仅原始字段+总分纠错'),('conditional','加246列分歧纠错')]:lines.append(f"|{label}|{r[arm+'_auc']:.10f}|{r[arm+'_delta']:+.10f}|{r[arm+'_rounds']}|")
    main_ci=next(z['delta_and_fixed_prediction_ci'] for z in v['fitted_prediction_checks'] if z.get('arm')=='conditional')
    lines+=['',f"主候选相对LR差值{r['conditional_delta']:+.10f}，固定预测的近似配对95%区间{main_ci['interval95']}；相对必要对照{r['conditional_vs_control']:+.10f}。区间包含零；同时未达到事先+0.00005及对照+0.00001准入门槛，停止该配方。没有重拟合outerT、没有评分outer fold0候选、没有继续其他折，也没有新测试预测。",'',
      '这是在同一内部停止区选树数后的探索结果，存在选择乐观性；区间还没有覆盖训练变化、上游OOF依赖和历史多次尝试，不应按独立确认或真实增益解读。', '',
      '## 核验与决策','',
      '全部1000万个配对的布尔判断和246列逐成员计数，采用另一种按行块的实现重建。五折LR评分全部从保存权重重算；最大误差约2.22e−15。原生划分与StratifiedKFold(5,shuffle=True,random_state=42)一致。两个纠错器在全部139927行使用独立编码实现及CPU树推理重建，最大误差约8.88e−16，AUC完全一致。内部LR训练梯度最大约4.54e−14；拟合/评分索引互斥，类别词表仅由拟合行构造。最佳四文件SHA256未变。', '',
      '当前结论：池内互补错误真实存在，但这一个低容量、强正则的条件纠错配方没有将它转为足够增益。不能据此否定所有条件门控，也不能把描述性的70%写成尚待回收的提分空间。停止继续调这套配方，保留0.96187；后续新训练应先有具体的池外监督信息或更强的跨折可预测错排证据。', '',
      '完整协议、输入指纹、索引、配对、权重、XGBoost模型、评分数组及核验JSON均保存在 `.artifact_runtime/pool_repairability_20261022/`。公开基础预测没有重新训练，整体仍是有上游依赖的条件二层检验，不能称全链路嵌套验证。']
    (ROOT/'POOL_REPAIRABILITY_REPORT_20261022.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary)

if __name__=='__main__':main()
