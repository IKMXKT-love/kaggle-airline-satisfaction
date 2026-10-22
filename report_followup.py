"""Summarize verified follow-up artifacts without fitting or tuning."""
from pathlib import Path
from experiment_realmlp_factorial import ROOT,read

def main():
    directory=Path(read(ROOT/'.artifact_runtime/followup_20261019/latest.json')['directory'])
    assert read(directory/'verification.json')['verified']
    screen=read(directory/'screen/summary.json');confirm=read(directory/'confirmation/summary.json')
    decision=read(directory/'decision.json')
    lines=['# 0.96104 之后的有限复核','', '日期：2026-10-19。公榜0.96104来自用户反馈，当前记录归属于已交付的80% v4＋20%精简RealMLP试提交。',
        '', '## RealMLP 稳定性', '', '| 分区 | 五分支基线 AUC | 种子42单模型 | 第二种子单模型 | 20%种子42融合 AUC |', '|---|---:|---:|---:|---:|']
    for name,s in [('筛选',screen),('复核',confirm)]:
        lines.append(f"| {name} | {s['baseline_auc']:.9f} | {s['seeds']['42']['auc']:.9f} | {s['seeds']['20261019']['auc']:.9f} | {s['current_auc']:.9f} |")
    lines+=['','| 分区/种子 | 固定20%融合相对基线的增益 | 样本层面约95%区间 |','|---|---:|---|']
    for name,s in [('筛选',screen),('复核',confirm)]:
        for seed,r in s['seeds'].items():
            d=r['fixed20_vs_base'];lo,hi=d['interval95']
            lines.append(f"| {name}/{seed} | {d['delta']:+.9f} | [{lo:+.9f}, {hi:+.9f}] |")
    lines+=['','## 六个固定融合设置','',
        '只在原筛选分区比较10%、20%、35%以及概率/logit两种形式，使用两个种子预测的均值。选择后冻结，复核分区只计算冻结设置。',
        '', '| 新模型权重 | 形式 | 筛选 AUC | 相对当前20%单种子方案 |','|---|---|---:|---:|']
    for r in screen['trials']:
        lines.append(f"| {r['weight']:.0%} | {r['form']} | {r['auc']:.9f} | {r['vs_current']['delta']:+.9f} |")
    b=confirm['best'];lo,hi=b['vs_current']['interval95']
    lines+=['',f"冻结设置：{b['weight']:.0%}、{b['form']}。复核 AUC {b['auc']:.9f}，相对当前方案 {b['vs_current']['delta']:+.9f}，约95%区间 [{lo:+.9f}, {hi:+.9f}]。",'',
        f"当前固定配方稳定性门槛通过：{decision['current_recipe_stable']}。融合调整的预设增益门槛通过：{decision['frozen_fusion_improves_current']}。",'',
        '## 方法与限制','',
        '- 筛选：411944拟合、35822内部停止、111942计分；复核：514931拟合、44777内部停止、139927计分。',
        '- 复核基线五个分支全部按相同拟合/停止/计分区重训。新增RealMLP固定3轮，特征只使用拟合部分，目标编码内部5折。',
        '- 原筛选分支复用前次预测；行序、源码与输入哈希已经核验。',
        '- 两种融合均为固定公式，没有在整份OOF上拟合元模型，也没有依照公榜分数搜权重。',
        '- 历史分区曾被查看，不称为全新盲测；区间仅覆盖固定预测的样本不确定性，不覆盖训练及筛选不确定性。',
        '- 不同训练协议的绝对分数不能直接比较；本地变化不能直接加到公榜0.96104。',
        f'- 明细目录：`{directory}`。',
        '', '## TabICLv2','']
    latest=ROOT/'.artifact_runtime/tabicl_probe_20261019/latest.json'
    if latest.exists():
        td=Path(read(latest)['directory']);summary=td/'summary.json'
        if summary.exists():
            ts=read(summary)
            lines+=['| 上下文行数 | 状态 | 预热后1024行秒数 | 预计全筛选秒数 | 单模型AUC | 对当前融合的增益 |', '|---|---|---:|---:|---:|---:|']
            for r in ts['rows']:
                lines.append(f"| {r['context']} | {r['status']} | {r.get('probe_seconds',0):.1f} | {r.get('projected_query_seconds',0):.1f} | {r.get('auc',float('nan')):.9f} | {r.get('paired',{}).get('delta',float('nan')):+.9f} |")
            lines+=['','使用官方v2权重、4个内部视图、原始21字段和8K/32K受限上下文。受限版本不等同于全部约70万行上下文。',
                '资源门槛：拟合超过300秒或预计完整筛选推理超过1200秒即停止该规模扩展。完整筛选固定20%加入当前融合，增益达到0.0002才进入后续复核。',
                '若因资源门槛停止，只能说明本机该配置的成本不合适，不能判断完整模型家族的准确率。',f'明细：`{td}`。']
        else:lines+=['资源实验尚未完成。']
    else:lines+=['资源实验尚未启动。']
    cvlatest=ROOT/'.artifact_runtime/lean_cv_followup/latest.json'
    neighbor_latest=ROOT/'.artifact_runtime/source_neighbors_20261019/latest.json'
    if neighbor_latest.exists():
        nd=Path(read(neighbor_latest)['directory'])
        if (nd/'decision.json').exists():
            ns=read(nd/'screen/summary.json');ndec=read(nd/'decision.json')
            lo,hi=ns['paired']['interval95']
            lines+=['','## 条件原始数据近邻','',
                '在性别、客户类型、出行目的、舱位相容的原始数据组内，对17个数值字段作原始数据IQR缩放（延误先log1p），计算16/64近邻。仅用公开原始标签生成平滑概率、距离和支持量，共10列。',
                '原始数据库已由既有数据加载器排除精确重合行，使用129859行；近邻标签是参考信息，不能视作比赛旅客的真标签。',
                '同一XGBoost从66列变为76列，其余训练配置和拟合/停止/计分行不变；只替换基线内部28%的新XGBoost分支，再应用已冻结融合公式。',
                f"单模型AUC {ns['auc']:.9f}，融合AUC {ns['blend_auc']:.9f}，相对冻结组合 {ns['paired']['delta']:+.9f}，约95%区间 [{lo:+.9f}, {hi:+.9f}]。状态：{ndec['status']}。",
                '该配置没有达到+0.0002筛选门槛，不进入确认或完整训练。三项条件分组、平滑公式、未知组回退及不吸收查询数据的契约检查通过。',
                f'明细：`{nd}`。']
    if cvlatest.exists():
        cd=Path(read(cvlatest)['directory'])
        if (cd/'summary.json').exists():
            cs=read(cd/'summary.json')
            cvverified=read(cd/'verification.json');assert cvverified['verified']
            lines+=['','## 完整10折输出','',
                f"新增RealMLP单模型OOF AUC：{cs['standalone_oof_auc']:.9f}；冻结融合OOF AUC：{cs['blended_oof_auc']:.9f}。", 
                f"v4历史OOF AUC：{cs['legacy_baseline_oof_auc']:.9f}；相对它改善{cs['blended_oof_auc']-cs['legacy_baseline_oof_auc']:+.9f}。",
                f"使用相同新RealMLP折外预测的20%概率融合对照：{cvverified['fixed_cv_probability20_auc']:.9f}；冻结35% logit相对它改善{cs['blended_oof_auc']-cvverified['fixed_cv_probability20_auc']:+.9f}。这项对照不同于已提交的全数据单次拟合版本。",
                '两组固定种子均完成10折，外层每折的训练区内再做5折目标编码，固定3轮训练。完整OOF分数没有用于重新选择配方或权重。',
                f"输出：`{cs['output']}`，SHA256：`{cs['output_sha256']}`。", 
                '新增分支外层计分标签不参与训练/早停，TE在外层训练区内部交叉拟合。现有v4的历史OOF含外层早停，因此融合OOF仍有该限制。',
                '独立核验从20份种子级缓存重建各折、完整OOF、测试平均与最终CSV；所有训练行恰好获得一次外层预测，299844个测试ID与顺序、概率范围、融合公式及受保护文件哈希均通过。',
                '未使用完整OOF再选择权重。新文件公榜未知，不能据此宣布超过0.96104。']
    (ROOT/'FOLLOWUP_20261019_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(ROOT/'FOLLOWUP_20261019_REPORT.md')

if __name__=='__main__':main()
