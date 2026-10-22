"""Report measured validation and separately attributed user leaderboard feedback."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def report():
    r=ROOT
    def read(p):return json.loads((r/p).read_text(encoding='utf-8'))
    rec=read('v3_recipe.json');m=read('submission_v3_metrics.json');v=read('v3_verification.json');a=read('v3_confirmation.json')
    leaderboard_status='尚未收到线上成绩反馈。'
    leaderboard_cell='尚无反馈'
    leaderboard_analysis=''
    if (r/'leaderboard_feedback.json').exists():
        feedback=read('leaderboard_feedback.json')
        current=feedback['current']
        actual_hash=hashlib.sha256((r/'submission_v3.csv').read_bytes()).hexdigest()
        if current['submission_sha256'] != actual_hash:
            raise ValueError('Leaderboard feedback belongs to a different submission file.')
        score=float(current['score']);old_score=float(feedback['previous']['score'])
        m.update(kaggle_score=score,submitted=True,kaggle_score_source='user_report',
                 kaggle_score_independently_verified=False,
                 leaderboard_feedback_file='leaderboard_feedback.json')
        (r/'submission_v3_metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        leaderboard_status=f'用户反馈最新提交成绩为 {score:.5f}；成绩来自用户，未通过 Kaggle 接口独立核实。'
        leaderboard_cell=f'{score:.5f}（用户反馈）'
        leaderboard_analysis=(f'用户反馈的线上增益为 **{score-old_score:+.5f}**，距离 0.962 仍差 **{.962-score:.5f}**。'
            f'本地 OOF 与线上差值为 {m["oof_auc"]-score:.9f}，旧方案为 {m["baseline_oof_auc"]-old_score:.9f}。'
            '本轮增益体现在线上结果中，但两次提交不足以建立固定的本地到线上换算关系，也不能证明没有过拟合。'
            '反馈按最近交付的 submission_v3.csv 记录；用户未提供提交编号或文件名。')
    wb,wx,wn=rec['weights']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    curves={}
    for name in ['cv.log','cv_resume.log']:
        path=r/'.artifact_runtime/v3'/name
        if not path.exists():continue
        parts=re.split(r'CV START (\d+) realmlp',path.read_text(encoding='utf-8',errors='replace'))
        for i in range(1,len(parts),2):
            values=re.findall(r'Epoch (\d+)/128: val 1-auc_ovr = ([0-9.]+)',parts[i+1])
            curves[int(parts[i])]=[(int(e),1-float(z)) for e,z in values]
    gains=np.array(m['fold_scores'])-m['baseline_fold_scores']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(range(1,11),gains,color=['#17876f' if z>0 else '#b64a45' for z in gains])
    axes[0].axhline(0,color='black',linewidth=.7)
    axes[0].set(title='Same-fold AUC gain over previous recipe',xlabel='Validation fold',ylabel='AUC difference',xticks=range(1,11))
    for fold in [1,8,10]:
        values=curves.get(fold,[])
        if values:
            epochs,auc=zip(*values);line=axes[1].plot(epochs,auc,label=f'Fold {fold}')[0]
            best=int(np.argmax(auc));axes[1].scatter(epochs[best],auc[best],s=28,color=line.get_color())
    axes[1].set(title='RealMLP validation curves (best epoch marked)',xlabel='Epoch',ylabel='Validation ROC-AUC')
    axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(r/'final_validation.png',dpi=150);plt.close(fig)
    rows='\n'.join(f'| {i} | {old:.7f} | {new:.7f} | {new-old:+.7f} |'
        for i,(old,new) in enumerate(zip(m['baseline_fold_scores'],m['fold_scores']),1))
    nn=rec['candidates']['realmlp'];xgb=rec['candidates']['xgb']
    target='达到' if m['oof_auc']>=.962 else '未达到'
    text=f'''# 本轮优化实测报告

本轮目标是争取 Kaggle ROC-AUC 0.962。已完成最终方案的 10 折训练和预测核验。{leaderboard_status}

## 实际结果

| 指标 | 旧方案 | 新方案 |
|---|---:|---:|
| 全部折外预测 OOF AUC | {m['baseline_oof_auc']:.9f} | {m['oof_auc']:.9f} |
| 10 折平均 AUC | {sum(m['baseline_fold_scores'])/10:.9f} | {m['mean_auc']:.9f} |
| 固定方案后的复核 AUC | {a['baseline_auc']:.9f} | {a['auc']:.9f} |
| Kaggle 提交 AUC | 0.96036（用户反馈） | {leaderboard_cell} |

OOF 变化：{v['oof_gain']:+.9f}；相同 10 折中 {v['improved_folds']} 折提高。新方案折间标准差 {m['std_auc']:.9f}。

本地 OOF **{target} 0.962**。本地 AUC 和线上 AUC 是两个结果，不能把本地分数直接当作线上成绩，也不能假设两者差值固定。

{leaderboard_analysis}

## 最终方法

新版本采用 66 个输入特征：21 个原始特征、21 个飞行距离分组统计、4 个频次统计、6 个折内目标编码、1 个原始数据模型的概率对数比、13 个服务评分辅助预测。

- 飞行距离分组统计只在当前训练分区学习，表示相同距离旅客的年龄、评分、客群组成与样本数；没有使用满意度标签。验证及测试只查询训练分区统计。
- 6 个目标编码键为飞行距离、年龄、飞行距离×年龄、飞行距离×舱位、飞行距离×出行目的、年龄×舱位×出行目的×客户类型。训练特征使用内部 5 折编码，验证及测试只用训练标签建立的映射。
- 原始数据模型沿用先前公开航空满意度数据训练的 XGBoost 先验。原始数据与比赛特征完全重合的 21 行已删除，保留 129859 行。
- 对每个服务评分，另用公开原始数据的其他 20 个共同特征训练一个回归器，预测期望评分。预测自身评分时排除该评分列；这些辅助回归器不使用满意度标签。
- 主模型候选为 XGBoost 和官方 RealMLP。数值填补、类别映射和所有训练型统计均按当前训练分区拟合。

最终固定权重：旧融合方案 {wb:.0%}、新 XGBoost {wx:.0%}、RealMLP {wn:.0%}。旧方案内部为 25% XGBoost 和 75% CatBoost，故最终 XGBoost 总权重 {wb*.25+wx:.0%}、CatBoost 权重 {wb*.75:.0%}。

## 筛选与失败实验

所有候选使用此前相同的 447766 行拟合集、111942 行筛选集。旧融合方案筛选 AUC {rec['baseline_selection_auc']:.9f}，最终新融合 {rec['selection_auc']:.9f}。

| 候选 | 筛选 AUC |
|---|---:|
| 旧 XGBoost（目标编码、原始数据统计及先验） | 0.959762207 |
| 新 XGBoost（66 特征） | {xgb['auc']:.9f} |
| 新 RealMLP（同样的 66 特征） | {nn['auc']:.9f} |
| 参数调整后的最佳 CatBoost | 0.960301261 |
| TabM：完整统计输入 | 0.959065739 |
| TabM：简化输入、加宽网络 | 0.958069605 |

新 XGBoost 的输入和参数同时变化，因此不能把它与旧 XGBoost 的全部差值都归因于某一个特征。单独新增原始数据 CatBoost 先验未取得收益。7 个 CatBoost 参数候选的最佳融合筛选 AUC 为 0.960328966，但固定轮数复核较同次旧方案低约 0.000015；该分支仅保留了 2 折对照缓存，没有把它称为完整 10 折结果。

本轮没有继续添加随机插值合成样本：此前它降低了验证成绩。额外信息来自公开原始样本训练的先验和辅助特征。

## 验证边界与过拟合

最终权重先在有限固定网格选择，然后冻结。复核阶段不根据复核标签调整权重；XGBoost 按筛选最优轮数的 1.1 倍训练，RealMLP 只在开发数据内部留出 5% 选模型，不使用复核标签早停。

复核集上一轮已经看过，并非全新的独立测试集。本轮也在相同筛选集比较了多个方案；最终 10 折和筛选数据存在重叠，且折模型会用本折验证标签选择最佳轮次。这些因素可能使本地估计偏乐观。复核阶段的 RealMLP 内部 5% 划分发生在开发集目标编码之后，内部早停指标也可能偏乐观，但外层复核标签始终未进入编码或训练。不能仅凭训练 AUC 高于验证 AUC 就判定线上差距全部来自过拟合。

| 折 | 旧 AUC | 新 AUC | 变化 |
|---|---:|---:|---:|
{rows}

![逐折增益与训练曲线]({(r/'final_validation.png').as_posix()})

曲线圆点标记最佳验证轮次。后期分数下降的轮次不作为最终模型；这说明轮次选择有作用，但不能单独证明线上差距来自哪一种原因。

## 交付与复现

- `submission_v3.csv`：可提交预测，共 {v['rows']} 行，列为 `id,satisfaction`，顺序与测试集及模板一致。
- `submission_v3_oof.csv`、`submission_v3_metrics.json`：完整 10 折预测与结果。
- `v3_recipe.json`：冻结配置和网格记录；`v3_confirmation.json`、`v3_verification.json`：复核及独立重建凭据。
- `main_v3.py`：最终训练入口；双击 `run_v3.cmd` 可复用缓存或继续训练并核验。
- 原 `main.py`、输入数据及旧提交完整保留，{v['preserved_files']} 个受保护文件的哈希均已核验。

续跑时发现项目根目录的 `train.csv`、`test.csv` 缺失，已从下载目录的原件恢复；两份文件的 SHA-256 与训练开始时的记录完全一致，下载目录原件未修改。续跑复用了已完成的 7 折和第 8 折 XGBoost 检查点。

复现筛选：先运行 `v3_features.py` 生成公开数据辅助预测，再分别运行 `v3_training.py xgb --device cpu`、`v3_training.py realmlp`；随后依次运行 `optimize_v3.py select`、`optimize_v3.py confirmation`、`main_v3.py`。完整复现环境见 `requirements-v3.txt`。CPU 辅助特征计算与 GPU 模型训练分别记录在 `.artifact_runtime/v3/` 中。

源方法来自[本场比赛公开方案](https://www.kaggle.com/code/sachith7/s6e10-what-each-step-was-worth)的路线统计、目标编码和辅助评分思路；本项目独立生成所有模型预测，没有导入他人 OOF 或测试预测。其页面报告的 0.96153 属于作者的多模型方案，不是本项目复现成绩。RealMLP 使用[官方 PyTabKit 实现](https://github.com/dholzmueller/pytabkit)，原始数据来自[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)。最初 S6E8 参考方案对应另一个比赛，其分数不能直接迁移到本场。
'''
    (r/'FINAL_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8')
    print('Report created: FINAL_OPTIMIZATION_REPORT.md')


if __name__=='__main__':report()
