"""Build the experiment report from measured artifacts."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def report():
    r=ROOT/'.artifact_runtime/v2'
    cats=json.loads((r/'cat_screen.json').read_text(encoding='utf-8'))
    recipe=json.loads((ROOT/'v2_recipe.json').read_text(encoding='utf-8'))
    nn=recipe['nn']
    metrics=json.loads((ROOT/'submission_v2_metrics.json').read_text(encoding='utf-8'))
    audit=json.loads((ROOT/'v2_confirmation.json').read_text(encoding='utf-8'))
    verification=json.loads((ROOT/'v2_verification.json').read_text(encoding='utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for d in cats:
        curves=d['curves'];auc=curves['validation']['AUC'];start=min(200,len(auc)-1)
        axes[0].plot(range(start+1,len(auc)+1),auc[start:],label=d['config']['name'])
        if d['config']['name'] in ['cat_control',recipe['cat_config']['name']]:
            for part,style in [('learn','--'),('validation','-')]:
                values=curves[part]['Logloss'];indices=np.arange(len(values))*25+1
                axes[1].plot(indices[indices>200],np.array(values)[indices>200],style,label=d['config']['name']+' '+part)
    axes[0].set(title='CatBoost selection curves',xlabel='Trees',ylabel='Selection ROC-AUC')
    axes[1].set(title='Training and selection loss',xlabel='Trees',ylabel='Log loss')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'v2_learning_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for item in recipe['nn_candidates']:
        ax.plot([v['epoch'] for v in item['curves']],[v['auc'] for v in item['curves']],label=item['config']['name'])
    ax.set(title='TabM selection curves',xlabel='Epoch',ylabel='Selection ROC-AUC')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(ROOT/'v2_tabm_curves.png',dpi=150);plt.close(fig)
    fold_rows='\n'.join(f'| {i} | {old:.7f} | {new:.7f} | {new-old:+.7f} |'
        for i,(old,new) in enumerate(zip(metrics['baseline_fold_scores'],metrics['fold_scores']),1))
    candidate_rows='\n'.join(f"| {d['config']['name']} | {d['auc']:.7f} | {d['train_auc']:.7f} | {d['rounds']} |" for d in cats)
    nn_rows='\n'.join(f"| {d['config']['name']} | {d['auc']:.7f} | {d['train_auc_sample']:.7f} | {d['epochs']} |" for d in recipe['nn_candidates'])
    wb,wc,wn=recipe['weights']
    text=f'''# 第二轮优化实测报告

目标：争取 Kaggle ROC-AUC 0.962。旧提交的 0.96036 为用户反馈；本轮没有上传 Kaggle，不能把本地分数当作线上分数。

## 最终结果

- 新方案 10 折平均 AUC：{metrics['mean_auc']:.9f}，折间标准差：{metrics['std_auc']:.9f}。
- 新方案 OOF AUC：{metrics['oof_auc']:.9f}。
- 旧方案 OOF AUC：{metrics['baseline_oof_auc']:.9f}，变化：{verification['oof_gain']:+.9f}。
- 同样的 10 个验证折中，{metrics['improved_folds']} 折提高。
- 本地 OOF 是否达到 0.962：{metrics['oof_auc']>=.962}。Kaggle 是否达到：尚未验证。

## 做了哪些实验

固定使用此前的 447766 行拟合集和 111942 行筛选集，比较 7 个 CatBoost 配置。保留原始数据统计、XGBoost 原始数据概率先验和数值的类别副本。新增的原始数据 CatBoost 先验仅使用去除完全重合行后的公开原始数据标签训练，不使用比赛验证标签。

| CatBoost 配置 | 筛选 AUC | 训练 AUC | 最优轮数 |
|---|---:|---:|---:|
{candidate_rows}

测试了带原始数据统计的 TabM，以及精简为原始特征和先验概率、加宽网络的 TabM。较好候选为 {nn['config']['name']}，筛选 AUC：{nn['auc']:.7f}，最优 {nn['epochs']} 轮。使用分段线性数值嵌入、类别输入、{nn['config']['k']} 个共享权重的成员；训练时分别计算成员损失，预测时平均概率。数值填补、分位数变换和类别编码仅在每次拟合分区学习。训练集抽样 AUC：{nn['train_auc_sample']:.7f}，与 CatBoost 的全训练集 AUC 口径不同。

| TabM 配置 | 筛选 AUC | 训练抽样 AUC | 最优轮数 |
|---|---:|---:|---:|
{nn_rows}

最终固定权重：旧方案 {wb:.0%}，新 CatBoost {wc:.0%}，TabM {wn:.0%}。旧方案内部为 25% XGBoost + 75% CatBoost；最终 XGBoost 权重为 {wb*.25:.0%}。融合只筛选有限的固定权重网格，不根据 Kaggle 分数调整。

## 验证边界与过拟合

此前的审计集已经看过，因此它不是全新的独立测试集。本轮先冻结模型及融合权重，再按固定训练轮数作一次复核；复核标签不参与训练、早停或重新选择权重。该复核新 AUC 为 {audit['auc']:.9f}，同次旧模型对照为 {audit['baseline_auc']:.9f}，差值 {audit['change']:+.9f}。

最终使用分层 10 折和旧方案相同的折划分，未给任一行生成包含自身训练标签的折外预测。每折早停会查看该折验证标签，且方案筛选与最终 CV 的数据存在重叠，因此 OOF 是比较证据，不能当作完全独立的泛化估计。训练与验证 AUC 有差距不等于已证明线上差距源于过拟合。

没有继续使用随机插值的人工合成样本，因为上一轮实测变差。使用的额外标签数据来自公开原始航空满意度数据；删除了与比赛数据特征完全相同的 21 行，保留 129859 行。

| 折 | 旧 AUC | 新 AUC | 变化 |
|---|---:|---:|---:|
{fold_rows}

## 文件和复现

- `submission_v2.csv`：本轮提交文件，{verification['test_rows']} 行，`id,satisfaction`，正类概率。
- `submission_v2_oof.csv`：每行恰好一次的 10 折验证预测。
- `submission_v2_metrics.json`、`v2_confirmation.json`、`v2_verification.json`：实测结果及独立重建核验。
- `v2_recipe.json`：冻结的模型配置、权重、实验记录和数据指纹。
- `v2_learning_curves.png`、`v2_tabm_curves.png`：学习曲线。
- `.artifact_runtime/v2/snapshot/`：旧版本快照；核验确认数据、旧代码和旧提交文件未变。

双击 `run_v2.cmd` 或用项目环境运行 `main_v2.py`，可复现冻结配方的 10 折训练与核验。

在项目目录使用 `.venv\\Scripts\\python.exe v2_training.py` 重现 CatBoost 筛选；使用 `.venv\\Scripts\\python.exe v2_tabm.py` 和追加 `--compact` 的同一命令重现两组 TabM 筛选。随后依次运行 `optimize_v2.py select`、`optimize_v2.py confirmation`、`optimize_v2.py cv`，最后运行 `verify_v2.py` 和 `v2_report.py`。同环境可安装 `requirements-v2.txt`；神经网络和本轮 CatBoost 实验需要 CUDA GPU。完整 CV 自动复用与指纹一致的已完成预测。旧方案及其 10 折缓存保留，融合时复用。

TabM 参考：[官方实现与训练说明](https://github.com/yandex-research/tabm)。原始数据：[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)。
'''
    (ROOT/'V2_REPORT.md').write_text(text,encoding='utf-8')
    print('Report saved:',ROOT/'V2_REPORT.md')


if __name__=='__main__':report()
