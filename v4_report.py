"""Refresh the v4 report with attributed feedback, without changing training fingerprints."""
import hashlib
from pathlib import Path

from optimize_v4 import ROOT, read, report, write_json


def refresh():
    recipe = read(ROOT/'v4_recipe.json')
    directory = ROOT/'.artifact_runtime/v4'/recipe['signature']
    confirmation = read(directory/'confirmation.json') if (directory/'confirmation.json').exists() else None
    metrics_path = ROOT/'submission_v4_metrics.json'
    metrics = read(metrics_path) if metrics_path.exists() else None
    if metrics and Path(metrics['directory']) != directory:
        raise ValueError('Metrics and report recipe do not match.')
    feedback_path = ROOT/'leaderboard_feedback_v4.json'
    feedback = read(feedback_path) if feedback_path.exists() else None
    if feedback:
        current = feedback['current']
        if current['submission_file'] != 'submission_v4.csv':
            raise ValueError('Unexpected feedback submission file.')
        actual = hashlib.sha256((ROOT/'submission_v4.csv').read_bytes()).hexdigest()
        if actual != current['submission_sha256']:
            raise ValueError('Feedback does not match the current submission hash.')
        score = float(current['score']); previous = float(feedback['previous']['score'])
        if not 0 <= score <= 1 or not 0 <= previous <= 1:
            raise ValueError('Invalid leaderboard AUC.')
        if metrics:
            metrics.update(kaggle_score=score, submitted=True, kaggle_score_source='user_report',
                           kaggle_score_independently_verified=False,
                           leaderboard_feedback_file=feedback_path.name)
            write_json(metrics_path, metrics)
    report(directory, recipe, confirmation, metrics)
    if feedback:
        path = ROOT/'V4_EXPERIMENT_REPORT.md'
        text = path.read_text(encoding='utf-8').replace('尚无本轮线上成绩。',
            f'用户反馈本轮线上成绩为 {score:.5f}；未通过 Kaggle 接口独立核实。')
        text += f'''
## 用户反馈的线上成绩

| 版本 | Kaggle AUC | 较上一版 |
|---|---:|---:|
| 初始优化版 | 0.96036 | — |
| v3 | {previous:.5f} | {previous-.96036:+.5f} |
| v4 | **{score:.5f}** | **{score-previous:+.5f}** |

相比最初 0.96036，累计提高 {score-.96036:.5f}；距离目标 0.962 仍差 {0.962-score:.5f}。
本地 OOF 与最新线上分数的差值为 {metrics['oof_auc']-score:.9f}。
本轮线上成绩支持新增融合有收益，但不能证明没有过拟合，也不能据此推断后续收益或调整最佳权重。
以上线上分数均来自用户反馈；最新分数按最近交付的 submission_v4.csv 关联，用户未提供文件名或提交编号。
feedback 文件独立于训练配置保存，训练代码、冻结方案及预测文件均未因记录成绩而改变。
'''
        path.write_text(text, encoding='utf-8')
    print('Updated V4_EXPERIMENT_REPORT.md; no training performed.')


if __name__ == '__main__':
    refresh()
