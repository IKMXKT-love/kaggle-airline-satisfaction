"""Write a factual report from saved evaluation and verification receipts."""
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'.artifact_runtime/public_pool_trial_20261020'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main():
    e = load(RUN/'evaluation.json')
    m = load(RUN/'matrix_manifest.json')
    v = load(RUN/'independent_verification.json')
    sub = load(ROOT/'submission_public_pool_20261020.json')
    initial = load(RUN/'protocol.json')
    confirm = load(RUN/'confirmation_protocol.json')
    cv = load(RUN/'confirmation_result.json')
    transition = load(RUN/'solver_transition.json')
    receipts = load(RUN/'download_receipts.json')
    assert v['passed'] and e['passes_material_gain_gate'] and not e['own_included']
    source = pd.read_csv(ROOT/'public_pool_sources_20261020.csv', float_precision='round_trip')
    notebooks = {'yekenot', 'hermengardo', 'kratosyan', 'kagan_enc', 'aman_v5'}
    source['source_url'] = ['https://www.kaggle.com/'+('code/' if x['source'] in notebooks else 'datasets/')+x['source_ref']
                            for x in m['columns'][:m['public_columns']]]
    source.to_csv(ROOT/'public_pool_sources_20261020.csv', index=False, lineterminator='\n')
    cost = {
        'initial_phase_approx_seconds': confirm['started_epoch']-initial['started_epoch'],
        'initial_target_seconds': 900,
        'initial_overrun_including_inflight_and_transition_seconds': confirm['started_epoch']-initial['screen_deadline_epoch'],
        'confirmation_elapsed_seconds': time.time()-confirm['started_epoch'],
        'converged_cv_fit_seconds': sum(x['fit']['seconds'] for x in cv['rows']),
        'interrupted_lbfgs_process_wall_seconds': transition['old_process_wall_seconds'],
        'full_refit_seconds': sub['full_refit']['seconds'],
        'base_model_fits': 0, 'base_model_inferences': 0, 'converged_meta_model_fits': 11,
        'download_files': len(receipts), 'saved_download_bytes': sum(x['bytes'] for x in receipts),
        'timing_note': 'Initial start retrospectively includes 100 seconds of setup allowance; initial wall time is approximate. Saved bytes are not cumulative network traffic.'}
    save(RUN/'cost_receipt.json', cost)
    save(ROOT/'CURRENT_RECOMMENDED_CANDIDATE.json', {
        'as_of_local_date': '2026-10-20', 'status': 'offline_verified_public_score_pending',
        'submission_file': 'submission_public_pool_20261020.csv', 'sha256': sub['submission_sha256'],
        'oof_auc': e['pooled_aucs']['public_only'], 'public_auc': None,
        'public_prediction_members': 246, 'own_branch_members_used': 0,
        'reason_own_members_omitted': 'No stable incremental five-fold improvement',
        'source_attribution': 'public_pool_sources_20261020.csv', 'report': 'PUBLIC_POOL_TRIAL_REPORT_20261020.md',
        'current_submitted_best_record_unchanged': True})
    scores = e['pooled_aucs']
    delta = scores['public_only']-scores['current_best']
    rows = '\n'.join(f"| {i+1} | {e['fold_scores']['current_best'][i]:.9f} | {e['fold_scores']['public_only'][i]:.9f} | {e['selected_deltas_vs_current'][i]:+.9f} | {e['own_paired_deltas'][i]:+.9f} |" for i in range(5))
    fit_total = cost['converged_cv_fit_seconds']+cost['interrupted_lbfgs_process_wall_seconds']+cost['full_refit_seconds']
    report = f'''# 公开基础预测池实验结果

日期：2026-10-20。已完成来源导入、两组固定配方的五折融合比较、全量二层重拟合、候选导出和独立核验。基础模型没有重新训练或推理，没有上传Kaggle。

## 推荐提交

`submission_public_pool_20261020.csv`，299844行，列名和ID与官方样例及本地测试集一致，概率有效。

SHA-256：`{sub['submission_sha256']}`。

方法是246个公开基础预测的logit向量加L2逻辑回归，固定C=0.003。来源涉及22位作者、16个有保留成员的来源，作者、来源版本、许可元数据、系数及链接见`public_pool_sources_20261020.csv`。主要增益来自这些作者训练的基础模型；我们完成导入、融合、比较与核验。

**离线0.962205不代表线上达到0.962。当前没有这份文件的公榜成绩。** 用户反馈公榜0.96114的旧最佳及CURRENT_BEST记录保持不变，新候选单独记录在CURRENT_RECOMMENDED_CANDIDATE.json。

## 五折比较

| 方案 | 合并OOF AUC | 相对当前 |
|---|---:|---:|
| 当前最佳历史OOF | {scores['current_best']:.9f} | — |
| 公开246成员池 | {scores['public_only']:.9f} | {delta:+.9f} |
| 公开池加本地七分支 | {scores['public_plus_own']:.9f} | {scores['public_plus_own']-scores['current_best']:+.9f} |

| 折 | 当前 | 公开池 | 公开池相对当前 | 再加本地分支的变化 |
|---|---:|---:|---:|---:|
{rows}

公开池五折全部提升，通过预设的平均折增益至少0.0003、至少4/5折正向的门槛。本地七分支带来的平均折变化为{e['own_delta_mean']:+.9f}，标准误{e['own_delta_se']:.9f}，仅3/5折正向，未通过评分前冻结的纳入规则。因此最终没有强行加入我们的模型。

本地七分支是旧XGBoost、旧CatBoost、较新XGBoost、原RealMLP、LightGBM、精简RealMLP两种子。缓存重建当前组合最大概率误差约2.05e-8，与基础logit的float32舍入一致。

## 公开参照与来源变化

参考：[Dariush Afshar的247 Public OOFs](https://www.kaggle.com/code/dariushafshar/s6e10-247-public-oofs-one-stack-lb-0-96187)。按冻结v1成员白名单导入，没有将后来加入的stack245等二层产物混入。

`kz_source_transfer`的原始OOF/test在对应公开输出中无法取得，明确排除，没有用最终submission或其他模型替换。其余246个成员逐个核对长度、有限值、尺度和OOF AUC，与冻结清单AUC差均在2e-6内。原始文件哈希和抓取的来源版本已保存；当前URL将来可能变化。

原247参照OOF为0.962205015，本次246成员为{scores['public_only']:.9f}，只低约0.000000259。测试集秩相关{v['test_rank_correlation_with_author_frozen247_reference']:.9f}，最大概率差约{v['max_abs_probability_difference_from_author_reference']:.6f}。相似预测不能保证相同公榜成绩。

247参照的0.96187为作者自报。此前已核对其历史得分文件哈希，没有独立查询比赛提交记录。本轮候选重新拟合246个基础预测，没有把作者成品改名当作新成果。

## 验证边界

二层采用原生StratifiedKFold五折seed42；本地基础模型保留原有10折缓存，公开成员的折数与种子也有不同。这是固定基础OOF上的二层交叉验证，不是端到端严格嵌套验证。基础模型的评分折早停、跨折间接标签依赖、历史选择及公开成员选择偏差仍存在。

18个公开成员有ID重排或独立ID侧文件证据；223个依赖作者声明的行顺序，另5个还核对了完整标签向量。高AUC不能证明所有上游训练无泄漏。强正则和五折一致改善有参考价值，不能保证私榜泛化。

没有搜索新权重、C值、客群门控或其他二层模型。只比较公开池和公开池加本地分支两组；选择规则在评分前保存。下一步应提交候选验证真实成绩，不继续进行小权重微调。

## 成本与预算执行

初筛目标15分钟，该阶段主要完成下载准备，没有候选融合分数。计入在途请求和阶段收尾，切换到确认约{cost['initial_phase_approx_seconds']/60:.2f}分钟，超出目标约{cost['initial_overrun_including_inflight_and_transition_seconds']/60:.2f}分钟，不能声称严格15分钟内完成。下载器已改为逐MiB检查截止时间，限制在途请求拖延。初筛起点包含对先前准备时间回溯100秒的保守估算。

初筛没有完成新的模型效果门槛检验；随后按原授权60分钟确认阶段推进，依据是上一轮已经复算的公开组合+0.000530证据和导入一致性检查，不是一个并不存在的新初筛成绩。确认阶段至本报告约{cost['confirmation_elapsed_seconds']/60:.2f}分钟，包括补齐下载、完整矩阵、比较、重拟合和核验，低于60分钟上限。

共保存{cost['download_files']}个下载文件，约{cost['saved_download_bytes']/1024**3:.2f}GiB，不等同于累计网络流量。采用磁盘映射、分块计算和逐折拟合，没有记录全程RAM峰值。

最初分块L-BFGS运行约{cost['interrupted_lbfgs_process_wall_seconds']:.2f}秒后中止并保存同折检查点，原因是收敛速度可能导致整体超时。替换为相同目标的分块阻尼Newton法，先与标准逻辑回归数值核对，系数最大差约1.36e-8；输入、C和划分不变。

十次完成的五折二层拟合计时合计{cost['converged_cv_fit_seconds']:.2f}秒，全量二层重拟合{cost['full_refit_seconds']:.2f}秒；连同中止的求解器，拟合相关计时合计约{fit_total/60:.2f}分钟，不是全部任务墙钟。基础模型拟合0次、重新推理0次。

## 核验与复现材料

独立核验通过：CSV列名/ID/行数、概率有效性、保存预测与导出一致、线性模型抽样重建、全量训练目标梯度、OOF复算、来源系数和旧最佳哈希保护。全量模型最大梯度{v['saved_full_model_max_gradient']:.3g}。

- 候选：`submission_public_pool_20261020.csv`及同名JSON。
- 来源：`public_pool_sources_20261020.csv`。
- 导入、二层拟合、核验：`public_pool_trial.py`、`fit_public_pool.py`、`verify_public_pool_trial.py`。
- 协议、原始文件、版本、哈希、矩阵、权重、折预测及计时：`.artifact_runtime/public_pool_trial_20261020/`。
- 旧脚本、缓存和提交全部保留，旧最佳SHA-256仍为`{v['previous_best_sha256']}`。
'''
    path = ROOT/'PUBLIC_POOL_TRIAL_REPORT_20261020.md'
    if path.exists(): raise FileExistsError('Preserve the existing report')
    path.write_text(report, encoding='utf-8')
    print(json.dumps({'report': str(path), 'oof_gain': delta, 'fit_seconds_including_interrupted': fit_total,
                      'source_rows': len(source), 'candidate_verified': v['passed']}, indent=2))


if __name__ == '__main__':
    main()
