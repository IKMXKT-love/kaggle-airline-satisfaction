# 方法与源码导航

20条是按机制归并的路线，不是20种原创算法。每条的具体变体及分数见[完整登记](../EXPERIMENT_SCORE_CATALOG_20261023.md)。历史失败和中止均保留。

| 序号 | 路线 | 入口/实现 | 结果证据 |
|---:|---|---|---|
| 1 | XGBoost基线 | [main.py](../main.py) | [submission_metrics.json](../submission_metrics.json) |
| 2 | 人工特征与插值合成 | [feature_engineering.py](../feature_engineering.py) | [enhancement_comparison.json](../enhancement_comparison.json) |
| 3 | 原始数据先验与交叉目标编码 | [optimized_features.py](../optimized_features.py)、[optimized_training.py](../optimized_training.py) | [OPTIMIZATION.md](../OPTIMIZATION.md) |
| 4 | XGB/CatBoost/LightGBM组合 | [optimize.py](../optimize.py)、[optimize_v4.py](../optimize_v4.py) | [V4_EXPERIMENT_REPORT.md](../V4_EXPERIMENT_REPORT.md) |
| 5 | 航程画像与辅助评分预测 | [v3_features.py](../v3_features.py)、[v3_training.py](../v3_training.py) | [FINAL_OPTIMIZATION_REPORT.md](../FINAL_OPTIMIZATION_REPORT.md) |
| 6 | RealMLP与精简配方 | [experiment_neural.py](../experiment_neural.py)、[experiment_realmlp_factorial.py](../experiment_realmlp_factorial.py)、[train_lean_cv_followup.py](../train_lean_cv_followup.py) | [REALMLP_FACTORIAL_REPORT.md](../REALMLP_FACTORIAL_REPORT.md) |
| 7 | TabM | [v2_tabm.py](../v2_tabm.py)、[optimize_v2.py](../optimize_v2.py) | [FINAL_OPTIMIZATION_REPORT.md](../FINAL_OPTIMIZATION_REPORT.md) |
| 8 | 原始域频率与条件评分统计 | [transfer_features.py](../transfer_features.py)、[experiment_transfer.py](../experiment_transfer.py) | [TRANSFER_EXPERIMENT_REPORT.md](../TRANSFER_EXPERIMENT_REPORT.md) |
| 9 | 教师logit起点修正 | [experiment_transfer.py](../experiment_transfer.py) | [TRANSFER_EXPERIMENT_REPORT.md](../TRANSFER_EXPERIMENT_REPORT.md) |
| 10 | 条件原始数据近邻 | [experiment_source_neighbors.py](../experiment_source_neighbors.py) | [FOLLOWUP_20261019_REPORT.md](../FOLLOWUP_20261019_REPORT.md) |
| 11 | TabICL冻结/微调/局部上下文 | [experiment_tabicl_probe.py](../experiment_tabicl_probe.py)、[experiment_tabicl_finetune.py](../experiment_tabicl_finetune.py)、[experiment_local_context.py](../experiment_local_context.py) | [TABICL_FINETUNE_REPORT.md](../TABICL_FINETUNE_REPORT.md)、[LOCAL_CONTEXT_REPORT_20261021.md](../LOCAL_CONTEXT_REPORT_20261021.md) |
| 12 | 遮蔽分类与评分恢复多任务 | [experiment_multitask.py](../experiment_multitask.py)、[multitask_support.py](../multitask_support.py) | [DENOISING_MULTITASK_REPORT.md](../DENOISING_MULTITASK_REPORT.md) |
| 13 | FFM字段交互与加性对照 | [experiment_ffm.py](../experiment_ffm.py)、[ffm_support.py](../ffm_support.py) | [FIELD_INTERACTION_REPORT.md](../FIELD_INTERACTION_REPORT.md) |
| 14 | 强组合堆叠、offset与成员分歧纠错 | [experiment_correction.py](../experiment_correction.py)、[experiment_pool_repairability.py](../experiment_pool_repairability.py) | [STRONG_CORRECTION_REPORT.md](../STRONG_CORRECTION_REPORT.md)、[POOL_REPAIRABILITY_REPORT_20261022.md](../POOL_REPAIRABILITY_REPORT_20261022.md) |
| 15 | 公开预测池融合及上游增量 | [public_pool_trial.py](../public_pool_trial.py)、[fit_public_pool.py](../fit_public_pool.py)、[experiment_joint_upstream.py](../experiment_joint_upstream.py) | [PUBLIC_POOL_TRIAL_REPORT_20261020.md](../PUBLIC_POOL_TRIAL_REPORT_20261020.md)、[JOINT_UPSTREAM_TRIAL_REPORT_20261021.md](../JOINT_UPSTREAM_TRIAL_REPORT_20261021.md) |
| 16 | 双域共享表示 | [experiment_domain_transfer.py](../experiment_domain_transfer.py)、[domain_transfer_support.py](../domain_transfer_support.py) | [DOMAIN_TRANSFER_REPORT_20261021.md](../DOMAIN_TRANSFER_REPORT_20261021.md) |
| 17 | 类别条件域密度比 | [experiment_density_ratio.py](../experiment_density_ratio.py)、[density_ratio_support.py](../density_ratio_support.py) | [DENSITY_RATIO_REPORT_20261021.md](../DENSITY_RATIO_REPORT_20261021.md) |
| 18 | 额外评分与条件积分 | [experiment_privileged_signal.py](../experiment_privileged_signal.py)、[experiment_privileged_marginal.py](../experiment_privileged_marginal.py) | [PRIVILEGED_SIGNAL_REPORT_20261021.md](../PRIVILEGED_SIGNAL_REPORT_20261021.md) |
| 19 | 排序损失融合 | [experiment_ranking_fusion.py](../experiment_ranking_fusion.py) | [RANKING_FUSION_REPORT_20261022.md](../RANKING_FUSION_REPORT_20261022.md) |
| 20 | ModernBERT与RTD预训练 | [experiment_rtd_trial.py](../experiment_rtd_trial.py)、[rtd_trial_support.py](../rtd_trial_support.py)、[evaluate_rtd_trial.py](../evaluate_rtd_trial.py) | [RTD_TRIAL_REPORT_20261023.md](../RTD_TRIAL_REPORT_20261023.md) |
