# 已尝试方法与得分完整登记（2026-10-23）

本清单只读取已有文件，没有启动训练、推理、调参或提交。之前的“20类”是人为归并的方法方向，不是独立算法数、实验次数或已提交版本数。下面展开记录具体变体。

**排序规则：公榜成绩单列降序；完整OOF单列降序；其余每张原始对照表在自身验证范围内降序。不同分区、单模型/融合、内部选型/外部计分和源域成绩不合并成一个真实优劣排行榜。** 没有线上记录的方案不补造公榜分数。

## 一、用户反馈的公榜成绩（完整六个版本）

成绩均来自用户反馈，未独立查询Kaggle提交记录；文件关联沿用项目当时的归档。

| 序号 | 方案 | 公榜AUC | 预测文件 |
|---|---|---|---|
| 1 | 公开246基础预测＋L2逻辑回归融合 | 0.96187 | submission_public_pool_20261020.csv（归档路径：`submission_public_pool_20261020.csv`） |
| 2 | v4＋双种子10折精简RealMLP，65/35 logit融合 | 0.96114 | submission_lean_realmlp_10fold.csv（归档路径：`submission_lean_realmlp_10fold.csv`） |
| 3 | v4＋全量训练精简RealMLP，80/20概率融合 | 0.96104 | submission_v4_lean_realmlp_trial.csv（归档路径：`submission_v4_lean_realmlp_trial.csv`） |
| 4 | v4：v3＋LightGBM | 0.96082 | submission_v4.csv（归档路径：`submission_v4.csv`） |
| 5 | v3：航程画像、目标编码、辅助评分预测＋XGB/RealMLP融合 | 0.96074 | submission_v3.csv（归档路径：`submission_v3.csv`） |
| 6 | 初始优化：原始数据先验/目标编码＋XGB/CatBoost融合 | 0.96036 | submission_optimized.csv（归档路径：`submission_optimized.csv`） |

## 二、完整训练集OOF结果（按数值降序）

五折公开预测融合与十折本地训练的上游及验证协议不同；下表的数值顺序不能替代同条件比较。OOF不等于公榜。

| 序号 | 方案 | 整体OOF AUC | 验证范围 | 证据 |
|---|---|---|---|---|
| 1 | 公开池＋Kumo | 0.962205401810 | 原生5折二层融合 | [.artifact_runtime/joint_upstream_20261021/evaluation.json](.artifact_runtime/joint_upstream_20261021/evaluation.json) |
| 2 | 公开246预测池 | 0.962204756097 | 原生5折二层融合 | [.artifact_runtime/public_pool_trial_20261020/evaluation.json](.artifact_runtime/public_pool_trial_20261020/evaluation.json) |
| 3 | 公开池＋本地七分支 | 0.962204080677 | 原生5折二层融合 | [.artifact_runtime/public_pool_trial_20261020/evaluation.json](.artifact_runtime/public_pool_trial_20261020/evaluation.json) |
| 4 | 公开池＋Kumo＋神经邻居 | 0.962203170789 | 原生5折二层融合 | [.artifact_runtime/joint_upstream_20261021/evaluation.json](.artifact_runtime/joint_upstream_20261021/evaluation.json) |
| 5 | 公开池＋神经邻居 | 0.962202584473 | 原生5折二层融合 | [.artifact_runtime/joint_upstream_20261021/evaluation.json](.artifact_runtime/joint_upstream_20261021/evaluation.json) |
| 6 | v4＋双种子精简RealMLP融合 | 0.961674583332 | 本地10折 | [submission_lean_realmlp_10fold.json](submission_lean_realmlp_10fold.json) |
| 7 | v4 | 0.961446183127 | 本地10折 | [submission_v4_metrics.json](submission_v4_metrics.json) |
| 8 | v3 | 0.961422659137 | 本地10折 | [submission_v3_metrics.json](submission_v3_metrics.json) |
| 9 | 双种子精简RealMLP单模型 | 0.961318956314 | 本地10折 | [submission_lean_realmlp_10fold.json](submission_lean_realmlp_10fold.json) |
| 10 | 初始XGB/CatBoost优化 | 0.961111828459 | 本地10折 | [submission_optimized_metrics.json](submission_optimized_metrics.json) |
| 11 | 原始XGBoost | 0.958695243856 | 本地10折 | [submission_metrics.json](submission_metrics.json) |
| 12 | 人工特征＋插值合成 | 0.958405629161 | 本地10折 | [submission_enhanced_metrics.json](submission_enhanced_metrics.json) |

## 三、逐项实验、对照、消融与融合变体

每组保留原表的全部字段与行；每表按标注的AUC列降序。逐折成绩、早停过程、分群分析只是对应实验的明细，不另算独立方法。只有差值的表按差值排序，明确不当成绝对AUC。

### 初始XGBoost、合成数据及原始数据先验

来源：[optimization_report.md](optimization_report.md)

**航空满意度优化结果**；排序列：`10 折平均 AUC`。

| 方案 | 10 折平均 AUC |
|---|---|
| 本次优化 | 0.961119 |
| 原始 XGBoost | 0.958703 |
| 上一轮特征工程与插值合成 | 0.958412 |

### v3、TabM与CatBoost参数对照

来源：[FINAL_OPTIMIZATION_REPORT.md](FINAL_OPTIMIZATION_REPORT.md)

**筛选与失败实验**；排序列：`筛选 AUC`。

| 候选 | 筛选 AUC |
|---|---|
| 新 XGBoost（66 特征） | 0.960318078 |
| 参数调整后的最佳 CatBoost | 0.960301261 |
| 新 RealMLP（同样的 66 特征） | 0.959918344 |
| 旧 XGBoost（目标编码、原始数据统计及先验） | 0.959762207 |
| TabM：完整统计输入 | 0.959065739 |
| TabM：简化输入、加宽网络 | 0.958069605 |

**验证边界与过拟合**；排序列：`旧 AUC`。

| 折 | 旧 AUC | 新 AUC | 变化 |
|---|---|---|---|
| 6 | 0.9620782 | 0.9623926 | +0.0003145 |
| 5 | 0.9617789 | 0.9620079 | +0.0002290 |
| 8 | 0.9616536 | 0.9620880 | +0.0004344 |
| 9 | 0.9612596 | 0.9614273 | +0.0001677 |
| 2 | 0.9612337 | 0.9614577 | +0.0002240 |
| 1 | 0.9610940 | 0.9615173 | +0.0004233 |
| 10 | 0.9610205 | 0.9613905 | +0.0003700 |
| 7 | 0.9606580 | 0.9609776 | +0.0003196 |
| 4 | 0.9605700 | 0.9609922 | +0.0004222 |
| 3 | 0.9598477 | 0.9600642 | +0.0002165 |

### v4特征消融、残差特征与LightGBM融合

来源：[V4_EXPERIMENT_REPORT.md](V4_EXPERIMENT_REPORT.md)

**特征消融**；排序列：`筛选 AUC`。

| 去掉的特征组 | 筛选 AUC | 相对完整对照变化 |
|---|---|---|
| frequency | 0.960220643 | -0.000077730 |
| auxiliary | 0.960104219 | -0.000194154 |
| target_encoding | 0.960062797 | -0.000235576 |
| route | 0.960026350 | -0.000272023 |
| teacher | 0.959848486 | -0.000449887 |

**固定候选与融合**；排序列：`筛选 AUC`。

| 候选 | 融合方式 | 权重 | 筛选 AUC | 相对 v3 |
|---|---|---|---|---|
| lgb_63 | mix | 0.3 | 0.960596145 | +0.000047233 |
| lgb_63 | mix | 0.4 | 0.960595639 | +0.000046727 |
| lgb_63 | mix | 0.2 | 0.960588783 | +0.000039871 |
| lgb_127 | mix | 0.2 | 0.960573254 | +0.000024342 |
| lgb_63 | mix | 0.1 | 0.960573154 | +0.000024242 |
| lgb_127 | mix | 0.3 | 0.960573016 | +0.000024104 |
| xgb_residual | replace_xgb | 0.4 | 0.960570264 | +0.000021352 |
| lgb_127 | mix | 0.1 | 0.960566090 | +0.000017178 |
| lgb_127 | mix | 0.4 | 0.960563984 | +0.000015072 |
| without_teacher | replace_xgb | 0.4 | 0.960556406 | +0.000007494 |
| lgb_15 | mix | 0.1 | 0.960546252 | -0.000002659 |
| lgb_15 | mix | 0.2 | 0.960537091 | -0.000011821 |
| lgb_15 | mix | 0.3 | 0.960520416 | -0.000028496 |
| without_frequency | replace_xgb | 0.4 | 0.960503557 | -0.000045355 |
| lgb_15 | mix | 0.4 | 0.960497983 | -0.000050929 |
| without_target_encoding | replace_xgb | 0.4 | 0.960475610 | -0.000073302 |
| without_auxiliary | replace_xgb | 0.4 | 0.960460302 | -0.000088610 |
| without_route | replace_xgb | 0.4 | 0.960406284 | -0.000142627 |

**用户反馈的线上成绩**；排序列：`Kaggle AUC`。

| 版本 | Kaggle AUC | 较上一版 |
|---|---|---|
| v4 | **0.96082** | **+0.00008** |
| v3 | 0.96074 | +0.00038 |
| 初始优化版 | 0.96036 | — |

### RealMLP航程类别及数值表示

来源：[NEURAL_EXPERIMENT_REPORT.md](NEURAL_EXPERIMENT_REPORT.md)

**单模型结果**；排序列：`筛选 AUC`。

| 设置 | 特征数 | 筛选 AUC | 训练及预测分钟 |
|---|---|---|---|
| 同预算 RealMLP 对照 | 66 | 0.959470271 | 3.0 |
| 数值嵌入尺度 32 | 66 | 0.959456436 | 4.0 |
| 新增航程类别嵌入 | 67 | 0.958647484 | 4.4 |

**融合诊断**；排序列：`融合 AUC`。

| 设置 | 网络权重 | 融合 AUC | 相对历史 v4 | 相对同权重对照 |
|---|---|---|---|---|
| 新增航程类别嵌入 | 14% | 0.960607667 | +0.000011522 | +0.000025703 |
| 数值嵌入尺度 32 | 14% | 0.960597021 | +0.000000876 | +0.000015057 |
| 数值嵌入尺度 32 | 30% | 0.960540889 | -0.000055257 | +0.000021333 |
| 新增航程类别嵌入 | 30% | 0.960540870 | -0.000055275 | +0.000021314 |
| 数值嵌入尺度 32 | 50% | 0.960363626 | -0.000232519 | +0.000016025 |
| 新增航程类别嵌入 | 50% | 0.960294000 | -0.000302145 | -0.000053601 |

**成对不确定性参考**；排序列：`单模型 AUC 差`。

| 设置 | 单模型 AUC 差 | 约 95% 区间 |
|---|---|---|
| 数值嵌入尺度 32 | -0.000013835 | [-0.000433437, +0.000405767] |
| 新增航程类别嵌入 | -0.000822787 | [-0.001323735, -0.000321839] |

### 源域统计、教师offset与4096分箱

来源：[TRANSFER_EXPERIMENT_REPORT.md](TRANSFER_EXPERIMENT_REPORT.md)

**筛选结果**；排序列：`替换后融合 AUC`。

| 设置 | 特征数 | 最佳轮数 | 单模型 AUC | 相对 XGB 对照 | 替换后融合 AUC | 相对融合对照 |
|---|---|---|---|---|---|---|
| 教师 logit 修正 | 66 | 1107 | 0.959875598 | -0.000201990 | 0.960437517 | +0.000033457 |
| 4096 分箱 | 66 | 621 | 0.960144959 | +0.000067371 | 0.960431988 | +0.000027928 |
| 评分条件分布 | 96 | 685 | 0.960092000 | +0.000014413 | 0.960424703 | +0.000020644 |
| 原始/合成频率比 | 90 | 856 | 0.960044111 | -0.000033477 | 0.960410107 | +0.000006047 |
| 频率比＋评分分布 | 120 | 603 | 0.960051356 | -0.000026232 | 0.960408713 | +0.000004653 |

### RealMLP特征/训练配方四组对照

来源：[REALMLP_FACTORIAL_REPORT.md](REALMLP_FACTORIAL_REPORT.md)

**RealMLP 四组对照实测**；排序列：`固定融合 AUC`。

| 设置 | 单模型 AUC | 固定融合 AUC | 融合增益 | 样本层面约95%成对区间 |
|---|---|---|---|---|
| 精简47特征＋新配方 | 0.960094921 | 0.960578835 | +0.000174775 | [+0.000103215, +0.000246335] |
| 精简47特征＋原配方 | 0.959014817 | 0.960477111 | +0.000073051 | [-0.000013874, +0.000159975] |
| 现有66特征＋新配方 | 0.960114463 | 0.960442570 | +0.000038510 | [-0.000007484, +0.000084505] |
| 现有66特征＋原配方 | 0.959845078 | 0.960405602 | +0.000001542 | [-0.000046491, +0.000049574] |

**对照能说明什么**；排序列：`单模型 AUC 差值`。

| 对照 | 单模型 AUC 差值 | 固定融合 AUC 差值 |
|---|---|---|
| 精简47特征＋新配方 相对 精简47特征＋原配方 | +0.001080104 | +0.000101724 |
| 现有66特征＋新配方 相对 现有66特征＋原配方 | +0.000269385 | +0.000036968 |
| 精简47特征＋新配方 相对 现有66特征＋新配方 | -0.000019542 | +0.000136265 |
| 精简47特征＋原配方 相对 现有66特征＋原配方 | -0.000830262 | +0.000071509 |

### 双种子融合及初始TabICL

来源：[FOLLOWUP_20261019_REPORT.md](FOLLOWUP_20261019_REPORT.md)

**RealMLP 稳定性**；排序列：`20%种子42融合 AUC`。

| 分区 | 五分支基线 AUC | 种子42单模型 | 第二种子单模型 | 20%种子42融合 AUC |
|---|---|---|---|---|
| 复核 | 0.961182879 | 0.960868027 | 0.960882749 | 0.961321495 |
| 筛选 | 0.960404060 | 0.960094921 | 0.960110689 | 0.960578836 |

**RealMLP 稳定性**；排序列：`固定20%融合相对基线的增益`。

| 分区/种子 | 固定20%融合相对基线的增益 | 样本层面约95%区间 |
|---|---|---|
| 筛选/20261019 | +0.000180945 | [+0.000109309, +0.000252580] |
| 筛选/42 | +0.000174776 | [+0.000103216, +0.000246336] |
| 复核/20261019 | +0.000152720 | [+0.000092245, +0.000213195] |
| 复核/42 | +0.000138616 | [+0.000079608, +0.000197623] |

**六个固定融合设置**；排序列：`筛选 AUC`。

| 新模型权重 | 形式 | 筛选 AUC | 相对当前20%单种子方案 |
|---|---|---|---|
| 35% | logit | 0.960633383 | +0.000054547 |
| 35% | probability | 0.960629036 | +0.000050201 |
| 20% | logit | 0.960585633 | +0.000006797 |
| 20% | probability | 0.960583787 | +0.000004951 |
| 10% | probability | 0.960513773 | -0.000065063 |
| 10% | logit | 0.960513163 | -0.000065673 |

**TabICLv2**；排序列：`单模型AUC`。

| 上下文行数 | 状态 | 预热后1024行秒数 | 预计全筛选秒数 | 单模型AUC | 对当前融合的增益 |
|---|---|---|---|---|---|
| 32768 | screen_complete | 0.3 | 34.6 | 0.955975887 | -0.000077056 |
| 8192 | screen_complete | 0.2 | 20.3 | 0.954103301 | -0.000322523 |

### TabICL冻结/微调与上下文容量

来源：[TABICL_FINETUNE_REPORT.md](TABICL_FINETUNE_REPORT.md)

**外部筛选结果**；排序列：`单模型AUC`。

| 状态 | 上下文行数 | 单模型AUC | 新成员权重 | 相对最新组合增益 | 固定预测差值约95%区间 |
|---|---|---|---|---|---|
| 冻结 | 65,536 | 0.956252648 | 5% | -0.000001695 | [-0.000024493, +0.000021103] |
| 微调 | 65,536 | 0.956117287 | 5% | -0.000010433 | [-0.000033081, +0.000012214] |
| 冻结 | 32,768 | 0.955978703 | 5% | +0.000005955 | [-0.000019216, +0.000031126] |
| 微调 | 32,768 | 0.955920566 | 5% | -0.000006723 | [-0.000030326, +0.000016880] |

**微调过程**；排序列：`AUC`。

| 内部验证次数（0为更新前） | AUC | 秒 |
|---|---|---|
| 1 | 0.956496953 | 17.7 |
| 3 | 0.956336795 | 18.4 |
| 0 | 0.956234806 | 18.4 |
| 2 | 0.956064899 | 18.3 |

### 遮蔽分类与评分恢复多任务

来源：[DENOISING_MULTITASK_REPORT.md](DENOISING_MULTITASK_REPORT.md)

**单模型**；排序列：`外部计分AUC`。

| 设置 | 外部计分AUC | 最佳总轮次 | 训练分钟 | 峰值分配显存GB |
|---|---|---|---|---|
| B：遮蔽分类→完整输入微调 | 0.959225679 | 13 | 1.07 | 0.39 |
| C：遮蔽分类＋评分恢复→完整输入微调 | 0.959005795 | 16 | 1.21 | 0.39 |
| A：完整输入分类 | 0.958515625 | 13 | 0.96 | 0.39 |

**两种预先固定的融合用途**；排序列：`融合AUC`。

| 设置 | 用途 | 融合AUC | 相对当前组合 | 固定预测差值约95%区间 |
|---|---|---|---|---|
| C_multitask | 20%加入当前组合 | 0.960644660 | +0.000011276 | [-0.000081808, +0.000104361] |
| B_masked | 20%加入当前组合 | 0.960622180 | -0.000011204 | [-0.000087493, +0.000065085] |
| A_clean | 20%加入当前组合 | 0.960611937 | -0.000021447 | [-0.000115436, +0.000072541] |
| C_multitask | 替换35%精简RealMLP | 0.960518695 | -0.000114689 | [-0.000273875, +0.000044496] |
| B_masked | 替换35%精简RealMLP | 0.960511770 | -0.000121614 | [-0.000256606, +0.000013377] |
| A_clean | 替换35%精简RealMLP | 0.960457497 | -0.000175887 | [-0.000335405, -0.000016369] |

**学习目标的消融**；排序列：`单模型AUC差值`。

| 比较 | 单模型AUC差值 | 约95%配对区间 |
|---|---|---|
| B_masked_minus_A_clean | +0.000710054 | [+0.000499385, +0.000920724] |
| C_multitask_minus_A_clean | +0.000490170 | [+0.000202163, +0.000778178] |
| C_multitask_minus_B_masked | -0.000219884 | [-0.000482245, +0.000042477] |

**训练预算的诊断**；排序列：`前阶段最高内部停止AUC`。

| 设置 | 前阶段最高内部停止AUC | 后阶段最高内部停止AUC | 后阶段减前阶段 |
|---|---|---|---|
| C_multitask | 0.960665951 | 0.959827539 | -0.000838412 |
| B_masked | 0.960538142 | 0.960100452 | -0.000437690 |
| A_clean | 0.960023832 | 0.959170533 | -0.000853299 |

**最好初筛用途的评分模式分层**；排序列：`候选融合AUC`。

| 模式是否在拟合区出现 | 行数 | 当前组合AUC | 候选融合AUC |
|---|---|---|---|
| seen_rating_pattern | 64737 | 0.962405886 | 0.962417567 |
| unseen_rating_pattern | 47205 | 0.951207728 | 0.951224434 |

### 普通堆叠与固定起点纠错

来源：[STRONG_CORRECTION_REPORT.md](STRONG_CORRECTION_REPORT.md)

**同折结果**；排序列：`候选AUC`。

| 外层折 | 候选 | 同协议基线AUC | 候选AUC | 差值 | 内部选择轮数 |
|---|---|---|---|---|---|
| 0 | 固定起点纠错 | 0.962195660 | 0.962233041 | +0.000037381 | 2488 |
| 0 | 普通堆叠 | 0.962195660 | 0.962208382 | +0.000012722 | 2302 |
| 1 | 固定起点纠错 | 0.961089904 | 0.961127941 | +0.000038037 | 1015 |
| 1 | 普通堆叠 | 0.961089904 | 0.961103525 | +0.000013621 | 721 |

**同折结果**；排序列：`AUC`。

| 候选 | AUC | 相对基线 | 固定预测差值约95%区间 | 通过门槛 |
|---|---|---|---|---|
| 固定起点纠错 | 0.961668379 | +0.000045313 | [-0.000014779, +0.000105404] | False |
| 普通堆叠 | 0.961639701 | +0.000016635 | [-0.000052535, +0.000085804] | False |

### FFM与加性模型

来源：[FIELD_INTERACTION_REPORT.md](FIELD_INTERACTION_REPORT.md)

**显式字段交互实验**；排序列：`固定融合AUC`。

| 外层折 | 模型 | 单模型AUC | 固定融合AUC | 相对同折当前配方基线 | 内部选择轮数 |
|---|---|---|---|---|---|
| 0 | ffm | 0.959121009 | 0.962127715 | -0.000067945 | 2 |
| 0 | additive | 0.951752321 | 0.961425958 | -0.000769703 | 16 |
| 1 | ffm | 0.957711065 | 0.960808483 | -0.000281421 | 1 |
| 1 | additive | 0.950455361 | 0.960252248 | -0.000837657 | 11 |

**显式字段交互实验**；排序列：`融合增益`。

| 模型 | 融合增益 | 固定预测差值约95%区间 | 通过门槛 |
|---|---|---|---|
| ffm | -0.000158552 | [-0.000285939, -0.000031165] | False |
| additive | -0.000793839 | [-0.000994919, -0.000592760] | False |

### 公开预测池及本地模型纳入对照

来源：[PUBLIC_POOL_TRIAL_REPORT_20261020.md](PUBLIC_POOL_TRIAL_REPORT_20261020.md)

**五折比较**；排序列：`合并OOF AUC`。

| 方案 | 合并OOF AUC | 相对当前 |
|---|---|---|
| 公开246成员池 | 0.962204756 | +0.000530173 |
| 公开池加本地七分支 | 0.962204081 | +0.000529497 |
| 当前最佳历史OOF | 0.961674583 | — |

### Kumo及神经邻居上游联合拟合

来源：[JOINT_UPSTREAM_TRIAL_REPORT_20261021.md](JOINT_UPSTREAM_TRIAL_REPORT_20261021.md)

**全部结果**；排序列：`整体 OOF AUC`。

| 组别 | 整体 OOF AUC | 整体差值 | 原生折平均差值 | 正向折数 |
|---|---|---|---|---|
| 加 Kumo | 0.962205401810 | +0.000000645713 | +0.000000419116 | 4/5 |
| 当前 246 成员，重算基线 | 0.962204756097 | 0 | 0 | — |
| 两者同时加入 | 0.962203170789 | -0.000001585308 | -0.000001817383 | 3/5 |
| 加神经邻居 | 0.962202584473 | -0.000002171624 | -0.000002213239 | 3/5 |

### 比赛/原始数据双域共享学习

来源：[DOMAIN_TRANSFER_REPORT_20261021.md](DOMAIN_TRANSFER_REPORT_20261021.md)

**结果**；排序列：`固定10%融合AUC`。

| 设置 | 内部选择轮数 | 单模型AUC | 固定10%融合AUC | 相对公开池 |
|---|---|---|---|---|
| 两域共用预测头 | 5 | 0.960112324 | 0.962348736 | -0.000010430 |
| 两域独立预测头 | 5 | 0.960455800 | 0.962345743 | -0.000013422 |
| 只学习比赛标签 | 5 | 0.960252026 | 0.962339307 | -0.000019859 |

**结果**；排序列：`5%融合差值`。

| 设置 | 5%融合差值 | 10%融合差值 | 20%融合差值 |
|---|---|---|---|
| 两域共用预测头 | +0.000001054 | -0.000010430 | -0.000071419 |
| 两域独立预测头 | -0.000001565 | -0.000013422 | -0.000067317 |
| 只学习比赛标签 | -0.000004061 | -0.000019859 | -0.000084688 |

### 类别条件域密度比

来源：[DENSITY_RATIO_REPORT_20261021.md](DENSITY_RATIO_REPORT_20261021.md)

**同区结果**；排序列：`10%融合AUC`。

| 训练方法 | 选定轮数 | 单模型AUC | 10%融合AUC | 相对当前池 | 配对95%区间 |
|---|---|---|---|---|---|
| 直接分类 | 8 | 0.9600394441 | 0.9627774146 | -0.0000215137 | [-0.0000607751, +0.0000177477] |
| 普通教师修正 | 15 | 0.9594294298 | 0.9627355338 | -0.0000633944 | [-0.0001322029, +0.0000054140] |
| 类别条件域密度比 | 1 | 0.9554107710 | 0.9623277553 | -0.0004711730 | [-0.0006742865, -0.0002680594] |

### TabICL局部上下文联合微调

来源：[LOCAL_CONTEXT_REPORT_20261021.md](LOCAL_CONTEXT_REPORT_20261021.md)

**同一计分行的结果**；排序列：`固定10%融合AUC`。

| 设置 | 单模型AUC | 固定10%融合AUC | 相对当前池 |
|---|---|---|---|
| 随机上下文，微调 | 0.952810460 | 0.962125947 | -0.000069880 |
| 随机上下文，冻结 | 0.951845352 | 0.962082109 | -0.000113718 |
| 局部上下文，局部微调 | 0.952188088 | 0.962022717 | -0.000173110 |
| 局部上下文，冻结 | 0.951751025 | 0.962018410 | -0.000177417 |

**同一计分行的结果**；排序列：`5%差值`。

| 设置 | 5%差值 | 10%差值 | 20%差值 |
|---|---|---|---|
| 随机上下文，微调 | -0.000007784 | -0.000069880 | -0.000334424 |
| 随机上下文，冻结 | -0.000021853 | -0.000113718 | -0.000477597 |
| 局部上下文，冻结 | -0.000060645 | -0.000177417 | -0.000568006 |
| 局部上下文，局部微调 | -0.000061822 | -0.000173110 | -0.000532562 |

### 额外评分信息及条件积分：仅原始源域

来源：[PRIVILEGED_SIGNAL_REPORT_20261021.md](PRIVILEGED_SIGNAL_REPORT_20261021.md)

**第一阶段：真实评分是否有价值**；排序列：`源留出AUC`。

| 源域教师 | 选定树数 | 源留出AUC | logloss | 比赛查询是否具备全部输入 |
|---|---|---|---|---|
| 21字段+真实额外评分 | 818 | 0.9954939473 | 0.08270787 | 不具备，仅作信息对照 |
| 21字段+打乱额外评分 | 751 | 0.9948236784 | 0.08868094 | 没有可用信息内容，仅作安慰剂对照 |
| 共同21字段 | 853 | 0.9948177526 | 0.08847191 | 具备 |

**第二阶段：不读取真实评分能否保留收益**；排序列：`源留出AUC`。

| 未知评分处理规则 | 源留出AUC | 角色 |
|---|---|---|
| 使用真实额外评分 | 0.9954939473 | 不可部署的信息对照 |
| 直接21字段教师 | 0.9948177526 | 主对照 |
| 按每行六类概率作条件积分 | 0.9947610401 | 唯一主候选 |
| 直接填预测最可能评分 | 0.9946045187 | 固定消融 |
| 按源训练区全局评分分布积分 | 0.9936145746 | 固定消融 |

### 排序损失融合：内部筛选

来源：[RANKING_FUSION_REPORT_20261022.md](RANKING_FUSION_REPORT_20261022.md)

**排序目标融合：限时初筛结果**；排序列：`候选AUC`。

| 分区 | 对照AUC | 候选AUC | 差值 | 配对95%区间 |
|---|---|---|---|---|
| inner | 0.9620133931 | 0.9620093335 | -0.0000040596 | [-1.738974751130854e-05, 9.270642604644489e-06] |

### 公开池成员分歧纠错：内部筛选

来源：[POOL_REPAIRABILITY_REPORT_20261022.md](POOL_REPAIRABILITY_REPORT_20261022.md)

**学习这些分歧是否有用**；排序列：`内部AUC`。

| 模型 | 内部AUC | 相对LR差值 | 内部选择树数 |
|---|---|---|---|
| 加246列分歧纠错 | 0.9620195209 | +0.0000061278 | 11 |
| 当前LR | 0.9620133931 | — | — |
| 仅原始字段+总分纠错 | 0.9620130918 | -0.0000003013 | 1 |

### Transformer与RTD同条件比较

来源：[RTD_TRIAL_REPORT_20261023.md](RTD_TRIAL_REPORT_20261023.md)

**实测结果**；排序列：`AUC`。

| 模型 | AUC | 与对应对照的差值 |
|---|---|---|
| 原有最优融合在该折的预测 | 0.962359165 | 新 RTD 融合低 0.000096571 |
| 同条件基线＋RTD 模型 | 0.962262594 | +0.000016589 |
| 同条件公开模型融合基线 | 0.962246005 | — |
| 同条件基线＋无预训练模型 | 0.962241734 | −0.000004271 |
| RTD 预训练 Transformer 单模型 | 0.961284184 | +0.002024719 |
| 无预训练 Transformer 单模型 | 0.959259465 | — |

### 原始数据条件近邻（报告正文中的结果）

| 设置 | 单模型AUC | 融合AUC | 相对同条件基线 | 证据 |
|---|---|---|---|---|
| 按客群条件检索16/64原始数据近邻 | 0.960031792 | 0.960634072 | +0.000000690 | [FOLLOWUP_20261019_REPORT.md](FOLLOWUP_20261019_REPORT.md) |

## 四、早期候选及参数/融合搜索逐项记录

下列为历史开发集选型成绩，没有重新评估；同一配置也可能在上面的最终报告中出现。配置名和全部权重保留，以便逐项追溯；不把不同权重等同于新算法。

### optimization_search.json / records

来源：[optimization_search.json](optimization_search.json)

| 序号 | 配置/权重 | 开发集AUC | JSON原索引 |
|---|---|---|---|
| 1 | `{"name":"catboost_allcat_original_d8","family":"catboost","use_te":false,"numeric_categories":true,"depth":8,"best_rounds":1054,"features":115,"original":true}` | 0.960281975054 | 15 |
| 2 | `{"name":"catboost_allcat_original","family":"catboost","use_te":false,"numeric_categories":true,"depth":6,"best_rounds":2068,"features":115,"original":true}` | 0.960248620686 | 14 |
| 3 | `{"name":"lightgbm_te_original","family":"lightgbm","use_te":true,"numeric_categories":false,"depth":-1,"best_rounds":495,"features":136,"original":true}` | 0.959812526560 | 16 |
| 4 | `{"name":"te_original_d7","use_te":true,"original":true,"depth":7,"child":20,"lr":0.04,"best_rounds":273,"features":136,"family":"xgboost"}` | 0.959762206602 | 4 |
| 5 | `{"name":"te_digits_original_d4","use_te":true,"original":true,"numeric_categories":false,"digits":true,"depth":4,"child":20,"lr":0.035,"best_rounds":1245,"features":160,"family":"xgboost"}` | 0.959742786154 | 7 |
| 6 | `{"name":"multi_prior_te_d5","family":"xgboost","use_te":true,"original":true,"multiple_priors":true,"extended_keys":false,"depth":5,"child":10,"lr":0.035,"best_rounds":897,"features":138}` | 0.959725797768 | 10 |
| 7 | `{"name":"digits_original_d5","use_te":false,"original":true,"numeric_categories":false,"digits":true,"depth":5,"child":10,"lr":0.035,"best_rounds":1199,"features":122,"family":"xgboost"}` | 0.959713806880 | 6 |
| 8 | `{"name":"te_original_d5","use_te":true,"original":true,"depth":5,"child":10,"lr":0.04,"best_rounds":359,"features":136,"family":"xgboost"}` | 0.959711884057 | 3 |
| 9 | `{"name":"interaction_multi_prior_d7","family":"xgboost","use_te":true,"original":true,"multiple_priors":true,"extended_keys":true,"depth":7,"child":10,"lr":0.035,"best_rounds":373,"features":168}` | 0.959693892227 | 12 |
| 10 | `{"name":"interaction_multi_prior_d5","family":"xgboost","use_te":true,"original":true,"multiple_priors":true,"extended_keys":true,"depth":5,"child":10,"lr":0.035,"best_rounds":695,"features":168}` | 0.959647806230 | 11 |
| 11 | `{"name":"catboost_original","family":"catboost","use_te":false,"numeric_categories":false,"depth":6,"best_rounds":1317,"features":98,"original":true}` | 0.959507687818 | 13 |
| 12 | `{"name":"original_d5","use_te":false,"original":true,"depth":5,"child":10,"lr":0.04,"best_rounds":614,"features":98,"family":"xgboost"}` | 0.959438835267 | 2 |
| 13 | `{"name":"allcat_original_d5","use_te":false,"original":true,"numeric_categories":true,"digits":false,"depth":5,"child":10,"lr":0.035,"best_rounds":247,"features":115,"family":"xgboost"}` | 0.958403061722 | 5 |
| 14 | `{"name":"raw_original_rows1","use_te":false,"original":false,"rows_weight":1.0,"depth":5,"child":10,"lr":0.035,"best_rounds":2563,"features":22,"family":"xgboost"}` | 0.958243491338 | 9 |
| 15 | `{"name":"raw_original_rows025","use_te":false,"original":false,"rows_weight":0.25,"depth":5,"child":10,"lr":0.035,"best_rounds":2145,"features":22,"family":"xgboost"}` | 0.958056182446 | 8 |
| 16 | `{"name":"raw_native_d5","use_te":false,"original":false,"depth":5,"child":10,"lr":0.04,"best_rounds":1541,"features":21,"family":"xgboost"}` | 0.957915725672 | 0 |
| 17 | `{"name":"raw_native_d7","use_te":false,"original":false,"depth":7,"child":20,"lr":0.04,"best_rounds":714,"features":21,"family":"xgboost"}` | 0.957877824040 | 1 |

### optimization_search.json / blend_trials

来源：[optimization_search.json](optimization_search.json)

| 序号 | 配置/权重 | 开发集AUC | JSON原索引 |
|---|---|---|---|
| 1 | `{"components":[{"name":"te_original_d7","weight":0.25},{"name":"catboost_allcat_original_d8","weight":0.75}]}` | 0.960270327215 | 1 |
| 2 | `{"components":[{"name":"te_digits_original_d4","weight":0.25},{"name":"catboost_allcat_original_d8","weight":0.75}]}` | 0.960259352865 | 8 |
| 3 | `{"components":[{"name":"multi_prior_te_d5","weight":0.25},{"name":"catboost_allcat_original_d8","weight":0.75}]}` | 0.960255269574 | 15 |
| 4 | `{"components":[{"name":"te_original_d7","weight":0.5},{"name":"catboost_allcat_original_d8","weight":0.5}]}` | 0.960178803814 | 2 |
| 5 | `{"components":[{"name":"te_digits_original_d4","weight":0.5},{"name":"catboost_allcat_original_d8","weight":0.5}]}` | 0.960157611698 | 9 |
| 6 | `{"components":[{"name":"multi_prior_te_d5","weight":0.5},{"name":"catboost_allcat_original_d8","weight":0.5}]}` | 0.960148933290 | 16 |
| 7 | `{"components":[{"name":"te_original_d7","weight":0.75},{"name":"catboost_allcat_original_d8","weight":0.25}]}` | 0.960010115620 | 3 |
| 8 | `{"components":[{"name":"te_digits_original_d4","weight":0.75},{"name":"catboost_allcat_original_d8","weight":0.25}]}` | 0.959984466448 | 10 |
| 9 | `{"components":[{"name":"multi_prior_te_d5","weight":0.75},{"name":"catboost_allcat_original_d8","weight":0.25}]}` | 0.959971660130 | 17 |
| 10 | `{"components":[{"name":"te_digits_original_d4","weight":0.5},{"name":"lightgbm_te_original","weight":0.5}]}` | 0.959874234403 | 12 |
| 11 | `{"components":[{"name":"te_digits_original_d4","weight":0.25},{"name":"lightgbm_te_original","weight":0.75}]}` | 0.959873223361 | 11 |
| 12 | `{"components":[{"name":"multi_prior_te_d5","weight":0.25},{"name":"lightgbm_te_original","weight":0.75}]}` | 0.959859148632 | 18 |
| 13 | `{"components":[{"name":"te_original_d7","weight":0.5},{"name":"lightgbm_te_original","weight":0.5}]}` | 0.959857064793 | 5 |
| 14 | `{"components":[{"name":"multi_prior_te_d5","weight":0.5},{"name":"lightgbm_te_original","weight":0.5}]}` | 0.959856146869 | 19 |
| 15 | `{"components":[{"name":"te_original_d7","weight":0.25},{"name":"lightgbm_te_original","weight":0.75}]}` | 0.959852878037 | 4 |
| 16 | `{"components":[{"name":"te_digits_original_d4","weight":0.75},{"name":"lightgbm_te_original","weight":0.25}]}` | 0.959829160418 | 13 |
| 17 | `{"components":[{"name":"te_original_d7","weight":0.75},{"name":"lightgbm_te_original","weight":0.25}]}` | 0.959827154177 | 6 |
| 18 | `{"components":[{"name":"multi_prior_te_d5","weight":0.75},{"name":"lightgbm_te_original","weight":0.25}]}` | 0.959810401011 | 20 |
| 19 | `{"components":[{"name":"te_original_d7","weight":1.0}]}` | 0.959762206602 | 0 |
| 20 | `{"components":[{"name":"te_digits_original_d4","weight":1.0}]}` | 0.959742786154 | 7 |
| 21 | `{"components":[{"name":"multi_prior_te_d5","weight":1.0}]}` | 0.959725797768 | 14 |

### v2_recipe.json / nn_candidates

来源：[v2_recipe.json](v2_recipe.json)

| 序号 | 配置/权重 | 开发集AUC | JSON原索引 |
|---|---|---|---|
| 1 | `{"config":{"name":"tabm_ple","k":16,"width":256,"blocks":2,"embedding":16,"bins":32,"dropout":0.1,"lr":0.002,"weight_decay":0.0003,"batch_size":1024,"max_epochs":60,"patience":10},"seed":42,"features":111}` | 0.959065738895 | 0 |
| 2 | `{"config":{"name":"tabm_compact","k":32,"width":512,"blocks":2,"embedding":16,"bins":48,"dropout":0.1,"lr":0.001,"weight_decay":0.0003,"batch_size":1024,"max_epochs":60,"patience":10,"raw_only":true},"seed":42,"features":35}` | 0.958069604701 | 1 |

### v2_recipe.json / trials

来源：[v2_recipe.json](v2_recipe.json)

| 序号 | 配置/权重 | 开发集AUC | JSON原索引 |
|---|---|---|---|
| 1 | `{"weights":[0.4,0.6,0.0]}` | 0.960328966365 | 12 |
| 2 | `{"weights":[0.6,0.4,0.0]}` | 0.960322829950 | 9 |
| 3 | `{"weights":[0.19999999999999996,0.8,0.0]}` | 0.960321393734 | 14 |
| 4 | `{"weights":[0.8,0.2,0.0]}` | 0.960303393498 | 5 |
| 5 | `{"weights":[0.2,0.6,0.2]}` | 0.960302501763 | 13 |
| 6 | `{"weights":[0.39999999999999997,0.4,0.2]}` | 0.960302027443 | 10 |
| 7 | `{"weights":[0.6000000000000001,0.2,0.2]}` | 0.960289801173 | 6 |
| 8 | `{"weights":[1.0,0.0,0.0]}` | 0.960270327215 | 0 |
| 9 | `{"weights":[0.8,0.0,0.2]}` | 0.960262245667 | 1 |
| 10 | `{"weights":[0.19999999999999996,0.4,0.4]}` | 0.960195308987 | 11 |
| 11 | `{"weights":[0.4,0.2,0.4]}` | 0.960185557135 | 7 |
| 12 | `{"weights":[0.6,0.0,0.4]}` | 0.960161928107 | 2 |
| 13 | `{"weights":[0.20000000000000007,0.2,0.6]}` | 0.959979470730 | 8 |
| 14 | `{"weights":[0.4,0.0,0.6]}` | 0.959958014133 | 3 |
| 15 | `{"weights":[0.19999999999999996,0.0,0.8]}` | 0.959623501127 | 4 |

### v3_recipe.json / trials

来源：[v3_recipe.json](v3_recipe.json)

| 序号 | 配置/权重 | 开发集AUC | JSON原索引 |
|---|---|---|---|
| 1 | `{"weights":[0.39999999999999997,0.4,0.2]}` | 0.960548911840 | 10 |
| 2 | `{"weights":[0.4,0.6,0.0]}` | 0.960518043854 | 12 |
| 3 | `{"weights":[0.6,0.4,0.0]}` | 0.960514936039 | 9 |
| 4 | `{"weights":[0.2,0.6,0.2]}` | 0.960514786986 | 13 |
| 5 | `{"weights":[0.6000000000000001,0.2,0.2]}` | 0.960505367513 | 6 |
| 6 | `{"weights":[0.19999999999999996,0.4,0.4]}` | 0.960496536172 | 11 |
| 7 | `{"weights":[0.4,0.2,0.4]}` | 0.960487589726 | 7 |
| 8 | `{"weights":[0.19999999999999996,0.8,0.0]}` | 0.960451001446 | 14 |
| 9 | `{"weights":[0.8,0.2,0.0]}` | 0.960434257657 | 5 |
| 10 | `{"weights":[0.6,0.0,0.4]}` | 0.960398281758 | 2 |
| 11 | `{"weights":[0.20000000000000007,0.2,0.6]}` | 0.960384079963 | 8 |
| 12 | `{"weights":[0.8,0.0,0.2]}` | 0.960380485218 | 1 |
| 13 | `{"weights":[0.4,0.0,0.6]}` | 0.960328597773 | 3 |
| 14 | `{"weights":[1.0,0.0,0.0]}` | 0.960270327215 | 0 |
| 15 | `{"weights":[0.19999999999999996,0.0,0.8]}` | 0.960170082726 | 4 |

### v4_recipe.json / trials

来源：[v4_recipe.json](v4_recipe.json)

| 序号 | 配置/权重 | 开发集AUC | JSON原索引 |
|---|---|---|---|
| 1 | `{"spec":{"name":"lgb_63","family":"lightgbm","transform":"full","leaves":63},"mode":"mix","weight":0.3,"rounds":696}` | 0.960596145323 | 12 |
| 2 | `{"spec":{"name":"lgb_63","family":"lightgbm","transform":"full","leaves":63},"mode":"mix","weight":0.4,"rounds":696}` | 0.960595638994 | 13 |
| 3 | `{"spec":{"name":"lgb_63","family":"lightgbm","transform":"full","leaves":63},"mode":"mix","weight":0.2,"rounds":696}` | 0.960588782531 | 11 |
| 4 | `{"spec":{"name":"lgb_127","family":"lightgbm","transform":"full","leaves":127},"mode":"mix","weight":0.2,"rounds":455}` | 0.960573253803 | 15 |
| 5 | `{"spec":{"name":"lgb_63","family":"lightgbm","transform":"full","leaves":63},"mode":"mix","weight":0.1,"rounds":696}` | 0.960573153572 | 10 |
| 6 | `{"spec":{"name":"lgb_127","family":"lightgbm","transform":"full","leaves":127},"mode":"mix","weight":0.3,"rounds":455}` | 0.960573015835 | 16 |
| 7 | `{"spec":{"name":"xgb_residual","family":"xgb","transform":"residual"},"mode":"replace_xgb","weight":0.4,"rounds":661}` | 0.960570264002 | 5 |
| 8 | `{"spec":{"name":"lgb_127","family":"lightgbm","transform":"full","leaves":127},"mode":"mix","weight":0.1,"rounds":455}` | 0.960566090180 | 14 |
| 9 | `{"spec":{"name":"lgb_127","family":"lightgbm","transform":"full","leaves":127},"mode":"mix","weight":0.4,"rounds":455}` | 0.960563983707 | 17 |
| 10 | `{"spec":{"name":"without_teacher","family":"xgb","transform":"without_teacher"},"mode":"replace_xgb","weight":0.4,"rounds":1241}` | 0.960556405903 | 3 |
| 11 | `{"spec":{"name":"lgb_15","family":"lightgbm","transform":"full","leaves":15},"mode":"mix","weight":0.1,"rounds":1584}` | 0.960546252479 | 6 |
| 12 | `{"spec":{"name":"lgb_15","family":"lightgbm","transform":"full","leaves":15},"mode":"mix","weight":0.2,"rounds":1584}` | 0.960537091021 | 7 |
| 13 | `{"spec":{"name":"lgb_15","family":"lightgbm","transform":"full","leaves":15},"mode":"mix","weight":0.3,"rounds":1584}` | 0.960520416101 | 8 |
| 14 | `{"spec":{"name":"without_frequency","family":"xgb","transform":"without_frequency"},"mode":"replace_xgb","weight":0.4,"rounds":612}` | 0.960503556884 | 1 |
| 15 | `{"spec":{"name":"lgb_15","family":"lightgbm","transform":"full","leaves":15},"mode":"mix","weight":0.4,"rounds":1584}` | 0.960497983058 | 9 |
| 16 | `{"spec":{"name":"without_target_encoding","family":"xgb","transform":"without_target_encoding"},"mode":"replace_xgb","weight":0.4,"rounds":806}` | 0.960475609507 | 2 |
| 17 | `{"spec":{"name":"without_auxiliary","family":"xgb","transform":"without_auxiliary"},"mode":"replace_xgb","weight":0.4,"rounds":619}` | 0.960460301935 | 4 |
| 18 | `{"spec":{"name":"without_route","family":"xgb","transform":"without_route"},"mode":"replace_xgb","weight":0.4,"rounds":636}` | 0.960406284414 | 0 |

### v4_recipe.json / ablations

来源：[v4_recipe.json](v4_recipe.json)

| 序号 | 配置/权重 | 开发集AUC | JSON原索引 |
|---|---|---|---|
| 1 | `{"group":"frequency"}` | 0.960220643240 | 1 |
| 2 | `{"group":"auxiliary"}` | 0.960104219167 | 4 |
| 3 | `{"group":"target_encoding"}` | 0.960062797479 | 2 |
| 4 | `{"group":"route"}` | 0.960026350493 | 0 |
| 5 | `{"group":"teacher"}` | 0.959848485969 | 3 |

### CatBoost七组参数候选

来源：[.artifact_runtime/v2/cat_screen.json](.artifact_runtime/v2/cat_screen.json)

| 序号 | 参数 | 开发集AUC |
|---|---|---|
| 1 | {"name": "cat_d7_l10", "depth": 7, "l2_leaf_reg": 10, "learning_rate": 0.04} | 0.960301261159 |
| 2 | {"name": "cat_control", "depth": 8, "l2_leaf_reg": 5, "learning_rate": 0.04} | 0.960267505544 |
| 3 | {"name": "cat_d8_l20_slow", "depth": 8, "l2_leaf_reg": 20, "learning_rate": 0.025} | 0.960250685450 |
| 4 | {"name": "cat_d8_l50", "depth": 8, "l2_leaf_reg": 50, "learning_rate": 0.04} | 0.960228427003 |
| 5 | {"name": "cat_d6_l20", "depth": 6, "l2_leaf_reg": 20, "learning_rate": 0.04} | 0.960210634989 |
| 6 | {"name": "cat_source_prior", "depth": 8, "l2_leaf_reg": 5, "learning_rate": 0.04, "source_prior": true} | 0.960207540431 |
| 7 | {"name": "cat_d9_l10", "depth": 9, "l2_leaf_reg": 10, "learning_rate": 0.04} | 0.960055372285 |

## 五、补充端点评分索引

包括报告中可能只展示最佳融合比例的明细；按文件分别排序。JSON路径是原始定位，不代表新增方法。

来源：[.artifact_runtime/followup_20261019/4261d13e038c1512ffc1/confirmation/summary.json](.artifact_runtime/followup_20261019/4261d13e038c1512ffc1/confirmation/summary.json)

| 指标路径 | AUC |
|---|---|
| $.trials[0].auc | 0.961383226277 |
| $.best.auc | 0.961383226277 |
| $.current_auc | 0.961321494923 |
| $.baseline_auc | 0.961182879148 |
| $.seeds.20261019.auc | 0.960882749346 |
| $.seeds.42.auc | 0.960868026588 |

来源：[.artifact_runtime/followup_20261019/4261d13e038c1512ffc1/screen/summary.json](.artifact_runtime/followup_20261019/4261d13e038c1512ffc1/screen/summary.json)

| 指标路径 | AUC |
|---|---|
| $.trials[5].auc | 0.960633382842 |
| $.best.auc | 0.960633382842 |
| $.trials[4].auc | 0.960629036363 |
| $.trials[3].auc | 0.960585632683 |
| $.trials[2].auc | 0.960583786651 |
| $.current_auc | 0.960578835551 |
| $.trials[0].auc | 0.960513772710 |
| $.trials[1].auc | 0.960513162916 |
| $.baseline_auc | 0.960404059766 |
| $.seeds.20261019.auc | 0.960110688608 |
| $.seeds.42.auc | 0.960094920942 |

来源：[.artifact_runtime/neural_representation/5fd485f228259b9bd183/summary.json](.artifact_runtime/neural_representation/5fd485f228259b9bd183/summary.json)

| 指标路径 | AUC |
|---|---|
| $.trials[0].auc | 0.960607667065 |
| $.best.auc | 0.960607667065 |
| $.trials[3].auc | 0.960597020891 |
| $.v4_selection_auc | 0.960596145323 |
| $.matched_control_blend_auc | 0.960581964220 |
| $.trials[4].auc | 0.960540888815 |
| $.trials[1].auc | 0.960540870385 |
| $.trials[5].auc | 0.960363626002 |
| $.trials[2].auc | 0.960293999892 |
| $.models[0].auc | 0.959470270853 |
| $.models[2].auc | 0.959456435548 |
| $.models[1].auc | 0.958647483826 |

来源：[.artifact_runtime/source_neighbors_20261019/849013e6a93e03c753dd/screen/summary.json](.artifact_runtime/source_neighbors_20261019/849013e6a93e03c753dd/screen/summary.json)

| 指标路径 | AUC |
|---|---|
| $.blend_auc | 0.960634072498 |
| $.anchor_auc | 0.960633382842 |
| $.auc | 0.960031792241 |

来源：[.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/data.json](.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/data.json)

| 指标路径 | AUC |
|---|---|
| $.current_auc | 0.960633382842 |

来源：[.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/decision.json](.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/decision.json)

| 指标路径 | AUC |
|---|---|
| $.cells[0].trials[0].auc | 0.960639337871 |
| $.trials[0].auc | 0.960639337871 |
| $.cells[0].baseline_auc | 0.960633382842 |
| $.cells[1].baseline_auc | 0.960633382842 |
| $.cells[2].baseline_auc | 0.960633382842 |
| $.cells[3].baseline_auc | 0.960633382842 |
| $.cells[1].trials[0].auc | 0.960631687965 |
| $.trials[2].auc | 0.960631687965 |
| $.cells[2].trials[0].auc | 0.960626659590 |
| $.trials[4].auc | 0.960626659590 |
| $.cells[3].trials[0].auc | 0.960622949418 |
| $.trials[6].auc | 0.960622949418 |
| $.cells[0].trials[1].auc | 0.960556314078 |
| $.trials[1].auc | 0.960556314078 |
| $.cells[1].trials[1].auc | 0.960536681690 |
| $.trials[3].auc | 0.960536681690 |
| $.cells[2].trials[1].auc | 0.960522763128 |
| $.trials[5].auc | 0.960522763128 |
| $.cells[3].trials[1].auc | 0.960508898562 |
| $.trials[7].auc | 0.960508898562 |
| $.cells[1].auc | 0.956252648335 |
| $.cells[3].auc | 0.956117287350 |
| $.cells[0].auc | 0.955978703284 |
| $.cells[2].auc | 0.955920566260 |

来源：[.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/finetuned_32768.json](.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/finetuned_32768.json)

| 指标路径 | AUC |
|---|---|
| $.baseline_auc | 0.960633382842 |
| $.trials[0].auc | 0.960626659590 |
| $.trials[1].auc | 0.960522763128 |
| $.auc | 0.955920566260 |

来源：[.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/finetuned_65536.json](.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/finetuned_65536.json)

| 指标路径 | AUC |
|---|---|
| $.baseline_auc | 0.960633382842 |
| $.trials[0].auc | 0.960622949418 |
| $.trials[1].auc | 0.960508898562 |
| $.auc | 0.956117287350 |

来源：[.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/frozen_32768.json](.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/frozen_32768.json)

| 指标路径 | AUC |
|---|---|
| $.trials[0].auc | 0.960639337871 |
| $.baseline_auc | 0.960633382842 |
| $.trials[1].auc | 0.960556314078 |
| $.auc | 0.955978703284 |

来源：[.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/frozen_65536.json](.artifact_runtime/tabicl_finetune_20261019/7d7dc513cbabbc16b430/screen/frozen_65536.json)

| 指标路径 | AUC |
|---|---|
| $.baseline_auc | 0.960633382842 |
| $.trials[0].auc | 0.960631687965 |
| $.trials[1].auc | 0.960536681690 |
| $.auc | 0.956252648335 |

来源：[.artifact_runtime/tabicl_finetune_20261019/93c9003c772d241f9dd5/screen/data.json](.artifact_runtime/tabicl_finetune_20261019/93c9003c772d241f9dd5/screen/data.json)

| 指标路径 | AUC |
|---|---|
| $.current_auc | 0.960633382842 |

来源：[.artifact_runtime/tabicl_finetune_20261019/93c9003c772d241f9dd5/screen/frozen_32768.json](.artifact_runtime/tabicl_finetune_20261019/93c9003c772d241f9dd5/screen/frozen_32768.json)

| 指标路径 | AUC |
|---|---|
| $.trials[0].auc | 0.960639337871 |
| $.baseline_auc | 0.960633382842 |
| $.trials[1].auc | 0.960556314078 |
| $.auc | 0.955978703284 |

来源：[.artifact_runtime/tabicl_finetune_20261019/f82b87bada67842d568f/screen/data.json](.artifact_runtime/tabicl_finetune_20261019/f82b87bada67842d568f/screen/data.json)

| 指标路径 | AUC |
|---|---|
| $.current_auc | 0.960633382842 |

来源：[.artifact_runtime/tabicl_probe_20261019/e1f1c42fb3af84f7815f/summary.json](.artifact_runtime/tabicl_probe_20261019/e1f1c42fb3af84f7815f/summary.json)

| 指标路径 | AUC |
|---|---|
| $.rows[0].current_auc | 0.960578836521 |
| $.rows[1].current_auc | 0.960578836521 |
| $.rows[1].blend_auc | 0.960501780205 |
| $.rows[0].blend_auc | 0.960256313595 |
| $.rows[1].auc | 0.955975886787 |
| $.rows[0].auc | 0.954103300956 |

## 六、没有独立可排名分数的内容

- 教师先验、目标编码、航程画像、频率、13项辅助评分预测等是组合的组成部分。其独立贡献以消融表记录，不能把完整组合分数冒充各特征自身的分数。
- 原始域额外真实评分是不可部署的信息对照；0.995左右的源域分数不属于本比赛，不进入公榜或比赛OOF排名。
- RTD检测损失、评分恢复准确率、训练集AUC、吞吐/资源探针以及中断且没有端点评分的配置不作为比赛成绩。
- 本轮引用的外部作者成绩、只阅读未运行的方法、新51成员的作者侧刷新试验不算我们实际完成的实验。
- 上述原始报告及历史JSON均保留。新清单只汇总，不修改历史实验记录与当前最佳文件。

## 登记范围统计

这些数量是证据行数，含对照、复核、逐折记录和重复引用；不能相加当作独立实验次数。

| 项目 | 数量 |
|---|---|
| public_versions | 6 |
| full_oof_rows | 12 |
| original_report_sources | 20 |
| auc_or_delta_tables | 39 |
| rows_in_report_tables | 162 |
| early_search_rows | 100 |
| supplementary_endpoint_metrics | 85 |
| counts_are_evidence_rows_not_unique_methods | True |
