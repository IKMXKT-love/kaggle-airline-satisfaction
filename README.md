# Kaggle Airline Satisfaction Research Archive

Playground Series S6E10 航空乘客满意度预测：方法、源码、完整实验记录与可恢复归档。

**用户反馈的最高公榜 AUC：0.96187。该结果主要来自复现公开预测池及其融合方案，并非本项目独立训练出的原创最高分。** 我们取得22位作者的246个基础预测，沿用logit＋L2逻辑回归（C=0.003）重新拟合与验证；本地七个基础分支没有稳定增益，最终未加入。

| 用户反馈公榜 AUC | 方法 |
|---:|---|
| 0.96187 | 公开246预测池＋L2逻辑回归融合 |
| 0.96114 | v4＋双种子十折精简RealMLP |
| 0.96104 | v4＋全量训练精简RealMLP |
| 0.96082 | v4：v3＋LightGBM |
| 0.96074 | v3：航程画像、辅助评分预测及树模型/RealMLP融合 |
| 0.96036 | 原始数据先验、目标编码及XGBoost/CatBoost融合 |

成绩来自用户反馈，没有独立查询Kaggle提交记录。本地OOF、源域成绩与公榜分开记录；不宣称达到0.962或保证私榜泛化。

## 从这里阅读

- [完整方法与得分登记](EXPERIMENT_SCORE_CATALOG_20261023.md)：公榜、OOF、逐项对照、消融及早期参数/权重搜索。
- [方法与源码导航](docs/METHODS.md)：各条路线对应的入口、支持模块、报告和核验程序。
- [复现与恢复](docs/REPRODUCIBILITY.md)：数据、环境、历史脚本限制及完整归档恢复。
- [数据、模型与分卷说明](docs/ARTIFACTS.md)：普通仓库和Release各保存什么。
- [来源与贡献边界](docs/ATTRIBUTION.md)：公开预测、参考方案与第三方许可范围。
- [当前最佳结果](CURRENT_BEST_RESULT.md)、[最后一次RTD实验报告](RTD_TRIAL_REPORT_20261023.md)。

## 源码布局

源码保持原始相对位置，便于保留既有导入及缓存定位：`main*.py` 为历史训练入口，`experiment_*.py` 为机制实验，`*_support.py` 为实现，`report_*.py` 为报告生成，`verify_*.py` 与 `tests/` 为核验，`research_s6e10/` 为研究和来源记录。根目录的 `main.py` 默认生成较早的 `submission_optimized.csv`，**不会直接复现最高分版本**。

## 完整归档

[Release archive-20261023](https://github.com/IKMXKT-love/kaggle-airline-satisfaction/releases/tag/archive-20261023) 保存当前项目全部普通文件的分卷ZIP，以及逐文件SHA-256清单。包括比赛/原始数据、历史预测、权重、缓存、依赖环境和原始Git元数据。符号链接记录目标但不遍历项目外的内容。各卷是独立ZIP，恢复程序会逐卷检查并恢复到新目录。

普通Git仓库提供便于阅读的源码、报告与小型元数据；不把未上传到普通Git的大文件描述为已经包含在源码克隆中。历史报告中的本机路径只作原始证据，不保证在另一台机器直接可用。

第三方代码、数据、预测和权重保留各自来源及许可；本归档未为这些材料重新授权。没有为整个混合归档附加统一开源许可证。
