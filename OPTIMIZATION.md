# 参考第二名方案的航空满意度优化

## 本次最终结果

完整 10 折平均 AUC 为 **0.961119**，整体折外 AUC 为 **0.961112**。
原方案平均 AUC 为 0.958703，提升 **0.002416**，10 折均改善；尚未达到 0.965。
独立保留集 AUC 为 0.960901。17 项测试、输入文件校验及 20 个模型检查点的预测重建核对通过。
正式结果为 `submission_optimized.csv`，汇总见 [optimization_report.md](optimization_report.md)。

## 来源与迁移范围

参考 [Xin Feng 的第二名方案](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/2nd-place-solution)。
它对应手机成瘾预测（S6E8），当前数据对应 [航空满意度预测（S6E10）](https://www.kaggle.com/competitions/playground-series-s6e10)。
本项目迁移原始数据补充信息、折内目标编码、分箱及数字特征对照、小范围模型筛选等方法，
没有照搬手机专用字段，也没有使用其他参赛者的测试集预测。

## 运行入口

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --device cuda --n-jobs 8
```

没有 NVIDIA 显卡时可用 `--device cpu`。当前目录的 `train.csv`、`test.csv` 和
`sample_submission.csv` 是比赛数据；`external_data` 中的同名文件是另外下载的公开原始航空数据，不能混淆。
程序不会覆盖这两组输入数据。

默认输出：

- `submission_optimized.csv`：299,844 条测试集概率预测，按模板 ID 排列。
- `submission_optimized_oof.csv`：699,635 条真实训练数据的折外预测，每行只属于一个验证折。
- `submission_optimized_metrics.json`：10 折评分、整体折外评分、各模型结果、配置和原始数据指纹。

默认优化流程不使用上一轮的人工插值合成。旧流程可通过 `--pipeline legacy` 运行；
`--demo` 仍自动使用旧流程，避免把模拟字段送入航空专用模型。

## 本次采用的技术

1. **保留数值 0。** 使用密集数值列和 XGBoost 原生类别，避免稀疏矩阵将数值 0 与缺失值混在一起。
2. **公开原始数据作为补充列。** 来源为 [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)。129,880 行中剔除与比赛任意一行共同特征完全重复的 21 行，保留 129,859 行。未共有的 `Inflight service`、原始 ID 和索引列不进入比赛模型。
3. **原始数据先验。** 仅用公开原始数据标签训练 XGBoost，预测比赛行；另外从公开数据计算平滑目标均值和计数。它们不读取比赛的验证标签。
4. **比赛数据目标编码。** XGBoost 的训练输入通过内部 5 折交叉拟合生成目标编码；外部验证折和测试集只映射。原始值、关键交互及连续变量分箱分别形成编码键。
5. **少量有证据的候选模型。** 分别比较 XGBoost、CatBoost、LightGBM，以及原始行追加、数字特征、数值转类别和额外交互。未采用所有特征机械堆叠。
6. **固定权重融合。** 当前配置保留 25% XGBoost，另有 75% CatBoost。权重仅在开发验证集上从预设网格选择，满足最初保留 XGBoost 的要求；并非声称这个组合超过所有单模型。

`optimized_recipe.json` 固定所选模型和权重。开发集最佳单个 CatBoost AUC 为
0.960282，所选含 XGBoost 的组合为 0.960270，二者仅差约 0.000012。
复杂融合没有稳定优势，因此未引入更多模型或高容量堆叠器。

## 验证设计

- 首先固定留出 20%（139,927 行），不用于本轮候选及权重选择。
- 剩余 80% 再划分为 447,766 行拟合、111,942 行开发验证，筛选候选方案。
- 方案固定后，在 139,927 行保留集评估；训练轮数提前由开发集最佳轮数乘 1.1 决定，不使用保留集标签早停，也不按保留集结果重新选型。
- 最后对全部真实训练数据运行分层 10 折，`shuffle=True, random_state=42`，与原版本使用相同的外部折。各折使用验证集早停，测试预测取 10 个折的平均。

保留集报告见 `optimization_audit.json`，候选和权重试验见 `optimization_search.json`。
10 折结果用于最终对比；由于超参数选择不是完整嵌套 10 折，不能把其分数看作完全独立的泛化估计。
本地 AUC 不代表 Kaggle 公榜或私榜，本次没有自动提交到 Kaggle。

## 文件作用和恢复

- `main.py`：日常运行入口及输出检查。
- `optimized_features.py`：原生类别、目标编码和原始数据统计。
- `optimized_training.py`：原始数据缓存、模型训练和 10 折融合。
- `optimize.py`：重放开发集模型选择及固定方案的保留集评估。
- `optimized_recipe.json`：本次选定配置，通常无需手工修改。
- `.artifact_runtime`：原始数据先验缓存、每折预测检查点和训练记录；输入或配置指纹变化时不会误用旧折结果。

运行中断后，重复相同 `main.py` 命令可恢复已完成的折与模型预测。
改变训练文件、标签、代码或配置后会使用新的检查点目录。

验证测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖 0 与缺失的区分、目标编码内部交叉拟合、未知类别、独立原始统计、
保留集标签不影响固定轮数模型的预测，以及 10 折权重汇总和检查点恢复。
