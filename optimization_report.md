# 航空满意度优化结果

已参考用户提供的第二名方案，将原始数据先验、分箱和折内目标编码迁移到航空满意度任务，
并完成真实数据的全部 10 折训练。当前固定配置是 25% XGBoost + 75% CatBoost。

| 方案 | 10 折平均 AUC |
| --- | ---: |
| 原始 XGBoost | 0.958703 |
| 上一轮特征工程与插值合成 | 0.958412 |
| 本次优化 | 0.961119 |

本次相对原始方案提高 0.002416，
10 折中 10 折改善；整体折外 AUC 为 0.961112。
尚未达到 0.965，差距为 0.003881。

另外，未参与本轮模型和权重选择的 139,927 行保留集上，固定方案 AUC 为
0.960901，同划分基线为 0.958414。
这些分数都是本地验证分数，本次没有向 Kaggle 提交，不能宣称榜单达到同样分数。

## 使用结果

正式预测文件是 submission_optimized.csv（完整归档内：`submission_optimized.csv`），
包含 299,844 行测试概率预测。原始数据、模板及原始预测均保留。

再次训练：

```powershell
.\.venv\Scripts\python.exe main.py --device cuda --n-jobs 8
```

输入、配置和代码一致时，程序会恢复完成的折预测检查点。

## 验证与限制

17 项测试通过，提交行数、列顺序、ID 和概率范围均验证通过。
每条真实训练数据恰有一个折外预测，最终概率已从 20 个折与模型检查点独立重建核对。
原训练集、测试集、提交模板、原预测及公开原始数据均通过 SHA-256 校验。
目标和 ID 未进入特征，比赛目标编码在外层训练部分内部交叉拟合。

所参考的 [第二名方案](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/2nd-place-solution)
属于手机成瘾预测，本项目的数据属于航空满意度预测；迁移的是方法，无法直接迁移其分数。
候选选择没有采用完整嵌套 10 折，因此最终 10 折分数与单独保留集结果应一起参考。

详细方法见 [OPTIMIZATION.md](OPTIMIZATION.md)，完整指标见
[submission_optimized_metrics.json](submission_optimized_metrics.json)，检查回执见
[optimization_verification.json](optimization_verification.json)。
