# XGBoost 预测与 10 折交叉验证

`main.py` 支持二分类和回归，每次固定使用 **10 折交叉验证**。
二分类使用分层划分，以 ROC-AUC 评估；回归使用普通随机划分，以 RMSE 评估。
自动模式把恰好有两个不同标签的目标视为二分类，其他目标视为回归；
如果两个数值实际上代表回归目标，请加 `--task regression`。
当前不支持多分类目标。

## 安装和试运行

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --demo
```

`--demo` 会在独立的 `demo_data` 目录创建 800 行模拟训练数据、200 行模拟测试数据及示例提交模板，随后实际运行 10 折训练。
**这些数据、分数和预测仅用于验证代码流程，不能用于比赛提交。**
原文件夹的 `sample_submission.csv` 会保留。

## 使用真实数据

将比赛的 `train.csv`、`test.csv` 放在本目录，与比赛对应的
`sample_submission.csv` 一起使用，然后运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

也可以指定外部路径、目标列和任务类型：

```powershell
.\.venv\Scripts\python.exe main.py --train "D:\data\train.csv" --test "D:\data\test.csv" --sample "D:\data\sample_submission.csv" --target satisfaction --task classification
```

CSV 需要有唯一且非空的 `id` 列，其他 ID 名称可用 `--id-column` 指定。
训练集比测试集多一个目标列，其他特征列需要一致。
没有模板时，会从训练集与测试集的列差异推断目标，并按测试集顺序输出。
二分类每个类别至少需要 10 行，输出标签编码后第二类的概率，运行日志和指标文件会记录类别顺序。

程序排除 ID 和目标列，支持数值、文本类别和缺失值；每折单独拟合类别编码，
数值缺失由 XGBoost 处理。每折最多训练 1500 轮，50 轮无提升时早停，
使用该折最佳模型预测测试集，最后平均 10 个模型的预测。
可用 `--n-estimators 500` 调整每折最大轮数。
随机划分适用于相互独立的样本；时间序列或同一主体的重复记录需要另行采用时间或分组划分。

## 输出

- `submission.csv`：测试集预测，按提交模板 ID 和列顺序排列。
- `submission_oof.csv`：每条训练数据在其验证折上的预测及折号。
- `submission_metrics.json`：10 个折的分数、均值、标准差、整体折外分数及最佳轮数。

示例运行的输出均在 `demo_data` 目录。`--output` 可以指定预测文件路径，
指标与折外预测会保存到同一目录。再次运行会更新同名输出文件。
