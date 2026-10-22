# XGBoost、特征工程、合成增强与 10 折交叉验证

`main.py` 支持二分类和回归，每次固定使用 **10 折交叉验证**。
二分类使用分层划分，以 ROC-AUC 评估；回归使用普通随机划分，以 RMSE 评估。
自动模式把恰好有两个不同标签的目标视为二分类，其他目标视为回归；
如果两个数值实际上代表回归目标，请加 `--task regression`。
当前不支持多分类目标。

## 新加入的特征工程

`feature_engineering.py` 针对当前航空乘客满意度数据创建特征：

- 服务、线上服务、客舱服务、地面服务的平均分、最低分、最高分、标准差、高低分数量及零分数量；0 分与低评分分别记录。
- 到达与出发延误的差值、总量、最大值、是否延误、对数变换，以及单位航程延误。
- 年龄段、儿童及年长乘客标志、航程分段和航程对数。
- 客户类型、出行目的、舱位之间的类别组合。
- 原始及派生类别的频次编码，未知类别映射为 0。

逐行特征不读取其他样本。频次编码仅从每折**真实训练部分**学习，验证集、测试集及合成行均不参与统计；类别编码也只在当前训练部分拟合。
原始 ID 和目标列不会作为特征。

## 合成数据增强

默认每折增加相当于真实训练行数 **15%** 的合成行，单行训练权重为真实样本的 **0.25**。
在类别上下文相同、年龄段与航程段相同的训练样本中选择两个不同父样本，
二分类还要求父样本标签相同。仅对年龄、航程、延误等连续数值做小幅插值，
原始文本类别和离散服务评分保留锚点取值；整数型变量取整，缺失值不凭空填充。
派生特征随后从合成后的原始变量重新计算，确保延误总量等关系一致。

验证集和测试集不做合成。每折单独生成合成行，不会先对全量数据增强再切分。
指标文件记录各折真实、合成及验证行数。回归任务使用相同系数插值数值与目标，
仅适用于局部插值有意义的数据。没有可用连续变量或父样本对时跳过增强。
合成数据不会增加独立观测信息，是否改善模型需要以真实验证集评分判断。

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

本机已验证 NVIDIA 显卡可用于训练，可加速运行：

```powershell
.\.venv\Scripts\python.exe main.py --device cuda --n-jobs 8
```

做同样 10 折划分的对照实验：

```powershell
# 仅特征工程，关闭合成增强
.\.venv\Scripts\python.exe main.py --synthetic-ratio 0 --output submission_features_only.csv
# 关闭两项新增功能，保留原始特征的 XGBoost 流程
.\.venv\Scripts\python.exe main.py --no-feature-engineering --synthetic-ratio 0 --output submission_control.csv
# 调整合成比例和权重
.\.venv\Scripts\python.exe main.py --synthetic-ratio 0.1 --synthetic-weight 0.2
```

## 输出

- `submission_enhanced.csv`：测试集预测，按提交模板 ID 和列顺序排列。
- `submission_enhanced_oof.csv`：每条真实训练数据在其验证折上的预测及折号，不含合成行。
- `submission_enhanced_metrics.json`：10 个折的分数、均值、标准差、整体折外分数、最佳轮数及增强配置。

原有 `submission.csv` 及其指标和折外预测保留，可用于比较。
如果目录中有 `synthetic_preview.csv`，它是第一折的合成行预览，附带两个父样本的训练 ID，
仅供检查，程序不会将其作为额外训练文件读取。

示例运行的输出均在 `demo_data` 目录。`--output` 可以指定预测文件路径，
指标与折外预测会保存到同一目录。再次运行会更新同名输出文件。

运行验证测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 本次真实数据验证（2026-10-17）

已用 699,635 条训练数据、299,844 条测试数据完成全部 10 折，运行命令为：

```powershell
.\.venv\Scripts\python.exe main.py --device cuda --n-jobs 8
```

| 方案 | 10 折平均 AUC | 整体折外 AUC |
| --- | --- | --- |
| 已有原始特征模型 | 0.958703 | 0.958695 |
| 特征工程 + 15% 合成行、0.25 权重 | 0.958412 | 0.958406 |

本次新增方案未提升，平均 AUC 下降约 0.000291，建议目前仍使用原有
`submission.csv`。新增功能和独立输出均保留，便于后续对照调整。
已确认两个结果使用完全相同的验证折；原结果没有记录训练设备，新增结果使用 CUDA，
该比较衡量整套改动，不能单独归因于某一特征或合成数据。

9 项验证测试通过；新提交的行数、列名、ID 顺序及概率范围均验证通过。
原训练集、测试集、模板和原预测文件的 SHA-256 校验值保持一致。
`enhancement_comparison.json` 保存分数对比与检查结果，
`synthetic_preview.csv` 保存 1,000 条第一折合成样本，
`synthetic_preview_metadata.json` 记录其用途与配置。
