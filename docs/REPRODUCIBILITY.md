# 复现与恢复

## 完整文件恢复

安装GitHub CLI并登录后，在新目录下载Release资产：

```powershell
gh release download archive-20261023 --repo IKMXKT-love/kaggle-airline-satisfaction --dir .
elease-assets
python .
elease-assets
estore.py --assets .
elease-assets --destination C:\Restore\kaggle-airline-satisfaction
```

恢复程序拒绝非空目标，先校验每卷SHA-256，再逐文件核对与解压。目录与文件清单覆盖当前源文件夹；已清理的历史缓存不能凭空恢复。符号链接/目录联接只保留目标记录，避免自动访问另一台电脑的外部绝对路径。

## 环境

原项目使用Windows、Python虚拟环境和NVIDIA GPU。`requirements*.txt`是各历史阶段依赖，`environment/`保存本次实际安装的包版本。`.artifact_runtime/tabicl_runtime`与`tabicl_finetune_deps`含单独依赖；完整Release保留这些文件。

恢复旧`.venv`不保证可在不同系统路径直接运行。建议另建虚拟环境并依照所选实验的依赖文件安装；需要GPU版本的PyTorch和相容驱动。这里没有重新安装环境或重新训练所有模型，因此不承诺跨平台一键复现全部分支。

## 查看已有最高分文件

恢复后，最高分文件为 `submission_public_pool_20261020.csv`，SHA-256：
`d3029122f333be8c98844b8af4c7f2f2fae91d74487e95994a488df1684a7728`。

不要把运行 `main.py` 与复现最高分混同。该入口默认运行较早的XGB/CatBoost配方。最高分相关源码为 `public_pool_trial.py`、`fit_public_pool.py`、`verify_public_pool_trial.py`，协议及权重保存在 `.artifact_runtime/public_pool_trial_20261020/`。

## 历史实验与核验

历史脚本按当时输入、缓存、原生折和固定时间协议编写；部分协议的绝对deadline现已过期，直接重跑可能停止或覆盖旧输出。应在恢复副本中新建运行目录与新协议，保留旧结果，不能仅修改分数文件或冒充原实验继续。RTD权重不含优化器状态，不能保证中断后逐步完全一致续训。

本归档只做文件、归档、远端哈希与源码语法核验，没有重新跑交叉验证或提交Kaggle。实验本身的核验范围以各报告及对应verification文件为准。公开OOF继承上游早停、间接标签依赖与历史选型限制，不能作为端到端严格嵌套验证。
