# 重排序后的评分融合

本仓库提供核心方法的参考实现。输入为已经取得两路分数的固定候选列表，代码不调用模型，不需要GPU。

保留的内容包括纯重排序、BM25破平局、平均名次、归一化凸组合、RRF、Logistic、普通与域均衡ListFuse、组内监督、同权重组内限制、oracle排序，以及基本评价与配对统计工具。

仓库不包含论文成绩、真实查询、评分缓存、模型文件、训练好的权重、论文文件或服务器环境配置。示例数据全部为人为构造，不代表实验结果。

## 开始使用

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
postfusion evaluate --input examples/synthetic.jsonl --method T
```

拟合与同权重限制的运行方式见英文README。合成示例只用于检查程序，不用于判断方法效果。真实评测必须分离源域训练和目标域测试。

## 与论文的关系

这是一份清理后的核心方法实现，不是完整实验复现包。具体输入、历史模型评分、原始响应、论文拟合参数和运行记录不在仓库里，因此不能仅凭本仓库复现论文所有表格。

方法说明写清了特征顺序、两种目标的权重、组内训练的具体定义、网格还原和评价分母。清理版采用稳定的log-softmax计算和明确的随机优先序，不声称与开发期间的脚本逐比特相同。

## 发布前检查

```sh
python tools/check_release.py
```

检查通过后仍应人工确认文件列表。代码许可证为MIT，不包括任何外部数据集或模型的授权。
