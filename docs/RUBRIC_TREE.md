# Rubric Tree：任务要求、评审依据与学习信号

Rubric Tree 将任务评价要求整理成带版本的结构。它让研究者看清一个任务检查哪些方面、各项如何分组，
以及哪些前提会阻断结论。CAUSALQUEST 已提供三类任务的树、结构校验和 JSON 导出。

## 三个组件分别做什么？

| 组件 | 输入与用途 | 当前实现 |
|---|---|---|
| 任务 Rubric Tree | 描述一类任务的标准、权重与硬门禁 | `RubricNode` / `RubricTree`；三类任务构建函数 |
| 历史确定性验证器 | 对报告、工具轨迹和评测侧 oracle 运行特定检查 | 各任务 `verify_*` 函数，自带检查权重与通过门槛 |
| 当前 RL 七维评审 | 根据模型已获得的公开证据，判断轨迹质量和组内次序 | 人工评审文件、协议校验、等级与 Pareto 比较 |

树的节点权重描述评价结构；历史验证器的检查项按各自实现聚合。当前代码没有把树递归求和作为通用
评分执行器，也没有将节点权重自动变成 RL 的 token 奖励。RL 实际消费的是七维评审结果。

## 节点包含什么？

```json
{
  "id": "statistics",
  "weight": 0.2,
  "description": "正确使用区间、检验和多重比较",
  "hard_gate": false,
  "children": []
}
```

- `id`：树内唯一标识，用于稳定引用。
- `weight`：相对于父节点的局部权重；同一父节点的子节点权重之和为 1。
- `description`：该项要求的含义。
- `hard_gate`：标记关键前提；具体如何阻断得分由使用它的评价规则落实。
- `children`：可继续拆分的子标准。

树包含独立 `version`，根节点权重为 1。结构校验检查节点 ID、说明、非负权重、同级权重和、全树 ID 唯一性等。
目前提供的三棵树均为根节点下的一层标准；类型支持更深层的结构。

## 三类任务的实际权重

| 标准 | 设计评审 | 健康诊断 | 实验后决策 |
|---|---:|---:|---:|
| 有效性 `validity` | 0.30 | 0.25 | 0.20 |
| 问题覆盖 `issue_coverage` | 0.20 | — | — |
| 功效 `power` | 0.15 | — | — |
| 异常定位 `localization` | — | 0.15 | — |
| 调查过程 `trajectory` | — | 0.15 | 0.10 |
| 估计对象 `estimand` | — | — | 0.10 |
| 统计判断 `statistics` | — | — | 0.20 |
| 业务意义 `practicality` | — | — | 0.15 |
| 证据 `evidence` | 0.15 | 0.20 | 0.15 |
| 决策 `decision` | 0.15 | 0.20 | 0.05 |
| 人工审批边界 `human_boundary` | 0.05 | 0.05 | 0.05 |

三棵树都将 `validity` 和 `human_boundary` 标记为硬门禁。
这些是项目当前的设计权重，尚未通过独立实验确立为最优权重。

原始树：[设计](../examples/rubrics/design_review.json)、
[健康](../examples/rubrics/health_diagnosis.json)、[实验后](../examples/rubrics/post_experiment.json)。

## 为什么适合因果场景？

实验后任务需要区分几个连续但不同的问题：

1. **数据能不能支持分析？** 检查来源、随机化风险及隔离条件。
2. **比较的究竟是什么？** 核对候选、对照、指标与总体。
3. **统计结果支持什么？** 检查差值、区间和多重比较。
4. **能否达到业务要求？** 将统计结果与已登记 MDE 对齐。
5. **建议与证据是否一致？** 检查选择、拒绝或重跑的理由，以及调查状态与后续行动。

例如，若区间整体高于 0，但尚未整体高于正的 MDE，模型应根据任务合同解释业务收益的不确定性。
比较工具可以准确计算区间与阈值的关系；选对对象、决定下一步仍由 Agent 完成。

这也解释了为何只统计最终枚举值或报告是否提交会遗漏错误：结论可能碰巧正确，但依据、状态或修订行动仍有问题。

## 如何读取和扩展？

```python
import json
from causalquest.evaluation.post_experiment import build_post_experiment_rubric, rubric_to_dict
from causalquest.schemas.rubric import RubricTree

tree = build_post_experiment_rubric()
payload = rubric_to_dict(tree)
print(json.dumps(payload, ensure_ascii=False, indent=2))
restored = RubricTree.from_mapping(payload)
assert restored == tree
```

新增任务时，先定义标准及其证据依据，再决定哪些可由程序核对、哪些需要语义评审。
节点只是评价合同；具体评分器需要明确实现并单独验证。修改权重或标准应更新版本，并保留原协议下的结果。

如果目标是局部信用分配，还需要定义动作对应的证据范围、局部结果和奖励归因方法。
一棵更细的树本身不会解决这一步。

## 参考答案与公开证据的边界

部分历史验证器使用 factory 生成的 oracle，设计树中的“问题覆盖”也沿用了这一表述。
这些文件服务于评测，不作为自主 Agent 的输入。
当前人工 RL 评审包仅包含公开请求、实际工具观测和输出，并要求证据出现在被评动作之前。

程序可验证的数值、项目编写的业务约束、专家示范与人工标签具有不同来源。
为新数据接入评测时，应核对来源和合同歧义；存在歧义的评审保留待裁定状态。
运行参考本身不能保证一个任务拥有唯一、普遍正确的因果决策答案。

相关实现：[`schemas/rubric.py`](../src/causalquest/schemas/rubric.py)、
[`evaluation/`](../src/causalquest/evaluation/)、[七维评审协议](REVIEW_PROTOCOL.md)。
