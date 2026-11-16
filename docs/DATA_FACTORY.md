# Data Factory：从来源数据到 Agent 的调查环境

数据工程负责准备可执行的证据空间。任务构造器将来源数据与业务合同组合成案例，模型再通过工具获取信息、
更新调查状态并提交报告。示范回放、SFT 数据准备与 RL 自主采样围绕相同环境展开。

## 来源、上下文与参考分开管理

| 层级 | 内容 | 使用边界 |
|---|---|---|
| `real_archive` | 公开真实实验的聚合臂记录 | 保持原始实验单位，解释范围受来源设计约束 |
| `authored_context` | 项目编写的 Brief、指标合同、MDE、候选比较与审批边界 | 明确标注人工上下文，不冒充源实验文档或预注册 |
| `controlled_lab` | 配置和随机种子生成的分配、曝光、事件 | 用于可复现的受控调查任务 |
| `controlled_fault` | 对日志副本施加的已知故障及干预记录 | 用于故障诊断和工具检查 |
| 评测侧参考 | 统计结果、规则检查项和任务 Rubric | 与 Agent 公开输入分离 |

真实档案的统计量、模拟器参数、项目决策政策和人工语义标签具有不同含义。
例如，生成器给定的效果参数不等于有限样本中的实测效果；项目后验编写的 MDE 也不等于源实验的预注册门槛。

## 真实档案适配：Upworthy

[`data/upworthy.py`](../src/causalquest/data/upworthy.py) 从原研究实现直接迁入，提供 Python API：

| API | 职责 |
|---|---|
| `download_upworthy_snapshot` | 显式许可确认后下载指定快照，核对大小与哈希，保存来源回执 |
| `audit_upworthy_snapshot` | 流式检查字段、实验单位、计数、重复、缺失、风险与分区交叉 |
| `build_upworthy_database` | 在审计通过后构建规范化 DuckDB、隔离记录、来源表及内容组件 |

下载需要使用者提供包含文件身份的来源配置，并明确确认相应许可；导入模块不会发起网络请求。
原始数据保存在调用方指定根目录的 `dataset/raw/`，规范化数据库位于 `dataset/processed/`。
真实档案不随代码发布。

规范化产物包括：

- `upworthy_arms`：保留实验臂、原始计数和来源行身份。
- `upworthy_tests`：汇总实验、风险、隔离原因和候选数据池。
- `quarantine_tests`：缺失关键文本或有效实验臂不足的隔离记录。
- `content_components`：关联同源内容的实验组件。
- `analysis_arms`：带组件与数据池信息的分析视图；消费者仍需检查适用性和隔离条件。
- `source_files`、`snapshot_metadata`：来源文件与快照身份。

### 先在合成小样本上运行整条路径

```bash
python scripts/prepare_archive_tutorial.py --output local_data/archive_tutorial
```

该命令生成四行自行编写的 CSV，运行实际审计与建库 API，并输出配置、审计和建库摘要。
所有输入均为教程数据，不下载真实档案。

```python
import json
from pathlib import Path
from causalquest.data.upworthy import audit_upworthy_snapshot

root = Path('local_data/archive_tutorial')
config = json.loads((root / 'source_config.json').read_text())
audit = audit_upworthy_snapshot(config, root)
print(audit['checks'])
```

真实数据需要重新准备来源配置和文件，不能沿用教程的文件身份或行数。

## 按内容组件和来源组隔离

仅按数据行切分可能让同一实验、同一内容或其派生案例同时进入训练和评测。
Upworthy 适配器利用规范化标题、slug 和图像标识建立内容连通组件。
通用 `share_text` 不作为强连接键；含 holdout 来源的关联组件进入锁定数据池。

[`data/splits.py`](../src/causalquest/data/splits.py) 提供独立的稳定组级划分：

```python
from causalquest.data.splits import assign_group

source_group = 'my-dataset|content-component-001'
research_split = assign_group(source_group, namespace='my-project-v1')
runtime_split = {'train': 'train', 'dev': 'validation',
                 'public_test': 'test', 'locked_private_test': 'test'}[research_split]
```

函数返回研究划分名；框架案例目录使用 `train/validation/test`。调用方需要保存这层映射与访问权限，
并决定哪些测试材料可以发布。将锁定集映射到运行时 `test` 不意味着可以将它用于训练、调参或公开分发。
原适配器的候选数据池同样需要经过案例选择和划分，才成为正式训练目录。

## 受控日志与故障

[`controlled_lab.py`](../src/causalquest/data/controlled_lab.py) 定义实验参数和生成机制，
[`experiment_db.py`](../src/causalquest/data/experiment_db.py) 将事件写入工具使用的数据库。
[`faults.py`](../src/causalquest/data/faults.py) 在副本上注入并检查 SRM、缺失、重复、污染、延迟等异常。
基础日志与故障清单保留独立身份，便于核对一次诊断对应的实际观测。

这些能力已经被健康任务构造器复用。构造受控任务时，应同步检查任务的事件、变体、指标和分析单位，
并在来源标签中保留其合成属性。

## 任务构造器连接环境

三类入口位于 [`tasks/`](../src/causalquest/tasks/)：

- `build_design_case` 将类型化设计与业务上下文组织为设计评审文档。
- `build_health_case` 将受控实验与故障配置组织为健康诊断环境。
- `build_post_experiment_case` 从规范化数据库提取实验，组合项目编写的候选比较政策和决策合同。

实验后构造器以只读方式访问来源数据库，并拒绝使用锁定私有测试来源来编写案例。
各案例具有本地公开材料、工具数据与独立的评测参考。公开教程可通过以下命令体验三类环境：

```bash
python scripts/prepare_domain_tutorial.py --output local_data/domain_tutorial
```

## 从环境到后训练数据

专家动作必须实际回放，后续上下文引用回放得到的工具观测；修订动作后，后继观测也需要重新构造。
SFT 保存“当前可见上下文 → 下一条 assistant 动作”，历史和工具消息不参与该步目标损失。
RL 则从任务初始状态采样完整轨迹组，再通过公开证据评审形成组内比较。

训练流程不会仅凭工具返回成功、结构完整或文件身份相同就判定语义正确。
评审检查结论、证据与行动的关系，完整规则见 [评审协议](REVIEW_PROTOCOL.md)。

## 迁入范围

本版迁入原研究的 Upworthy 获取、审计、规范化和组级切分模块及其原有测试；
Controlled Lab、故障注入、三类任务构造器和领域工具沿用当前框架已有实现。
Open Bandit 的 OPE 路径及原研究专属数据生产脚本未作为本版接口发布。
