# 插画式框架图与 Rubric Tree 图

两张图采用白底、彩色细描边、人物与工具插图、轨迹小卡片、编号清单、粗箭头和括号分组。
图形语言参照本轮提供的参考图，内容按 CAUSALQUEST 的实际流程重新组织。中文文字与数学符号均可重新生成。

## 图 1：自主调查与策略优化

[![CAUSALQUEST 自主调查与策略优化](figures/framework_architecture.png)](figures/framework_architecture.svg)

专家示范经 SFT 初始化 Agent。模型选择工具、操作数、证据与行动，环境执行查询和计算并返回观测。
同一案例的完整轨迹组进入人工七维公开证据评审，再通过等级、语义 Pareto 比较与核心支持门控得到策略优势。
裁剪策略更新保留固定 SFT 参考约束，保存并独立重载后，用新策略重新采样。

- 蓝色表示示范与 SFT，绿色表示自主执行和轨迹，紫色表示评审与比较，红色表示参数更新。
- 实线表示数据或执行依赖；绿色虚线表示新策略进入下一轮采样。
- 人物插图表示专家示范和 Agent 策略，图中没有增加额外教师模型阶段。
- $a_t$ 为模型动作，$o_{t+1}$ 为工具观测，$\tau_i$ 为同一案例的一条完整轨迹。

## 图 2：任务评价结构与 RL 评分接口

[![任务评价结构与 RL 评分接口](figures/rubric_tree_and_reward.png)](figures/rubric_tree_and_reward.svg)

左侧按任务类型分为设计评审、健康诊断和实验后决策。卡片内的标准、版本、权重和硬门禁均读取
`examples/rubrics/*.json`。红色盾牌及 † 标记硬门禁；每棵树的同级权重之和为 1。

右侧以实际轨迹和公开观测作为独立输入，展示当前的人工七维评审、四等级、组内优势和策略更新接口。
任务树描述评价结构；当前七维评审单独定义排序，树权重不直接相加为 RL 奖励。
更多定义见 [Rubric Tree 说明](RUBRIC_TREE.md)。

## 文件与重新生成

| 图片 | README 展示 | 矢量原图 |
|---|---|---|
| 自主调查与策略优化 | [PNG](figures/framework_architecture.png) | [SVG](figures/framework_architecture.svg) |
| 任务评价结构与 RL 评分 | [PNG](figures/rubric_tree_and_reward.png) | [SVG](figures/rubric_tree_and_reward.svg) |

```bash
python -m pip install -e '.[figures]'
python scripts/draw_framework_figures.py
```

绘图入口为 `scripts/draw_framework_figures.py`，插画式布局与图标绘制在 `scripts/illustrated_diagrams.py`。
重新生成需要本地中文字体，例如 Noto Sans CJK SC 或 Microsoft YaHei；可用 `--font` 指定字体。
字体文件不随仓库分发。SVG 将文字转为矢量轮廓，图标也保留矢量路径；PNG 以 180 dpi 导出。
脚本检查文字是否超出画布，最终布局另经目视核对。

## 插图来源与许可

人物、剪贴板及天平采用 [Microsoft Fluent Emoji](https://github.com/microsoft/fluentui-emoji) 的 Flat 素材，
版权归 Microsoft Corporation，使用 [MIT 许可](figures/icons/LICENSE.txt)。
[素材记录](figures/icons/sources.json) 列出源文件和校验值。素材副本随图保存，重新生成无需联网。

图中对部分图标进行了配色适配，并省略教师图标在小尺寸下不必要的模糊效果。
其余线框图标、轨迹卡片、连接关系和排版由本项目绘制。完整图及素材复用时请保留上述许可与来源说明。
