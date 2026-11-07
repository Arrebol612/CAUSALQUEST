# Framework v0.1: design, execution plan and scope

## 2026-11-06：首版目标

本轮将既有研究实现整理为独立的开源候选，目录为 `CausalQuest_framework/`。原实验数据、权重、
评分和版本化脚本保留。框架的验收重点是可复现的训练与评测闭环，效果由具体配方及同协议对照衡量。

首版顺序：提取领域核心 → 统一模型/轨迹接口 → SFT和完整分组采样 → 人工评审导入 → 单次RL更新
→ 质量统计 → 测试与打包。当前首轮执行只使用本地CPU测试模型及合成工具环境；4B/A40验收属于后续
真实资源阶段，需使用原模型快照和适配后的同协议数据，不能以toy分数代替。

## Components

```text
src/causalquest/
  data/, tasks/, tools/, schemas/   reusable source/contract/statistical core
  evaluation/                      task Rubric Trees and legacy verifiers
  harness/                         public environment adapter contracts
  models/                          native tool parsing and public tool schema
  framework/
    config.py                      portable configuration and explicit paths
    backends.py                    shared generation/teacher-forced score boundary
    environments.py                separate toy and CausalQuest sessions
    objectives.py                  ordinal preferences, clipping, reference term
    workflows.py                   bounded SFT, sample and RL stages
    reviews.py                     frozen packets, manual labels and evaluation
    artifacts.py                   strict JSON, hashing and atomic writes
    cli.py                         explicit user entry points
```

The domain dependency closure is recorded in `core_provenance.json`. Reused numerical tools have one
sibling import converted to a package import. Historical scripts are not imported or executed through
AST extraction. The clean domain adapter calls those tools through the same shared call-count budget.
The imported domain snapshot does not expose the later experimental tool-cost accounting as a unified
cost model; record calls/tokens and do not equate them with every historical cost column.

## Model and optimization contract

HF generation and scoring share one local snapshot and tokenizer. Native generation tokens are kept,
including stop/truncated outputs. SFT renders the same chat template twice and requires the prompt to
be an exact prefix of the full next-assistant sequence. Only completion positions enter the loss.
Sequences above the context budget fail explicitly. Tool schemas are rendered in canonical JSON key
order for live requests, persisted requests and SFT rows. HF sampling defaults to temperature 1 and
accepts an explicit positive `sampling_temperature`. Top-p stays 1 and top-k 0 without repetition
penalties. It disables model-default fallback and records effective settings alongside native tokens.
Raw frozen-HF scores remain distinct from the actual tempered sampling probabilities.

Grad-enabled scoring uses decoder training mode with stochastic dropout disabled so that the actual
Transformers checkpoint path executes. No-grad scoring and generation use eval mode. Each gradient
forward checks for real checkpoint-function calls, and SFT step metrics record the cumulative count.
The first A40 attempt exposed an eval-mode checkpoint bypass before any optimizer update; the corrected
path passed local gradient-equivalence checks and the A40 run: one SFT update and fresh reload used
16194.9 MiB peak allocated VRAM. Corrected sampling passed native-token/config checks. The latest
temperature-0.6 six-draw cohort used 11445.3 MiB peak allocated VRAM and supplied one supported
policy-advantage group. Returned native records report one successful real-model RL update,
exact old/current cache agreement and fresh parameter reload at 24028.8 MiB peak allocated VRAM.
All 223 files in the update archive and the 91 native generation receipts were verified locally.

The old raw HF scores and actual generation-distribution scores are separate. The RL ratio is current
policy / frozen old policy. Detached old / behavior correction has upper cap 2 and no lower floor.
Advantages come from within-group ordinal preferences. An all-wrong group has zero direct policy
advantage. Each trajectory receives equal batch mass; target tokens are averaged within the full
trajectory, not separately averaged per response. Zero-advantage examples retain the fixed-reference
term and stay in the denominator.

All old scores are checked before any RL update. Runtime-source identity, model fingerprint, full group
population, review identity and train split must match. Unknown or missing scores stop the update.
The framework does not infer semantic correctness from accepted tools, format or hashes.

Checkpoints include model files, Adam state, Torch/CUDA RNG and clocks. Each saved model is freshly
loaded and its parameter identity checked. SFT continuation restores the cursor and complete dataset
identity. SFT-to-RL is an explicit fresh-Adam boundary; RL-to-RL restores its optimizer and fixed
reference identity. A partially written output is never an automatic resume point. RL writes update
intent, then saves the returned update before collecting post-update diagnostics.

## Data formats

Case catalog entries include:

```json
{"case_id":"tutorial_design","source_group":"tutorial_source","split":"train",
 "task_type":"experiment_design_review","case_dir":"cases/tutorial_design",
 "case_snapshot_id":"the public snapshot identity","suite_snapshot_id":"tutorial-v1"}
```

Use builder-produced identities rather than inventing snapshot strings for real cases. `case_dir`
resolves relative to the catalog. A source group cannot cross catalog splits. Three existing case
builders remain available through Python; `cq data build-design` is the first CLI builder.

Reviewed SFT JSONL rows contain `id`, `case_id`, `source_group`, `split: train`,
`review_status: accepted`, `protocol: cq-public-history-v1`, `messages`, and `tools`. The final message
is the new assistant action, including native tool-call name and arguments where applicable. Content
can be null when a native tool call supplies the target. Prior assistant messages remain context.
`review_status` is a provenance assertion from the reviewer, not semantic certification by the loader.
If a case catalog is supplied, row source/case identity must match its TRAIN portion. Independently
check all derivatives when creating a larger train/validation split.

Changed demonstration actions must be replayed to rebuild successor evidence. This first clean release
does not automatically convert the historical report-v2 datasets or manufacture expert actions from
successful tool calls. The demo's two toy TRAIN cases and generated target rows are only workflow fixtures.

## Review and quality metrics

The manual interface is fully file-based, described in `REVIEW_PROTOCOL.md`. A hosted reviewer is not
required. API-judge configuration and calibration can be added behind that interface later. Missing
semantic judgments remain pending. Completed model failures are retained in the cohort; infrastructure
failures leave the sampling stage incomplete and block review export/update.

Evaluation reports submission, supported core, complete success, hard violations, task breakdown,
source count and generated tokens. Source-level clustering is necessary for inferential comparisons.
Checkpoint fit and RL KL/clip metrics are separate JSON artifacts. Success counts under different
contracts, tools, panels or assisted starting states are not pooled into one curve.

## Data and evaluation boundaries

The sandbox is a controlled software environment with read-only database tools. It is not an OS-level
container for arbitrary untrusted Python. Call permissions, case isolation, public evidence and model
inputs are enforced at the application boundary. Execute it under normal process/filesystem isolation
when accepting third-party tools or datasets.

Public tutorial values are authored synthetic data. No archive rows, private test cases, user credentials,
original model weights or server paths are included. Upworthy is an external aggregate-data source with
its own license and documented validity limits; business contracts and MDEs are project-authored.
Programmatic statistics, business policies, expert demonstrations and semantic review labels have
different reference status. No reference here certifies a true population causal effect in all cases.

Private held-out cases must not be in a public training release. The framework samples only explicitly
configured catalogs and never opens the research workspace's locked-test directories during its demo.
Human reviewer packets contain public requests/observations, not hidden evaluator files. Peer reviewers
must distinguish ambiguous contracts from wrong model decisions and use pending/adjudication when needed.

## Acceptance and next research steps

2026-11-07：公共包测试、独立 wheel 安装闭环和真实 4B/A40 验收已完成。真实验收包含一次 SFT 更新、
最终六条轨迹的公开证据评审、一次 RL 更新及保存重载。223 个导出文件与 91 次原生生成记录均已核对。
完整数值见 [Validation](VALIDATION.md)，三类任务评价结构见 [Rubric Tree](RUBRIC_TREE.md)。

最终更新前核心支持 1/6，严格完整成功 0/6；尚未测量独立的更新后任务质量。
后续能力研究应建立同一输入、工具和报告协议下的基线对照，按来源分组统计，并保留全部模型失败。
不同协议的历史分数只作背景，不合并用于选模或宣称训练收益。

公开仓库为 `Arrebol612/CAUSALQUEST`；原研究库以私有 `CausalQuest-research` 保留。
发布仅包含白名单源码、示例、测试和文档。服务器运行配置、模型、原研究数据和私有评测留在各自原位置。
首版支持 JSON 指标和人工语义评审；TensorBoard 导出、vLLM 训练适配和自动裁判按后续研究需要接入。
