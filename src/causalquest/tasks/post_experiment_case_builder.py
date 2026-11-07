"""Build isolated T3 cases from the normalized Upworthy archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from causalquest.evaluation.post_experiment import build_post_experiment_rubric, rubric_to_dict
from causalquest.schemas import DataLayer, LifecycleStage, TaskSpec, TaskType

from .post_experiment import PostExperimentPolicy, assess_post_experiment


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _render_brief(raw_case: dict[str, Any], policy: PostExperimentPolicy) -> str:
    return "\n".join(
        (
            f"# {raw_case['title']}",
            "",
            f"业务背景：{raw_case['business_context']}",
            "",
            f"反事实决策合同指定参考臂：`{policy.reference_arm_id}`",
            "",
            f"反事实决策合同指定候选臂：`{policy.candidate_arm_id}`",
            "",
            f"主指标：聚合点击率；最小业务收益（绝对值）：{policy.mde_absolute:.6f}",
            "",
            f"显著性水平：{policy.alpha:.3f}；目标功效：{policy.target_power:.2f}",
            "",
            (
                "注意：实验已经结束，臂选择与 MDE 是本项目为可验证训练任务后验撰写的"
                "决策合同，不是源实验的预注册信息。请判断是否选择该内容 package、进入"
                "有限验证、按新协议重跑、迭代或拒绝；不得外推为产品级全量上线。"
                "最终动作由人类审批。"
            ),
            "",
        )
    )


def _render_metric_definition() -> str:
    return (
        "# 指标定义\n\n"
        "CTR = clicks / impressions。每行是聚合 package 实验臂，不是用户事件。"
        "效应定义为候选臂 CTR 减参考臂 CTR，只允许在同一 clickability test 内比较。\n"
    )


def _render_playbook() -> str:
    return (
        "# 实验后决策规范\n\n"
        "先核验实验来源、风险和隔离状态，再读取全部实验臂。用 Wilson 区间与 Newcombe "
        "差值区间表达不确定性；多臂实验以合同声明参考臂形成比较族，p 值使用 Holm 校正，"
        "差值区间使用 Bonferroni 同时置信水平。"
        "统计显著不等于达到 MDE；来源 significance、first_place 和 winner 均不是真值。"
        "风险期实验先触发有效性硬门禁，不再计算效果。历史实验不能继续收流量；证据不足时"
        "只能提出带新登记协议的重跑建议。任何内容选择都需人工批准，且不代表产品级上线。\n"
    )


def _rows_to_dicts(cursor) -> list[dict[str, Any]]:
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def build_post_experiment_case(
    raw_case: dict[str, Any],
    source_database: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize one non-locked experiment into a case-local read-only warehouse."""

    import duckdb

    source_database = Path(source_database)
    output = Path(output_root) / raw_case["case_id"]
    policy = PostExperimentPolicy.from_mapping(raw_case["decision_policy"])
    source = duckdb.connect(str(source_database), read_only=True)
    try:
        metadata = source.execute(
            "SELECT schema_version, snapshot_id FROM snapshot_metadata"
        ).fetchone()
        context_rows = _rows_to_dicts(
            source.execute(
                """
                SELECT source_partition, test_id, component_id, assigned_pool,
                       randomization_risk, quarantine_reasons, schema_eligible,
                       eligible_for_positive_causal_case
                FROM upworthy_tests WHERE test_id = ?
                """,
                [raw_case["test_id"]],
            )
        )
        if len(context_rows) != 1:
            raise ValueError(f"expected one source test: {raw_case['test_id']}")
        context = context_rows[0]
        if context["assigned_pool"] == "locked_private_test":
            raise PermissionError("locked private tests cannot be used for case authoring")
        arms = _rows_to_dicts(
            source.execute(
                """
                SELECT arm_id, source_row_index, headline, slug, eyecatcher_id,
                       impressions, clicks, ctr, source_significance_undocumented,
                       editor_first_place, editor_winner
                FROM upworthy_arms
                WHERE source_partition = ? AND test_id = ?
                ORDER BY arm_id
                """,
                [context["source_partition"], context["test_id"]],
            )
        )
    finally:
        source.close()

    if len(arms) < 2:
        raise ValueError("a T3 comparison case requires at least two source arms")
    assessment = assess_post_experiment(
        arms,
        policy,
        randomization_risk=bool(context["randomization_risk"]),
        quarantine_reasons=context["quarantine_reasons"],
    )
    diagnostic_only = raw_case.get("diagnostic_only", False) is True
    if (context["randomization_risk"] or context["quarantine_reasons"]) and not diagnostic_only:
        raise ValueError("risky or quarantined source tests must be declared diagnostic_only")

    documents = {
        "decision_brief": "decision_brief.md",
        "metric_definition": "metric_definition.md",
        "method_playbook": "method_playbook.md",
    }
    brief = _render_brief(raw_case, policy)
    metric_definition = _render_metric_definition()
    playbook = _render_playbook()
    visible_payload = {
        "case_id": raw_case["case_id"],
        "source_snapshot_id": metadata[1],
        "test_id": context["test_id"],
        "context": context,
        "arms": arms,
        "decision_policy": policy.to_dict(),
        "documents": {
            "brief": brief,
            "metric_definition": metric_definition,
            "playbook": playbook,
        },
    }
    case_snapshot_id = hashlib.sha256(_canonical_json(visible_payload).encode()).hexdigest()
    allowed_tools = (
        "get_document",
        "get_experiment",
        "get_metric_definition",
        "describe_relation",
        "query_warehouse",
        "get_arm_results",
        "estimate_ab_effect",
        "adjust_multiple_testing",
        "update_investigation_state",
        "request_additional_evidence",
        "submit_review",
    )
    task_spec = TaskSpec(
        case_id=raw_case["case_id"],
        suite="post_experiment_seed_v1",
        task_type=TaskType.POST_EXPERIMENT_DECISION,
        lifecycle_stage=LifecycleStage.COMPLETED,
        scenario_family=raw_case["scenario_family"],
        data_layer=DataLayer.MIXED_EVIDENCE,
        randomization_unit="viewer_assignment_not_individually_observed",
        exposure_unit="aggregate_impression",
        analysis_unit="aggregate_experiment_arm",
        estimand="within_test_click_probability_difference",
        visible_evidence=(
            "decision_brief",
            "metric_definition",
            "method_playbook",
            "case_warehouse",
        ),
        hidden_evidence=("post_experiment_oracle", "post_experiment_rubric"),
        allowed_tools=allowed_tools,
        max_tool_calls=14,
        decision_schema="post-decision-v1",
        template_family_id=raw_case["template_family_id"],
        source_group_id=str(context["component_id"]),
    )
    manifest = {
        **task_spec.to_public_dict(),
        "case_snapshot_id": case_snapshot_id,
        "source_snapshot_id": metadata[1],
        "database": "warehouse.duckdb",
        "documents": documents,
        "primary_metric": "aggregate_click_through_rate",
        "decision_policy": policy.to_dict(),
        "action_boundary": {
            "permitted": "select_or_reject_one_editorial_package_for_the_stated_CTR_objective",
            "prohibited": [
                "product_wide_rollout_claim",
                "retention_or_revenue_claim",
                "automatic_execution_without_human_approval",
            ],
            "required_human_checks": [
                "editorial_safety_and_factual_accuracy",
                "brand_and_policy_compliance",
                "new_registered_quality_guardrails_before_product_rollout",
            ],
        },
        "evidence_provenance": {
            "arm_results": "real_upworthy_aggregate_archive",
            "randomization_risk": "derived_from_published_archive_date_warning",
            "decision_policy": "authored_counterfactual_decision_contract_not_source_preregistration",
            "chinese_documents": "authored_benchmark_context",
            "statistical_outputs": "deterministic_recomputation",
        },
        "available_relations": [
            "experiment_context",
            "arm_results",
            "metric_catalog",
            "snapshot_metadata",
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "warehouse.duckdb.tmp"
    if temporary.exists():
        temporary.unlink()
    warehouse = duckdb.connect(str(temporary))
    try:
        warehouse.execute(
            """
            CREATE TABLE experiment_context (
                source_partition VARCHAR, test_id VARCHAR, component_id VARCHAR,
                assigned_pool VARCHAR, randomization_risk BOOLEAN,
                randomization_risk_source VARCHAR, quarantine_reasons VARCHAR,
                reference_arm_id VARCHAR, candidate_arm_id VARCHAR,
                mde_absolute DOUBLE, alpha DOUBLE, target_power DOUBLE,
                policy_origin VARCHAR, action_scope VARCHAR
            )
            """
        )
        warehouse.execute(
            "INSERT INTO experiment_context VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                context["source_partition"],
                context["test_id"],
                context["component_id"],
                context["assigned_pool"],
                context["randomization_risk"],
                "published_2024_date_window_derived",
                context["quarantine_reasons"],
                policy.reference_arm_id,
                policy.candidate_arm_id,
                policy.mde_absolute,
                policy.alpha,
                policy.target_power,
                policy.policy_origin,
                policy.action_scope,
            ],
        )
        warehouse.execute(
            """
            CREATE TABLE arm_results (
                arm_id VARCHAR PRIMARY KEY, source_row_index BIGINT, headline VARCHAR,
                slug VARCHAR, eyecatcher_id VARCHAR, impressions BIGINT, clicks BIGINT,
                ctr DOUBLE, source_significance_undocumented VARCHAR,
                editor_first_place BOOLEAN, editor_winner BOOLEAN
            )
            """
        )
        warehouse.executemany(
            "INSERT INTO arm_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    arm["arm_id"],
                    arm["source_row_index"],
                    arm["headline"],
                    arm["slug"],
                    arm["eyecatcher_id"],
                    arm["impressions"],
                    arm["clicks"],
                    arm["ctr"],
                    arm["source_significance_undocumented"],
                    arm["editor_first_place"],
                    arm["editor_winner"],
                )
                for arm in arms
            ],
        )
        warehouse.execute(
            """
            CREATE TABLE metric_catalog AS SELECT
                'aggregate_click_through_rate'::VARCHAR AS metric_id,
                'primary'::VARCHAR AS metric_role,
                'clicks / impressions within one test'::VARCHAR AS definition,
                'increase'::VARCHAR AS direction,
                'aggregate_experiment_arm'::VARCHAR AS unit
            """
        )
        warehouse.execute(
            """
            CREATE TABLE snapshot_metadata AS SELECT
                ?::VARCHAR AS schema_version, ?::VARCHAR AS case_snapshot_id,
                ?::VARCHAR AS source_schema_version, ?::VARCHAR AS source_snapshot_id
            """,
            ["post-experiment-case-v1", case_snapshot_id, metadata[0], metadata[1]],
        )
        warehouse.execute("CHECKPOINT")
    finally:
        warehouse.close()
    temporary.replace(output / "warehouse.duckdb")

    (output / "prompt.md").write_text(
        "调查该已完成实验，核验有效性、效应、不确定性与多重比较后提交结构化决策建议。\n",
        encoding="utf-8",
    )
    docs = output / "docs"
    docs.mkdir(exist_ok=True)
    (docs / documents["decision_brief"]).write_text(brief, encoding="utf-8")
    (docs / documents["metric_definition"]).write_text(metric_definition, encoding="utf-8")
    (docs / documents["method_playbook"]).write_text(playbook, encoding="utf-8")
    _write_json(output / "manifest.json", manifest)
    oracle = assessment.to_oracle_dict()
    oracle.update(
        {
            "evaluator_only": True,
            "source_snapshot_id": metadata[1],
            "test_id": context["test_id"],
            "action_boundary": manifest["action_boundary"],
            "evidence_provenance": manifest["evidence_provenance"],
        }
    )
    _write_json(output / "hidden" / "oracle.json", oracle)
    _write_json(
        output / "hidden" / "rubric.json",
        rubric_to_dict(build_post_experiment_rubric()),
    )
    return {
        "case_id": raw_case["case_id"],
        "test_id": context["test_id"],
        "source_group_id": context["component_id"],
        "template_family_id": raw_case["template_family_id"],
        "assigned_pool": context["assigned_pool"],
        "case_snapshot_id": case_snapshot_id,
        "expected_decision": assessment.decision.value,
        "required_issue_codes": [issue.code for issue in assessment.issues],
        "passed": True,
    }
