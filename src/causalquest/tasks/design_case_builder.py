"""Build Agent-visible T1 seed cases and isolated evaluator artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from causalquest.evaluation.design_review import build_design_review_rubric, rubric_to_dict
from causalquest.schemas import DataLayer, LifecycleStage, TaskSpec, TaskType

from .design_review import DesignDraft, assess_design


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_brief(case: dict[str, Any], draft: DesignDraft) -> str:
    return "\n".join(
        (
            f"# {case['title']}",
            "",
            f"业务背景：{case['business_context']}",
            "",
            f"假设：{draft.hypothesis or '未提供'}",
            "",
            f"预期决策动作：{draft.decision_action or '未提供'}",
            "",
            "请在占用实验流量前评审该方案；不要假定未登记的信息。",
            "",
        )
    )


def _render_metric_plan(draft: DesignDraft) -> str:
    lines = ["# 指标计划", ""]
    if not draft.metrics:
        return "\n".join((*lines, "未登记指标。", ""))
    lines.extend(
        (
            "| metric | role | unit | window | direction | threshold | aggregation |",
            "|---|---|---|---|---|---|---|",
        )
    )
    lines.extend(
        f"| {item.metric_id} | {item.role} | {item.unit} | "
        f"{item.attribution_window} | {item.direction} | "
        f"{item.decision_threshold or '未登记'} | "
        f"{item.aggregation_to_analysis_unit or '未登记'} |"
        for item in draft.metrics
    )
    lines.append("")
    return "\n".join(lines)


def build_design_case(
    raw_case: dict[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Build one seed case and return evaluator-side catalog metadata."""

    output = Path(output_root) / raw_case["case_id"]
    draft = DesignDraft.from_mapping(raw_case["design"])
    assessment = assess_design(draft)
    documents = {
        "experiment_brief": "experiment_brief.md",
        "metric_plan": "metric_plan.md",
        "traffic_forecast": "traffic_forecast.md",
        "decision_charter": "decision_charter.md",
        "method_playbook": "method_playbook.md",
    }
    brief = _render_brief(raw_case, draft)
    metric_plan = _render_metric_plan(draft)
    traffic_forecast = (
        "# 流量与排期预测\n\n"
        f"每日预计合格单位：{draft.daily_eligible_units or '未登记'}。\n\n"
        f"最长实验周期：{draft.max_duration_days or '未登记'} 天。\n"
    )
    decision_charter = (
        "# 决策合同\n\n"
        f"多指标策略：{draft.multiplicity_policy or '未登记'}。\n\n"
        "设计批准只代表允许占用实验流量；任何灰度或全量动作仍由业务、数据科学和工程负责人联合审批。\n"
    )
    playbook = (
        "# 实验设计评审方法\n\n"
        "依次检查假设与决策、随机化/曝光/分析单位、分析总体、指标角色、功效和干扰。\n"
        "不能用上线后的行为筛选 ITT 总体；高干扰场景应考虑 cluster 或 switchback。\n"
    )
    visible_payload = {
        "case_id": raw_case["case_id"],
        "design": draft.to_dict(),
        "documents": {
            "brief": brief,
            "metric_plan": metric_plan,
            "traffic_forecast": traffic_forecast,
            "decision_charter": decision_charter,
            "playbook": playbook,
        },
        "related_experiments": raw_case.get("related_experiments", []),
    }
    case_snapshot_id = hashlib.sha256(_canonical_json(visible_payload).encode("utf-8")).hexdigest()
    task_spec = TaskSpec(
        case_id=raw_case["case_id"],
        suite="design_review_seed_v1",
        task_type=TaskType.DESIGN_REVIEW,
        lifecycle_stage=LifecycleStage.DESIGN,
        scenario_family=draft.scenario_family,
        data_layer=DataLayer.AUTHORED_CONTEXT,
        randomization_unit=draft.randomization_unit,
        exposure_unit=draft.exposure_unit,
        analysis_unit=draft.analysis_unit,
        estimand=draft.estimand,
        visible_evidence=(
            "experiment_draft",
            "experiment_brief",
            "metric_plan",
            "traffic_forecast",
            "decision_charter",
            "method_playbook",
            "historical_experiment_registry",
        ),
        hidden_evidence=("design_oracle", "design_rubric"),
        allowed_tools=(
            "get_document",
            "get_experiment",
            "get_metric_definition",
            "estimate_power",
            "find_related_experiments",
            "update_investigation_state",
            "request_additional_evidence",
            "submit_review",
        ),
        max_tool_calls=16,
        decision_schema="design-v1",
        template_family_id=raw_case["template_family_id"],
        source_group_id=raw_case["source_group_id"],
    )
    manifest = {
        **task_spec.to_public_dict(),
        "case_snapshot_id": case_snapshot_id,
        "documents": documents,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.md").write_text(
        "请审查该实验设计，使用工具核对关键定义与功效，并提交带证据的结构化建议。\n",
        encoding="utf-8",
    )
    docs_dir = output / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / documents["experiment_brief"]).write_text(brief, encoding="utf-8")
    (docs_dir / documents["metric_plan"]).write_text(metric_plan, encoding="utf-8")
    (docs_dir / documents["traffic_forecast"]).write_text(traffic_forecast, encoding="utf-8")
    (docs_dir / documents["decision_charter"]).write_text(decision_charter, encoding="utf-8")
    (docs_dir / documents["method_playbook"]).write_text(playbook, encoding="utf-8")
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "experiment.json", draft.to_dict())
    _write_json(output / "related_experiments.json", raw_case.get("related_experiments", []))
    _write_json(output / "hidden" / "oracle.json", assessment.to_oracle_dict())
    _write_json(
        output / "hidden" / "rubric.json",
        rubric_to_dict(build_design_review_rubric()),
    )
    return {
        "case_id": raw_case["case_id"],
        "scenario_family": draft.scenario_family,
        "template_family_id": raw_case["template_family_id"],
        "source_group_id": raw_case["source_group_id"],
        "case_snapshot_id": case_snapshot_id,
        "expected_decision": assessment.decision.value,
        "required_issue_codes": [issue.code for issue in assessment.issues],
        "passed": True,
    }
