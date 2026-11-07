"""Build T2 health-diagnosis cases with isolated evaluator artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from causalquest.data.controlled_lab import ControlledExperimentData
from causalquest.data.experiment_db import build_controlled_lab_database
from causalquest.data.faults import FaultSpec, inject_fault, verify_fault_injection
from causalquest.data.health_checks import (
    analyze_slice_health,
    check_assignment_exposure,
    check_event_timeliness,
    check_metric_completeness,
    check_pre_experiment_balance,
    check_sample_ratio,
)
from causalquest.evaluation.health_diagnosis import (
    build_health_diagnosis_rubric,
    rubric_to_dict,
)
from causalquest.schemas import DataLayer, LifecycleStage, TaskSpec, TaskType

from .health_diagnosis import expected_health_assessment


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_incident_ticket(raw_case: dict[str, Any], data: ControlledExperimentData) -> str:
    return "\n".join(
        (
            f"# {raw_case['title']}",
            "",
            f"业务背景：{raw_case['business_context']}",
            "",
            f"实验：{data.config.experiment_id}",
            "",
            f"当前关注指标：{raw_case['monitored_metric']}",
            "",
            f"业务入口：{data.config.surface}；对照方案：{data.config.control_variant}；候选方案：{data.config.treatment_variant}",
            "",
            "请先验证实验健康度，再决定是否允许团队解释处理效应。不要假设告警即根因。",
            "",
        )
    )


def _render_runbook() -> str:
    return (
        "# 运行中实验调查规范\n\n"
        "先确认实验登记信息和指标口径，再检查分流、实验前属性平衡、分配—曝光链路、"
        "指标完整性与时间窗口。异常必须定位到可验证机制或切片；不得在健康门禁失败时"
        "直接解释 treatment effect。切片诊断不等于确认性异质效应分析。\n"
    )


def _diagnostic_snapshot(database_path: Path, metric_id: str) -> dict[str, Any]:
    import duckdb

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        return {
            "sample_ratio": check_sample_ratio(connection),
            "pre_experiment_balance": check_pre_experiment_balance(connection),
            "assignment_exposure": check_assignment_exposure(connection),
            "metric_completeness": check_metric_completeness(connection, metric_id),
            "event_timeliness": check_event_timeliness(connection, metric_id),
            "platform_slice": analyze_slice_health(connection, "platform", metric_id),
            "app_version_slice": analyze_slice_health(connection, "app_version", metric_id),
        }
    finally:
        connection.close()


def build_health_case(
    raw_case: dict[str, Any],
    clean_data: ControlledExperimentData,
    faults: dict[str, FaultSpec],
    output_root: str | Path,
) -> dict[str, Any]:
    """Build one Agent-visible warehouse and its evaluator-only expected diagnosis."""

    fault_id = raw_case.get("fault_id")
    fault_result = None
    if fault_id is None:
        data = clean_data
        fault_type = None
    else:
        if fault_id not in faults:
            raise ValueError(f"unknown fault_id: {fault_id}")
        fault_spec = faults[fault_id]
        if raw_case.get("fault_overrides"):
            fault_spec = FaultSpec.from_mapping(
                {**fault_spec.to_dict(), **raw_case["fault_overrides"]}
            )
        fault_result = inject_fault(clean_data, fault_spec)
        verification = verify_fault_injection(clean_data, fault_result)
        if not verification["passed"]:
            raise ValueError(f"fault verification failed for {fault_id}: {verification}")
        data = fault_result.data
        fault_type = fault_result.spec.fault_type

    output = Path(output_root) / raw_case["case_id"]
    output.mkdir(parents=True, exist_ok=True)
    database_name = "warehouse.duckdb"
    experiment_start = datetime.fromisoformat(
        data.config.start_time_utc.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    analysis_as_of_time = (
        experiment_start + timedelta(days=data.config.duration_days, hours=24)
    ).isoformat()
    database_report = build_controlled_lab_database(
        data,
        output / database_name,
        allow_controlled_faults=fault_result is not None,
        analysis_as_of_time=analysis_as_of_time,
    )
    documents = {
        "incident_ticket": "incident_ticket.md",
        "monitoring_runbook": "monitoring_runbook.md",
    }
    incident_ticket = _render_incident_ticket(raw_case, data)
    runbook = _render_runbook()
    visible_payload = {
        "case_id": raw_case["case_id"],
        "data_snapshot_id": data.snapshot_id,
        "analysis_as_of_time": analysis_as_of_time,
        "documents": {"incident_ticket": incident_ticket, "monitoring_runbook": runbook},
        "monitored_metric": raw_case["monitored_metric"],
    }
    case_snapshot_id = hashlib.sha256(_canonical_json(visible_payload).encode()).hexdigest()
    allowed_tools = (
        "get_document",
        "get_experiment",
        "get_metric_definition",
        "describe_relation",
        "profile_relation",
        "query_warehouse",
        "check_sample_ratio",
        "check_pre_experiment_balance",
        "check_assignment_exposure",
        "check_metric_completeness",
        "check_event_timeliness",
        "analyze_slice_health",
        "update_investigation_state",
        "request_additional_evidence",
        "submit_review",
    )
    task_spec = TaskSpec(
        case_id=raw_case["case_id"],
        suite="health_diagnosis_seed_v1",
        task_type=TaskType.HEALTH_DIAGNOSIS,
        lifecycle_stage=LifecycleStage.RUNNING,
        scenario_family=raw_case.get("scenario_family", "feed_experiment_health"),
        data_layer=DataLayer.CONTROLLED_LAB,
        randomization_unit=data.config.randomization_unit,
        exposure_unit="exposure",
        analysis_unit=data.config.analysis_unit,
        estimand="triggered_population_ate",
        visible_evidence=("incident_ticket", "monitoring_runbook", "experiment_warehouse"),
        hidden_evidence=("fault_oracle", "health_rubric"),
        allowed_tools=allowed_tools,
        max_tool_calls=14,
        decision_schema="health-v1",
        template_family_id=raw_case["template_family_id"],
        source_group_id=raw_case["source_group_id"],
    )
    manifest = {
        **task_spec.to_public_dict(),
        "case_snapshot_id": case_snapshot_id,
        "data_snapshot_id": data.snapshot_id,
        "analysis_as_of_time": analysis_as_of_time,
        "database": database_name,
        "documents": documents,
        "monitored_metric": raw_case["monitored_metric"],
        "available_relations": [
            "experiment_registry",
            "assignments",
            "exposures",
            "events_visible",
            "metric_catalog",
            "snapshot_metadata",
            "variant_metrics",
            "case_runtime",
        ],
    }
    docs = output / "docs"
    docs.mkdir(exist_ok=True)
    (docs / documents["incident_ticket"]).write_text(incident_ticket, encoding="utf-8")
    (docs / documents["monitoring_runbook"]).write_text(runbook, encoding="utf-8")
    (output / "prompt.md").write_text(
        "调查该运行中实验的健康状态，更新调查假设，并提交带证据的结构化建议。\n",
        encoding="utf-8",
    )
    _write_json(output / "manifest.json", manifest)
    assessment = expected_health_assessment(fault_type)
    oracle = assessment.to_oracle_dict()
    oracle.update(
        {
            "evaluator_only": True,
            "fault": fault_result.oracle_manifest() if fault_result else None,
            "diagnostic_snapshot": _diagnostic_snapshot(
                output / database_name, raw_case["monitored_metric"]
            ),
            "database_report": database_report,
        }
    )
    _write_json(output / "hidden" / "oracle.json", oracle)
    _write_json(
        output / "hidden" / "rubric.json",
        rubric_to_dict(build_health_diagnosis_rubric()),
    )
    return {
        "case_id": raw_case["case_id"],
        "template_family_id": raw_case["template_family_id"],
        "source_group_id": raw_case["source_group_id"],
        "case_snapshot_id": case_snapshot_id,
        "data_snapshot_id": data.snapshot_id,
        "expected_decision": assessment.decision.value,
        "required_issue_codes": [issue.code for issue in assessment.issues],
        "fault_type": fault_type.value if fault_type else None,
        "passed": True,
    }
