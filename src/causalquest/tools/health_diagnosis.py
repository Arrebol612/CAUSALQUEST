"""Budgeted Agent-facing tools for running-experiment health investigations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causalquest.data.health_checks import (
    analyze_slice_health,
    check_assignment_exposure,
    check_event_timeliness,
    check_metric_completeness,
    check_pre_experiment_balance,
    check_sample_ratio,
)
from causalquest.schemas import EvidenceRecord, ReviewReport, TaskType

from .case_environment import ToolBudgetExceeded
from .readonly_duckdb import ReadOnlyDuckDB

TOOL_VERSION = "health-diagnosis-tools-v1"
CONTROLLED_RELATIONS = {
    "experiment_registry",
    "assignments",
    "exposures",
    "events_visible",
    "metric_catalog",
    "snapshot_metadata",
    "variant_metrics",
    "case_runtime",
}
INVESTIGATION_STATUSES = {"open", "supported", "rejected", "blocked"}


class HealthDiagnosisEnvironment:
    """Expose read-only evidence and trusted checks without revealing fault oracles."""

    def __init__(self, case_dir: str | Path) -> None:
        self.case_dir = Path(case_dir)
        self.manifest = json.loads((self.case_dir / "manifest.json").read_text(encoding="utf-8"))
        self.database = ReadOnlyDuckDB(
            self.case_dir / self.manifest["database"],
            allowed_relations=CONTROLLED_RELATIONS,
        )
        self.allowed_tools = set(self.manifest["allowed_tools"])
        self.max_tool_calls = int(self.manifest["max_tool_calls"])
        self.trace: list[dict[str, Any]] = []
        self.investigation_state: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self.database.close()

    def _before_call(self, tool_name: str) -> None:
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"tool is not allowed for this case: {tool_name}")
        if len(self.trace) >= self.max_tool_calls:
            raise ToolBudgetExceeded(
                f"tool budget exhausted: {len(self.trace)}/{self.max_tool_calls}"
            )

    def _record(self, tool_name: str, arguments: dict[str, Any], result: Any) -> dict[str, Any]:
        evidence = EvidenceRecord.build(
            source_type="health_diagnosis_tool",
            source_ref=f"{self.manifest['case_id']}:{tool_name}",
            source_version=TOOL_VERSION,
            arguments=arguments,
            result=result,
            data_snapshot_id=self.manifest["case_snapshot_id"],
            tool_version=TOOL_VERSION,
        )
        record = {
            "step": len(self.trace) + 1,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "evidence_id": evidence.evidence_id,
            "result_sha256": evidence.result_sha256,
        }
        self.trace.append(record)
        return {"evidence_id": evidence.evidence_id, "result": result}

    def get_document(self, document_id: str) -> dict[str, Any]:
        self._before_call("get_document")
        documents = self.manifest.get("documents", {})
        if document_id not in documents:
            raise ValueError(f"unknown public document: {document_id}")
        root = (self.case_dir / "docs").resolve()
        path = (root / documents[document_id]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PermissionError("document path escapes the public docs directory") from exc
        return self._record(
            "get_document",
            {"document_id": document_id},
            {"document_id": document_id, "text": path.read_text(encoding="utf-8")},
        )

    def get_experiment(self) -> dict[str, Any]:
        self._before_call("get_experiment")
        result = self.database.query(
            """
            SELECT experiment_id, template_id, randomization_unit, analysis_unit,
                   start_time_utc, duration_days, control_variant, treatment_variant,
                   treatment_share, analysis_as_of_time
            FROM experiment_registry
            CROSS JOIN case_runtime
            """
        )
        return self._record("get_experiment", {}, result)

    def get_metric_definition(self, metric_id: str) -> dict[str, Any]:
        self._before_call("get_metric_definition")
        result = self.database.query(
            "SELECT * FROM metric_catalog WHERE metric_id = ?", [metric_id]
        )
        if result["row_count"] == 0:
            raise ValueError(f"unknown metric: {metric_id}")
        return self._record("get_metric_definition", {"metric_id": metric_id}, result)

    def describe_relation(self, relation: str) -> dict[str, Any]:
        self._before_call("describe_relation")
        return self._record(
            "describe_relation", {"relation": relation}, self.database.describe_table(relation)
        )

    def profile_relation(self, relation: str) -> dict[str, Any]:
        self._before_call("profile_relation")
        if relation not in CONTROLLED_RELATIONS:
            raise ValueError(f"relation is not allowlisted: {relation}")
        result = self.database.query(f"SELECT COUNT(*) AS row_count FROM {relation}")
        return self._record("profile_relation", {"relation": relation}, result)

    def query_warehouse(self, sql: str, parameters: list[Any] | None = None) -> dict[str, Any]:
        self._before_call("query_warehouse")
        result = self.database.query(sql, parameters)
        return self._record("query_warehouse", {"sql": sql, "parameters": parameters or []}, result)

    def check_sample_ratio(self) -> dict[str, Any]:
        self._before_call("check_sample_ratio")
        return self._record("check_sample_ratio", {}, check_sample_ratio(self.database.connection))

    def check_pre_experiment_balance(self) -> dict[str, Any]:
        self._before_call("check_pre_experiment_balance")
        return self._record(
            "check_pre_experiment_balance",
            {},
            check_pre_experiment_balance(self.database.connection),
        )

    def check_assignment_exposure(self) -> dict[str, Any]:
        self._before_call("check_assignment_exposure")
        return self._record(
            "check_assignment_exposure", {}, check_assignment_exposure(self.database.connection)
        )

    def check_metric_completeness(self, metric_id: str) -> dict[str, Any]:
        self._before_call("check_metric_completeness")
        return self._record(
            "check_metric_completeness",
            {"metric_id": metric_id},
            check_metric_completeness(self.database.connection, metric_id),
        )

    def check_event_timeliness(self, metric_id: str) -> dict[str, Any]:
        self._before_call("check_event_timeliness")
        return self._record(
            "check_event_timeliness",
            {"metric_id": metric_id},
            check_event_timeliness(self.database.connection, metric_id),
        )

    def analyze_slice_health(self, dimension: str, metric_id: str) -> dict[str, Any]:
        self._before_call("analyze_slice_health")
        return self._record(
            "analyze_slice_health",
            {"dimension": dimension, "metric_id": metric_id},
            analyze_slice_health(self.database.connection, dimension, metric_id),
        )

    def update_investigation_state(
        self,
        hypothesis_id: str,
        status: str,
        summary: str,
        evidence_ids: list[str],
    ) -> dict[str, Any]:
        self._before_call("update_investigation_state")
        if not hypothesis_id.strip() or not summary.strip():
            raise ValueError("hypothesis_id and summary must not be empty")
        if status not in INVESTIGATION_STATUSES:
            raise ValueError(f"unsupported investigation status: {status}")
        trace_evidence = {item["evidence_id"] for item in self.trace}
        unknown = sorted(set(evidence_ids).difference(trace_evidence))
        if unknown:
            raise ValueError(f"investigation state cites unknown evidence: {unknown}")
        state = {
            "hypothesis_id": hypothesis_id,
            "status": status,
            "summary": summary,
            "evidence_ids": list(evidence_ids),
        }
        self.investigation_state[hypothesis_id] = state
        return self._record("update_investigation_state", state, state)

    def request_additional_evidence(self, missing_evidence: str, reason: str) -> dict[str, Any]:
        self._before_call("request_additional_evidence")
        if not missing_evidence.strip() or not reason.strip():
            raise ValueError("missing_evidence and reason must not be empty")
        result = {
            "status": "recorded_for_human_follow_up",
            "missing_evidence": missing_evidence,
            "reason": reason,
        }
        return self._record(
            "request_additional_evidence",
            {"missing_evidence": missing_evidence, "reason": reason},
            result,
        )

    def submit_review(self, response: dict[str, Any]) -> dict[str, Any]:
        self._before_call("submit_review")
        report = ReviewReport.from_mapping(response)
        if report.task_type is not TaskType.HEALTH_DIAGNOSIS:
            raise ValueError("health environment accepts only experiment_health_diagnosis reports")
        cited = {
            evidence_id for claim in report.claim_evidence for evidence_id in claim.evidence_ids
        }
        trace_evidence = {item["evidence_id"] for item in self.trace}
        unknown = sorted(cited.difference(trace_evidence))
        if unknown:
            raise ValueError(f"report cites unknown evidence: {unknown}")
        return self._record(
            "submit_review",
            {"response": response},
            {
                "accepted": True,
                "schema_version": report.schema_version,
                "investigation_hypotheses": len(self.investigation_state),
            },
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()
