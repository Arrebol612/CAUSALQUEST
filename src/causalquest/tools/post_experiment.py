"""Budgeted Agent-facing tools for completed Upworthy experiment decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causalquest.data.post_experiment import analyze_multiarm_binary, compare_binary_arms
from causalquest.schemas import EvidenceRecord, ReviewReport, TaskType

from .case_environment import ToolBudgetExceeded
from .readonly_duckdb import ReadOnlyDuckDB

TOOL_VERSION = "post-experiment-tools-v1"
VISIBLE_RELATIONS = {
    "arm_results",
    "experiment_context",
    "metric_catalog",
    "snapshot_metadata",
}
INVESTIGATION_STATUSES = {"open", "supported", "rejected", "blocked"}


class PostExperimentEnvironment:
    """Expose one experiment, public documents and deterministic statistical tools."""

    def __init__(self, case_dir: str | Path) -> None:
        self.case_dir = Path(case_dir)
        self.manifest = json.loads((self.case_dir / "manifest.json").read_text(encoding="utf-8"))
        self.database = ReadOnlyDuckDB(
            self.case_dir / self.manifest["database"],
            allowed_relations=VISIBLE_RELATIONS,
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
            source_type="post_experiment_tool",
            source_ref=f"{self.manifest['case_id']}:{tool_name}",
            source_version=TOOL_VERSION,
            arguments=arguments,
            result=result,
            data_snapshot_id=self.manifest["case_snapshot_id"],
            tool_version=TOOL_VERSION,
        )
        self.trace.append(
            {
                "step": len(self.trace) + 1,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "evidence_id": evidence.evidence_id,
                "result_sha256": evidence.result_sha256,
            }
        )
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
        result = self.database.query("SELECT * FROM experiment_context")
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

    def query_warehouse(self, sql: str, parameters: list[Any] | None = None) -> dict[str, Any]:
        self._before_call("query_warehouse")
        result = self.database.query(sql, parameters)
        return self._record("query_warehouse", {"sql": sql, "parameters": parameters or []}, result)

    def get_arm_results(self) -> dict[str, Any]:
        self._before_call("get_arm_results")
        result = self.database.query(
            """
            SELECT arm_id, headline, slug, eyecatcher_id, impressions, clicks, ctr,
                   editor_first_place, editor_winner
            FROM arm_results ORDER BY arm_id
            """
        )
        return self._record("get_arm_results", {}, result)

    def _arm(self, arm_id: str) -> dict[str, Any]:
        result = self.database.query(
            "SELECT arm_id, impressions, clicks FROM arm_results WHERE arm_id = ?", [arm_id]
        )
        if result["row_count"] != 1:
            raise ValueError(f"unknown arm_id: {arm_id}")
        return dict(zip(result["columns"], result["rows"][0]))

    def estimate_ab_effect(self, candidate_arm_id: str, reference_arm_id: str) -> dict[str, Any]:
        self._before_call("estimate_ab_effect")
        result = compare_binary_arms(
            self._arm(candidate_arm_id),
            self._arm(reference_arm_id),
            confidence_level=1 - float(self.manifest["decision_policy"]["alpha"]),
        )
        return self._record(
            "estimate_ab_effect",
            {"candidate_arm_id": candidate_arm_id, "reference_arm_id": reference_arm_id},
            result,
        )

    def adjust_multiple_testing(self, reference_arm_id: str) -> dict[str, Any]:
        self._before_call("adjust_multiple_testing")
        result = self.database.query(
            "SELECT arm_id, impressions, clicks FROM arm_results ORDER BY arm_id"
        )
        arms = [dict(zip(result["columns"], row)) for row in result["rows"]]
        adjusted = analyze_multiarm_binary(
            arms,
            reference_arm_id=reference_arm_id,
            alpha=float(self.manifest["decision_policy"]["alpha"]),
        )
        return self._record(
            "adjust_multiple_testing", {"reference_arm_id": reference_arm_id}, adjusted
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

    def request_additional_evidence(self, evidence_type: str, reason: str) -> dict[str, Any]:
        self._before_call("request_additional_evidence")
        if not evidence_type.strip() or not reason.strip():
            raise ValueError("evidence_type and reason must not be empty")
        return self._record(
            "request_additional_evidence",
            {"evidence_type": evidence_type, "reason": reason},
            {
                "status": "requires_new_registered_run_or_human_owner",
                "evidence_type": evidence_type,
            },
        )

    def submit_review(self, response: dict[str, Any]) -> dict[str, Any]:
        self._before_call("submit_review")
        report = ReviewReport.from_mapping(response)
        if report.task_type is not TaskType.POST_EXPERIMENT_DECISION:
            raise ValueError("post-experiment environment accepts only post_experiment reports")
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
