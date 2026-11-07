"""Budgeted Agent-facing tools for pre-launch experiment design review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from causalquest.schemas import EvidenceRecord, ReviewReport, TaskType
from causalquest.tasks.design_review import DesignDraft, estimate_binary_power

from .case_environment import ToolBudgetExceeded

TOOL_VERSION = "design-review-tools-v1"
INVESTIGATION_STATUSES = {"open", "supported", "rejected", "blocked"}


class DesignReviewEnvironment:
    """Expose only public case files and deterministic T1 analysis tools."""

    def __init__(self, case_dir: str | Path, metric_catalog_path: str | Path) -> None:
        self.case_dir = Path(case_dir)
        self.manifest = json.loads((self.case_dir / "manifest.json").read_text(encoding="utf-8"))
        self.design = DesignDraft.from_mapping(
            json.loads((self.case_dir / "experiment.json").read_text(encoding="utf-8"))
        )
        metric_payload = yaml.safe_load(Path(metric_catalog_path).read_text(encoding="utf-8"))
        self.metrics = {item["metric_id"]: item for item in metric_payload["metrics"]}
        self.allowed_tools = set(self.manifest["allowed_tools"])
        self.max_tool_calls = int(self.manifest["max_tool_calls"])
        self.trace: list[dict[str, Any]] = []
        related_path = self.case_dir / "related_experiments.json"
        self.related_experiments = (
            json.loads(related_path.read_text(encoding="utf-8")) if related_path.is_file() else []
        )
        self.investigation_state: dict[str, dict[str, Any]] = {}

    def _before_call(self, tool_name: str) -> None:
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"tool is not allowed for this case: {tool_name}")
        if len(self.trace) >= self.max_tool_calls:
            raise ToolBudgetExceeded(
                f"tool budget exhausted: {len(self.trace)}/{self.max_tool_calls}"
            )

    def _record(self, tool_name: str, arguments: dict[str, Any], result: Any) -> dict[str, Any]:
        evidence = EvidenceRecord.build(
            source_type="design_review_tool",
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
        docs_root = (self.case_dir / "docs").resolve()
        path = (docs_root / documents[document_id]).resolve()
        try:
            path.relative_to(docs_root)
        except ValueError as exc:
            raise PermissionError("document path escapes the public docs directory") from exc
        result = {"document_id": document_id, "text": path.read_text(encoding="utf-8")}
        return self._record("get_document", {"document_id": document_id}, result)

    def get_experiment(self) -> dict[str, Any]:
        self._before_call("get_experiment")
        return self._record("get_experiment", {}, self.design.to_dict())

    def get_metric_definition(self, metric_id: str) -> dict[str, Any]:
        self._before_call("get_metric_definition")
        if metric_id not in self.metrics:
            raise ValueError(f"unknown metric: {metric_id}")
        return self._record(
            "get_metric_definition", {"metric_id": metric_id}, self.metrics[metric_id]
        )

    def estimate_power(self) -> dict[str, Any]:
        self._before_call("estimate_power")
        values = (
            self.design.baseline_rate,
            self.design.mde_absolute,
            self.design.planned_sample_per_arm,
        )
        if any(value is None for value in values):
            result = {
                "status": "insufficient_inputs",
                "missing": [
                    name
                    for name, value in zip(
                        ("baseline_rate", "mde_absolute", "planned_sample_per_arm"),
                        values,
                        strict=True,
                    )
                    if value is None
                ],
            }
        else:
            result = estimate_binary_power(
                baseline_rate=self.design.baseline_rate,
                mde_absolute=self.design.mde_absolute,
                planned_sample_per_arm=self.design.planned_sample_per_arm,
                alpha=self.design.alpha,
                target_power=self.design.target_power,
            ).to_dict()
        return self._record("estimate_power", {}, result)

    def find_related_experiments(self, query: str = "") -> dict[str, Any]:
        self._before_call("find_related_experiments")
        needle = query.strip().lower()
        rows = [
            item
            for item in self.related_experiments
            if not needle or needle in json.dumps(item, ensure_ascii=False).lower()
        ]
        return self._record(
            "find_related_experiments",
            {"query": query},
            {"count": len(rows), "experiments": rows},
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
        known = {item["evidence_id"] for item in self.trace}
        unknown = sorted(set(evidence_ids).difference(known))
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
            {"status": "queued_for_human_owner", "evidence_type": evidence_type},
        )

    def submit_review(self, response: dict[str, Any]) -> dict[str, Any]:
        self._before_call("submit_review")
        report = ReviewReport.from_mapping(response)
        if report.task_type is not TaskType.DESIGN_REVIEW:
            raise ValueError("design environment accepts only experiment_design_review reports")
        cited = {
            evidence_id for claim in report.claim_evidence for evidence_id in claim.evidence_ids
        }
        known = {item["evidence_id"] for item in self.trace}
        unknown = sorted(cited.difference(known))
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
