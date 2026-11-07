"""Budgeted case environment that never exposes hidden evaluator files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from causalquest.data.analysis import (
    assess_logging_health,
    compare_observed_ctr,
    policy_window_summary,
)

from .readonly_duckdb import ReadOnlyDuckDB

DOCUMENTS = {
    "experiment_brief": ("experiment_brief.md", "experiment_brief"),
    "metric_definition": ("metric_definition.md", "metric_definition_ctr"),
    "method_playbook": ("method_playbook.md", "method_playbook_logging"),
}


class ToolBudgetExceeded(RuntimeError):
    """Raised when a trajectory exceeds the case's explicit tool budget."""


class CaseEnvironment:
    """Expose visible documents and trusted analyses through a logged tool protocol."""

    def __init__(self, case_dir: str | Path, database_path: str | Path) -> None:
        self.case_dir = Path(case_dir)
        self.manifest = json.loads((self.case_dir / "manifest.json").read_text(encoding="utf-8"))
        self.database = ReadOnlyDuckDB(database_path)
        self.max_tool_calls = int(self.manifest["max_tool_calls"])
        self.allowed_tools = set(self.manifest["allowed_tools"])
        self.trace: list[dict[str, Any]] = []

    def close(self) -> None:
        self.database.close()

    def _before_call(self, tool_name: str) -> None:
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"tool is not allowed for this case: {tool_name}")
        if len(self.trace) >= self.max_tool_calls:
            raise ToolBudgetExceeded(
                f"tool budget exhausted: {len(self.trace)}/{self.max_tool_calls}"
            )

    def _record(
        self, tool_name: str, arguments: dict[str, Any], result: Any, evidence_id: str
    ) -> dict[str, Any]:
        payload = json.dumps(
            {"tool": tool_name, "arguments": arguments, "result": result},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        record = {
            "step": len(self.trace) + 1,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "evidence_id": evidence_id,
            "result_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
        self.trace.append(record)
        return {"evidence_id": evidence_id, "result": result}

    def retrieve_doc(self, document: str) -> dict[str, Any]:
        self._before_call("retrieve_doc")
        if document not in DOCUMENTS:
            raise ValueError(f"unknown visible document: {document}")
        filename, evidence_id = DOCUMENTS[document]
        text = (self.case_dir / "docs" / filename).read_text(encoding="utf-8")
        return self._record("retrieve_doc", {"document": document}, {"text": text}, evidence_id)

    def describe_table(self, table: str) -> dict[str, Any]:
        self._before_call("describe_table")
        result = self.database.describe_table(table)
        return self._record("describe_table", {"table": table}, result, f"schema_{table}")

    def query_duckdb(self, sql: str, parameters: list[Any] | None = None) -> dict[str, Any]:
        self._before_call("query_duckdb")
        result = self.database.query(sql, parameters)
        digest = hashlib.sha256(
            json.dumps({"sql": sql, "parameters": parameters}, sort_keys=True).encode()
        ).hexdigest()[:16]
        return self._record(
            "query_duckdb",
            {"sql": sql, "parameters": parameters or []},
            result,
            f"query_{digest}",
        )

    def check_logging_health(self) -> dict[str, Any]:
        self._before_call("check_logging_health")
        start, end = self.manifest["analysis_window"]
        rows = policy_window_summary(
            self.database.connection, self.manifest["campaign"], start, end
        )
        result = assess_logging_health(rows)
        return self._record(
            "check_logging_health",
            {"campaign": self.manifest["campaign"], "start": start, "end": end},
            result,
            "propensity_health",
        )

    def run_ab_analysis(self) -> dict[str, Any]:
        self._before_call("run_ab_analysis")
        start, end = self.manifest["analysis_window"]
        rows = policy_window_summary(
            self.database.connection, self.manifest["campaign"], start, end
        )
        result = compare_observed_ctr(rows)
        return self._record(
            "run_ab_analysis",
            {"campaign": self.manifest["campaign"], "start": start, "end": end},
            result,
            "daily_policy_metrics",
        )

    def submit_review(self, response: dict[str, Any]) -> dict[str, Any]:
        self._before_call("submit_review")
        required = {"recommendation", "metrics", "evidence_ids", "limitations"}
        missing = sorted(required.difference(response))
        result = {"accepted": not missing, "missing_fields": missing}
        return self._record("submit_review", {"response": response}, result, "submitted_review")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()
