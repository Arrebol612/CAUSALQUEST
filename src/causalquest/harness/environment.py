"""Adapters that expose existing business tools through one safe agent interface."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from causalquest.data.hashing import sha256_file
from causalquest.evaluation.design_review import verify_design_review
from causalquest.evaluation.health_diagnosis import verify_health_diagnosis
from causalquest.evaluation.post_experiment import verify_post_experiment
from causalquest.tools.design_review import DesignReviewEnvironment
from causalquest.tools.health_diagnosis import HealthDiagnosisEnvironment
from causalquest.tools.post_experiment import PostExperimentEnvironment

from .contracts import CaseSpec, payload_hash

CANONICAL_ACTION_TOOLS = {"submit_review", "update_investigation_state"}


def _json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is not None:
        if origin in (list, tuple, set):
            return "array"
        if origin in (dict, Mapping):
            return "object"
        args = [item for item in get_args(annotation) if item is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else "string"
    return {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }.get(annotation, "string")


class ToolRegistry:
    """Allowlisted dispatcher with model-facing schemas derived from Python signatures."""

    def __init__(self, environment: Any) -> None:
        self.environment = environment
        self.allowed_tools = frozenset(environment.allowed_tools)

    def invoke(self, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if tool_name in CANONICAL_ACTION_TOOLS:
            raise PermissionError(f"{tool_name} must use its canonical action kind")
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"tool is not allowed for this case: {tool_name}")
        method = getattr(self.environment, tool_name, None)
        if method is None or not callable(method):
            raise PermissionError(f"allowed tool has no callable implementation: {tool_name}")
        return method(**dict(arguments))

    def update_state(self, patch: Mapping[str, Any]) -> dict[str, Any]:
        if "update_investigation_state" not in self.allowed_tools:
            raise PermissionError("state updates are not allowed for this case")
        return self.environment.update_investigation_state(**dict(patch))

    def submit(self, report: Mapping[str, Any]) -> dict[str, Any]:
        if "submit_review" not in self.allowed_tools:
            raise PermissionError("final report submission is not allowed for this case")
        return self.environment.submit_review(dict(report))

    def specs(self) -> list[dict[str, Any]]:
        specs = []
        for name in sorted(self.allowed_tools.difference(CANONICAL_ACTION_TOOLS)):
            method = getattr(self.environment, name)
            signature = inspect.signature(method)
            hints = get_type_hints(method)
            properties: dict[str, Any] = {}
            required: list[str] = []
            for parameter in signature.parameters.values():
                if parameter.name == "self":
                    continue
                properties[parameter.name] = {
                    "type": _json_type(hints.get(parameter.name, parameter.annotation))
                }
                if parameter.default is inspect.Parameter.empty:
                    required.append(parameter.name)
            specs.append(
                {
                    "name": name,
                    "description": (inspect.getdoc(method) or f"Execute {name}.").splitlines()[0],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            )
        return specs


class EnvironmentSession:
    """One isolated rollout environment and its evaluator-only verifier handle."""

    def __init__(self, spec: CaseSpec, environment: Any) -> None:
        self.spec = spec
        self.environment = environment
        self.tools = ToolRegistry(environment)
        self.case_dir = Path(spec.case_dir)

    def public_task(self) -> dict[str, Any]:
        manifest = dict(self.environment.manifest)
        for key in tuple(manifest):
            if "hidden" in key.lower() or key in {"database"}:
                manifest.pop(key)
        prompt_path = self.case_dir / "prompt.md"
        prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""
        return {
            **self.spec.public_identity(),
            "lifecycle_stage": manifest.get("lifecycle_stage"),
            "scenario_family": manifest.get("scenario_family"),
            "estimand": manifest.get("estimand"),
            "monitored_metric": manifest.get("monitored_metric"),
            "decision_policy": manifest.get("decision_policy"),
            "documents": sorted(manifest.get("documents", {})),
            "prompt": prompt,
            "tools": self.tools.specs(),
            "max_tool_calls": self.environment.max_tool_calls,
        }

    def state_snapshot(self) -> dict[str, Any]:
        return dict(self.environment.investigation_state)

    def state_hash(self) -> str:
        return payload_hash(self.state_snapshot())

    def observed_evidence(self) -> list[dict[str, str]]:
        return [
            {"tool_name": str(item["tool_name"]), "evidence_id": str(item["evidence_id"])}
            for item in self.environment.trace
        ]

    def latest_trace_record(self) -> dict[str, Any] | None:
        return dict(self.environment.trace[-1]) if self.environment.trace else None

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()


class EnvironmentFactory:
    """Create task-specific environments while keeping the LoopRunner task-agnostic."""

    def __init__(self, metric_catalog_path: str | Path) -> None:
        self.metric_catalog_path = Path(metric_catalog_path)

    def create(self, spec: CaseSpec) -> EnvironmentSession:
        if spec.task_type == "experiment_design_review":
            environment = DesignReviewEnvironment(spec.case_dir, self.metric_catalog_path)
        elif spec.task_type == "experiment_health_diagnosis":
            environment = HealthDiagnosisEnvironment(spec.case_dir)
        elif spec.task_type == "post_experiment_decision":
            environment = PostExperimentEnvironment(spec.case_dir)
        else:
            raise ValueError(f"unsupported task type: {spec.task_type}")
        return EnvironmentSession(spec, environment)

    @staticmethod
    def verify(session: EnvironmentSession, report: Mapping[str, Any]) -> dict[str, Any]:
        oracle_path = session.case_dir / "hidden" / "oracle.json"
        if (
            session.spec.oracle_sha256 is not None
            and sha256_file(oracle_path) != session.spec.oracle_sha256
        ):
            raise ValueError(f"oracle hash mismatch for {session.spec.case_id}")
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        trace = list(session.environment.trace)
        if session.spec.task_type == "experiment_design_review":
            return verify_design_review(report, oracle, trace)
        if session.spec.task_type == "experiment_health_diagnosis":
            return verify_health_diagnosis(report, oracle, trace)
        if session.spec.task_type == "post_experiment_decision":
            return verify_post_experiment(report, oracle, trace)
        raise ValueError(f"unsupported task type: {session.spec.task_type}")
