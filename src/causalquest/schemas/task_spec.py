"""Typed contract for a CausalQuest case."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """Task families supported by the project blueprint."""

    DESIGN_REVIEW = "experiment_design_review"
    HEALTH_DIAGNOSIS = "experiment_health_diagnosis"
    POST_EXPERIMENT_DECISION = "post_experiment_decision"
    RECOMMENDATION_POLICY_OPE = "recommendation_policy_ope"


class LifecycleStage(str, Enum):
    """Experiment stage visible to the agent."""

    DESIGN = "design"
    PRELAUNCH = "prelaunch"
    RUNNING = "running"
    COMPLETED = "completed"


class DataLayer(str, Enum):
    """Truth and provenance class for case evidence."""

    REAL_ARCHIVE = "real_archive"
    MIXED_EVIDENCE = "mixed_evidence"
    AUTHORED_CONTEXT = "authored_context"
    CONTROLLED_LAB = "controlled_lab"
    CONTROLLED_FAULT = "controlled_fault"
    HIDDEN_ORACLE = "hidden_oracle"


EXPECTED_STAGE = {
    TaskType.DESIGN_REVIEW: LifecycleStage.DESIGN,
    TaskType.HEALTH_DIAGNOSIS: LifecycleStage.RUNNING,
    TaskType.POST_EXPERIMENT_DECISION: LifecycleStage.COMPLETED,
    TaskType.RECOMMENDATION_POLICY_OPE: LifecycleStage.PRELAUNCH,
}


def _required_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _string_tuple(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of strings")
    result = tuple(_required_text(f"{name} item", item) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class TaskSpec:
    """Minimum versioned metadata required for every generated case."""

    case_id: str
    suite: str
    task_type: TaskType
    lifecycle_stage: LifecycleStage
    scenario_family: str
    data_layer: DataLayer
    randomization_unit: str
    exposure_unit: str
    analysis_unit: str
    estimand: str
    visible_evidence: tuple[str, ...]
    hidden_evidence: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    max_tool_calls: int
    decision_schema: str
    template_family_id: str
    source_group_id: str
    schema_version: str = "task-spec-v1"

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "suite",
            "scenario_family",
            "randomization_unit",
            "exposure_unit",
            "analysis_unit",
            "estimand",
            "decision_schema",
            "template_family_id",
            "source_group_id",
            "schema_version",
        ):
            _required_text(name, getattr(self, name))
        if EXPECTED_STAGE[self.task_type] != self.lifecycle_stage:
            raise ValueError(
                f"{self.task_type.value} requires lifecycle_stage "
                f"{EXPECTED_STAGE[self.task_type].value}"
            )
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")
        if not self.allowed_tools:
            raise ValueError("allowed_tools must not be empty")
        overlap = set(self.visible_evidence).intersection(self.hidden_evidence)
        if overlap:
            raise ValueError(f"visible and hidden evidence overlap: {sorted(overlap)}")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TaskSpec:
        """Parse and validate a JSON/YAML-compatible task mapping."""

        return cls(
            case_id=_required_text("case_id", payload.get("case_id")),
            suite=_required_text("suite", payload.get("suite")),
            task_type=TaskType(payload.get("task_type")),
            lifecycle_stage=LifecycleStage(payload.get("lifecycle_stage")),
            scenario_family=_required_text("scenario_family", payload.get("scenario_family")),
            data_layer=DataLayer(payload.get("data_layer")),
            randomization_unit=_required_text(
                "randomization_unit", payload.get("randomization_unit")
            ),
            exposure_unit=_required_text("exposure_unit", payload.get("exposure_unit")),
            analysis_unit=_required_text("analysis_unit", payload.get("analysis_unit")),
            estimand=_required_text("estimand", payload.get("estimand")),
            visible_evidence=_string_tuple("visible_evidence", payload.get("visible_evidence", [])),
            hidden_evidence=_string_tuple("hidden_evidence", payload.get("hidden_evidence", [])),
            allowed_tools=_string_tuple("allowed_tools", payload.get("allowed_tools", [])),
            max_tool_calls=int(payload.get("max_tool_calls", 0)),
            decision_schema=_required_text("decision_schema", payload.get("decision_schema")),
            template_family_id=_required_text(
                "template_family_id", payload.get("template_family_id")
            ),
            source_group_id=_required_text("source_group_id", payload.get("source_group_id")),
            schema_version=_required_text(
                "schema_version", payload.get("schema_version", "task-spec-v1")
            ),
        )

    def split_group_key(self) -> str:
        """Return a leakage-resistant group key for dataset splitting."""

        return (
            f"{self.source_group_id}|{self.template_family_id}|"
            f"{self.scenario_family}|{self.task_type.value}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete evaluator-side task contract."""

        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "suite": self.suite,
            "task_type": self.task_type.value,
            "lifecycle_stage": self.lifecycle_stage.value,
            "scenario_family": self.scenario_family,
            "data_layer": self.data_layer.value,
            "randomization_unit": self.randomization_unit,
            "exposure_unit": self.exposure_unit,
            "analysis_unit": self.analysis_unit,
            "estimand": self.estimand,
            "visible_evidence": list(self.visible_evidence),
            "hidden_evidence": list(self.hidden_evidence),
            "allowed_tools": list(self.allowed_tools),
            "max_tool_calls": self.max_tool_calls,
            "decision_schema": self.decision_schema,
            "template_family_id": self.template_family_id,
            "source_group_id": self.source_group_id,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize the Agent-visible contract without hidden evidence references."""

        payload = self.to_dict()
        payload.pop("hidden_evidence")
        return payload
