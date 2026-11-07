"""Stage-specific review report contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .task_spec import TaskType


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewDecision(str, Enum):
    APPROVE_DESIGN = "approve_design"
    REVISE_DESIGN = "revise_design"
    REJECT_DESIGN = "reject_design"
    INSUFFICIENT_DESIGN_EVIDENCE = "insufficient_design_evidence"
    HEALTHY_CONTINUE = "healthy_continue"
    CONTINUE_WITH_WARNING = "continue_with_warning"
    PAUSE_AND_INVESTIGATE = "pause_and_investigate"
    FIX_AND_RESTART = "fix_and_restart"
    INVALID_FOR_DECISION = "invalid_for_decision"
    SHIP = "ship"
    GUARDED_ROLLOUT = "guarded_rollout"
    CONTINUE_EXPERIMENT = "continue_experiment"
    ITERATE_AND_RETEST = "iterate_and_retest"
    DO_NOT_SHIP = "do_not_ship"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SELECT_CANDIDATE = "select_candidate"
    LIMITED_VALIDATION = "limited_validation"
    RERUN_WITH_REGISTERED_PROTOCOL = "rerun_with_registered_protocol"
    REJECT_CANDIDATE = "reject_candidate"
    ADVANCE_TO_ONLINE_TEST = "advance_to_online_test"
    HOLD_FOR_MORE_OFFLINE_EVIDENCE = "hold_for_more_offline_evidence"
    REJECT_OFFLINE_CANDIDATE = "reject_offline_candidate"


ALLOWED_DECISIONS = {
    TaskType.DESIGN_REVIEW: {
        ReviewDecision.APPROVE_DESIGN,
        ReviewDecision.REVISE_DESIGN,
        ReviewDecision.REJECT_DESIGN,
        ReviewDecision.INSUFFICIENT_DESIGN_EVIDENCE,
    },
    TaskType.HEALTH_DIAGNOSIS: {
        ReviewDecision.HEALTHY_CONTINUE,
        ReviewDecision.CONTINUE_WITH_WARNING,
        ReviewDecision.PAUSE_AND_INVESTIGATE,
        ReviewDecision.FIX_AND_RESTART,
        ReviewDecision.INVALID_FOR_DECISION,
    },
    TaskType.POST_EXPERIMENT_DECISION: {
        ReviewDecision.SHIP,
        ReviewDecision.GUARDED_ROLLOUT,
        ReviewDecision.CONTINUE_EXPERIMENT,
        ReviewDecision.ITERATE_AND_RETEST,
        ReviewDecision.DO_NOT_SHIP,
        ReviewDecision.INSUFFICIENT_EVIDENCE,
        ReviewDecision.SELECT_CANDIDATE,
        ReviewDecision.LIMITED_VALIDATION,
        ReviewDecision.RERUN_WITH_REGISTERED_PROTOCOL,
        ReviewDecision.REJECT_CANDIDATE,
    },
    TaskType.RECOMMENDATION_POLICY_OPE: {
        ReviewDecision.ADVANCE_TO_ONLINE_TEST,
        ReviewDecision.HOLD_FOR_MORE_OFFLINE_EVIDENCE,
        ReviewDecision.REJECT_OFFLINE_CANDIDATE,
        ReviewDecision.INSUFFICIENT_EVIDENCE,
    },
}


def _text_tuple(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of strings")
    result = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(result) != len(value):
        raise ValueError(f"{name} must contain only non-empty strings")
    return result


@dataclass(frozen=True)
class ClaimEvidence:
    """One report claim and the evidence records that support it."""

    claim: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("claim must not be empty")
        if not self.evidence_ids:
            raise ValueError("every claim must cite at least one evidence ID")
        if any(not value.startswith("ev_") for value in self.evidence_ids):
            raise ValueError("evidence IDs must use the ev_ prefix")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ClaimEvidence:
        return cls(
            claim=str(payload.get("claim", "")).strip(),
            evidence_ids=_text_tuple("evidence_ids", payload.get("evidence_ids", [])),
        )


@dataclass(frozen=True)
class ReviewReport:
    """Human-reviewable final output submitted by an agent."""

    task_type: TaskType
    decision: ReviewDecision
    confidence: Confidence
    estimand: str
    validated_findings: tuple[str, ...]
    blocking_issues: tuple[str, ...]
    guardrail_findings: tuple[str, ...]
    uncertainties: tuple[str, ...]
    next_actions: tuple[str, ...]
    claim_evidence: tuple[ClaimEvidence, ...]
    requires_human_approval: bool = True
    schema_version: str = "review-report-v1"

    def __post_init__(self) -> None:
        if self.decision not in ALLOWED_DECISIONS[self.task_type]:
            raise ValueError(
                f"decision {self.decision.value} is invalid for {self.task_type.value}"
            )
        if not self.estimand.strip():
            raise ValueError("estimand must not be empty")
        if not self.requires_human_approval:
            raise ValueError("CausalQuest reports must require human approval")
        cited_claims = {item.claim for item in self.claim_evidence}
        missing = set(self.validated_findings).difference(cited_claims)
        if missing:
            raise ValueError(f"validated findings lack evidence mappings: {sorted(missing)}")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ReviewReport:
        """Parse a JSON-compatible report and enforce stage-specific decisions."""

        return cls(
            task_type=TaskType(payload.get("task_type")),
            decision=ReviewDecision(payload.get("decision")),
            confidence=Confidence(payload.get("confidence")),
            estimand=str(payload.get("estimand", "")).strip(),
            validated_findings=_text_tuple(
                "validated_findings", payload.get("validated_findings", [])
            ),
            blocking_issues=_text_tuple("blocking_issues", payload.get("blocking_issues", [])),
            guardrail_findings=_text_tuple(
                "guardrail_findings", payload.get("guardrail_findings", [])
            ),
            uncertainties=_text_tuple("uncertainties", payload.get("uncertainties", [])),
            next_actions=_text_tuple("next_actions", payload.get("next_actions", [])),
            claim_evidence=tuple(
                ClaimEvidence.from_mapping(item) for item in payload.get("claim_evidence", [])
            ),
            requires_human_approval=payload.get("requires_human_approval", True) is True,
            schema_version=str(payload.get("schema_version", "review-report-v1")),
        )
