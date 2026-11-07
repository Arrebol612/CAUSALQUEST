"""Typed health-diagnosis expectations derived from controlled fault mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causalquest.data.faults import FaultType
from causalquest.schemas import ReviewDecision


@dataclass(frozen=True)
class HealthIssue:
    code: str
    severity: str
    message: str
    hard_gate: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class HealthAssessment:
    decision: ReviewDecision
    issues: tuple[HealthIssue, ...]
    required_tools: tuple[str, ...]
    diagnostic_evidence_tools: tuple[str, ...]
    required_documents: tuple[str, ...] = ("incident_ticket", "monitoring_runbook")

    def to_oracle_dict(self) -> dict[str, Any]:
        return {
            "expected_decision": self.decision.value,
            "required_issue_codes": [issue.code for issue in self.issues],
            "hard_gate_issue_codes": [issue.code for issue in self.issues if issue.hard_gate],
            "issues": [issue.to_dict() for issue in self.issues],
            "required_tools": list(self.required_tools),
            "diagnostic_evidence_tools": list(self.diagnostic_evidence_tools),
            "required_documents": list(self.required_documents),
        }


def _issue(code: str, message: str, *, hard_gate: bool = True) -> HealthIssue:
    return HealthIssue(code=code, severity="critical", message=message, hard_gate=hard_gate)


def expected_health_assessment(fault_type: FaultType | None) -> HealthAssessment:
    """Map a controlled mechanism to an evaluator-only operational expectation."""

    base_tools = {
        "get_document",
        "get_experiment",
        "get_metric_definition",
        "check_sample_ratio",
        "check_pre_experiment_balance",
        "check_assignment_exposure",
        "update_investigation_state",
    }
    if fault_type is None:
        evidence_tools = {
            "check_sample_ratio",
            "check_assignment_exposure",
            "check_metric_completeness",
            "check_event_timeliness",
        }
        return HealthAssessment(
            decision=ReviewDecision.HEALTHY_CONTINUE,
            issues=(),
            required_tools=tuple(sorted(base_tools | evidence_tools)),
            diagnostic_evidence_tools=tuple(sorted(evidence_tools)),
        )

    mapping = {
        FaultType.SAMPLE_RATIO_MISMATCH: (
            ReviewDecision.PAUSE_AND_INVESTIGATE,
            _issue(
                "severe_sample_ratio_mismatch",
                "观测分流显著偏离登记比例，当前数据不能支持处理效应解释",
            ),
            {"check_sample_ratio"},
        ),
        FaultType.TELEMETRY_MISSINGNESS: (
            ReviewDecision.PAUSE_AND_INVESTIGATE,
            _issue(
                "primary_metric_telemetry_missingness",
                "主指标在特定平台、版本和实验组存在系统性缺失",
            ),
            {"check_metric_completeness", "analyze_slice_health"},
        ),
        FaultType.DELAYED_EVENTS: (
            ReviewDecision.PAUSE_AND_INVESTIGATE,
            _issue(
                "metric_arrival_delay_or_immaturity",
                "指标到达延迟超过登记 SLA，且截至分析时间的数据可能尚未成熟",
            ),
            {"check_event_timeliness", "analyze_slice_health"},
        ),
        FaultType.DUPLICATE_EXPOSURE: (
            ReviewDecision.PAUSE_AND_INVESTIGATE,
            _issue(
                "duplicate_exposure_records",
                "同一实验单位存在重复曝光，指标可能被重复计数",
            ),
            {"check_assignment_exposure"},
        ),
        FaultType.CROSS_GROUP_CONTAMINATION: (
            ReviewDecision.INVALID_FOR_DECISION,
            _issue(
                "cross_group_contamination",
                "分配组与实际接收版本不一致，组间对比失去可解释性",
            ),
            {"check_assignment_exposure"},
        ),
    }
    decision, issue, evidence_tools = mapping[fault_type]
    return HealthAssessment(
        decision=decision,
        issues=(issue,),
        required_tools=tuple(sorted(base_tools | evidence_tools)),
        diagnostic_evidence_tools=tuple(sorted(evidence_tools)),
    )
