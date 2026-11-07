"""Deterministic policy and evaluator Oracle for completed aggregate experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalquest.data.post_experiment import analyze_multiarm_binary, minimum_sample_per_arm
from causalquest.schemas import ReviewDecision


@dataclass(frozen=True)
class PostExperimentPolicy:
    reference_arm_id: str
    candidate_arm_id: str
    mde_absolute: float
    alpha: float = 0.05
    target_power: float = 0.80
    action_scope: str = "editorial_package_selection"
    policy_origin: str = "authored_counterfactual_decision_contract"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PostExperimentPolicy:
        policy = cls(
            reference_arm_id=str(payload.get("reference_arm_id", "")).strip(),
            candidate_arm_id=str(payload.get("candidate_arm_id", "")).strip(),
            mde_absolute=float(payload.get("mde_absolute", 0)),
            alpha=float(payload.get("alpha", 0.05)),
            target_power=float(payload.get("target_power", 0.80)),
            action_scope=str(
                payload.get("action_scope", "editorial_package_selection")
            ).strip(),
            policy_origin=str(
                payload.get("policy_origin", "authored_counterfactual_decision_contract")
            ).strip(),
        )
        if not policy.reference_arm_id or not policy.candidate_arm_id:
            raise ValueError("reference_arm_id and candidate_arm_id are required")
        if policy.reference_arm_id == policy.candidate_arm_id:
            raise ValueError("reference and candidate arms must differ")
        if policy.mde_absolute <= 0:
            raise ValueError("mde_absolute must be positive")
        if not 0 < policy.alpha < 1 or not 0 < policy.target_power < 1:
            raise ValueError("alpha and target_power must be between zero and one")
        if not policy.action_scope or not policy.policy_origin:
            raise ValueError("action_scope and policy_origin are required")
        return policy

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DecisionIssue:
    code: str
    severity: str
    message: str
    hard_gate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PostExperimentAssessment:
    decision: ReviewDecision
    issues: tuple[DecisionIssue, ...]
    analysis: dict[str, Any]
    required_tools: tuple[str, ...]
    diagnostic_evidence_tools: tuple[str, ...]
    required_documents: tuple[str, ...]

    def to_oracle_dict(self) -> dict[str, Any]:
        return {
            "expected_decision": self.decision.value,
            "required_issue_codes": [issue.code for issue in self.issues],
            "hard_gate_issue_codes": [issue.code for issue in self.issues if issue.hard_gate],
            "issues": [issue.to_dict() for issue in self.issues],
            "analysis": self.analysis,
            "required_tools": list(self.required_tools),
            "diagnostic_evidence_tools": list(self.diagnostic_evidence_tools),
            "required_documents": list(self.required_documents),
        }


def _issue(
    code: str, message: str, *, severity: str = "warning", hard_gate: bool = False
) -> DecisionIssue:
    return DecisionIssue(code, severity, message, hard_gate)


def assess_post_experiment(
    arms: Sequence[Mapping[str, Any]],
    policy: PostExperimentPolicy,
    *,
    randomization_risk: bool,
    quarantine_reasons: str | None,
) -> PostExperimentAssessment:
    """Apply validity, multiplicity, uncertainty and practical-significance gates."""

    by_id = {str(arm["arm_id"]): dict(arm) for arm in arms}
    if len(by_id) != len(arms):
        raise ValueError("arm IDs must be unique")
    for arm_id in (policy.reference_arm_id, policy.candidate_arm_id):
        if arm_id not in by_id:
            raise ValueError(f"declared arm is absent: {arm_id}")

    validity_tools = (
        "get_document",
        "get_experiment",
        "update_investigation_state",
    )
    if quarantine_reasons or randomization_risk:
        issue = (
            _issue(
                "source_test_quarantined",
                f"源实验已隔离：{quarantine_reasons}",
                severity="critical",
                hard_gate=True,
            )
            if quarantine_reasons
            else _issue(
                "published_randomization_risk_period",
                "实验位于维护者公告的随机化风险日期窗，不能支持正向因果决策",
                severity="critical",
                hard_gate=True,
            )
        )
        return PostExperimentAssessment(
            ReviewDecision.INSUFFICIENT_EVIDENCE,
            (issue,),
            {
                "policy": policy.to_dict(),
                "analysis_skipped": True,
                "skip_reason": issue.code,
                "randomization_risk": bool(randomization_risk),
                "quarantine_reasons": quarantine_reasons,
            },
            validity_tools,
            ("get_experiment",),
            ("decision_brief", "method_playbook"),
        )

    family = analyze_multiarm_binary(
        arms,
        reference_arm_id=policy.reference_arm_id,
        alpha=policy.alpha,
    )
    comparison = next(
        item
        for item in family["comparisons"]
        if item["candidate_arm_id"] == policy.candidate_arm_id
    )
    required_per_arm = minimum_sample_per_arm(
        comparison["reference"]["rate"],
        policy.mde_absolute,
        alpha=policy.alpha,
        target_power=policy.target_power,
        family_size=family["comparison_count"],
    )
    observed_min = min(
        comparison["reference"]["impressions"], comparison["candidate"]["impressions"]
    )
    information_fraction = min(1.0, observed_min / required_per_arm)
    analysis = {
        "policy": policy.to_dict(),
        "family": family,
        "candidate_comparison": comparison,
        "required_sample_per_arm_approx": required_per_arm,
        "observed_min_sample_per_declared_arm": observed_min,
        "information_fraction": information_fraction,
        "randomization_risk": bool(randomization_risk),
        "quarantine_reasons": quarantine_reasons,
    }
    base_tools = (
        "adjust_multiple_testing",
        "get_arm_results",
        "get_document",
        "get_experiment",
        "get_metric_definition",
        "update_investigation_state",
    )
    lift = comparison["absolute_lift"]
    lower = comparison["difference_ci_low"]
    upper = comparison["difference_ci_high"]
    significant = comparison["reject_null_adjusted"]
    if significant and lower >= policy.mde_absolute:
        decision = ReviewDecision.SELECT_CANDIDATE
        issues: tuple[DecisionIssue, ...] = ()
    elif significant and lower > 0 and lift >= policy.mde_absolute:
        decision = ReviewDecision.LIMITED_VALIDATION
        issues = (
            _issue(
                "practical_magnitude_uncertain",
                "效应方向可靠且点估计达到 MDE，但同时区间下界尚未达到直接选择阈值；需新验证实验补齐质量护栏",
            ),
        )
    elif upper <= 0:
        decision = ReviewDecision.REJECT_CANDIDATE
        issues = (
            _issue(
                "candidate_harm_detected",
                "候选实验臂的点击率差值区间上界不高于零",
                severity="critical",
            ),
        )
    elif significant and lift > 0:
        decision = ReviewDecision.ITERATE_AND_RETEST
        issues = (
            _issue(
                "positive_but_below_registered_mde",
                "候选方向为正但点估计未达到任务合同的最小业务收益，建议迭代方案后重测",
            ),
        )
    elif upper < policy.mde_absolute and information_fraction >= 1:
        decision = ReviewDecision.REJECT_CANDIDATE
        issues = (
            _issue(
                "practical_benefit_ruled_out",
                "当前同时区间已排除任务合同的最小业务收益",
            ),
        )
    elif information_fraction < 1 and lower <= 0 <= upper:
        decision = ReviewDecision.RERUN_WITH_REGISTERED_PROTOCOL
        issues = (
            _issue(
                "insufficient_information_for_registered_mde",
                "历史实验已结束且信息量不足；应按新登记协议重跑，而不是虚构继续收流量",
            ),
        )
    else:
        decision = ReviewDecision.ITERATE_AND_RETEST
        issues = (
            _issue(
                "registered_success_criterion_not_met",
                "多重校正、方向和实际显著性联合门禁未通过",
            ),
        )
    return PostExperimentAssessment(
        decision,
        issues,
        analysis,
        base_tools,
        ("adjust_multiple_testing", "get_arm_results"),
        ("decision_brief", "metric_definition", "method_playbook"),
    )
