"""Deterministic design-review domain model and pre-launch checks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from causalquest.schemas import ReviewDecision


@dataclass(frozen=True)
class MetricDraft:
    metric_id: str
    role: str
    unit: str
    attribution_window: str
    direction: str
    decision_threshold: str
    aggregation_to_analysis_unit: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MetricDraft:
        return cls(
            metric_id=str(payload.get("metric_id", "")).strip(),
            role=str(payload.get("role", "")).strip(),
            unit=str(payload.get("unit", "")).strip(),
            attribution_window=str(payload.get("attribution_window", "")).strip(),
            direction=str(payload.get("direction", "")).strip(),
            decision_threshold=str(payload.get("decision_threshold", "")).strip(),
            aggregation_to_analysis_unit=str(
                payload.get("aggregation_to_analysis_unit", "")
            ).strip(),
        )


@dataclass(frozen=True)
class DesignDraft:
    experiment_id: str
    scenario_family: str
    hypothesis: str
    decision_action: str
    randomization_unit: str
    exposure_unit: str
    analysis_unit: str
    analysis_population: str
    estimand: str
    exposure_definition: str
    design_type: str
    interference_risk: str
    cluster_adjustment: bool
    planned_sample_per_arm: int | None
    baseline_rate: float | None
    mde_absolute: float | None
    alpha: float
    target_power: float
    multiplicity_policy: str
    daily_eligible_units: int | None
    max_duration_days: int | None
    industrial_review: bool
    metrics: tuple[MetricDraft, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DesignDraft:
        def optional_number(name: str, cast):
            value = payload.get(name)
            return None if value is None else cast(value)

        return cls(
            experiment_id=str(payload.get("experiment_id", "")).strip(),
            scenario_family=str(payload.get("scenario_family", "")).strip(),
            hypothesis=str(payload.get("hypothesis", "")).strip(),
            decision_action=str(payload.get("decision_action", "")).strip(),
            randomization_unit=str(payload.get("randomization_unit", "")).strip(),
            exposure_unit=str(payload.get("exposure_unit", "")).strip(),
            analysis_unit=str(payload.get("analysis_unit", "")).strip(),
            analysis_population=str(payload.get("analysis_population", "")).strip(),
            estimand=str(payload.get("estimand", "")).strip(),
            exposure_definition=str(payload.get("exposure_definition", "")).strip(),
            design_type=str(payload.get("design_type", "")).strip(),
            interference_risk=str(payload.get("interference_risk", "")).strip(),
            cluster_adjustment=payload.get("cluster_adjustment", False) is True,
            planned_sample_per_arm=optional_number("planned_sample_per_arm", int),
            baseline_rate=optional_number("baseline_rate", float),
            mde_absolute=optional_number("mde_absolute", float),
            alpha=float(payload.get("alpha", 0.05)),
            target_power=float(payload.get("target_power", 0.8)),
            multiplicity_policy=str(payload.get("multiplicity_policy", "")).strip(),
            daily_eligible_units=optional_number("daily_eligible_units", int),
            max_duration_days=optional_number("max_duration_days", int),
            industrial_review=payload.get("industrial_review", False) is True,
            metrics=tuple(MetricDraft.from_mapping(item) for item in payload.get("metrics", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scenario_family": self.scenario_family,
            "hypothesis": self.hypothesis,
            "decision_action": self.decision_action,
            "randomization_unit": self.randomization_unit,
            "exposure_unit": self.exposure_unit,
            "analysis_unit": self.analysis_unit,
            "analysis_population": self.analysis_population,
            "estimand": self.estimand,
            "exposure_definition": self.exposure_definition,
            "design_type": self.design_type,
            "interference_risk": self.interference_risk,
            "cluster_adjustment": self.cluster_adjustment,
            "planned_sample_per_arm": self.planned_sample_per_arm,
            "baseline_rate": self.baseline_rate,
            "mde_absolute": self.mde_absolute,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "multiplicity_policy": self.multiplicity_policy,
            "daily_eligible_units": self.daily_eligible_units,
            "max_duration_days": self.max_duration_days,
            "industrial_review": self.industrial_review,
            "metrics": [metric.__dict__ for metric in self.metrics],
        }


@dataclass(frozen=True)
class PowerEstimate:
    required_sample_per_arm: int
    planned_sample_per_arm: int
    achieved_power: float
    baseline_rate: float
    treatment_rate_at_mde: float
    mde_absolute: float
    alpha: float
    target_power: float
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def estimate_binary_power(
    *,
    baseline_rate: float,
    mde_absolute: float,
    planned_sample_per_arm: int,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> PowerEstimate:
    """Approximate equal-allocation two-sided power for a difference in proportions."""

    treatment_rate = baseline_rate + mde_absolute
    if not 0 < baseline_rate < 1 or not 0 < treatment_rate < 1:
        raise ValueError("baseline rate and treatment rate at MDE must be in (0, 1)")
    if mde_absolute <= 0:
        raise ValueError("mde_absolute must be positive")
    if planned_sample_per_arm <= 0:
        raise ValueError("planned_sample_per_arm must be positive")
    if not 0 < alpha < 1 or not 0.5 < target_power < 1:
        raise ValueError("alpha and target_power are outside supported ranges")

    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(target_power)
    pooled = (baseline_rate + treatment_rate) / 2
    null_variance = 2 * pooled * (1 - pooled)
    alternative_variance = baseline_rate * (1 - baseline_rate) + treatment_rate * (
        1 - treatment_rate
    )
    numerator = z_alpha * math.sqrt(null_variance) + z_power * math.sqrt(alternative_variance)
    required = math.ceil((numerator / mde_absolute) ** 2)

    signal = mde_absolute * math.sqrt(planned_sample_per_arm)
    critical_adjustment = z_alpha * math.sqrt(null_variance)
    achieved = normal.cdf((signal - critical_adjustment) / math.sqrt(alternative_variance))
    return PowerEstimate(
        required_sample_per_arm=required,
        planned_sample_per_arm=planned_sample_per_arm,
        achieved_power=max(0.0, min(1.0, achieved)),
        baseline_rate=baseline_rate,
        treatment_rate_at_mde=treatment_rate,
        mde_absolute=mde_absolute,
        alpha=alpha,
        target_power=target_power,
        assumptions=(
            "two_sided_normal_approximation",
            "equal_allocation",
            "independent_analysis_units",
            "no_multiple_testing_adjustment",
        ),
    )


@dataclass(frozen=True)
class DesignIssue:
    code: str
    severity: str
    message: str
    hard_gate: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DesignAssessment:
    decision: ReviewDecision
    issues: tuple[DesignIssue, ...]
    power: PowerEstimate | None
    required_tools: tuple[str, ...]
    diagnostic_evidence_tools: tuple[str, ...]
    required_documents: tuple[str, ...]

    def to_oracle_dict(self) -> dict[str, Any]:
        return {
            "expected_decision": self.decision.value,
            "required_issue_codes": [issue.code for issue in self.issues],
            "hard_gate_issue_codes": [issue.code for issue in self.issues if issue.hard_gate],
            "issues": [issue.to_dict() for issue in self.issues],
            "power": self.power.to_dict() if self.power else None,
            "required_tools": list(self.required_tools),
            "diagnostic_evidence_tools": list(self.diagnostic_evidence_tools),
            "required_documents": list(self.required_documents),
        }


def _issue(code: str, severity: str, message: str, *, hard_gate: bool = False) -> DesignIssue:
    return DesignIssue(code=code, severity=severity, message=message, hard_gate=hard_gate)


def assess_design(draft: DesignDraft) -> DesignAssessment:
    """Apply deterministic design gates and produce an evaluator-side oracle."""

    issues: list[DesignIssue] = []
    if not draft.hypothesis:
        issues.append(_issue("missing_falsifiable_hypothesis", "major", "缺少可证伪假设"))
    if not draft.decision_action:
        issues.append(_issue("missing_decision_action", "major", "缺少实验结果对应的决策动作"))
    if not draft.exposure_definition:
        issues.append(
            _issue(
                "missing_exposure_definition",
                "critical",
                "未定义何时视为真正接受 treatment",
                hard_gate=True,
            )
        )
    if draft.randomization_unit != draft.analysis_unit and not draft.cluster_adjustment:
        issues.append(
            _issue(
                "analysis_unit_ignores_randomization",
                "critical",
                "分析单位细于随机化单位且未声明聚类调整",
                hard_gate=True,
            )
        )
    if draft.estimand == "assignment_itt" and draft.analysis_population != "assigned_units":
        issues.append(
            _issue(
                "post_treatment_population_filter",
                "critical",
                "ITT estimand 使用了分组后的行为筛选总体",
                hard_gate=True,
            )
        )
    if draft.interference_risk == "high" and draft.design_type == "individual_ab":
        issues.append(
            _issue(
                "interference_requires_alternative_design",
                "critical",
                "高干扰场景不应直接采用个体级 A/B",
                hard_gate=True,
            )
        )

    roles = {metric.role for metric in draft.metrics}
    if "primary" not in roles:
        issues.append(
            _issue(
                "missing_primary_metric",
                "critical",
                "指标计划没有唯一明确的主指标",
                hard_gate=True,
            )
        )
    if "guardrail" not in roles:
        issues.append(_issue("missing_guardrail_metric", "major", "指标计划缺少护栏指标"))

    if draft.industrial_review:
        primary_metrics = [metric for metric in draft.metrics if metric.role == "primary"]
        guardrails = [metric for metric in draft.metrics if metric.role == "guardrail"]
        if len(primary_metrics) > 1:
            issues.append(
                _issue(
                    "primary_metric_not_unique",
                    "critical",
                    "工业评审要求唯一主指标；多主指标必须重写决策合同",
                    hard_gate=True,
                )
            )
        if any(not metric.decision_threshold for metric in guardrails):
            issues.append(
                _issue(
                    "guardrail_threshold_missing",
                    "critical",
                    "护栏只有方向而没有可执行阈值",
                    hard_gate=True,
                )
            )
        if any(
            metric.unit not in {draft.analysis_unit, "assigned_unit", "exposed_unit"}
            and not metric.aggregation_to_analysis_unit
            for metric in draft.metrics
        ):
            issues.append(
                _issue(
                    "metric_aggregation_undefined",
                    "critical",
                    "session/request 级指标没有声明如何聚合到分析单位",
                    hard_gate=True,
                )
            )
        if not draft.multiplicity_policy:
            issues.append(
                _issue(
                    "multiplicity_policy_missing",
                    "major",
                    "未声明多指标决策族及校正策略",
                )
            )
        traffic_inputs = (draft.daily_eligible_units, draft.max_duration_days)
        if any(value is None for value in traffic_inputs):
            issues.append(
                _issue(
                    "traffic_budget_missing",
                    "major",
                    "缺少每日合格流量或最长实验周期，无法验证排期可执行性",
                )
            )

    complete_power_inputs = all(
        value is not None
        for value in (draft.baseline_rate, draft.mde_absolute, draft.planned_sample_per_arm)
    )
    power = None
    if complete_power_inputs:
        power = estimate_binary_power(
            baseline_rate=draft.baseline_rate,
            mde_absolute=draft.mde_absolute,
            planned_sample_per_arm=draft.planned_sample_per_arm,
            alpha=draft.alpha,
            target_power=draft.target_power,
        )
        if power.planned_sample_per_arm < power.required_sample_per_arm:
            issues.append(
                _issue(
                    "underpowered_for_registered_mde",
                    "critical",
                    "计划样本量无法达到登记 MDE 对应的目标功效",
                    hard_gate=True,
                )
            )
        if (
            draft.industrial_review
            and draft.daily_eligible_units is not None
            and draft.max_duration_days is not None
            and 2 * power.required_sample_per_arm
            > draft.daily_eligible_units * draft.max_duration_days
        ):
            issues.append(
                _issue(
                    "traffic_budget_infeasible",
                    "critical",
                    "登记 MDE 所需总样本超过最长周期内的可用合格流量",
                    hard_gate=True,
                )
            )
    else:
        issues.append(
            _issue(
                "insufficient_power_inputs",
                "critical",
                "缺少 baseline、MDE 或计划样本量，无法评估功效",
                hard_gate=True,
            )
        )

    if any(issue.code == "insufficient_power_inputs" for issue in issues):
        decision = ReviewDecision.INSUFFICIENT_DESIGN_EVIDENCE
    elif issues:
        decision = ReviewDecision.REVISE_DESIGN
    else:
        decision = ReviewDecision.APPROVE_DESIGN
    required_tools = ["get_experiment", "get_metric_definition", "estimate_power"]
    diagnostic_tools = ["get_experiment", "estimate_power"]
    required_documents: list[str] = []
    if draft.industrial_review:
        required_tools.extend(
            ["get_document", "find_related_experiments", "update_investigation_state"]
        )
        diagnostic_tools.extend(["get_document", "find_related_experiments"])
        required_documents.extend(
            ["experiment_brief", "metric_plan", "traffic_forecast", "decision_charter"]
        )
    return DesignAssessment(
        decision=decision,
        issues=tuple(issues),
        power=power,
        required_tools=tuple(required_tools),
        diagnostic_evidence_tools=tuple(diagnostic_tools),
        required_documents=tuple(required_documents),
    )
