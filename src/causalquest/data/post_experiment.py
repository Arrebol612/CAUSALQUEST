"""Trusted statistics for aggregate binary A/B/n experiment decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from .statistics import wilson_interval


def _validate_arm(arm: Mapping[str, Any]) -> tuple[str, int, int]:
    arm_id = str(arm.get("arm_id", "")).strip()
    if not arm_id:
        raise ValueError("arm_id must not be empty")
    impressions = int(arm.get("impressions", 0))
    clicks = int(arm.get("clicks", -1))
    if impressions <= 0:
        raise ValueError(f"arm {arm_id} impressions must be positive")
    if clicks < 0 or clicks > impressions:
        raise ValueError(f"arm {arm_id} clicks must be between zero and impressions")
    return arm_id, impressions, clicks


def compare_binary_arms(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compare two aggregate binomial arms with Wilson/Newcombe uncertainty."""

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    candidate_id, candidate_n, candidate_clicks = _validate_arm(candidate)
    reference_id, reference_n, reference_clicks = _validate_arm(reference)
    if candidate_id == reference_id:
        raise ValueError("candidate and reference arms must differ")

    alpha = 1 - confidence_level
    z = NormalDist().inv_cdf(1 - alpha / 2)
    candidate_estimate = wilson_interval(candidate_clicks, candidate_n, z)
    reference_estimate = wilson_interval(reference_clicks, reference_n, z)
    difference = candidate_estimate.rate - reference_estimate.rate

    # Newcombe method 10: combine the two Wilson score intervals without pooling.
    lower = difference - math.sqrt(
        (candidate_estimate.rate - candidate_estimate.ci_low) ** 2
        + (reference_estimate.ci_high - reference_estimate.rate) ** 2
    )
    upper = difference + math.sqrt(
        (candidate_estimate.ci_high - candidate_estimate.rate) ** 2
        + (reference_estimate.rate - reference_estimate.ci_low) ** 2
    )
    pooled = (candidate_clicks + reference_clicks) / (candidate_n + reference_n)
    pooled_se = math.sqrt(pooled * (1 - pooled) * (1 / candidate_n + 1 / reference_n))
    if pooled_se:
        z_score = difference / pooled_se
        p_value = 2 * (1 - NormalDist().cdf(abs(z_score)))
    else:
        z_score = 0.0
        p_value = 1.0
    relative_lift = difference / reference_estimate.rate if reference_estimate.rate else None
    return {
        "method": "newcombe_wilson_difference_with_pooled_two_sided_z_test",
        "confidence_level": confidence_level,
        "candidate_arm_id": candidate_id,
        "reference_arm_id": reference_id,
        "candidate": {
            "clicks": candidate_clicks,
            "impressions": candidate_n,
            "rate": candidate_estimate.rate,
            "wilson_ci_low": candidate_estimate.ci_low,
            "wilson_ci_high": candidate_estimate.ci_high,
        },
        "reference": {
            "clicks": reference_clicks,
            "impressions": reference_n,
            "rate": reference_estimate.rate,
            "wilson_ci_low": reference_estimate.ci_low,
            "wilson_ci_high": reference_estimate.ci_high,
        },
        "absolute_lift": difference,
        "relative_lift": relative_lift,
        "difference_ci_low": lower,
        "difference_ci_high": upper,
        "z_score": z_score,
        "p_value_unadjusted": max(0.0, min(1.0, p_value)),
    }


def holm_adjust(comparisons: Sequence[Mapping[str, Any]], *, alpha: float = 0.05) -> list[dict]:
    """Return Holm-adjusted p-values in the original comparison order."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not comparisons:
        raise ValueError("at least one comparison is required")
    ranked = sorted(enumerate(comparisons), key=lambda item: float(item[1]["p_value_unadjusted"]))
    adjusted_by_index: dict[int, float] = {}
    running_max = 0.0
    count = len(ranked)
    for rank, (index, comparison) in enumerate(ranked):
        raw = float(comparison["p_value_unadjusted"])
        if not 0 <= raw <= 1:
            raise ValueError("p-values must be between zero and one")
        adjusted = min(1.0, (count - rank) * raw)
        running_max = max(running_max, adjusted)
        adjusted_by_index[index] = running_max
    return [
        {
            **dict(comparison),
            "multiplicity_method": "holm",
            "family_size": count,
            "p_value_adjusted": adjusted_by_index[index],
            "reject_null_adjusted": adjusted_by_index[index] <= alpha,
        }
        for index, comparison in enumerate(comparisons)
    ]


def analyze_multiarm_binary(
    arms: Sequence[Mapping[str, Any]],
    *,
    reference_arm_id: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare every non-reference arm with one declared reference and apply Holm."""

    validated = {_validate_arm(arm)[0]: dict(arm) for arm in arms}
    if len(validated) != len(arms):
        raise ValueError("arm IDs must be unique")
    if reference_arm_id not in validated:
        raise ValueError("reference arm is absent")
    family_size = len(validated) - 1
    simultaneous_confidence_level = 1 - alpha / family_size
    comparisons = [
        compare_binary_arms(
            arm,
            validated[reference_arm_id],
            confidence_level=simultaneous_confidence_level,
        )
        for arm_id, arm in sorted(validated.items())
        if arm_id != reference_arm_id
    ]
    return {
        "reference_arm_id": reference_arm_id,
        "alpha": alpha,
        "multiplicity_method": "holm",
        "simultaneous_ci_method": "bonferroni_newcombe",
        "simultaneous_confidence_level": simultaneous_confidence_level,
        "comparison_count": len(comparisons),
        "comparisons": holm_adjust(comparisons, alpha=alpha),
    }


def minimum_sample_per_arm(
    baseline_rate: float,
    mde_absolute: float,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    family_size: int = 1,
) -> int:
    """Approximate equal-allocation sample size for a two-sided binary comparison."""

    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between zero and one")
    if mde_absolute <= 0 or baseline_rate + mde_absolute >= 1:
        raise ValueError("mde_absolute must be positive and keep the target rate below one")
    if not 0 < alpha < 1 or not 0 < target_power < 1:
        raise ValueError("alpha and target_power must be between zero and one")
    if family_size <= 0:
        raise ValueError("family_size must be positive")
    alternative_rate = baseline_rate + mde_absolute
    pooled_rate = (baseline_rate + alternative_rate) / 2
    z_alpha = NormalDist().inv_cdf(1 - alpha / (2 * family_size))
    z_power = NormalDist().inv_cdf(target_power)
    numerator = (
        z_alpha * math.sqrt(2 * pooled_rate * (1 - pooled_rate))
        + z_power
        * math.sqrt(baseline_rate * (1 - baseline_rate) + alternative_rate * (1 - alternative_rate))
    ) ** 2
    return math.ceil(numerator / mde_absolute**2)
