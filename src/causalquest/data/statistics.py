"""Small, dependency-free statistical helpers used by case builders."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProportionEstimate:
    successes: int
    trials: int
    rate: float
    ci_low: float
    ci_high: float


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> ProportionEstimate:
    """Return a Wilson score interval for a binomial proportion."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    rate = successes / trials
    denominator = 1 + z**2 / trials
    centre = rate + z**2 / (2 * trials)
    radius = z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2))
    return ProportionEstimate(
        successes=successes,
        trials=trials,
        rate=rate,
        ci_low=(centre - radius) / denominator,
        ci_high=(centre + radius) / denominator,
    )


def two_proportion_difference(
    treatment_successes: int,
    treatment_trials: int,
    control_successes: int,
    control_trials: int,
    z: float = 1.959963984540054,
) -> dict[str, float]:
    """Compute an unpooled interval and a pooled two-sided z-test."""

    treatment = wilson_interval(treatment_successes, treatment_trials)
    control = wilson_interval(control_successes, control_trials)
    difference = treatment.rate - control.rate
    standard_error = math.sqrt(
        treatment.rate * (1 - treatment.rate) / treatment_trials
        + control.rate * (1 - control.rate) / control_trials
    )
    pooled = (treatment_successes + control_successes) / (treatment_trials + control_trials)
    pooled_se = math.sqrt(pooled * (1 - pooled) * (1 / treatment_trials + 1 / control_trials))
    z_score = difference / pooled_se if pooled_se else 0.0
    p_value = math.erfc(abs(z_score) / math.sqrt(2))
    relative_lift = difference / control.rate if control.rate else math.nan
    return {
        "absolute_lift": difference,
        "relative_lift": relative_lift,
        "ci_low": difference - z * standard_error,
        "ci_high": difference + z * standard_error,
        "z_score": z_score,
        "p_value": p_value,
    }
