"""Trusted deterministic analyses shared by tools and hidden oracle builders."""

from __future__ import annotations

from datetime import date
from typing import Any

from .statistics import two_proportion_difference, wilson_interval


def policy_window_summary(
    connection, campaign: str, start: date | str, end: date | str
) -> dict[str, dict[str, Any]]:
    """Aggregate policy logging and click metrics in an inclusive UTC date window."""

    rows = connection.execute(
        """
        SELECT
            behavior_policy,
            COUNT(*) AS impressions,
            SUM(reward_click) AS clicks,
            MIN(propensity) AS propensity_min,
            MAX(propensity) AS propensity_max,
            SUM(CASE WHEN propensity < 0.001 THEN 1 ELSE 0 END) AS very_low_propensity,
            COUNT(DISTINCT position) AS position_count
        FROM exposures
        WHERE campaign = ? AND event_date_utc BETWEEN ? AND ?
        GROUP BY behavior_policy
        ORDER BY behavior_policy
        """,
        [campaign, start, end],
    ).fetchall()
    return {
        policy: {
            "impressions": int(impressions),
            "clicks": int(clicks),
            "propensity_min": float(propensity_min),
            "propensity_max": float(propensity_max),
            "very_low_propensity": int(very_low),
            "position_count": int(position_count),
        }
        for policy, impressions, clicks, propensity_min, propensity_max, very_low, position_count in rows
    }


def compare_observed_ctr(policy_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare observed BTS and Random CTR without claiming user-level causality."""

    random = policy_rows["random"]
    bts = policy_rows["bts"]
    random_estimate = wilson_interval(random["clicks"], random["impressions"])
    bts_estimate = wilson_interval(bts["clicks"], bts["impressions"])
    difference = two_proportion_difference(
        bts["clicks"], bts["impressions"], random["clicks"], random["impressions"]
    )
    if min(random["impressions"], bts["impressions"]) < 500:
        recommendation = "insufficient_sample"
    elif difference["p_value"] >= 0.05:
        recommendation = "inconclusive"
    elif difference["absolute_lift"] > 0:
        recommendation = "bts_observed_ctr_higher"
    else:
        recommendation = "random_observed_ctr_higher"
    return {
        "recommendation": recommendation,
        "metrics": {
            "random_ctr": random_estimate.rate,
            "bts_ctr": bts_estimate.rate,
            "absolute_lift": difference["absolute_lift"],
            "p_value": difference["p_value"],
        },
        "intervals": {
            "random_ctr": [random_estimate.ci_low, random_estimate.ci_high],
            "bts_ctr": [bts_estimate.ci_low, bts_estimate.ci_high],
            "absolute_lift": [difference["ci_low"], difference["ci_high"]],
        },
        "interpretation_limit": "observed policy comparison; not user-level causal A/B or direct OPE",
    }


def assess_logging_health(policy_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the first deterministic logging gate."""

    invalid = any(
        row["propensity_min"] <= 0 or row["propensity_max"] > 1 or row["position_count"] != 3
        for row in policy_rows.values()
    )
    too_small = any(row["impressions"] < 500 for row in policy_rows.values())
    recommendation = (
        "block_invalid_logging"
        if invalid
        else "request_more_data"
        if too_small
        else "logging_checks_pass"
    )
    return {
        "recommendation": recommendation,
        "metrics": {
            "random_propensity_min": policy_rows["random"]["propensity_min"],
            "bts_propensity_min": policy_rows["bts"]["propensity_min"],
            "random_impressions": policy_rows["random"]["impressions"],
            "bts_impressions": policy_rows["bts"]["impressions"],
        },
        "diagnostics": policy_rows,
    }
