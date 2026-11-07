"""Deterministic, auditable health checks for running controlled experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any

BALANCE_DIMENSIONS = ("platform", "region", "acquisition_channel")
SLICE_DIMENSIONS = (*BALANCE_DIMENSIONS, "app_version")


def _experiment(connection) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT experiment_id, control_variant, treatment_variant, treatment_share,
               start_time_utc, duration_days, config_json
        FROM experiment_registry
        """
    ).fetchone()
    if row is None:
        raise ValueError("experiment_registry is empty")
    config = json.loads(row[6]) if isinstance(row[6], str) else row[6]
    return {
        "experiment_id": row[0],
        "control_variant": row[1],
        "treatment_variant": row[2],
        "treatment_share": float(row[3]),
        "start_time_utc": row[4].isoformat(),
        "duration_days": int(row[5]),
        "config": config,
    }


def check_sample_ratio(connection, *, alpha: float = 0.001) -> dict[str, Any]:
    """Run a two-cell chi-square allocation check against the registered split."""

    experiment = _experiment(connection)
    counts = dict(
        connection.execute(
            "SELECT variant_id, COUNT(*) FROM assignments GROUP BY variant_id"
        ).fetchall()
    )
    control = experiment["control_variant"]
    treatment = experiment["treatment_variant"]
    total = int(counts.get(control, 0) + counts.get(treatment, 0))
    if total == 0:
        return {"status": "insufficient_data", "reason": "no_assignments"}
    expected_treatment = total * experiment["treatment_share"]
    expected_control = total - expected_treatment
    if min(expected_control, expected_treatment) < 5:
        return {
            "status": "insufficient_data",
            "reason": "expected_cell_count_below_5",
            "observed": {control: counts.get(control, 0), treatment: counts.get(treatment, 0)},
        }
    chi_square = sum(
        (observed - expected) ** 2 / expected
        for observed, expected in (
            (counts.get(control, 0), expected_control),
            (counts.get(treatment, 0), expected_treatment),
        )
    )
    p_value = math.erfc(math.sqrt(chi_square / 2))
    return {
        "status": "mismatch" if p_value < alpha else "within_expectation",
        "observed": {control: counts.get(control, 0), treatment: counts.get(treatment, 0)},
        "expected": {control: expected_control, treatment: expected_treatment},
        "observed_treatment_share": counts.get(treatment, 0) / total,
        "registered_treatment_share": experiment["treatment_share"],
        "chi_square": chi_square,
        "p_value": p_value,
        "alpha": alpha,
        "assumptions": ["registered_allocation_is_correct", "independent_assignment_units"],
    }


def check_pre_experiment_balance(
    connection,
    *,
    dimensions: Iterable[str] = BALANCE_DIMENSIONS,
    max_share_gap: float = 0.05,
) -> dict[str, Any]:
    """Compare pre-treatment categorical distributions between experiment arms."""

    requested = tuple(dimensions)
    unknown = sorted(set(requested).difference(BALANCE_DIMENSIONS))
    if unknown:
        raise ValueError(f"unsupported balance dimension(s): {', '.join(unknown)}")
    experiment = _experiment(connection)
    variants = (experiment["control_variant"], experiment["treatment_variant"])
    results: list[dict[str, Any]] = []
    for dimension in requested:
        rows = connection.execute(
            f"""
            SELECT variant_id, {dimension}, COUNT(*) AS units
            FROM assignments
            GROUP BY variant_id, {dimension}
            ORDER BY {dimension}, variant_id
            """
        ).fetchall()
        totals = {variant: 0 for variant in variants}
        categories: dict[str, dict[str, int]] = {}
        for variant, category, units in rows:
            totals[variant] = totals.get(variant, 0) + int(units)
            categories.setdefault(str(category), {})[variant] = int(units)
        category_results = []
        max_gap = 0.0
        for category, category_counts in sorted(categories.items()):
            shares = {
                variant: category_counts.get(variant, 0) / totals[variant]
                if totals.get(variant, 0)
                else 0.0
                for variant in variants
            }
            gap = abs(shares[variants[0]] - shares[variants[1]])
            max_gap = max(max_gap, gap)
            category_results.append(
                {"category": category, "counts": category_counts, "shares": shares, "gap": gap}
            )
        results.append(
            {
                "dimension": dimension,
                "status": "imbalanced" if max_gap > max_share_gap else "balanced",
                "max_share_gap": max_gap,
                "categories": category_results,
            }
        )
    return {
        "status": "imbalanced"
        if any(item["status"] == "imbalanced" for item in results)
        else "balanced",
        "max_share_gap_threshold": max_share_gap,
        "dimensions": results,
        "boundary": "diagnostic_only_not_stratified_randomization_or_causal_adjustment",
    }


def check_assignment_exposure(connection) -> dict[str, Any]:
    """Check assignment, exposure, duplication, orphaning, and received-variant consistency."""

    rows = connection.execute(
        """
        SELECT
            a.variant_id,
            COUNT(DISTINCT a.unit_id) AS assigned_units,
            COUNT(DISTINCT e.unit_id) AS exposed_units
        FROM assignments a
        LEFT JOIN exposures e USING (experiment_id, unit_id)
        GROUP BY a.variant_id
        ORDER BY a.variant_id
        """
    ).fetchall()
    by_variant = [
        {
            "variant_id": row[0],
            "assigned_units": int(row[1]),
            "exposed_units": int(row[2]),
            "exposure_rate": row[2] / row[1] if row[1] else None,
        }
        for row in rows
    ]
    duplicate_units = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT experiment_id, unit_id
                FROM exposures
                GROUP BY experiment_id, unit_id
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    contamination_rows = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM exposures e
            JOIN assignments a USING (experiment_id, unit_id)
            WHERE e.variant_id <> a.variant_id
            """
        ).fetchone()[0]
    )
    orphan_exposures = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM exposures e
            LEFT JOIN assignments a USING (experiment_id, unit_id)
            WHERE a.unit_id IS NULL
            """
        ).fetchone()[0]
    )
    status = "healthy"
    if contamination_rows or orphan_exposures:
        status = "invalid_assignment_exposure_link"
    elif duplicate_units:
        status = "duplicate_exposure"
    return {
        "status": status,
        "by_variant": by_variant,
        "duplicate_exposure_units": duplicate_units,
        "cross_group_exposure_rows": contamination_rows,
        "orphan_exposure_rows": orphan_exposures,
    }


def _event_name(connection, metric_id: str) -> str:
    row = connection.execute(
        "SELECT event_name FROM metric_catalog WHERE metric_id = ?", [metric_id]
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown metric: {metric_id}")
    return str(row[0])


def check_metric_completeness(
    connection,
    metric_id: str,
    *,
    minimum_completeness: float = 0.95,
) -> dict[str, Any]:
    """Measure expected one-row-per-exposure metric coverage across operational slices."""

    event_name = _event_name(connection, metric_id)
    rows = connection.execute(
        """
        SELECT e.variant_id, a.platform, e.app_version,
               COUNT(DISTINCT e.exposure_id) AS expected_rows,
               COUNT(v.event_id) AS observed_rows
        FROM exposures e
        JOIN assignments a USING (experiment_id, unit_id)
        LEFT JOIN events_visible v
          ON v.experiment_id = e.experiment_id
         AND v.exposure_id = e.exposure_id
         AND v.event_name = ?
        GROUP BY e.variant_id, a.platform, e.app_version
        ORDER BY e.variant_id, a.platform, e.app_version
        """,
        [event_name],
    ).fetchall()
    slices = []
    for variant, platform, app_version, expected, observed in rows:
        completeness = observed / expected if expected else None
        slices.append(
            {
                "variant_id": variant,
                "platform": platform,
                "app_version": app_version,
                "expected_rows": int(expected),
                "observed_rows": int(observed),
                "completeness": completeness,
                "status": "incomplete"
                if completeness is not None and completeness < minimum_completeness
                else "complete",
            }
        )
    incomplete = [item for item in slices if item["status"] == "incomplete"]
    return {
        "status": "incomplete" if incomplete else "complete",
        "metric_id": metric_id,
        "event_name": event_name,
        "minimum_completeness": minimum_completeness,
        "slices": slices,
        "incomplete_slices": incomplete,
        "normalization_contract": "one_metric_observation_per_exposure",
    }


def check_event_timeliness(
    connection,
    metric_id: str,
    *,
    expected_arrival_sla_minutes: int = 15,
) -> dict[str, Any]:
    """Check visible event-time validity and arrival lag against an explicit as-of cutoff."""

    if expected_arrival_sla_minutes <= 0:
        raise ValueError("expected_arrival_sla_minutes must be positive")

    event_name = _event_name(connection, metric_id)
    rows = connection.execute(
        """
        WITH registry AS (
            SELECT start_time_utc + duration_days * INTERVAL '1 day' AS experiment_end,
                   analysis_as_of_time
            FROM experiment_registry
            CROSS JOIN case_runtime
        )
        SELECT v.variant_id, a.platform,
               COUNT(*) AS observed_rows,
               COUNT(*) FILTER (
                   WHERE v.arrival_time - v.event_time > ? * INTERVAL '1 minute'
               ) AS late_arrival_rows,
               COUNT(*) FILTER (
                   WHERE v.event_time > r.experiment_end
               ) AS outside_outcome_window_rows,
               MAX(v.arrival_time - v.event_time) AS maximum_arrival_lag,
               MAX(v.event_time - e.exposed_at) AS maximum_outcome_lag
        FROM events_visible v
        JOIN exposures e USING (experiment_id, exposure_id, unit_id)
        JOIN assignments a USING (experiment_id, unit_id)
        CROSS JOIN registry r
        WHERE v.event_name = ?
        GROUP BY v.variant_id, a.platform
        ORDER BY v.variant_id, a.platform
        """,
        [expected_arrival_sla_minutes, event_name],
    ).fetchall()
    slices = [
        {
            "variant_id": row[0],
            "platform": row[1],
            "observed_rows": int(row[2]),
            "late_arrival_rows": int(row[3]),
            "late_arrival_rate": row[3] / row[2] if row[2] else None,
            "outside_outcome_window_rows": int(row[4]),
            "maximum_arrival_lag": str(row[5]),
            "maximum_outcome_lag": str(row[6]),
        }
        for row in rows
    ]
    late_arrival_rows = sum(item["late_arrival_rows"] for item in slices)
    outside_outcome_window_rows = sum(item["outside_outcome_window_rows"] for item in slices)
    analysis_as_of_time = connection.execute(
        "SELECT analysis_as_of_time FROM case_runtime"
    ).fetchone()[0]
    if outside_outcome_window_rows:
        status = "events_outside_outcome_window"
    elif late_arrival_rows:
        status = "late_arrivals_detected"
    else:
        status = "arrival_within_sla"
    return {
        "status": status,
        "metric_id": metric_id,
        "event_name": event_name,
        "analysis_as_of_time": analysis_as_of_time.isoformat(),
        "expected_arrival_sla_minutes": expected_arrival_sla_minutes,
        "late_arrival_rows": late_arrival_rows,
        "outside_outcome_window_rows": outside_outcome_window_rows,
        "slices": slices,
        "boundary": "arrival_time_is_collector_ingestion_time_not_warehouse_etl_completion",
    }


def analyze_slice_health(connection, dimension: str, metric_id: str) -> dict[str, Any]:
    """Compare allocation, exposure, and metric coverage for a pre-approved slice dimension."""

    if dimension not in SLICE_DIMENSIONS:
        raise ValueError(f"unsupported slice dimension: {dimension}")
    event_name = _event_name(connection, metric_id)
    if dimension == "app_version":
        dimension_sql = "e.app_version"
    else:
        dimension_sql = f"a.{dimension}"
    rows = connection.execute(
        f"""
        SELECT a.variant_id, {dimension_sql} AS slice_value,
               COUNT(DISTINCT a.unit_id) AS assigned_units,
               COUNT(DISTINCT e.exposure_id) AS exposed_units,
               COUNT(v.event_id) AS metric_rows
        FROM assignments a
        LEFT JOIN exposures e USING (experiment_id, unit_id)
        LEFT JOIN events_visible v
          ON v.experiment_id = e.experiment_id
         AND v.exposure_id = e.exposure_id
         AND v.event_name = ?
        GROUP BY a.variant_id, slice_value
        ORDER BY slice_value, a.variant_id
        """,
        [event_name],
    ).fetchall()
    return {
        "dimension": dimension,
        "metric_id": metric_id,
        "rows": [
            {
                "variant_id": row[0],
                "slice_value": row[1],
                "assigned_units": int(row[2]),
                "exposed_units": int(row[3]),
                "exposure_rate": row[3] / row[2] if row[2] else None,
                "metric_rows": int(row[4]),
                "metric_completeness": row[4] / row[3] if row[3] else None,
            }
            for row in rows
        ],
        "boundary": "diagnostic_slice_not_confirmatory_heterogeneous_effect_inference",
    }
