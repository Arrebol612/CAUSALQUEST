"""Materialize Controlled Experiment Lab snapshots as auditable DuckDB files."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .controlled_lab import ControlledExperimentData, audit_clean_experiment


def _load_duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - CLI error path
        raise RuntimeError("duckdb is required; install the project dependencies first") from exc
    return duckdb


def _rows(rows: tuple[dict[str, Any], ...], columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
    return [tuple(row[column] for column in columns) for row in rows]


def build_controlled_lab_database(
    data: ControlledExperimentData,
    output_path: str | Path,
    *,
    allow_controlled_faults: bool = False,
    analysis_as_of_time: str | None = None,
) -> dict[str, Any]:
    """Write a logical snapshot to DuckDB, preserving deliberate benchmark faults."""

    audit = audit_clean_experiment(data)
    if not audit["passed"] and not allow_controlled_faults:
        raise ValueError(f"controlled lab audit failed: {audit['issues']}")

    duckdb = _load_duckdb()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    connection = duckdb.connect(str(output))
    connection.execute("SET TimeZone = 'UTC'")
    try:
        connection.execute(
            """
            CREATE TABLE experiment_registry (
                experiment_id VARCHAR PRIMARY KEY,
                template_id VARCHAR NOT NULL,
                schema_version VARCHAR NOT NULL,
                randomization_unit VARCHAR NOT NULL,
                analysis_unit VARCHAR NOT NULL,
                start_time_utc TIMESTAMPTZ NOT NULL,
                duration_days INTEGER NOT NULL,
                control_variant VARCHAR NOT NULL,
                treatment_variant VARCHAR NOT NULL,
                treatment_share DOUBLE NOT NULL,
                config_json JSON NOT NULL
            )
            """
        )
        config = data.config
        connection.execute(
            "INSERT INTO experiment_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                config.experiment_id,
                config.template_id,
                config.schema_version,
                config.randomization_unit,
                config.analysis_unit,
                config.start_time_utc,
                config.duration_days,
                config.control_variant,
                config.treatment_variant,
                config.treatment_share,
                json.dumps(config.to_dict(), sort_keys=True),
            ],
        )
        assignment_columns = (
            "assignment_id",
            "experiment_id",
            "unit_id",
            "variant_id",
            "assigned_at",
            "region",
            "platform",
            "acquisition_channel",
            "allocation_score",
        )
        connection.execute(
            """
            CREATE TABLE assignments (
                assignment_id VARCHAR PRIMARY KEY,
                experiment_id VARCHAR NOT NULL,
                unit_id VARCHAR NOT NULL,
                variant_id VARCHAR NOT NULL,
                assigned_at TIMESTAMPTZ NOT NULL,
                region VARCHAR NOT NULL,
                platform VARCHAR NOT NULL,
                acquisition_channel VARCHAR NOT NULL,
                allocation_score DOUBLE NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _rows(data.assignments, assignment_columns),
        )
        connection.execute("CREATE UNIQUE INDEX assignments_unit ON assignments(unit_id)")

        exposure_columns = (
            "exposure_id",
            "experiment_id",
            "unit_id",
            "variant_id",
            "session_id",
            "exposed_at",
            "surface",
            "app_version",
        )
        connection.execute(
            """
            CREATE TABLE exposures (
                exposure_id VARCHAR PRIMARY KEY,
                experiment_id VARCHAR NOT NULL,
                unit_id VARCHAR NOT NULL,
                variant_id VARCHAR NOT NULL,
                session_id VARCHAR NOT NULL,
                exposed_at TIMESTAMPTZ NOT NULL,
                surface VARCHAR NOT NULL,
                app_version VARCHAR NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO exposures VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            _rows(data.exposures, exposure_columns),
        )
        exposure_index = "CREATE INDEX" if allow_controlled_faults else "CREATE UNIQUE INDEX"
        connection.execute(f"{exposure_index} exposures_unit ON exposures(unit_id)")

        event_columns = (
            "event_id",
            "experiment_id",
            "exposure_id",
            "unit_id",
            "variant_id",
            "event_name",
            "event_value",
            "event_time",
            "arrival_time",
        )
        connection.execute(
            """
            CREATE TABLE events (
                event_id VARCHAR PRIMARY KEY,
                experiment_id VARCHAR NOT NULL,
                exposure_id VARCHAR NOT NULL,
                unit_id VARCHAR NOT NULL,
                variant_id VARCHAR NOT NULL,
                event_name VARCHAR NOT NULL,
                event_value DOUBLE NOT NULL,
                event_time TIMESTAMPTZ NOT NULL,
                arrival_time TIMESTAMPTZ NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _rows(data.events, event_columns),
        )
        connection.execute("CREATE INDEX events_exposure ON events(exposure_id)")
        connection.execute("CREATE INDEX events_name_variant ON events(event_name, variant_id)")

        if analysis_as_of_time is None:
            start = datetime.fromisoformat(config.start_time_utc.replace("Z", "+00:00"))
            analysis_as_of_time = (
                start.astimezone(timezone.utc)
                + timedelta(days=config.duration_days, hours=24 * 365)
            ).isoformat()
        connection.execute(
            """
            CREATE TABLE case_runtime (
                analysis_as_of_time TIMESTAMPTZ NOT NULL
            )
            """
        )
        connection.execute("INSERT INTO case_runtime VALUES (?)", [analysis_as_of_time])
        connection.execute(
            """
            CREATE VIEW events_visible AS
            SELECT e.*
            FROM events e
            CROSS JOIN case_runtime r
            WHERE e.arrival_time <= r.analysis_as_of_time
            """
        )
        connection.execute(
            """
            CREATE TABLE metric_catalog AS
            SELECT * FROM (VALUES (?, ?, 'exposed_unit', 'increase'),
                                  (?, ?, 'exposed_unit', 'increase'),
                                  (?, ?, 'exposed_unit', 'decrease'),
                                  (?, ?, 'exposed_unit', 'decrease'))
            AS t(metric_id, event_name, unit, direction)
            """
            ,
            [
                config.primary_metric_id,
                config.primary_event_name,
                config.conversion_metric_id,
                config.conversion_event_name,
                config.reliability_metric_id,
                config.reliability_event_name,
                config.latency_metric_id,
                config.latency_event_name,
            ],
        )
        connection.execute(
            f"""
            CREATE VIEW variant_metrics AS
            SELECT
                variant_id,
                COUNT(DISTINCT exposure_id) AS exposed_units,
                AVG(event_value) FILTER (WHERE event_name = '{config.primary_event_name}') AS primary_rate,
                AVG(event_value) FILTER (WHERE event_name = '{config.conversion_event_name}')
                    AS conversion_rate,
                AVG(event_value) FILTER (WHERE event_name = '{config.reliability_event_name}') AS reliability_rate,
                AVG(event_value) FILTER (WHERE event_name = '{config.latency_event_name}') AS mean_latency_ms
            FROM events_visible
            GROUP BY variant_id
            """
        )
        connection.execute(
            """
            CREATE TABLE snapshot_metadata (
                snapshot_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                generation_method VARCHAR NOT NULL,
                assignment_rows BIGINT NOT NULL,
                exposure_rows BIGINT NOT NULL,
                event_rows BIGINT NOT NULL,
                analysis_as_of_time TIMESTAMPTZ NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO snapshot_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                data.snapshot_id,
                config.schema_version,
                "deterministic_sha256_draws",
                len(data.assignments),
                len(data.exposures),
                len(data.events),
                analysis_as_of_time,
            ],
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

    return {
        "dataset": "controlled_experiment_lab",
        "data_layer": "controlled_fault" if allow_controlled_faults else "controlled_lab",
        "schema_version": data.config.schema_version,
        "experiment_id": data.config.experiment_id,
        "template_id": data.config.template_id,
        "seed": data.config.seed,
        "generation_method": "deterministic_sha256_draws",
        "snapshot_id": data.snapshot_id,
        "assignment_rows": len(data.assignments),
        "exposure_rows": len(data.exposures),
        "event_rows": len(data.events),
        "database_path": output.as_posix(),
        "database_bytes": output.stat().st_size,
        "audit": audit,
        "controlled_faults_allowed": allow_controlled_faults,
        "analysis_as_of_time": analysis_as_of_time,
        "passed": True,
    }
