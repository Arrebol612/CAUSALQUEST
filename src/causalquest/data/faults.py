"""Deterministic fault injection with evaluator-only oracles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from .controlled_lab import ControlledExperimentData, audit_clean_experiment


class FaultType(str, Enum):
    """Fault families supported by the first controlled benchmark."""

    SAMPLE_RATIO_MISMATCH = "sample_ratio_mismatch"
    TELEMETRY_MISSINGNESS = "telemetry_missingness"
    DELAYED_EVENTS = "delayed_events"
    DUPLICATE_EXPOSURE = "duplicate_exposure"
    CROSS_GROUP_CONTAMINATION = "cross_group_contamination"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("optional text fields must be non-empty when provided")
    return value.strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_fraction(seed: int, *parts: str) -> float:
    payload = "|".join((str(seed), *parts)).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class FaultSpec:
    """One typed and reproducible fault recipe."""

    fault_id: str
    fault_type: FaultType
    seed: int
    fraction: float
    target_variant: str | None = None
    replacement_variant: str | None = None
    target_platform: str | None = None
    target_app_version: str | None = None
    target_event_name: str | None = None
    delay_hours: int = 0
    schema_version: str = "controlled-fault-v1"

    def __post_init__(self) -> None:
        if not self.fault_id.strip():
            raise ValueError("fault_id must not be empty")
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if self.fault_type is FaultType.SAMPLE_RATIO_MISMATCH and not self.target_variant:
            raise ValueError("sample_ratio_mismatch requires target_variant")
        if (
            self.fault_type
            in {
                FaultType.TELEMETRY_MISSINGNESS,
                FaultType.DELAYED_EVENTS,
            }
            and not self.target_event_name
        ):
            raise ValueError(f"{self.fault_type.value} requires target_event_name")
        if self.fault_type is FaultType.DELAYED_EVENTS and self.delay_hours <= 0:
            raise ValueError("delayed_events requires positive delay_hours")
        if self.fault_type is FaultType.CROSS_GROUP_CONTAMINATION:
            if not self.target_variant or not self.replacement_variant:
                raise ValueError("cross_group_contamination requires both variants")
            if self.target_variant == self.replacement_variant:
                raise ValueError("contamination variants must differ")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FaultSpec:
        """Parse one YAML/JSON-compatible fault recipe."""

        return cls(
            fault_id=str(payload.get("fault_id", "")).strip(),
            fault_type=FaultType(payload.get("fault_type")),
            seed=int(payload.get("seed", 0)),
            fraction=float(payload.get("fraction", 0)),
            target_variant=_optional_text(payload.get("target_variant")),
            replacement_variant=_optional_text(payload.get("replacement_variant")),
            target_platform=_optional_text(payload.get("target_platform")),
            target_app_version=_optional_text(payload.get("target_app_version")),
            target_event_name=_optional_text(payload.get("target_event_name")),
            delay_hours=int(payload.get("delay_hours", 0)),
            schema_version=str(payload.get("schema_version", "controlled-fault-v1")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "seed": self.seed,
            "fraction": self.fraction,
            "target_variant": self.target_variant,
            "replacement_variant": self.replacement_variant,
            "target_platform": self.target_platform,
            "target_app_version": self.target_app_version,
            "target_event_name": self.target_event_name,
            "delay_hours": self.delay_hours,
        }


@dataclass(frozen=True)
class FaultInjectionResult:
    """Faulted public data plus a separately serializable hidden oracle."""

    data: ControlledExperimentData
    parent_snapshot_id: str
    spec: FaultSpec
    affected_record_ids: tuple[str, ...]
    oracle_details: dict[str, Any]

    def public_manifest(self) -> dict[str, Any]:
        """Return provenance and row counts without revealing the fault label."""

        return {
            "schema_version": self.data.config.schema_version,
            "dataset": "controlled_experiment_lab",
            "data_layer": "controlled_lab",
            "experiment_id": self.data.config.experiment_id,
            "snapshot_id": self.data.snapshot_id,
            "available_relations": ["experiment_registry", "assignments", "exposures", "events"],
        }

    def oracle_manifest(self) -> dict[str, Any]:
        """Return evaluator-only labels and exact affected record IDs."""

        return {
            "schema_version": self.spec.schema_version,
            "evaluator_only": True,
            "data_layer": "controlled_fault",
            "snapshot_id": self.data.snapshot_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "fault": self.spec.to_dict(),
            "affected_count": len(self.affected_record_ids),
            "affected_record_ids": list(self.affected_record_ids),
            "details": self.oracle_details,
        }


def _selected(spec: FaultSpec, record_id: str) -> bool:
    return _stable_fraction(spec.seed, spec.fault_id, record_id) < spec.fraction


def _ensure_selection(eligible: list[str], selected: set[str]) -> set[str]:
    if eligible and not selected:
        selected.add(min(eligible))
    return selected


def _matching_exposure_units(
    data: ControlledExperimentData,
    spec: FaultSpec,
) -> set[str]:
    assignments = {row["unit_id"]: row for row in data.assignments}
    eligible: list[str] = []
    for exposure in data.exposures:
        assignment = assignments[exposure["unit_id"]]
        if spec.target_variant and exposure["variant_id"] != spec.target_variant:
            continue
        if spec.target_platform and assignment["platform"] != spec.target_platform:
            continue
        if spec.target_app_version and exposure["app_version"] != spec.target_app_version:
            continue
        eligible.append(exposure["unit_id"])
    return set(eligible)


def _snapshot(
    source: ControlledExperimentData,
    assignments: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> ControlledExperimentData:
    payload = {
        "config": source.config.to_dict(),
        "assignments": assignments,
        "exposures": exposures,
        "events": events,
    }
    snapshot_id = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return ControlledExperimentData(
        config=source.config,
        assignments=tuple(assignments),
        exposures=tuple(exposures),
        events=tuple(events),
        snapshot_id=snapshot_id,
    )


def inject_fault(data: ControlledExperimentData, spec: FaultSpec) -> FaultInjectionResult:
    """Return a faulted copy; never mutate the clean parent snapshot."""

    assignments = [dict(row) for row in data.assignments]
    exposures = [dict(row) for row in data.exposures]
    events = [dict(row) for row in data.events]
    affected: set[str] = set()
    details: dict[str, Any] = {}

    if spec.fault_type is FaultType.SAMPLE_RATIO_MISMATCH:
        eligible = [
            row["unit_id"] for row in assignments if row["variant_id"] == spec.target_variant
        ]
        affected = _ensure_selection(
            eligible, {unit_id for unit_id in eligible if _selected(spec, unit_id)}
        )
        before_counts = {
            variant: sum(row["variant_id"] == variant for row in assignments)
            for variant in (data.config.control_variant, data.config.treatment_variant)
        }
        assignments = [row for row in assignments if row["unit_id"] not in affected]
        exposures = [row for row in exposures if row["unit_id"] not in affected]
        events = [row for row in events if row["unit_id"] not in affected]
        after_counts = {
            variant: sum(row["variant_id"] == variant for row in assignments)
            for variant in (data.config.control_variant, data.config.treatment_variant)
        }
        details = {
            "mechanism": "assignment_log_loss_with_downstream_rows_removed",
            "before_variant_counts": before_counts,
            "after_variant_counts": after_counts,
        }

    elif spec.fault_type is FaultType.TELEMETRY_MISSINGNESS:
        eligible_units = _matching_exposure_units(data, spec)
        eligible = [
            row["event_id"]
            for row in events
            if row["unit_id"] in eligible_units and row["event_name"] == spec.target_event_name
        ]
        affected = _ensure_selection(
            eligible, {event_id for event_id in eligible if _selected(spec, event_id)}
        )
        events = [row for row in events if row["event_id"] not in affected]
        details = {
            "mechanism": "metric_observation_rows_dropped",
            "target_event_name": spec.target_event_name,
            "target_platform": spec.target_platform,
            "target_app_version": spec.target_app_version,
        }

    elif spec.fault_type is FaultType.DELAYED_EVENTS:
        eligible_units = _matching_exposure_units(data, spec)
        eligible = [
            row["event_id"]
            for row in events
            if row["unit_id"] in eligible_units and row["event_name"] == spec.target_event_name
        ]
        affected = _ensure_selection(
            eligible, {event_id for event_id in eligible if _selected(spec, event_id)}
        )
        for row in events:
            if row["event_id"] in affected:
                row["arrival_time"] = _iso(
                    _parse_utc(row["arrival_time"]) + timedelta(hours=spec.delay_hours)
                )
        details = {
            "mechanism": "event_arrival_timestamp_shift",
            "target_event_name": spec.target_event_name,
            "delay_hours": spec.delay_hours,
        }

    elif spec.fault_type is FaultType.DUPLICATE_EXPOSURE:
        eligible_units = _matching_exposure_units(data, spec)
        eligible = [row["exposure_id"] for row in exposures if row["unit_id"] in eligible_units]
        affected = _ensure_selection(
            eligible, {exposure_id for exposure_id in eligible if _selected(spec, exposure_id)}
        )
        source_events = {exposure_id: [] for exposure_id in affected}
        for event in events:
            if event["exposure_id"] in source_events:
                source_events[event["exposure_id"]].append(event)
        duplicates: list[dict[str, Any]] = []
        duplicate_events: list[dict[str, Any]] = []
        for exposure in exposures:
            if exposure["exposure_id"] not in affected:
                continue
            duplicate = dict(exposure)
            duplicate["exposure_id"] = f"{exposure['exposure_id']}__retry"
            duplicates.append(duplicate)
            for event in source_events[exposure["exposure_id"]]:
                duplicate_event = dict(event)
                duplicate_event["event_id"] = f"{event['event_id']}__retry"
                duplicate_event["exposure_id"] = duplicate["exposure_id"]
                duplicate_events.append(duplicate_event)
        exposures.extend(duplicates)
        events.extend(duplicate_events)
        details = {
            "mechanism": "client_retry_duplicates_exposure_and_metric_rows",
            "duplicate_exposure_rows": len(duplicates),
            "duplicate_event_rows": len(duplicate_events),
        }

    elif spec.fault_type is FaultType.CROSS_GROUP_CONTAMINATION:
        eligible_units = _matching_exposure_units(data, spec)
        eligible = [row["exposure_id"] for row in exposures if row["unit_id"] in eligible_units]
        affected = _ensure_selection(
            eligible, {exposure_id for exposure_id in eligible if _selected(spec, exposure_id)}
        )
        affected_units: set[str] = set()
        for exposure in exposures:
            if exposure["exposure_id"] in affected:
                exposure["variant_id"] = spec.replacement_variant
                affected_units.add(exposure["unit_id"])
        for event in events:
            if event["unit_id"] in affected_units:
                event["variant_id"] = spec.replacement_variant
        details = {
            "mechanism": "assigned_units_receive_opposite_variant",
            "assigned_variant": spec.target_variant,
            "received_variant": spec.replacement_variant,
        }

    else:  # pragma: no cover - Enum exhaustiveness guard
        raise ValueError(f"unsupported fault type: {spec.fault_type}")

    faulted = _snapshot(data, assignments, exposures, events)
    return FaultInjectionResult(
        data=faulted,
        parent_snapshot_id=data.snapshot_id,
        spec=spec,
        affected_record_ids=tuple(sorted(affected)),
        oracle_details=details,
    )


def verify_fault_injection(
    source: ControlledExperimentData,
    result: FaultInjectionResult,
) -> dict[str, Any]:
    """Verify that one fault recipe changed the expected relation without parent mutation."""

    audit = audit_clean_experiment(result.data)
    checks = {
        "parent_snapshot": result.parent_snapshot_id == source.snapshot_id,
        "snapshot_changed": result.data.snapshot_id != source.snapshot_id,
        "affected_nonempty": bool(result.affected_record_ids),
    }
    fault_type = result.spec.fault_type
    if fault_type is FaultType.SAMPLE_RATIO_MISMATCH:
        checks["signature"] = len(result.data.assignments) < len(source.assignments) and len(
            result.data.exposures
        ) <= len(source.exposures)
    elif fault_type is FaultType.TELEMETRY_MISSINGNESS:
        checks["signature"] = (
            len(result.data.events) < len(source.events)
            and "event_metric_cardinality_mismatch" in audit["issues"]
        )
    elif fault_type is FaultType.DELAYED_EVENTS:
        checks["signature"] = len(result.data.events) == len(source.events) and (
            "arrival_after_experiment_end" in audit["issues"]
        )
    elif fault_type is FaultType.DUPLICATE_EXPOSURE:
        checks["signature"] = len(result.data.exposures) > len(source.exposures) and (
            "duplicate_exposure_unit" in audit["issues"]
        )
    else:
        checks["signature"] = "assignment_exposure_variant_mismatch" in audit["issues"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "clean_audit_issues": audit["issues"],
    }
