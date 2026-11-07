"""Deterministic clean experiment generator for the Controlled Experiment Lab."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

SCHEMA_VERSION = "controlled-lab-v1"
REQUIRED_EVENT_NAMES = (
    "click_observation",
    "conversion_observation",
    "crash_observation",
    "latency_ms",
)


def _required_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _probability(name: str, value: Any, *, allow_one: bool = False) -> float:
    parsed = float(value)
    upper_valid = parsed <= 1 if allow_one else parsed < 1
    if parsed < 0 or not upper_valid:
        boundary = "[0, 1]" if allow_one else "[0, 1)"
        raise ValueError(f"{name} must be in {boundary}")
    return parsed


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("start_time_utc must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_fraction(seed: int, *parts: str) -> float:
    payload = "|".join((str(seed), *parts)).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def _stable_choice(options: tuple[str, ...], seed: int, *parts: str) -> str:
    index = min(int(_stable_fraction(seed, *parts) * len(options)), len(options) - 1)
    return options[index]


def _stable_normal(seed: int, *parts: str) -> float:
    first = max(_stable_fraction(seed, *parts, "normal_u1"), 1e-12)
    second = _stable_fraction(seed, *parts, "normal_u2")
    return math.sqrt(-2 * math.log(first)) * math.cos(2 * math.pi * second)


@dataclass(frozen=True)
class ControlledLabConfig:
    """Configuration for one clean, synthetic internet-product experiment."""

    experiment_id: str
    template_id: str
    seed: int
    population_size: int
    start_time_utc: str
    duration_days: int
    randomization_unit: str
    analysis_unit: str
    control_variant: str
    treatment_variant: str
    treatment_share: float
    exposure_rate: float
    baseline_ctr: float
    treatment_ctr_absolute_lift: float
    baseline_conversion_rate: float
    treatment_conversion_absolute_lift: float
    baseline_crash_rate: float
    treatment_crash_absolute_lift: float
    baseline_latency_ms: float
    treatment_latency_delta_ms: float
    surface: str = "home_feed"
    primary_metric_id: str = "click_through_rate"
    conversion_metric_id: str = "conversion_rate"
    reliability_metric_id: str = "crash_rate"
    latency_metric_id: str = "mean_latency_ms"
    primary_event_name: str = "click_observation"
    conversion_event_name: str = "conversion_observation"
    reliability_event_name: str = "crash_observation"
    latency_event_name: str = "latency_ms"
    max_arrival_delay_seconds: int = 300
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "template_id",
            "start_time_utc",
            "randomization_unit",
            "analysis_unit",
            "control_variant",
            "treatment_variant",
            "schema_version",
            "surface",
            "primary_metric_id",
            "conversion_metric_id",
            "reliability_metric_id",
            "latency_metric_id",
            "primary_event_name",
            "conversion_event_name",
            "reliability_event_name",
            "latency_event_name",
        ):
            _required_text(name, getattr(self, name))
        _parse_utc(self.start_time_utc)
        if self.population_size < 100:
            raise ValueError("population_size must be at least 100")
        if self.duration_days <= 0:
            raise ValueError("duration_days must be positive")
        if self.randomization_unit != self.analysis_unit:
            raise ValueError("clean v1 requires matching randomization and analysis units")
        if self.control_variant == self.treatment_variant:
            raise ValueError("control and treatment variants must differ")
        _probability("treatment_share", self.treatment_share)
        if self.treatment_share == 0:
            raise ValueError("treatment_share must be greater than zero")
        _probability("exposure_rate", self.exposure_rate, allow_one=True)
        for name in ("baseline_ctr", "baseline_conversion_rate", "baseline_crash_rate"):
            _probability(name, getattr(self, name), allow_one=True)
        for baseline_name, lift_name in (
            ("baseline_ctr", "treatment_ctr_absolute_lift"),
            ("baseline_conversion_rate", "treatment_conversion_absolute_lift"),
            ("baseline_crash_rate", "treatment_crash_absolute_lift"),
        ):
            treatment_rate = getattr(self, baseline_name) + getattr(self, lift_name)
            _probability(f"{baseline_name} + {lift_name}", treatment_rate, allow_one=True)
        if self.baseline_latency_ms <= 0:
            raise ValueError("baseline_latency_ms must be positive")
        if self.baseline_latency_ms + self.treatment_latency_delta_ms <= 0:
            raise ValueError("treatment latency must be positive")
        if self.max_arrival_delay_seconds < 0:
            raise ValueError("max_arrival_delay_seconds must not be negative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ControlledLabConfig:
        """Parse a YAML/JSON-compatible mapping."""

        return cls(
            experiment_id=_required_text("experiment_id", payload.get("experiment_id")),
            template_id=_required_text("template_id", payload.get("template_id")),
            seed=int(payload.get("seed", 0)),
            population_size=int(payload.get("population_size", 0)),
            start_time_utc=_required_text("start_time_utc", payload.get("start_time_utc")),
            duration_days=int(payload.get("duration_days", 0)),
            randomization_unit=_required_text(
                "randomization_unit", payload.get("randomization_unit")
            ),
            analysis_unit=_required_text("analysis_unit", payload.get("analysis_unit")),
            control_variant=_required_text("control_variant", payload.get("control_variant")),
            treatment_variant=_required_text("treatment_variant", payload.get("treatment_variant")),
            treatment_share=float(payload.get("treatment_share", 0)),
            exposure_rate=float(payload.get("exposure_rate", 0)),
            baseline_ctr=float(payload.get("baseline_ctr", 0)),
            treatment_ctr_absolute_lift=float(payload.get("treatment_ctr_absolute_lift", 0)),
            baseline_conversion_rate=float(payload.get("baseline_conversion_rate", 0)),
            treatment_conversion_absolute_lift=float(
                payload.get("treatment_conversion_absolute_lift", 0)
            ),
            baseline_crash_rate=float(payload.get("baseline_crash_rate", 0)),
            treatment_crash_absolute_lift=float(payload.get("treatment_crash_absolute_lift", 0)),
            baseline_latency_ms=float(payload.get("baseline_latency_ms", 0)),
            treatment_latency_delta_ms=float(payload.get("treatment_latency_delta_ms", 0)),
            surface=_required_text("surface", payload.get("surface", "home_feed")),
            primary_metric_id=_required_text(
                "primary_metric_id", payload.get("primary_metric_id", "click_through_rate")
            ),
            conversion_metric_id=_required_text(
                "conversion_metric_id", payload.get("conversion_metric_id", "conversion_rate")
            ),
            reliability_metric_id=_required_text(
                "reliability_metric_id", payload.get("reliability_metric_id", "crash_rate")
            ),
            latency_metric_id=_required_text(
                "latency_metric_id", payload.get("latency_metric_id", "mean_latency_ms")
            ),
            primary_event_name=_required_text(
                "primary_event_name", payload.get("primary_event_name", "click_observation")
            ),
            conversion_event_name=_required_text(
                "conversion_event_name",
                payload.get("conversion_event_name", "conversion_observation"),
            ),
            reliability_event_name=_required_text(
                "reliability_event_name", payload.get("reliability_event_name", "crash_observation")
            ),
            latency_event_name=_required_text(
                "latency_event_name", payload.get("latency_event_name", "latency_ms")
            ),
            max_arrival_delay_seconds=int(payload.get("max_arrival_delay_seconds", 300)),
            schema_version=_required_text(
                "schema_version", payload.get("schema_version", SCHEMA_VERSION)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ControlledExperimentData:
    """Logical snapshot produced by the clean generator."""

    config: ControlledLabConfig
    assignments: tuple[dict[str, Any], ...]
    exposures: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    snapshot_id: str


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _event_row(
    *,
    event_id: str,
    experiment_id: str,
    exposure_id: str,
    unit_id: str,
    variant_id: str,
    event_name: str,
    event_value: float,
    event_time: datetime,
    arrival_time: datetime,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "experiment_id": experiment_id,
        "exposure_id": exposure_id,
        "unit_id": unit_id,
        "variant_id": variant_id,
        "event_name": event_name,
        "event_value": event_value,
        "event_time": _iso(event_time),
        "arrival_time": _iso(arrival_time),
    }


def generate_clean_experiment(config: ControlledLabConfig) -> ControlledExperimentData:
    """Generate a clean snapshot with deterministic hash-based pseudo-random draws."""

    start = _parse_utc(config.start_time_utc)
    duration_seconds = config.duration_days * 24 * 60 * 60
    assignment_window = min(6 * 60 * 60, duration_seconds // 10)
    exposure_window = max(duration_seconds - assignment_window - 7 * 60 * 60, 1)

    assignments: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for index in range(config.population_size):
        unit_id = f"usr_{index:07d}"
        assignment_score = _stable_fraction(config.seed, unit_id, "assignment")
        variant_id = (
            config.treatment_variant
            if assignment_score < config.treatment_share
            else config.control_variant
        )
        assigned_at = start + timedelta(
            seconds=int(_stable_fraction(config.seed, unit_id, "assigned_at") * assignment_window)
        )
        platform = _stable_choice(("android", "ios"), config.seed, unit_id, "platform")
        region = _stable_choice(("east", "south", "north", "west"), config.seed, unit_id, "region")
        channel = _stable_choice(("organic", "paid", "referral"), config.seed, unit_id, "channel")
        assignments.append(
            {
                "assignment_id": f"asn_{config.experiment_id}_{index:07d}",
                "experiment_id": config.experiment_id,
                "unit_id": unit_id,
                "variant_id": variant_id,
                "assigned_at": _iso(assigned_at),
                "region": region,
                "platform": platform,
                "acquisition_channel": channel,
                "allocation_score": assignment_score,
            }
        )

        if _stable_fraction(config.seed, unit_id, "exposure") >= config.exposure_rate:
            continue

        exposed_at = assigned_at + timedelta(
            seconds=int(_stable_fraction(config.seed, unit_id, "exposed_at") * exposure_window)
        )
        exposure_id = f"exp_{config.experiment_id}_{index:07d}"
        app_version = (
            _stable_choice(("9.3.0", "9.4.0"), config.seed, unit_id, "app_version")
            if platform == "android"
            else _stable_choice(("8.8.0", "8.9.0"), config.seed, unit_id, "app_version")
        )
        exposures.append(
            {
                "exposure_id": exposure_id,
                "experiment_id": config.experiment_id,
                "unit_id": unit_id,
                "variant_id": variant_id,
                "session_id": f"ses_{config.experiment_id}_{index:07d}",
                "exposed_at": _iso(exposed_at),
                "surface": config.surface,
                "app_version": app_version,
            }
        )

        is_treatment = variant_id == config.treatment_variant
        ctr = config.baseline_ctr + (config.treatment_ctr_absolute_lift if is_treatment else 0)
        conversion_rate = config.baseline_conversion_rate + (
            config.treatment_conversion_absolute_lift if is_treatment else 0
        )
        crash_rate = config.baseline_crash_rate + (
            config.treatment_crash_absolute_lift if is_treatment else 0
        )
        latency_mean = config.baseline_latency_ms + (
            config.treatment_latency_delta_ms if is_treatment else 0
        )
        outcomes = {
            config.primary_event_name: float(
                _stable_fraction(config.seed, unit_id, "primary_outcome") < ctr
            ),
            config.conversion_event_name: float(
                _stable_fraction(config.seed, unit_id, "conversion") < conversion_rate
            ),
            config.reliability_event_name: float(
                _stable_fraction(config.seed, unit_id, "crash") < crash_rate
            ),
            config.latency_event_name: round(
                max(1.0, latency_mean + 35 * _stable_normal(config.seed, unit_id, "latency")),
                3,
            ),
        }
        event_names = (
            config.primary_event_name,
            config.conversion_event_name,
            config.reliability_event_name,
            config.latency_event_name,
        )
        for event_index, event_name in enumerate(event_names, start=1):
            event_time = exposed_at + timedelta(
                seconds=event_index
                + int(_stable_fraction(config.seed, unit_id, event_name, "event_time") * 21_600)
            )
            arrival_time = event_time + timedelta(
                seconds=int(
                    _stable_fraction(config.seed, unit_id, event_name, "arrival_time")
                    * (config.max_arrival_delay_seconds + 1)
                )
            )
            events.append(
                _event_row(
                    event_id=f"evt_{config.experiment_id}_{index:07d}_{event_index}",
                    experiment_id=config.experiment_id,
                    exposure_id=exposure_id,
                    unit_id=unit_id,
                    variant_id=variant_id,
                    event_name=event_name,
                    event_value=outcomes[event_name],
                    event_time=event_time,
                    arrival_time=arrival_time,
                )
            )

    logical_payload = {
        "config": config.to_dict(),
        "assignments": assignments,
        "exposures": exposures,
        "events": events,
    }
    snapshot_id = hashlib.sha256(_canonical_json(logical_payload).encode("utf-8")).hexdigest()
    return ControlledExperimentData(
        config=config,
        assignments=tuple(assignments),
        exposures=tuple(exposures),
        events=tuple(events),
        snapshot_id=snapshot_id,
    )


def audit_clean_experiment(data: ControlledExperimentData) -> dict[str, Any]:
    """Check clean-snapshot keys, references, variants, cardinality, and time order."""

    issues: list[str] = []
    assignments_by_unit = {row["unit_id"]: row for row in data.assignments}
    exposures_by_id = {row["exposure_id"]: row for row in data.exposures}
    assignment_ids = {row["assignment_id"] for row in data.assignments}
    exposure_ids = list(exposures_by_id)
    exposure_units = [row["unit_id"] for row in data.exposures]
    event_ids = [row["event_id"] for row in data.events]

    if len(assignment_ids) != len(data.assignments):
        issues.append("duplicate_assignment_id")
    if len(assignments_by_unit) != len(data.assignments):
        issues.append("duplicate_assignment_unit")
    if len(exposure_ids) != len(data.exposures):
        issues.append("duplicate_exposure_id")
    if len(exposure_units) != len(set(exposure_units)):
        issues.append("duplicate_exposure_unit")
    if len(event_ids) != len(set(event_ids)):
        issues.append("duplicate_event_id")

    event_names_by_exposure: dict[str, list[str]] = {}
    for exposure in data.exposures:
        assignment = assignments_by_unit.get(exposure["unit_id"])
        if assignment is None:
            issues.append("exposure_without_assignment")
            continue
        if assignment["variant_id"] != exposure["variant_id"]:
            issues.append("assignment_exposure_variant_mismatch")
        if _parse_utc(exposure["exposed_at"]) < _parse_utc(assignment["assigned_at"]):
            issues.append("exposure_before_assignment")

    experiment_end = _parse_utc(data.config.start_time_utc) + timedelta(
        days=data.config.duration_days
    )
    for event in data.events:
        exposure = exposures_by_id.get(event["exposure_id"])
        if exposure is None:
            issues.append("event_without_exposure")
            continue
        event_names_by_exposure.setdefault(event["exposure_id"], []).append(event["event_name"])
        if event["unit_id"] != exposure["unit_id"]:
            issues.append("exposure_event_unit_mismatch")
        if event["variant_id"] != exposure["variant_id"]:
            issues.append("exposure_event_variant_mismatch")
        if _parse_utc(event["event_time"]) < _parse_utc(exposure["exposed_at"]):
            issues.append("event_before_exposure")
        if _parse_utc(event["event_time"]) > experiment_end:
            issues.append("event_after_experiment_end")
        arrival_time = event.get("arrival_time")
        if not arrival_time:
            issues.append("missing_arrival_time")
        else:
            parsed_arrival = _parse_utc(arrival_time)
            if parsed_arrival < _parse_utc(event["event_time"]):
                issues.append("arrival_before_event")
            if parsed_arrival > experiment_end:
                issues.append("arrival_after_experiment_end")

    expected_names = sorted(
        (
            data.config.primary_event_name,
            data.config.conversion_event_name,
            data.config.reliability_event_name,
            data.config.latency_event_name,
        )
    )
    if any(sorted(names) != expected_names for names in event_names_by_exposure.values()):
        issues.append("event_metric_cardinality_mismatch")
    if len(event_names_by_exposure) != len(data.exposures):
        issues.append("exposure_without_complete_events")

    variant_counts = {
        variant: sum(row["variant_id"] == variant for row in data.assignments)
        for variant in (data.config.control_variant, data.config.treatment_variant)
    }
    return {
        "schema_version": data.config.schema_version,
        "snapshot_id": data.snapshot_id,
        "assignment_rows": len(data.assignments),
        "exposure_rows": len(data.exposures),
        "event_rows": len(data.events),
        "variant_counts": variant_counts,
        "observed_treatment_share": (
            variant_counts[data.config.treatment_variant] / len(data.assignments)
        ),
        "issues": sorted(set(issues)),
        "passed": not issues,
    }
