"""Acquisition, audit, and normalization primitives for the Upworthy archive."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import unicodedata
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {
    "created_at",
    "clickability_test_id",
    "headline",
    "impressions",
    "clicks",
    "test_week",
}
# Strong stimulus/article identifiers.  ``share_text`` is deliberately excluded:
# generic values such as "Wow." recur across unrelated stories and create giant,
# semantically invalid connected components.
CONTENT_LINK_COLUMNS = ("headline", "slug", "eyecatcher_id")
NORMALIZATION_SCHEMA_VERSION = "upworthy-canonical-v1"


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    actual_bytes = path.stat().st_size
    actual_sha256 = _file_sha256(path)
    return {
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "passed": actual_bytes == expected_bytes and actual_sha256 == expected_sha256,
    }


def download_upworthy_snapshot(
    config: Mapping[str, Any],
    repository_root: str | Path,
    receipt_path: str | Path,
    *,
    accept_license: bool,
    force: bool = False,
) -> dict[str, Any]:
    """Download the approved files atomically and verify their pinned bytes and hashes."""

    if not accept_license:
        raise ValueError("download requires explicit CC-BY-4.0 acceptance")
    if config.get("license", {}).get("dataset") != "CC-BY-4.0":
        raise ValueError("unexpected dataset license")

    repo = Path(repository_root).resolve()
    allowed_root = (repo / "dataset" / "raw").resolve()
    destination_root = (repo / str(config["acquisition"]["destination"])).resolve()
    if not _inside(allowed_root, destination_root):
        raise ValueError("configured destination escapes dataset/raw")
    destination_root.mkdir(parents=True, exist_ok=True)

    records = []
    for item in config["acquisition"]["files"]:
        destination = (destination_root / str(item["name"])).resolve()
        if not _inside(destination_root, destination):
            raise ValueError(f"unsafe file name: {item['name']}")
        expected_bytes = int(item["bytes"])
        expected_sha256 = str(item["sha256"])
        status = "downloaded"
        if destination.exists() and not force:
            verification = _verify_file(destination, expected_bytes, expected_sha256)
            if not verification["passed"]:
                raise ValueError(f"existing file does not match proposal: {destination.name}")
            status = "reused_verified"
        else:
            partial = destination.with_name(destination.name + ".part")
            if partial.exists():
                if not force:
                    raise ValueError(f"stale partial download exists: {partial.name}")
                partial.unlink()
            request = urllib.request.Request(
                str(item["url"]), headers={"User-Agent": "CausalQuest/0.1 (+data-audit)"}
            )
            digest = hashlib.sha256()
            byte_count = 0
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                partial.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if byte_count != expected_bytes or digest.hexdigest() != expected_sha256:
                raise ValueError(
                    f"download verification failed for {destination.name}: "
                    f"bytes={byte_count}, sha256={digest.hexdigest()}"
                )
            partial.replace(destination)
            verification = _verify_file(destination, expected_bytes, expected_sha256)

        records.append(
            {
                "role": item["role"],
                "osf_guid": item["osf_guid"],
                "source_url": item["url"],
                "path": destination.relative_to(repo).as_posix(),
                "expected_bytes": expected_bytes,
                "actual_bytes": verification["bytes"],
                "expected_sha256": expected_sha256,
                "actual_sha256": verification["sha256"],
                "status": status,
                "passed": verification["passed"],
            }
        )

    receipt = {
        "schema_version": "upworthy-acquisition-receipt-v1",
        "dataset_id": config["dataset_id"],
        "source_version": config["version"],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": config["license"]["dataset"],
        "files": records,
        "total_bytes": sum(item["actual_bytes"] for item in records),
        "passed": bool(records) and all(item["passed"] for item in records),
    }
    output_path = Path(receipt_path)
    if not output_path.is_absolute():
        output_path = repo / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = output_path.with_name(output_path.name + ".tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.move(str(temporary_receipt), str(output_path))
    return receipt


def _parse_nonnegative_integer(value: str) -> int:
    parsed = float(value)
    if not parsed.is_integer() or parsed < 0:
        raise ValueError(value)
    return int(parsed)


def _parse_date(value: str):
    return datetime.fromisoformat(value.strip()).date()


def _audit_upworthy_file(
    path: Path,
    specification: Mapping[str, Any],
    risk_aliases: list[str],
    risk_start,
    risk_end,
) -> tuple[dict[str, Any], set[str], set[str]]:
    report: dict[str, Any] = {
        "role": specification["role"],
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "rows": 0,
        "tests": 0,
        "columns": [],
        "unnamed_index_column": False,
        "risk_indicator_column": None,
        "risk_indicator_mode": None,
        "risky_rows": 0,
        "risky_tests": 0,
        "exact_duplicate_rows": 0,
        "duplicate_source_indices": 0,
        "tests_with_fewer_than_two_valid_arms": 0,
        "null_counts": {},
        "invalid": {
            "missing_test_id": 0,
            "missing_headline": 0,
            "invalid_created_at": 0,
            "invalid_impressions": 0,
            "invalid_clicks": 0,
            "clicks_exceed_impressions": 0,
            "nonpositive_impressions": 0,
        },
        "warnings": [],
    }
    test_valid_arms: dict[str, int] = {}
    test_risk: dict[str, bool] = {}
    missing_headline_tests: set[str] = set()
    seen_rows: set[str] = set()
    seen_indices: set[str] = set()
    content_keys: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        report["columns"] = columns
        missing = sorted(REQUIRED_COLUMNS.difference(columns))
        if missing:
            raise ValueError(f"{path.name} missing required columns: {missing}")
        report["unnamed_index_column"] = "" in columns
        risk_column = next((alias for alias in risk_aliases if alias in columns), None)
        report["risk_indicator_column"] = risk_column
        report["risk_indicator_mode"] = (
            "source_column" if risk_column else "published_date_window_derived"
        )
        if risk_column is None:
            report["warnings"].append("source_randomization_risk_column_absent")
        null_counts = {column or "__unnamed_index__": 0 for column in columns}
        canonical_columns = [column for column in columns if column]
        for row in reader:
            report["rows"] += 1
            for column in columns:
                if not str(row.get(column, "")).strip():
                    null_counts[column or "__unnamed_index__"] += 1
            source_index = str(row.get("", "")).strip()
            if source_index:
                if source_index in seen_indices:
                    report["duplicate_source_indices"] += 1
                seen_indices.add(source_index)
            canonical = "\x1f".join(str(row.get(column, "")) for column in canonical_columns)
            canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if canonical_hash in seen_rows:
                report["exact_duplicate_rows"] += 1
            seen_rows.add(canonical_hash)

            test_id = str(row.get("clickability_test_id", "")).strip()
            if not test_id:
                report["invalid"]["missing_test_id"] += 1
            if not str(row.get("headline", "")).strip():
                report["invalid"]["missing_headline"] += 1
                if test_id:
                    missing_headline_tests.add(test_id)
            try:
                created_date = _parse_date(str(row.get("created_at", "")))
            except ValueError:
                created_date = None
                report["invalid"]["invalid_created_at"] += 1
            try:
                impressions = _parse_nonnegative_integer(str(row.get("impressions", "")))
            except ValueError:
                impressions = None
                report["invalid"]["invalid_impressions"] += 1
            try:
                clicks = _parse_nonnegative_integer(str(row.get("clicks", "")))
            except ValueError:
                clicks = None
                report["invalid"]["invalid_clicks"] += 1
            if impressions is not None and impressions <= 0:
                report["invalid"]["nonpositive_impressions"] += 1
            if impressions is not None and clicks is not None and clicks > impressions:
                report["invalid"]["clicks_exceed_impressions"] += 1

            derived_risk = bool(created_date and risk_start <= created_date <= risk_end)
            if risk_column:
                source_risk = str(row.get(risk_column, "")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                }
                row_risk = source_risk
            else:
                row_risk = derived_risk
            if row_risk:
                report["risky_rows"] += 1
            if test_id:
                test_risk[test_id] = test_risk.get(test_id, False) or row_risk
                valid_arm = (
                    impressions is not None
                    and impressions > 0
                    and clicks is not None
                    and clicks <= impressions
                )
                test_valid_arms[test_id] = test_valid_arms.get(test_id, 0) + int(valid_arm)
                content_key = "\x1f".join(
                    (
                        str(row.get("headline", "")).strip().casefold(),
                        str(row.get("slug", "")).strip().casefold(),
                        str(row.get("eyecatcher_id", "")).strip(),
                    )
                )
                content_keys.add(hashlib.sha256(content_key.encode("utf-8")).hexdigest())

    report["null_counts"] = null_counts
    report["tests"] = len(test_valid_arms)
    report["risky_tests"] = sum(test_risk.values())
    tests_with_fewer_than_two_valid_arms = {
        test_id for test_id, count in test_valid_arms.items() if count < 2
    }
    report["tests_with_fewer_than_two_valid_arms"] = len(tests_with_fewer_than_two_valid_arms)
    report["quarantined_tests"] = len(missing_headline_tests | tests_with_fewer_than_two_valid_arms)
    expected_rows = int(specification["expected_packages"])
    expected_tests = int(specification["expected_tests"])
    report["expected_rows"] = expected_rows
    report["expected_tests"] = expected_tests
    report["checks"] = {
        "bytes": report["bytes"] == int(specification["bytes"]),
        "sha256": report["sha256"] == specification["sha256"],
        "row_count": report["rows"] == expected_rows,
        "test_count": report["tests"] == expected_tests,
        "causal_count_values": not any(
            report["invalid"][name]
            for name in (
                "missing_test_id",
                "invalid_created_at",
                "invalid_impressions",
                "invalid_clicks",
                "clicks_exceed_impressions",
                "nonpositive_impressions",
            )
        ),
        "exact_rows_unique": report["exact_duplicate_rows"] == 0,
        "source_indices_unique": report["duplicate_source_indices"] == 0,
        "risk_resolvable": risk_column is not None or report["invalid"]["invalid_created_at"] == 0,
    }
    report["quarantine_required"] = {
        "missing_headline_rows": report["invalid"]["missing_headline"],
        "tests_with_fewer_than_two_valid_arms": report["tests_with_fewer_than_two_valid_arms"],
    }
    report["passed"] = all(report["checks"].values())
    return report, set(test_valid_arms), content_keys


def audit_upworthy_snapshot(
    config: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Any]:
    """Stream the approved archive and audit schema, units, partitions, and causal gates."""

    repo = Path(repository_root).resolve()
    raw_root = (repo / str(config["acquisition"]["destination"])).resolve()
    allowed_root = (repo / "dataset" / "raw").resolve()
    if not _inside(allowed_root, raw_root):
        raise ValueError("configured destination escapes dataset/raw")
    risk_start = datetime.fromisoformat(config["quality_policy"]["risky_period_start"]).date()
    risk_end = datetime.fromisoformat(config["quality_policy"]["risky_period_end"]).date()
    risk_aliases = list(config["expected_schema"]["randomization_risk_aliases"])
    reports = []
    tests_by_role: dict[str, set[str]] = {}
    content_by_role: dict[str, set[str]] = {}
    for specification in config["acquisition"]["files"]:
        path = raw_root / specification["name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        report, test_ids, content_keys = _audit_upworthy_file(
            path, specification, risk_aliases, risk_start, risk_end
        )
        reports.append(report)
        tests_by_role[str(specification["role"])] = test_ids
        content_by_role[str(specification["role"])] = content_keys

    roles = sorted(tests_by_role)
    test_overlap: dict[str, int] = {}
    content_overlap: dict[str, int] = {}
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            pair = f"{left}__{right}"
            test_overlap[pair] = len(tests_by_role[left].intersection(tests_by_role[right]))
            content_overlap[pair] = len(content_by_role[left].intersection(content_by_role[right]))
    checks = {
        "all_files_pass": all(report["passed"] for report in reports),
        "test_partitions_disjoint": not any(test_overlap.values()),
        "risk_policy_available": all(
            report["risk_indicator_column"] is not None
            or report["invalid"]["invalid_created_at"] == 0
            for report in reports
        ),
    }
    return {
        "schema_version": "upworthy-raw-audit-v1",
        "dataset_id": config["dataset_id"],
        "source_version": config["version"],
        "row_unit": config["expected_schema"]["row_unit"],
        "experiment_key": config["expected_schema"]["experiment_key"],
        "files": reports,
        "total_rows": sum(report["rows"] for report in reports),
        "total_tests": sum(report["tests"] for report in reports),
        "total_risky_tests": sum(report["risky_tests"] for report in reports),
        "test_partition_overlap": test_overlap,
        "content_fingerprint_overlap": content_overlap,
        "quarantine_required": {
            "missing_headline_rows": sum(
                report["invalid"]["missing_headline"] for report in reports
            ),
            "tests_with_fewer_than_two_valid_arms": sum(
                report["tests_with_fewer_than_two_valid_arms"] for report in reports
            ),
            "quarantined_tests": sum(report["quarantined_tests"] for report in reports),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _normalize_content(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _content_components(
    config: Mapping[str, Any], raw_root: Path
) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    parents: dict[str, str] = {}
    sizes: dict[str, int] = {}
    partition_by_test: dict[str, str] = {}
    owner_by_key: dict[tuple[str, str], str] = {}

    def find(test_id: str) -> str:
        parent = parents[test_id]
        if parent != test_id:
            parents[test_id] = find(parent)
        return parents[test_id]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    for specification in config["acquisition"]["files"]:
        role = str(specification["role"])
        path = raw_root / specification["name"]
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                test_id = str(row["clickability_test_id"]).strip()
                if not test_id:
                    continue
                if test_id not in parents:
                    parents[test_id] = test_id
                    sizes[test_id] = 1
                    partition_by_test[test_id] = role
                elif partition_by_test[test_id] != role:
                    raise ValueError(f"test ID crosses source partitions: {test_id}")
                for column in CONTENT_LINK_COLUMNS:
                    normalized = _normalize_content(str(row.get(column, "")))
                    if not normalized:
                        continue
                    key = (column, normalized)
                    owner = owner_by_key.get(key)
                    if owner is None:
                        owner_by_key[key] = test_id
                    else:
                        union(test_id, owner)

    members: dict[str, list[str]] = {}
    for test_id in parents:
        members.setdefault(find(test_id), []).append(test_id)

    records: list[tuple[Any, ...]] = []
    assigned_counts: dict[str, int] = {}
    spanning_components = 0
    nonholdout_tests_locked = 0
    confirmatory_tests_to_development = 0
    largest_component_tests = 0
    for test_ids in members.values():
        test_ids.sort()
        partitions = sorted({partition_by_test[test_id] for test_id in test_ids})
        if "holdout" in partitions:
            assigned_role = "locked_private_test"
        elif "exploratory" in partitions:
            assigned_role = "development"
        else:
            assigned_role = "candidate_train"
        component_id = "ugc_" + hashlib.sha256("\x1f".join(test_ids).encode()).hexdigest()[:24]
        spans = len(partitions) > 1
        spanning_components += int(spans)
        largest_component_tests = max(largest_component_tests, len(test_ids))
        for test_id in test_ids:
            source_partition = partition_by_test[test_id]
            assigned_counts[assigned_role] = assigned_counts.get(assigned_role, 0) + 1
            nonholdout_tests_locked += int(
                assigned_role == "locked_private_test" and source_partition != "holdout"
            )
            confirmatory_tests_to_development += int(
                assigned_role == "development" and source_partition == "confirmatory"
            )
            records.append(
                (
                    test_id,
                    source_partition,
                    component_id,
                    assigned_role,
                    spans,
                    len(partitions),
                    len(test_ids),
                )
            )
    return records, {
        "link_columns": list(CONTENT_LINK_COLUMNS),
        "component_count": len(members),
        "cross_partition_components": spanning_components,
        "largest_component_tests": largest_component_tests,
        "assigned_test_counts": assigned_counts,
        "nonholdout_tests_locked_by_content": nonholdout_tests_locked,
        "confirmatory_tests_moved_to_development": confirmatory_tests_to_development,
    }


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_upworthy_database(
    config: Mapping[str, Any],
    repository_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build an immutable-source canonical DB with quarantine and leakage components."""

    import duckdb

    repo = Path(repository_root).resolve()
    raw_root = (repo / str(config["acquisition"]["destination"])).resolve()
    allowed_raw = (repo / "dataset" / "raw").resolve()
    if not _inside(allowed_raw, raw_root):
        raise ValueError("configured destination escapes dataset/raw")
    audit = audit_upworthy_snapshot(config, repo)
    if not audit["passed"]:
        raise ValueError("raw audit has unresolved hard failures")

    output = Path(output_path)
    if not output.is_absolute():
        output = repo / output
    output = output.resolve()
    allowed_processed = (repo / "dataset" / "processed").resolve()
    if not _inside(allowed_processed, output):
        raise ValueError("output escapes dataset/processed")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    component_records, component_report = _content_components(config, raw_root)
    risk_start = _sql_literal(config["quality_policy"]["risky_period_start"])
    risk_end = _sql_literal(config["quality_policy"]["risky_period_end"])
    selects = []
    source_records = []
    for specification in config["acquisition"]["files"]:
        role = str(specification["role"])
        source = raw_root / specification["name"]
        source_path = _sql_literal(source.as_posix())
        source_name = _sql_literal(source.name)
        role_literal = _sql_literal(role)
        selects.append(
            f"""
            SELECT
                'upworthy:' || {role_literal} || ':' || column00 AS arm_id,
                {role_literal} AS source_partition,
                CAST(column00 AS BIGINT) AS source_row_index,
                {source_name} AS source_file,
                clickability_test_id AS test_id,
                CAST(created_at AS TIMESTAMP) AS created_at_unknown_tz,
                TRY_CAST(updated_at AS TIMESTAMP) AS updated_at_unknown_tz,
                CAST(test_week AS INTEGER) AS test_week,
                excerpt,
                headline,
                lede,
                slug,
                eyecatcher_id,
                share_text,
                square,
                CAST(impressions AS BIGINT) AS impressions,
                CAST(clicks AS BIGINT) AS clicks,
                CAST(clicks AS DOUBLE) / NULLIF(CAST(impressions AS BIGINT), 0) AS ctr,
                significance AS source_significance_undocumented,
                lower(first_place) = 'true' AS editor_first_place,
                lower(winner) = 'true' AS editor_winner,
                CAST(created_at AS DATE) BETWEEN DATE {risk_start} AND DATE {risk_end}
                    AS randomization_risk,
                'published_2024_date_window_derived' AS randomization_risk_source
            FROM read_csv_auto({source_path}, header=true, all_varchar=true)
            """
        )
        source_records.append(
            (
                source.name,
                role,
                int(specification["bytes"]),
                str(specification["sha256"]),
            )
        )

    snapshot_payload = {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "risk_window": [
            config["quality_policy"]["risky_period_start"],
            config["quality_policy"]["risky_period_end"],
        ],
        "content_link_columns": list(CONTENT_LINK_COLUMNS),
        "sources": [
            {"role": item["role"], "sha256": item["sha256"]}
            for item in config["acquisition"]["files"]
        ],
    }
    snapshot_id = hashlib.sha256(
        json.dumps(snapshot_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    connection = duckdb.connect(str(temporary))
    try:
        connection.execute("CREATE TABLE upworthy_arms AS " + " UNION ALL ".join(selects))
        connection.execute("CREATE UNIQUE INDEX upworthy_arm_id ON upworthy_arms(arm_id)")
        connection.execute(
            """
            CREATE TABLE content_components (
                test_id VARCHAR PRIMARY KEY,
                source_partition VARCHAR NOT NULL,
                component_id VARCHAR NOT NULL,
                assigned_pool VARCHAR NOT NULL,
                spans_source_partitions BOOLEAN NOT NULL,
                source_partition_count INTEGER NOT NULL,
                component_test_count INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO content_components VALUES (?, ?, ?, ?, ?, ?, ?)",
            component_records,
        )
        connection.execute(
            """
            CREATE TABLE quarantine_tests AS
            WITH reasons AS (
                SELECT DISTINCT source_partition, test_id, 'missing_headline' AS reason
                FROM upworthy_arms
                WHERE headline IS NULL OR trim(headline) = ''
                UNION ALL
                SELECT source_partition, test_id, 'fewer_than_two_valid_arms' AS reason
                FROM upworthy_arms
                GROUP BY source_partition, test_id
                HAVING COUNT(*) FILTER (
                    WHERE impressions > 0 AND clicks BETWEEN 0 AND impressions
                ) < 2
            )
            SELECT source_partition, test_id, string_agg(DISTINCT reason, ',' ORDER BY reason)
                AS reasons
            FROM reasons
            GROUP BY source_partition, test_id
            """
        )
        connection.execute(
            """
            CREATE TABLE upworthy_tests AS
            SELECT
                a.source_partition,
                a.test_id,
                c.component_id,
                c.assigned_pool,
                c.spans_source_partitions,
                COUNT(*) AS arm_count,
                SUM(a.impressions) AS impressions,
                SUM(a.clicks) AS clicks,
                SUM(a.clicks)::DOUBLE / NULLIF(SUM(a.impressions), 0) AS pooled_ctr,
                MIN(a.created_at_unknown_tz) AS created_at_min_unknown_tz,
                MAX(a.created_at_unknown_tz) AS created_at_max_unknown_tz,
                bool_or(a.randomization_risk) AS randomization_risk,
                q.reasons AS quarantine_reasons,
                q.test_id IS NULL AS schema_eligible,
                q.test_id IS NULL AND NOT bool_or(a.randomization_risk)
                    AS eligible_for_positive_causal_case
            FROM upworthy_arms a
            JOIN content_components c USING (source_partition, test_id)
            LEFT JOIN quarantine_tests q USING (source_partition, test_id)
            GROUP BY a.source_partition, a.test_id, c.component_id, c.assigned_pool,
                     c.spans_source_partitions, q.reasons, q.test_id
            """
        )
        connection.execute(
            """
            CREATE VIEW analysis_arms AS
            SELECT a.*, t.component_id, t.assigned_pool,
                   t.randomization_risk AS test_randomization_risk,
                   t.quarantine_reasons, t.schema_eligible,
                   t.eligible_for_positive_causal_case
            FROM upworthy_arms a
            JOIN upworthy_tests t USING (source_partition, test_id)
            """
        )
        connection.execute(
            """
            CREATE TABLE source_files (
                source_file VARCHAR PRIMARY KEY,
                source_partition VARCHAR NOT NULL,
                bytes BIGINT NOT NULL,
                sha256 VARCHAR NOT NULL
            )
            """
        )
        connection.executemany("INSERT INTO source_files VALUES (?, ?, ?, ?)", source_records)
        observed_rows = connection.execute("SELECT COUNT(*) FROM upworthy_arms").fetchone()[0]
        observed_tests = connection.execute("SELECT COUNT(*) FROM upworthy_tests").fetchone()[0]
        quarantined_tests = connection.execute("SELECT COUNT(*) FROM quarantine_tests").fetchone()[
            0
        ]
        risky_tests = connection.execute(
            "SELECT COUNT(*) FROM upworthy_tests WHERE randomization_risk"
        ).fetchone()[0]
        eligible_tests = connection.execute(
            "SELECT COUNT(*) FROM upworthy_tests WHERE eligible_for_positive_causal_case"
        ).fetchone()[0]
        connection.execute(
            """
            CREATE TABLE snapshot_metadata AS
            SELECT ?::VARCHAR AS schema_version,
                   ?::VARCHAR AS snapshot_id,
                   ?::TIMESTAMPTZ AS built_at_utc,
                   ?::BIGINT AS arm_rows,
                   ?::BIGINT AS tests
            """,
            [
                NORMALIZATION_SCHEMA_VERSION,
                snapshot_id,
                datetime.now(timezone.utc),
                observed_rows,
                observed_tests,
            ],
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

    temporary.replace(output)
    checks = {
        "raw_audit": audit["passed"],
        "arm_count": observed_rows == audit["total_rows"],
        "test_count": observed_tests == audit["total_tests"],
        "quarantine_materialized": quarantined_tests
        == audit["quarantine_required"]["quarantined_tests"],
        "risk_count": risky_tests == audit["total_risky_tests"],
    }
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "dataset_id": config["dataset_id"],
        "source_version": config["version"],
        "snapshot_id": snapshot_id,
        "output": output.relative_to(repo).as_posix(),
        "database_bytes": output.stat().st_size,
        "arm_rows": observed_rows,
        "tests": observed_tests,
        "quarantined_tests": quarantined_tests,
        "randomization_risk_tests": risky_tests,
        "eligible_positive_causal_tests": eligible_tests,
        "component_audit": component_report,
        "checks": checks,
        "passed": all(checks.values()),
    }
