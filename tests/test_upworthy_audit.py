import csv
import hashlib

from causalquest.data.upworthy import audit_upworthy_snapshot

COLUMNS = [
    "",
    "created_at",
    "clickability_test_id",
    "headline",
    "slug",
    "eyecatcher_id",
    "impressions",
    "clicks",
    "test_week",
]


def _write_partition(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _rows(test_id, date="2014-11-20 06:43:16.005"):
    return [
        {
            "": str(index),
            "created_at": date,
            "clickability_test_id": test_id,
            "headline": f"headline {index}",
            "slug": f"slug-{index}",
            "eyecatcher_id": f"image-{index}",
            "impressions": "100",
            "clicks": str(10 + index),
            "test_week": "201446",
        }
        for index in range(2)
    ]


def _config(repository, role_to_rows):
    destination = repository / "dataset/raw/upworthy/test"
    destination.mkdir(parents=True)
    files = []
    for role, rows in role_to_rows.items():
        path = destination / f"{role}.csv"
        _write_partition(path, rows)
        payload = path.read_bytes()
        files.append(
            {
                "role": role,
                "name": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "expected_packages": len(rows),
                "expected_tests": len({row["clickability_test_id"] for row in rows}),
            }
        )
    return {
        "dataset_id": "upworthy_test",
        "version": "test-v1",
        "acquisition": {"destination": "dataset/raw/upworthy/test", "files": files},
        "expected_schema": {
            "row_unit": "package_or_experiment_arm",
            "experiment_key": "clickability_test_id",
            "randomization_risk_aliases": ["problem"],
        },
        "quality_policy": {
            "risky_period_start": "2013-06-25",
            "risky_period_end": "2014-01-10",
        },
    }


def test_audit_accepts_disjoint_valid_partitions_and_derives_risk(tmp_path):
    repository = tmp_path / "repo"
    config = _config(repository, {"exploratory": _rows("test-a"), "holdout": _rows("test-b")})
    report = audit_upworthy_snapshot(config, repository)
    assert report["passed"] is True
    assert report["total_rows"] == 4
    assert report["test_partition_overlap"]["exploratory__holdout"] == 0
    assert all(
        item["risk_indicator_mode"] == "published_date_window_derived" for item in report["files"]
    )


def test_audit_rejects_invalid_counts_and_partition_leakage(tmp_path):
    repository = tmp_path / "repo"
    invalid = _rows("shared-test", date="2013-07-01 00:00:00")
    invalid[0]["clicks"] = "101"
    config = _config(
        repository,
        {"exploratory": invalid, "holdout": _rows("shared-test")},
    )
    report = audit_upworthy_snapshot(config, repository)
    assert report["passed"] is False
    assert report["test_partition_overlap"]["exploratory__holdout"] == 1
    assert report["total_risky_tests"] == 1
    exploratory = next(item for item in report["files"] if item["role"] == "exploratory")
    assert exploratory["invalid"]["clicks_exceed_impressions"] == 1
