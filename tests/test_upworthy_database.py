import csv
import hashlib

import pytest

duckdb = pytest.importorskip("duckdb")

from causalquest.data.upworthy import build_upworthy_database

COLUMNS = [
    "",
    "created_at",
    "updated_at",
    "clickability_test_id",
    "excerpt",
    "headline",
    "lede",
    "slug",
    "eyecatcher_id",
    "impressions",
    "clicks",
    "significance",
    "first_place",
    "winner",
    "share_text",
    "square",
    "test_week",
]


def _rows(test_id, *, content="shared", missing_headline=False):
    return [
        {
            "": str(index),
            "created_at": "2014-11-20 06:43:16.005",
            "updated_at": "2016-04-02 16:33:38.062",
            "clickability_test_id": test_id,
            "excerpt": "excerpt",
            "headline": f"{content} headline" if not missing_headline or index else "",
            "lede": "lede",
            "slug": f"{content}-slug",
            "eyecatcher_id": f"{content}-image",
            "impressions": "100",
            "clicks": str(10 + index),
            "significance": "1.0",
            "first_place": "False",
            "winner": "False",
            "share_text": "shared text",
            "square": "",
            "test_week": "201446",
        }
        for index in range(2)
    ]


def _config(repository, *, shared_strong_content=True):
    destination = repository / "dataset/raw/upworthy/test"
    destination.mkdir(parents=True)
    files = []
    for role, rows in (
        (
            "confirmatory",
            _rows("train-test", content="shared" if shared_strong_content else "train"),
        ),
        (
            "holdout",
            _rows(
                "holdout-test",
                content="shared" if shared_strong_content else "holdout",
                missing_headline=True,
            ),
        ),
    ):
        path = destination / f"{role}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        payload = path.read_bytes()
        files.append(
            {
                "role": role,
                "name": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "expected_packages": 2,
                "expected_tests": 1,
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


def test_database_quarantines_invalid_text_and_locks_connected_content(tmp_path):
    repository = tmp_path / "repo"
    config = _config(repository)
    output = repository / "dataset/processed/upworthy/test.duckdb"
    report = build_upworthy_database(config, repository, output)
    assert report["passed"] is True
    assert report["quarantined_tests"] == 1
    replacement_report = build_upworthy_database(config, repository, output)
    assert replacement_report["snapshot_id"] == report["snapshot_id"]
    connection = duckdb.connect(str(output), read_only=True)
    try:
        pools = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT assigned_pool FROM upworthy_tests"
            ).fetchall()
        }
        quarantine = connection.execute("SELECT COUNT(*) FROM quarantine_tests").fetchone()[0]
        rows = connection.execute("SELECT COUNT(*) FROM analysis_arms").fetchone()[0]
    finally:
        connection.close()
    assert pools == {"locked_private_test"}
    assert quarantine == 1
    assert rows == 4


def test_generic_share_text_alone_does_not_link_partitions(tmp_path):
    repository = tmp_path / "repo"
    config = _config(repository, shared_strong_content=False)
    output = repository / "dataset/processed/upworthy/test.duckdb"
    report = build_upworthy_database(config, repository, output)
    assert report["passed"] is True
    assert report["component_audit"]["component_count"] == 2
    assert report["component_audit"]["nonholdout_tests_locked_by_content"] == 0
    connection = duckdb.connect(str(output), read_only=True)
    try:
        pools = dict(
            connection.execute(
                "SELECT source_partition, assigned_pool FROM upworthy_tests"
            ).fetchall()
        )
    finally:
        connection.close()
    assert pools == {
        "confirmatory": "candidate_train",
        "holdout": "locked_private_test",
    }
