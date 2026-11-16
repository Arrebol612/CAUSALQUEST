import hashlib

import pytest

from causalquest.data.upworthy import download_upworthy_snapshot


def _config(source, destination, payload):
    return {
        "dataset_id": "upworthy_test",
        "version": "test-v1",
        "license": {"dataset": "CC-BY-4.0"},
        "acquisition": {
            "destination": destination,
            "files": [
                {
                    "role": "exploratory",
                    "osf_guid": "local",
                    "name": "sample.csv",
                    "url": source.as_uri(),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
        },
    }


def test_download_is_atomic_verified_and_reusable(tmp_path):
    repository = tmp_path / "repo"
    source = tmp_path / "source.csv"
    payload = b"experiment,impressions,clicks\nexp-1,10,2\n"
    source.write_bytes(payload)
    config = _config(source, "dataset/raw/upworthy/test-v1", payload)
    first = download_upworthy_snapshot(
        config,
        repository,
        "dataset/manifests/receipt.json",
        accept_license=True,
    )
    second = download_upworthy_snapshot(
        config,
        repository,
        "dataset/manifests/receipt.json",
        accept_license=True,
    )
    assert first["passed"] is True
    assert first["files"][0]["status"] == "downloaded"
    assert second["files"][0]["status"] == "reused_verified"
    assert not (repository / "dataset/raw/upworthy/test-v1/sample.csv.part").exists()


def test_download_requires_license_acceptance_and_safe_destination(tmp_path):
    source = tmp_path / "source.csv"
    payload = b"a\n1\n"
    source.write_bytes(payload)
    config = _config(source, "../outside", payload)
    with pytest.raises(ValueError, match="acceptance"):
        download_upworthy_snapshot(config, tmp_path / "repo", "receipt.json", accept_license=False)
    with pytest.raises(ValueError, match="escapes"):
        download_upworthy_snapshot(config, tmp_path / "repo", "receipt.json", accept_license=True)
