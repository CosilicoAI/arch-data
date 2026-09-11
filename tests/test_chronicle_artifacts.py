"""Tests for Chronicle source artifact acquisition and storage metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pytest
import yaml

from chronicle.cli import main as cli_main
from chronicle.artifacts import (
    AmbiguousManifestError,
    ArtifactCommandResult,
    ArtifactFilenameError,
    MalformedManifestError,
    ManifestNameError,
    RecordedR2LocatorError,
    SourceArtifactManifestError,
    SourceArtifactRevisionError,
    build_artifact_key,
    build_artifact_rows,
    build_derived_r2_key,
    bootstrap_r2_buckets,
    build_r2_key,
    fetch_source_artifact,
    infer_r2_country,
    infer_build_id,
    inventory_source_artifacts,
    publish_derived_artifacts,
    publish_source_artifacts,
)
from chronicle.epoch import Epoch
from chronicle.harness import main as harness_main


def test_build_r2_key_is_content_addressed():
    key = build_r2_key(
        source_id="irs_soi",
        package_id="soi-table-1-2",
        year=2023,
        sha256="abc123",
        filename="23in12ms.xls",
    )

    assert key == "raw/irs_soi/soi-table-1-2/2023/abc123/23in12ms.xls"


def test_build_derived_r2_key_is_build_scoped():
    key = build_derived_r2_key(
        source_id="irs_soi",
        package_id="soi-table-1-1",
        year=2023,
        build_id="ledger.build.v1:abc123",
        artifact_name="reports/build_summary.json",
    )

    assert key == (
        "derived/irs_soi/soi-table-1-1/2023/"
        "ledger.build.v1:abc123/reports/build_summary.json"
    )


def test_build_artifact_key_hashes_one_payload_across_epochs():
    ledger = build_artifact_key(
        build_id="ledger.build.v1:abc123",
        artifact_name="reports/build_summary.json",
        sha256="def456",
    )
    chronicle = build_artifact_key(
        build_id="chronicle.build.v2:abc123",
        artifact_name="reports/build_summary.json",
        sha256="def456",
        epoch=Epoch.CHRONICLE,
    )

    assert ledger.startswith("ledger.build_artifact.v1:")
    assert chronicle.startswith("chronicle.build_artifact.v2:")
    assert ledger.split(":", maxsplit=1)[1] == chronicle.split(":", maxsplit=1)[1]


def test_build_artifact_key_rejects_unknown_build_epoch():
    with pytest.raises(
        ValueError,
        match=(
            "ledger[.]build[.]v1.*chronicle[.]build[.]v2|"
            "chronicle[.]build[.]v2.*ledger[.]build[.]v1"
        ),
    ):
        build_artifact_key(
            build_id="future.build.v9:abc123",
            artifact_name="facts.jsonl",
            sha256="def456",
        )


@pytest.mark.parametrize(
    ("source_id", "package_path", "expected_country"),
    [
        ("ird", "db/data/ird/wff", "nz"),
        ("ons", "db/data/ons/mye", "uk"),
        ("irs_soi", "db/data/irs_soi/table_1_1", None),
        ("irs_soi", "/work/ons/chronicle/db/data/irs_soi/table_1_1", None),
        ("ird", "/work/ons/chronicle/db/data/ird/wff", "nz"),
        ("ird", "/work/packages/ons/chronicle/db/data/ird/wff", "nz"),
        ("ird", "/work/ons/chronicle/packages/ird/wff", "nz"),
        ("ird", "/repo/db/data/ird/packages", "nz"),
        ("ird", "/repo/db/data/ird/packages/manifest.yaml", "nz"),
        ("ird", "/repo/packages/ird/packages/source_package.yaml", "nz"),
        ("irs_soi", "/repo/db/data/irs_soi/packages/manifest.yaml", None),
    ],
)
def test_infer_r2_country_uses_publisher_directory(
    source_id, package_path, expected_country
):
    assert (
        infer_r2_country(source_id=source_id, package_path=package_path)
        == expected_country
    )


def test_country_aware_r2_keys_preserve_legacy_us_layout():
    nz_key = build_r2_key(
        source_id="ird",
        package_id="ird-working-for-families-statistics-sept-2025",
        year=2024,
        sha256="abc123",
        filename="wff.xlsx",
    )
    uk_key = build_derived_r2_key(
        source_id="ons",
        package_id="ons-mye-2024-uk",
        year=2024,
        build_id="ledger.build.v1:abc123",
        artifact_name="facts.jsonl",
    )

    assert nz_key.startswith("raw/nz/ird/")
    assert uk_key.startswith("derived/uk/ons/")


def test_country_aware_r2_prefix_rejects_wrong_country():
    with pytest.raises(ValueError, match="disagrees with publisher country"):
        build_r2_key(
            source_id="ird",
            package_id="wff",
            year=2024,
            sha256="abc123",
            filename="wff.xlsx",
            prefix="raw/uk",
        )


def test_fetch_source_artifact_writes_manifest_and_inventory(tmp_path):
    source = tmp_path / "source.xls"
    content = b"chronicle artifact fixture"
    source.write_bytes(content)
    output_dir = tmp_path / "data" / "irs_soi" / "table_1_2"

    report = fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table-1-2",
        year=2023,
        output_dir=output_dir,
        source_page="https://example.test/source-page",
        table="Publication 1304 Table 1.2",
    )

    expected_sha = hashlib.sha256(content).hexdigest()
    manifest = yaml.safe_load((output_dir / "manifest.yaml").read_text())
    inventory = inventory_source_artifacts(output_dir)

    assert report.valid
    assert report.sha256 == expected_sha
    assert (output_dir / "source.xls").read_bytes() == content
    assert manifest["source_id"] == "irs_soi"
    assert manifest["package_id"] == "soi-table-1-2"
    assert manifest["files"][2023]["sha256"] == expected_sha
    assert manifest["files"][2023]["source_url"] == str(source)
    assert inventory.valid
    assert inventory.counts == {
        "artifact_count": 1,
        "checksum_mismatch_count": 0,
        "manifest_count": 1,
        "missing_count": 0,
        "r2_link_count": 0,
    }


def test_fetch_rejects_wrong_country_before_reading_or_overwriting_cache(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"original official artifact")
    output_dir = tmp_path / "db" / "data" / "ird" / "wff"
    fetch_source_artifact(
        str(source),
        source_id="ird",
        package_id="ird-wff",
        year=2024,
        output_dir=output_dir,
    )
    artifact_path = output_dir / source.name
    manifest_path = output_dir / "manifest.yaml"
    original_artifact = artifact_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("A rejected route must not read the source artifact")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)
    with pytest.raises(ValueError, match="disagrees with publisher country"):
        fetch_source_artifact(
            str(source),
            source_id="ird",
            package_id="ird-wff",
            year=2024,
            output_dir=output_dir,
            r2_prefix="raw/uk",
        )

    assert artifact_path.read_bytes() == original_artifact
    assert manifest_path.read_bytes() == original_manifest


def test_publish_source_artifacts_uploads_manifest_entries(tmp_path):
    output_dir = tmp_path / "data" / "irs_soi" / "table_1_2"
    source = tmp_path / "source.xls"
    source.write_bytes(b"raw artifact")
    fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table-1-2",
        year=2023,
        output_dir=output_dir,
    )
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)

    report = publish_source_artifacts(output_dir, wrangler_command=str(wrangler))
    manifest = yaml.safe_load((output_dir / "manifest.yaml").read_text())
    storage = manifest["files"][2023]["storage"]["r2"]

    assert report.valid
    assert report.counts == {
        "artifact_count": 1,
        "failed_count": 0,
        "manifest_count": 1,
        "r2_link_count": 1,
        "skipped_count": 0,
        "uploaded_count": 1,
    }
    assert storage["bucket"] == "ledger-raw"
    assert storage["key"].startswith("raw/irs_soi/soi-table-1-2/2023/")
    assert "ledger-raw/raw/irs_soi/soi-table-1-2/2023/" in log.read_text()


def test_publish_source_artifacts_handles_label_year_entries(tmp_path):
    output_dir = tmp_path / "data" / "ssa" / "ssi_monthly_statistics_2024_12"
    source = tmp_path / "table01.csv"
    source.write_bytes(b"csv artifact")
    fetch_source_artifact(
        str(source),
        source_id="ssa",
        package_id="ssa-ssi-monthly-statistics-2024-12",
        year=2024,
        output_dir=output_dir,
    )
    capture = output_dir / "table01.html"
    capture_bytes = b"<html>capture</html>"
    capture.write_bytes(capture_bytes)
    manifest_path = output_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["files"]["source_capture"] = {
        "filename": capture.name,
        "sha256": hashlib.sha256(capture_bytes).hexdigest(),
        "size_bytes": len(capture_bytes),
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)

    report = publish_source_artifacts(output_dir, wrangler_command=str(wrangler))
    manifest = yaml.safe_load(manifest_path.read_text())
    storage = manifest["files"]["source_capture"]["storage"]["r2"]

    assert report.valid
    assert report.counts["uploaded_count"] == 2
    assert report.counts["failed_count"] == 0
    assert storage["key"] == (
        "raw/ssa/ssa-ssi-monthly-statistics-2024-12/source_capture/"
        f"{hashlib.sha256(capture_bytes).hexdigest()}/table01.html"
    )


def test_publish_source_artifacts_uses_country_for_each_manifest(tmp_path):
    data_root = tmp_path / "ons" / "chronicle" / "db" / "data"
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)
    for publisher, package_id, year in (
        ("ird", "ird-wff", 2024),
        ("ons", "ons-mye", 2024),
        ("irs_soi", "soi-table", 2023),
    ):
        output_dir = data_root / publisher / package_id
        source = tmp_path / f"{publisher}.csv"
        source.write_bytes(publisher.encode())
        fetch_source_artifact(
            str(source),
            source_id=publisher,
            package_id=package_id,
            year=year,
            output_dir=output_dir,
        )

    report = publish_source_artifacts(data_root, wrangler_command=str(wrangler))

    assert report.valid
    commands = log.read_text()
    assert "ledger-raw/raw/nz/ird/ird-wff/2024/" in commands
    assert "ledger-raw/raw/uk/ons/ons-mye/2024/" in commands
    assert "ledger-raw/raw/irs_soi/soi-table/2023/" in commands


def test_publish_source_artifacts_preserves_a_legacy_countryless_key(tmp_path):
    output_dir = tmp_path / "data" / "ird" / "wff"
    source = tmp_path / "wff.xlsx"
    source.write_bytes(b"official WFF workbook")
    fetch_source_artifact(
        str(source),
        source_id="ird",
        package_id="ird-wff",
        year=2024,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    artifact = manifest["files"][2024]
    artifact["storage"] = {
        "r2": {
            "provider": "r2",
            "bucket": "ledger-raw",
            "key": (
                f"raw/ird/ird-wff/2024/{artifact['sha256']}/{artifact['filename']}"
            ),
        }
    }
    artifact["storage"]["r2"]["uri"] = (
        f"r2://ledger-raw/{artifact['storage']['r2']['key']}"
    )
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)

    report = publish_source_artifacts(output_dir, wrangler_command=str(wrangler))

    assert report.valid
    assert report.entries[0].upload is None
    assert report.entries[0].errors == ()
    assert report.entries[0].skipped == "recorded_r2_already_published"
    assert report.entries[0].r2_location.key == artifact["storage"]["r2"]["key"]
    assert not log.exists()


def test_publish_source_artifacts_accepts_package_named_packages(tmp_path):
    output_dir = tmp_path / "db" / "data" / "ird" / "packages"
    source = tmp_path / "wff.xlsx"
    source.write_bytes(b"official WFF workbook")
    fetched = fetch_source_artifact(
        str(source),
        source_id="ird",
        package_id="packages",
        year=2024,
        output_dir=output_dir,
    )
    wrangler = tmp_path / "wrangler"
    wrangler.write_text("#!/bin/sh\necho ok\n")
    wrangler.chmod(0o755)

    published = publish_source_artifacts(output_dir, wrangler_command=str(wrangler))

    assert fetched.valid
    assert published.valid
    assert published.counts["uploaded_count"] == 1
    manifest = yaml.safe_load((output_dir / "manifest.yaml").read_text())
    assert manifest["files"][2024]["storage"]["r2"]["key"].startswith(
        "raw/nz/ird/packages/2024/"
    )


def test_inventory_source_artifacts_catches_checksum_mismatch(tmp_path):
    source = tmp_path / "source.xls"
    source.write_bytes(b"original")
    output_dir = tmp_path / "data"
    fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table-1-2",
        year=2023,
        output_dir=output_dir,
    )
    (output_dir / "source.xls").write_bytes(b"changed")

    report = inventory_source_artifacts(output_dir)

    assert not report.valid
    assert report.counts["checksum_mismatch_count"] == 1
    assert report.entries[0].errors == ("checksum_mismatch",)


def test_artifact_cli_commands_emit_json(tmp_path, capsys):
    source = tmp_path / "source.xls"
    source.write_bytes(b"cli artifact")
    output_dir = tmp_path / "artifact-dir"

    exit_code = harness_main(
        [
            "fetch-artifact",
            "--url",
            str(source),
            "--source-id",
            "irs_soi",
            "--package-id",
            "soi-table-cli",
            "--year",
            "2023",
            "--out-dir",
            str(output_dir),
        ]
    )
    fetch_payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert fetch_payload["valid"]
    assert (output_dir / "manifest.yaml").exists()

    exit_code = harness_main(
        [
            "inventory-artifacts",
            "--root",
            str(output_dir),
        ]
    )
    inventory_payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert inventory_payload["valid"]
    assert inventory_payload["counts"]["artifact_count"] == 1

    wrangler = tmp_path / "wrangler"
    wrangler.write_text("#!/bin/sh\necho ok\n")
    wrangler.chmod(0o755)
    exit_code = harness_main(
        [
            "publish-raw",
            "--root",
            str(output_dir),
            "--wrangler-command",
            str(wrangler),
        ]
    )
    raw_payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert raw_payload["valid"]
    assert raw_payload["counts"]["uploaded_count"] == 1


def test_bootstrap_r2_reports_missing_authentication(tmp_path):
    wrangler = tmp_path / "wrangler"
    wrangler.write_text("#!/bin/sh\necho 'You are not authenticated.'\n")
    wrangler.chmod(0o755)

    report = bootstrap_r2_buckets(wrangler_command=str(wrangler))

    assert not report.valid
    assert not report.authenticated
    assert "wrangler_not_authenticated" in report.errors[0]


def test_publish_derived_artifacts_uploads_build_directory(tmp_path):
    suite = tmp_path / "suite"
    reports = suite / "reports"
    reports.mkdir(parents=True)
    build_id = "ledger.build.v1:test123"
    (reports / "database.json").write_text(json.dumps({"build_id": build_id}))
    (reports / "build_summary.json").write_text(
        json.dumps({"reports": {"database": {"build_id": build_id}}})
    )
    (suite / "facts.jsonl").write_text("{}\n")
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)

    report = publish_derived_artifacts(
        suite,
        source_id="irs_soi",
        package_id="soi-table-1-1",
        year=2023,
        build_artifacts_output=tmp_path / "build_artifacts.jsonl",
        wrangler_command=str(wrangler),
    )

    uploaded_names = {entry.artifact_name for entry in report.entries}
    command_log = log.read_text()
    build_artifact_rows = [
        json.loads(line)
        for line in (tmp_path / "build_artifacts.jsonl").read_text().splitlines()
    ]

    assert report.valid
    assert infer_build_id(suite) == build_id
    assert report.build_id == build_id
    assert report.build_artifacts_path == str(tmp_path / "build_artifacts.jsonl")
    assert uploaded_names == {
        "reports/build_summary.json",
        "reports/database.json",
        "facts.jsonl",
    }
    assert len(build_artifact_rows) == 3
    assert build_artifact_rows[0]["build_id"] == build_id
    assert build_artifact_rows[0]["r2_bucket"] == "ledger-derived"
    assert "ledger-derived/derived/irs_soi/soi-table-1-1/2023/" in command_log
    assert "reports/build_summary.json" in command_log


@pytest.mark.parametrize("write_build_artifacts", [False, True])
def test_publish_derived_artifacts_rejects_unknown_build_before_side_effects(
    tmp_path,
    write_build_artifacts,
):
    suite = tmp_path / "suite"
    reports = suite / "reports"
    reports.mkdir(parents=True)
    (reports / "database.json").write_text(
        json.dumps({"build_id": "future.build.v9:invalid"})
    )
    (suite / "facts.jsonl").write_text("{}\n")
    upload_log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {upload_log}\necho ok\n")
    wrangler.chmod(0o755)
    build_artifacts_path = tmp_path / "build_artifacts.jsonl"
    build_artifacts_path.write_text("sentinel\n")

    report = publish_derived_artifacts(
        suite,
        source_id="irs_soi",
        package_id="soi-table-1-1",
        year=2023,
        build_artifacts_output=(
            build_artifacts_path if write_build_artifacts else None
        ),
        wrangler_command=str(wrangler),
    )

    assert report.errors == ("malformed_build_id",)
    assert report.entries == ()
    assert not upload_log.exists()
    assert build_artifacts_path.read_text() == "sentinel\n"


def test_build_artifact_rows_skips_failed_uploads(tmp_path):
    suite = tmp_path / "suite"
    reports = suite / "reports"
    reports.mkdir(parents=True)
    (reports / "database.json").write_text(
        json.dumps({"build_id": "ledger.build.v1:failed-row"})
    )
    (suite / "facts.jsonl").write_text("{}\n")
    wrangler = tmp_path / "wrangler"
    wrangler.write_text("#!/bin/sh\nexit 1\n")
    wrangler.chmod(0o755)

    report = publish_derived_artifacts(
        suite,
        source_id="irs_soi",
        package_id="soi-table-1-1",
        year=2023,
        wrangler_command=str(wrangler),
    )

    assert not report.valid
    assert build_artifact_rows(report) == ()


def test_publish_derived_cli_emits_json(tmp_path, capsys):
    suite = tmp_path / "suite"
    reports = suite / "reports"
    reports.mkdir(parents=True)
    (reports / "database.json").write_text(
        json.dumps({"build_id": "ledger.build.v1:cli"})
    )
    (suite / "ledger.db").write_bytes(b"db")
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)

    exit_code = harness_main(
        [
            "publish-derived",
            "--dir",
            str(suite),
            "--source-id",
            "irs_soi",
            "--package-id",
            "soi-table-1-1",
            "--year",
            "2023",
            "--wrangler-command",
            str(wrangler),
            "--build-artifacts-out",
            str(tmp_path / "build_artifacts.jsonl"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["valid"]
    assert payload["build_id"] == "ledger.build.v1:cli"
    assert payload["counts"]["artifact_count"] == 2
    assert (tmp_path / "build_artifacts.jsonl").exists()


def test_top_level_cli_dispatches_publish_derived(tmp_path, capsys, monkeypatch):
    suite = tmp_path / "suite"
    reports = suite / "reports"
    reports.mkdir(parents=True)
    (reports / "database.json").write_text(
        json.dumps({"build_id": "ledger.build.v1:top-cli"})
    )
    (suite / "facts.jsonl").write_text("{}\n")
    wrangler = tmp_path / "wrangler"
    wrangler.write_text("#!/bin/sh\necho ok\n")
    wrangler.chmod(0o755)
    monkeypatch.setattr(
        "sys.argv",
        [
            "chronicle",
            "publish-derived",
            "--dir",
            str(suite),
            "--source-id",
            "irs_soi",
            "--package-id",
            "soi-table-1-1",
            "--year",
            "2023",
            "--wrangler-command",
            str(wrangler),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli_main()
    payload = json.loads(capsys.readouterr().out)

    assert exc.value.code == 0
    assert payload["valid"]


def _sqlite_build(path, build_id):
    """Write a minimal build database carrying one ledger_builds row."""
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE ledger_builds (build_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO ledger_builds VALUES (?)", (build_id,))


@pytest.mark.parametrize("db_name", ["chronicle.db", "ledger.db"])
def test_infer_build_id_reads_new_and_legacy_database_names(tmp_path, db_name):
    suite = tmp_path / "suite"
    suite.mkdir()
    _sqlite_build(suite / db_name, "ledger.build.v1:from-db")

    assert infer_build_id(suite) == "ledger.build.v1:from-db"


def test_infer_build_id_prefers_the_chronicle_database(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    _sqlite_build(suite / "chronicle.db", "ledger.build.v1:chronicle")
    _sqlite_build(suite / "ledger.db", "ledger.build.v1:legacy")

    assert infer_build_id(suite) == "ledger.build.v1:chronicle"


@pytest.mark.parametrize("db_name", ["chronicle.db", "ledger.db"])
def test_publish_derived_classifies_both_database_names(tmp_path, db_name):
    suite = tmp_path / "suite"
    reports = suite / "reports"
    reports.mkdir(parents=True)
    build_id = "ledger.build.v1:kind"
    (reports / "database.json").write_text(json.dumps({"build_id": build_id}))
    (suite / db_name).write_bytes(b"db")
    wrangler = tmp_path / "wrangler"
    wrangler.write_text("#!/bin/sh\necho ok\n")
    wrangler.chmod(0o755)

    report = publish_derived_artifacts(
        suite,
        source_id="irs_soi",
        package_id="soi-table-1-1",
        year=2023,
        wrangler_command=str(wrangler),
    )
    rows = {row["artifact_name"]: row for row in build_artifact_rows(report)}

    assert rows[db_name]["artifact_kind"] == "sqlite_database"


def test_publish_derived_uses_the_configured_bucket(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONICLE_R2_DERIVED_BUCKET", "chronicle-derived")
    suite = tmp_path / "suite"
    reports = suite / "reports"
    reports.mkdir(parents=True)
    (reports / "database.json").write_text(
        json.dumps({"build_id": "ledger.build.v1:bucket"})
    )
    (suite / "facts.jsonl").write_text("{}\n")
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)

    report = publish_derived_artifacts(
        suite,
        source_id="irs_soi",
        package_id="soi-table-1-1",
        year=2023,
        wrangler_command=str(wrangler),
    )

    assert report.valid
    assert report.entries[0].r2_location.bucket == "chronicle-derived"
    assert "chronicle-derived/derived/irs_soi/" in log.read_text()


def test_publish_raw_skips_an_object_already_held_by_a_preserved_bucket(
    tmp_path, monkeypatch
):
    """A recorded storage.r2 bucket is preserved history, not a publish target.

    Archived witness records pin raw R2 URLs by hash, so backfilling the same
    bytes into a renamed bucket must not rewrite the manifest. The entry is
    already published, so the sweep reports it skipped and stays green: after
    the bucket-default flip every entry published before it takes this path.
    """
    monkeypatch.setenv("CHRONICLE_R2_RAW_BUCKET", "chronicle-raw")
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-1-1"
    source = tmp_path / "soi.xlsx"
    source.write_bytes(b"official SOI workbook")
    fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table-1-1",
        year=2023,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    artifact = manifest["files"][2023]
    recorded_key = (
        f"raw/irs_soi/soi-table-1-1/2023/{artifact['sha256']}/{artifact['filename']}"
    )
    artifact["storage"] = {
        "r2": {
            "provider": "r2",
            "bucket": "ledger-raw",
            "key": recorded_key,
            "uri": f"r2://ledger-raw/{recorded_key}",
        }
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)

    before = manifest_path.read_bytes()
    report = publish_source_artifacts(output_dir, wrangler_command=str(wrangler))
    entry = report.entries[0]

    assert report.valid
    assert entry.errors == ()
    assert entry.upload is None
    assert entry.skipped == (
        "recorded_r2_bucket_is_preserved_history:"
        "recorded=ledger-raw:requested=chronicle-raw"
    )
    assert entry.r2_location is not None
    assert entry.r2_location.bucket == "ledger-raw"
    assert entry.r2_location.key == recorded_key
    assert entry.to_dict()["skipped"] == entry.skipped
    assert report.counts["skipped_count"] == 1
    assert report.counts["uploaded_count"] == 0
    assert report.counts["failed_count"] == 0
    assert not log.exists()
    assert manifest_path.read_bytes() == before


def test_documented_bucket_cutover_sweep_accepts_the_tracked_registry(
    tmp_path, monkeypatch, capsys
):
    """The documented bucket flip is green for every recorded historical key."""
    tracked_data = Path(__file__).resolve().parents[1] / "db" / "data"
    copied_data = tmp_path / "data"
    shutil.copytree(tracked_data, copied_data)
    manifest_bytes = {
        path.relative_to(copied_data): path.read_bytes()
        for path in copied_data.rglob("*")
        if path.is_file() and path.name.lower().startswith("manifest")
    }
    uploads = []

    def non_writing_uploader(location, local_path, *, wrangler_command):
        uploads.append((location, local_path, wrangler_command))
        return ArtifactCommandResult(
            command=("non-writing-uploader",),
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setenv("CHRONICLE_R2_RAW_BUCKET", "chronicle-raw")
    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", non_writing_uploader)

    exit_code = harness_main(
        [
            "publish-raw",
            "--root",
            str(copied_data),
            "--wrangler-command",
            "non-writing-uploader",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    counts = report["counts"]

    # The tracked registry grows as packages land, so the sweep is pinned by
    # its invariants rather than by today's exact counts: every artifact is a
    # preserved-bucket skip with an R2 link, nothing uploads or fails, no
    # manifest-level error, exit 0. The floors keep the test meaningful.
    observed = (exit_code, report["valid"], len(report["errors"]))
    assert observed == (0, True, 0), json.dumps(
        {"observed": observed, "counts": counts}, sort_keys=True
    )
    assert counts["uploaded_count"] == 0, counts
    assert counts["failed_count"] == 0, counts
    assert (
        counts["skipped_count"] == counts["artifact_count"] == counts["r2_link_count"]
    ), counts
    assert counts["artifact_count"] >= 194, counts
    assert counts["manifest_count"] >= 161, counts
    assert all(entry["skipped"] or entry["upload"] for entry in report["entries"])
    assert uploads == []
    assert {
        path.relative_to(copied_data): path.read_bytes()
        for path in copied_data.rglob("*")
        if path.is_file() and path.name.lower().startswith("manifest")
    } == manifest_bytes


def test_fetch_artifact_keeps_an_already_recorded_bucket(tmp_path, monkeypatch):
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-1-1"
    source = tmp_path / "soi.xlsx"
    source.write_bytes(b"official SOI workbook")
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)
    fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table-1-1",
        year=2023,
        output_dir=output_dir,
        upload_r2=True,
        wrangler_command=str(wrangler),
    )
    manifest_path = output_dir / "manifest.yaml"
    first = yaml.safe_load(manifest_path.read_text())
    assert first["files"][2023]["storage"]["r2"]["bucket"] == "ledger-raw"

    monkeypatch.setenv("CHRONICLE_R2_RAW_BUCKET", "chronicle-raw")
    report = fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table-1-1",
        year=2023,
        output_dir=output_dir,
        upload_r2=True,
        wrangler_command=str(wrangler),
    )
    second = yaml.safe_load(manifest_path.read_text())

    # The backfill copy really is uploaded to the new bucket, but the manifest
    # keeps recording where the bytes were first published.
    assert report.r2_location.bucket == "chronicle-raw"
    assert "chronicle-raw" in log.read_text()
    assert (
        second["files"][2023]["storage"]["r2"] == first["files"][2023]["storage"]["r2"]
    )


# ---------------------------------------------------------------------------
# Publisher revisions
#
# A raw R2 key is content-addressed, so a recorded storage.r2 block is a claim
# about specific bytes. On 2026-09-02 the IRS re-published 22in05ira.xlsx and
# 22in06ira.xlsx under their existing URLs (PolicyEngine/chronicle#225): a
# repeated fetch must never pair those new bytes with the old object's URI.
# ---------------------------------------------------------------------------

REPUBLISHED_URL = "https://www.irs.gov/pub/irs-soi/22in05ira.xlsx"
REPUBLISHED_FILENAME = "22in05ira.xlsx"
FIRST_PUBLICATION = b"IRA table 5, first publication"
SECOND_PUBLICATION = b"IRA table 5, silently re-published with revised rows"


def _wrangler_stub(tmp_path, log):
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)
    return wrangler


def _serve(monkeypatch, content):
    """Serve ``content`` from the publisher URL, without touching the network."""

    def _fake_read_artifact(source_url):
        assert source_url == REPUBLISHED_URL
        return content, REPUBLISHED_FILENAME

    monkeypatch.setattr("chronicle.artifacts._read_artifact", _fake_read_artifact)


def _fetch_republished(output_dir, wrangler, *, upload_r2=True, **kwargs):
    return fetch_source_artifact(
        REPUBLISHED_URL,
        source_id="irs_soi",
        package_id="soi-table-5",
        year=2022,
        output_dir=output_dir,
        upload_r2=upload_r2,
        wrangler_command=str(wrangler),
        **kwargs,
    )


def test_repeated_fetch_of_identical_bytes_preserves_the_recorded_block(
    tmp_path, monkeypatch
):
    """Same bytes: the recorded block survives whatever bucket is configured."""
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)
    manifest_path = output_dir / "manifest.yaml"
    _serve(monkeypatch, FIRST_PUBLICATION)
    _fetch_republished(output_dir, wrangler)
    first = yaml.safe_load(manifest_path.read_text())["files"][2022]

    monkeypatch.setenv("CHRONICLE_R2_RAW_BUCKET", "chronicle-raw")
    report = _fetch_republished(output_dir, wrangler)
    second = yaml.safe_load(manifest_path.read_text())["files"][2022]

    assert report.valid
    # The backfill copy really goes to the renamed bucket, but the manifest
    # keeps recording where these bytes were first published.
    assert report.r2_location.bucket == "chronicle-raw"
    assert "chronicle-raw" in log.read_text()
    assert second["storage"] == first["storage"]
    # Field order too, so the block is byte-for-byte identical once dumped.
    assert list(second["storage"]["r2"].items()) == list(first["storage"]["r2"].items())
    assert second["storage"]["r2"]["bucket"] == "ledger-raw"
    assert "previous_r2" not in second["storage"]
    assert second["sha256"] == first["sha256"]


@pytest.mark.parametrize(
    ("upload_r2", "configured_bucket"),
    [
        pytest.param(True, None, id="reuploaded"),
        # The two routes that reached a manifest in the wild: a fetch that only
        # registers the bytes, and a fetch once the bucket default has moved.
        # Both preserved the recorded block while rewriting sha256/size_bytes.
        pytest.param(False, None, id="registered-without-upload"),
        pytest.param(True, "chronicle-raw", id="after-the-bucket-rename"),
    ],
)
def test_repeated_fetch_of_different_bytes_is_refused(
    tmp_path, monkeypatch, upload_r2, configured_bucket
):
    """A publisher revision must not inherit the recorded object's provenance."""
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)
    manifest_path = output_dir / "manifest.yaml"
    artifact_path = output_dir / REPUBLISHED_FILENAME
    _serve(monkeypatch, FIRST_PUBLICATION)
    first_report = _fetch_republished(output_dir, wrangler)
    manifest_before = manifest_path.read_bytes()
    uploads_before = log.read_text()

    if configured_bucket:
        monkeypatch.setenv("CHRONICLE_R2_RAW_BUCKET", configured_bucket)
    _serve(monkeypatch, SECOND_PUBLICATION)
    with pytest.raises(SourceArtifactRevisionError) as raised:
        _fetch_republished(output_dir, wrangler, upload_r2=upload_r2)

    message = str(raised.value)
    assert first_report.sha256 in message
    assert hashlib.sha256(SECOND_PUBLICATION).hexdigest() in message
    assert f"size_bytes={len(FIRST_PUBLICATION)}" in message
    assert f"size_bytes={len(SECOND_PUBLICATION)}" in message
    assert "release revision" in message
    assert "--record-revision" in message
    # Nothing was overwritten, copied or uploaded on the way to the refusal.
    assert manifest_path.read_bytes() == manifest_before
    assert artifact_path.read_bytes() == FIRST_PUBLICATION
    assert log.read_text() == uploads_before


def test_record_revision_writes_a_new_key_and_keeps_the_previous_object(
    tmp_path, monkeypatch
):
    """The opt-in records the new bytes' own key under the configured bucket."""
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)
    manifest_path = output_dir / "manifest.yaml"
    _serve(monkeypatch, FIRST_PUBLICATION)
    first_report = _fetch_republished(output_dir, wrangler)
    superseded = yaml.safe_load(manifest_path.read_text())["files"][2022]

    monkeypatch.setenv("CHRONICLE_R2_RAW_BUCKET", "chronicle-raw")
    _serve(monkeypatch, SECOND_PUBLICATION)
    report = _fetch_republished(output_dir, wrangler, record_revision=True)
    revised = yaml.safe_load(manifest_path.read_text())["files"][2022]
    revised_sha256 = hashlib.sha256(SECOND_PUBLICATION).hexdigest()

    assert report.valid
    # storage.r2 names the object that holds the entry's current bytes...
    assert revised["sha256"] == revised_sha256
    assert revised["size_bytes"] == len(SECOND_PUBLICATION)
    assert revised["storage"]["r2"]["bucket"] == "chronicle-raw"
    assert revised["storage"]["r2"]["key"] == (
        f"raw/irs_soi/soi-table-5/2022/{revised_sha256}/{REPUBLISHED_FILENAME}"
    )
    assert revised["storage"]["r2"]["uri"] == (
        f"r2://chronicle-raw/{revised['storage']['r2']['key']}"
    )
    # ...and never the superseded key, which stays addressable as history.
    previous = revised["storage"]["previous_r2"]
    assert [entry["uri"] for entry in previous] == [superseded["storage"]["r2"]["uri"]]
    assert previous[0]["bucket"] == "ledger-raw"
    assert previous[0]["sha256"] == first_report.sha256
    assert previous[0]["size_bytes"] == len(FIRST_PUBLICATION)
    assert previous[0]["fetched_at"] == superseded["fetched_at"]
    assert previous[0]["superseded_at"] == revised["fetched_at"]
    assert (output_dir / REPUBLISHED_FILENAME).read_bytes() == SECOND_PUBLICATION
    assert f"chronicle-raw/{revised['storage']['r2']['key']}" in log.read_text()


def test_a_revised_manifest_still_reads_as_one_r2_linked_artifact(
    tmp_path, monkeypatch
):
    """storage.previous_r2 is a sibling key, so every storage.r2 reader is intact."""
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    wrangler = _wrangler_stub(tmp_path, tmp_path / "wrangler.log")
    _serve(monkeypatch, FIRST_PUBLICATION)
    _fetch_republished(output_dir, wrangler)
    _serve(monkeypatch, SECOND_PUBLICATION)
    _fetch_republished(output_dir, wrangler, record_revision=True)

    inventory = inventory_source_artifacts(output_dir)

    assert inventory.valid
    assert inventory.counts["r2_link_count"] == 1
    assert inventory.counts["checksum_mismatch_count"] == 0
    assert inventory.entries[0].r2["bucket"] == "ledger-raw"
    assert inventory.entries[0].sha256_actual == (
        hashlib.sha256(SECOND_PUBLICATION).hexdigest()
    )


def test_publish_raw_refuses_a_file_the_recorded_object_does_not_hold(
    tmp_path, monkeypatch
):
    """Recorded sha256 != local sha256 is a revision, not a backfill."""
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)
    manifest_path = output_dir / "manifest.yaml"
    _serve(monkeypatch, FIRST_PUBLICATION)
    first_report = _fetch_republished(output_dir, wrangler)

    # Reproduce the state a pre-fix fetch left behind: new bytes on disk, the
    # entry's own hash rewritten, the recorded key still addressing the old
    # bytes.
    revised_sha256 = hashlib.sha256(SECOND_PUBLICATION).hexdigest()
    (output_dir / REPUBLISHED_FILENAME).write_bytes(SECOND_PUBLICATION)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["files"][2022]["sha256"] = revised_sha256
    manifest["files"][2022]["size_bytes"] = len(SECOND_PUBLICATION)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    manifest_before = manifest_path.read_bytes()
    uploads_before = log.read_text()

    report = publish_source_artifacts(output_dir, wrangler_command=str(wrangler))

    assert not report.valid
    assert report.entries[0].upload is None
    assert report.entries[0].r2_location is None
    assert report.entries[0].errors == (
        "recorded_r2_identity_mismatch:"
        f"recorded_sha256={first_report.sha256}:"
        f"recorded_filename={REPUBLISHED_FILENAME}:"
        f"local_sha256={revised_sha256}:"
        f"local_filename={REPUBLISHED_FILENAME}",
    )
    assert log.read_text() == uploads_before
    assert manifest_path.read_bytes() == manifest_before


def test_publish_raw_uploads_a_registered_revision_and_keeps_its_history(
    tmp_path, monkeypatch
):
    """Once the revision is registered, publishing it is ordinary work."""
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)
    manifest_path = output_dir / "manifest.yaml"
    _serve(monkeypatch, FIRST_PUBLICATION)
    first_report = _fetch_republished(output_dir, wrangler)
    _serve(monkeypatch, SECOND_PUBLICATION)
    _fetch_republished(output_dir, wrangler, record_revision=True)
    revised_sha256 = hashlib.sha256(SECOND_PUBLICATION).hexdigest()

    report = publish_source_artifacts(output_dir, wrangler_command=str(wrangler))
    published = yaml.safe_load(manifest_path.read_text())["files"][2022]

    assert report.valid
    assert published["storage"]["r2"]["key"].endswith(
        f"/{revised_sha256}/{REPUBLISHED_FILENAME}"
    )
    assert [entry["sha256"] for entry in published["storage"]["previous_r2"]] == [
        first_report.sha256
    ]


def test_fetch_artifact_cli_refuses_a_revision_then_records_it_on_request(
    tmp_path, monkeypatch, capsys
):
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    wrangler = _wrangler_stub(tmp_path, tmp_path / "wrangler.log")
    argv = [
        "fetch-artifact",
        "--url",
        REPUBLISHED_URL,
        "--source-id",
        "irs_soi",
        "--package-id",
        "soi-table-5",
        "--year",
        "2022",
        "--out-dir",
        str(output_dir),
        "--upload-r2",
        "--wrangler-command",
        str(wrangler),
    ]
    _serve(monkeypatch, FIRST_PUBLICATION)
    assert harness_main(argv) == 0
    capsys.readouterr()

    _serve(monkeypatch, SECOND_PUBLICATION)
    refused = harness_main(argv)
    refusal = capsys.readouterr()

    assert refused == 1
    assert "--record-revision" in refusal.err
    assert refusal.out == ""

    assert harness_main([*argv, "--record-revision"]) == 0
    recorded = json.loads(capsys.readouterr().out)

    assert recorded["sha256"] == hashlib.sha256(SECOND_PUBLICATION).hexdigest()
    assert recorded["r2_location"]["key"].endswith(
        f"/{recorded['sha256']}/{REPUBLISHED_FILENAME}"
    )


def test_record_revision_without_an_upload_records_no_current_object(
    tmp_path, monkeypatch
):
    """An offline revision keeps history without claiming the new bytes exist.

    Registering a revision without ``--upload-r2`` leaves the entry with no
    ``storage.r2`` at all rather than a pointer to bytes R2 does not hold. The
    superseded object stays addressable, and a later publish-raw completes the
    registration.
    """
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)
    manifest_path = output_dir / "manifest.yaml"
    _serve(monkeypatch, FIRST_PUBLICATION)
    first_report = _fetch_republished(output_dir, wrangler)

    _serve(monkeypatch, SECOND_PUBLICATION)
    _fetch_republished(output_dir, wrangler, upload_r2=False, record_revision=True)
    registered = yaml.safe_load(manifest_path.read_text())["files"][2022]

    assert "r2" not in registered["storage"]
    assert [entry["sha256"] for entry in registered["storage"]["previous_r2"]] == [
        first_report.sha256
    ]

    report = publish_source_artifacts(output_dir, wrangler_command=str(wrangler))
    published = yaml.safe_load(manifest_path.read_text())["files"][2022]
    revised_sha256 = hashlib.sha256(SECOND_PUBLICATION).hexdigest()

    assert report.valid
    assert published["storage"]["r2"]["key"].endswith(
        f"/{revised_sha256}/{REPUBLISHED_FILENAME}"
    )
    assert [entry["sha256"] for entry in published["storage"]["previous_r2"]] == [
        first_report.sha256
    ]


def _shared_archive_entry(content, *, package_id, year, filename="shared.zip"):
    sha256 = hashlib.sha256(content).hexdigest()
    key = f"raw/usda_snap/{package_id}/{year}/{sha256}/{filename}"
    return {
        "filename": filename,
        "source_url": "https://example.test/shared.zip",
        "sha256": sha256,
        "size_bytes": len(content),
        "fetched_at": "2026-05-11T11:57:29+00:00",
        "storage": {
            "r2": {
                "provider": "r2",
                "bucket": "ledger-raw",
                "key": key,
                "uri": f"r2://ledger-raw/{key}",
            }
        },
    }


def test_shared_archive_revision_is_refused_through_an_unregistered_owner(tmp_path):
    """A selected empty vintage cannot bypass another manifest's identity."""
    package = tmp_path / "db" / "data" / "usda_snap" / "fy69_to_current"
    package.mkdir(parents=True)
    original = b"USDA archive, first publication"
    revised = b"USDA archive, revised publication"
    filename = "snap-zip-fy69tocurrent-6.zip"
    (package / filename).write_bytes(original)
    primary_path = package / "manifest.yaml"
    primary_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "usda_snap",
                "package_id": "usda-snap-fy69-to-current",
                "files": {},
            },
            sort_keys=False,
        )
    )
    sibling_path = package / "manifest_fy2025_monthly_source_package.yaml"
    sibling_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "usda_snap",
                "package_id": "usda-snap-fy2025-monthly-state-caseloads",
                "files": {
                    2025: _shared_archive_entry(
                        original,
                        package_id="usda-snap-fy69-to-current",
                        year=2024,
                        filename=filename,
                    )
                },
            },
            sort_keys=False,
        )
    )
    publisher = _publish(tmp_path, filename, revised)
    before = {path: path.read_bytes() for path in (primary_path, sibling_path)}

    with pytest.raises(SourceArtifactRevisionError):
        fetch_source_artifact(
            str(publisher),
            source_id="usda_snap",
            package_id="usda-snap-fy69-to-current",
            year=2024,
            output_dir=package,
        )

    assert (package / filename).read_bytes() == original
    assert {path: path.read_bytes() for path in before} == before


def test_record_revision_updates_every_owner_of_usda_shared_archive(tmp_path):
    """The tracked USDA two-manifest shape has one physical archive."""
    package = tmp_path / "db" / "data" / "usda_snap" / "fy69_to_current"
    package.mkdir(parents=True)
    original = b"USDA archive, first publication"
    revised = b"USDA archive, revised publication"
    revised_sha256 = hashlib.sha256(revised).hexdigest()
    filename = "snap-zip-fy69tocurrent-6.zip"
    (package / filename).write_bytes(original)
    manifests = (
        (
            package / "manifest.yaml",
            "usda-snap-fy69-to-current",
            2024,
            "usda-snap-fy69-to-current",
            2024,
        ),
        (
            package / "manifest_fy2025_monthly_source_package.yaml",
            "usda-snap-fy2025-monthly-state-caseloads",
            2025,
            "usda-snap-fy69-to-current",
            2024,
        ),
    )
    previous_uris = {}
    for path, package_id, vintage, route_package, route_year in manifests:
        entry = _shared_archive_entry(
            original,
            package_id=route_package,
            year=route_year,
            filename=filename,
        )
        entry["source_table"] = f"owner {vintage}"
        previous_uris[path] = entry["storage"]["r2"]["uri"]
        path.write_text(
            yaml.safe_dump(
                {
                    "source_id": "usda_snap",
                    "package_id": package_id,
                    "files": {vintage: entry},
                },
                sort_keys=False,
            )
        )
    publisher = _publish(tmp_path, filename, revised)

    fetch_source_artifact(
        str(publisher),
        source_id="usda_snap",
        package_id="usda-snap-fy69-to-current",
        year=2024,
        output_dir=package,
        record_revision=True,
    )

    assert (package / filename).read_bytes() == revised
    for path, _package_id, vintage, _route_package, _route_year in manifests:
        entry = yaml.safe_load(path.read_text())["files"][vintage]
        assert entry["sha256"] == revised_sha256
        assert entry["size_bytes"] == len(revised)
        assert entry["source_table"] == f"owner {vintage}"
        assert "r2" not in entry["storage"]
        assert [item["uri"] for item in entry["storage"]["previous_r2"]] == [
            previous_uris[path]
        ]


def test_record_revision_updates_every_same_manifest_owner(tmp_path):
    """SSA-style semantic aliases of one file share one byte identity."""
    package = tmp_path / "db" / "data" / "ssa" / "supplement"
    package.mkdir(parents=True)
    original = b"SSA extracted table, first publication"
    revised = b"SSA extracted table, revised publication"
    revised_sha256 = hashlib.sha256(revised).hexdigest()
    filename = "ssa_oasdi_ssi_2024.csv"
    (package / filename).write_bytes(original)
    manifest_path = package / "manifest.yaml"
    entries = {
        2024: _shared_archive_entry(
            original,
            package_id="ssa-annual-statistical-supplement-2025",
            year=2024,
            filename=filename,
        ),
        "extracted_targets": _shared_archive_entry(
            original,
            package_id="ssa-annual-statistical-supplement-2025",
            year="extracted_targets",
            filename=filename,
        ),
    }
    for entry in entries.values():
        entry["source_url"] = "https://example.test/ssa.csv"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "ssa",
                "package_id": "ssa-annual-statistical-supplement-2025",
                "files": entries,
            },
            sort_keys=False,
        )
    )
    publisher = _publish(tmp_path, filename, revised)

    fetch_source_artifact(
        str(publisher),
        source_id="ssa",
        package_id="ssa-annual-statistical-supplement-2025",
        year=2024,
        output_dir=package,
        record_revision=True,
    )

    updated = yaml.safe_load(manifest_path.read_text())["files"]
    assert {entry["sha256"] for entry in updated.values()} == {revised_sha256}
    assert {
        item["sha256"]
        for entry in updated.values()
        for item in entry["storage"]["previous_r2"]
    } == {hashlib.sha256(original).hexdigest()}


def test_a_recorded_block_that_only_carries_a_uri_is_still_recognized(
    tmp_path, monkeypatch
):
    """Identity reads the URI when a hand-written block records no key."""
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    wrangler = _wrangler_stub(tmp_path, tmp_path / "wrangler.log")
    manifest_path = output_dir / "manifest.yaml"
    _serve(monkeypatch, FIRST_PUBLICATION)
    _fetch_republished(output_dir, wrangler)
    manifest = yaml.safe_load(manifest_path.read_text())
    recorded = manifest["files"][2022]["storage"]["r2"]
    manifest["files"][2022]["storage"]["r2"] = {
        "provider": recorded["provider"],
        "bucket": recorded["bucket"],
        "uri": recorded["uri"],
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    uri_only = manifest["files"][2022]["storage"]["r2"]

    report = _fetch_republished(output_dir, wrangler)
    preserved = yaml.safe_load(manifest_path.read_text())["files"][2022]

    assert report.valid
    assert preserved["storage"]["r2"] == uri_only

    _serve(monkeypatch, SECOND_PUBLICATION)
    with pytest.raises(SourceArtifactRevisionError):
        _fetch_republished(output_dir, wrangler)


# ---------------------------------------------------------------------------
# Manifest addressing, entry identity, and recorded locators
#
# Everything below concerns the state a fetch reads before it writes: which
# manifest it reads, what that manifest's entry says its vintage holds, and
# whether the recorded R2 block names one object or two.
# ---------------------------------------------------------------------------

TRADITIONAL_MANIFEST = "manifest_traditional_source_package.yaml"
ROTH_MANIFEST = "manifest_roth_source_package.yaml"


def _publish(tmp_path, name, content):
    """Write bytes a fetch can read as a local publisher path."""
    path = tmp_path / "publisher" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _fetch_local(output_dir, source_path, *, package_id="soi-table-5", **kwargs):
    return fetch_source_artifact(
        str(source_path),
        source_id="irs_soi",
        package_id=package_id,
        year=2022,
        output_dir=output_dir,
        **kwargs,
    )


def _entry(manifest_path):
    return yaml.safe_load(manifest_path.read_text())["files"][2022]


def test_fetch_artifact_writes_the_manifest_it_was_given(tmp_path):
    """One publisher directory, two source packages, two manifests.

    db/data/irs_soi/ira_contributions keeps the traditional and Roth IRA
    packages side by side. A fetch that always wrote manifest.yaml would write
    a third manifest neither package reads.
    """
    package = tmp_path / "db" / "data" / "irs_soi" / "ira_contributions"
    traditional = _publish(tmp_path, "22in05ira.xlsx", b"traditional IRA table")
    roth = _publish(tmp_path, "22in06ira.xlsx", b"roth IRA table")
    package.mkdir(parents=True)
    for name, package_id in (
        (TRADITIONAL_MANIFEST, "soi-ira-traditional-contributions-2022"),
        (ROTH_MANIFEST, "soi-ira-roth-contributions-2022"),
    ):
        (package / name).write_text(
            yaml.safe_dump(
                {"source_id": "irs_soi", "package_id": package_id, "files": {}},
                sort_keys=False,
            )
        )

    _fetch_local(
        package,
        traditional,
        package_id="soi-ira-traditional-contributions-2022",
        manifest_filename=TRADITIONAL_MANIFEST,
    )
    _fetch_local(
        package,
        roth,
        package_id="soi-ira-roth-contributions-2022",
        manifest_filename=ROTH_MANIFEST,
    )

    assert sorted(path.name for path in package.glob("manifest*.yaml")) == [
        ROTH_MANIFEST,
        TRADITIONAL_MANIFEST,
    ]
    assert not (package / "manifest.yaml").exists()
    assert _entry(package / TRADITIONAL_MANIFEST)["filename"] == "22in05ira.xlsx"
    assert _entry(package / ROTH_MANIFEST)["filename"] == "22in06ira.xlsx"
    assert _entry(package / TRADITIONAL_MANIFEST)["sha256"] == (
        hashlib.sha256(b"traditional IRA table").hexdigest()
    )


def test_a_revision_is_refused_in_the_manifest_that_records_it(tmp_path):
    """The IRA revision workflow the docs cite, on a two-manifest package."""
    package = tmp_path / "db" / "data" / "irs_soi" / "ira_contributions"
    traditional = _publish(tmp_path, "22in05ira.xlsx", b"traditional IRA table")
    _fetch_local(
        package,
        traditional,
        package_id="soi-ira-traditional-contributions-2022",
        manifest_filename=TRADITIONAL_MANIFEST,
    )
    recorded = (package / TRADITIONAL_MANIFEST).read_bytes()

    # The IRS re-publishes under the same URL and vintage.
    traditional.write_bytes(b"traditional IRA table, revised rows")

    with pytest.raises(SourceArtifactRevisionError) as raised:
        _fetch_local(
            package,
            traditional,
            package_id="soi-ira-traditional-contributions-2022",
            manifest_filename=TRADITIONAL_MANIFEST,
        )

    assert TRADITIONAL_MANIFEST in str(raised.value)
    assert (package / TRADITIONAL_MANIFEST).read_bytes() == recorded
    assert (package / "22in05ira.xlsx").read_bytes() == b"traditional IRA table"

    # Without the flag the same fetch would address a manifest.yaml that no
    # package reads and that protects nothing: the #225 path. It is refused,
    # naming the manifests the directory keeps, and nothing is written.
    with pytest.raises(AmbiguousManifestError) as stray:
        _fetch_local(
            package,
            traditional,
            package_id="soi-ira-traditional-contributions-2022",
        )

    assert TRADITIONAL_MANIFEST in str(stray.value)
    assert "--manifest" in str(stray.value)
    assert not (package / "manifest.yaml").exists()
    assert (package / TRADITIONAL_MANIFEST).read_bytes() == recorded
    assert (package / "22in05ira.xlsx").read_bytes() == b"traditional IRA table"


def test_fetch_artifact_cli_refuses_a_stray_default_manifest(tmp_path, capsys):
    package = tmp_path / "db" / "data" / "irs_soi" / "ira_contributions"
    traditional = _publish(tmp_path, "22in05ira.xlsx", b"traditional IRA table")
    _fetch_local(
        package,
        traditional,
        package_id="soi-ira-traditional-contributions-2022",
        manifest_filename=TRADITIONAL_MANIFEST,
    )
    argv = [
        "fetch-artifact",
        "--url",
        str(traditional),
        "--source-id",
        "irs_soi",
        "--package-id",
        "soi-ira-traditional-contributions-2022",
        "--year",
        "2022",
        "--out-dir",
        str(package),
    ]

    assert harness_main(argv) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert TRADITIONAL_MANIFEST in err
    assert not (package / "manifest.yaml").exists()


def test_fetch_refuses_a_stray_default_beside_a_case_variant_named_manifest(
    tmp_path, monkeypatch
):
    package = tmp_path / "db" / "data" / "irs_soi" / "ira_contributions"
    package.mkdir(parents=True)
    named_manifest = package / "MANIFEST_TRADITIONAL.YML"
    named_manifest.write_text("source_id: irs_soi\nfiles: {}\n")
    source = _publish(tmp_path, "22in05ira.xlsx", b"traditional IRA table")

    def unexpected_read(_source_url):
        raise AssertionError("a case-variant named manifest did not block I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(AmbiguousManifestError, match="MANIFEST_TRADITIONAL.YML"):
        _fetch_local(package, source)

    assert not (package / "manifest.yaml").exists()


def test_a_same_bytes_rename_is_refused_by_name_not_as_a_revision(tmp_path):
    """Identical bytes under another filename are neither a revision nor a
    re-fetch: the entry's filename must keep agreeing with its recorded key."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    source = _publish(tmp_path, "22in05ira.xlsx", b"IRA table 5")
    _fetch_local(package, source, filename="table-5.xlsx", upload_r2=False)
    recorded = (package / "manifest.yaml").read_bytes()

    for record_revision in (False, True):
        with pytest.raises(SourceArtifactRevisionError) as raised:
            _fetch_local(
                package, source, upload_r2=False, record_revision=record_revision
            )
        message = str(raised.value)
        assert "rename is not a release revision" in message
        assert "filename=table-5.xlsx" in message
        assert "names them 22in05ira.xlsx" in message
        assert "--filename table-5.xlsx" in message

    assert (package / "manifest.yaml").read_bytes() == recorded
    assert not (package / "22in05ira.xlsx").exists()
    assert (package / "table-5.xlsx").read_bytes() == b"IRA table 5"


@pytest.mark.parametrize(
    "manifest_filename",
    ["../manifest.yaml", "nested/manifest.yaml", "", "   ", ".", ".."],
)
def test_a_manifest_name_must_stay_inside_the_package(tmp_path, manifest_filename):
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    source = _publish(tmp_path, "table.xlsx", b"table")

    with pytest.raises(ValueError, match="inside the package directory"):
        _fetch_local(package, source, manifest_filename=manifest_filename)

    assert not package.exists()


@pytest.mark.parametrize(
    ("source_url", "filename", "message"),
    [
        pytest.param("publisher.csv", "manifest.yaml", "manifest name", id="default"),
        pytest.param(
            "publisher.csv", "MANIFEST_NAMED.YML", "manifest name", id="named"
        ),
        pytest.param(
            "publisher.csv", "nested/publisher.csv", "bare filename", id="nested"
        ),
        pytest.param(
            "https://publisher.test/manifest.yaml",
            None,
            "manifest name",
            id="inferred",
        ),
    ],
)
def test_artifact_filename_is_refused_before_publisher_io(
    tmp_path, monkeypatch, source_url, filename, message
):
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text("source_id: irs_soi\npackage_id: soi-table\nfiles: {}\n")
    before = manifest_path.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("an invalid artifact filename reached publisher I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(SourceArtifactManifestError, match=message):
        fetch_source_artifact(
            source_url,
            source_id="irs_soi",
            package_id="soi-table",
            year=2024,
            output_dir=package,
            filename=filename,
        )

    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    ("source_id", "package_id"),
    [
        pytest.param("/", "package", id="source-id"),
        pytest.param("publisher", "/", id="package-id"),
    ],
)
def test_fetch_refuses_invalid_r2_identity_before_publisher_io(
    tmp_path, monkeypatch, source_id, package_id
):
    package = tmp_path / "db" / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    artifact_path = package / "table.csv"
    artifact_path.write_bytes(b"registered publisher bytes")

    def unexpected_read(_source_url):
        raise AssertionError("an invalid R2 identity reached publisher I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(ValueError, match="canonical R2 key segment"):
        fetch_source_artifact(
            "https://publisher.test/table.csv",
            source_id=source_id,
            package_id=package_id,
            year=2024,
            output_dir=package,
        )

    assert artifact_path.read_bytes() == b"registered publisher bytes"
    assert not (package / "manifest.yaml").exists()


def test_manifest_name_must_be_discoverable_before_publisher_io(tmp_path, monkeypatch):
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table"
    source = _publish(tmp_path, "table.csv", b"publisher table")

    def unexpected_read(_source_url):
        raise AssertionError("an undiscoverable manifest name reached publisher I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(ManifestNameError, match="invisible"):
        _fetch_local(package, source, manifest_filename="custom.yaml")

    assert not package.exists()


def test_fetch_artifact_cli_reports_a_manifest_name_outside_the_package(
    tmp_path, capsys
):
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    source = _publish(tmp_path, "table.xlsx", b"table")
    argv = [
        "fetch-artifact",
        "--url",
        str(source),
        "--source-id",
        "irs_soi",
        "--package-id",
        "soi-table-5",
        "--year",
        "2022",
        "--out-dir",
        str(package),
        "--manifest",
        "../manifest.yaml",
    ]

    assert harness_main(argv) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "inside the package directory" in err
    assert not package.exists()
    assert not (tmp_path / "db" / "data" / "irs_soi" / "manifest.yaml").exists()


def test_fetch_artifact_cli_targets_the_named_manifest(tmp_path, capsys):
    package = tmp_path / "db" / "data" / "irs_soi" / "ira_contributions"
    traditional = _publish(tmp_path, "22in05ira.xlsx", b"traditional IRA table")
    argv = [
        "fetch-artifact",
        "--url",
        str(traditional),
        "--source-id",
        "irs_soi",
        "--package-id",
        "soi-ira-traditional-contributions-2022",
        "--year",
        "2022",
        "--out-dir",
        str(package),
        "--manifest",
        TRADITIONAL_MANIFEST,
    ]

    assert harness_main(argv) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["manifest_path"].endswith(TRADITIONAL_MANIFEST)
    assert not (package / "manifest.yaml").exists()

    traditional.write_bytes(b"traditional IRA table, revised rows")

    assert harness_main(argv) == 1
    assert TRADITIONAL_MANIFEST in capsys.readouterr().err


@pytest.mark.parametrize(
    ("existing_name", "requested_name"),
    [
        pytest.param("manifest.yml", "manifest.yaml", id="yml-default"),
        pytest.param("Manifest.yaml", "manifest.yaml", id="case-variant-default"),
        pytest.param(
            "manifest_monthly_source_package.yaml",
            "manifest_monthy_source_package.yaml",
            id="mistyped-named-manifest",
        ),
    ],
)
def test_fetch_refuses_to_create_any_manifest_beside_an_existing_registry(
    tmp_path, monkeypatch, existing_name, requested_name
):
    package = tmp_path / "db" / "data" / "usda_snap" / "fy69_to_current"
    package.mkdir(parents=True)
    existing = package / existing_name
    existing.write_text(
        "source_id: usda_snap\npackage_id: usda-snap-fy69-to-current\nfiles: {}\n"
    )
    before = existing.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("ambiguous manifest creation reached publisher I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(AmbiguousManifestError, match=existing_name):
        fetch_source_artifact(
            "https://example.test/snap.zip",
            source_id="usda_snap",
            package_id="usda-snap-fy69-to-current",
            year=2024,
            output_dir=package,
            manifest_filename=requested_name,
        )

    assert existing.read_bytes() == before
    assert requested_name not in {path.name for path in package.iterdir()}


def test_fetch_refuses_a_symlinked_manifest_before_publisher_io(tmp_path, monkeypatch):
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table"
    package.mkdir(parents=True)
    outside_manifest = tmp_path / "outside-manifest.yaml"
    outside_manifest.write_text(
        "source_id: irs_soi\npackage_id: soi-table\nfiles: {}\n"
    )
    manifest_path = package / "manifest.yaml"
    manifest_path.symlink_to(outside_manifest)
    before = outside_manifest.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("a symlinked manifest reached publisher I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(MalformedManifestError, match="symlink"):
        fetch_source_artifact(
            "https://example.test/table.xlsx",
            source_id="irs_soi",
            package_id="soi-table",
            year=2024,
            output_dir=package,
        )

    assert manifest_path.is_symlink()
    assert outside_manifest.read_bytes() == before


def test_fetch_refuses_physically_distinct_normalized_manifest_aliases(
    tmp_path, monkeypatch
):
    package = tmp_path / "db" / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text("source_id: publisher\npackage_id: package\nfiles: {}\n")
    case_alias = package / "Manifest.yaml"
    monkeypatch.setattr(
        "chronicle.artifacts.package_manifest_paths",
        lambda _package: [manifest_path, case_alias],
    )

    def unexpected_read(_source_url):
        raise AssertionError("normalized manifest aliases reached publisher I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(AmbiguousManifestError, match="normalized manifest name"):
        fetch_source_artifact(
            "https://example.test/table.csv",
            source_id="publisher",
            package_id="package",
            year=2024,
            output_dir=package,
        )


def test_fetch_refuses_a_symlinked_artifact_target_before_publisher_io(
    tmp_path, monkeypatch
):
    package = tmp_path / "db" / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text("source_id: publisher\npackage_id: package\nfiles: {}\n")
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"outside bytes")
    artifact_path = package / "table.csv"
    artifact_path.symlink_to(outside)
    before = {manifest_path: manifest_path.read_bytes(), outside: outside.read_bytes()}

    def unexpected_read(_source_url):
        raise AssertionError("symlinked artifact target reached publisher I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(ArtifactFilenameError, match="symbolic link"):
        fetch_source_artifact(
            "https://example.test/table.csv",
            source_id="publisher",
            package_id="package",
            year=2024,
            output_dir=package,
        )

    assert artifact_path.is_symlink()
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "manifest_filename",
    [
        pytest.param("../manifest.yaml", id="parent"),
        pytest.param("manifest_*.yaml", id="star-glob"),
        pytest.param("manifest_?.yml", id="question-glob"),
        pytest.param("manifest_[ab].yaml", id="character-class-glob"),
    ],
)
def test_sweep_manifest_selector_must_be_a_literal_supported_filename(
    tmp_path, manifest_filename
):
    root = tmp_path / "requested-root"
    package = root / "package"
    package.mkdir(parents=True)
    content = b"publisher table"
    (package / "table.csv").write_bytes(content)
    (package / "manifest_a.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "publisher",
                "package_id": "package",
                "files": {
                    2024: {
                        "filename": "table.csv",
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                },
            },
            sort_keys=False,
        )
    )
    outside_manifest = root.parent / "manifest.yaml"
    outside_manifest.write_text("files: {}\n")
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)
    before = {
        path: path.read_bytes()
        for path in (package / "manifest_a.yaml", outside_manifest)
    }

    with pytest.raises(ManifestNameError, match="Manifest"):
        publish_source_artifacts(
            root,
            manifest_filename=manifest_filename,
            wrangler_command=str(wrangler),
        )

    assert {path: path.read_bytes() for path in before} == before
    assert not log.exists()


@pytest.mark.parametrize(
    "operation", [inventory_source_artifacts, publish_source_artifacts]
)
def test_invalid_sweep_manifest_selector_is_refused_even_when_root_is_missing(
    tmp_path, operation
):
    with pytest.raises(ManifestNameError):
        operation(tmp_path / "missing", manifest_filename="../manifest.yaml")


@pytest.mark.parametrize("command", ["inventory-artifacts", "publish-raw"])
def test_sweep_cli_reports_an_invalid_manifest_selector(command, tmp_path, capsys):
    exit_code = harness_main(
        [
            command,
            "--root",
            str(tmp_path),
            "--manifest",
            "../manifest.yaml",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")


# ---------------------------------------------------------------------------
# Identity without a recorded R2 object
# ---------------------------------------------------------------------------


def _failing_wrangler(tmp_path, log):
    wrangler = tmp_path / "failing-wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\nexit 1\n")
    wrangler.chmod(0o755)
    return wrangler


def test_a_registered_entry_is_protected_before_it_is_ever_published(tmp_path):
    """No storage.r2 yet is not no identity: the entry declares its bytes."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    source = _publish(tmp_path, "22in05ira.xlsx", b"IRA table 5, first publication")
    first = _fetch_local(package, source, upload_r2=False)
    recorded = (package / "manifest.yaml").read_bytes()

    assert "storage" not in _entry(package / "manifest.yaml")

    # Same bytes: an ordinary repeated fetch, not a revision.
    assert _fetch_local(package, source, upload_r2=False).sha256 == first.sha256

    source.write_bytes(b"IRA table 5, silently re-published")
    with pytest.raises(SourceArtifactRevisionError) as raised:
        _fetch_local(package, source, upload_r2=False)

    message = str(raised.value)
    assert first.sha256 in message
    assert hashlib.sha256(b"IRA table 5, silently re-published").hexdigest() in message
    assert "size_bytes=30" in message
    assert "--record-revision" in message
    assert (package / "manifest.yaml").read_bytes() == recorded
    assert (package / "22in05ira.xlsx").read_bytes() == (
        b"IRA table 5, first publication"
    )


def test_a_failed_upload_does_not_disable_revision_protection(tmp_path):
    """The state #225 hit: bytes registered, upload failed, no storage.r2."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    log = tmp_path / "wrangler.log"
    wrangler = _failing_wrangler(tmp_path, log)
    source = _publish(tmp_path, "22in05ira.xlsx", b"IRA table 5, first publication")

    report = _fetch_local(
        package, source, upload_r2=True, wrangler_command=str(wrangler)
    )
    recorded = (package / "manifest.yaml").read_bytes()

    assert report.errors == ("r2_upload_failed",)
    assert "storage" not in _entry(package / "manifest.yaml")

    source.write_bytes(b"IRA table 5, silently re-published")
    with pytest.raises(SourceArtifactRevisionError):
        _fetch_local(package, source, upload_r2=True, wrangler_command=str(wrangler))

    assert (package / "manifest.yaml").read_bytes() == recorded


def test_record_revision_over_an_unpublished_entry_supersedes_nothing(tmp_path):
    """There is no object to keep, so the entry gets no previous_r2 key."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    source = _publish(tmp_path, "22in05ira.xlsx", b"IRA table 5, first publication")
    _fetch_local(package, source, upload_r2=False)

    source.write_bytes(b"IRA table 5, silently re-published")
    report = _fetch_local(package, source, upload_r2=False, record_revision=True)
    revised = _entry(package / "manifest.yaml")

    assert report.valid
    assert revised["sha256"] == (
        hashlib.sha256(b"IRA table 5, silently re-published").hexdigest()
    )
    assert "storage" not in revised


# ---------------------------------------------------------------------------
# Recorded locator cross-checks
# ---------------------------------------------------------------------------


def _recorded_package(tmp_path, content=b"IRA table 5, first publication"):
    """A package whose entry records a published, content-addressed object."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    wrangler = _wrangler_stub(tmp_path, tmp_path / "wrangler.log")
    source = _publish(tmp_path, "22in05ira.xlsx", content)
    report = _fetch_local(
        package, source, upload_r2=True, wrangler_command=str(wrangler)
    )
    return package, source, report


def _rewrite_recorded_r2(package, mutate, manifest="manifest.yaml"):
    manifest_path = package / manifest
    payload = yaml.safe_load(manifest_path.read_text())
    mutate(payload["files"][2022]["storage"])
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return manifest_path


def _other_sha256():
    return hashlib.sha256(b"some other object entirely").hexdigest()


@pytest.mark.parametrize(
    "previous_r2",
    [
        pytest.param({}, id="mapping"),
        pytest.param("not a list", id="scalar"),
        pytest.param(None, id="null"),
    ],
)
def test_fetch_refuses_non_list_previous_r2_before_publisher_io(
    tmp_path, monkeypatch, previous_r2
):
    """Malformed archived provenance must not be replaced by a new history."""
    package, source, _report = _recorded_package(tmp_path)
    manifest_path = _rewrite_recorded_r2(
        package,
        lambda storage: storage.__setitem__("previous_r2", previous_r2),
    )
    artifact_path = package / "22in05ira.xlsx"
    before = {
        manifest_path: manifest_path.read_bytes(),
        artifact_path: artifact_path.read_bytes(),
    }
    source.write_bytes(b"IRA table 5, revised publication")

    def unexpected_read(_source_url):
        raise AssertionError("malformed previous_r2 reached publisher I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(
        MalformedManifestError,
        match=r"storage[.]previous_r2 must be a list",
    ):
        _fetch_local(package, source, upload_r2=False, record_revision=True)

    assert {path: path.read_bytes() for path in before} == before


def _contradict_key(storage):
    key = storage["r2"]["key"]
    storage["r2"]["key"] = key.replace(key.split("/")[-2], _other_sha256())


def _contradict_bucket(storage):
    storage["r2"]["bucket"] = "some-other-bucket"


def _contradict_provider(storage):
    storage["r2"]["provider"] = "s3"


def _mangle_uri(storage):
    storage["r2"]["uri"] = "r2:/ledger-raw-missing-a-slash"


def _drop_the_locator(storage):
    storage["r2"] = {"provider": "r2", "bucket": "ledger-raw"}


def _flatten_the_key(storage):
    storage["r2"]["key"] = "raw/irs_soi/22in05ira.xlsx"
    storage["r2"]["uri"] = f"r2://ledger-raw/{storage['r2']['key']}"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(_contradict_key, "contradicts uri", id="key-vs-uri"),
        pytest.param(_contradict_bucket, "contradicts uri", id="bucket-vs-uri"),
        pytest.param(
            _contradict_provider, "does not identify R2", id="provider-vs-uri"
        ),
        pytest.param(_mangle_uri, "is not provider://bucket/key", id="uri-shape"),
        pytest.param(_drop_the_locator, "records no uri", id="no-locator"),
        pytest.param(
            _flatten_the_key, "is not content-addressed", id="not-content-addressed"
        ),
    ],
)
def test_a_recorded_block_that_names_two_objects_is_refused(tmp_path, mutate, expected):
    """A contradictory locator is an error, never a silently preserved block.

    The key-vs-uri case is the one that used to pass: identity was read from
    the key alone, so a block whose uri named different bytes was carried
    forward verbatim, and the manifest kept publishing a URI for an object it
    no longer described.
    """
    package, source, _ = _recorded_package(tmp_path)
    manifest_path = _rewrite_recorded_r2(package, mutate)
    recorded = manifest_path.read_bytes()

    # Identical bytes: the fetch would otherwise preserve the recorded block.
    with pytest.raises(RecordedR2LocatorError) as raised:
        _fetch_local(package, source, upload_r2=False)

    assert expected in str(raised.value)
    assert manifest_path.read_bytes() == recorded


def test_a_malformed_storage_block_is_not_treated_as_absent(tmp_path):
    package, source, _ = _recorded_package(tmp_path)
    manifest_path = _rewrite_recorded_r2(
        package, lambda storage: storage.update({"r2": ["r2://ledger-raw/raw/key"]})
    )
    recorded = manifest_path.read_bytes()

    with pytest.raises(MalformedManifestError, match="must be a mapping"):
        _fetch_local(package, source, upload_r2=False)

    assert manifest_path.read_bytes() == recorded


def test_publish_raw_refuses_a_contradictory_recorded_block(tmp_path):
    """Nothing is uploaded under a block that does not name one object."""
    package, _, _ = _recorded_package(tmp_path)
    log = tmp_path / "publish.log"
    wrangler = _wrangler_stub(tmp_path, log)
    manifest_path = _rewrite_recorded_r2(package, _contradict_key)
    recorded = manifest_path.read_bytes()

    report = publish_source_artifacts(package, wrangler_command=str(wrangler))

    assert not report.valid
    assert report.entries[0].upload is None
    assert report.entries[0].errors[0].startswith("recorded_r2_locator_invalid:")
    assert "contradicts uri" in report.entries[0].errors[0]
    assert not log.exists()
    assert manifest_path.read_bytes() == recorded


# ---------------------------------------------------------------------------
# Malformed manifests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "document",
    [
        pytest.param("- one entry\n- another\n", id="list"),
        pytest.param("a bare scalar\n", id="scalar"),
        pytest.param("files: [\n", id="unparseable"),
    ],
)
def test_a_malformed_manifest_is_refused_before_anything_is_fetched(tmp_path, document):
    """Not an absent manifest: refusing it protects what it still records.

    The publisher path does not exist, so reaching the fetch at all would raise
    FileNotFoundError instead. Getting MalformedManifestError is what says the
    manifest was read and refused first.
    """
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(document)

    with pytest.raises(MalformedManifestError):
        _fetch_local(package, tmp_path / "publisher" / "never-read.xlsx")

    assert manifest_path.read_text() == document
    assert list(package.iterdir()) == [manifest_path]


@pytest.mark.parametrize(
    "document",
    [
        "files:\n- not a mapping\n",
        "files: 3\n",
        "source_id: irs_soi\nfiles: text\n",
    ],
)
def test_a_non_mapping_files_block_is_refused_before_anything_is_fetched(
    tmp_path, document
):
    """The same document inventory-artifacts and publish-raw report as
    'files must be a mapping'; a fetch must not overwrite the artifact first."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(document)

    with pytest.raises(MalformedManifestError, match="files must be a mapping"):
        _fetch_local(package, tmp_path / "publisher" / "never-read.xlsx")

    assert manifest_path.read_text() == document
    assert list(package.iterdir()) == [manifest_path]


@pytest.mark.parametrize(
    "document",
    [
        "",
        "\n",
        "{}\n",
        "# only a comment\n",
        "files:\n",
        "source_id: irs_soi\nfiles:\n",
    ],
)
def test_an_empty_manifest_still_reads_as_absent(tmp_path, document):
    """Including a bare ``files:`` line, which parses as an explicit null: the
    fetch records into a fresh mapping rather than failing after the write."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    package.mkdir(parents=True)
    (package / "manifest.yaml").write_text(document)
    source = _publish(tmp_path, "22in05ira.xlsx", b"IRA table 5")

    report = _fetch_local(package, source, upload_r2=False)

    assert report.valid
    assert _entry(package / "manifest.yaml")["filename"] == "22in05ira.xlsx"


def test_a_malformed_manifest_is_reported_by_inventory_and_publish(tmp_path):
    """Neither sweep may crash on, or silently skip, a document it cannot read."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    package.mkdir(parents=True)
    (package / "manifest.yaml").write_text("- not a mapping\n")

    inventory = inventory_source_artifacts(package)
    published = publish_source_artifacts(package)

    assert not inventory.valid
    assert inventory.entries == ()
    assert "must be a YAML mapping" in inventory.errors[0]
    assert not published.valid
    assert published.entries == ()
    assert "must be a YAML mapping" in published.errors[0]


@pytest.mark.parametrize(
    "duplicate_document",
    [
        pytest.param(
            "source_id: hidden_source\n"
            "source_id: irs_soi\n"
            "package_id: soi-table-5\n"
            "files: {}\n",
            id="source-id",
        ),
        pytest.param(
            "source_id: irs_soi\n"
            "package_id: hidden-package\n"
            "package_id: soi-table-5\n"
            "files: {}\n",
            id="package-id",
        ),
        pytest.param(
            "source_id: irs_soi\n"
            "package_id: soi-table-5\n"
            "files:\n"
            "  2022:\n"
            "    filename: hidden.xlsx\n"
            f"    sha256: {hashlib.sha256(b'hidden bytes').hexdigest()}\n"
            "files: {}\n",
            id="files",
        ),
        pytest.param(
            "source_id: irs_soi\n"
            "package_id: soi-table-5\n"
            "files:\n"
            "  2022:\n"
            "    filename: hidden.xlsx\n"
            f"    sha256: {hashlib.sha256(b'hidden bytes').hexdigest()}\n"
            "  2022: {}\n",
            id="vintage",
        ),
    ],
)
def test_fetch_refuses_duplicate_manifest_keys_before_publisher_io(
    tmp_path, monkeypatch, duplicate_document
):
    """A lossy YAML parse must never decide which identity gets rewritten."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(duplicate_document)
    before = manifest_path.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("duplicate manifest keys reached publisher I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(MalformedManifestError, match="duplicate key"):
        fetch_source_artifact(
            "https://example.test/table.xlsx",
            source_id="irs_soi",
            package_id="soi-table-5",
            year=2022,
            output_dir=package,
        )

    assert manifest_path.read_bytes() == before


# ---------------------------------------------------------------------------
# Sol gate round 3: fetch preflight and in-place manifest updates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "mismatched_field",
        "declared_source_id",
        "declared_package_id",
        "source_id",
        "package_id",
    ),
    [
        pytest.param(
            "source_id",
            "other_source",
            "requested-package",
            "requested_source",
            "requested-package",
            id="source-id",
        ),
        pytest.param(
            "package_id",
            "usda_snap",
            "usda-snap-fy69-to-current",
            "usda_snap",
            "usda-snap-fy2025-monthly-state-caseloads",
            id="package-id",
        ),
    ],
)
def test_fetch_refuses_a_selected_manifest_for_another_package_before_io(
    tmp_path,
    monkeypatch,
    mismatched_field,
    declared_source_id,
    declared_package_id,
    source_id,
    package_id,
):
    package = tmp_path / "db" / "data" / "usda_snap" / "fy69_to_current"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": declared_source_id,
                "package_id": declared_package_id,
                "files": {},
            },
            sort_keys=False,
        )
    )
    before = manifest_path.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("a mismatched manifest must be refused before I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(SourceArtifactManifestError) as raised:
        fetch_source_artifact(
            "https://example.test/snap-zip-fy69tocurrent-6.zip",
            source_id=source_id,
            package_id=package_id,
            year=2025,
            output_dir=package,
        )

    message = str(raised.value)
    declared = {
        "source_id": declared_source_id,
        "package_id": declared_package_id,
    }[mismatched_field]
    requested = {"source_id": source_id, "package_id": package_id}[mismatched_field]
    assert f"{mismatched_field}={declared!r}" in message
    assert f"{mismatched_field}={requested!r}" in message
    assert manifest_path.read_bytes() == before
    assert list(package.iterdir()) == [manifest_path]


def test_fetch_uses_a_quoted_year_key_for_revision_protection(tmp_path):
    package = tmp_path / "db" / "data" / "irs_soi" / "table"
    package.mkdir(parents=True)
    artifact_path = package / "table.xlsx"
    original = b"original publisher bytes"
    revised = b"silently revised publisher bytes"
    artifact_path.write_bytes(original)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "irs_soi",
                "package_id": "soi-table",
                "files": {
                    "2024": {
                        "filename": artifact_path.name,
                        "source_url": "https://example.test/table.xlsx",
                        "sha256": hashlib.sha256(original).hexdigest(),
                        "size_bytes": len(original),
                    }
                },
            },
            sort_keys=False,
        )
    )
    source = _publish(tmp_path, artifact_path.name, revised)
    before = manifest_path.read_bytes()

    with pytest.raises(SourceArtifactRevisionError):
        fetch_source_artifact(
            str(source),
            source_id="irs_soi",
            package_id="soi-table",
            year=2024,
            output_dir=package,
        )

    assert manifest_path.read_bytes() == before
    assert artifact_path.read_bytes() == original


def test_fetch_refuses_both_spellings_of_one_year_before_io(tmp_path, monkeypatch):
    package = tmp_path / "db" / "data" / "irs_soi" / "table"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "irs_soi",
                "package_id": "soi-table",
                "files": {
                    2024: {"filename": "numeric.xlsx"},
                    "2024": {"filename": "quoted.xlsx"},
                },
            },
            sort_keys=False,
        )
    )
    before = manifest_path.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("ambiguous year keys must be refused before I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(MalformedManifestError, match="both keys"):
        fetch_source_artifact(
            "https://example.test/table.xlsx",
            source_id="irs_soi",
            package_id="soi-table",
            year=2024,
            output_dir=package,
        )

    assert manifest_path.read_bytes() == before
    assert list(package.iterdir()) == [manifest_path]


@pytest.mark.parametrize(
    "file_spec",
    [
        pytest.param([], id="list"),
        pytest.param("not a mapping", id="string"),
        pytest.param(0, id="zero"),
        pytest.param(False, id="false"),
        pytest.param(None, id="null"),
    ],
)
def test_fetch_refuses_a_non_mapping_year_entry_before_io(
    tmp_path, monkeypatch, file_spec
):
    package = tmp_path / "db" / "data" / "irs_soi" / "table"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "irs_soi",
                "package_id": "soi-table",
                "files": {2024: file_spec},
            },
            sort_keys=False,
        )
    )
    before = manifest_path.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("a malformed year entry must be refused before I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(MalformedManifestError, match="entry 2024.*mapping"):
        fetch_source_artifact(
            "https://example.test/table.xlsx",
            source_id="irs_soi",
            package_id="soi-table",
            year=2024,
            output_dir=package,
        )

    assert manifest_path.read_bytes() == before
    assert list(package.iterdir()) == [manifest_path]


@pytest.mark.parametrize("revision", [False, True], ids=["refetch", "revision"])
def test_fetch_carries_forward_fields_it_does_not_own(tmp_path, revision):
    package = tmp_path / "db" / "data" / "irs_soi" / "table"
    source = _publish(tmp_path, "table.xlsx", b"original publisher bytes")
    fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table",
        year=2024,
        output_dir=package,
    )
    manifest_path = package / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    metadata = {
        "source_table": "Publisher table 7",
        "notes": "Keep this review note.",
        "source_urls": ["https://example.test/landing-page"],
        "archive_member": "table.csv",
        "year": 2024,
    }
    manifest["files"][2024].update(metadata)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    if revision:
        source.write_bytes(b"publisher revision")

    fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table",
        year=2024,
        output_dir=package,
        record_revision=revision,
    )

    updated = yaml.safe_load(manifest_path.read_text())["files"][2024]
    for field, value in metadata.items():
        assert updated.get(field) == value


# ---------------------------------------------------------------------------
# Sol gate round 3: whole-tree manifest discovery and files-block shape
# ---------------------------------------------------------------------------


def _write_sweep_manifests(root):
    manifest_names = (
        "manifest.yaml",
        "manifest.yml",
        "manifest_named.yaml",
        "manifest_named.yml",
        "Manifest_Mixed.YAML",
    )
    for index, manifest_name in enumerate(manifest_names):
        package = root / f"package-{index}"
        package.mkdir(parents=True)
        content = f"publisher artifact {index}".encode()
        filename = f"artifact-{index}.csv"
        (package / filename).write_bytes(content)
        (package / manifest_name).write_text(
            yaml.safe_dump(
                {
                    "source_id": "publisher",
                    "package_id": f"package-{index}",
                    "files": {
                        2024: {
                            "filename": filename,
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "size_bytes": len(content),
                        }
                    },
                },
                sort_keys=False,
            )
        )
    decoy = root / "decoy" / "manifest-not-a-package.yaml"
    decoy.parent.mkdir()
    decoy.write_text("this: is not a package manifest\n")


def test_inventory_default_sweep_discovers_every_package_manifest(tmp_path):
    root = tmp_path / "data"
    _write_sweep_manifests(root)

    report = inventory_source_artifacts(root)

    assert report.valid
    assert report.counts["manifest_count"] == 5
    assert report.counts["artifact_count"] == 5
    assert {entry.manifest_path.rsplit("/", 1)[-1] for entry in report.entries} == {
        "manifest.yaml",
        "manifest.yml",
        "manifest_named.yaml",
        "manifest_named.yml",
        "Manifest_Mixed.YAML",
    }


def test_publish_default_sweep_discovers_every_package_manifest(tmp_path):
    root = tmp_path / "data"
    _write_sweep_manifests(root)
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)

    report = publish_source_artifacts(root, wrangler_command=str(wrangler))

    assert report.valid
    assert report.counts["manifest_count"] == 5
    assert report.counts["artifact_count"] == 5
    assert report.counts["uploaded_count"] == 5
    assert len(log.read_text().splitlines()) == 5


@pytest.mark.parametrize(
    "files",
    [
        pytest.param([], id="empty-list"),
        pytest.param("", id="empty-string"),
        pytest.param(0, id="zero"),
        pytest.param(False, id="false"),
    ],
)
def test_sweeps_reject_falsy_non_mapping_files_blocks(tmp_path, files):
    package = tmp_path / "data" / "package"
    package.mkdir(parents=True)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "publisher",
                "package_id": "package",
                "files": files,
            },
            sort_keys=False,
        )
    )
    before = manifest_path.read_bytes()
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)

    inventory = inventory_source_artifacts(package)
    published = publish_source_artifacts(package, wrangler_command=str(wrangler))

    assert (inventory.valid, published.valid) == (False, False)
    assert "files must be a mapping" in inventory.errors[0]
    assert "files must be a mapping" in published.errors[0]
    assert inventory.entries == ()
    assert published.entries == ()
    assert not log.exists()
    assert manifest_path.read_bytes() == before


def test_sweeps_treat_a_null_files_block_as_absent(tmp_path):
    package = tmp_path / "data" / "package"
    package.mkdir(parents=True)
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "publisher",
                "package_id": "package",
                "files": None,
            },
            sort_keys=False,
        )
    )

    inventory = inventory_source_artifacts(package)
    published = publish_source_artifacts(package)

    assert inventory.valid
    assert published.valid


# ---------------------------------------------------------------------------
# Manifest-declared artifact paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path_kind", ["absolute", "parent"])
def test_sweeps_refuse_non_bare_artifact_filenames_without_reading_them(
    tmp_path, path_kind
):
    package = tmp_path / "data" / "package"
    package.mkdir(parents=True)
    outside = tmp_path / "data" / "outside.csv"
    outside.write_bytes(b"outside publisher bytes")
    filename = str(outside) if path_kind == "absolute" else "../outside.csv"
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "publisher",
                "package_id": "package",
                "files": {
                    2024: {
                        "filename": filename,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                },
            },
            sort_keys=False,
        )
    )
    before = manifest_path.read_bytes()
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)

    inventory = inventory_source_artifacts(package)
    published = publish_source_artifacts(package, wrangler_command=str(wrangler))
    expected = f"non_canonical_filename:{filename}"

    assert not inventory.valid
    assert inventory.entries[0].errors == (expected,)
    assert inventory.entries[0].local_path == str(package)
    assert not published.valid
    assert published.entries[0].errors == (expected,)
    assert published.entries[0].upload is None
    assert published.entries[0].local_path == str(package)
    assert not log.exists()
    assert manifest_path.read_bytes() == before


def test_sweeps_refuse_a_symlinked_artifact_without_reading_it(tmp_path):
    package = tmp_path / "data" / "package"
    package.mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"outside publisher bytes")
    artifact_path = package / "table.csv"
    artifact_path.symlink_to(outside)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "publisher",
                "package_id": "package",
                "files": {
                    2024: {
                        "filename": artifact_path.name,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                },
            },
            sort_keys=False,
        )
    )
    before = manifest_path.read_bytes()
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)

    inventory = inventory_source_artifacts(package)
    published = publish_source_artifacts(package, wrangler_command=str(wrangler))
    expected = "artifact_path_is_symlink:table.csv"

    assert not inventory.valid
    assert inventory.entries[0].errors == (expected,)
    assert not inventory.entries[0].exists
    assert not published.valid
    assert published.entries[0].errors == (expected,)
    assert published.entries[0].upload is None
    assert not log.exists()
    assert manifest_path.read_bytes() == before
    assert artifact_path.is_symlink()


@pytest.mark.parametrize(
    "bad_kind",
    [
        pytest.param("parent", id="parent-path"),
        pytest.param("symlink", id="symlink"),
        pytest.param("manifest-name", id="manifest-name"),
        pytest.param("previous-r2", id="malformed-history"),
    ],
)
def test_publish_preflights_every_entry_before_any_upload(
    tmp_path, monkeypatch, bad_kind
):
    package = tmp_path / "data" / "package"
    package.mkdir(parents=True)
    first = b"first publisher table"
    second = b"second publisher table"
    (package / "one.csv").write_bytes(first)
    outside = tmp_path / "data" / "outside.csv"
    outside.write_bytes(second)
    second_path = package / "two.csv"
    bad_filename = "two.csv"
    bad_storage = None
    if bad_kind == "parent":
        bad_filename = "../outside.csv"
    elif bad_kind == "symlink":
        second_path.symlink_to(outside)
    elif bad_kind == "manifest-name":
        bad_filename = "manifest.yaml"
    else:
        second_path.write_bytes(second)
        bad_storage = {"previous_r2": {"not": "a list"}}
    bad_entry = {
        "filename": bad_filename,
        "source_url": "https://example.test/two.csv",
        "sha256": hashlib.sha256(second).hexdigest(),
        "size_bytes": len(second),
    }
    if bad_storage is not None:
        bad_entry["storage"] = bad_storage
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "publisher",
                "package_id": "package",
                "files": {
                    2023: {
                        "filename": "one.csv",
                        "source_url": "https://example.test/one.csv",
                        "sha256": hashlib.sha256(first).hexdigest(),
                        "size_bytes": len(first),
                    },
                    2024: bad_entry,
                },
            },
            sort_keys=False,
        )
    )
    before = manifest_path.read_bytes()
    uploads = []

    def non_writing_uploader(location, local_path, *, wrangler_command):
        uploads.append((location, local_path, wrangler_command))
        return ArtifactCommandResult(
            command=("non-writing-uploader",),
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", non_writing_uploader)

    report = publish_source_artifacts(package)

    assert not report.valid
    assert uploads == []
    assert manifest_path.read_bytes() == before


def test_publish_preflights_every_sibling_manifest_before_any_upload(
    tmp_path, monkeypatch
):
    package = tmp_path / "data" / "package"
    package.mkdir(parents=True)
    content = b"publisher table"
    (package / "table.csv").write_bytes(content)
    manifests = {
        package / "manifest_a.yaml": {
            "source_id": "publisher",
            "package_id": "package-a",
            "files": {
                2024: {
                    "filename": "table.csv",
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            },
        },
        package / "manifest_b.yaml": {
            "source_id": "publisher",
            "package_id": "package-b",
            "files": {
                2024: {
                    "filename": "manifest.yaml",
                    "sha256": hashlib.sha256(b"not a manifest").hexdigest(),
                }
            },
        },
    }
    for path, payload in manifests.items():
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = {path: path.read_bytes() for path in manifests}
    uploads = []

    def non_writing_uploader(location, local_path, *, wrangler_command):
        uploads.append((location, local_path, wrangler_command))
        return ArtifactCommandResult(
            command=("non-writing-uploader",),
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", non_writing_uploader)

    report = publish_source_artifacts(package)

    assert not report.valid
    assert uploads == []
    assert any(
        "manifest_named_filename:manifest.yaml" in entry.errors
        for entry in report.entries
    )
    assert {path: path.read_bytes() for path in manifests} == before


def test_publish_preflights_entire_root_before_any_upload(tmp_path, monkeypatch):
    root = tmp_path / "data"
    good_package = root / "a_good"
    bad_package = root / "z_bad"
    good_package.mkdir(parents=True)
    bad_package.mkdir(parents=True)
    good_content = b"good publisher table"
    bad_content = b"bad publisher table"
    (good_package / "good.csv").write_bytes(good_content)
    (bad_package / "bad.csv").write_bytes(bad_content)
    manifests = {
        good_package / "manifest.yaml": {
            "source_id": "publisher",
            "package_id": "good-package",
            "files": {
                2024: {
                    "filename": "good.csv",
                    "sha256": hashlib.sha256(good_content).hexdigest(),
                }
            },
        },
        bad_package / "manifest.yaml": {
            "source_id": "publisher",
            "package_id": "bad-package",
            "files": {
                2024: {
                    "filename": "../bad.csv",
                    "sha256": hashlib.sha256(bad_content).hexdigest(),
                }
            },
        },
    }
    for path, payload in manifests.items():
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = {path: path.read_bytes() for path in manifests}
    uploads = []

    def non_writing_uploader(location, local_path, *, wrangler_command):
        uploads.append((location, local_path, wrangler_command))
        return ArtifactCommandResult(
            command=("non-writing-uploader",),
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", non_writing_uploader)

    report = publish_source_artifacts(root)

    assert not report.valid
    assert uploads == []
    assert any(
        "non_canonical_filename:../bad.csv" in entry.errors for entry in report.entries
    )
    assert {path: path.read_bytes() for path in manifests} == before


def test_sweeps_refuse_conflicting_owners_across_package_manifests(
    tmp_path, monkeypatch
):
    package = tmp_path / "data" / "package"
    package.mkdir(parents=True)
    content = b"publisher table"
    filename = "table.csv"
    (package / filename).write_bytes(content)
    manifest_paths = (
        package / "manifest_a.yaml",
        package / "manifest_b.yaml",
    )
    for path, sha256 in zip(
        manifest_paths,
        (hashlib.sha256(content).hexdigest(), hashlib.sha256(b"other").hexdigest()),
    ):
        path.write_text(
            yaml.safe_dump(
                {
                    "source_id": "publisher",
                    "package_id": path.stem,
                    "files": {
                        2024: {
                            "filename": filename,
                            "sha256": sha256,
                            "size_bytes": len(content),
                        }
                    },
                },
                sort_keys=False,
            )
        )
    before = {path: path.read_bytes() for path in manifest_paths}
    uploads = []

    def non_writing_uploader(location, local_path, *, wrangler_command):
        uploads.append((location, local_path, wrangler_command))
        return ArtifactCommandResult(
            command=("non-writing-uploader",),
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", non_writing_uploader)

    inventory = inventory_source_artifacts(package)
    published = publish_source_artifacts(package)

    assert not inventory.valid
    assert not published.valid
    assert any("identify different bytes" in error for error in inventory.errors)
    assert any("identify different bytes" in error for error in published.errors)
    assert uploads == []
    assert {path: path.read_bytes() for path in manifest_paths} == before


# ---------------------------------------------------------------------------
# Sol gate round 3: canonical R2 locators before bucket-cutover skips
# ---------------------------------------------------------------------------


def test_publish_preserves_an_explicit_historical_route_during_bucket_cutover(
    tmp_path, monkeypatch
):
    package = tmp_path / "db" / "data" / "irs_soi" / "table"
    source = _publish(tmp_path, "table.xlsx", b"publisher table")
    fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table",
        year=2024,
        output_dir=package,
    )
    manifest_path = package / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    spec = manifest["files"][2024]
    wrong_key = f"raw/irs_soi/other-package/2023/{spec['sha256']}/{spec['filename']}"
    spec["storage"] = {
        "r2": {
            "provider": "r2",
            "bucket": "ledger-raw",
            "key": wrong_key,
            "uri": f"r2://ledger-raw/{wrong_key}",
        }
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = manifest_path.read_bytes()
    monkeypatch.setenv("CHRONICLE_R2_RAW_BUCKET", "chronicle-raw")
    log = tmp_path / "wrangler.log"
    wrangler = _wrangler_stub(tmp_path, log)

    report = publish_source_artifacts(package, wrangler_command=str(wrangler))

    assert report.valid
    assert report.entries[0].upload is None
    assert report.entries[0].errors == ()
    assert report.entries[0].skipped == (
        "recorded_r2_bucket_is_preserved_history:"
        "recorded=ledger-raw:requested=chronicle-raw"
    )
    assert report.entries[0].r2_location.key == wrong_key
    assert not log.exists()
    assert manifest_path.read_bytes() == before


def _make_recorded_locator_use_s3(package):
    manifest_path = package / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    r2 = manifest["files"][2022]["storage"]["r2"]
    r2["provider"] = "s3"
    r2["uri"] = f"s3://{r2['bucket']}/{r2['key']}"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path


def test_fetch_refuses_a_self_consistent_non_r2_locator_before_io(
    tmp_path, monkeypatch
):
    package, source, _report = _recorded_package(tmp_path)
    manifest_path = _make_recorded_locator_use_s3(package)
    before = manifest_path.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("a non-R2 storage.r2 locator must be refused before I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(RecordedR2LocatorError, match="provider.*r2"):
        _fetch_local(package, source, upload_r2=False)

    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("missing_field", ["provider", "uri"])
def test_fetch_refuses_an_incomplete_r2_locator_before_io(
    tmp_path, monkeypatch, missing_field
):
    package, source, _report = _recorded_package(tmp_path)
    manifest_path = package / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["files"][2022]["storage"]["r2"].pop(missing_field)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = manifest_path.read_bytes()

    def unexpected_read(_source_url):
        raise AssertionError("an incomplete storage.r2 locator reached I/O")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)

    with pytest.raises(RecordedR2LocatorError, match=missing_field):
        _fetch_local(package, source, upload_r2=False)

    assert manifest_path.read_bytes() == before


def test_publish_refuses_a_self_consistent_non_r2_locator(tmp_path, monkeypatch):
    package, _source, _report = _recorded_package(tmp_path)
    manifest_path = _make_recorded_locator_use_s3(package)
    before = manifest_path.read_bytes()
    monkeypatch.setenv("CHRONICLE_R2_RAW_BUCKET", "chronicle-raw")
    log = tmp_path / "publish.log"
    wrangler = _wrangler_stub(tmp_path, log)

    report = publish_source_artifacts(package, wrangler_command=str(wrangler))

    assert not report.valid
    assert report.entries[0].upload is None
    assert report.entries[0].skipped is None
    assert report.entries[0].errors[0].startswith("recorded_r2_locator_invalid:")
    assert "provider" in report.entries[0].errors[0]
    assert not log.exists()
    assert manifest_path.read_bytes() == before


# ---------------------------------------------------------------------------
# Sol gate round 3: identity segments, alias enumeration, non-regular manifests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["irs soi", "a/b", "..", " irs_soi", "irs_soi ", "a\\b", "a\tb"],
)
@pytest.mark.parametrize("field", ["source_id", "package_id"])
def test_fetch_refuses_noncanonical_identity_segments_before_io(
    tmp_path, monkeypatch, bad_id, field
):
    """A registration identity that _clean_key_part would rewrite (or that
    embeds separators) must be refused, never normalized into a different
    R2 namespace."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    source = _publish(tmp_path, "table.xlsx", b"table")

    def unexpected_read(_url):
        raise AssertionError("publisher read reached with a bad identity")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)
    kwargs = {"source_id": "irs_soi", "package_id": "soi-table-5"}
    kwargs[field] = bad_id

    with pytest.raises(SourceArtifactManifestError, match="segment"):
        fetch_source_artifact(
            str(source),
            year=2022,
            output_dir=package,
            **kwargs,
        )

    assert not package.exists()


def test_matching_directory_entry_refuses_multiple_normalized_aliases():
    """Two physical entries sharing one normalized key are a package defect;
    returning the first spelling would silently ignore the other bytes."""
    from types import SimpleNamespace

    from chronicle.registration import matching_directory_entry

    entries = [
        SimpleNamespace(name="TABLE.CSV"),
        SimpleNamespace(name="other.csv"),
        SimpleNamespace(name="table.csv"),
    ]
    directory = SimpleNamespace(is_dir=lambda: True, iterdir=lambda: iter(entries))

    with pytest.raises(ValueError, match="TABLE.CSV.*table.csv|table.csv.*TABLE.CSV"):
        matching_directory_entry(directory, "table.csv")

    assert matching_directory_entry(directory, "other.csv").name == "other.csv"


def test_publish_and_inventory_report_duplicate_artifact_aliases(tmp_path, monkeypatch):
    """A duplicate-alias defect surfaces as an entry error, not a crash and
    not a silent first-match read."""
    output_dir = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    source = _publish(tmp_path, "22in05ira.xlsx", b"IRA table 5")
    _fetch_local(output_dir, source, upload_r2=False)

    def duplicate_alias(_directory, filename):
        raise ValueError(f"{filename!r} matches two physical spellings in the package.")

    monkeypatch.setattr("chronicle.artifacts.matching_directory_entry", duplicate_alias)

    inventory = inventory_source_artifacts(output_dir)
    published = publish_source_artifacts(output_dir)

    assert not inventory.valid
    assert any(
        "duplicate_artifact_spellings" in error
        for entry in inventory.entries
        for error in entry.errors
    )
    assert not published.valid
    assert any(
        "duplicate_artifact_spellings" in error
        for entry in published.entries
        for error in entry.errors
    )


@pytest.mark.parametrize("shape", ["dangling", "directory"])
def test_sweeps_refuse_non_regular_manifest_entries(tmp_path, shape):
    """A manifest-named entry that is not a regular file must fail the sweep
    loudly instead of vanishing from it."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    package.mkdir(parents=True)
    target = package / "manifest.yaml"
    if shape == "dangling":
        target.symlink_to(package / "nowhere.yaml")
    else:
        target.mkdir()

    with pytest.raises(SourceArtifactManifestError, match="regular file"):
        inventory_source_artifacts(tmp_path / "db" / "data")
    with pytest.raises(SourceArtifactManifestError, match="regular file"):
        publish_source_artifacts(tmp_path / "db" / "data")


def test_fetch_refuses_a_dangling_manifest_symlink_instead_of_creating_one(
    tmp_path,
):
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    package.mkdir(parents=True)
    (package / "manifest.yaml").symlink_to(package / "nowhere.yaml")
    source = _publish(tmp_path, "table.xlsx", b"table")

    with pytest.raises(SourceArtifactManifestError, match="regular file"):
        _fetch_local(package, source, upload_r2=False)

    assert (package / "manifest.yaml").is_symlink()
    assert not (package / "table.xlsx").exists()


def test_record_revision_updates_every_owner_even_when_yaml_aliases_share_one_entry(
    tmp_path,
):
    """Two vintages sharing one physical file are two owners even when the
    manifest spelled them with a YAML anchor and alias (one dict object);
    identifying the selected owner by object identity skipped both."""
    package = tmp_path / "db" / "data" / "irs_soi" / "soi-table-5"
    source = _publish(tmp_path, "archive.zip", b"first publication")
    _fetch_local(package, source, upload_r2=False)
    manifest_path = package / "manifest.yaml"
    entry = _entry(manifest_path)
    lines = [
        "source_id: irs_soi",
        "package_id: soi-table-5",
        "files:",
        "  2022: &shared",
    ]
    for field, value in entry.items():
        lines.append(f"    {field}: {json.dumps(value)}")
    lines.append("  2023: *shared")
    manifest_path.write_text("\n".join(lines) + "\n")
    old_sha = entry["sha256"]

    source.write_bytes(b"second publication, revised rows")
    report = fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table-5",
        year=2023,
        output_dir=package,
        upload_r2=False,
        record_revision=True,
    )

    assert report.valid
    manifest = yaml.safe_load(manifest_path.read_text())
    new_sha = hashlib.sha256(b"second publication, revised rows").hexdigest()
    assert manifest["files"][2023]["sha256"] == new_sha
    assert manifest["files"][2022]["sha256"] == new_sha, (
        "the aliased owner kept the superseded checksum"
    )
    assert old_sha != new_sha


def test_identical_byte_refetch_keeps_r2_identified_shared_file_valid(tmp_path):
    """Two manifests may identify one package-local file through identical
    content-addressed R2 locators without declaring ``sha256``. Refetching
    the same bytes records ``sha256`` on the selected manifest only; the
    sibling's effective identity (its recorded R2 key) still agrees, so the
    package directory must stay valid for every sweep."""
    from chronicle.artifacts import (
        _effective_recorded_digest,
        default_r2_raw_bucket,
    )
    from chronicle.registration import validate_package_directory

    def collisions(manifests):
        # The sweeps resolve each entry's effective identity (its recorded
        # content-addressed R2 key when no ``sha256`` is declared) exactly
        # like this before comparing owners.
        return validate_package_directory(
            manifests, entry_digest=_effective_recorded_digest
        )

    package = tmp_path / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    content = b"shared publisher table"
    filename = "shared.csv"
    sha256 = hashlib.sha256(content).hexdigest()
    (package / filename).write_bytes(content)
    source = tmp_path / "publisher-download.csv"
    source.write_bytes(content)
    bucket = default_r2_raw_bucket()
    manifest_paths = (package / "manifest_a.yaml", package / "manifest_b.yaml")
    for path in manifest_paths:
        key = build_r2_key(
            source_id="publisher",
            package_id=path.stem,
            year=2024,
            sha256=sha256,
            filename=filename,
        )
        path.write_text(
            yaml.safe_dump(
                {
                    "source_id": "publisher",
                    "package_id": path.stem,
                    "files": {
                        2024: {
                            "filename": filename,
                            "source_url": str(source),
                            "storage": {
                                "r2": {
                                    "provider": "r2",
                                    "bucket": bucket,
                                    "key": key,
                                    "uri": f"r2://{bucket}/{key}",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            )
        )
    manifests = {str(path): yaml.safe_load(path.read_text()) for path in manifest_paths}
    assert collisions(manifests) == ()
    assert inventory_source_artifacts(package).valid

    report = fetch_source_artifact(
        str(source),
        source_id="publisher",
        package_id="manifest_a",
        year=2024,
        output_dir=package,
        filename=filename,
        manifest_filename="manifest_a.yaml",
    )
    assert report.valid
    assert report.sha256 == sha256

    manifests = {str(path): yaml.safe_load(path.read_text()) for path in manifest_paths}
    assert manifests[str(manifest_paths[0])]["files"][2024]["sha256"] == sha256
    assert "sha256" not in manifests[str(manifest_paths[1])]["files"][2024]
    assert collisions(manifests) == ()

    inventory = inventory_source_artifacts(package)
    published = publish_source_artifacts(package)
    for sweep in (inventory, published):
        assert not any("filename_collision" in error for error in sweep.errors)
        assert not any("identify different bytes" in error for error in sweep.errors)
        assert sweep.valid


def _write_unidentified_shared_manifests(package, source, *, filename="shared.csv"):
    """Two manifests naming one package-local file with no identity yet."""
    manifest_paths = (package / "manifest_a.yaml", package / "manifest_b.yaml")
    for path in manifest_paths:
        path.write_text(
            yaml.safe_dump(
                {
                    "source_id": "publisher",
                    "package_id": path.stem,
                    "files": {
                        2024: {
                            "filename": filename,
                            "source_url": str(source),
                        }
                    },
                },
                sort_keys=False,
            )
        )
    return manifest_paths


def _fake_wrangler(tmp_path):
    log = tmp_path / "wrangler.log"
    wrangler = tmp_path / "wrangler"
    wrangler.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\necho ok\n")
    wrangler.chmod(0o755)
    return wrangler, log


def test_selected_publication_of_a_shared_unidentified_file_keeps_siblings_valid(
    tmp_path,
):
    """Two manifests may name one package-local file before either records an
    identity. Publishing only one of them records its checksum and locator;
    the sibling, which still records nothing, must not become a false
    collision, and its own later publication must record the same identity."""
    package = tmp_path / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    content = b"shared publisher table"
    (package / "shared.csv").write_bytes(content)
    source = tmp_path / "publisher-download.csv"
    source.write_bytes(content)
    manifest_a, manifest_b = _write_unidentified_shared_manifests(package, source)
    wrangler, log = _fake_wrangler(tmp_path)
    expected_sha256 = hashlib.sha256(content).hexdigest()

    assert inventory_source_artifacts(package).valid

    selected = publish_source_artifacts(
        package, manifest_filename="manifest_a.yaml", wrangler_command=str(wrangler)
    )
    assert selected.valid
    assert selected.counts["uploaded_count"] == 1
    assert yaml.safe_load(manifest_a.read_text())["files"][2024]["sha256"] == (
        expected_sha256
    )
    assert "sha256" not in yaml.safe_load(manifest_b.read_text())["files"][2024]

    inventory = inventory_source_artifacts(package)
    assert inventory.valid, inventory.errors
    assert not any("filename_collision" in error for error in inventory.errors)

    sibling = publish_source_artifacts(
        package, manifest_filename="manifest_b.yaml", wrangler_command=str(wrangler)
    )
    assert sibling.valid, sibling.errors
    assert sibling.counts["uploaded_count"] == 1
    recorded_b = yaml.safe_load(manifest_b.read_text())["files"][2024]
    recorded_a = yaml.safe_load(manifest_a.read_text())["files"][2024]
    assert recorded_b["sha256"] == expected_sha256
    assert recorded_b["storage"]["r2"]["key"].endswith(f"/{expected_sha256}/shared.csv")
    assert recorded_a["storage"]["r2"]["key"].endswith(f"/{expected_sha256}/shared.csv")
    assert len(log.read_text().splitlines()) == 2

    everything = publish_source_artifacts(package, wrangler_command=str(wrangler))
    assert everything.valid, everything.errors
    assert everything.counts["uploaded_count"] == 0
    assert everything.counts["skipped_count"] == 2


def test_partial_upload_failure_of_a_shared_file_stays_retryable(tmp_path, monkeypatch):
    """A full sweep whose second upload fails records the first manifest's
    identity only. The retry must not report a collision for the sibling that
    still records nothing, and must finish recording the same identity."""
    package = tmp_path / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    content = b"shared publisher table"
    (package / "shared.csv").write_bytes(content)
    source = tmp_path / "publisher-download.csv"
    source.write_bytes(content)
    manifest_a, manifest_b = _write_unidentified_shared_manifests(package, source)
    expected_sha256 = hashlib.sha256(content).hexdigest()
    uploads = []
    failures = {"remaining": 1}

    def uploader(location, local_path, *, wrangler_command):
        uploads.append(location)
        if len(uploads) == 2 and failures["remaining"]:
            failures["remaining"] -= 1
            return ArtifactCommandResult(
                command=("failing-uploader",), returncode=1, stdout="", stderr="boom"
            )
        return ArtifactCommandResult(
            command=("uploader",), returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", uploader)

    first = publish_source_artifacts(package)
    assert not first.valid
    assert first.counts["uploaded_count"] == 1
    assert first.counts["failed_count"] == 1
    recorded = {
        path.name: yaml.safe_load(path.read_text())["files"][2024]
        for path in (manifest_a, manifest_b)
    }
    identified = [name for name, entry in recorded.items() if "sha256" in entry]
    assert len(identified) == 1

    inventory = inventory_source_artifacts(package)
    assert inventory.valid, inventory.errors
    assert not any("filename_collision" in error for error in inventory.errors)

    retry = publish_source_artifacts(package)
    assert retry.valid, retry.errors
    assert retry.counts["uploaded_count"] == 1
    assert retry.counts["skipped_count"] == 1
    assert retry.counts["failed_count"] == 0
    for path in (manifest_a, manifest_b):
        entry = yaml.safe_load(path.read_text())["files"][2024]
        assert entry["sha256"] == expected_sha256
        assert entry["storage"]["r2"]["key"].endswith(f"/{expected_sha256}/shared.csv")
    assert len(uploads) == 3


def test_sweeps_refuse_identifying_a_shared_file_whose_bytes_changed(
    tmp_path, monkeypatch
):
    """An unidentified sibling is not a collision, but identifying it would
    hash the package-local bytes. When those bytes no longer match what the
    identified owner records, every sweep refuses before any upload."""
    package = tmp_path / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    old_content = b"shared publisher table"
    new_content = b"shared publisher table, revised"
    (package / "shared.csv").write_bytes(new_content)
    source = tmp_path / "publisher-download.csv"
    source.write_bytes(new_content)
    manifest_a, manifest_b = _write_unidentified_shared_manifests(package, source)
    identified = yaml.safe_load(manifest_a.read_text())
    identified["files"][2024]["sha256"] = hashlib.sha256(old_content).hexdigest()
    identified["files"][2024]["size_bytes"] = len(old_content)
    manifest_a.write_text(yaml.safe_dump(identified, sort_keys=False))
    uploads = []

    def unexpected_uploader(location, local_path, *, wrangler_command):
        uploads.append(location)
        return ArtifactCommandResult(
            command=("uploader",), returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", unexpected_uploader)

    inventory = inventory_source_artifacts(package)
    published = publish_source_artifacts(package)
    sibling_only = publish_source_artifacts(
        package, manifest_filename="manifest_b.yaml"
    )

    for sweep in (inventory, published, sibling_only):
        assert not sweep.valid
        assert any("two identities" in error for error in sweep.errors), sweep.errors
    assert uploads == []
    assert "sha256" not in yaml.safe_load(manifest_b.read_text())["files"][2024]


@pytest.mark.parametrize("record_revision", [False, True])
def test_first_fetch_initializes_a_predeclared_entry_without_identity(
    tmp_path, record_revision
):
    """A manifest may predeclare an entry with only ``filename`` and
    ``source_url``. Its first fetch identifies it: the entry is the one being
    initialized, not an unidentifiable owner, so both the fetch preflight
    and the manifest writer must accept it (default and --record-revision)."""
    package = tmp_path / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    source = tmp_path / "publisher-download.csv"
    content = b"first publisher bytes"
    source.write_bytes(content)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "publisher",
                "package_id": "package",
                "files": {2024: {"filename": "table.csv", "source_url": str(source)}},
            },
            sort_keys=False,
        )
    )
    assert not (package / "table.csv").exists()

    report = fetch_source_artifact(
        str(source),
        source_id="publisher",
        package_id="package",
        year=2024,
        output_dir=package,
        filename="table.csv",
        record_revision=record_revision,
    )

    assert report.valid, report.errors
    expected_sha256 = hashlib.sha256(content).hexdigest()
    assert report.sha256 == expected_sha256
    assert (package / "table.csv").read_bytes() == content
    entry = yaml.safe_load(manifest_path.read_text())["files"][2024]
    assert entry["filename"] == "table.csv"
    assert entry["sha256"] == expected_sha256
    assert entry["size_bytes"] == len(content)
    assert inventory_source_artifacts(package).valid

    again = fetch_source_artifact(
        str(source),
        source_id="publisher",
        package_id="package",
        year=2024,
        output_dir=package,
        filename="table.csv",
    )
    assert again.valid, again.errors
    assert again.sha256 == expected_sha256


def test_fetch_still_refuses_an_unidentified_owner_in_another_manifest(tmp_path):
    """Only the entry being initialized is exempt: another manifest naming the
    same package-local file without an identity keeps refusing the fetch
    before any byte is written, because the fetched bytes would silently
    define what that sibling means."""
    package = tmp_path / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    source = tmp_path / "publisher-download.csv"
    source.write_bytes(b"first publisher bytes")
    manifest_a, manifest_b = _write_unidentified_shared_manifests(
        package, source, filename="table.csv"
    )
    before = {path: path.read_text() for path in (manifest_a, manifest_b)}

    with pytest.raises(MalformedManifestError, match="records no sha256 identity"):
        fetch_source_artifact(
            str(source),
            source_id="publisher",
            package_id="manifest_a",
            year=2024,
            output_dir=package,
            filename="table.csv",
            manifest_filename="manifest_a.yaml",
        )

    assert not (package / "table.csv").exists()
    assert {path: path.read_text() for path in (manifest_a, manifest_b)} == before


@pytest.mark.parametrize("missing", ["source_id", "package_id"])
def test_selected_publication_overrides_apply_only_to_the_selected_manifest(
    tmp_path, missing
):
    """``--source-id`` / ``--package-id`` complete the selected manifest's
    identity. An unselected sibling that declares a different identifier is
    preflighted with its own identifiers, not the override, so the selected
    publication succeeds and the sibling is left untouched."""
    package = tmp_path / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    content_a = b"table a"
    content_b = b"table b"
    (package / "table_a.csv").write_bytes(content_a)
    (package / "table_b.csv").write_bytes(content_b)
    identity_a = {"source_id": "publisher_a", "package_id": "package_a"}
    identity_b = {"source_id": "publisher_b", "package_id": "package_b"}
    override = {missing: identity_a[missing]}
    declared_a = {key: value for key, value in identity_a.items() if key != missing}
    manifest_a = package / "manifest_a.yaml"
    manifest_b = package / "manifest_b.yaml"
    manifest_a.write_text(
        yaml.safe_dump(
            {
                **declared_a,
                "files": {
                    2024: {
                        "filename": "table_a.csv",
                        "source_url": "https://publisher.test/a",
                        "sha256": hashlib.sha256(content_a).hexdigest(),
                        "size_bytes": len(content_a),
                    }
                },
            },
            sort_keys=False,
        )
    )
    manifest_b.write_text(
        yaml.safe_dump(
            {
                **identity_b,
                "files": {
                    2024: {
                        "filename": "table_b.csv",
                        "source_url": "https://publisher.test/b",
                        "sha256": hashlib.sha256(content_b).hexdigest(),
                        "size_bytes": len(content_b),
                    }
                },
            },
            sort_keys=False,
        )
    )
    before_b = manifest_b.read_text()
    wrangler, log = _fake_wrangler(tmp_path)

    report = publish_source_artifacts(
        package,
        manifest_filename="manifest_a.yaml",
        wrangler_command=str(wrangler),
        **override,
    )

    assert report.valid, report.errors
    assert report.counts["uploaded_count"] == 1
    assert report.counts["failed_count"] == 0
    recorded = yaml.safe_load(manifest_a.read_text())
    assert recorded[missing] == identity_a[missing]
    assert recorded["files"][2024]["storage"]["r2"]["key"].startswith(
        "raw/publisher_a/package_a/2024/"
    )
    assert manifest_b.read_text() == before_b
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.parametrize("missing", ["source_id", "package_id"])
def test_default_sweep_overrides_apply_to_every_selected_sibling(tmp_path, missing):
    """A default sweep selects every manifest in the directory. The override
    must complete the manifest that lacks the identifier and confirm the
    sibling that declares the same value, whichever order they are processed
    in -- a selected sibling met through another selected manifest's package
    preflight is not an unselected one."""
    package = tmp_path / "data" / "publisher" / "package"
    package.mkdir(parents=True)
    content_a = b"table a"
    content_b = b"table b"
    (package / "table_a.csv").write_bytes(content_a)
    (package / "table_b.csv").write_bytes(content_b)
    identity = {"source_id": "publisher", "package_id": "package"}
    override = {missing: identity[missing]}
    declared_a = {key: value for key, value in identity.items() if key != missing}
    manifest_a = package / "manifest_a.yaml"
    manifest_b = package / "manifest_b.yaml"
    manifest_a.write_text(
        yaml.safe_dump(
            {
                **declared_a,
                "files": {
                    2024: {
                        "filename": "table_a.csv",
                        "source_url": "https://publisher.test/a",
                        "sha256": hashlib.sha256(content_a).hexdigest(),
                        "size_bytes": len(content_a),
                    }
                },
            },
            sort_keys=False,
        )
    )
    manifest_b.write_text(
        yaml.safe_dump(
            {
                **identity,
                "files": {
                    2024: {
                        "filename": "table_b.csv",
                        "source_url": "https://publisher.test/b",
                        "sha256": hashlib.sha256(content_b).hexdigest(),
                        "size_bytes": len(content_b),
                    }
                },
            },
            sort_keys=False,
        )
    )
    wrangler, log = _fake_wrangler(tmp_path)

    report = publish_source_artifacts(
        package, wrangler_command=str(wrangler), **override
    )

    assert report.valid, report.errors
    assert report.counts["uploaded_count"] == 2
    assert report.counts["failed_count"] == 0
    assert not any("r2_identity_invalid" in error for error in report.errors)
    for path in (manifest_a, manifest_b):
        recorded = yaml.safe_load(path.read_text())
        assert recorded[missing] == identity[missing]
        assert recorded["files"][2024]["storage"]["r2"]["key"].startswith(
            "raw/publisher/package/2024/"
        )
    assert len(log.read_text().splitlines()) == 2
