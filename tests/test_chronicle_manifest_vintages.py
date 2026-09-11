"""Every artifact boundary refuses duplicate logical manifest vintages."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from chronicle.artifacts import (
    MalformedManifestError,
    fetch_source_artifact,
    inventory_source_artifacts,
    publish_source_artifacts,
)
from chronicle.source_package import SourceArtifactSpec


@pytest.fixture
def duplicate_vintage_package(tmp_path):
    package = tmp_path / "data" / "irs_soi" / "table"
    package.mkdir(parents=True)
    files = {}
    for vintage, filename in (
        (2023, "selected.csv"),
        (2024, "numeric.csv"),
        ("2024", "quoted.csv"),
    ):
        content = f"publisher bytes for {filename}\n".encode()
        (package / filename).write_bytes(content)
        files[vintage] = {
            "filename": filename,
            "source_url": f"https://example.test/{filename}",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "irs_soi",
                "package_id": "soi-table",
                "files": files,
            },
            sort_keys=False,
        )
    )
    return package


def _unexpected_artifact_io(*_args, **_kwargs):
    raise AssertionError("duplicate logical vintages reached artifact I/O")


@pytest.mark.parametrize("year", [2023, 2024])
def test_fetch_refuses_logical_vintage_duplicates_anywhere_before_io(
    duplicate_vintage_package, monkeypatch, year
):
    package = duplicate_vintage_package
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    monkeypatch.setattr("chronicle.artifacts._read_artifact", _unexpected_artifact_io)
    monkeypatch.setattr(
        "chronicle.artifacts._upload_r2_object", _unexpected_artifact_io
    )

    with pytest.raises(MalformedManifestError, match="both keys"):
        fetch_source_artifact(
            "https://example.test/selected.csv",
            source_id="irs_soi",
            package_id="soi-table",
            year=year,
            output_dir=package,
            upload_r2=True,
        )

    assert {path.name: path.read_bytes() for path in package.iterdir()} == before


@pytest.mark.parametrize(
    "operation", [publish_source_artifacts, inventory_source_artifacts]
)
def test_sweeps_refuse_logical_vintage_duplicates_before_artifact_io(
    duplicate_vintage_package, monkeypatch, operation
):
    package = duplicate_vintage_package
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    read_bytes = Path.read_bytes

    def checked_read_bytes(path):
        if path.parent == package and path.suffix == ".csv":
            _unexpected_artifact_io()
        return read_bytes(path)

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "read_bytes", checked_read_bytes)
        guarded.setattr(
            "chronicle.artifacts._upload_r2_object", _unexpected_artifact_io
        )
        report = operation(package)

    assert not report.valid
    assert report.entries == ()
    assert any("both keys" in error for error in report.errors)
    assert {path.name: path.read_bytes() for path in package.iterdir()} == before


@pytest.mark.parametrize("year", [2023, 2024])
def test_source_loader_refuses_logical_vintage_duplicates_anywhere_before_io(
    duplicate_vintage_package, tmp_path, monkeypatch, year
):
    package = duplicate_vintage_package
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    monkeypatch.setattr("chronicle.source_package.files", lambda _package: tmp_path)
    monkeypatch.setattr(
        "chronicle.source_package._read_source_artifact_content",
        _unexpected_artifact_io,
    )
    artifact = SourceArtifactSpec(
        source_name="irs_soi",
        source_table="Table",
        resource_package="test_resources",
        resource_directory="data/irs_soi/table",
        manifest="manifest.yaml",
        vintage="2024",
        extracted_at="2026-09-04",
        extraction_method="test",
    )

    with pytest.raises(ValueError, match="both keys"):
        artifact._artifact_content(year)

    assert {path.name: path.read_bytes() for path in package.iterdir()} == before
