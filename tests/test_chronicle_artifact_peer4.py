"""Failing-first regressions for PR #226 peer round 4."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from chronicle.artifacts import (
    ArtifactCommandResult,
    SourceArtifactManifestError,
    fetch_source_artifact,
    inventory_source_artifacts,
    publish_derived_artifacts,
    publish_source_artifacts,
)
from chronicle.cli import main as cli_main
from chronicle.harness import main as harness_main


def _package(tmp_path, *, filename="table.csv", manifest_name="manifest.yaml"):
    package = tmp_path / "package"
    package.mkdir()
    content = b"publisher,value\nexample,123\n"
    (package / filename).write_bytes(content)
    manifest_path = package / manifest_name
    manifest = {
        "source_id": "publisher",
        "package_id": "package",
        "files": {
            2024: {
                "filename": filename,
                "source_url": "https://example.test/table.csv",
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return package, manifest_path, manifest


def _no_upload(monkeypatch):
    def unexpected_upload(*args, **kwargs):
        pytest.fail("uploader reached before refusal")

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", unexpected_upload)


@pytest.mark.parametrize("shape", ["file-link", "directory-link", "dangling", "fifo"])
def test_derived_preflights_complete_tree_before_reads_or_uploads(
    tmp_path, monkeypatch, shape
):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "a_good.jsonl").write_bytes(b"{}\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_bytes(b"outside bytes")
    unsafe = suite / "z_unsafe"
    if shape == "file-link":
        unsafe.symlink_to(outside / "secret.csv")
    elif shape == "directory-link":
        unsafe.symlink_to(outside, target_is_directory=True)
    elif shape == "dangling":
        unsafe.symlink_to(outside / "missing")
    else:
        os.mkfifo(unsafe)
    output = tmp_path / "registry.jsonl"
    output.write_text("sentinel\n")
    _no_upload(monkeypatch)

    def unexpected_read(*args, **kwargs):
        pytest.fail("build read reached before tree refusal")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    monkeypatch.setattr("chronicle.artifacts.infer_build_id", unexpected_read)
    report = publish_derived_artifacts(
        suite,
        source_id="publisher",
        package_id="package",
        year=2024,
        build_artifacts_output=output,
    )
    assert not report.valid
    assert report.entries == ()
    assert any("regular" in error or "symlink" in error for error in report.errors)
    assert output.read_text() == "sentinel\n"


@pytest.mark.parametrize("field", ["source_id", "package_id"])
@pytest.mark.parametrize("bad_id", ["foo bar", "a/b", "..", 123, None, ""])
@pytest.mark.parametrize("operation", ["raw", "derived"])
def test_new_publication_refuses_noncanonical_identity_before_reads(
    tmp_path, monkeypatch, field, bad_id, operation
):
    package, manifest_path, manifest = _package(tmp_path)
    kwargs = {"source_id": "publisher", "package_id": "package"}
    kwargs[field] = bad_id
    if operation == "raw":
        manifest[field] = bad_id
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = manifest_path.read_text()
    _no_upload(monkeypatch)

    def unexpected_read(*args, **kwargs):
        pytest.fail("artifact read reached with noncanonical publication identity")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    monkeypatch.setattr("chronicle.artifacts.infer_build_id", unexpected_read)
    if operation == "raw":
        report = publish_source_artifacts(package)
    else:
        report = publish_derived_artifacts(package, year=2024, **kwargs)
    assert not report.valid
    assert any(
        "identity" in error or "source_id" in error or "package_id" in error
        for error in (
            *report.errors,
            *(e for entry in report.entries for e in entry.errors),
        )
    )
    assert manifest_path.read_text() == before


@pytest.mark.parametrize("field", ["source_id", "package_id"])
@pytest.mark.parametrize("declaration", [" padded ", 123, None, "   ", ""])
def test_fetch_refuses_noncanonical_present_manifest_identity_before_io(
    tmp_path, monkeypatch, field, declaration
):
    package, manifest_path, manifest = _package(tmp_path)
    args = {"source_id": "publisher", "package_id": "package"}
    args[field] = "padded" if declaration == " padded " else "123"
    manifest[field] = declaration
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = manifest_path.read_text()
    _no_upload(monkeypatch)

    def unexpected_read(*args, **kwargs):
        pytest.fail("publisher read reached with noncanonical manifest declaration")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)
    with pytest.raises(SourceArtifactManifestError, match=field):
        fetch_source_artifact(
            "https://example.test/table.csv",
            year=2024,
            output_dir=package,
            filename="table.csv",
            **args,
        )
    assert manifest_path.read_text() == before


@pytest.mark.parametrize("operation", ["publish", "inventory"])
@pytest.mark.parametrize(
    ("physical", "declared"),
    [("table.csv", "TABLE.csv"), ("café.csv", "cafe\u0301.csv")],
)
def test_sweeps_refuse_single_normalized_artifact_alias_before_read(
    tmp_path, monkeypatch, operation, physical, declared
):
    package, manifest_path, manifest = _package(tmp_path, filename=physical)
    # On filesystems that normalize Unicode on creation, choose the other
    # logical spelling from the actual directory entry returned by iterdir.
    actual = next(
        path.name for path in package.iterdir() if path.name != manifest_path.name
    )
    if actual == declared:
        declared = physical
    manifest["files"][2024]["filename"] = declared
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = manifest_path.read_text()
    _no_upload(monkeypatch)

    def unexpected_read(*args, **kwargs):
        pytest.fail("artifact read reached through a normalized alias")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    function = (
        publish_source_artifacts
        if operation == "publish"
        else inventory_source_artifacts
    )
    report = function(package)
    assert not report.valid
    assert any(
        "spelling" in error for entry in report.entries for error in entry.errors
    )
    assert manifest_path.read_text() == before


@pytest.mark.parametrize("operation", ["publish-raw", "inventory-artifacts"])
@pytest.mark.parametrize("shape", ["directory", "dangling", "fifo"])
@pytest.mark.parametrize("entrypoint", ["function", "harness", "cli"])
def test_explicit_sweep_reports_nonregular_manifest_sibling(
    tmp_path, monkeypatch, capsys, operation, shape, entrypoint
):
    package, manifest_path, _manifest = _package(
        tmp_path, manifest_name="manifest_selected.yaml"
    )
    sibling = package / "manifest_sibling.yaml"
    if shape == "directory":
        sibling.mkdir()
    elif shape == "dangling":
        sibling.symlink_to(package / "missing")
    else:
        os.mkfifo(sibling)
    before = manifest_path.read_text()
    _no_upload(monkeypatch)
    if entrypoint == "function":
        function = (
            publish_source_artifacts
            if operation == "publish-raw"
            else inventory_source_artifacts
        )
        report = function(package, manifest_filename=manifest_path.name)
        assert not report.valid
        assert "regular file" in " ".join(report.errors)
    else:
        args = [operation, "--root", str(package), "--manifest", manifest_path.name]
        if entrypoint == "harness":
            assert harness_main(args) == 1
        else:
            monkeypatch.setattr("sys.argv", ["chronicle", *args])
            with pytest.raises(SystemExit) as exit_info:
                cli_main()
            assert exit_info.value.code == 1
        assert "regular file" in json.dumps(json.loads(capsys.readouterr().out))
    assert manifest_path.read_text() == before


@pytest.mark.parametrize("location", ["root", "nested", "excluded-registry"])
def test_derived_preflight_rejects_symlinks_at_every_tree_boundary(
    tmp_path, monkeypatch, location
):
    suite = tmp_path / "suite"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_bytes(b"outside")
    if location == "root":
        suite.symlink_to(outside, target_is_directory=True)
    else:
        suite.mkdir()
        target = suite / "build_artifacts.jsonl"
        if location == "nested":
            (suite / "reports").mkdir()
            target = suite / "reports" / "database.json"
        target.symlink_to(outside / "secret.csv")
    _no_upload(monkeypatch)
    report = publish_derived_artifacts(
        suite,
        source_id="publisher",
        package_id="package",
        year=2024,
        build_id="ledger.build.v1:peer4",
    )
    assert not report.valid
    assert report.entries == ()


@pytest.mark.parametrize("field", ["source_id", "package_id"])
@pytest.mark.parametrize("bad_id", ["foo bar", "a/b", "..", 123, ""])
def test_raw_publication_refuses_noncanonical_explicit_identity_before_read(
    tmp_path, monkeypatch, field, bad_id
):
    package, manifest_path, _manifest = _package(tmp_path)
    before = manifest_path.read_text()
    _no_upload(monkeypatch)

    def unexpected_read(*args, **kwargs):
        pytest.fail("artifact read reached with invalid explicit identity")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    report = publish_source_artifacts(package, **{field: bad_id})
    assert not report.valid
    assert manifest_path.read_text() == before


@pytest.mark.parametrize(
    "defect",
    [
        "contradictory",
        "incomplete",
        "non-r2",
        "empty",
        "sha256",
        "filename",
        "local-digest",
    ],
)
def test_inventory_refuses_invalid_recorded_r2_and_does_not_count_link(
    tmp_path, monkeypatch, defect
):
    package, manifest_path, manifest = _package(tmp_path)
    spec = manifest["files"][2024]
    key = f"raw/publisher/package/2024/{spec['sha256']}/table.csv"
    block = {
        "provider": "r2",
        "bucket": "archive",
        "key": key,
        "uri": f"r2://archive/{key}",
    }
    if defect == "contradictory":
        block["bucket"] = "different"
    elif defect == "incomplete":
        del block["provider"]
    elif defect == "non-r2":
        block["provider"] = "s3"
        block["uri"] = f"s3://archive/{key}"
    elif defect == "empty":
        block = {}
    elif defect == "sha256":
        spec["sha256"] = "0" * 64
    elif defect == "filename":
        block["key"] = key.replace("table.csv", "other.csv")
        block["uri"] = f"r2://archive/{block['key']}"
    else:
        del spec["sha256"]
        (package / "table.csv").write_bytes(b"different local bytes")
    spec["storage"] = {"r2": block}
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = manifest_path.read_text()
    if defect != "local-digest":

        def unexpected_read(*args, **kwargs):
            pytest.fail("inventory read bytes before refusing invalid R2 identity")

        monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    report = inventory_source_artifacts(package)
    assert not report.valid
    assert report.counts["r2_link_count"] == 0
    assert report.entries[0].r2 is None
    assert any("recorded_r2" in error for error in report.entries[0].errors)
    assert manifest_path.read_text() == before


def test_inventory_accepts_consistent_uri_only_r2_locator(tmp_path):
    package, manifest_path, manifest = _package(tmp_path)
    spec = manifest["files"][2024]
    spec["storage"] = {
        "r2": {
            "provider": "r2",
            "uri": f"r2://archive/raw/publisher/package/2024/{spec['sha256']}/table.csv",
        }
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    report = inventory_source_artifacts(package)
    assert report.valid
    assert report.counts["r2_link_count"] == 1


def test_derived_publication_refuses_unrecognized_explicit_route_before_read(
    tmp_path, monkeypatch
):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "facts.jsonl").write_bytes(b"{}\n")
    _no_upload(monkeypatch)

    def unexpected_read(*args, **kwargs):
        pytest.fail("build read reached for an unrecognizable derived route")

    monkeypatch.setattr("chronicle.artifacts.infer_build_id", unexpected_read)
    report = publish_derived_artifacts(
        suite,
        source_id="publisher",
        package_id="package",
        year=2024,
        r2_bucket="chronicle-builds",
        r2_prefix="builds",
    )
    assert not report.valid
    assert "derived_route" in " ".join(report.errors)


@pytest.mark.parametrize(
    "env_prefix", ["CHRONICLE_", "POLICYENGINE_LEDGER_", "LEDGER_"]
)
def test_derived_publication_propagates_configured_prefix(
    tmp_path, monkeypatch, env_prefix
):
    from chronicle.consumer_contract import _points_at_derived

    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "facts.jsonl").write_bytes(b"{}\n")
    monkeypatch.setenv(f"{env_prefix}R2_DERIVED_BUCKET", "chronicle-builds")
    monkeypatch.setenv(f"{env_prefix}R2_DERIVED_PREFIX", "builds")
    uploaded = []

    def upload(location, *_args, **_kwargs):
        uploaded.append(location)
        return ArtifactCommandResult(
            command=("test",), returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", upload)
    report = publish_derived_artifacts(
        suite,
        source_id="publisher",
        package_id="package",
        year=2024,
        build_id="ledger.build.v1:peer4",
    )
    assert report.valid
    assert len(uploaded) == 1
    assert uploaded[0].key.startswith("builds/")
    assert _points_at_derived(uploaded[0].bucket, uploaded[0].key)


@pytest.mark.parametrize("field", ["source_id", "package_id"])
@pytest.mark.parametrize("declaration", [None, 123, "historical name"])
def test_raw_publication_preserves_history_without_new_identity_requirements(
    tmp_path, monkeypatch, field, declaration
):
    package, manifest_path, manifest = _package(tmp_path)
    spec = manifest["files"][2024]
    key = f"historical/route/{spec['sha256']}/table.csv"
    spec["storage"] = {"r2": {"provider": "r2", "uri": f"r2://archive/{key}"}}
    manifest[field] = declaration
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = manifest_path.read_text()
    _no_upload(monkeypatch)
    report = publish_source_artifacts(package)
    assert report.valid
    assert report.entries[0].skipped
    assert report.entries[0].r2_location.uri == f"r2://archive/{key}"
    assert manifest_path.read_text() == before


@pytest.mark.parametrize("field", ["source_id", "package_id"])
@pytest.mark.parametrize("yaml_identity", ["2024-01-01", "!!set {legacy: null}"])
@pytest.mark.parametrize("entrypoint", ["harness", "cli"])
def test_raw_cli_serializes_noncanonical_yaml_identity_refusals(
    tmp_path, monkeypatch, capsys, field, yaml_identity, entrypoint
):
    package, manifest_path, manifest = _package(tmp_path)
    manifest[field] = yaml.safe_load(yaml_identity)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = manifest_path.read_text()
    _no_upload(monkeypatch)
    args = ["publish-raw", "--root", str(package)]
    if entrypoint == "harness":
        assert harness_main(args) == 1
    else:
        monkeypatch.setattr("sys.argv", ["chronicle", *args])
        with pytest.raises(SystemExit) as exit_info:
            cli_main()
        assert exit_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert not payload["valid"]
    assert isinstance(payload["entries"][0][field], str)
    assert "r2_identity_invalid" in " ".join(payload["entries"][0]["errors"])
    assert manifest_path.read_text() == before


@pytest.mark.parametrize("bad_year", ["/2024", "..", "2024/elsewhere"])
@pytest.mark.parametrize("operation", ["raw", "derived"])
def test_publication_refuses_vintage_namespace_escape_before_reads(
    tmp_path, monkeypatch, bad_year, operation
):
    package, manifest_path, manifest = _package(tmp_path)
    manifest["files"][bad_year] = manifest["files"].pop(2024)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    _no_upload(monkeypatch)

    def unexpected_read(*args, **kwargs):
        pytest.fail("publication read reached with a vintage namespace escape")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    if operation == "raw":
        report = publish_source_artifacts(package)
    else:
        report = publish_derived_artifacts(
            package,
            source_id="publisher",
            package_id="package",
            year=bad_year,
            build_id="ledger.build.v1:peer4",
            r2_bucket="custom-store",
        )
    assert not report.valid


@pytest.mark.parametrize(
    "bad_build_id", ["ledger.build.v1:bad/id", "ledger.build.v1:bad id"]
)
def test_derived_publication_refuses_noncanonical_build_segment(
    tmp_path, monkeypatch, bad_build_id
):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "facts.jsonl").write_bytes(b"{}\n")
    _no_upload(monkeypatch)
    report = publish_derived_artifacts(
        suite,
        source_id="publisher",
        package_id="package",
        year=2024,
        build_id=bad_build_id,
    )
    assert not report.valid
    assert report.errors == ("malformed_build_id",)


@pytest.mark.parametrize("bad_year", ["/2024", "..", "2024/elsewhere"])
def test_fetch_refuses_vintage_namespace_escape_before_publisher_io(
    tmp_path, monkeypatch, bad_year
):
    package, _manifest_path, _manifest = _package(tmp_path)

    def unexpected_read(*args, **kwargs):
        pytest.fail("fetch reached publisher with a vintage namespace escape")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)
    with pytest.raises(SourceArtifactManifestError, match="year"):
        fetch_source_artifact(
            "https://example.test/table.csv",
            source_id="publisher",
            package_id="package",
            year=bad_year,
            output_dir=package,
        )
