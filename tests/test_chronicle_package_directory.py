"""The byte boundary spans every manifest a package directory keeps.

A publisher directory may keep several manifests -- ``manifest.yaml`` beside
``manifest_<package>.yaml`` files -- and ``fetch-artifact --manifest`` (PR
#226) selects which one a fetch records into. The boundary is the file in the
directory, not the manifest: a name or a digest registered hash-only in any
manifest there must not be fetched, published, parsed, or reclassified
through another. An artifact may not be named like a manifest, a fetch may
not record into a manifest that identifies another package, and
``register-artifact`` addresses a named manifest exactly as a fetch does.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import uuid

import pytest
import yaml

from chronicle.artifacts import (
    AmbiguousManifestError,
    _package_manifests,
    inventory_source_artifacts,
    publish_source_artifacts,
)
from chronicle.harness import main as harness_main
from chronicle.registration import (
    ArtifactFilenameError,
    HashOnlyRegistrationError,
    ManifestAccessError,
    is_manifest_filename,
    register_hash_only_artifact,
    validate_file_entry,
    validate_package_directory,
)
from chronicle.source_package import SOURCE_ARTIFACT_FETCH_ENV, SourceArtifactSpec
from tests.test_chronicle_microdata_registration import (
    ATTESTED,
    EVIDENCE,
    FIXTURE_SHA,
    LICENSED_BYTES,
    PUBLIC_BYTES,
    PUBLIC_SHA,
    _attested_entry,
    _fetch_release,
    _fetch_table,
    _forbid_uploads,
    _isolated_reader,
    _record_uploads,
    _refuse_read,
    _register,
    _serve,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LICENSED_SHA = hashlib.sha256(LICENSED_BYTES).hexdigest()


def _write(path: Path, payload: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path.read_bytes()


def _table_manifest(**fields: object) -> dict:
    payload = {
        "source_id": "dwp",
        "package_id": "dwp-frs-2023-24",
        "kind": "publisher_table",
        "files": {},
    }
    payload.update(fields)
    return payload


def _public_table_entry(filename: str, content: bytes) -> dict:
    return {
        "filename": filename,
        "source_url": f"https://publisher.example/{filename}",
        "access": "public",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _hash_only_manifest(**entry: object) -> dict:
    return {
        "source_id": "dwp",
        "package_id": "dwp-frs-2023-24",
        "kind": "microdata_release",
        "files": {2023: [_attested_entry(**entry)]},
    }


def _snapshot(package: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(package.iterdir())}


# --------------------------------------------------------------------------
# fetch-artifact
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "sibling"),
    [
        ("manifest_codebook.yaml", "manifest.yaml"),
        ("manifest.yaml", "manifest_release.yaml"),
        ("manifest_a.yaml", "manifest_b.yml"),
    ],
)
def test_fetch_refuses_a_name_a_sibling_manifest_registers_hash_only(
    tmp_path, monkeypatch, target, sibling
):
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    _write(package / sibling, _hash_only_manifest())
    # The target exists so the stray-default rule is not what refuses.
    _write(package / target, _table_manifest())
    before = _snapshot(package)
    reads = _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    with pytest.raises(ManifestAccessError, match=sibling) as refused:
        _fetch_table(
            tmp_path / "adult.tab",
            package,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            filename="adult.tab",
            manifest_filename=target,
            upload_r2=True,
        )

    assert "access='licensed'" in str(refused.value)
    assert reads == []
    assert _snapshot(package) == before


def test_fetch_refuses_bytes_a_sibling_manifest_registers_hash_only(
    tmp_path, monkeypatch
):
    """The same bytes under another name are the same gated artifact."""
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    _write(package / "manifest.yaml", _hash_only_manifest(sha256=LICENSED_SHA))
    _write(package / "manifest_tables.yaml", _table_manifest())
    before = _snapshot(package)

    # Pinned to the gated digest: refused before the read.
    reads = _refuse_read(monkeypatch)
    with pytest.raises(ManifestAccessError, match="manifest.yaml"):
        _fetch_table(
            tmp_path / "other.tab",
            package,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            filename="other.tab",
            manifest_filename="manifest_tables.yaml",
            expected_sha256=LICENSED_SHA,
        )
    assert reads == []

    # Not pinned: the served bytes turn out to be the gated ones, and are
    # refused before anything is written or uploaded.
    _serve(monkeypatch, LICENSED_BYTES)
    _forbid_uploads(monkeypatch)
    with pytest.raises(ManifestAccessError, match="manifest.yaml"):
        _fetch_table(
            tmp_path / "other.tab",
            package,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            filename="other.tab",
            manifest_filename="manifest_tables.yaml",
            upload_r2=True,
        )
    assert _snapshot(package) == before


def test_fetch_refuses_bytes_its_own_manifest_registers_hash_only(
    tmp_path, monkeypatch
):
    """A release manifest never archives a digest it registers hash-only."""
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    _write(package / "manifest.yaml", _hash_only_manifest(sha256=LICENSED_SHA))
    before = _snapshot(package)
    _serve(monkeypatch, LICENSED_BYTES)
    _forbid_uploads(monkeypatch)

    with pytest.raises(ManifestAccessError, match="adult.tab"):
        _fetch_release(
            package,
            staging_dir=tmp_path / "staging",
            filename="codebook.pdf",
            content=LICENSED_BYTES,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            vintage="2023_24",
            licence="OGL-UK-3.0",
            publisher="Department for Work and Pensions",
            licence_evidence={**EVIDENCE, "issuer": "DWP"},
            upload_r2=True,
        )

    assert _snapshot(package) == before
    assert not (tmp_path / "staging").exists()


def test_fetch_refuses_to_overwrite_a_file_a_sibling_manifest_records(
    tmp_path, monkeypatch
):
    """Two manifests may record one public file only as the same bytes."""
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    original = b"the table both manifests record"
    (package / "table.ods").parent.mkdir(parents=True)
    (package / "table.ods").write_bytes(original)
    _write(
        package / "manifest_source_package.yaml",
        _table_manifest(files={2024: _public_table_entry("table.ods", original)}),
    )
    _write(package / "manifest.yaml", _table_manifest())
    before = _snapshot(package)
    _serve(monkeypatch, b"a different table")

    with pytest.raises(ManifestAccessError, match="manifest_source_package.yaml"):
        _fetch_table(
            tmp_path / "table.ods",
            package,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            filename="table.ods",
        )

    assert _snapshot(package) == before

    # The same bytes are the same file: recorded twice, overwritten never.
    _serve(monkeypatch, original)
    report = _fetch_table(
        tmp_path / "table.ods",
        package,
        source_id="dwp",
        package_id="dwp-frs-2023-24",
        filename="table.ods",
    )
    assert report.valid
    assert (package / "table.ods").read_bytes() == original


def test_fetch_strictly_validates_a_sibling_before_any_publisher_read(
    tmp_path, monkeypatch
):
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    _write(package / "manifest_tables.yaml", _table_manifest())
    _write(
        package / "manifest.yaml",
        _table_manifest(
            files={
                2023: {
                    "filename": "adult.tab",
                    "Access": "licensed",
                    "sha256": LICENSED_SHA,
                }
            }
        ),
    )
    before = _snapshot(package)
    reads = _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    with pytest.raises(ManifestAccessError, match="misspelled_field:Access"):
        _fetch_table(
            tmp_path / "adult.tab",
            package,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            filename="adult.tab",
            manifest_filename="manifest_tables.yaml",
            upload_r2=True,
        )

    assert reads == []
    assert _snapshot(package) == before


def test_fetch_validation_finds_a_normalized_alias_of_hash_only_bytes(
    tmp_path, monkeypatch
):
    """Simulate a case-sensitive directory while running on folded APFS."""
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    _write(package / "manifest.yaml", _hash_only_manifest(sha256=LICENSED_SHA))
    actual_path = package / "ADULT.TAB"
    actual_path.write_bytes(LICENSED_BYTES)
    declared_path = package / "adult.tab"
    real_exists = Path.exists

    def case_sensitive_exists(path: Path) -> bool:
        if path == declared_path:
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", case_sensitive_exists)
    reads = _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    with pytest.raises(ManifestAccessError, match="bytes_present_for_hash_only_entry"):
        _fetch_release(
            package,
            staging_dir=tmp_path / "staging",
            filename="codebook.pdf",
            content=PUBLIC_BYTES,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            licence="OGL-UK-3.0",
            publisher="Department for Work and Pensions",
            vintage="2023_24",
            licence_evidence={**EVIDENCE, "issuer": "DWP"},
            upload_r2=True,
        )

    assert reads == []
    assert actual_path.read_bytes() == LICENSED_BYTES


def test_release_fetch_finds_a_normalized_alias_beside_the_manifest(
    tmp_path, monkeypatch
):
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    _write(
        package / "manifest.yaml",
        _hash_only_manifest(files={}) | {"files": {}, "publisher": "DWP"},
    )
    actual_path = package / "CODEBOOK.PDF"
    actual_path.write_bytes(PUBLIC_BYTES)
    declared_path = package / "codebook.pdf"
    real_exists = Path.exists

    def case_sensitive_exists(path: Path) -> bool:
        if path == declared_path:
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", case_sensitive_exists)
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="exists beside the manifest"):
        _fetch_release(
            package,
            staging_dir=tmp_path / "staging",
            filename="codebook.pdf",
            content=PUBLIC_BYTES,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            licence="OGL-UK-3.0",
            publisher="DWP",
            vintage="2023_24",
            licence_evidence={**EVIDENCE, "issuer": "DWP"},
            upload_r2=True,
        )

    assert reads == []


def test_package_manifests_refuses_distinct_normalized_name_aliases(
    tmp_path, monkeypatch
):
    """Portable simulation of two case-distinct files on a sensitive volume."""
    package = tmp_path / "package"
    selected = package / "MANIFEST.YAML"
    sibling = tmp_path / "case-distinct-entry" / "manifest.yaml"
    _write(selected, _table_manifest())
    _write(sibling, _hash_only_manifest())
    monkeypatch.setattr(
        "chronicle.artifacts.package_manifest_paths",
        lambda _output: [selected, sibling],
    )

    with pytest.raises(AmbiguousManifestError, match="normalized manifest name"):
        _package_manifests(package, selected, _table_manifest())


@pytest.mark.parametrize(
    "name",
    ["manifest.yaml", "MANIFEST.YAML", "manifest.yml", "manifest_tables.yaml"],
)
def test_an_artifact_may_not_be_named_like_a_manifest(tmp_path, monkeypatch, name):
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ArtifactFilenameError, match="manifest"):
        _fetch_table(
            tmp_path / name,
            package,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            filename=name,
        )
    with pytest.raises(ArtifactFilenameError, match="manifest"):
        _fetch_table(
            tmp_path / "publisher" / name,
            package,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
        )
    with pytest.raises(HashOnlyRegistrationError, match="manifest"):
        _register(package, filename=name)

    assert reads == []
    assert not package.exists()
    assert is_manifest_filename(name)
    assert f"manifest_named_filename:{name}" in validate_file_entry(
        _attested_entry(filename=name),
        kind="microdata_release",
        manifest={},
        local_file_exists=False,
    )


def test_fetch_refuses_a_manifest_that_identifies_another_package(
    tmp_path, monkeypatch
):
    package = tmp_path / "db" / "data" / "irs_soi" / "table_1_2"
    original = _write(
        package / "manifest.yaml",
        {
            "source_id": "irs_soi",
            "package_id": "soi-table-1-2",
            "kind": "publisher_table",
        },
    )
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="package_id='soi-table-1-2'"):
        _fetch_table(tmp_path / "t.xlsx", package, package_id="soi-table-9")
    with pytest.raises(ManifestAccessError, match="source_id='irs_soi'"):
        _fetch_table(tmp_path / "t.xlsx", package, source_id="irs")

    assert reads == []
    assert (package / "manifest.yaml").read_bytes() == original


# --------------------------------------------------------------------------
# publish-raw and inventory-artifacts
# --------------------------------------------------------------------------


def _mixed_directory(tmp_path: Path) -> Path:
    """manifest.yaml registers adult.tab hash-only; a sibling table manifest
    records the same name, and another file with the same bytes, as public."""
    root = tmp_path / "data"
    package = root / "dwp" / "frs_2023_24"
    _write(package / "manifest.yaml", _hash_only_manifest(sha256=LICENSED_SHA))
    (package / "adult.tab").write_bytes(LICENSED_BYTES)
    (package / "extract.tab").write_bytes(LICENSED_BYTES)
    (package / "table.ods").write_bytes(b"a public table")
    _write(
        package / "manifest_tables.yaml",
        _table_manifest(
            files={
                2023: _public_table_entry("adult.tab", LICENSED_BYTES),
                2022: _public_table_entry("extract.tab", LICENSED_BYTES),
                2021: _public_table_entry("table.ods", b"a public table"),
            }
        ),
    )
    return root


def _named_manifest_tree(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for index, manifest_name in enumerate(
        ("manifest_tables.yaml", "manifest_release.yml", "MANIFEST_UPPER.YAML")
    ):
        package = root / "dwp" / f"package_{index}"
        content = f"public table {index}".encode()
        filename = f"table_{index}.ods"
        package.mkdir(parents=True)
        (package / filename).write_bytes(content)
        _write(
            package / manifest_name,
            _table_manifest(
                files={2020 + index: _public_table_entry(filename, content)}
            ),
        )
    return root


def test_default_inventory_discovers_named_yaml_and_yml_manifests(tmp_path):
    root = _named_manifest_tree(tmp_path)

    report = inventory_source_artifacts(root)
    explicit = inventory_source_artifacts(
        root, manifest_filename="manifest_tables.yaml"
    )

    assert report.valid
    assert report.counts["manifest_count"] == 3
    assert report.counts["artifact_count"] == 3
    assert explicit.counts["manifest_count"] == 1
    assert explicit.counts["artifact_count"] == 1


def test_default_publish_discovers_named_yaml_and_yml_manifests(tmp_path, monkeypatch):
    root = _named_manifest_tree(tmp_path)
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(root)

    assert report.valid
    assert report.counts["manifest_count"] == 3
    assert report.counts["artifact_count"] == 3
    assert report.counts["uploaded_count"] == 3
    assert len(uploads) == 3


@pytest.mark.parametrize("operation", ["inventory", "publish"])
def test_artifact_sweeps_find_a_normalized_alias_of_hash_only_bytes(
    tmp_path, monkeypatch, operation
):
    package = tmp_path / "data" / "dwp" / "frs_2023_24"
    _write(package / "manifest.yaml", _hash_only_manifest(sha256=LICENSED_SHA))
    actual_path = package / "ADULT.TAB"
    actual_path.write_bytes(LICENSED_BYTES)
    declared_path = package / "adult.tab"
    real_exists = Path.exists

    def case_sensitive_exists(path: Path) -> bool:
        if path == declared_path:
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", case_sensitive_exists)
    if operation == "inventory":
        report = inventory_source_artifacts(package)
    else:
        uploads = _record_uploads(monkeypatch)
        report = publish_source_artifacts(package, skip_hash_only=True)
        assert uploads == []

    assert not report.valid
    assert any(
        "bytes_present_for_hash_only_entry" in error
        for entry in report.entries
        for error in entry.errors
    )


def test_publish_raw_never_uploads_what_a_sibling_manifest_registers_hash_only(
    tmp_path, monkeypatch
):
    root = _mixed_directory(tmp_path)
    package = root / "dwp" / "frs_2023_24"
    before = _snapshot(package)
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(root, manifest_filename="manifest_tables.yaml")

    assert not report.valid
    assert uploads == []
    assert report.entries == ()
    assert any(
        error.startswith("filename_collision_across_manifests:adult.tab")
        for error in report.errors
    )
    assert any(
        error.startswith(f"sha256_collision_across_manifests:{LICENSED_SHA}")
        for error in report.errors
    )
    assert _snapshot(package) == before


def test_publish_refuses_an_unpinned_public_alias_of_hash_only_bytes(
    tmp_path, monkeypatch
):
    package = tmp_path / "data" / "dwp" / "frs_2023_24"
    public = _public_table_entry("public-alias.tab", LICENSED_BYTES)
    public.pop("sha256")
    _write(
        package / "manifest_tables.yaml",
        _table_manifest(files={2023: public}),
    )
    _write(
        package / "manifest_release.yaml",
        _hash_only_manifest(sha256=LICENSED_SHA),
    )
    (package / "public-alias.tab").write_bytes(LICENSED_BYTES)
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(package, manifest_filename="manifest_tables.yaml")

    assert not report.valid
    assert uploads == []
    assert any(
        error.startswith(f"sha256_collision_across_manifests:{LICENSED_SHA}")
        for entry in report.entries
        for error in entry.errors
    )


def test_publish_strictly_validates_a_sibling_before_any_upload(tmp_path, monkeypatch):
    root = tmp_path / "data"
    package = root / "dwp" / "frs_2023_24"
    malformed = _hash_only_manifest(sha256=LICENSED_SHA)
    malformed["files"][2023][0]["Access"] = "licensed"
    malformed["files"][2023][0].pop("access")
    _write(package / "manifest.yaml", malformed)
    _write(
        package / "manifest_tables.yaml",
        _table_manifest(files={2023: _public_table_entry("adult.tab", LICENSED_BYTES)}),
    )
    (package / "adult.tab").write_bytes(LICENSED_BYTES)
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(root, manifest_filename="manifest_tables.yaml")

    assert not report.valid
    assert uploads == []
    assert any("misspelled_field:Access" in error for error in report.errors)


def test_publish_validates_a_sibling_before_reading_selected_bytes(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    package = root / "dwp" / "frs_2023_24"
    malformed = _hash_only_manifest(sha256=LICENSED_SHA)
    malformed["files"][2023][0]["Access"] = "licensed"
    malformed["files"][2023][0].pop("access")
    _write(package / "manifest.yaml", malformed)
    _write(
        package / "manifest_tables.yaml",
        _table_manifest(files={2023: _public_table_entry("adult.tab", LICENSED_BYTES)}),
    )
    artifact_path = package / "adult.tab"
    artifact_path.write_bytes(LICENSED_BYTES)
    reads = []
    real_read_bytes = Path.read_bytes

    def record_artifact_read(path):
        if path == artifact_path:
            reads.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_artifact_read)
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(root, manifest_filename="manifest_tables.yaml")

    assert not report.valid
    assert reads == []
    assert uploads == []
    assert any("misspelled_field:Access" in error for error in report.errors)


def test_publish_validates_the_complete_selected_manifest_before_any_upload(
    tmp_path, monkeypatch
):
    package = tmp_path / "data" / "dwp" / "frs_2023_24"
    malformed = _public_table_entry("other.tab", PUBLIC_BYTES)
    malformed["Access"] = "licensed"
    malformed.pop("access")
    _write(
        package / "manifest_tables.yaml",
        _table_manifest(
            files={
                2023: _public_table_entry("adult.tab", LICENSED_BYTES),
                2022: malformed,
            }
        ),
    )
    (package / "adult.tab").write_bytes(LICENSED_BYTES)
    (package / "other.tab").write_bytes(PUBLIC_BYTES)
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(package, manifest_filename="manifest_tables.yaml")

    assert not report.valid
    assert uploads == []
    assert any("misspelled_field:Access" in error for error in report.errors)


def test_publish_validates_selected_manifest_before_reading_any_bytes(
    tmp_path, monkeypatch
):
    package = tmp_path / "data" / "dwp" / "frs_2023_24"
    malformed = _public_table_entry("other.tab", PUBLIC_BYTES)
    malformed["Access"] = "licensed"
    malformed.pop("access")
    _write(
        package / "manifest_tables.yaml",
        _table_manifest(
            files={
                2023: _public_table_entry("adult.tab", LICENSED_BYTES),
                2022: malformed,
            }
        ),
    )
    artifact_paths = {package / "adult.tab", package / "other.tab"}
    (package / "adult.tab").write_bytes(LICENSED_BYTES)
    (package / "other.tab").write_bytes(PUBLIC_BYTES)
    reads = []
    real_read_bytes = Path.read_bytes

    def record_artifact_read(path):
        if path in artifact_paths:
            reads.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_artifact_read)
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(package, manifest_filename="manifest_tables.yaml")

    assert not report.valid
    assert reads == []
    assert uploads == []
    assert any("misspelled_field:Access" in error for error in report.errors)


def test_inventory_reports_collisions_across_manifests(tmp_path):
    root = _mixed_directory(tmp_path)

    report = inventory_source_artifacts(root, manifest_filename="manifest_tables.yaml")

    assert not report.valid
    codes = {error.split(": ")[0] for error in report.errors}
    assert "filename_collision_across_manifests:adult.tab" in codes
    assert f"sha256_collision_across_manifests:{LICENSED_SHA}" in codes
    # table.ods is recorded once and reported only for itself.
    assert not any("table.ods" in code for code in codes)


def test_inventory_does_not_read_bytes_for_an_invalid_entry(tmp_path, monkeypatch):
    package = tmp_path / "data" / "dwp" / "frs_2023_24"
    entry = _public_table_entry("adult.tab", LICENSED_BYTES)
    entry["Access"] = "licensed"
    entry.pop("access")
    _write(
        package / "manifest_tables.yaml",
        _table_manifest(files={2023: entry}),
    )
    artifact_path = package / "adult.tab"
    artifact_path.write_bytes(LICENSED_BYTES)
    reads = []
    real_read_bytes = Path.read_bytes

    def record_artifact_read(path):
        if path == artifact_path:
            reads.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_artifact_read)

    report = inventory_source_artifacts(
        package, manifest_filename="manifest_tables.yaml"
    )

    assert not report.valid
    assert reads == []
    assert "misspelled_field:Access" in report.entries[0].errors


def test_inventory_does_not_read_bytes_after_locator_validation_error(
    tmp_path, monkeypatch
):
    package = tmp_path / "data" / "dwp" / "frs_2023_24"
    entry = _public_table_entry("adult.tab", LICENSED_BYTES)
    key = f"raw/dwp/dwp-frs-2023-24/2023/{entry['sha256']}/{entry['filename']}"
    entry["storage"] = {
        "r2": {
            "provider": "r2",
            "bucket": "ledger-raw",
            "key": key,
            "uri": f"r2://other-bucket/{key}",
        }
    }
    _write(
        package / "manifest_tables.yaml",
        _table_manifest(files={2023: entry}),
    )
    artifact_path = package / "adult.tab"
    artifact_path.write_bytes(LICENSED_BYTES)
    reads = []
    real_read_bytes = Path.read_bytes

    def record_artifact_read(path):
        if path == artifact_path:
            reads.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_artifact_read)

    report = inventory_source_artifacts(
        package, manifest_filename="manifest_tables.yaml"
    )

    assert not report.valid
    assert reads == []
    assert report.entries[0].sha256_actual is None
    assert any(
        error.startswith("recorded_r2_locator_invalid:")
        for error in report.entries[0].errors
    )


def test_two_manifests_may_record_one_public_file_as_the_same_bytes(tmp_path):
    """The tracked shape: manifest.yaml and manifest_<package>.yaml both
    record one publisher file with one digest (db/data/usda_snap/...)."""
    root = tmp_path / "data"
    package = root / "usda_snap" / "fy69_to_current"
    content = b"snap zip"
    (package).mkdir(parents=True)
    (package / "snap.zip").write_bytes(content)
    entry = _public_table_entry("snap.zip", content)
    _write(package / "manifest.yaml", _table_manifest(files={2024: entry}))
    _write(package / "manifest_fy2025.yaml", _table_manifest(files={2025: entry}))

    assert (
        validate_package_directory(
            {
                "manifest.yaml": yaml.safe_load(
                    (package / "manifest.yaml").read_text()
                ),
                "manifest_fy2025.yaml": yaml.safe_load(
                    (package / "manifest_fy2025.yaml").read_text()
                ),
            }
        )
        == ()
    )
    assert inventory_source_artifacts(root).valid
    assert inventory_source_artifacts(
        root, manifest_filename="manifest_fy2025.yaml"
    ).valid


def test_the_committed_tree_has_no_collisions_across_manifests():
    report = inventory_source_artifacts(REPO_ROOT / "db" / "data")
    assert not [error for error in report.errors if "across_manifests" in error]


# --------------------------------------------------------------------------
# register-artifact
# --------------------------------------------------------------------------


def test_register_refuses_an_identity_a_sibling_manifest_holds_public(tmp_path):
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    archived = _public_table_entry("adult.tab", PUBLIC_BYTES)
    archived["storage"] = {
        "r2": {
            "provider": "r2",
            "bucket": "ledger-raw",
            "key": f"raw/uk/dwp/dwp-frs-2023-24/2023/{PUBLIC_SHA}/adult.tab",
            "uri": f"r2://ledger-raw/raw/uk/dwp/dwp-frs-2023-24/2023/{PUBLIC_SHA}/adult.tab",
        }
    }
    _write(
        package / "manifest_tables.yaml",
        _table_manifest(
            files={
                2023: archived,
                2022: _public_table_entry("extract.tab", b"unarchived public bytes"),
            }
        ),
    )
    _write(
        package / "manifest.yaml", {"source_id": "dwp", "package_id": "dwp-frs-2023-24"}
    )
    before = _snapshot(package)

    # By name: the sibling archived adult.tab.
    with pytest.raises(HashOnlyRegistrationError, match="manifest_tables.yaml"):
        _register(package)
    # By digest: the sibling archived these bytes under another name.
    with pytest.raises(HashOnlyRegistrationError, match="manifest_tables.yaml"):
        _register(package, filename="other.tab", sha256=PUBLIC_SHA)
    # A public record with no object is still an explicit class change.
    with pytest.raises(HashOnlyRegistrationError, match="manifest_tables.yaml"):
        _register(package, filename="extract.tab")

    assert _snapshot(package) == before


def test_register_refuses_a_stray_default_manifest_beside_named_ones(tmp_path):
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    _write(package / "manifest_tables.yaml", _table_manifest())
    before = _snapshot(package)

    with pytest.raises(HashOnlyRegistrationError, match="--manifest"):
        _register(package)

    assert _snapshot(package) == before


def test_register_targets_the_named_manifest(tmp_path, capsys):
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    _write(package / "manifest_tables.yaml", _table_manifest())

    report = _register(package, manifest_filename="manifest_release.yaml")

    assert report.valid
    assert report.manifest_path.endswith("manifest_release.yaml")
    written = yaml.safe_load((package / "manifest_release.yaml").read_text())
    assert written["kind"] == "microdata_release"
    assert [entry["filename"] for entry in written["files"][2023]] == ["adult.tab"]
    assert not (package / "manifest.yaml").exists()

    with pytest.raises(HashOnlyRegistrationError, match="manifest"):
        _register(package, manifest_filename="../manifest.yaml")
    with pytest.raises(HashOnlyRegistrationError, match="manifest"):
        _register(package, manifest_filename="adult.tab")

    argv = [
        "register-artifact",
        "--source-id",
        "dwp",
        "--package-id",
        "dwp-frs-2023-24",
        "--year",
        "2023",
        "--out-dir",
        str(package),
        "--manifest",
        "manifest_release.yaml",
        "--filename",
        "benefits.tab",
        "--sha256",
        FIXTURE_SHA,
        "--vintage",
        "2023_24",
        "--licence",
        "UK Data Service End User Licence",
        "--access",
        "licensed",
        "--doi",
        "10.5255/UKDA-SN-9367-2",
        "--hash-source",
        "consumer_attested",
        "--attested-by",
        ATTESTED["attested_by"],
        "--attestation-evidence",
        ATTESTED["attestation_evidence"],
        "--verified-at",
        ATTESTED["verified_at"],
    ]
    assert harness_main(argv) == 0
    capsys.readouterr()
    written = yaml.safe_load((package / "manifest_release.yaml").read_text())
    assert [entry["filename"] for entry in written["files"][2023]] == [
        "adult.tab",
        "benefits.tab",
    ]


# --------------------------------------------------------------------------
# The source-package byte reader
# --------------------------------------------------------------------------


@pytest.mark.parametrize("defect", ["selected_access_spelling", "same_manifest_pin"])
def test_byte_reader_validates_the_complete_selected_manifest(
    tmp_path, monkeypatch, defect
):
    _isolated_reader(tmp_path, monkeypatch)
    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "frs"
    resource_dir.mkdir(parents=True)
    selected = _public_table_entry("adult.tab", LICENSED_BYTES)
    files: dict[int, dict] = {2023: selected}
    if defect == "selected_access_spelling":
        selected["Access"] = "licensed"
        selected.pop("access")
    else:
        files[2022] = _attested_entry(
            filename="other.tab",
            sha256=LICENSED_SHA,
            source_url="https://ukdataservice.example/other.tab",
        )
    _write(resource_dir / "manifest.yaml", _table_manifest(files=files))
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Family Resources Survey",
        resource_package=package_name,
        resource_directory="data/dwp/frs",
        manifest="manifest.yaml",
        vintage="2023_24",
        extracted_at="2026-09-02",
        extraction_method="none",
        parser="delimited_text_full_rows",
        delimiter="\t",
        artifact_year=2023,
    )

    with pytest.raises(ManifestAccessError):
        spec.assert_parseable(2023)


def test_byte_reader_finds_a_normalized_alias_for_another_current_entry(
    tmp_path, monkeypatch
):
    _isolated_reader(tmp_path, monkeypatch)
    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "frs"
    resource_dir.mkdir(parents=True)
    files = {
        2023: _public_table_entry("table.csv", PUBLIC_BYTES),
        2022: _attested_entry(filename="adult.tab", sha256=LICENSED_SHA),
    }
    _write(resource_dir / "manifest.yaml", _table_manifest(files=files))
    (resource_dir / "table.csv").write_bytes(PUBLIC_BYTES)
    actual_path = resource_dir / "ADULT.TAB"
    actual_path.write_bytes(LICENSED_BYTES)
    declared_path = resource_dir / "adult.tab"
    real_is_file = Path.is_file

    def case_sensitive_is_file(path: Path) -> bool:
        if path == declared_path:
            return False
        return real_is_file(path)

    monkeypatch.setattr(Path, "is_file", case_sensitive_is_file)
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Family Resources Survey",
        resource_package=package_name,
        resource_directory="data/dwp/frs",
        manifest="manifest.yaml",
        vintage="2023_24",
        extracted_at="2026-09-02",
        extraction_method="none",
        parser="delimited_text_full_rows",
        delimiter=",",
        artifact_year=2023,
    )

    with pytest.raises(ManifestAccessError, match="bytes_present_for_hash_only_entry"):
        spec.assert_parseable(2023)


def test_byte_reader_refuses_a_file_a_sibling_manifest_registers_hash_only(
    tmp_path, monkeypatch
):
    import sys
    import uuid

    _isolated_reader(tmp_path, monkeypatch)
    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "frs"
    resource_dir.mkdir(parents=True)
    _write(
        resource_dir / "manifest.yaml",
        _table_manifest(files={2023: _public_table_entry("adult.tab", LICENSED_BYTES)}),
    )
    _write(
        resource_dir / "manifest_release.yaml",
        _hash_only_manifest(sha256=LICENSED_SHA),
    )
    (resource_dir / "adult.tab").write_bytes(LICENSED_BYTES)
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Family Resources Survey",
        resource_package=package_name,
        resource_directory="data/dwp/frs",
        manifest="manifest.yaml",
        vintage="2023_24",
        extracted_at="2026-09-02",
        extraction_method="none",
        parser="delimited_text_full_rows",
        delimiter="\t",
        artifact_year=2023,
    )

    with pytest.raises(ManifestAccessError, match="manifest_release.yaml"):
        spec.assert_parseable(2023)
    with pytest.raises(ManifestAccessError, match="manifest_release.yaml"):
        spec.build_source_rows(2023)


def test_byte_reader_strictly_validates_a_sibling_manifest(tmp_path, monkeypatch):
    _isolated_reader(tmp_path, monkeypatch)
    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "frs"
    resource_dir.mkdir(parents=True)
    _write(
        resource_dir / "manifest.yaml",
        _table_manifest(files={2023: _public_table_entry("adult.tab", PUBLIC_BYTES)}),
    )
    malformed = _hash_only_manifest(sha256=LICENSED_SHA)
    malformed["files"][2023][0]["Access"] = "licensed"
    malformed["files"][2023][0].pop("access")
    _write(resource_dir / "manifest_release.yaml", malformed)
    (resource_dir / "adult.tab").write_bytes(PUBLIC_BYTES)
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Family Resources Survey",
        resource_package=package_name,
        resource_directory="data/dwp/frs",
        manifest="manifest.yaml",
        vintage="2023_24",
        extracted_at="2026-09-02",
        extraction_method="none",
        parser="delimited_text_full_rows",
        delimiter="\t",
        artifact_year=2023,
    )

    with pytest.raises(ManifestAccessError, match="manifest_release.yaml"):
        spec.assert_parseable(2023)


def test_byte_reader_refuses_a_public_filename_collision_before_reading(
    tmp_path, monkeypatch
):
    from chronicle import source_package

    _isolated_reader(tmp_path, monkeypatch)
    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "frs"
    resource_dir.mkdir(parents=True)
    _write(
        resource_dir / "manifest.yaml",
        _table_manifest(files={2023: _public_table_entry("adult.tab", LICENSED_BYTES)}),
    )
    _write(
        resource_dir / "manifest_tables.yaml",
        _table_manifest(files={2023: _public_table_entry("ADULT.TAB", PUBLIC_BYTES)}),
    )
    (resource_dir / "adult.tab").write_bytes(LICENSED_BYTES)
    reads = []
    real_read = source_package._read_source_artifact_content

    def record_read(path, artifact):
        reads.append(path)
        return real_read(path, artifact)

    monkeypatch.setattr(source_package, "_read_source_artifact_content", record_read)
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Family Resources Survey",
        resource_package=package_name,
        resource_directory="data/dwp/frs",
        manifest="manifest.yaml",
        vintage="2023_24",
        extracted_at="2026-09-04",
        extraction_method="none",
        parser="delimited_text_full_rows",
        delimiter="\t",
        artifact_year=2023,
    )

    with pytest.raises(
        ManifestAccessError,
        match="filename_collision_across_manifests:adult.tab",
    ):
        spec._artifact_content(2023)
    assert reads == []


def test_byte_reader_refuses_an_unpinned_public_alias_of_hash_only_bytes(
    tmp_path, monkeypatch
):
    _isolated_reader(tmp_path, monkeypatch)
    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "frs"
    resource_dir.mkdir(parents=True)
    public = _public_table_entry("public-alias.tab", LICENSED_BYTES)
    public.pop("sha256")
    _write(
        resource_dir / "manifest_tables.yaml",
        _table_manifest(files={2023: public}),
    )
    _write(
        resource_dir / "manifest_release.yaml",
        _hash_only_manifest(sha256=LICENSED_SHA),
    )
    (resource_dir / "public-alias.tab").write_bytes(LICENSED_BYTES)
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Family Resources Survey",
        resource_package=package_name,
        resource_directory="data/dwp/frs",
        manifest="manifest_tables.yaml",
        vintage="2023_24",
        extracted_at="2026-09-04",
        extraction_method="none",
        parser="delimited_text_full_rows",
        delimiter="\t",
        artifact_year=2023,
    )

    with pytest.raises(ManifestAccessError, match=LICENSED_SHA):
        spec._artifact_content(2023)


@pytest.mark.parametrize(
    "declared_sha256",
    [None, "not-a-digest"],
    ids=("omitted", "malformed"),
)
def test_byte_reader_does_not_cache_an_unpinned_alias_before_refusal(
    tmp_path, monkeypatch, declared_sha256
):
    from chronicle import source_package

    cache_root = _isolated_reader(tmp_path, monkeypatch)
    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "frs"
    resource_dir.mkdir(parents=True)
    public = _public_table_entry("public-alias.tab", LICENSED_BYTES)
    if declared_sha256 is None:
        public.pop("sha256")
    else:
        public["sha256"] = declared_sha256
    _write(
        resource_dir / "manifest_tables.yaml",
        _table_manifest(files={2023: public}),
    )
    _write(
        resource_dir / "manifest_release.yaml",
        _hash_only_manifest(sha256=LICENSED_SHA),
    )
    fetches = []

    def record_fetch(source_url):
        fetches.append(source_url)
        return LICENSED_BYTES

    monkeypatch.setattr(source_package, "_fetch_source_artifact_content", record_fetch)
    monkeypatch.setenv(SOURCE_ARTIFACT_FETCH_ENV, "1")
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Family Resources Survey",
        resource_package=package_name,
        resource_directory="data/dwp/frs",
        manifest="manifest_tables.yaml",
        vintage="2023_24",
        extracted_at="2026-09-04",
        extraction_method="none",
        parser="delimited_text_full_rows",
        delimiter="\t",
        artifact_year=2023,
    )

    with pytest.raises((ManifestAccessError, ValueError)):
        spec._artifact_content(2023)
    assert fetches == []
    assert not cache_root.exists()


@pytest.mark.parametrize(
    "locator",
    [
        {
            "provider": "r2",
            "bucket": "ledger-raw",
            "key": (f"raw/dwp/dwp-frs-2023-24/2023/{LICENSED_SHA}/adult.tab"),
            "uri": (
                "r2://other-bucket/raw/dwp/dwp-frs-2023-24/2023/"
                f"{LICENSED_SHA}/adult.tab"
            ),
        },
        ["r2://ledger-raw/raw/dwp/dwp-frs-2023-24/2023/object"],
    ],
    ids=("contradictory-fields", "non-mapping"),
)
def test_source_reader_validates_r2_locator_before_reading(
    tmp_path, monkeypatch, locator
):
    from chronicle import source_package

    _isolated_reader(tmp_path, monkeypatch)
    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "frs"
    resource_dir.mkdir(parents=True)
    entry = _public_table_entry("adult.tab", LICENSED_BYTES)
    entry["storage"] = {"r2": locator}
    _write(
        resource_dir / "manifest.yaml",
        _table_manifest(files={2023: entry}),
    )
    (resource_dir / "adult.tab").write_bytes(LICENSED_BYTES)
    reads = []
    real_read = source_package._read_source_artifact_content

    def record_read(path, artifact):
        reads.append(path)
        return real_read(path, artifact)

    monkeypatch.setattr(source_package, "_read_source_artifact_content", record_read)
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Family Resources Survey",
        resource_package=package_name,
        resource_directory="data/dwp/frs",
        manifest="manifest.yaml",
        vintage="2023_24",
        extracted_at="2026-09-04",
        extraction_method="none",
        parser="delimited_text_full_rows",
        delimiter="\t",
        artifact_year=2023,
    )

    with pytest.raises(ManifestAccessError, match="storage.r2"):
        spec._artifact_content(2023)
    assert reads == []


def test_the_stray_default_manifest_rule_reaches_register_before_any_write(tmp_path):
    package = tmp_path / "db" / "data" / "irs_soi" / "ira_contributions"
    _write(package / "manifest_roth_source_package.yaml", _table_manifest())
    _write(package / "manifest_traditional_source_package.yaml", _table_manifest())
    before = _snapshot(package)

    with pytest.raises((HashOnlyRegistrationError, AmbiguousManifestError)):
        register_hash_only_artifact(
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            output_dir=package,
            filename="adult.tab",
            sha256=FIXTURE_SHA,
            licence="UK Data Service End User Licence",
            access="licensed",
            vintage="2023_24",
            doi="10.5255/UKDA-SN-9367-2",
            **ATTESTED,
        )

    assert _snapshot(package) == before


@pytest.mark.parametrize("upload_r2", [False, True])
def test_fetch_validates_proposed_cross_vintage_table_before_writes(
    tmp_path, monkeypatch, upload_r2
):
    package = tmp_path / "package"
    original = b"published table for 2022"
    _write(
        package / "manifest.yaml",
        _table_manifest(files={2022: _public_table_entry("table.csv", original)}),
    )
    (package / "table.csv").write_bytes(original)
    before = _snapshot(package)
    _serve(monkeypatch, b"different publisher bytes for 2023")
    uploads = _record_uploads(monkeypatch)
    writes = []

    def record_bytes(path, content):
        writes.append((path.name, content))
        return len(content)

    def record_text(path, content, *args, **kwargs):
        writes.append((path.name, content))
        return len(content)

    monkeypatch.setattr(Path, "write_bytes", record_bytes)
    monkeypatch.setattr(Path, "write_text", record_text)
    refusal = None
    try:
        _fetch_table(
            tmp_path / "table.csv",
            package,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            filename="table.csv",
            upload_r2=upload_r2,
        )
    except ManifestAccessError as error:
        refusal = error

    assert writes == [], "proposed manifest must be valid before writing any bytes"
    assert uploads == []
    assert refusal is not None
    assert "filename_collision:table.csv" in str(refusal)
    assert _snapshot(package) == before
