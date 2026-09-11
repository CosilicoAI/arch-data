"""Public microdata staging stays outside repositories and package trees."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
from pathlib import Path

import pytest
import yaml

from chronicle.artifacts import (
    ArtifactCommandResult,
    fetch_source_artifact,
    inventory_source_artifacts,
    microdata_staging_path,
    publish_source_artifacts,
)
from chronicle.registration import ManifestAccessError, validate_file_entry
from tests.test_chronicle_microdata_registration import (
    PUBLIC_BYTES,
    PUBLIC_SHA,
    _fetch_release,
    _record_uploads,
    _refuse_read,
    _serve,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(
    params=[
        "output-directory",
        "nested-output-directory",
        "repository-directory",
        "repository-data-directory",
        "another-package-directory",
        "another-package-yml",
        "another-named-package",
        "another-git-checkout",
        "another-git-worktree",
        "bare-git-repository",
        "symlink-component",
        "symlink-before-parent-component",
        "symlink-identity-component",
        "symlink-filename-component",
    ],
)
def unsafe_microdata_staging(tmp_path, request):
    destination = request.param
    output = tmp_path / "package"
    staging = tmp_path / "staging"
    if destination == "output-directory":
        staging = output
    elif destination == "nested-output-directory":
        staging = output / "nested" / "staging"
    elif destination == "repository-directory":
        staging = REPO_ROOT / ".chronicle-test-staging-refusal" / "nested"
    elif destination == "repository-data-directory":
        staging = REPO_ROOT / "db" / "data" / ".chronicle-test-staging-refusal"
    elif destination in {
        "another-package-directory",
        "another-package-yml",
        "another-named-package",
    }:
        package = tmp_path / "another-package"
        package.mkdir()
        manifest_name = {
            "another-package-directory": "manifest.yaml",
            "another-package-yml": "Manifest.yml",
            "another-named-package": "manifest_public.YAML",
        }[destination]
        (package / manifest_name).write_text(
            yaml.safe_dump({"kind": "publisher_table", "files": {}})
        )
        staging = package / "nested" / "staging"
    elif destination in {
        "another-git-checkout",
        "another-git-worktree",
        "bare-git-repository",
    }:
        repository = tmp_path / "another-repository"
        repository.mkdir()
        if destination == "another-git-checkout":
            (repository / ".git").mkdir()
        elif destination == "another-git-worktree":
            (repository / ".git").write_text("gitdir: /elsewhere/worktrees/example\n")
        else:
            (repository / "HEAD").write_text("ref: refs/heads/main\n")
            (repository / "objects").mkdir()
            (repository / "refs").mkdir()
        staging = repository / "nested" / "staging"
    else:
        outside = tmp_path / "outside" / "child"
        outside.mkdir(parents=True)
        if destination == "symlink-identity-component":
            staging.mkdir()
            (staging / "census_acs").symlink_to(outside, target_is_directory=True)
        elif destination == "symlink-filename-component":
            identity = (
                staging
                / "census_acs"
                / "census-acs-pums-2022-1yr"
                / "2022"
                / PUBLIC_SHA
            )
            identity.mkdir(parents=True)
            (identity / "csv_hus.zip").symlink_to(outside / "missing.zip")
        else:
            alias = tmp_path / "alias"
            alias.symlink_to(outside, target_is_directory=True)
            staging = (
                alias / ".." / "staging"
                if destination == "symlink-before-parent-component"
                else alias / "staging"
            )

    return output, staging


def test_fetch_refuses_unsafe_microdata_staging_before_publisher_read(
    monkeypatch, unsafe_microdata_staging
):
    output, staging = unsafe_microdata_staging
    effects = []
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: effects.append(("lock", path)) or nullcontext(),
    )
    monkeypatch.setattr(
        "chronicle.artifacts._read_artifact",
        lambda url: (
            effects.append(("publisher_read", url)) or (PUBLIC_BYTES, "csv_hus.zip")
        ),
    )
    # Record every mutation instead of putting release bytes anywhere, even
    # while this regression runs against the vulnerable implementation.
    monkeypatch.setattr(
        Path, "mkdir", lambda path, **kwargs: effects.append(("mkdir", path))
    )
    monkeypatch.setattr(
        Path,
        "write_bytes",
        lambda path, content: effects.append(("write_bytes", path)),
    )
    monkeypatch.setattr(
        "chronicle.artifacts._upsert_manifest",
        lambda path, **kwargs: effects.append(("manifest_write", path)),
    )
    monkeypatch.setattr(
        "chronicle.artifacts._upload_r2_object",
        lambda location, path, **kwargs: (
            effects.append(("upload", path))
            or ArtifactCommandResult(
                command=("stub",), returncode=0, stdout="", stderr=""
            )
        ),
    )

    with pytest.raises(ManifestAccessError, match="[Ss]taging"):
        _fetch_release(output, staging_dir=staging, upload_r2=True)

    assert effects == []


def test_publish_refuses_unsafe_microdata_staging_before_read_or_upload(
    tmp_path, monkeypatch, unsafe_microdata_staging
):
    output, staging = unsafe_microdata_staging
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output, staging_dir=tmp_path / "safe-staging")
    manifest_path = output / "manifest.yaml"
    before = manifest_path.read_bytes()
    staged = microdata_staging_path(
        staging_dir=staging,
        source_id="census_acs",
        package_id="census-acs-pums-2022-1yr",
        year=2022,
        sha256=PUBLIC_SHA,
        filename="csv_hus.zip",
    )
    effects = []
    original_is_file = Path.is_file
    original_read_bytes = Path.read_bytes
    # Simulate existing staged bytes without writing into any unsafe tree.
    monkeypatch.setattr(
        Path, "is_file", lambda path: path == staged or original_is_file(path)
    )

    def read_bytes(path):
        if path == staged:
            effects.append(("artifact_read", path))
            return PUBLIC_BYTES
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: effects.append(("lock", path)) or nullcontext(),
    )
    monkeypatch.setattr(
        "chronicle.artifacts._upload_r2_object",
        lambda location, path, **kwargs: (
            effects.append(("upload", path))
            or ArtifactCommandResult(
                command=("stub",), returncode=0, stdout="", stderr=""
            )
        ),
    )
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda path, content, **kwargs: effects.append(("manifest_write", path)),
    )

    report = publish_source_artifacts(output, staging_dir=staging)

    assert effects == [], report
    assert not report.valid
    assert any(
        ("staging" in error.lower() or "artifact_path_is_symlink" in error)
        for error in (
            *report.errors,
            *(e for entry in report.entries for e in entry.errors),
        )
    )
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("spelling", ["absolute", "relative", "parent-component"])
def test_fetch_accepts_external_microdata_staging(tmp_path, monkeypatch, spelling):
    output = tmp_path / "package"
    external = tmp_path / "package-external"
    staging = external
    if spelling == "relative":
        monkeypatch.chdir(tmp_path)
        staging = Path("package-external")
    elif spelling == "parent-component":
        (tmp_path / "existing").mkdir()
        staging = tmp_path / "existing" / ".." / "package-external"
    _serve(monkeypatch, PUBLIC_BYTES)
    uploads = _record_uploads(monkeypatch)

    report = _fetch_release(output, staging_dir=staging, upload_r2=True)

    assert report.valid
    staged = Path(report.local_path)
    assert staged.is_relative_to(external)
    assert staged.read_bytes() == PUBLIC_BYTES
    assert uploads == [(report.r2_location.uri, str(staged))]
    assert sorted(path.name for path in output.iterdir()) == ["manifest.yaml"]


def _table_fetch_beside_public_microdata(package, *, alias, identity):
    """Build valid metadata; only the selected table's classification is wrong."""
    content = b"person_id,age\n1,45\n2,31\n"
    digest = hashlib.sha256(content).hexdigest()

    def locator(filename, sha256):
        key = f"raw/publisher/package/2023/{sha256}/{filename}"
        return {
            "provider": "r2",
            "bucket": "ledger-raw",
            "key": key,
            "uri": f"r2://ledger-raw/{key}",
        }

    entry = {
        "filename": "table.csv" if alias == "filename" else "microdata.csv",
        "sha256": digest,
        "size_bytes": len(content),
        "source_url": "https://publisher.example/microdata.csv",
        "access": "public",
        "licence": "CC0-1.0",
        "vintage": "2023",
        "hash_source": "chronicle_fetch",
        "attested_by": "chronicle",
        "verified_at": "2026-09-05",
    }
    history = []
    if alias.startswith("archived-"):
        entry.update(filename="new-microdata.csv", sha256="a" * 64)
        history.append(
            locator("table.csv", "b" * 64)
            if alias == "archived-filename"
            else locator("old-microdata.csv", digest)
        )
    entry["storage"] = {
        "r2": locator(entry["filename"], entry["sha256"]),
        "previous_r2": history,
    }
    entry["licence_evidence"] = {
        "issuer": "Fixture publisher",
        "scope": "This fixture public microdata release is dedicated to CC0.",
        "url": "https://publisher.example/licence",
        "licence": entry["licence"],
        "sha256": entry["sha256"],
    }
    assert (
        validate_file_entry(
            entry, kind="microdata_release", manifest={}, local_file_exists=False
        )
        == ()
    )
    selected = {
        "filename": "table.csv",
        "source_url": "https://publisher.example/table.csv",
    }
    if identity == "declared":
        selected["sha256"] = digest
    elif identity == "r2-only":
        selected["storage"] = {"r2": locator("table.csv", digest)}
    package.mkdir()
    for name, kind, spec in (
        ("manifest_tables.yaml", "publisher_table", selected),
        ("manifest_release.yaml", "microdata_release", [entry]),
    ):
        (package / name).write_text(
            yaml.safe_dump(
                {
                    "source_id": "publisher",
                    "package_id": "package",
                    "kind": kind,
                    "files": {2023: spec},
                }
            )
        )
    (package / "README.txt").write_bytes(b"preserve unrelated package bytes\n")
    kwargs = {
        "source_id": "publisher",
        "package_id": "package",
        "year": 2023,
        "output_dir": package,
        "filename": "table.csv",
        "manifest_filename": "manifest_tables.yaml",
    }
    if identity == "expected":
        kwargs["expected_sha256"] = digest
    return content, kwargs


@pytest.mark.parametrize(
    "alias", ["filename", "sha256", "archived-filename", "archived-sha256"]
)
@pytest.mark.parametrize("identity", ["declared", "r2-only", "expected", "observed"])
@pytest.mark.parametrize("upload", [False, True])
def test_table_fetch_refuses_public_microdata_alias_without_package_writes(
    tmp_path, monkeypatch, alias, identity, upload
):
    package = tmp_path / "package"
    content, kwargs = _table_fetch_beside_public_microdata(
        package, alias=alias, identity=identity
    )
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    reads = _serve(monkeypatch, content)
    uploads = _record_uploads(monkeypatch)
    locks = []
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )
    # The only permitted operation for an unknown checksum is the in-memory
    # publisher stub; known identities must be refused before even the lock.
    known = identity != "observed" or alias.endswith("filename")
    try:
        with pytest.raises(ManifestAccessError, match="microdata"):
            fetch_source_artifact(
                "https://publisher.example/table.csv", upload_r2=upload, **kwargs
            )
    finally:
        assert {path.name: path.read_bytes() for path in package.iterdir()} == before
        assert uploads == []
        assert reads == ([] if known else ["https://publisher.example/table.csv"])
        assert locks == ([] if known else [package])


@pytest.mark.parametrize("identified", [False, True])
@pytest.mark.parametrize("upload", [False, True])
def test_table_fetch_accepts_distinct_bytes_beside_public_microdata(
    tmp_path, monkeypatch, identified, upload
):
    package = tmp_path / "package"
    _content, kwargs = _table_fetch_beside_public_microdata(
        package, alias="sha256", identity="observed"
    )
    content = b"year,total_people\n2023,1234\n"
    digest = hashlib.sha256(content).hexdigest()
    manifest = package / "manifest_tables.yaml"
    if identified:
        payload = yaml.safe_load(manifest.read_text())
        payload["files"][2023]["sha256"] = digest
        manifest.write_text(yaml.safe_dump(payload))
    sibling = package / "manifest_release.yaml"
    before = sibling.read_bytes()
    reads = _serve(monkeypatch, content)
    uploads = _record_uploads(monkeypatch)
    # Exercise relative output paths as well as the absolute refusal fixtures.
    monkeypatch.chdir(tmp_path)
    kwargs["output_dir"] = Path("package")

    report = fetch_source_artifact(
        "https://publisher.example/table.csv", upload_r2=upload, **kwargs
    )

    assert report.valid
    assert reads == ["https://publisher.example/table.csv"]
    assert (package / "table.csv").read_bytes() == content
    assert sibling.read_bytes() == before
    assert len(uploads) == int(upload)


def test_table_fetch_rechecks_public_microdata_identity_under_lock(
    tmp_path, monkeypatch
):
    package = tmp_path / "package"
    content, kwargs = _table_fetch_beside_public_microdata(
        package, alias="sha256", identity="observed"
    )
    table_path = package / "manifest_tables.yaml"
    table = yaml.safe_load(table_path.read_text())
    table["files"][2023]["sha256"] = "b" * 64
    table_path.write_text(yaml.safe_dump(table))
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    reads = _serve(monkeypatch, content)
    uploads = _record_uploads(monkeypatch)
    locks = []

    @contextmanager
    def change_identity_while_acquiring_lock(output):
        locks.append(output)
        table["files"][2023]["sha256"] = hashlib.sha256(content).hexdigest()
        table_path.write_text(yaml.safe_dump(table))
        before[table_path.name] = table_path.read_bytes()
        yield

    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock", change_identity_while_acquiring_lock
    )
    with pytest.raises(ManifestAccessError, match="microdata"):
        fetch_source_artifact(
            "https://publisher.example/table.csv", upload_r2=True, **kwargs
        )

    assert locks == [package]
    assert reads == []
    assert uploads == []
    assert {path.name: path.read_bytes() for path in package.iterdir()} == before


TABLE_BYTES = b"person_id,age\n1,45\n2,31\n"
TABLE_SHA = hashlib.sha256(TABLE_BYTES).hexdigest()
RELEASE_BYTES = b"public household pums beside a publisher table"
RELEASE_SHA = hashlib.sha256(RELEASE_BYTES).hexdigest()


def _table_beside_public_microdata(package, staging, *, alias, identity):
    """Stage a package whose table bytes carry a public release's identity.

    Mirrors ``_table_fetch_beside_public_microdata`` for the publish and
    inventory paths: the table file is already in the package directory and the
    release's own bytes are staged outside it, so every entry is otherwise
    publishable. Only the selected table's classification is wrong.
    """

    def locator(filename, sha256):
        key = f"raw/publisher/package/2023/{sha256}/{filename}"
        return {
            "provider": "r2",
            "bucket": "ledger-raw",
            "key": key,
            "uri": f"r2://ledger-raw/{key}",
        }

    release_name = "table.csv" if alias == "filename" else "microdata.csv"
    release_sha = TABLE_SHA if alias == "sha256" else RELEASE_SHA
    history = []
    if alias == "archived-filename":
        history.append(locator("table.csv", "b" * 64))
    elif alias == "archived-sha256":
        history.append(locator("old-microdata.csv", TABLE_SHA))
    release_bytes = TABLE_BYTES if release_sha == TABLE_SHA else RELEASE_BYTES
    entry = {
        "filename": release_name,
        "sha256": release_sha,
        "size_bytes": len(release_bytes),
        "source_url": f"https://publisher.example/{release_name}",
        "access": "public",
        "licence": "CC0-1.0",
        "vintage": "2023",
        "hash_source": "chronicle_fetch",
        "attested_by": "chronicle",
        "verified_at": "2026-09-05",
        "storage": {"r2": locator(release_name, release_sha), "previous_r2": history},
        "licence_evidence": {
            "issuer": "Fixture publisher",
            "scope": "This fixture public microdata release is dedicated to CC0.",
            "url": "https://publisher.example/licence",
            "licence": "CC0-1.0",
            "sha256": release_sha,
        },
    }
    assert (
        validate_file_entry(
            entry, kind="microdata_release", manifest={}, local_file_exists=False
        )
        == ()
    )
    selected = {
        "filename": "table.csv",
        "source_url": "https://publisher.example/table.csv",
        "access": "public",
        "licence": "CC0-1.0",
    }
    if identity == "declared":
        selected["sha256"] = TABLE_SHA
        selected["size_bytes"] = len(TABLE_BYTES)
    elif identity == "r2-only":
        selected["storage"] = {"r2": locator("table.csv", TABLE_SHA)}
    package.mkdir(parents=True)
    for name, kind, spec in (
        ("manifest_tables.yaml", "publisher_table", selected),
        ("manifest_release.yaml", "microdata_release", [entry]),
    ):
        (package / name).write_text(
            yaml.safe_dump(
                {
                    "source_id": "publisher",
                    "package_id": "package",
                    "kind": kind,
                    "files": {2023: spec},
                }
            )
        )
    (package / "table.csv").write_bytes(TABLE_BYTES)
    (package / "README.txt").write_bytes(b"preserve unrelated package bytes\n")
    staged = microdata_staging_path(
        staging_dir=staging,
        source_id="publisher",
        package_id="package",
        year=2023,
        sha256=release_sha,
        filename=release_name,
    )
    staged.parent.mkdir(parents=True)
    staged.write_bytes(release_bytes)
    return staged


def _report_error_codes(report):
    return [
        *report.errors,
        *(code for entry in report.entries for code in entry.errors),
    ]


@pytest.mark.parametrize("alias", ["sha256", "archived-filename", "archived-sha256"])
@pytest.mark.parametrize("identity", ["declared", "observed", "r2-only"])
def test_publish_raw_refuses_public_microdata_alias_without_upload_or_rewrite(
    tmp_path, monkeypatch, alias, identity
):
    package = tmp_path / "package"
    staging = tmp_path / "staging"
    staged = _table_beside_public_microdata(
        package, staging, alias=alias, identity=identity
    )
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    staged_before = staged.read_bytes()
    _refuse_read(monkeypatch, "a publisher was read")
    uploads = _record_uploads(monkeypatch)
    writes = []
    reads = []
    original_read_bytes = Path.read_bytes

    def record_read(path):
        if path == package / "table.csv":
            reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    # Record every manifest rewrite instead of performing it, so this
    # regression cannot corrupt the package while the fix is missing.
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda path, content, **kwargs: writes.append(path),
    )

    report = publish_source_artifacts(
        package, manifest_filename="manifest_tables.yaml", staging_dir=staging
    )

    codes = _report_error_codes(report)
    assert not report.valid, codes
    assert any("bytes_identified_by_microdata_release" in code for code in codes), codes
    assert uploads == []
    assert writes == []
    # Read the log before the snapshot assertions below read the tree again.
    # Only an undeclared, unrecorded table digest needs the local bytes to be
    # classified; every other alias is known from metadata alone.
    known = identity != "observed" or alias == "archived-filename"
    assert reads == ([] if known else [package / "table.csv"])
    assert {
        path.name: original_read_bytes(path) for path in package.iterdir()
    } == before
    assert original_read_bytes(staged) == staged_before


@pytest.mark.parametrize("alias", ["sha256", "archived-filename", "archived-sha256"])
@pytest.mark.parametrize("identity", ["declared", "observed", "r2-only"])
def test_inventory_reports_public_microdata_alias_for_table_entry(
    tmp_path, alias, identity
):
    package = tmp_path / "package"
    staging = tmp_path / "staging"
    _table_beside_public_microdata(package, staging, alias=alias, identity=identity)

    report = inventory_source_artifacts(package, staging_dir=staging)

    codes = _report_error_codes(report)
    assert not report.valid, codes
    assert any("bytes_identified_by_microdata_release" in code for code in codes), codes
    assert any("manifest_release.yaml" in code for code in codes), codes
    # A metadata alias is a package defect, reported with the registration that
    # carries it as well as the release that already identifies those bytes.
    if identity != "observed" or alias == "archived-filename":
        assert any(
            "manifest_tables.yaml" in code and "manifest_release.yaml" in code
            for code in codes
        ), codes


@pytest.mark.parametrize("identity", ["declared", "observed"])
def test_publish_raw_accepts_distinct_table_bytes_beside_public_microdata(
    tmp_path, monkeypatch, identity
):
    package = tmp_path / "package"
    staging = tmp_path / "staging"
    staged = _table_beside_public_microdata(
        package, staging, alias="sha256", identity="observed"
    )
    content = b"year,total_people\n2023,1234\n"
    digest = hashlib.sha256(content).hexdigest()
    (package / "table.csv").write_bytes(content)
    if identity == "declared":
        manifest_path = package / "manifest_tables.yaml"
        payload = yaml.safe_load(manifest_path.read_text())
        payload["files"][2023]["sha256"] = digest
        payload["files"][2023]["size_bytes"] = len(content)
        manifest_path.write_text(yaml.safe_dump(payload))
    release_before = (package / "manifest_release.yaml").read_bytes()
    staged_before = staged.read_bytes()
    _refuse_read(monkeypatch, "a publisher was read")
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(
        package, manifest_filename="manifest_tables.yaml", staging_dir=staging
    )
    inventory = inventory_source_artifacts(package, staging_dir=staging)

    assert report.valid, _report_error_codes(report)
    assert inventory.valid, _report_error_codes(inventory)
    assert len(uploads) == 1
    assert uploads[0][0].endswith(f"/2023/{digest}/table.csv")
    assert (package / "manifest_release.yaml").read_bytes() == release_before
    assert staged.read_bytes() == staged_before


def test_publish_raw_keeps_two_manifests_sharing_one_public_table(
    tmp_path, monkeypatch
):
    """The tracked shape: two non-release manifests record one public file."""
    package = tmp_path / "package"
    package.mkdir()
    content = b"year,total_people\n2023,1234\n"
    digest = hashlib.sha256(content).hexdigest()
    entry = {
        "filename": "table.csv",
        "sha256": digest,
        "size_bytes": len(content),
        "source_url": "https://publisher.example/table.csv",
        "access": "public",
        "licence": "CC0-1.0",
    }
    for name in ("manifest.yaml", "manifest_tables.yaml"):
        (package / name).write_text(
            yaml.safe_dump(
                {
                    "source_id": "publisher",
                    "package_id": "package",
                    "kind": "publisher_table",
                    "files": {2023: dict(entry)},
                }
            )
        )
    (package / "table.csv").write_bytes(content)
    _refuse_read(monkeypatch, "a publisher was read")
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(package, manifest_filename="manifest_tables.yaml")
    inventory = inventory_source_artifacts(package)

    assert report.valid, _report_error_codes(report)
    assert inventory.valid, _report_error_codes(inventory)
    assert len(uploads) == 1


def _release_fetch_beside_public_table(package, *, alias):
    """Register a public table, then prepare a release fetch for its identity.

    The mirror of :func:`_table_beside_public_microdata`: the contradiction is
    created by the release fetch rather than found by publish, so nothing may
    reach the publisher, the staging directory, or either manifest.
    """

    def locator(filename, sha256):
        key = f"raw/census_acs/census-acs-pums-2022-1yr/2022/{sha256}/{filename}"
        return {
            "provider": "r2",
            "bucket": "ledger-raw",
            "key": key,
            "uri": f"r2://ledger-raw/{key}",
        }

    other = b"a second public release file in the same vintage"
    other_sha = hashlib.sha256(other).hexdigest()
    table = {
        "filename": "table.csv",
        "source_url": "https://publisher.example/table.csv",
        "access": "public",
        "licence": "CC0-1.0",
    }
    if alias == "sha256":
        table["sha256"] = PUBLIC_SHA
        table["size_bytes"] = len(PUBLIC_BYTES)
    else:
        table["storage"] = {"r2": locator("table.csv", PUBLIC_SHA)}
        table["sha256"] = PUBLIC_SHA
        table["size_bytes"] = len(PUBLIC_BYTES)
    existing = {
        "filename": "csv_pus.zip",
        "sha256": other_sha,
        "size_bytes": len(other),
        "source_url": "https://publisher.example/pums/csv_pus.zip",
        "access": "public",
        "licence": "US-Government-Work",
        "vintage": "2022",
        "hash_source": "chronicle_fetch",
        "attested_by": "chronicle",
        "verified_at": "2026-09-05",
        "storage": {"r2": locator("csv_pus.zip", other_sha)},
        "licence_evidence": {
            "issuer": "U.S. Census Bureau",
            "scope": "Public-use file of a federal agency; 17 U.S.C. 105",
            "url": "https://publisher.example/licence",
            "licence": "US-Government-Work",
            "sha256": other_sha,
        },
    }
    package.mkdir(parents=True)
    for name, kind, spec in (
        ("manifest_tables.yaml", "publisher_table", table),
        ("manifest_release.yaml", "microdata_release", [existing]),
    ):
        (package / name).write_text(
            yaml.safe_dump(
                {
                    "source_id": "census_acs",
                    "package_id": "census-acs-pums-2022-1yr",
                    "kind": kind,
                    "files": {2022: spec},
                }
            )
        )
    (package / "table.csv").write_bytes(PUBLIC_BYTES)


@pytest.mark.parametrize("alias", ["sha256", "r2-only"])
@pytest.mark.parametrize("upload", [False, True])
def test_release_fetch_refuses_identity_a_public_table_already_claims(
    tmp_path, monkeypatch, alias, upload
):
    package = tmp_path / "package"
    staging = tmp_path / "staging"
    _release_fetch_beside_public_table(package, alias=alias)
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    reads = _refuse_read(monkeypatch, "a publisher was read")
    uploads = _record_uploads(monkeypatch)
    writes = []
    monkeypatch.setattr(
        Path, "write_bytes", lambda path, content: writes.append(("stage", path))
    )
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda path, content, **kwargs: writes.append(("manifest", path)),
    )

    with pytest.raises(ManifestAccessError, match="microdata release"):
        _fetch_release(
            package,
            staging_dir=staging,
            manifest_filename="manifest_release.yaml",
            upload_r2=upload,
        )

    assert reads == []
    assert uploads == []
    assert writes == []
    assert {path.name: path.read_bytes() for path in package.iterdir()} == before
    assert not staging.exists()


@pytest.mark.parametrize("upload", [False, True])
def test_release_fetch_accepts_a_distinct_public_table_sibling(
    tmp_path, monkeypatch, upload
):
    package = tmp_path / "package"
    staging = tmp_path / "staging"
    _release_fetch_beside_public_table(package, alias="sha256")
    table_path = package / "manifest_tables.yaml"
    payload = yaml.safe_load(table_path.read_text())
    content = b"year,total_people\n2022,1234\n"
    payload["files"][2022]["sha256"] = hashlib.sha256(content).hexdigest()
    payload["files"][2022]["size_bytes"] = len(content)
    payload["files"][2022].pop("storage", None)
    table_path.write_text(yaml.safe_dump(payload))
    (package / "table.csv").write_bytes(content)
    table_before = table_path.read_bytes()
    _serve(monkeypatch, PUBLIC_BYTES)
    uploads = _record_uploads(monkeypatch)

    report = _fetch_release(
        package,
        staging_dir=staging,
        manifest_filename="manifest_release.yaml",
        upload_r2=upload,
    )

    assert report.valid
    assert Path(report.local_path).read_bytes() == PUBLIC_BYTES
    assert table_path.read_bytes() == table_before
    assert len(uploads) == int(upload)
