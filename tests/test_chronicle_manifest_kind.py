"""The explicit-kind rule and the repository guard for microdata bytes.

Every manifest created or modified after the microdata-identity ADR declares
``kind``. Manifests that predate the rule are frozen, byte for byte, in
``chronicle/grandfathered_manifests.py``; any other kindless manifest is an
error at every entry point -- ``fetch-artifact``, ``publish-raw``,
``inventory-artifacts``, ``validate-package`` and the source-package byte
reader -- and never a publisher table by default. A ``kind: microdata_release``
package directory holds only manifests: public release bytes are archived in
R2 from a staging directory outside the tree, never committed.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from chronicle.artifacts import (
    fetch_source_artifact,
    inventory_source_artifacts,
    publish_source_artifacts,
)
from chronicle.grandfathered_manifests import (
    GRANDFATHERED_KINDLESS_MANIFESTS,
    grandfathered_manifest_key,
    is_grandfathered_manifest,
    manifest_digest,
)
from chronicle.registration import (
    ManifestAccessError,
    ManifestKindError,
    manifest_kind,
    safe_manifest_kind,
)
from chronicle.source_package import SourceArtifactSpec, validate_source_package


REPO_ROOT = Path(__file__).resolve().parents[1]
# The ceiling ratchets down with the list. The freeze held 168 entries at the
# ``ba8147a7`` snapshot; every removal lowers this bound, so nothing can be
# added back into a slot a removal vacated.
FREEZE_SIZE = 163


def _tracked_manifests() -> list[Path]:
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "db/data"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [
        REPO_ROOT / path
        for path in listed
        if Path(path).name.startswith("manifest") and path.endswith((".yaml", ".yml"))
    ]


def _kindless(path: Path) -> bool:
    payload = yaml.safe_load(path.read_text()) or {}
    return isinstance(payload, dict) and "kind" not in payload


def test_every_kindless_manifest_in_the_tree_is_frozen_unmodified():
    tracked = _tracked_manifests()
    assert tracked
    kindless = {
        path.relative_to(REPO_ROOT).as_posix(): path
        for path in tracked
        if _kindless(path)
    }

    missing = sorted(set(kindless) - set(GRANDFATHERED_KINDLESS_MANIFESTS))
    assert missing == [], (
        "kindless manifests outside the frozen list; declare `kind:` in them: "
        f"{missing}"
    )
    modified = sorted(
        key
        for key, path in kindless.items()
        if manifest_digest(path) != GRANDFATHERED_KINDLESS_MANIFESTS[key]
    )
    assert modified == [], (
        "frozen manifests modified without declaring `kind:`; declare it and "
        f"drop them from the frozen list: {modified}"
    )


def test_the_frozen_list_only_shrinks():
    # Entries leave the list once a manifest declares its kind; none is added.
    assert len(GRANDFATHERED_KINDLESS_MANIFESTS) <= FREEZE_SIZE
    stale = sorted(
        key
        for key in GRANDFATHERED_KINDLESS_MANIFESTS
        if not (REPO_ROOT / key).exists() or not _kindless(REPO_ROOT / key)
    )
    assert stale == [], f"drop from the frozen list, the kind is declared: {stale}"


def _frozen_copy(tmp_path: Path) -> tuple[Path, str]:
    key = "db/data/irs_soi/table_1_2/manifest.yaml"
    assert key in GRANDFATHERED_KINDLESS_MANIFESTS
    copy = tmp_path / key
    copy.parent.mkdir(parents=True)
    copy.write_bytes((REPO_ROOT / key).read_bytes())
    return copy, key


def test_a_frozen_manifest_reads_as_a_publisher_table_until_it_is_modified(tmp_path):
    copy, key = _frozen_copy(tmp_path)
    payload = yaml.safe_load(copy.read_text())

    assert grandfathered_manifest_key(copy) == key
    assert is_grandfathered_manifest(copy)
    assert manifest_kind(payload, manifest_path=copy) == "publisher_table"
    assert safe_manifest_kind(payload, manifest_path=copy) == ("publisher_table", None)

    copy.write_bytes(copy.read_bytes() + b"# touched\n")
    assert not is_grandfathered_manifest(copy)
    with pytest.raises(ManifestKindError, match="declares no kind"):
        manifest_kind(payload, manifest_path=copy)
    assert safe_manifest_kind(payload, manifest_path=copy) == (
        "publisher_table",
        "manifest_kind_missing",
    )


def test_a_fetch_into_a_frozen_manifest_declares_its_kind(tmp_path):
    copy, _key = _frozen_copy(tmp_path)
    source = tmp_path / "24in12ms.xls"
    source.write_bytes(b"the next vintage")

    report = fetch_source_artifact(
        str(source),
        source_id="irs_soi",
        package_id="soi-table-1-2",
        year=2099,
        output_dir=copy.parent,
    )
    payload = yaml.safe_load(copy.read_text())

    assert report.valid
    assert payload["kind"] == "publisher_table"
    assert list(payload)[:3] == ["source_id", "package_id", "kind"]
    # The manifest left the freeze by declaring itself.
    assert not is_grandfathered_manifest(copy)
    assert manifest_kind(payload, manifest_path=copy) == "publisher_table"


def _kindless_package(tmp_path: Path) -> Path:
    package = tmp_path / "db" / "data" / "dwp" / "new_tables"
    package.mkdir(parents=True)
    (package / "table.ods").write_bytes(b"a table")
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-new-tables",
                "files": {
                    2023: {
                        "filename": "table.ods",
                        "source_url": "https://publisher.example/table.ods",
                    }
                },
            },
            sort_keys=False,
        )
    )
    return package


def test_a_new_kindless_manifest_is_refused_by_fetch_before_reading(
    tmp_path, monkeypatch
):
    package = _kindless_package(tmp_path)
    original = (package / "manifest.yaml").read_bytes()
    reads: list[str] = []
    monkeypatch.setattr(
        "chronicle.artifacts._read_artifact",
        lambda url: reads.append(url) or (b"x", "other.ods"),
    )

    for kind in (None, "publisher_table", "microdata_release"):
        with pytest.raises(ManifestAccessError, match="declares no kind"):
            fetch_source_artifact(
                "https://publisher.example/other.ods",
                source_id="dwp",
                package_id="dwp-new-tables",
                year=2024,
                output_dir=package,
                kind=kind,
                licence="OGL-UK-3.0",
            )

    assert reads == []
    assert (package / "manifest.yaml").read_bytes() == original


def test_a_new_kindless_manifest_is_reported_and_skipped_by_the_sweeps(
    tmp_path, monkeypatch
):
    package = _kindless_package(tmp_path)
    original = (package / "manifest.yaml").read_bytes()
    monkeypatch.setattr(
        "chronicle.artifacts._upload_r2_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no upload")),
    )

    inventory = inventory_source_artifacts(tmp_path / "db" / "data")
    published = publish_source_artifacts(tmp_path / "db" / "data")

    assert not inventory.valid
    assert any(error.startswith("manifest_kind_missing") for error in inventory.errors)
    assert not published.valid
    assert published.entries == ()
    assert any(error.startswith("manifest_kind_missing") for error in published.errors)
    assert (package / "manifest.yaml").read_bytes() == original


def test_a_new_kindless_manifest_is_refused_by_the_byte_reader(tmp_path, monkeypatch):
    package_name = f"chronicle_kind_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "tables"
    resource_dir.mkdir(parents=True)
    (resource_dir / "table.csv").write_bytes(b"a,b\n1,2\n")
    (resource_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-tables",
                "files": {2023: {"filename": "table.csv", "source_url": "x"}},
            }
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Tables",
        resource_package=package_name,
        resource_directory="data/dwp/tables",
        manifest="manifest.yaml",
        vintage="2023",
        extracted_at="2026-09-02",
        extraction_method="none",
        parser="delimited_text_full_rows",
        artifact_year=2023,
    )

    with pytest.raises(ManifestKindError):
        spec.assert_parseable_manifest()
    with pytest.raises(ManifestKindError):
        spec.build_source_rows(2023)

    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "source_package.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "ledger.source_package.v1",
                "package_id": "dwp-tables-parse-attempt",
                "label": "A kindless manifest",
                "artifact": {
                    "source_name": "dwp",
                    "source_table": "Tables",
                    "resource_package": package_name,
                    "resource_directory": "data/dwp/tables",
                    "manifest": "manifest.yaml",
                    "vintage": "2023",
                    "extracted_at": "2026-09-02",
                    "extraction_method": "none",
                    "parser": "delimited_text_full_rows",
                    "artifact_year": 2023,
                },
                "record_sets": [],
            },
            sort_keys=False,
        )
    )
    report = validate_source_package(package_dir, year=2023)
    assert not report.valid
    assert [issue.code for issue in report.errors] == ["manifest_kind_missing"]


def test_no_tracked_microdata_bytes():
    """The repository guard: a release package holds manifests and nothing else."""
    release_dirs = {
        path.parent
        for path in _tracked_manifests()
        if (yaml.safe_load(path.read_text()) or {}).get("kind") == "microdata_release"
    }
    assert release_dirs

    tracked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "db/data"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for directory in release_dirs:
        relative = directory.relative_to(REPO_ROOT).as_posix()
        files = sorted(
            Path(path).name
            for path in tracked
            if Path(path).parent.as_posix() == relative
        )
        assert files == ["manifest.yaml"], f"{relative} tracks microdata bytes: {files}"
        assert sorted(item.name for item in directory.iterdir()) == ["manifest.yaml"]


def test_the_committed_tree_passes_the_kind_rule():
    report = inventory_source_artifacts(REPO_ROOT / "db" / "data")
    assert not [error for error in report.errors if "manifest_kind" in error]
