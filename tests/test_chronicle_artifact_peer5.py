"""Failing-first regressions for PR #227 peer round 5.

Two residual refusal gaps: ``fetch-artifact`` and ``register-artifact`` never
ran the package-wide microdata identity sweep that ``publish-raw`` and
``inventory-artifacts`` run, and ``storage.previous_r2`` list elements were
never validated, so unreadable archived provenance was refused by registration
and silently ignored by the artifact commands.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path

import pytest
import yaml

from chronicle.artifacts import (
    SourceArtifactManifestError,
    fetch_source_artifact,
    inventory_source_artifacts,
    publish_source_artifacts,
)
from chronicle.registration import HashOnlyRegistrationError, register_hash_only_artifact
from tests.test_chronicle_microdata_registration import (
    ATTESTED,
    _record_uploads,
    _refuse_read,
)

RELEASE_BYTES = b"person_id,age\n1,45\n2,31\n"
RELEASE_SHA = hashlib.sha256(RELEASE_BYTES).hexdigest()
OTHER_SHA = "d" * 64
ARCHIVED_SHA = "e" * 64
MICRODATA_CODE = "bytes_identified_by_microdata_release"


def _locator(filename: str, sha256: str, year: int = 2023) -> dict[str, str]:
    key = f"raw/publisher/package/{year}/{sha256}/{filename}"
    return {
        "provider": "r2",
        "bucket": "ledger-raw",
        "key": key,
        "uri": f"r2://ledger-raw/{key}",
    }


def _release_manifest() -> dict[str, object]:
    entry = {
        "filename": "microdata.csv",
        "sha256": RELEASE_SHA,
        "size_bytes": len(RELEASE_BYTES),
        "source_url": "https://publisher.example/microdata.csv",
        "access": "public",
        "licence": "CC0-1.0",
        "vintage": "2023",
        "hash_source": "chronicle_fetch",
        "attested_by": "chronicle",
        "verified_at": "2026-09-05",
        "storage": {"r2": _locator("microdata.csv", RELEASE_SHA)},
        "licence_evidence": {
            "issuer": "Fixture publisher",
            "scope": "This fixture public microdata release is dedicated to CC0.",
            "url": "https://publisher.example/licence",
            "licence": "CC0-1.0",
            "sha256": RELEASE_SHA,
        },
    }
    return {
        "source_id": "publisher",
        "package_id": "package",
        "kind": "microdata_release",
        "files": {2023: [entry]},
    }


def _aliasing_entry(alias: str) -> dict[str, object]:
    """A public publisher-table entry carrying the release's identity."""
    entry: dict[str, object] = {
        "filename": "other-table.csv",
        "source_url": "https://publisher.example/2022.csv",
    }
    if alias == "filename":
        entry["filename"] = "microdata.csv"
    elif alias == "sha256":
        entry["sha256"] = RELEASE_SHA
    elif alias == "archived-filename":
        entry["sha256"] = OTHER_SHA
        entry["storage"] = {
            "r2": _locator("other-table.csv", OTHER_SHA, year=2022),
            "previous_r2": [_locator("microdata.csv", ARCHIVED_SHA, year=2022)],
        }
    else:
        assert alias == "archived-sha256", alias
        entry["sha256"] = OTHER_SHA
        entry["storage"] = {
            "r2": _locator("other-table.csv", OTHER_SHA, year=2022),
            "previous_r2": [_locator("old-table.csv", RELEASE_SHA, year=2022)],
        }
    return entry


def _package_with_aliasing_table(
    package: Path, *, alias: str, where: str
) -> dict[str, object]:
    """A package whose *other* vintage aliases a public microdata release.

    The vintage a fetch or a registration would touch is untouched by the
    collision: only the package-wide sweep sees it.
    """
    package.mkdir(parents=True)
    selected = {
        "source_id": "publisher",
        "package_id": "package",
        "kind": "publisher_table",
        "files": {
            2024: {
                "filename": "table.csv",
                "source_url": "https://publisher.example/table.csv",
            }
        },
    }
    aliasing = _aliasing_entry(alias)
    if where == "same-manifest":
        selected["files"][2022] = aliasing
    else:
        assert where == "sibling-manifest", where
        (package / "manifest_other.yaml").write_text(
            yaml.safe_dump(
                {
                    "source_id": "publisher",
                    "package_id": "package",
                    "kind": "publisher_table",
                    "files": {2022: aliasing},
                }
            )
        )
    (package / "manifest_tables.yaml").write_text(yaml.safe_dump(selected))
    (package / "manifest_release.yaml").write_text(yaml.safe_dump(_release_manifest()))
    return {
        "source_id": "publisher",
        "package_id": "package",
        "year": 2024,
        "output_dir": package,
        "filename": "table.csv",
        "manifest_filename": "manifest_tables.yaml",
    }


ALIASES = ["filename", "sha256", "archived-filename", "archived-sha256"]


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("where", ["same-manifest", "sibling-manifest"])
@pytest.mark.parametrize("upload", [False, True])
def test_fetch_refuses_a_package_microdata_alias_before_publisher_io(
    tmp_path, monkeypatch, alias, where, upload
):
    """fetch-artifact runs the sweep publish-raw and inventory already run."""
    package = tmp_path / "package"
    kwargs = _package_with_aliasing_table(package, alias=alias, where=where)
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    reads = _refuse_read(monkeypatch)
    uploads = _record_uploads(monkeypatch)
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(SourceArtifactManifestError, match=MICRODATA_CODE):
        fetch_source_artifact(
            "https://publisher.example/table.csv", upload_r2=upload, **kwargs
        )

    assert {path.name: path.read_bytes() for path in package.iterdir()} == before
    assert reads == []
    assert uploads == []
    assert locks == []


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("where", ["same-manifest", "sibling-manifest"])
def test_inventory_reports_the_same_package_alias_fetch_refuses(
    tmp_path, alias, where
):
    """The fetch refusal uses inventory-artifacts' vocabulary, on one tree."""
    package = tmp_path / "package"
    _package_with_aliasing_table(package, alias=alias, where=where)

    report = inventory_source_artifacts(package)

    assert any(MICRODATA_CODE in error for error in report.errors), report.errors


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("where", ["same-manifest", "sibling-manifest"])
def test_registration_refuses_a_package_microdata_alias(
    tmp_path, monkeypatch, alias, where
):
    """register-artifact stops at owner agreement without the package sweep."""
    package = tmp_path / "package"
    _package_with_aliasing_table(package, alias=alias, where=where)
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.registration._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(HashOnlyRegistrationError, match=MICRODATA_CODE):
        register_hash_only_artifact(
            source_id="publisher",
            package_id="package",
            year=2025,
            output_dir=package,
            manifest_filename="manifest_release.yaml",
            filename="adult.tab",
            sha256="f" * 64,
            licence="UK Data Service End User Licence",
            access="licensed",
            vintage="2025",
            size_bytes=1024,
            doi="10.5255/UKDA-SN-9367-2",
            **ATTESTED,
        )

    assert {path.name: path.read_bytes() for path in package.iterdir()} == before
    assert locks == []


# ---------------------------------------------------------------------------
# Unreadable archived provenance
# ---------------------------------------------------------------------------

TABLE_BYTES = b"publisher,value\nexample,123\n"
TABLE_SHA = hashlib.sha256(TABLE_BYTES).hexdigest()
TABLE_KEY = f"raw/publisher/package/2024/{TABLE_SHA}/table.csv"
ARCHIVED_KEY = f"raw/publisher/package/2024/{ARCHIVED_SHA}/archived.csv"
VALID_ARCHIVED = {
    "provider": "r2",
    "bucket": "archive",
    "key": ARCHIVED_KEY,
    "uri": f"r2://archive/{ARCHIVED_KEY}",
}
UNREADABLE_ARCHIVED = {
    # The peer's case: a bare locator string is not a mapping, so
    # _recorded_object_identities skips it and every archived-alias check goes
    # blind on an identity the key plainly carries.
    "bare-string": f"r2://archive/{ARCHIVED_KEY}",
    "unparseable-uri": {"provider": "r2", "uri": "not-an-r2-locator"},
    "contradictory": {
        "provider": "r2",
        "bucket": "different",
        "key": ARCHIVED_KEY,
        "uri": f"r2://archive/{ARCHIVED_KEY}",
    },
    "no-provider": {"uri": f"r2://archive/{ARCHIVED_KEY}"},
    "not-content-addressed": {
        "provider": "r2",
        "uri": "r2://archive/raw/publisher/package/2024/archived.csv",
    },
    "empty": {},
}
ELEMENTS = sorted(UNREADABLE_ARCHIVED)


def _package_with_history(package: Path, *, element: object) -> Path:
    """A published publisher table whose archived object is the argument."""
    package.mkdir(parents=True)
    (package / "table.csv").write_bytes(TABLE_BYTES)
    manifest_path = package / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "kind": "publisher_table",
                "source_id": "publisher",
                "package_id": "package",
                "files": {
                    2024: {
                        "filename": "table.csv",
                        "source_url": "https://publisher.example/table.csv",
                        "sha256": TABLE_SHA,
                        "size_bytes": len(TABLE_BYTES),
                        "storage": {
                            "r2": {
                                "provider": "r2",
                                "bucket": "archive",
                                "key": TABLE_KEY,
                                "uri": f"r2://archive/{TABLE_KEY}",
                            },
                            "previous_r2": [element],
                        },
                    }
                },
            },
            sort_keys=False,
        )
    )
    return manifest_path


@pytest.mark.parametrize("element", ELEMENTS)
def test_inventory_refuses_unreadable_archived_provenance(
    tmp_path, monkeypatch, element
):
    """An archived block Chronicle cannot read is not a valid registration."""
    package = tmp_path / "package"
    manifest_path = _package_with_history(package, element=UNREADABLE_ARCHIVED[element])
    before = manifest_path.read_text()

    def unexpected_read(*args, **kwargs):
        pytest.fail("inventory read bytes before refusing unreadable history")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)

    report = inventory_source_artifacts(package)

    assert not report.valid
    assert any(
        "previous_r2" in error
        for error in (*report.errors, *report.entries[0].errors)
    ), (report.errors, report.entries[0].errors)
    assert manifest_path.read_text() == before


@pytest.mark.parametrize("element", ELEMENTS)
def test_publish_refuses_unreadable_archived_provenance(tmp_path, monkeypatch, element):
    package = tmp_path / "package"
    manifest_path = _package_with_history(package, element=UNREADABLE_ARCHIVED[element])
    before = manifest_path.read_text()
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(package)

    assert not report.valid
    assert any(
        "previous_r2" in error
        for error in (*report.errors, *(e for entry in report.entries for e in entry.errors))
    ), report
    assert uploads == []
    assert manifest_path.read_text() == before


@pytest.mark.parametrize("element", ELEMENTS)
def test_fetch_refuses_unreadable_archived_provenance(tmp_path, monkeypatch, element):
    package = tmp_path / "package"
    manifest_path = _package_with_history(package, element=UNREADABLE_ARCHIVED[element])
    before = manifest_path.read_text()
    reads = _refuse_read(monkeypatch)
    uploads = _record_uploads(monkeypatch)
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(SourceArtifactManifestError, match="previous_r2"):
        fetch_source_artifact(
            "https://publisher.example/table.csv",
            source_id="publisher",
            package_id="package",
            year=2024,
            output_dir=package,
            filename="table.csv",
        )

    assert manifest_path.read_text() == before
    assert reads == []
    assert uploads == []
    assert locks == []


@pytest.mark.parametrize("element", ELEMENTS)
def test_registration_refuses_unreadable_archived_provenance(
    tmp_path, monkeypatch, element
):
    """The command that already refused it: the other three now agree."""
    package = tmp_path / "package"
    manifest_path = _package_with_history(package, element=UNREADABLE_ARCHIVED[element])
    before = manifest_path.read_text()
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.registration._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(HashOnlyRegistrationError, match="previous_r2"):
        register_hash_only_artifact(
            source_id="publisher",
            package_id="package",
            year=2025,
            output_dir=package,
            manifest_filename="manifest_release.yaml",
            filename="adult.tab",
            sha256="f" * 64,
            licence="UK Data Service End User Licence",
            access="licensed",
            vintage="2025",
            size_bytes=1024,
            doi="10.5255/UKDA-SN-9367-2",
            **ATTESTED,
        )

    assert manifest_path.read_text() == before
    assert locks == []


def test_unreadable_archived_provenance_cannot_hide_a_release_identity(
    tmp_path, monkeypatch
):
    """The blinding case: the bare string carries a release's exact identity."""
    package = tmp_path / "package"
    package.mkdir()
    (package / "table.csv").write_bytes(TABLE_BYTES)
    hidden_key = f"raw/publisher/package/2023/{RELEASE_SHA}/microdata.csv"
    manifest_path = package / "manifest_tables.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "kind": "publisher_table",
                "source_id": "publisher",
                "package_id": "package",
                "files": {
                    2024: {
                        "filename": "table.csv",
                        "source_url": "https://publisher.example/table.csv",
                        "sha256": TABLE_SHA,
                        "size_bytes": len(TABLE_BYTES),
                        "storage": {"previous_r2": [f"r2://ledger-raw/{hidden_key}"]},
                    }
                },
            },
            sort_keys=False,
        )
    )
    (package / "manifest_release.yaml").write_text(yaml.safe_dump(_release_manifest()))
    before = manifest_path.read_text()
    _record_uploads(monkeypatch)

    report = inventory_source_artifacts(package)

    assert not report.valid
    assert any(
        "previous_r2" in error or MICRODATA_CODE in error
        for error in (*report.errors, *(e for entry in report.entries for e in entry.errors))
    ), report
    assert manifest_path.read_text() == before


def test_valid_archived_provenance_is_still_read_and_kept(tmp_path, monkeypatch):
    """The control: a well-formed archived object stays a valid registration."""
    package = tmp_path / "package"
    manifest_path = _package_with_history(package, element=VALID_ARCHIVED)
    before = manifest_path.read_text()
    _record_uploads(monkeypatch)

    report = inventory_source_artifacts(package)

    assert report.valid, (report.errors, report.entries[0].errors)
    assert manifest_path.read_text() == before
    from chronicle.artifacts import _recorded_identity_aliases

    entry = yaml.safe_load(manifest_path.read_text())["files"][2024]
    names, digests = _recorded_identity_aliases(entry)
    assert "archived.csv" in names and ARCHIVED_SHA in digests


@pytest.mark.parametrize("recorded", ["current", "archived"])
def test_an_identityless_release_manifest_refuses_rather_than_crashes(
    tmp_path, recorded
):
    """Binding a locator to a missing identity is a refusal, not a crash.

    ``_clean_key_part`` was reached with the manifest's absent ``source_id``
    and raised ``AttributeError``, which no ``SourceArtifactManifestError``
    handler catches. Validating archived objects reaches the same binding, so
    the failure has to be the controlled one.
    """
    package = tmp_path / "package"
    package.mkdir()
    key = f"raw/pub/pack/2024/{TABLE_SHA}/table.csv"
    block = {
        "provider": "r2",
        "bucket": "ledger-raw",
        "key": key,
        "uri": f"r2://ledger-raw/{key}",
    }
    entry = {
        "filename": "table.csv",
        "sha256": TABLE_SHA,
        "size_bytes": len(TABLE_BYTES),
        "source_url": "https://publisher.example/table.csv",
        "access": "public",
        "licence": "CC0-1.0",
        "vintage": "2024",
        "hash_source": "chronicle_fetch",
        "attested_by": "chronicle",
        "verified_at": "2026-09-05",
        "storage": {"r2": block} if recorded == "current" else {"previous_r2": [block]},
    }
    (package / "manifest.yaml").write_text(
        yaml.safe_dump({"kind": "microdata_release", "files": {2024: [entry]}})
    )

    report = inventory_source_artifacts(package)

    assert not report.valid
    assert any(
        "recorded_r2" in error for error in report.entries[0].errors
    ), report.entries[0].errors
