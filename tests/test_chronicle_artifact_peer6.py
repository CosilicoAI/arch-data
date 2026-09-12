"""Failing-first regressions for PR #227 peer round 6.

The fetch preflight validated recorded R2 locators only for the entry it
selected and for entries naming the same package-local file. A malformed
``storage.r2`` or ``storage.previous_r2`` element anywhere else in the package
-- another vintage of the selected manifest, or a sibling manifest -- reached
``_upsert_manifest``, which refuses it only after the registration lock is
held and after the publisher has been read. ``fetch-artifact`` documents that
"Every refusal happens before the publisher is read"; registration already
validates every entry's locator before it takes its lock.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
from pathlib import Path
import re

import pytest
import yaml

from chronicle.artifacts import (
    SourceArtifactManifestError,
    fetch_source_artifact,
    inventory_source_artifacts,
)
from chronicle.registration import (
    HashOnlyRegistrationError,
    register_hash_only_artifact,
)
from tests.test_chronicle_microdata_registration import (
    ATTESTED,
    _record_uploads,
    _refuse_read,
    _serve,
)

TABLE_BYTES = b"publisher,value\nexample,123\n"
TABLE_SHA = hashlib.sha256(TABLE_BYTES).hexdigest()
OTHER_SHA = "d" * 64
ARCHIVED_SHA = "e" * 64
RELEASE_BYTES = b"person_id,age\n1,45\n2,31\n"
RELEASE_SHA = hashlib.sha256(RELEASE_BYTES).hexdigest()

TABLE_KEY = f"raw/publisher/package/2024/{TABLE_SHA}/table.csv"
OTHER_KEY = f"raw/publisher/package/2022/{OTHER_SHA}/other-table.csv"
ARCHIVED_KEY = f"raw/publisher/package/2022/{ARCHIVED_SHA}/archived.csv"
RELEASE_KEY = f"raw/publisher/package/2023/{RELEASE_SHA}/microdata.csv"
RELEASE_ARCHIVED_KEY = f"raw/publisher/package/2023/{ARCHIVED_SHA}/old-microdata.csv"


def _locator(bucket: str, key: str) -> dict[str, str]:
    return {
        "provider": "r2",
        "bucket": bucket,
        "key": key,
        "uri": f"r2://{bucket}/{key}",
    }


# Every shape ``_validated_recorded_r2`` refuses, in the vocabulary
# inventory-artifacts already reports for them.
UNREADABLE = {
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
        "uri": "r2://archive/raw/publisher/package/2022/archived.csv",
    },
    "empty": {},
}
ELEMENTS = sorted(UNREADABLE)
PLACEMENTS = ["other-vintage", "sibling-manifest"]
POSITIONS = ["storage.r2", "previous_r2"]


def _selected_entry() -> dict[str, object]:
    return {
        "filename": "table.csv",
        "source_url": "https://publisher.example/table.csv",
        "sha256": TABLE_SHA,
        "size_bytes": len(TABLE_BYTES),
        "storage": {"r2": _locator("archive", TABLE_KEY)},
    }


def _unselected_entry(*, position: str, element: object) -> dict[str, object]:
    """An entry the fetch never selects: a different vintage and filename.

    A different filename keeps it out of ``_manifest_file_owners``, which is
    the only preflight path that validated a non-selected entry's locator.
    """
    entry: dict[str, object] = {
        "filename": "other-table.csv",
        "source_url": "https://publisher.example/2022.csv",
        "sha256": OTHER_SHA,
        "size_bytes": 12,
    }
    if position == "previous_r2":
        entry["storage"] = {
            "r2": _locator("archive", OTHER_KEY),
            "previous_r2": [element],
        }
    else:
        assert position == "storage.r2", position
        entry["storage"] = {"r2": element}
    return entry


def _table_manifest(files: dict[object, object]) -> dict[str, object]:
    return {
        "kind": "publisher_table",
        "source_id": "publisher",
        "package_id": "package",
        "files": files,
    }


def _build_package(
    package: Path, *, where: str, position: str, element: object
) -> dict[str, object]:
    """A package whose *unselected* provenance is the one Chronicle cannot read."""
    package.mkdir(parents=True)
    (package / "table.csv").write_bytes(TABLE_BYTES)
    unselected = _unselected_entry(position=position, element=element)
    files: dict[object, object] = {2024: _selected_entry()}
    if where == "other-vintage":
        files[2022] = unselected
    else:
        assert where == "sibling-manifest", where
        (package / "manifest_other.yaml").write_text(
            yaml.safe_dump(_table_manifest({2022: unselected}), sort_keys=False)
        )
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(_table_manifest(files), sort_keys=False)
    )
    return {
        "source_id": "publisher",
        "package_id": "package",
        "year": 2024,
        "output_dir": package,
        "filename": "table.csv",
    }


def _snapshot(package: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(package.iterdir())}


def _expected_fragment(position: str) -> str:
    return "storage.r2" if position == "storage.r2" else "storage.previous_r2[0]"


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("where", PLACEMENTS)
def test_fetch_refuses_an_unreadable_locator_outside_the_selected_vintage(
    tmp_path, monkeypatch, where, position, element
):
    """The preflight validates every entry's locators, not just the selected one."""
    package = tmp_path / "package"
    kwargs = _build_package(
        package, where=where, position=position, element=UNREADABLE[element]
    )
    before = _snapshot(package)
    reads = _refuse_read(monkeypatch)
    uploads = _record_uploads(monkeypatch)
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(
        SourceArtifactManifestError, match=re.escape(_expected_fragment(position))
    ):
        fetch_source_artifact("https://publisher.example/table.csv", **kwargs)

    assert _snapshot(package) == before
    assert reads == []
    assert uploads == []
    assert locks == []


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("where", PLACEMENTS)
def test_inventory_reports_the_locator_fetch_refuses(
    tmp_path, where, position, element
):
    """The control: one tree, and inventory-artifacts already names it."""
    package = tmp_path / "package"
    _build_package(package, where=where, position=position, element=UNREADABLE[element])

    report = inventory_source_artifacts(package)

    assert not report.valid
    reported = [
        error
        for error in (
            *report.errors,
            *(message for entry in report.entries for message in entry.errors),
        )
        if "storage.r2" in error or "previous_r2" in error
    ]
    assert reported, (report.errors, [entry.errors for entry in report.entries])


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("where", PLACEMENTS)
def test_registration_already_refuses_the_same_locator(
    tmp_path, monkeypatch, where, position, element
):
    """The parity control: registration validated every entry all along."""
    package = tmp_path / "package"
    _build_package(package, where=where, position=position, element=UNREADABLE[element])
    before = _snapshot(package)
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.registration._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(
        HashOnlyRegistrationError, match=re.escape(_expected_fragment(position))
    ):
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

    assert _snapshot(package) == before
    assert locks == []


def _release_manifest(*, previous: object | None) -> dict[str, object]:
    """A valid public microdata release, optionally carrying archived history."""
    storage: dict[str, object] = {"r2": _locator("ledger-raw", RELEASE_KEY)}
    if previous is not None:
        storage["previous_r2"] = [previous]
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
        "storage": storage,
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


def _build_release_sibling_package(
    package: Path, *, previous: object | None
) -> dict[str, object]:
    package.mkdir(parents=True)
    (package / "table.csv").write_bytes(TABLE_BYTES)
    (package / "manifest_tables.yaml").write_text(
        yaml.safe_dump(_table_manifest({2024: _selected_entry()}), sort_keys=False)
    )
    (package / "manifest_release.yaml").write_text(
        yaml.safe_dump(_release_manifest(previous=previous), sort_keys=False)
    )
    return {
        "source_id": "publisher",
        "package_id": "package",
        "year": 2024,
        "output_dir": package,
        "filename": "table.csv",
        "manifest_filename": "manifest_tables.yaml",
    }


@pytest.mark.parametrize("element", ELEMENTS)
def test_fetch_refuses_an_unreadable_locator_in_a_release_sibling(
    tmp_path, monkeypatch, element
):
    """A release sibling's archived provenance binds the registration identity."""
    package = tmp_path / "package"
    kwargs = _build_release_sibling_package(package, previous=UNREADABLE[element])
    before = _snapshot(package)
    reads = _refuse_read(monkeypatch)
    uploads = _record_uploads(monkeypatch)
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(
        SourceArtifactManifestError, match=re.escape("storage.previous_r2[0]")
    ):
        fetch_source_artifact("https://publisher.example/table.csv", **kwargs)

    assert _snapshot(package) == before
    assert reads == []
    assert uploads == []
    assert locks == []


def test_a_release_sibling_bound_to_another_registration_is_refused(
    tmp_path, monkeypatch
):
    """``bind_registration_identity`` still applies to the hoisted validation."""
    package = tmp_path / "package"
    foreign = "raw/other/package/2023/" + ARCHIVED_SHA + "/old-microdata.csv"
    kwargs = _build_release_sibling_package(
        package, previous=_locator("ledger-raw", foreign)
    )
    before = _snapshot(package)
    reads = _refuse_read(monkeypatch)
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(SourceArtifactManifestError, match="complete registration"):
        fetch_source_artifact("https://publisher.example/table.csv", **kwargs)

    assert _snapshot(package) == before
    assert reads == []
    assert locks == []


@pytest.mark.parametrize("where", PLACEMENTS)
def test_readable_provenance_outside_the_selected_vintage_still_fetches(
    tmp_path, monkeypatch, where
):
    """The no-over-refusal control: valid locators elsewhere change nothing."""
    package = tmp_path / "package"
    kwargs = _build_package(
        package,
        where=where,
        position="previous_r2",
        element=_locator("archive", ARCHIVED_KEY),
    )
    reads = _serve(monkeypatch, TABLE_BYTES)
    _record_uploads(monkeypatch)

    report = fetch_source_artifact("https://publisher.example/table.csv", **kwargs)

    assert reads == ["https://publisher.example/table.csv"]
    assert report.sha256 == TABLE_SHA
    assert (package / "table.csv").read_bytes() == TABLE_BYTES
    unselected = yaml.safe_load((package / "manifest.yaml").read_text())["files"]
    if where == "other-vintage":
        assert unselected[2022]["storage"]["previous_r2"] == [
            _locator("archive", ARCHIVED_KEY)
        ]


def test_a_release_sibling_with_readable_history_still_fetches(tmp_path, monkeypatch):
    """The release-side control: well-formed archived history is kept."""
    package = tmp_path / "package"
    kwargs = _build_release_sibling_package(
        package, previous=_locator("ledger-raw", RELEASE_ARCHIVED_KEY)
    )
    reads = _serve(monkeypatch, TABLE_BYTES)
    _record_uploads(monkeypatch)

    report = fetch_source_artifact("https://publisher.example/table.csv", **kwargs)

    assert reads == ["https://publisher.example/table.csv"]
    assert report.sha256 == TABLE_SHA
    release = yaml.safe_load((package / "manifest_release.yaml").read_text())
    assert release["files"][2023][0]["storage"]["previous_r2"] == [
        _locator("ledger-raw", RELEASE_ARCHIVED_KEY)
    ]


@pytest.mark.parametrize("where", PLACEMENTS)
def test_the_under_lock_loop_still_rechecks_a_locator_written_during_the_read(
    tmp_path, monkeypatch, where
):
    """The hoisted pass is an addition: ``_upsert_manifest`` remains the recheck.

    The publisher read is the widest window in which another writer can land a
    manifest the preflight already accepted. Nothing may be written or
    uploaded on the strength of the first two passes alone.
    """
    package = tmp_path / "package"
    kwargs = _build_package(
        package,
        where=where,
        position="previous_r2",
        element=_locator("archive", ARCHIVED_KEY),
    )
    target = (
        package / "manifest.yaml"
        if where == "other-vintage"
        else package / "manifest_other.yaml"
    )
    uploads = _record_uploads(monkeypatch)
    reads: list[str] = []

    def corrupt_then_serve(source_url):
        reads.append(source_url)
        payload = yaml.safe_load(target.read_text())
        payload["files"][2022]["storage"]["previous_r2"] = [
            UNREADABLE["unparseable-uri"]
        ]
        target.write_text(yaml.safe_dump(payload, sort_keys=False))
        return TABLE_BYTES, "table.csv"

    monkeypatch.setattr("chronicle.artifacts._read_artifact", corrupt_then_serve)

    locks: list[Path] = []

    @contextmanager
    def recording_lock(path):
        locks.append(path)
        yield

    monkeypatch.setattr("chronicle.artifacts._registration_lock", recording_lock)

    with pytest.raises(
        SourceArtifactManifestError, match=re.escape("storage.previous_r2[0]")
    ):
        fetch_source_artifact("https://publisher.example/table.csv", **kwargs)

    assert reads == ["https://publisher.example/table.csv"]
    assert locks == [package]
    assert uploads == []
    # The proposal is validated before any manifest is rewritten, so the entry
    # this fetch would have recorded never gained its fetch metadata.
    selected = yaml.safe_load((package / "manifest.yaml").read_text())["files"][2024]
    assert "fetched_at" not in selected, selected


# ---------------------------------------------------------------------------
# The selected entry: the one place the existing tree differs from the proposed
# ---------------------------------------------------------------------------

MISMATCHED = {
    # The recorded object's key says one thing and the entry says another.
    "checksum": {"sha256": OTHER_SHA, "key_sha256": TABLE_SHA, "key_name": "table.csv"},
    "filename": {
        "sha256": TABLE_SHA,
        "key_sha256": TABLE_SHA,
        "key_name": "renamed.csv",
    },
}


def _package_with_a_mismatched_selected_entry(package: Path, *, defect: str) -> None:
    package.mkdir(parents=True)
    (package / "table.csv").write_bytes(TABLE_BYTES)
    spec = MISMATCHED[defect]
    key = f"raw/publisher/package/2024/{spec['key_sha256']}/{spec['key_name']}"
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            _table_manifest(
                {
                    2024: {
                        "filename": "table.csv",
                        "source_url": "https://publisher.example/table.csv",
                        "sha256": spec["sha256"],
                        "size_bytes": len(TABLE_BYTES),
                        "storage": {"r2": _locator("archive", key)},
                    }
                }
            ),
            sort_keys=False,
        )
    )


@pytest.mark.parametrize("defect", sorted(MISMATCHED))
def test_inventory_calls_a_mismatched_selected_entry_invalid(tmp_path, defect):
    """The control that makes the refusal below the documented contract.

    ``fetch-artifact`` will not carry an invalid registration forward, and
    ``inventory-artifacts`` reports the same codes.
    """
    package = tmp_path / "package"
    _package_with_a_mismatched_selected_entry(package, defect=defect)

    report = inventory_source_artifacts(package)

    assert not report.valid
    assert any(
        "recorded_r2_identity_mismatch" in error
        for entry in report.entries
        for error in entry.errors
    ), [entry.errors for entry in report.entries]


@pytest.mark.parametrize("record_revision", [False, True])
@pytest.mark.parametrize("defect", sorted(MISMATCHED))
def test_fetch_refuses_a_selected_entry_that_contradicts_its_own_locator(
    tmp_path, monkeypatch, defect, record_revision
):
    """The preflight reads the recorded tree, so the entry it rewrites is checked too.

    The fetch used to accept this directory and repair the contradiction in
    the rewrite, including under ``--record-revision``, which archived a
    superseded object the entry never agreed with. Validating every entry
    before the publisher is read refuses it instead, in the vocabulary
    ``inventory-artifacts`` already reports for the same tree.
    """
    package = tmp_path / "package"
    _package_with_a_mismatched_selected_entry(package, defect=defect)
    before = _snapshot(package)
    reads = _refuse_read(monkeypatch)
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(
        SourceArtifactManifestError, match="recorded_r2_identity_mismatch"
    ):
        fetch_source_artifact(
            "https://publisher.example/table.csv",
            source_id="publisher",
            package_id="package",
            year=2024,
            output_dir=package,
            filename="table.csv",
            record_revision=record_revision,
            expected_sha256=TABLE_SHA if record_revision else None,
        )

    assert _snapshot(package) == before
    assert reads == []
    assert locks == []


def test_provenance_the_sweep_cannot_read_is_still_refused_before_the_read(
    tmp_path, monkeypatch
):
    """Why the hoisted loop follows the package sweep rather than preceding it.

    The sweep resolves locators non-raisingly, so what it finds on unreadable
    provenance is true and its message is the one the caller needs -- round 5
    pins that. What it *misses* is this: an archived element that is not a
    mapping is skipped, so a table entry whose only carrier of the release's
    identity is that element passes the sweep in silence. The locator loop
    that runs next refuses it anyway, still before the publisher is read.
    """
    package = tmp_path / "package"
    package.mkdir(parents=True)
    hidden = f"raw/publisher/package/2023/{RELEASE_SHA}/microdata.csv"
    aliasing = {
        "filename": "other-table.csv",
        "sha256": OTHER_SHA,
        "source_url": "https://publisher.example/2022.csv",
        "storage": {
            "r2": _locator("archive", OTHER_KEY),
            # Not a mapping: _recorded_object_identities skips it, so the
            # release identity its key carries is invisible to the sweep.
            "previous_r2": [f"r2://ledger-raw/{hidden}"],
        },
    }
    (package / "manifest_tables.yaml").write_text(
        yaml.safe_dump(
            _table_manifest(
                {
                    2022: aliasing,
                    2024: {
                        "filename": "table.csv",
                        "source_url": "https://publisher.example/table.csv",
                    },
                }
            ),
            sort_keys=False,
        )
    )
    (package / "manifest_release.yaml").write_text(
        yaml.safe_dump(_release_manifest(previous=None), sort_keys=False)
    )
    before = _snapshot(package)
    reads = _refuse_read(monkeypatch)
    uploads = _record_uploads(monkeypatch)
    locks: list[Path] = []
    monkeypatch.setattr(
        "chronicle.artifacts._registration_lock",
        lambda path: locks.append(path) or nullcontext(),
    )

    with pytest.raises(SourceArtifactManifestError) as refusal:
        fetch_source_artifact(
            "https://publisher.example/table.csv",
            source_id="publisher",
            package_id="package",
            year=2024,
            output_dir=package,
            filename="table.csv",
            manifest_filename="manifest_tables.yaml",
        )

    # The sweep stayed silent on it: this is the locator refusal, not the
    # package identity one.
    assert "storage.previous_r2[0]" in str(refusal.value)
    assert "bytes_identified_by_microdata_release" not in str(refusal.value)
    assert _snapshot(package) == before
    assert reads == []
    assert uploads == []
    assert locks == []
