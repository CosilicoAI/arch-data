"""Tests for microdata-release registration: identity without content.

Chronicle registers every raw microdata release its consumers build from and
stores the bytes of only the ones a publisher permits it to redistribute
(``docs/adr-chronicle-raw-microdata-identity.md``). These tests pin the whole
refusal surface: which access classes may carry bytes, which commands refuse
them, that every refusal happens before any filesystem or network side effect,
and that a hash-only registration is a complete, valid artifact record with no
local file and no R2 key.

The catalogue and the checked-in synthetic consumer manifests are covered in
``tests/test_chronicle_microdata_catalogue.py``; the explicit-kind rule and the
repository guard in ``tests/test_chronicle_manifest_kind.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import sys
import threading
import uuid
from pathlib import Path

import pytest
import yaml

from chronicle.artifacts import (
    ArtifactCommandResult,
    ExpectedArtifactIdentityError,
    MalformedManifestError,
    SourceArtifactRevisionError,
    _expected_identity,
    _upsert_manifest,
    fetch_source_artifact,
    inventory_source_artifacts,
    microdata_staging_path,
    publish_source_artifacts,
)
from chronicle.harness import main as harness_main
from chronicle.registration import (
    ACCESS_CLASSES,
    HASH_SOURCES,
    AmbiguousVintageKeyError,
    ArtifactFilenameError,
    HashOnlyRegistrationError,
    ListSpecRejected,
    ManifestAccessError,
    MicrodataReleaseNotParseableError,
    bare_filename,
    entry_access,
    filename_key,
    has_file_entries,
    is_bare_filename,
    is_hash_only,
    is_microdata_release,
    iter_file_specs,
    manifest_kind,
    normalize_access,
    register_hash_only_artifact,
    registration_id,
    resolve_vintage_key,
    safe_entry_access,
    stores_bytes,
    validate_file_entry,
    validate_manifest_files,
    vintage_key_forms,
    _registration_lock_path,
)
from chronicle.source_package import (
    SOURCE_ARTIFACT_CACHE_ENV,
    SOURCE_ARTIFACT_FETCH_ENV,
    SourceArtifactSpec,
    _read_source_artifact_content,
    validate_source_package,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FRS_PACKAGE = REPO_ROOT / "db" / "data" / "dwp" / "frs_2023_24"
SPI_PACKAGE = REPO_ROOT / "db" / "data" / "hmrc" / "spi_public_use_tape_2022_23"

# A syntactically valid checksum that identifies no real publisher bytes.
FIXTURE_SHA = "a" * 64
OTHER_SHA = "b" * 64
FIXTURE_COMMIT = "2fb2e2f8a99c37725bd6e7a15ff4c2595c912b77"

ATTESTED = {
    "hash_source": "consumer_attested",
    "attested_by": "PolicyEngine/microcosm",
    "attestation_evidence": (
        "microcosm uk/source_stages.json frs_spine pin, verified against the "
        "licensed copy"
    ),
    "verified_at": "2026-09-02",
}
PINNED = {
    "hash_source": "consumer_pin",
    "attested_by": "PolicyEngine/microcosm",
    "pinned_from": {
        "repository": "PolicyEngine/microcosm",
        "path": "packages/microcosm-build/src/microcosm/build/uk/source_stages.json",
        "commit": FIXTURE_COMMIT,
    },
}

# Evidence binding a public release to an allowlisted term. The URL is a
# test value; the check is that it is a durable http(s) location.
EVIDENCE = {
    "issuer": "U.S. Census Bureau",
    "scope": (
        "Public-use microdata file published by the U.S. Census Bureau, a "
        "federal agency; a work of the United States Government under 17 "
        "U.S.C. §105."
    ),
    "url": "https://evidence.example/census/public-use-files",
}
PUBLIC_BYTES = b"public household pums"
PUBLIC_SHA = hashlib.sha256(PUBLIC_BYTES).hexdigest()


def _register(output_dir: Path, **overrides: object) -> object:
    """Register a fixture licensed artifact, with per-test overrides."""
    kwargs: dict[str, object] = {
        "source_id": "dwp",
        "package_id": "dwp-frs-2023-24",
        "year": 2023,
        "output_dir": output_dir,
        "filename": "adult.tab",
        "sha256": FIXTURE_SHA,
        "licence": "UK Data Service End User Licence",
        "access": "licensed",
        "vintage": "2023_24",
        "size_bytes": 35323384,
        "doi": "10.5255/UKDA-SN-9367-2",
        **ATTESTED,
    }
    kwargs.update(overrides)
    return register_hash_only_artifact(**kwargs)  # type: ignore[arg-type]


def _manifest(output_dir: Path) -> dict:
    return yaml.safe_load((output_dir / "manifest.yaml").read_text())


def _refuse_read(monkeypatch, message: str = "the publisher was read") -> list:
    """Make any publisher read an ordering violation; return the read log."""
    reads: list[str] = []

    def unexpected_read(source_url):
        reads.append(source_url)
        raise AssertionError(f"ORDERING VIOLATION: {message} before the refusal")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)
    return reads


def _serve(monkeypatch, content: bytes, filename: str | None = None) -> list:
    """Serve ``content`` for any URL, recording each read."""
    reads: list[str] = []

    def fake_read(source_url):
        reads.append(source_url)
        return content, filename or Path(source_url).name

    monkeypatch.setattr("chronicle.artifacts._read_artifact", fake_read)
    return reads


def _record_uploads(monkeypatch) -> list[tuple[str, str]]:
    uploads: list[tuple[str, str]] = []

    def fake_upload(location, local_path, *, wrangler_command):
        uploads.append((location.uri, str(local_path)))
        return ArtifactCommandResult(
            command=("stub",), returncode=0, stdout="ok", stderr=""
        )

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", fake_upload)
    return uploads


def _forbid_uploads(monkeypatch) -> None:
    monkeypatch.setattr(
        "chronicle.artifacts._upload_r2_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("nothing may be uploaded on the way to a refusal")
        ),
    )


def _fetch_table(source: Path, output_dir: Path, **overrides: object):
    kwargs: dict[str, object] = {
        "source_id": "irs_soi",
        "package_id": "soi-table-1-2",
        "year": 2023,
        "output_dir": output_dir,
    }
    kwargs.update(overrides)
    return fetch_source_artifact(str(source), **kwargs)  # type: ignore[arg-type]


def _fetch_release(
    output_dir: Path,
    *,
    staging_dir: Path,
    filename: str = "csv_hus.zip",
    content: bytes = PUBLIC_BYTES,
    **overrides: object,
):
    """Fetch a public microdata release with complete evidence."""
    kwargs: dict[str, object] = {
        "source_id": "census_acs",
        "package_id": "census-acs-pums-2022-1yr",
        "year": 2022,
        "output_dir": output_dir,
        "filename": filename,
        "licence": "US-Government-Work",
        "access": "public",
        "kind": "microdata_release",
        "publisher": "U.S. Census Bureau",
        "vintage": "2022",
        "expected_sha256": hashlib.sha256(content).hexdigest(),
        "licence_evidence": EVIDENCE,
        "staging_dir": staging_dir,
    }
    kwargs.update(overrides)
    return fetch_source_artifact(
        f"https://publisher.example/pums/{filename}",
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Access, kind, filename and vintage vocabularies
# --------------------------------------------------------------------------


def test_access_classes_are_the_closed_contract_set():
    assert ACCESS_CLASSES == ("public", "licensed", "restricted")


@pytest.mark.parametrize("access", ACCESS_CLASSES)
def test_only_public_access_may_carry_bytes(access):
    assert stores_bytes(access) is (access == "public")
    assert is_hash_only(access) is (access != "public")


def test_absent_access_is_inferred_public_for_a_table_entry():
    assert normalize_access(None) == "public"
    assert entry_access({"filename": "table.xlsx"}) == "public"


def test_unknown_access_class_is_refused():
    with pytest.raises(ManifestAccessError, match="Unknown access class"):
        normalize_access("internal")


def test_unknown_access_class_falls_back_to_restricted_not_public():
    # Never upload bytes because a class failed to parse.
    assert safe_entry_access({"access": "internal"}) == "restricted"


def test_manifest_kind_is_explicit_except_for_an_absent_manifest():
    assert manifest_kind(None) == "publisher_table"
    assert manifest_kind({}) == "publisher_table"
    assert is_microdata_release({"kind": "microdata_release"}) is True
    with pytest.raises(ManifestAccessError, match="Unknown manifest kind"):
        manifest_kind({"kind": "microdata_rows"})
    # A manifest with content and no kind is an error, never a table.
    with pytest.raises(ManifestAccessError, match="declares no kind"):
        manifest_kind({"files": {2023: {"filename": "table.xlsx"}}})


def test_a_manifest_with_no_entries_has_nothing_to_classify():
    # A bare ``files:`` line or an empty mapping declares no entry that could
    # be misread as a publisher table, so the manifest reads like an absent
    # one and the command writing its first entry declares the kind.
    assert has_file_entries({"source_id": "irs_soi", "files": None}) is False
    assert has_file_entries({"source_id": "irs_soi", "files": {}}) is False
    assert has_file_entries({"files": {2023: {"filename": "t.xlsx"}}}) is True
    assert has_file_entries({"files": {2023: []}}) is False
    assert has_file_entries({"files": ["not", "a", "mapping"]}) is True
    assert manifest_kind({"source_id": "irs_soi", "files": None}) == "publisher_table"
    assert manifest_kind({"source_id": "irs_soi", "files": {}}) == "publisher_table"
    with pytest.raises(ManifestAccessError, match="declares no kind"):
        manifest_kind({"source_id": "irs_soi", "files": {2023: {"filename": "t"}}})


def test_hash_sources_are_the_closed_contract_set():
    assert HASH_SOURCES == ("chronicle_fetch", "consumer_attested", "consumer_pin")


def test_registration_identity_is_the_contract_tuple():
    assert (
        registration_id(
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            sha256=FIXTURE_SHA,
            filename="adult.tab",
        )
        == f"dwp/dwp-frs-2023-24/2023/{FIXTURE_SHA}/adult.tab"
    )


@pytest.mark.parametrize(
    "alias",
    [
        "./adult.tab",
        "sub/../adult.tab",
        "adult.tab/",
        "../pkg/adult.tab",
        "/x/adult.tab",
        ".",
        "..",
        " adult.tab",
        "",
    ],
)
def test_bare_filename_refuses_every_alias(alias):
    assert not is_bare_filename(alias)
    with pytest.raises(ArtifactFilenameError, match="bare filename"):
        bare_filename(alias)


def test_bare_filename_accepts_a_plain_name_and_keys_case_folded():
    assert bare_filename("adult.tab") == "adult.tab"
    assert filename_key("ADULT.TAB") == filename_key("adult.tab")
    assert filename_key("cafe\N{COMBINING ACUTE ACCENT}.tab") == filename_key(
        "caf\N{LATIN SMALL LETTER E WITH ACUTE}.tab"
    )
    assert filename_key("./Adult.tab") == "adult.tab"


def test_vintage_key_forms_pair_the_two_spellings_of_a_year_only():
    assert vintage_key_forms(2023) == (2023, "2023")
    assert vintage_key_forms("2023") == ("2023", 2023)
    assert vintage_key_forms("A_1") == ("A_1",)
    assert vintage_key_forms("0123") == ("0123",)
    assert resolve_vintage_key({"2023": {}}, 2023) == "2023"
    assert resolve_vintage_key({2023: {}}, "2023") == 2023
    assert resolve_vintage_key({}, 2023) is None
    with pytest.raises(AmbiguousVintageKeyError, match="both keys"):
        resolve_vintage_key({2023: {}, "2023": {}}, 2023)


# --------------------------------------------------------------------------
# Entry and manifest validation vocabulary
# --------------------------------------------------------------------------


def test_microdata_release_entry_requires_access_licence_and_attestation():
    errors = validate_file_entry(
        {"filename": "adult.tab", "sha256": FIXTURE_SHA},
        kind="microdata_release",
        manifest={},
        local_file_exists=False,
    )

    assert "missing_access" in errors
    assert "missing_licence" in errors
    assert "missing_hash_source" in errors


def _attested_entry(**mutation: object) -> dict:
    entry = {
        "filename": "adult.tab",
        "access": "licensed",
        "licence": "UK Data Service End User Licence",
        "vintage": "2023_24",
        "sha256": FIXTURE_SHA,
        "doi": "10.5255/UKDA-SN-9367-2",
        **ATTESTED,
    }
    entry.update(mutation)
    return {key: value for key, value in entry.items() if value is not None}


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"licence": None}, "missing_licence"),
        ({"sha256": None}, "missing_sha256"),
        ({"sha256": "not-a-checksum"}, "malformed_sha256"),
        ({"sha256": FIXTURE_SHA.upper()}, "malformed_sha256"),
        ({"vintage": None}, "missing_vintage"),
        ({"doi": None}, "missing_access_route"),
        ({"hash_source": None}, "missing_hash_source"),
        ({"hash_source": "transcribed"}, "unknown_hash_source:transcribed"),
        ({"attested_by": None}, "missing_attested_by"),
        ({"attestation_evidence": None}, "missing_attestation_evidence"),
        ({"verified_at": None}, "missing_verified_at"),
        ({"filename": "./adult.tab"}, "non_canonical_filename:./adult.tab"),
    ],
)
def test_hash_only_entry_reports_each_missing_field(mutation, expected_code):
    errors = validate_file_entry(
        _attested_entry(**mutation),
        kind="microdata_release",
        manifest={},
        local_file_exists=False,
    )

    assert expected_code in errors


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"filename": True}, "missing_filename"),
        ({"licence": ["UKDS"]}, "missing_licence"),
        ({"vintage": {"year": 2023}}, "missing_vintage"),
        ({"doi": None, "access_route": True}, "missing_access_route"),
        ({"attested_by": ["microcosm"]}, "missing_attested_by"),
        (
            {"attestation_evidence": {"manifest": "source_stages.json"}},
            "missing_attestation_evidence",
        ),
        ({"verified_at": True}, "missing_verified_at"),
    ],
    ids=(
        "filename",
        "licence",
        "vintage",
        "access-route",
        "attested-by",
        "evidence",
        "verified-at",
    ),
)
def test_required_text_fields_reject_non_string_yaml_values(mutation, expected_code):
    errors = validate_file_entry(
        _attested_entry(**mutation),
        kind="microdata_release",
        manifest={},
        local_file_exists=False,
    )

    assert expected_code in errors


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"pinned_from": None}, "missing_pinned_from"),
        ({"pinned_from": "microcosm@abc"}, "malformed_pinned_from"),
        (
            {"pinned_from": {"repository": "PolicyEngine/microcosm"}},
            "pinned_from_missing_field:commit",
        ),
        (
            {"pinned_from": {**PINNED["pinned_from"], "commit": "abc123"}},
            "malformed_pinned_from_commit",
        ),
        ({"verified_at": "2026-09-02"}, "verified_at_forbidden_for_consumer_pin"),
    ],
)
def test_consumer_pin_entry_reports_its_own_fields(mutation, expected_code):
    entry = _attested_entry(attestation_evidence=None, verified_at=None, **PINNED)
    entry.update(mutation)
    entry = {key: value for key, value in entry.items() if value is not None}

    errors = validate_file_entry(
        entry, kind="microdata_release", manifest={}, local_file_exists=False
    )

    assert expected_code in errors


def test_chronicle_fetch_entry_is_attested_by_chronicle_with_a_date():
    entry = _attested_entry(
        hash_source="chronicle_fetch", attested_by="microcosm", verified_at=None
    )
    errors = validate_file_entry(
        entry, kind="microdata_release", manifest={}, local_file_exists=False
    )
    assert "attested_by_not_chronicle" in errors
    assert "missing_verified_at" in errors


def test_hash_only_entry_flags_bytes_and_r2_locations():
    entry = _attested_entry(
        access="restricted",
        storage={
            "r2": {"bucket": "ledger-raw", "key": "raw/x/y/z"},
            "previous_r2": [{"uri": "r2://ledger-raw/raw/x/y/old"}],
        },
    )

    errors = validate_file_entry(
        entry,
        kind="microdata_release",
        manifest={},
        local_file_exists=True,
    )

    assert "bytes_present_for_hash_only_entry" in errors
    assert "r2_location_for_hash_only_entry" in errors
    assert "r2_history_for_hash_only_entry" in errors


def test_public_table_entry_needs_no_licence_access_route_or_attestation():
    assert (
        validate_file_entry(
            {"filename": "table.xlsx", "sha256": FIXTURE_SHA},
            kind="publisher_table",
            manifest={},
            local_file_exists=True,
        )
        == ()
    )


def _public_release_entry(**mutation: object) -> dict:
    entry = {
        "filename": "csv_hus.zip",
        "access": "public",
        "licence": "US-Government-Work",
        "licence_evidence": {
            **EVIDENCE,
            "licence": "US-Government-Work",
            "sha256": PUBLIC_SHA,
        },
        "vintage": "2022",
        "sha256": PUBLIC_SHA,
        "hash_source": "chronicle_fetch",
        "attested_by": "chronicle",
        "verified_at": "2026-09-03",
    }
    entry.update(mutation)
    return {key: value for key, value in entry.items() if value is not None}


def test_public_release_entry_is_complete_with_allowlisted_evidence():
    assert (
        validate_file_entry(
            _public_release_entry(),
            kind="microdata_release",
            manifest={},
            local_file_exists=False,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            {"licence": "U.S. Census Bureau public-use file"},
            "licence_not_redistributable:U.S. Census Bureau public-use file",
        ),
        ({"licence_evidence": None}, "missing_licence_evidence"),
        ({"licence_evidence": "see website"}, "malformed_licence_evidence"),
        (
            {
                "licence_evidence": {
                    **EVIDENCE,
                    "licence": "US-Government-Work",
                    "sha256": PUBLIC_SHA,
                    "url": "",
                }
            },
            "licence_evidence_missing_field:url",
        ),
        (
            {
                "licence_evidence": {
                    **EVIDENCE,
                    "licence": "CC0-1.0",
                    "sha256": PUBLIC_SHA,
                }
            },
            "licence_evidence_licence_mismatch",
        ),
        (
            {
                "licence_evidence": {
                    **EVIDENCE,
                    "licence": "US-Government-Work",
                    "sha256": OTHER_SHA,
                }
            },
            "licence_evidence_sha256_mismatch",
        ),
        (
            {
                "licence_evidence": {
                    **EVIDENCE,
                    "licence": "US-Government-Work",
                    "sha256": PUBLIC_SHA,
                    "url": "see the website",
                }
            },
            "licence_evidence_url_not_durable",
        ),
        ({"vintage": None}, "missing_vintage"),
        ({"sha256": None}, "missing_sha256"),
        ({"hash_source": None}, "missing_hash_source"),
    ],
)
def test_public_release_entry_reports_missing_evidence(mutation, expected_code):
    errors = validate_file_entry(
        _public_release_entry(**mutation),
        kind="microdata_release",
        manifest={},
        local_file_exists=False,
    )
    assert expected_code in errors


def test_public_release_bytes_beside_the_manifest_are_a_repository_violation():
    errors = validate_file_entry(
        _public_release_entry(),
        kind="microdata_release",
        manifest={},
        local_file_exists=True,
    )
    assert "bytes_present_for_microdata_release_entry" in errors


def test_list_file_spec_expands_only_for_a_microdata_release():
    specs = [{"filename": "adult.tab"}, {"filename": "child.tab"}]

    assert iter_file_specs(specs, kind="microdata_release") == tuple(specs)

    rejected = iter_file_specs(specs, kind="publisher_table")
    assert len(rejected) == 1
    assert isinstance(rejected[0], ListSpecRejected)
    assert validate_file_entry(
        rejected[0],
        kind="publisher_table",
        manifest={},
        local_file_exists=False,
    ) == ("list_file_spec_requires_microdata_release_kind",)


def test_manifest_level_validation_reports_keys_names_and_collisions():
    manifest = {
        "files": {
            2023: [
                {"filename": "adult.tab", "access": "licensed", "sha256": FIXTURE_SHA},
                {"filename": "./adult.tab", "access": "public", "sha256": OTHER_SHA},
                {"filename": "child.tab", "sha256": FIXTURE_SHA},
                {"filename": "child.tab", "sha256": FIXTURE_SHA},
            ],
            "2023": {"filename": "other.csv"},
            2022: {"filename": "table.xlsx", "sha256": FIXTURE_SHA},
            2021: {"filename": "TABLE.xlsx", "sha256": OTHER_SHA},
        }
    }

    errors = validate_manifest_files(manifest)

    assert "duplicate_vintage_key:2023" in errors
    assert "non_canonical_filename:./adult.tab" in errors
    assert "filename_collision:adult.tab" in errors
    assert "duplicate_filename_in_vintage:child.tab" in errors
    # Two public entries, one path, different bytes: the tree can hold one.
    assert "filename_collision:table.xlsx" in errors


def test_manifest_level_validation_accepts_the_same_file_under_two_keys():
    # The SSA manifests register one file under a year and a label key.
    manifest = {
        "files": {
            2024: {"filename": "ssa.csv", "sha256": FIXTURE_SHA},
            "extracted_targets": {"filename": "ssa.csv", "sha256": FIXTURE_SHA},
        }
    }
    assert validate_manifest_files(manifest) == ()


def test_manifest_level_validation_accepts_a_hash_only_reissue():
    manifest = {
        "files": {
            2023: [
                {"filename": "adult.tab", "access": "licensed", "sha256": FIXTURE_SHA},
                {"filename": "adult.tab", "access": "licensed", "sha256": OTHER_SHA},
            ]
        }
    }
    assert validate_manifest_files(manifest) == ()
    assert validate_manifest_files({"files": ["not", "a", "mapping"]}) == (
        "files_not_a_mapping",
    )


# --------------------------------------------------------------------------
# register-artifact
# --------------------------------------------------------------------------


def test_register_writes_identity_without_bytes_or_an_r2_key(tmp_path):
    output_dir = tmp_path / "pkg"

    report = _register(output_dir)
    manifest = _manifest(output_dir)
    entry = manifest["files"][2023][0]

    assert report.valid
    assert report.replaced is False
    assert report.hash_source == "consumer_attested"
    assert report.attested_by == "PolicyEngine/microcosm"
    assert report.registration == f"dwp/dwp-frs-2023-24/2023/{FIXTURE_SHA}/adult.tab"
    assert manifest["kind"] == "microdata_release"
    assert entry["access"] == "licensed"
    assert entry["sha256"] == FIXTURE_SHA
    assert entry["hash_source"] == "consumer_attested"
    assert entry["attested_by"] == "PolicyEngine/microcosm"
    assert entry["attestation_evidence"] == ATTESTED["attestation_evidence"]
    assert entry["verified_at"] == "2026-09-02"
    assert "storage" not in entry
    assert sorted(path.name for path in output_dir.iterdir()) == ["manifest.yaml"]


def test_register_a_consumer_pin_records_where_it_was_read(tmp_path):
    output_dir = tmp_path / "pkg"

    _register(output_dir, attestation_evidence=None, verified_at=None, **PINNED)
    entry = _manifest(output_dir)["files"][2023][0]

    assert entry["hash_source"] == "consumer_pin"
    assert entry["pinned_from"] == PINNED["pinned_from"]
    assert "verified_at" not in entry
    assert list(entry).index("hash_source") < list(entry).index("pinned_from")


def test_register_refuses_public_access(tmp_path):
    with pytest.raises(HashOnlyRegistrationError, match="refuses access='public'"):
        _register(tmp_path / "pkg", access="public")


@pytest.mark.parametrize("sha256", ["", "abc123", FIXTURE_SHA.upper(), "z" * 64])
def test_register_never_invents_a_hash(tmp_path, sha256):
    with pytest.raises(HashOnlyRegistrationError, match="Never invent a hash"):
        _register(tmp_path / "pkg", sha256=sha256)
    assert not (tmp_path / "pkg").exists()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"licence": ""}, "must record the publisher licence"),
        ({"vintage": ""}, "must record the artifact vintage"),
        ({"filename": "../adult.tab"}, "must be a bare filename"),
        ({"filename": "./adult.tab"}, "must be a bare filename"),
        ({"hash_source": "chronicle_fetch"}, "does not apply"),
        ({"hash_source": "guessed"}, "Unknown hash_source"),
        ({"attested_by": ""}, "pass --attested-by"),
        ({"attestation_evidence": None}, "pass --attestation-evidence"),
        ({"verified_at": None}, "pass --verified-at"),
    ],
)
def test_register_refuses_an_incomplete_registration(tmp_path, overrides, expected):
    with pytest.raises(HashOnlyRegistrationError, match=expected):
        _register(tmp_path / "pkg", **overrides)
    assert not (tmp_path / "pkg").exists()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"pinned_from": None}, "pass --pinned-from-repository"),
        ({"pinned_from": {"repository": "PolicyEngine/microcosm"}}, "40-hex commit"),
        ({"verified_at": "2026-09-02"}, "carries no verified_at"),
    ],
)
def test_register_refuses_an_incomplete_consumer_pin(tmp_path, overrides, expected):
    kwargs = {"hash_source": None, "attestation_evidence": None, "verified_at": None}
    kwargs.update(PINNED)
    kwargs.update(overrides)
    with pytest.raises(HashOnlyRegistrationError, match=expected):
        _register(tmp_path / "pkg", **kwargs)


def test_register_refuses_without_an_access_route(tmp_path):
    with pytest.raises(HashOnlyRegistrationError, match="how the bytes are reached"):
        _register(tmp_path / "pkg", doi=None)


def test_register_refuses_while_the_bytes_are_present(tmp_path):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    (output_dir / "adult.tab").write_bytes(b"licensed microdata must not live here")

    with pytest.raises(HashOnlyRegistrationError, match="while its bytes"):
        _register(output_dir)


def test_register_refuses_a_normalized_alias_of_local_bytes(tmp_path, monkeypatch):
    """Simulate a case-sensitive directory while running on folded APFS."""
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    actual_path = output_dir / "ADULT.TAB"
    content = b"licensed microdata must not live here"
    actual_path.write_bytes(content)
    requested_path = output_dir / "adult.tab"
    real_exists = Path.exists

    def case_sensitive_exists(path: Path) -> bool:
        if path == requested_path:
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", case_sensitive_exists)

    with pytest.raises(HashOnlyRegistrationError, match="ADULT.TAB"):
        _register(output_dir, filename="adult.tab")

    assert actual_path.read_bytes() == content
    assert not (output_dir / "manifest.yaml").exists()


@pytest.mark.parametrize(
    "document",
    ["[]\n", "false\n", "0\n", "''\n"],
    ids=["empty-list", "false", "zero", "empty-string"],
)
def test_register_refuses_a_falsy_non_mapping_manifest(tmp_path, document):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.yaml"
    manifest_path.write_text(document)
    original = manifest_path.read_bytes()

    with pytest.raises(HashOnlyRegistrationError, match="must be a mapping"):
        _register(output_dir)

    assert manifest_path.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "blank"),
    [("source_id", None), ("source_id", "   "), ("package_id", "")],
)
def test_register_replaces_blank_manifest_identity_fields(tmp_path, field, blank):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    payload = {
        "source_id": "dwp",
        "package_id": "dwp-frs-2023-24",
        "kind": "microdata_release",
        "files": {},
    }
    payload[field] = blank
    (output_dir / "manifest.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    report = _register(output_dir)
    manifest = _manifest(output_dir)

    assert report.source_id == "dwp"
    assert report.package_id == "dwp-frs-2023-24"
    assert manifest["source_id"] == "dwp"
    assert manifest["package_id"] == "dwp-frs-2023-24"


def test_register_replaces_a_blank_manifest_level_access_route(tmp_path):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "kind": "microdata_release",
                "source_page": None,
                "files": {},
            },
            sort_keys=False,
        )
    )
    source_page = "https://publisher.example/frs"

    _register(output_dir, doi=None, source_page=source_page)
    manifest = _manifest(output_dir)
    entry = manifest["files"][2023][0]

    assert manifest["source_page"] == source_page
    assert "missing_access_route" not in validate_file_entry(
        entry,
        kind="microdata_release",
        manifest=manifest,
        local_file_exists=False,
    )


@pytest.mark.parametrize("outside_exists", [True, False], ids=["existing", "dangling"])
def test_register_refuses_a_symlinked_manifest_target_before_writing(
    tmp_path, outside_exists
):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    outside = tmp_path / "outside.yaml"
    original = yaml.safe_dump(
        {
            "source_id": "dwp",
            "package_id": "dwp-frs-2023-24",
            "kind": "microdata_release",
            "files": {},
        },
        sort_keys=False,
    ).encode()
    if outside_exists:
        outside.write_bytes(original)
    manifest_path = output_dir / "manifest.yaml"
    manifest_path.symlink_to(outside)

    with pytest.raises(HashOnlyRegistrationError, match="symbolic link"):
        _register(output_dir)

    assert manifest_path.is_symlink()
    if outside_exists:
        assert outside.read_bytes() == original
    else:
        assert not outside.exists()


@pytest.mark.parametrize("target_exists", [True, False], ids=("existing", "dangling"))
def test_register_refuses_a_symlinked_output_directory_before_writing(
    tmp_path, target_exists
):
    outside = tmp_path / "outside"
    if target_exists:
        outside.mkdir()
    output_dir = tmp_path / "requested"
    output_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(HashOnlyRegistrationError, match="symbolic link"):
        _register(output_dir)

    assert output_dir.is_symlink()
    assert not (outside / "manifest.yaml").exists()


def test_register_checks_symlinks_before_resolving_parent_segments(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    linked_parent = outside / "linked-parent"
    linked_parent.mkdir(parents=True)
    (outside / "escaped").mkdir()
    (inside / "escaped").mkdir()
    link = inside / "link"
    link.symlink_to(linked_parent, target_is_directory=True)
    output_dir = link / ".." / "escaped"

    with pytest.raises(HashOnlyRegistrationError, match="symbolic link"):
        _register(output_dir)

    assert not (outside / "escaped" / "manifest.yaml").exists()
    assert not (inside / "escaped" / "manifest.yaml").exists()


def test_registration_persists_with_atomic_replace_under_an_exclusive_lock(
    tmp_path, monkeypatch
):
    events: list[tuple[str, int | None]] = []
    real_flock = fcntl.flock
    real_replace = os.replace

    def observed_flock(fd, operation):
        events.append(("flock", operation))
        return real_flock(fd, operation)

    def observed_replace(source, destination):
        assert events and events[-1] == ("flock", fcntl.LOCK_EX)
        events.append(("replace", None))
        return real_replace(source, destination)

    monkeypatch.setattr(fcntl, "flock", observed_flock)
    monkeypatch.setattr(os, "replace", observed_replace)

    report = _register(tmp_path / "pkg")

    assert report.valid
    assert events == [
        ("flock", fcntl.LOCK_EX),
        ("replace", None),
        ("flock", fcntl.LOCK_UN),
    ]


def test_registration_lock_identity_folds_case_variant_package_paths(tmp_path):
    assert _registration_lock_path(tmp_path / "Package") == _registration_lock_path(
        tmp_path / "package"
    )


def test_concurrent_registrations_preserve_both_manifest_updates(tmp_path, monkeypatch):
    output_dir = tmp_path / "pkg"
    first_inside_replace = threading.Event()
    release_first_replace = threading.Event()
    second_inside_replace = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()
    from chronicle import registration

    real_atomic_replace = registration._atomic_replace_manifest

    def coordinated_replace(manifest_path, document):
        nonlocal call_count
        with call_count_lock:
            call_index = call_count
            call_count += 1
        if call_index == 0:
            first_inside_replace.set()
            assert release_first_replace.wait(5)
        else:
            second_inside_replace.set()
        real_atomic_replace(manifest_path, document)

    monkeypatch.setattr(registration, "_atomic_replace_manifest", coordinated_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_register, output_dir)
        assert first_inside_replace.wait(5)
        second = executor.submit(
            _register, output_dir, filename="child.tab", sha256=OTHER_SHA
        )
        try:
            assert not second_inside_replace.wait(0.25)
        finally:
            release_first_replace.set()
        assert first.result(timeout=5).valid
        assert second.result(timeout=5).valid

    entries = _manifest(output_dir)["files"][2023]
    assert {(entry["filename"], entry["sha256"]) for entry in entries} == {
        ("adult.tab", FIXTURE_SHA),
        ("child.tab", OTHER_SHA),
    }


def test_fetch_shares_the_registration_lock_before_publisher_read(
    tmp_path, monkeypatch
):
    from chronicle import registration

    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    registration_inside_replace = threading.Event()
    release_registration = threading.Event()
    publisher_read = threading.Event()
    real_atomic_replace = registration._atomic_replace_manifest

    def blocked_registration_replace(manifest_path, document):
        registration_inside_replace.set()
        assert release_registration.wait(5)
        real_atomic_replace(manifest_path, document)

    def served_after_lock(_source_url):
        publisher_read.set()
        return PUBLIC_BYTES, "public-alias.zip"

    monkeypatch.setattr(
        registration, "_atomic_replace_manifest", blocked_registration_replace
    )
    monkeypatch.setattr("chronicle.artifacts._read_artifact", served_after_lock)
    uploads = _record_uploads(monkeypatch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        registration_future = executor.submit(
            _register,
            output_dir,
            source_id="census_acs",
            package_id="census-acs-pums-2022-1yr",
            year=2022,
            filename="adult.tab",
            sha256=PUBLIC_SHA,
            vintage="2022",
        )
        assert registration_inside_replace.wait(5)
        fetch_future = executor.submit(
            _fetch_release,
            output_dir,
            staging_dir=staging,
            filename="public-alias.zip",
            content=PUBLIC_BYTES,
            upload_r2=True,
        )
        try:
            assert not publisher_read.wait(0.25)
        finally:
            release_registration.set()
        assert registration_future.result(timeout=5).valid
        with pytest.raises(ManifestAccessError):
            fetch_future.result(timeout=5)

    assert uploads == []
    assert not staging.exists()
    entries = _manifest(output_dir)["files"][2022]
    assert [(entry["filename"], entry["access"]) for entry in entries] == [
        ("adult.tab", "licensed")
    ]


def test_publish_holds_the_registration_lock_through_manifest_write(
    tmp_path, monkeypatch
):
    from chronicle import registration

    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output_dir, staging_dir=staging)
    upload_started = threading.Event()
    release_upload = threading.Event()
    registration_inside_replace = threading.Event()
    real_atomic_replace = registration._atomic_replace_manifest

    def blocked_upload(location, local_path, *, wrangler_command):
        upload_started.set()
        assert release_upload.wait(5)
        return ArtifactCommandResult(
            command=("stub",), returncode=0, stdout="ok", stderr=""
        )

    def observed_registration_replace(manifest_path, document):
        registration_inside_replace.set()
        real_atomic_replace(manifest_path, document)

    monkeypatch.setattr("chronicle.artifacts._upload_r2_object", blocked_upload)
    monkeypatch.setattr(
        registration, "_atomic_replace_manifest", observed_registration_replace
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        publish_future = executor.submit(
            publish_source_artifacts,
            output_dir,
            staging_dir=staging,
            skip_hash_only=True,
        )
        assert upload_started.wait(5)
        registration_future = executor.submit(
            _register,
            output_dir,
            source_id="census_acs",
            package_id="census-acs-pums-2022-1yr",
            year=2022,
            filename="child.tab",
            vintage="2022",
        )
        try:
            assert not registration_inside_replace.wait(0.25)
        finally:
            release_upload.set()
        assert publish_future.result(timeout=5).valid
        assert registration_future.result(timeout=5).valid

    entries = _manifest(output_dir)["files"][2022]
    assert {entry["filename"] for entry in entries} == {"csv_hus.zip", "child.tab"}
    public = next(entry for entry in entries if entry["filename"] == "csv_hus.zip")
    assert public["storage"]["r2"]["provider"] == "r2"


def test_atomic_registration_failure_preserves_the_original_manifest(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "kind": "microdata_release",
                "files": {},
            },
            sort_keys=False,
        )
    )
    original = manifest_path.read_bytes()

    monkeypatch.setattr(
        os,
        "replace",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        _register(output_dir)

    assert manifest_path.read_bytes() == original
    assert not list(output_dir.glob(".manifest.yaml.*.tmp"))


def test_register_refuses_a_publisher_table_manifest(tmp_path):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "kind": "publisher_table",
                "files": {2023: {"filename": "table.ods", "sha256": FIXTURE_SHA}},
            }
        )
    )

    with pytest.raises(HashOnlyRegistrationError, match="publisher_table manifest"):
        _register(output_dir)


def test_register_refuses_a_kindless_manifest_with_content(tmp_path):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "files": {2023: {"filename": "table.ods", "sha256": FIXTURE_SHA}},
            }
        )
    )
    original = (output_dir / "manifest.yaml").read_bytes()

    with pytest.raises(HashOnlyRegistrationError, match="declares no kind"):
        _register(output_dir)

    assert (output_dir / "manifest.yaml").read_bytes() == original


def _entryless_manifest(tmp_path: Path, **fields: object) -> Path:
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    payload = {"source_id": "dwp", "package_id": "dwp-frs-2023-24", **fields}
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False).replace("files: null", "files:")
    )
    return output_dir


@pytest.mark.parametrize("files", [None, {}], ids=["explicit-null", "empty-mapping"])
def test_register_declares_the_kind_of_an_entryless_kindless_manifest(tmp_path, files):
    output_dir = _entryless_manifest(tmp_path, files=files)

    report = _register(output_dir)
    manifest = _manifest(output_dir)

    assert report.valid
    assert manifest["kind"] == "microdata_release"
    assert manifest["files"][2023][0]["filename"] == "adult.tab"


def test_register_never_reclassifies_a_declared_publisher_table(tmp_path):
    # An explicit kind is fixed even when the manifest holds no entry yet.
    output_dir = _entryless_manifest(tmp_path, kind="publisher_table", files={})
    original = (output_dir / "manifest.yaml").read_bytes()

    with pytest.raises(HashOnlyRegistrationError, match="publisher_table manifest"):
        _register(output_dir)

    assert (output_dir / "manifest.yaml").read_bytes() == original


def test_register_refuses_a_manifest_for_a_different_source(tmp_path):
    output_dir = tmp_path / "pkg"
    _register(output_dir)

    with pytest.raises(HashOnlyRegistrationError, match="declares source_id='dwp'"):
        _register(output_dir, source_id="ons", filename="other.tab")


def test_register_is_idempotent_and_byte_stable(tmp_path):
    output_dir = tmp_path / "pkg"
    _register(output_dir)
    first = (output_dir / "manifest.yaml").read_bytes()

    report = _register(output_dir)

    assert report.replaced is True
    assert (output_dir / "manifest.yaml").read_bytes() == first


def test_register_refuses_a_reissue_unless_it_is_asked_for(tmp_path):
    output_dir = tmp_path / "pkg"
    _register(output_dir)
    original = (output_dir / "manifest.yaml").read_bytes()

    with pytest.raises(HashOnlyRegistrationError, match="pass --allow-reissue"):
        _register(output_dir, sha256=OTHER_SHA)
    assert (output_dir / "manifest.yaml").read_bytes() == original

    report = _register(output_dir, sha256=OTHER_SHA, allow_reissue=True)
    entries = _manifest(output_dir)["files"][2023]

    assert report.replaced is False
    assert [entry["sha256"] for entry in entries] == [FIXTURE_SHA, OTHER_SHA]


def test_register_keeps_distinct_files_under_one_vintage(tmp_path):
    output_dir = tmp_path / "pkg"
    _register(output_dir, filename="adult.tab")
    _register(output_dir, filename="child.tab", sha256=OTHER_SHA)

    entries = _manifest(output_dir)["files"][2023]

    assert [entry["filename"] for entry in entries] == ["adult.tab", "child.tab"]


def test_reregistering_the_current_pin_stays_idempotent_after_a_reissue(tmp_path):
    output_dir = tmp_path / "pkg"
    _register(output_dir, sha256=FIXTURE_SHA)
    _register(output_dir, sha256=OTHER_SHA, allow_reissue=True)

    report = _register(output_dir, sha256=OTHER_SHA)

    assert report.replaced is True
    entries = _manifest(output_dir)["files"][2023]
    assert [entry["sha256"] for entry in entries] == [FIXTURE_SHA, OTHER_SHA]


# Finding 7: a quoted string year key is the same vintage as the integer.


def _quoted_year_release(tmp_path: Path, **entry_overrides: object) -> Path:
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    entry = _attested_entry(**entry_overrides)
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "kind": "microdata_release",
                "files": {"2023": [entry]},
            },
            sort_keys=False,
        )
    )
    return output_dir


def test_register_sees_a_registration_under_a_quoted_year_key(tmp_path):
    output_dir = _quoted_year_release(tmp_path)
    original = (output_dir / "manifest.yaml").read_bytes()

    with pytest.raises(HashOnlyRegistrationError, match="already registers"):
        _register(output_dir, sha256=OTHER_SHA)
    assert (output_dir / "manifest.yaml").read_bytes() == original

    report = _register(output_dir)
    manifest = _manifest(output_dir)

    assert report.replaced is True
    # The manifest's own key spelling is retained; no parallel key appears.
    assert list(manifest["files"]) == ["2023"]


def test_register_refuses_a_vintage_recorded_under_both_key_spellings(tmp_path):
    output_dir = _quoted_year_release(tmp_path)
    manifest = _manifest(output_dir)
    manifest["files"][2023] = [_attested_entry(filename="child.tab")]
    (output_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    original = (output_dir / "manifest.yaml").read_bytes()

    with pytest.raises(HashOnlyRegistrationError, match="duplicate_vintage_key"):
        _register(output_dir, filename="job.tab", sha256=OTHER_SHA)

    assert (output_dir / "manifest.yaml").read_bytes() == original


# Finding 3: a public identity with an object in R2 is not reclassified.


def _archived_release(tmp_path: Path, monkeypatch, *, filename="asecpub23csv.zip"):
    """A public release whose bytes were archived, then cleaned up locally."""
    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    uploads = _record_uploads(monkeypatch)
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(
        output_dir,
        staging_dir=staging,
        filename=filename,
        source_id="census_cps",
        package_id="census-cps-asec-2023",
        year=2023,
        upload_r2=True,
    )
    assert uploads
    staged = microdata_staging_path(
        staging_dir=staging,
        source_id="census_cps",
        package_id="census-cps-asec-2023",
        year=2023,
        sha256=PUBLIC_SHA,
        filename=filename,
    )
    staged.unlink()
    return output_dir, uploads


@pytest.mark.parametrize("access", ["licensed", "restricted"])
def test_register_refuses_to_reclassify_an_identity_r2_still_holds(
    tmp_path, monkeypatch, access
):
    output_dir, uploads = _archived_release(tmp_path, monkeypatch)
    original = (output_dir / "manifest.yaml").read_bytes()

    with pytest.raises(HashOnlyRegistrationError, match="records the R2 object"):
        _register(
            output_dir,
            source_id="census_cps",
            package_id="census-cps-asec-2023",
            filename="asecpub23csv.zip",
            sha256=PUBLIC_SHA,
            access=access,
            licence="Some licence",
            vintage="2023",
        )

    assert (output_dir / "manifest.yaml").read_bytes() == original
    entry = _manifest(output_dir)["files"][2023][0]
    assert entry["storage"]["r2"]["uri"] == uploads[0][0]
    inventory = inventory_source_artifacts(output_dir, staging_dir=tmp_path / "staging")
    assert inventory.counts["r2_link_count"] == 1


def test_register_refuses_to_reclassify_an_unarchived_public_entry(tmp_path):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "census_cps",
                "package_id": "census-cps-asec-2023",
                "kind": "microdata_release",
                "files": {2023: [_public_release_entry(filename="asecpub23csv.zip")]},
            },
            sort_keys=False,
        )
    )
    original = (output_dir / "manifest.yaml").read_bytes()

    with pytest.raises(HashOnlyRegistrationError, match="explicit decision"):
        _register(
            output_dir,
            source_id="census_cps",
            package_id="census-cps-asec-2023",
            filename="ASECPUB23CSV.ZIP",
            sha256=PUBLIC_SHA,
            licence="Some licence",
            vintage="2023",
        )

    assert (output_dir / "manifest.yaml").read_bytes() == original


def test_register_refuses_a_collision_with_a_public_alias(tmp_path):
    # Finding 1 from the registration side: a manifest never holds one path
    # under two access classes, whatever spelling the other entry uses.
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "kind": "microdata_release",
                "files": {2022: [_public_release_entry(filename="Adult.tab")]},
            },
            sort_keys=False,
        )
    )

    with pytest.raises(HashOnlyRegistrationError, match="explicit decision"):
        _register(output_dir)


# --------------------------------------------------------------------------
# fetch-artifact
# --------------------------------------------------------------------------


def test_fetch_refuses_a_hash_only_access_class(tmp_path, monkeypatch):
    source = tmp_path / "adult.tab"
    source.write_bytes(b"licensed microdata")
    _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="fetch-artifact stores bytes"):
        fetch_source_artifact(
            str(source),
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            output_dir=tmp_path / "pkg",
            access="licensed",
        )


def test_fetch_refuses_to_pull_bytes_over_a_hash_only_registration(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    _register(output_dir)
    original_manifest = (output_dir / "manifest.yaml").read_bytes()
    source = tmp_path / "adult.tab"
    source.write_bytes(b"licensed microdata")
    _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="Its bytes must not enter"):
        fetch_source_artifact(
            str(source),
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            output_dir=output_dir,
            filename="adult.tab",
            access="public",
        )

    assert not (output_dir / "adult.tab").exists()
    assert (output_dir / "manifest.yaml").read_bytes() == original_manifest


# Finding 5: the inferred filename is refused before the read, not after.


@pytest.mark.parametrize("year", [2023, 2022], ids=["same-vintage", "other-vintage"])
def test_fetch_without_filename_refuses_a_hash_only_registration_before_reading(
    tmp_path, monkeypatch, year
):
    output_dir = tmp_path / "pkg"
    _register(output_dir)
    original_manifest = (output_dir / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch, "the artifact was downloaded")

    with pytest.raises(ManifestAccessError, match="Its bytes must not enter"):
        fetch_source_artifact(
            "https://beta.ukdataservice.ac.uk/Umbraco/Surface/Download/9367/adult.tab",
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=year,
            output_dir=output_dir,
            licence="UK Data Service End User Licence",
            access="public",
        )

    assert reads == []
    assert (output_dir / "manifest.yaml").read_bytes() == original_manifest
    assert sorted(path.name for path in output_dir.iterdir()) == ["manifest.yaml"]


def test_fetch_refuses_an_uninferrable_or_unsupported_source_before_reading(
    tmp_path, monkeypatch
):
    reads = _refuse_read(monkeypatch)
    common = {
        "source_id": "irs_soi",
        "package_id": "soi-table-1-2",
        "year": 2023,
        "output_dir": tmp_path / "pkg",
    }
    with pytest.raises(ArtifactFilenameError, match="inferred from the URL"):
        fetch_source_artifact("https://publisher.example/", **common)
    with pytest.raises(ValueError, match="Unsupported source URL scheme"):
        fetch_source_artifact("ftp://publisher.example/table.xlsx", **common)
    assert reads == []
    assert not (tmp_path / "pkg").exists()


# Finding 1: no alias of a registered name slips past the byte boundary.

ALIASES = {
    "dot-slash": lambda out: "./adult.tab",
    "dot-dot-segment": lambda out: "sub/../adult.tab",
    "trailing-slash": lambda out: "adult.tab/",
    "parent-then-back": lambda out: f"../{out.name}/adult.tab",
    "absolute": lambda out: str(out / "adult.tab"),
    "case": lambda out: "ADULT.TAB",
}


@pytest.mark.parametrize("alias", ALIASES.values(), ids=ALIASES.keys())
def test_fetch_refuses_every_alias_of_a_hash_only_registration_before_reading(
    tmp_path, monkeypatch, alias
):
    output_dir = tmp_path / "pkg"
    _register(output_dir)
    original_manifest = (output_dir / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    with pytest.raises(ManifestAccessError):
        fetch_source_artifact(
            "https://publisher.example/frs/adult.tab",
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            output_dir=output_dir,
            filename=alias(output_dir),
            licence="Open Government Licence v3",
            access="public",
            upload_r2=True,
        )

    assert reads == []
    assert sorted(path.name for path in output_dir.iterdir()) == ["manifest.yaml"]
    assert (output_dir / "manifest.yaml").read_bytes() == original_manifest


def test_fetch_refuses_a_path_that_escapes_the_package_directory(tmp_path, monkeypatch):
    reads = _refuse_read(monkeypatch)
    with pytest.raises(ArtifactFilenameError, match="bare filename"):
        _fetch_table(
            tmp_path / "table.xlsx",
            tmp_path / "pkg",
            filename="../../escaped.xlsx",
        )
    assert reads == []
    assert not (tmp_path / "escaped.xlsx").exists()


def test_fetch_writes_the_access_class_and_kind_explicitly(tmp_path):
    source = tmp_path / "table.xlsx"
    source.write_bytes(b"publisher table")
    output_dir = tmp_path / "pkg"

    _fetch_table(source, output_dir)
    manifest = _manifest(output_dir)

    assert list(manifest)[:3] == ["source_id", "package_id", "kind"]
    assert manifest["kind"] == "publisher_table"
    assert manifest["files"][2023]["access"] == "public"


def test_fetch_into_a_microdata_release_manifest_requires_the_evidence(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    _register(output_dir)
    original = (output_dir / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch)
    base = {
        "source_id": "dwp",
        "package_id": "dwp-frs-2023-24",
        "year": 2023,
        "output_dir": output_dir,
        "filename": "codebook.pdf",
        "access": "public",
        "licence": "OGL-UK-3.0",
        "publisher": "Department for Work and Pensions",
        "vintage": "2023_24",
        # Not the digest the manifest registers hash-only: those bytes are
        # refused as the gated artifact whatever name they arrive under.
        "expected_sha256": OTHER_SHA,
        "licence_evidence": {**EVIDENCE, "issuer": "DWP"},
        "staging_dir": tmp_path / "staging",
    }
    cases = [
        ({"licence": None}, "pass --licence"),
        ({"vintage": None}, "pass --vintage"),
        ({"publisher": None}, "pass --publisher"),
        ({"expected_sha256": None}, "pass --expected-sha256"),
        (
            {"licence": "UK Data Service End User Licence"},
            "not on Chronicle's allowlist",
        ),
        ({"licence_evidence": None}, "licence_evidence_missing_field:issuer"),
        (
            {"licence_evidence": {**EVIDENCE, "url": "ask the archive"}},
            "url_not_durable",
        ),
    ]
    for overrides, expected in cases:
        kwargs = {**base, **overrides}
        with pytest.raises(ManifestAccessError, match=expected):
            fetch_source_artifact("https://publisher.example/codebook.pdf", **kwargs)

    assert reads == []
    assert (output_dir / "manifest.yaml").read_bytes() == original


def test_fetch_refuses_a_registration_recorded_under_another_vintage(
    tmp_path, monkeypatch
):
    # The write target is a path in the package directory, not a year, so a
    # registration under 2022 must still block a fetch requested for 2023.
    package = tmp_path / "pkg"
    _register(package, year=2022, vintage="2022_23")
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="for 2022 as access='licensed'"):
        fetch_source_artifact(
            "https://publisher.example/adult.tab",
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            output_dir=package,
            filename="adult.tab",
            licence="OGL-UK-3.0",
            access="public",
        )

    assert reads == []
    assert not (package / "adult.tab").exists()


def test_fetch_refuses_a_list_entry_in_a_manifest_without_a_kind(tmp_path, monkeypatch):
    # A missing kind must not make the guard blind to the entries the
    # manifest actually holds: the byte boundary is refused first.
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "files": {2023: [_attested_entry()]},
            }
        )
    )
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="Its bytes must not enter"):
        fetch_source_artifact(
            "https://publisher.example/adult.tab",
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            output_dir=package,
            filename="adult.tab",
            access="public",
        )

    assert reads == []
    assert not (package / "adult.tab").exists()


# Finding 4: the existing manifest is validated strictly before any I/O.


def test_fetch_refuses_a_kind_that_conflicts_with_the_manifest(tmp_path, monkeypatch):
    release = tmp_path / "release"
    _register(release)
    table = tmp_path / "table"
    (tmp_path / "t.xlsx").write_bytes(b"table")
    _fetch_table(tmp_path / "t.xlsx", table)
    reads = _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)
    before = {path: (path / "manifest.yaml").read_bytes() for path in (release, table)}

    with pytest.raises(ManifestAccessError, match="is a microdata_release manifest"):
        fetch_source_artifact(
            "https://publisher.example/codebook.pdf",
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            output_dir=release,
            kind="publisher_table",
            upload_r2=True,
        )
    with pytest.raises(ManifestAccessError, match="is a publisher_table manifest"):
        fetch_source_artifact(
            "https://publisher.example/22in05ira.xlsx",
            source_id="irs_soi",
            package_id="soi-table-1-2",
            year=2022,
            output_dir=table,
            kind="microdata_release",
            licence="US-Government-Work",
            upload_r2=True,
        )

    assert reads == []
    for path, original in before.items():
        assert (path / "manifest.yaml").read_bytes() == original
        assert not (path / "codebook.pdf").exists()


@pytest.mark.parametrize("files", [None, {}], ids=["explicit-null", "empty-mapping"])
def test_fetch_declares_the_requested_kind_on_an_entryless_kindless_manifest(
    tmp_path, monkeypatch, files
):
    output_dir = _entryless_manifest(tmp_path, files=files)
    _serve(monkeypatch, PUBLIC_BYTES)

    report = _fetch_release(
        output_dir,
        staging_dir=tmp_path / "staging",
        source_id="dwp",
        package_id="dwp-frs-2023-24",
        year=2023,
        licence="OGL-UK-3.0",
        publisher="Department for Work and Pensions",
        vintage="2023_24",
        licence_evidence={**EVIDENCE, "issuer": "DWP"},
    )
    manifest = _manifest(output_dir)

    assert report.valid
    assert manifest["kind"] == "microdata_release"
    assert [entry["filename"] for entry in manifest["files"][2023]] == ["csv_hus.zip"]


def test_fetch_never_reclassifies_a_declared_kind_without_entries(
    tmp_path, monkeypatch
):
    output_dir = _entryless_manifest(tmp_path, kind="publisher_table", files={})
    original = (output_dir / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="is a publisher_table manifest"):
        _fetch_release(
            output_dir,
            staging_dir=tmp_path / "staging",
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            licence="OGL-UK-3.0",
            publisher="Department for Work and Pensions",
            vintage="2023_24",
            licence_evidence={**EVIDENCE, "issuer": "DWP"},
        )

    assert reads == []
    assert (output_dir / "manifest.yaml").read_bytes() == original


def test_fetch_refuses_a_stored_unknown_kind_even_with_an_explicit_kind(
    tmp_path, monkeypatch
):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "kind": "microdata_rows",
                "files": {2023: {"filename": "table.ods", "sha256": FIXTURE_SHA}},
            }
        )
    )
    original = (package / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch)

    for kind in (None, "publisher_table"):
        with pytest.raises(ManifestAccessError, match="unknown manifest kind"):
            fetch_source_artifact(
                "https://publisher.example/codebook.pdf",
                source_id="dwp",
                package_id="dwp-frs-2023-24",
                year=2023,
                output_dir=package,
                kind=kind,
            )

    assert reads == []
    assert (package / "manifest.yaml").read_bytes() == original


def test_fetch_refuses_a_release_entry_without_an_access_class(tmp_path, monkeypatch):
    package = tmp_path / "pkg"
    package.mkdir()
    entry = _attested_entry()
    del entry["access"]
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "kind": "microdata_release",
                "publisher": "DWP",
                "files": {2023: [entry]},
            }
        )
    )
    original = (package / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    with pytest.raises(ManifestAccessError, match="missing_access"):
        fetch_source_artifact(
            "https://publisher.example/adult.tab",
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
            output_dir=package,
            filename="adult.tab",
            licence="OGL-UK-3.0",
            vintage="2023_24",
            expected_sha256=FIXTURE_SHA,
            licence_evidence={**EVIDENCE, "issuer": "DWP"},
            upload_r2=True,
        )

    assert reads == []
    assert (package / "manifest.yaml").read_bytes() == original
    assert not (package / "adult.tab").exists()


def test_fetch_refuses_an_invalid_manifest_before_reading(tmp_path, monkeypatch):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "kind": "microdata_release",
                "files": {
                    2023: [
                        _attested_entry(filename="adult.tab"),
                        _public_release_entry(filename="./adult.tab"),
                    ]
                },
            }
        )
    )
    original = (package / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="not a valid microdata_release"):
        _fetch_release(
            package,
            staging_dir=tmp_path / "staging",
            filename="job.tab",
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            year=2023,
        )

    assert reads == []
    assert (package / "manifest.yaml").read_bytes() == original


def test_fetch_never_drops_an_existing_registration(tmp_path, monkeypatch):
    output_dir = tmp_path / "pkg"
    _register(output_dir)
    _serve(monkeypatch, b"a public codebook in the same package")

    _fetch_release(
        output_dir,
        staging_dir=tmp_path / "staging",
        filename="codebook.pdf",
        content=b"a public codebook in the same package",
        source_id="dwp",
        package_id="dwp-frs-2023-24",
        year=2023,
        licence="OGL-UK-3.0",
        publisher="Department for Work and Pensions",
        vintage="2023_24",
        licence_evidence={**EVIDENCE, "issuer": "DWP"},
    )

    entries = _manifest(output_dir)["files"][2023]

    assert [entry["filename"] for entry in entries] == ["adult.tab", "codebook.pdf"]
    assert entries[0]["access"] == "licensed"
    assert entries[0]["sha256"] == FIXTURE_SHA
    assert entries[0]["attestation_evidence"] == ATTESTED["attestation_evidence"]


# --------------------------------------------------------------------------
# Public microdata releases: bytes only with evidence, staged outside the tree
# --------------------------------------------------------------------------


def test_fetch_archives_a_public_release_from_a_staging_directory(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "db" / "data" / "census" / "acs_pums_2022_1yr"
    staging = tmp_path / "staging"
    uploads = _record_uploads(monkeypatch)
    _serve(monkeypatch, PUBLIC_BYTES)

    report = _fetch_release(output_dir, staging_dir=staging, upload_r2=True)
    person = b"public person pums"
    _serve(monkeypatch, person)
    _fetch_release(
        output_dir,
        staging_dir=staging,
        filename="csv_pus.zip",
        content=person,
        upload_r2=True,
    )
    manifest = _manifest(output_dir)
    entries = manifest["files"][2022]

    staged = microdata_staging_path(
        staging_dir=staging,
        source_id="census_acs",
        package_id="census-acs-pums-2022-1yr",
        year=2022,
        sha256=PUBLIC_SHA,
        filename="csv_hus.zip",
    )
    assert report.valid
    assert Path(report.local_path) == staged
    assert staged.read_bytes() == PUBLIC_BYTES
    # No release bytes ever land beside the manifest.
    assert sorted(path.name for path in output_dir.iterdir()) == ["manifest.yaml"]
    assert uploads[0][1] == str(staged)
    assert uploads[0][0].endswith(f"/2022/{PUBLIC_SHA}/csv_hus.zip")
    assert manifest["kind"] == "microdata_release"
    assert manifest["publisher"] == "U.S. Census Bureau"
    assert [entry["filename"] for entry in entries] == ["csv_hus.zip", "csv_pus.zip"]
    first = entries[0]
    assert first["access"] == "public"
    assert first["licence"] == "US-Government-Work"
    assert first["licence_evidence"] == {
        **EVIDENCE,
        "licence": "US-Government-Work",
        "sha256": PUBLIC_SHA,
    }
    assert first["vintage"] == "2022"
    assert first["hash_source"] == "chronicle_fetch"
    assert first["attested_by"] == "chronicle"
    assert first["verified_at"] == first["fetched_at"][:10]
    assert first["storage"]["r2"]["uri"] == uploads[0][0]

    inventory = inventory_source_artifacts(output_dir, staging_dir=staging)
    assert inventory.valid
    assert inventory.counts["hash_only_count"] == 0
    assert inventory.counts["r2_link_count"] == 2
    assert all(entry.exists for entry in inventory.entries)


def test_a_public_release_without_a_recorded_object_is_incomplete(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output_dir, staging_dir=staging)

    inventory = inventory_source_artifacts(output_dir, staging_dir=staging)

    assert not inventory.valid
    assert inventory.entries[0].errors == ("r2_object_not_recorded",)


@pytest.mark.parametrize(
    ("locator", "expected_error"),
    [
        pytest.param({}, "recorded_r2_locator_invalid:", id="empty"),
        pytest.param(
            {
                "provider": "r2",
                "bucket": "ledger-raw",
                "key": (
                    "raw/census_acs/census-acs-pums-2022-1yr/2022/"
                    f"{PUBLIC_SHA}/csv_hus.zip"
                ),
                "uri": (
                    "r2://ledger-raw/raw/census_acs/"
                    "census-acs-pums-2022-1yr/2022/"
                    f"{OTHER_SHA}/csv_hus.zip"
                ),
            },
            "recorded_r2_locator_invalid:",
            id="contradictory",
        ),
        pytest.param(
            {
                "provider": "r2",
                "bucket": "ledger-raw",
                "key": (
                    "raw/census_acs/census-acs-pums-2022-1yr/2022/"
                    f"{OTHER_SHA}/other.zip"
                ),
                "uri": (
                    "r2://ledger-raw/raw/census_acs/"
                    "census-acs-pums-2022-1yr/2022/"
                    f"{OTHER_SHA}/other.zip"
                ),
            },
            "recorded_r2_identity_mismatch:",
            id="wrong-entry-identity",
        ),
        pytest.param(
            {
                "provider": "s3",
                "bucket": "ledger-raw",
                "key": (
                    "raw/census_acs/census-acs-pums-2022-1yr/2022/"
                    f"{PUBLIC_SHA}/csv_hus.zip"
                ),
                "uri": (
                    "s3://ledger-raw/raw/census_acs/"
                    "census-acs-pums-2022-1yr/2022/"
                    f"{PUBLIC_SHA}/csv_hus.zip"
                ),
            },
            "recorded_r2_locator_invalid:",
            id="wrong-provider",
        ),
        pytest.param(
            {
                "provider": "r2",
                "bucket": "ledger-raw",
                "key": (
                    f"raw/wrong-source/wrong-package/1999/{PUBLIC_SHA}/csv_hus.zip"
                ),
                "uri": (
                    "r2://ledger-raw/raw/wrong-source/wrong-package/1999/"
                    f"{PUBLIC_SHA}/csv_hus.zip"
                ),
            },
            "recorded_r2_locator_invalid:",
            id="wrong-registration-identity",
        ),
    ],
)
def test_inventory_refuses_an_invalid_or_identity_mismatched_r2_locator(
    tmp_path, monkeypatch, locator, expected_error
):
    output_dir = tmp_path / "pkg"
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output_dir, staging_dir=tmp_path / "unused-staging")
    manifest = _manifest(output_dir)
    manifest["files"][2022][0]["storage"] = {"r2": locator}
    (output_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    inventory = inventory_source_artifacts(
        output_dir, staging_dir=tmp_path / "no-staged-bytes"
    )

    assert not inventory.valid
    assert inventory.counts["r2_link_count"] == 0
    assert inventory.entries[0].r2 is None
    assert "r2_object_not_recorded" in inventory.entries[0].errors
    assert any(
        error.startswith(expected_error) for error in inventory.entries[0].errors
    )


# Finding 8: the fetch refuses bytes the reviewed pin does not cover.


def test_fetch_refuses_a_reissue_before_any_side_effect(tmp_path, monkeypatch):
    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    _serve(monkeypatch, b"silently re-published bytes")
    _forbid_uploads(monkeypatch)

    with pytest.raises(ExpectedArtifactIdentityError) as raised:
        _fetch_release(
            output_dir,
            staging_dir=staging,
            expected_sha256=PUBLIC_SHA,
            upload_r2=True,
        )

    message = str(raised.value)
    assert PUBLIC_SHA in message
    assert hashlib.sha256(b"silently re-published bytes").hexdigest() in message
    assert "unreviewed reissue" in message
    assert not output_dir.exists()
    assert not staging.exists()


def test_fetch_refuses_a_size_that_disagrees_with_the_pin(tmp_path, monkeypatch):
    _serve(monkeypatch, PUBLIC_BYTES)
    with pytest.raises(ExpectedArtifactIdentityError, match="size_bytes=1"):
        _fetch_release(
            tmp_path / "pkg",
            staging_dir=tmp_path / "staging",
            expected_size_bytes=1,
        )
    assert not (tmp_path / "pkg").exists()


def test_record_revision_does_not_override_the_expected_identity(tmp_path, monkeypatch):
    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output_dir, staging_dir=staging)
    original = (output_dir / "manifest.yaml").read_bytes()
    _serve(monkeypatch, b"revised bytes")

    with pytest.raises(ExpectedArtifactIdentityError):
        _fetch_release(
            output_dir,
            staging_dir=staging,
            expected_sha256=PUBLIC_SHA,
            record_revision=True,
        )

    assert (output_dir / "manifest.yaml").read_bytes() == original


def test_an_expectation_that_contradicts_the_record_is_refused_before_reading(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output_dir, staging_dir=staging)
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ExpectedArtifactIdentityError, match="already records"):
        _fetch_release(
            output_dir,
            staging_dir=staging,
            expected_sha256=OTHER_SHA,
            licence_evidence=EVIDENCE,
        )

    assert reads == []


@pytest.mark.parametrize("bad", ["", "abc", FIXTURE_SHA.upper()])
def test_fetch_never_accepts_an_invented_expected_hash(tmp_path, monkeypatch, bad):
    reads = _refuse_read(monkeypatch)
    with pytest.raises(ExpectedArtifactIdentityError, match="Never invent a hash"):
        _fetch_table(tmp_path / "t.xlsx", tmp_path / "pkg", expected_sha256=bad)
    assert reads == []


def test_a_table_fetch_honours_an_expected_hash_before_writing(tmp_path, monkeypatch):
    source = tmp_path / "table.xlsx"
    source.write_bytes(b"publisher table")
    output_dir = tmp_path / "pkg"

    with pytest.raises(ExpectedArtifactIdentityError):
        _fetch_table(source, output_dir, expected_sha256=OTHER_SHA)
    assert not output_dir.exists()

    report = _fetch_table(
        source,
        output_dir,
        expected_sha256=hashlib.sha256(b"publisher table").hexdigest(),
        expected_size_bytes=len(b"publisher table"),
    )
    assert report.valid


def test_fetch_refuses_release_bytes_already_tracked_beside_the_manifest(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    (output_dir / "csv_hus.zip").write_bytes(PUBLIC_BYTES)
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="staged outside the package tree"):
        _fetch_release(output_dir, staging_dir=tmp_path / "staging")

    assert reads == []


# Finding 6: a release vintage is a list, and every entry keeps its identity.


def test_refetching_a_release_file_with_different_bytes_is_refused(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    uploads = _record_uploads(monkeypatch)
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output_dir, staging_dir=staging, upload_r2=True)
    person = b"public person pums"
    _serve(monkeypatch, person)
    _fetch_release(
        output_dir,
        staging_dir=staging,
        filename="csv_pus.zip",
        content=person,
        upload_r2=True,
    )
    original = (output_dir / "manifest.yaml").read_bytes()
    revised = b"household pums, silently re-published"
    reads = _refuse_read(monkeypatch)
    uploads_before = list(uploads)

    # The list entry's recorded identity is live: a reviewed pin for other
    # bytes contradicts it, and the contradiction is refused before the read.
    with pytest.raises(
        ExpectedArtifactIdentityError, match="already records"
    ) as raised:
        _fetch_release(output_dir, staging_dir=staging, content=revised, upload_r2=True)

    assert PUBLIC_SHA in str(raised.value)
    assert "csv_hus.zip" in str(raised.value)
    assert reads == []
    assert (output_dir / "manifest.yaml").read_bytes() == original
    assert uploads == uploads_before

    # Served bytes that differ from the pin the record and the evidence agree
    # on are refused after the read, before any write or upload.
    _serve(monkeypatch, revised)
    with pytest.raises(ExpectedArtifactIdentityError, match="unreviewed reissue"):
        _fetch_release(
            output_dir,
            staging_dir=staging,
            content=revised,
            upload_r2=True,
            expected_sha256=PUBLIC_SHA,
        )
    assert (output_dir / "manifest.yaml").read_bytes() == original
    assert uploads == uploads_before
    assert not microdata_staging_path(
        staging_dir=staging,
        source_id="census_acs",
        package_id="census-acs-pums-2022-1yr",
        year=2022,
        sha256=hashlib.sha256(revised).hexdigest(),
        filename="csv_hus.zip",
    ).exists()


def test_refetching_identical_release_bytes_preserves_the_entry_in_place(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    _record_uploads(monkeypatch)
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output_dir, staging_dir=staging, upload_r2=True)
    person = b"public person pums"
    _serve(monkeypatch, person)
    _fetch_release(
        output_dir,
        staging_dir=staging,
        filename="csv_pus.zip",
        content=person,
        upload_r2=True,
    )
    before = _manifest(output_dir)["files"][2022]

    monkeypatch.setenv("CHRONICLE_R2_RAW_BUCKET", "chronicle-raw")
    _serve(monkeypatch, PUBLIC_BYTES)
    report = _fetch_release(output_dir, staging_dir=staging, upload_r2=True)
    after = _manifest(output_dir)["files"][2022]

    assert report.valid
    assert [entry["filename"] for entry in after] == ["csv_hus.zip", "csv_pus.zip"]
    # The recorded object is history: the backfill copy does not restate it.
    assert after[0]["storage"] == before[0]["storage"]
    assert after[0]["storage"]["r2"]["bucket"] == "ledger-raw"
    assert after[1] == before[1]


def test_record_revision_of_a_release_file_supersedes_only_that_entry(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    _record_uploads(monkeypatch)
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output_dir, staging_dir=staging, upload_r2=True)
    person = b"public person pums"
    _serve(monkeypatch, person)
    _fetch_release(
        output_dir,
        staging_dir=staging,
        filename="csv_pus.zip",
        content=person,
        upload_r2=True,
    )
    before = _manifest(output_dir)["files"][2022]
    revised = b"household pums, revised"
    _serve(monkeypatch, revised)

    _fetch_release(
        output_dir,
        staging_dir=staging,
        content=revised,
        upload_r2=True,
        record_revision=True,
    )
    after = _manifest(output_dir)["files"][2022]

    assert [entry["filename"] for entry in after] == ["csv_hus.zip", "csv_pus.zip"]
    assert after[0]["sha256"] == hashlib.sha256(revised).hexdigest()
    assert after[0]["licence_evidence"]["sha256"] == after[0]["sha256"]
    assert [entry["uri"] for entry in after[0]["storage"]["previous_r2"]] == [
        before[0]["storage"]["r2"]["uri"]
    ]
    assert after[1] == before[1]


def test_a_second_file_never_turns_a_publisher_table_vintage_into_a_list(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    (tmp_path / "22in05ira.xlsx").write_bytes(b"IRA table 5")
    _fetch_table(tmp_path / "22in05ira.xlsx", output_dir, year=2022)
    original = (output_dir / "manifest.yaml").read_bytes()
    (tmp_path / "22in05ira_rev.xlsx").write_bytes(b"IRA table 5, renamed")

    with pytest.raises(SourceArtifactRevisionError):
        _fetch_table(tmp_path / "22in05ira_rev.xlsx", output_dir, year=2022)
    assert (output_dir / "manifest.yaml").read_bytes() == original

    _fetch_table(
        tmp_path / "22in05ira_rev.xlsx", output_dir, year=2022, record_revision=True
    )
    revised = _manifest(output_dir)["files"][2022]

    assert isinstance(revised, dict)
    assert revised["filename"] == "22in05ira_rev.xlsx"


def test_the_manifest_write_refuses_a_same_bytes_rename_by_itself(tmp_path):
    # PR #226's rule: identical bytes under another filename are a rename,
    # not a revision, so --record-revision does not apply. The write path
    # repeats the guard so no caller reaches a false-provenance write.
    output_dir = tmp_path / "pkg"
    (tmp_path / "22in05ira.xlsx").write_bytes(b"IRA table 5")
    _fetch_table(tmp_path / "22in05ira.xlsx", output_dir, year=2022)
    original = (output_dir / "manifest.yaml").read_bytes()

    for record_revision in (False, True):
        with pytest.raises(SourceArtifactRevisionError, match="rename is not"):
            _upsert_manifest(
                output_dir / "manifest.yaml",
                source_id="irs_soi",
                package_id="soi-table-1-2",
                dataset="irs_soi_soi-table-1-2",
                source_page=None,
                table=None,
                publisher=None,
                year=2022,
                filename="table-5.xlsx",
                source_url="https://publisher.example/table-5.xlsx",
                sha256=hashlib.sha256(b"IRA table 5").hexdigest(),
                size_bytes=len(b"IRA table 5"),
                fetched_at="2026-09-04T00:00:00+00:00",
                access="public",
                licence=None,
                kind="publisher_table",
                vintage=None,
                licence_evidence=None,
                expected=_expected_identity(None, None),
                r2_location=None,
                record_revision=record_revision,
            )

    assert (output_dir / "manifest.yaml").read_bytes() == original


def test_a_second_file_over_an_unidentified_table_entry_is_refused(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    output_dir.mkdir()
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "irs_soi",
                "package_id": "soi-table-1-2",
                "kind": "publisher_table",
                "files": {2022: {"filename": "22in05ira.xlsx", "source_url": "x"}},
            }
        )
    )
    original = (output_dir / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch)

    with pytest.raises(MalformedManifestError, match="one file per vintage"):
        _fetch_table(tmp_path / "other.xlsx", output_dir, year=2022)

    assert reads == []
    assert (output_dir / "manifest.yaml").read_bytes() == original


# Finding 7 on the fetch side.


def test_fetch_refuses_a_revision_recorded_under_a_quoted_year_key(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    (tmp_path / "t.xlsx").write_bytes(b"first")
    _fetch_table(tmp_path / "t.xlsx", output_dir, year=2022)
    manifest = _manifest(output_dir)
    manifest["files"] = {"2022": manifest["files"][2022]}
    (output_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    original = (output_dir / "manifest.yaml").read_bytes()
    (tmp_path / "t.xlsx").write_bytes(b"second")

    with pytest.raises(SourceArtifactRevisionError, match="entry '2022'"):
        _fetch_table(tmp_path / "t.xlsx", output_dir, year=2022)
    assert (output_dir / "manifest.yaml").read_bytes() == original

    (tmp_path / "t.xlsx").write_bytes(b"first")
    _fetch_table(tmp_path / "t.xlsx", output_dir, year=2022)
    assert list(_manifest(output_dir)["files"]) == ["2022"]


def test_fetch_refuses_a_vintage_recorded_under_both_key_spellings(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "pkg"
    (tmp_path / "t.xlsx").write_bytes(b"first")
    _fetch_table(tmp_path / "t.xlsx", output_dir, year=2022)
    manifest = _manifest(output_dir)
    manifest["files"]["2022"] = dict(manifest["files"][2022])
    (output_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    original = (output_dir / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch)

    with pytest.raises(ManifestAccessError, match="duplicate_vintage_key"):
        _fetch_table(tmp_path / "t.xlsx", output_dir, year=2022)

    assert reads == []
    assert (output_dir / "manifest.yaml").read_bytes() == original


# --------------------------------------------------------------------------
# publish-raw
# --------------------------------------------------------------------------


def _hash_only_tree(tmp_path: Path) -> Path:
    """Build a data root holding one hash-only registration."""
    root = tmp_path / "data"
    _register(root / "dwp" / "frs_2023_24")
    return root


def test_publish_raw_refuses_hash_only_entries_without_reading_bytes(
    tmp_path, monkeypatch
):
    root = _hash_only_tree(tmp_path)
    _forbid_uploads(monkeypatch)

    report = publish_source_artifacts(root)

    assert not report.valid
    assert report.counts["hash_only_refused_count"] == 1
    assert report.counts["uploaded_count"] == 0
    entry = report.entries[0]
    assert entry.skipped == "hash_only_access:licensed"
    assert entry.errors == ("hash_only_access_refuses_bytes:licensed",)
    assert entry.r2_location is None
    assert entry.upload is None


def test_publish_raw_skip_hash_only_reports_the_skip_without_failing(
    tmp_path, monkeypatch
):
    root = _hash_only_tree(tmp_path)
    _forbid_uploads(monkeypatch)

    report = publish_source_artifacts(root, skip_hash_only=True)

    assert report.valid
    assert report.counts["hash_only_refused_count"] == 1
    assert report.counts["uploaded_count"] == 0
    assert report.entries[0].errors == ()
    assert report.entries[0].r2_location is None


def test_publish_raw_leaves_the_manifest_untouched(tmp_path, monkeypatch):
    root = _hash_only_tree(tmp_path)
    manifest_path = root / "dwp" / "frs_2023_24" / "manifest.yaml"
    original = manifest_path.read_bytes()
    _forbid_uploads(monkeypatch)

    publish_source_artifacts(root, skip_hash_only=True)

    assert manifest_path.read_bytes() == original


def test_publish_raw_reports_a_violation_even_when_skipping(tmp_path, monkeypatch):
    root = _hash_only_tree(tmp_path)
    (root / "dwp" / "frs_2023_24" / "adult.tab").write_bytes(b"leaked bytes")
    _forbid_uploads(monkeypatch)

    report = publish_source_artifacts(root, skip_hash_only=True)

    # --skip-hash-only turns off the refusal, not the contract check.
    assert not report.valid
    assert "bytes_present_for_hash_only_entry" in report.entries[0].errors


@pytest.mark.parametrize("skip_hash_only", [False, True])
def test_publish_raw_never_uploads_a_hash_only_file_through_a_public_alias(
    tmp_path, monkeypatch, skip_hash_only
):
    # Finding 1: a hand-edited public alias of the licensed entry must not
    # carry its bytes to the bucket; the manifest is refused whole.
    root = _hash_only_tree(tmp_path)
    package = root / "dwp" / "frs_2023_24"
    (package / "adult.tab").write_bytes(b"leaked licensed bytes")
    manifest = _manifest(package)
    manifest["files"][2023].append(
        {
            **_public_release_entry(filename="./adult.tab"),
            "sha256": hashlib.sha256(b"leaked licensed bytes").hexdigest(),
        }
    )
    (package / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    original = (package / "manifest.yaml").read_bytes()
    _forbid_uploads(monkeypatch)

    report = publish_source_artifacts(root, skip_hash_only=skip_hash_only)

    assert not report.valid
    assert report.entries == ()
    assert any("filename_collision:adult.tab" in error for error in report.errors)
    assert any("non_canonical_filename:./adult.tab" in error for error in report.errors)
    assert (package / "manifest.yaml").read_bytes() == original


def test_publish_raw_uploads_a_staged_release_and_never_the_tree(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "census" / "acs_pums_2022_1yr"
    staging = tmp_path / "staging"
    _serve(monkeypatch, PUBLIC_BYTES)
    _fetch_release(output_dir, staging_dir=staging)
    uploads = _record_uploads(monkeypatch)

    report = publish_source_artifacts(tmp_path / "data", staging_dir=staging)

    assert report.valid
    assert report.counts["uploaded_count"] == 1
    assert uploads[0][1].startswith(str(staging))
    assert (
        _manifest(output_dir)["files"][2022][0]["storage"]["r2"]["uri"]
        == (uploads[0][0])
    )

    (output_dir / "csv_hus.zip").write_bytes(PUBLIC_BYTES)
    report = publish_source_artifacts(tmp_path / "data", staging_dir=staging)
    assert not report.valid
    assert "bytes_present_for_microdata_release_entry" in report.entries[0].errors


# --------------------------------------------------------------------------
# inventory-artifacts
# --------------------------------------------------------------------------


def test_inventory_accepts_a_hash_only_entry_with_no_local_file(tmp_path):
    root = _hash_only_tree(tmp_path)

    report = inventory_source_artifacts(root)

    assert report.valid
    assert report.counts["hash_only_count"] == 1
    assert report.counts["missing_count"] == 0
    entry = report.entries[0]
    assert entry.valid
    assert entry.exists is False
    assert entry.hash_only is True
    assert entry.access == "licensed"
    assert entry.licence == "UK Data Service End User Licence"
    assert entry.sha256_expected == FIXTURE_SHA
    assert entry.sha256_actual is None
    # The size is the publisher's, recorded rather than measured.
    assert entry.size_bytes == 35323384


def test_inventory_flags_a_hash_only_entry_whose_bytes_are_present(tmp_path):
    root = _hash_only_tree(tmp_path)
    (root / "dwp" / "frs_2023_24" / "adult.tab").write_bytes(b"leaked bytes")

    report = inventory_source_artifacts(root)

    assert not report.valid
    assert "bytes_present_for_hash_only_entry" in report.entries[0].errors


def test_inventory_rejects_a_list_entry_outside_a_microdata_release(tmp_path):
    package = tmp_path / "data" / "dwp" / "tables"
    package.mkdir(parents=True)
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-tables",
                "kind": "publisher_table",
                "files": {2023: [{"filename": "a.ods", "sha256": FIXTURE_SHA}]},
            }
        )
    )

    report = inventory_source_artifacts(tmp_path / "data")

    assert not report.valid
    assert report.entries[0].errors == (
        "list_file_spec_requires_microdata_release_kind",
    )


def test_inventory_reports_manifest_level_defects(tmp_path):
    package = tmp_path / "data" / "dwp" / "tables"
    package.mkdir(parents=True)
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": "dwp",
                "package_id": "dwp-tables",
                "kind": "publisher_table",
                "files": {
                    2023: {"filename": "../a.ods", "sha256": FIXTURE_SHA},
                    "2023": {"filename": "b.ods", "sha256": FIXTURE_SHA},
                },
            }
        )
    )

    report = inventory_source_artifacts(tmp_path / "data")

    assert not report.valid
    assert any("duplicate_vintage_key:2023" in error for error in report.errors)
    # The shared loader refuses the whole document before entry/path inspection.
    assert report.entries == ()


# --------------------------------------------------------------------------
# Source packages never parse a microdata release or a hash-only entry
# --------------------------------------------------------------------------


def test_source_artifact_spec_refuses_to_parse_a_registered_release():
    spec = SourceArtifactSpec(
        source_name="dwp",
        source_table="Family Resources Survey 2023-24",
        resource_package="db",
        resource_directory="data/dwp/frs_2023_24",
        manifest="manifest.yaml",
        vintage="2023_24",
        extracted_at="2026-09-02",
        extraction_method="none",
        parser="delimited_text_full_rows",
        artifact_year=2023,
    )

    with pytest.raises(MicrodataReleaseNotParseableError, match="registers a"):
        spec.assert_parseable_manifest()
    with pytest.raises(MicrodataReleaseNotParseableError):
        spec._artifact_content(2023)


def test_year_mapping_refuses_a_multi_file_vintage_and_both_key_spellings():
    from chronicle.source_package import _year_mapping

    with pytest.raises(ValueError, match="list of 2 entries"):
        _year_mapping(
            {2023: [{"filename": "adult.tab"}, {"filename": "child.tab"}]}, 2023
        )
    with pytest.raises(ValueError, match="both keys"):
        _year_mapping({2023: {"filename": "a"}, "2023": {"filename": "b"}}, 2023)
    assert _year_mapping({"2023": {"filename": "a"}}, 2023) == {"filename": "a"}


def test_validate_package_reports_a_microdata_release_carve_out(tmp_path):
    package_dir = tmp_path / "frs_2023_24"
    package_dir.mkdir()
    (package_dir / "source_package.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "ledger.source_package.v1",
                "package_id": "dwp-frs-2023-24-parse-attempt",
                "label": "Attempt to parse a registered microdata release",
                "artifact": {
                    "source_name": "dwp",
                    "source_table": "Family Resources Survey 2023-24",
                    "resource_package": "db",
                    "resource_directory": "data/dwp/frs_2023_24",
                    "manifest": "manifest.yaml",
                    "vintage": "2023_24",
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
    assert "microdata_release_not_parseable" in {issue.code for issue in report.errors}


# Finding 2: the byte reader refuses a hash-only entry under any manifest kind.

LICENSED_BYTES = b"sernum\tage\n1\t45\n2\t31\n"


class _UnreadableArtifactPath:
    """Bytes sit in the package tree; a refusal must not read them."""

    def read_bytes(self) -> bytes:
        raise AssertionError("a hash-only entry's bytes must not be read")


def _isolated_reader(tmp_path: Path, monkeypatch) -> Path:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv(SOURCE_ARTIFACT_CACHE_ENV, str(cache_root))
    monkeypatch.setattr(
        "chronicle.source_package._fetch_source_artifact_content",
        lambda _url: (_ for _ in ()).throw(
            AssertionError("a hash-only entry must never be fetched")
        ),
    )
    return cache_root


def _licensed_table_spec(
    tmp_path: Path, monkeypatch, *, kind: str | None, access: str
) -> tuple[SourceArtifactSpec, Path]:
    """A publisher-table package whose single entry is hash-only.

    The resource package is a namespace package on ``sys.path`` with a unique
    name, so ``importlib.resources.files`` resolves it exactly as it resolves
    the repository's ``db`` package.
    """
    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "dwp" / "frs"
    resource_dir.mkdir(parents=True)
    manifest: dict = {"source_id": "dwp", "package_id": "dwp-frs"}
    if kind is not None:
        manifest["kind"] = kind
    entry = _attested_entry(
        access=access,
        sha256=hashlib.sha256(LICENSED_BYTES).hexdigest(),
        source_url="https://ukdataservice.example/adult.tab",
    )
    manifest["files"] = {2023: entry}
    (resource_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
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
    return spec, resource_dir


@pytest.mark.parametrize("access", ["licensed", "restricted"])
def test_byte_reader_refuses_a_hash_only_entry_before_any_store(
    tmp_path, monkeypatch, access
):
    cache_root = _isolated_reader(tmp_path, monkeypatch)
    entry = _attested_entry(access=access, source_url="https://x.example/adult.tab")
    cached = cache_root / FIXTURE_SHA / "adult.tab"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(LICENSED_BYTES)
    monkeypatch.setenv(SOURCE_ARTIFACT_FETCH_ENV, "1")

    with pytest.raises(ManifestAccessError, match=f"access={access!r}"):
        _read_source_artifact_content(_UnreadableArtifactPath(), entry)


def test_byte_reader_treats_an_unknown_access_class_as_unreadable(
    tmp_path, monkeypatch
):
    _isolated_reader(tmp_path, monkeypatch)
    entry = _attested_entry(access="internal")
    with pytest.raises(ManifestAccessError, match="Unknown access class 'internal'"):
        _read_source_artifact_content(_UnreadableArtifactPath(), entry)


@pytest.mark.parametrize("kind", ["publisher_table"])
@pytest.mark.parametrize("access", ["licensed", "restricted"])
def test_artifact_content_refuses_a_hash_only_mapping_entry(
    tmp_path, monkeypatch, kind, access
):
    _isolated_reader(tmp_path, monkeypatch)
    spec, _resource_dir = _licensed_table_spec(
        tmp_path, monkeypatch, kind=kind, access=access
    )

    with pytest.raises(ManifestAccessError, match="identity only"):
        spec.assert_parseable(2023)
    with pytest.raises(ManifestAccessError):
        spec._artifact_content(2023)
    with pytest.raises(ManifestAccessError):
        spec.build_source_rows(2023)


def test_validate_package_reports_a_hash_only_entry_without_reading(
    tmp_path, monkeypatch
):
    _isolated_reader(tmp_path, monkeypatch)
    spec, _resource_dir = _licensed_table_spec(
        tmp_path, monkeypatch, kind="publisher_table", access="licensed"
    )
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "source_package.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "ledger.source_package.v1",
                "package_id": "dwp-frs-parse-attempt",
                "label": "Attempt to parse a hash-only table entry",
                "artifact": {
                    "source_name": spec.source_name,
                    "source_table": spec.source_table,
                    "resource_package": spec.resource_package,
                    "resource_directory": spec.resource_directory,
                    "manifest": "manifest.yaml",
                    "vintage": "2023_24",
                    "extracted_at": "2026-09-02",
                    "extraction_method": "none",
                    "parser": "delimited_text_full_rows",
                    "delimiter": "\t",
                    "artifact_year": 2023,
                },
                "record_sets": [],
            },
            sort_keys=False,
        )
    )

    report = validate_source_package(package_dir, year=2023)

    assert not report.valid
    assert [issue.code for issue in report.errors] == [
        "hash_only_artifact_not_parseable"
    ]


def test_build_suite_refuses_a_hash_only_entry_before_touching_the_output(
    tmp_path, monkeypatch
):
    from chronicle.suite import build_source_suite

    _isolated_reader(tmp_path, monkeypatch)
    spec, _resource_dir = _licensed_table_spec(
        tmp_path, monkeypatch, kind="publisher_table", access="licensed"
    )
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "source_package.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "ledger.source_package.v1",
                "package_id": "dwp-frs-parse-attempt",
                "label": "Attempt to build a hash-only table entry",
                "artifact": {
                    "source_name": spec.source_name,
                    "source_table": spec.source_table,
                    "resource_package": spec.resource_package,
                    "resource_directory": spec.resource_directory,
                    "manifest": "manifest.yaml",
                    "vintage": "2023_24",
                    "extracted_at": "2026-09-02",
                    "extraction_method": "none",
                    "parser": "delimited_text_full_rows",
                    "delimiter": "\t",
                    "artifact_year": 2023,
                },
                "record_sets": [],
            },
            sort_keys=False,
        )
    )
    output_dir = tmp_path / "suite"

    with pytest.raises(ManifestAccessError):
        build_source_suite(package_dir, output_dir, year=2023)

    assert not output_dir.exists()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_register_artifact_round_trips(tmp_path, capsys):
    output_dir = tmp_path / "pkg"

    exit_code = harness_main(
        [
            "register-artifact",
            "--source-id",
            "hmrc",
            "--package-id",
            "hmrc-spi-public-use-tape-2022-23",
            "--year",
            "2022",
            "--out-dir",
            str(output_dir),
            "--filename",
            "put2223uk.tab",
            "--sha256",
            FIXTURE_SHA,
            "--size-bytes",
            "141323762",
            "--vintage",
            "2022-23",
            "--licence",
            "UK Data Service End User Licence (study SN 9422)",
            "--access",
            "restricted",
            "--doi",
            "10.5255/UKDA-SN-9422-1",
            "--hash-source",
            "consumer_pin",
            "--attested-by",
            "PolicyEngine/microcosm",
            "--pinned-from-repository",
            "PolicyEngine/microcosm",
            "--pinned-from-path",
            "packages/microcosm-build/src/microcosm/build/uk/hmrc_income_source_stages.json",
            "--pinned-from-commit",
            FIXTURE_COMMIT,
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["valid"] is True
    assert payload["access"] == "restricted"
    assert payload["hash_source"] == "consumer_pin"
    assert payload["r2_location"] is None
    assert payload["registration"] == (
        f"hmrc/hmrc-spi-public-use-tape-2022-23/2022/{FIXTURE_SHA}/put2223uk.tab"
    )
    assert not (output_dir / "put2223uk.tab").exists()
    entry = _manifest(output_dir)["files"][2022][0]
    assert entry["pinned_from"]["commit"] == FIXTURE_COMMIT


def test_cli_register_artifact_rejects_public_access(tmp_path):
    with pytest.raises(SystemExit):
        harness_main(
            [
                "register-artifact",
                "--source-id",
                "census_cps",
                "--package-id",
                "census-cps-asec-2023",
                "--year",
                "2023",
                "--out-dir",
                str(tmp_path / "pkg"),
                "--filename",
                "asecpub23csv.zip",
                "--sha256",
                FIXTURE_SHA,
                "--vintage",
                "2023",
                "--licence",
                "Public domain",
                "--access",
                "public",
                "--hash-source",
                "consumer_pin",
                "--attested-by",
                "x",
            ]
        )


def test_cli_refusals_print_an_error_and_exit_1(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "pkg"
    _register(output_dir)
    reads = _refuse_read(monkeypatch)

    exit_code = harness_main(
        [
            "fetch-artifact",
            "--url",
            "https://publisher.example/adult.tab",
            "--source-id",
            "dwp",
            "--package-id",
            "dwp-frs-2023-24",
            "--year",
            "2023",
            "--out-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "Its bytes must not enter" in captured.err
    assert reads == []

    exit_code = harness_main(
        [
            "register-artifact",
            "--source-id",
            "dwp",
            "--package-id",
            "dwp-frs-2023-24",
            "--year",
            "2023",
            "--out-dir",
            str(output_dir),
            "--filename",
            "adult.tab",
            "--sha256",
            OTHER_SHA,
            "--vintage",
            "2023_24",
            "--licence",
            "UK Data Service End User Licence",
            "--access",
            "licensed",
            "--doi",
            "10.5255/UKDA-SN-9367-2",
            "--hash-source",
            "consumer_pin",
            "--attested-by",
            "PolicyEngine/microcosm",
            "--pinned-from-repository",
            "PolicyEngine/microcosm",
            "--pinned-from-path",
            "p",
            "--pinned-from-commit",
            FIXTURE_COMMIT,
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "pass --allow-reissue" in captured.err


def test_cli_fetch_archives_a_release_with_the_reviewed_identity(
    tmp_path, monkeypatch, capsys
):
    output_dir = tmp_path / "pkg"
    staging = tmp_path / "staging"
    uploads = _record_uploads(monkeypatch)
    _serve(monkeypatch, PUBLIC_BYTES)
    argv = [
        "fetch-artifact",
        "--url",
        "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
        "--source-id",
        "census_cps",
        "--package-id",
        "census-cps-asec-2023",
        "--year",
        "2023",
        "--out-dir",
        str(output_dir),
        "--publisher",
        "U.S. Census Bureau",
        "--vintage",
        "2023 ASEC / 2022 income reference year",
        "--access",
        "public",
        "--licence",
        "US-Government-Work",
        "--kind",
        "microdata_release",
        "--expected-sha256",
        PUBLIC_SHA,
        "--expected-size-bytes",
        str(len(PUBLIC_BYTES)),
        "--licence-evidence-issuer",
        EVIDENCE["issuer"],
        "--licence-evidence-scope",
        EVIDENCE["scope"],
        "--licence-evidence-url",
        EVIDENCE["url"],
        "--staging-dir",
        str(staging),
        "--upload-r2",
    ]

    assert harness_main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    entry = _manifest(output_dir)["files"][2023][0]

    assert payload["valid"] is True
    assert payload["sha256"] == PUBLIC_SHA
    assert entry["filename"] == "asecpub23csv.zip"
    assert entry["vintage"] == "2023 ASEC / 2022 income reference year"
    assert entry["licence_evidence"]["sha256"] == PUBLIC_SHA
    assert uploads[0][0].endswith("/asecpub23csv.zip")

    _serve(monkeypatch, b"a reissue")
    assert harness_main(argv) == 1
    captured = capsys.readouterr()
    assert PUBLIC_SHA in captured.err
    assert captured.out == ""


# --------------------------------------------------------------------------
# The committed registrations
# --------------------------------------------------------------------------

#: Every committed hash-only pin, keyed by registration identity. Values are
#: (sha256, size_bytes, vintage, access, licence); the checksums are the
#: reviewed consumer pins and any change here is a change of identity.
GOLDEN_PINS: dict[tuple[str, str, int, str], tuple[str, int, str, str, str]] = {
    ("dwp", "dwp-frs-2023-24", 2023, "accounts.tab"): (
        "c5e31932bfd06087f835d2c83c0984c85a93409bf5ef85b699cb0958abcba1ea",
        1807921,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "adult.tab"): (
        "e09f9647d03585c81a528636028b2ed495f8f1fbcf64c5e7b4fe521b67367e06",
        35323384,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "benefits.tab"): (
        "ff30d054cc659bcf23b44c492d98cfd701c0bfdb63e8e9aa9769b490ba9d636b",
        4460292,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "benunit.tab"): (
        "88946815eace8561516d5cbb442c27e319c1e90abc381fb2338f0126e3b9e05b",
        21213867,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "child.tab"): (
        "b5dc84fe8b002ee925e61fae23fed27b11537af9fb174f1d07d9cc1748b9702e",
        2913156,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "chldcare.tab"): (
        "566e0ebca1d5e2f3e424e556c91f4cb583d17dadfdfa59feb3841eda7e5976a3",
        273837,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "extchild.tab"): (
        "8d358d7ee66ee4a7ceab87b4f24fbbf21ac86dc038dc7831e51fb271f96a57ec",
        18677,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "househol.tab"): (
        "5fd26b8b675f33b3b30c9ac789a18da17de734790f77e00ded287d1c3a187b30",
        12387117,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "job.tab"): (
        "88b77ffe06865f029f713bb1d55ff12bdea8a1234de5bc293e72458fe64f3a74",
        10934873,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "maint.tab"): (
        "f2dc924eb5a51b0c357791693d15b431327dc39c6421011efb313d88bf839695",
        15440,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "mortgage.tab"): (
        "ce36b477d67837c469608a0d68f7ef269ac04758974235f1157d2f6b92cdbfdc",
        631783,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "oddjob.tab"): (
        "b4ba3dd3151f73a01422983c60514a3e38458ddfa4fb33ae4ed0326873406305",
        5165,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "penprov.tab"): (
        "ee001461c40306ec24b38b2881e1774121114266a2ee449d606cd0a811c37731",
        522313,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("dwp", "dwp-frs-2023-24", 2023, "pension.tab"): (
        "150d6fad1fce81254fb7aea1526fbb00b63d4027d6e2ac4c26bb90aea3127eb7",
        1225838,
        "2023_24",
        "licensed",
        "UK Data Service End User Licence",
    ),
    ("hmrc", "hmrc-spi-public-use-tape-2022-23", 2022, "put2223uk.tab"): (
        "5ef829461060c91a2a47be59ad541d9b519fc3976d66ca80d4920f711bb96f66",
        141323762,
        "2022-23",
        "restricted",
        "UK Data Service End User Licence (study SN 9422)",
    ),
}

CONSUMER_PIN_COMMITS = {
    "packages/microcosm-build/src/microcosm/build/uk/source_stages.json": (
        "2fb2e2f8a99c37725bd6e7a15ff4c2595c912b77"
    ),
    "packages/microcosm-build/src/microcosm/build/uk/hmrc_income_source_stages.json": (
        "de7451bd19ca46d2967e73cdf393908d29e72542"
    ),
}


def _committed_entries(package: Path) -> list[tuple[dict, int, dict]]:
    manifest = yaml.safe_load((package / "manifest.yaml").read_text())
    assert manifest["kind"] == "microdata_release"
    return [
        (manifest, year, entry)
        for year, entries in manifest["files"].items()
        for entry in entries
    ]


def test_committed_pins_match_the_golden_mapping_exactly():
    committed = {
        (manifest["source_id"], manifest["package_id"], year, entry["filename"]): (
            entry["sha256"],
            entry["size_bytes"],
            entry["vintage"],
            entry["access"],
            entry["licence"],
        )
        for package in (FRS_PACKAGE, SPI_PACKAGE)
        for manifest, year, entry in _committed_entries(package)
    }

    assert committed == GOLDEN_PINS
    assert len(GOLDEN_PINS) == 15


def test_committed_frs_registration_covers_every_pinned_tab():
    filenames = [entry["filename"] for _m, _y, entry in _committed_entries(FRS_PACKAGE)]
    assert filenames == sorted(
        key[3] for key in GOLDEN_PINS if key[1] == "dwp-frs-2023-24"
    )


@pytest.mark.parametrize("package", [FRS_PACKAGE, SPI_PACKAGE])
def test_committed_registrations_are_consumer_pins_without_bytes(package):
    entries = _committed_entries(package)

    assert entries
    for _manifest_payload, _year, entry in entries:
        assert entry["access"] in {"licensed", "restricted"}
        assert entry["hash_source"] == "consumer_pin"
        assert entry["attested_by"] == "PolicyEngine/microcosm"
        assert entry["pinned_from"]["repository"] == "PolicyEngine/microcosm"
        assert (
            entry["pinned_from"]["commit"]
            == (CONSUMER_PIN_COMMITS[entry["pinned_from"]["path"]])
        )
        assert "verified_at" not in entry
        assert "storage" not in entry
        # No bytes accompany a hash-only registration.
        assert not (package / entry["filename"]).exists()
    assert sorted(path.name for path in package.iterdir()) == ["manifest.yaml"]


def test_committed_registrations_pass_inventory_with_the_golden_identities():
    report = inventory_source_artifacts(REPO_ROOT / "db" / "data")
    hash_only = [entry for entry in report.entries if entry.hash_only]

    assert report.valid
    assert all(entry.valid and not entry.exists for entry in hash_only)
    assert all(entry.r2 is None for entry in hash_only)
    identities = {
        (
            yaml.safe_load(Path(entry.manifest_path).read_text())["source_id"],
            yaml.safe_load(Path(entry.manifest_path).read_text())["package_id"],
            int(entry.year),
            entry.filename,
        ): (entry.sha256_expected, entry.size_bytes)
        for entry in hash_only
    }
    assert identities == {
        key: (value[0], value[1]) for key, value in GOLDEN_PINS.items()
    }


def test_no_committed_registration_computes_an_r2_key():
    for package in (FRS_PACKAGE, SPI_PACKAGE):
        text = (package / "manifest.yaml").read_text()
        assert "storage:" not in text
        assert "ledger-raw" not in text
        assert "r2://" not in text


def test_registration_hashes_are_not_hashes_of_anything_chronicle_holds():
    # A registered checksum identifies publisher bytes Chronicle never sees;
    # it must never coincide with the hash of the manifest that records it.
    for package in (FRS_PACKAGE, SPI_PACKAGE):
        manifest_hash = hashlib.sha256(
            (package / "manifest.yaml").read_bytes()
        ).hexdigest()
        assert manifest_hash not in (package / "manifest.yaml").read_text()


def _public_manifest_with_archived_revision() -> dict:
    """A renamed public release keeps the old immutable object's identity."""
    current_key = f"raw/census_cps/census-cps-asec-2023/2023/{PUBLIC_SHA}/current.zip"
    archived_key = (
        f"raw/census_cps/census-cps-asec-2023/2023/{FIXTURE_SHA}/archived.tab"
    )
    return {
        "kind": "microdata_release",
        "source_id": "census_cps",
        "package_id": "census-cps-asec-2023",
        "files": {
            2023: [
                _public_release_entry(
                    filename="current.zip",
                    storage={
                        "r2": {
                            "provider": "r2",
                            "uri": f"r2://ledger-raw/{current_key}",
                        },
                        "previous_r2": [
                            {
                                "provider": "r2",
                                "key": archived_key,
                                "uri": f"r2://ledger-raw/{archived_key}",
                            }
                        ],
                    },
                )
            ]
        },
    }


@pytest.mark.parametrize("manifest_scope", ["selected", "sibling"])
@pytest.mark.parametrize(
    ("filename", "sha256"),
    [("licensed-alias.tab", FIXTURE_SHA), ("ARCHIVED.TAB", OTHER_SHA)],
    ids=["archived-digest", "archived-filename"],
)
def test_registration_refuses_archived_revision_alias_before_mutation(
    tmp_path, monkeypatch, manifest_scope, filename, sha256
):
    package = tmp_path / "package"
    package.mkdir()
    public_manifest = _public_manifest_with_archived_revision()
    selected = package / "manifest.yaml"
    if manifest_scope == "selected":
        selected.write_text(yaml.safe_dump(public_manifest, sort_keys=False))
    else:
        selected.write_text(
            yaml.safe_dump({**public_manifest, "files": {}}, sort_keys=False)
        )
        (package / "manifest_public.yaml").write_text(
            yaml.safe_dump(public_manifest, sort_keys=False)
        )
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    def unexpected_mutation(*args, **kwargs):
        pytest.fail("archived revision alias reached a filesystem mutation")

    monkeypatch.setattr(Path, "mkdir", unexpected_mutation)
    monkeypatch.setattr(
        "chronicle.registration._registration_lock", unexpected_mutation
    )
    monkeypatch.setattr(
        "chronicle.registration._atomic_replace_manifest", unexpected_mutation
    )
    with pytest.raises(HashOnlyRegistrationError, match="records the R2 object"):
        _register(
            package,
            source_id="census_cps",
            package_id="census-cps-asec-2023",
            filename=filename,
            sha256=sha256,
        )

    assert {path.name: path.read_bytes() for path in package.iterdir()} == before


@pytest.mark.parametrize("manifest_scope", ["selected", "sibling"])
@pytest.mark.parametrize(
    ("filename", "sha256", "expected_error"),
    [
        (
            "licensed-alias.tab",
            FIXTURE_SHA,
            f"archived_sha256_collision:{FIXTURE_SHA}",
        ),
        ("ARCHIVED.TAB", OTHER_SHA, "archived_filename_collision:archived.tab"),
    ],
    ids=["archived-digest", "archived-filename"],
)
def test_package_validation_refuses_hash_only_archived_revision_alias(
    manifest_scope, filename, sha256, expected_error
):
    from chronicle.registration import validate_package_directory

    public_manifest = _public_manifest_with_archived_revision()
    licensed_entry = _attested_entry(filename=filename, sha256=sha256)
    if manifest_scope == "selected":
        public_manifest["files"][2023].append(licensed_entry)
        manifests = {"manifest.yaml": public_manifest}
    else:
        manifests = {
            "manifest_public.yaml": public_manifest,
            "manifest.yaml": {
                "kind": "microdata_release",
                "files": {2023: [licensed_entry]},
            },
        }

    assert expected_error in validate_package_directory(manifests)


def _registration_sibling_collision(package: Path, collision: str) -> str:
    """Write valid siblings whose disagreement is unrelated to the new entry."""
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "kind": "microdata_release",
                "source_id": "dwp",
                "package_id": "dwp-frs-2023-24",
                "files": {},
            }
        )
    )
    if collision.startswith("archived-"):
        first = _public_manifest_with_archived_revision()
        filename = (
            "archived-alias.tab" if collision == "archived-digest" else "ARCHIVED.TAB"
        )
        sha256 = FIXTURE_SHA if collision == "archived-digest" else OTHER_SHA
        second = {
            "kind": "microdata_release",
            "files": {2023: [_attested_entry(filename=filename, sha256=sha256)]},
        }
        expected_error = (
            f"archived_sha256_collision:{FIXTURE_SHA}"
            if collision == "archived-digest"
            else "archived_filename_collision:archived.tab"
        )
    else:
        manifests = []
        for digest in (FIXTURE_SHA, OTHER_SHA):
            entry = {"filename": "table.csv"}
            if collision == "r2-digest":
                key = f"raw/dwp/dwp-tables/2023/{digest}/table.csv"
                entry["storage"] = {
                    "r2": {"provider": "r2", "uri": f"r2://ledger-raw/{key}"}
                }
            else:
                entry["sha256"] = digest
            manifests.append({"kind": "publisher_table", "files": {2023: entry}})
        first, second = manifests
        expected_error = "filename_collision_across_manifests:table.csv"
    for name, manifest in (("first", first), ("second", second)):
        (package / f"manifest_{name}.yaml").write_text(yaml.safe_dump(manifest))
    return expected_error


@pytest.mark.parametrize(
    "collision",
    ["declared-digest", "r2-digest", "archived-digest", "archived-filename"],
)
def test_registration_refuses_unrelated_sibling_collision_before_mutation(
    tmp_path, monkeypatch, collision
):
    package = tmp_path / "package"
    expected_error = _registration_sibling_collision(package, collision)
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    lock_path = _registration_lock_path(package)
    assert not lock_path.exists()
    _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    def unexpected_mutation(*args, **kwargs):
        pytest.fail("existing sibling collision reached a filesystem mutation")

    monkeypatch.setattr(Path, "mkdir", unexpected_mutation)
    monkeypatch.setattr(
        "chronicle.registration._registration_lock", unexpected_mutation
    )
    monkeypatch.setattr(
        "chronicle.registration._atomic_replace_manifest", unexpected_mutation
    )
    with pytest.raises(HashOnlyRegistrationError, match=expected_error):
        _register(package, filename="unrelated.tab", sha256="c" * 64)

    assert not lock_path.exists()
    assert {path.name: path.read_bytes() for path in package.iterdir()} == before


def test_registration_rechecks_sibling_collision_under_lock_before_replacement(
    tmp_path, monkeypatch
):
    package = tmp_path / "package"
    expected_error = _registration_sibling_collision(package, "declared-digest")
    sibling = package / "manifest_second.yaml"
    sibling_bytes = sibling.read_bytes()
    sibling.unlink()
    original = (package / "manifest.yaml").read_bytes()
    _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)
    lock_entries = []

    @contextmanager
    def collision_while_acquiring_lock(output):
        assert output == package
        lock_entries.append(output)
        sibling.write_bytes(sibling_bytes)
        yield

    def unexpected_replacement(*args, **kwargs):
        pytest.fail("sibling collision under the lock reached manifest replacement")

    monkeypatch.setattr(
        "chronicle.registration._registration_lock", collision_while_acquiring_lock
    )
    monkeypatch.setattr(
        "chronicle.registration._atomic_replace_manifest", unexpected_replacement
    )
    with pytest.raises(HashOnlyRegistrationError, match=expected_error):
        _register(package, filename="unrelated.tab", sha256="c" * 64)

    assert lock_entries == [package]
    assert (package / "manifest.yaml").read_bytes() == original
    assert sibling.read_bytes() == sibling_bytes


def _registration_recorded_sibling(defect: str | None = None) -> tuple[dict, str]:
    """A sibling whose effective owners and R2 locators must all be valid."""

    def entry(year, digest=FIXTURE_SHA, filename="table.csv"):
        key = f"raw/dwp/dwp-tables/{year}/{digest}/{filename}"
        return {
            "filename": filename,
            "storage": {
                "r2": {
                    "provider": "r2",
                    "bucket": "ledger-raw",
                    "key": key,
                    "uri": f"r2://ledger-raw/{key}",
                }
            },
        }

    first = entry(2023)
    files = {2023: first}
    expected_error = ""
    if defect == "effective-owner":
        files[2024] = entry(2024, OTHER_SHA)
        expected_error = "identify different bytes"
    elif defect == "malformed-locator":
        first["storage"]["r2"]["uri"] = "not-an-r2-locator"
        expected_error = "storage.r2"
    elif defect == "checksum-mismatch":
        first["sha256"] = OTHER_SHA
        expected_error = "recorded_r2_identity_mismatch"
    elif defect == "filename-mismatch":
        first["filename"] = "renamed.csv"
        expected_error = "recorded_r2_identity_mismatch"
    elif defect == "malformed-archived-locator":
        first["storage"]["previous_r2"] = [
            {"provider": "r2", "uri": "not-an-r2-locator"}
        ]
        expected_error = "storage.previous_r2"
    else:
        assert defect is None
        files[2024] = entry(2024)
        first["storage"]["previous_r2"] = [
            entry(2023, OTHER_SHA, "archived.csv")["storage"]["r2"]
        ]
    return {
        "kind": "publisher_table",
        "source_id": "dwp",
        "package_id": "dwp-tables",
        "files": files,
    }, expected_error


@pytest.mark.parametrize(
    "defect",
    [
        "effective-owner",
        "malformed-locator",
        "checksum-mismatch",
        "filename-mismatch",
        "malformed-archived-locator",
    ],
)
def test_registration_refuses_invalid_recorded_sibling_before_mutation(
    tmp_path, monkeypatch, defect
):
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump({"kind": "microdata_release", "files": {}})
    )
    sibling, expected_error = _registration_recorded_sibling(defect)
    (package / "manifest_tables.yaml").write_text(yaml.safe_dump(sibling))
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    lock_path = _registration_lock_path(package)
    assert not lock_path.exists()
    _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    def unexpected_mutation(*args, **kwargs):
        pytest.fail("invalid recorded sibling reached a filesystem mutation")

    monkeypatch.setattr(Path, "mkdir", unexpected_mutation)
    monkeypatch.setattr(
        "chronicle.registration._registration_lock", unexpected_mutation
    )
    monkeypatch.setattr(
        "chronicle.registration._atomic_replace_manifest", unexpected_mutation
    )
    with pytest.raises(HashOnlyRegistrationError, match=expected_error):
        _register(package, filename="unrelated.tab", sha256="c" * 64)

    assert not lock_path.exists()
    assert {path.name: path.read_bytes() for path in package.iterdir()} == before


@pytest.mark.parametrize(
    "defect",
    [
        "effective-owner",
        "malformed-locator",
        "checksum-mismatch",
        "filename-mismatch",
        "malformed-archived-locator",
    ],
)
def test_registration_rechecks_recorded_sibling_under_lock_before_replacement(
    tmp_path, monkeypatch, defect
):
    package = tmp_path / "package"
    package.mkdir()
    selected = package / "manifest.yaml"
    selected.write_text(yaml.safe_dump({"kind": "microdata_release", "files": {}}))
    original = selected.read_bytes()
    sibling_path = package / "manifest_tables.yaml"
    valid_sibling, _ = _registration_recorded_sibling()
    sibling_path.write_text(yaml.safe_dump(valid_sibling))
    invalid_sibling, expected_error = _registration_recorded_sibling(defect)
    invalid_document = yaml.safe_dump(invalid_sibling)
    _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)
    lock_entries = []

    @contextmanager
    def invalidation_while_acquiring_lock(output):
        assert output == package
        lock_entries.append(output)
        sibling_path.write_text(invalid_document)
        yield

    def unexpected_replacement(*args, **kwargs):
        pytest.fail("invalid recorded sibling under lock reached replacement")

    monkeypatch.setattr(
        "chronicle.registration._registration_lock", invalidation_while_acquiring_lock
    )
    monkeypatch.setattr(
        "chronicle.registration._atomic_replace_manifest", unexpected_replacement
    )
    with pytest.raises(HashOnlyRegistrationError, match=expected_error):
        _register(package, filename="unrelated.tab", sha256="c" * 64)

    assert lock_entries == [package]
    assert selected.read_bytes() == original
    assert sibling_path.read_text() == invalid_document
    assert not _registration_lock_path(package).exists()
    assert {path.name for path in package.iterdir()} == {
        "manifest.yaml",
        "manifest_tables.yaml",
    }


def test_registration_accepts_agreeing_recorded_sibling_owners_and_history(
    tmp_path, monkeypatch
):
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump({"kind": "microdata_release", "files": {}})
    )
    sibling, _ = _registration_recorded_sibling()
    sibling_path = package / "manifest_tables.yaml"
    sibling_path.write_text(yaml.safe_dump(sibling))
    original = sibling_path.read_bytes()
    _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    report = _register(package, filename="unrelated.tab", sha256="c" * 64)

    assert report.filename == "unrelated.tab"
    assert sibling_path.read_bytes() == original


def test_registration_does_not_read_unidentified_sibling_owner_bytes(
    tmp_path, monkeypatch
):
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump({"kind": "microdata_release", "files": {}})
    )
    content = b"public publisher table"
    table_path = package / "table.csv"
    table_path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    sibling, _ = _registration_recorded_sibling()
    locator = sibling["files"][2023]["storage"]["r2"]
    for field in ("key", "uri"):
        locator[field] = locator[field].replace(FIXTURE_SHA, digest)
    sibling["files"][2024] = {"filename": "table.csv"}
    sibling_path = package / "manifest_tables.yaml"
    sibling_path.write_text(yaml.safe_dump(sibling))
    original = sibling_path.read_bytes()
    original_read_bytes = Path.read_bytes
    _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    def refuse_artifact_read(path):
        if path == table_path:
            pytest.fail("hash-only registration read a sibling artifact's bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_artifact_read)
    report = _register(package, filename="unrelated.tab", sha256="c" * 64)

    assert report.filename == "unrelated.tab"
    assert sibling_path.read_bytes() == original
    assert original_read_bytes(table_path) == content


@pytest.mark.parametrize("empty", [False, True], ids=["populated", "empty"])
def test_registration_refuses_publisher_table_list_sibling_before_mutation(
    tmp_path, monkeypatch, empty
):
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump({"kind": "microdata_release", "files": {}})
    )
    sibling, _ = _registration_recorded_sibling()
    sibling["files"][2023] = [] if empty else [sibling["files"][2023]]
    (package / "manifest_tables.yaml").write_text(yaml.safe_dump(sibling))
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    lock_path = _registration_lock_path(package)
    assert not lock_path.exists()
    _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    def unexpected_mutation(*args, **kwargs):
        pytest.fail("publisher-table list sibling reached a filesystem mutation")

    monkeypatch.setattr(Path, "mkdir", unexpected_mutation)
    monkeypatch.setattr(
        "chronicle.registration._registration_lock", unexpected_mutation
    )
    monkeypatch.setattr(
        "chronicle.registration._atomic_replace_manifest", unexpected_mutation
    )
    with pytest.raises(
        HashOnlyRegistrationError,
        match="list_file_spec_requires_microdata_release_kind",
    ):
        _register(package, filename="unrelated.tab", sha256="c" * 64)

    assert not lock_path.exists()
    assert {path.name: path.read_bytes() for path in package.iterdir()} == before


@pytest.mark.parametrize("empty", [False, True], ids=["populated", "empty"])
def test_registration_rechecks_publisher_table_list_sibling_under_lock(
    tmp_path, monkeypatch, empty
):
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text(
        yaml.safe_dump({"kind": "microdata_release", "files": {}})
    )
    sibling_path = package / "manifest_tables.yaml"
    sibling, _ = _registration_recorded_sibling()
    sibling_path.write_text(yaml.safe_dump(sibling))
    sibling["files"][2023] = [] if empty else [sibling["files"][2023]]
    invalid_document = yaml.safe_dump(sibling)
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    before[sibling_path.name] = invalid_document.encode()
    _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)
    lock_entries = []

    @contextmanager
    def invalidation_while_acquiring_lock(output):
        assert output == package
        lock_entries.append(output)
        sibling_path.write_text(invalid_document)
        yield

    def unexpected_replacement(*args, **kwargs):
        pytest.fail("publisher-table list sibling under lock reached replacement")

    monkeypatch.setattr(
        "chronicle.registration._registration_lock", invalidation_while_acquiring_lock
    )
    monkeypatch.setattr(
        "chronicle.registration._atomic_replace_manifest", unexpected_replacement
    )
    with pytest.raises(
        HashOnlyRegistrationError,
        match="list_file_spec_requires_microdata_release_kind",
    ):
        _register(package, filename="unrelated.tab", sha256="c" * 64)

    assert lock_entries == [package]
    assert not _registration_lock_path(package).exists()
    assert {path.name: path.read_bytes() for path in package.iterdir()} == before
