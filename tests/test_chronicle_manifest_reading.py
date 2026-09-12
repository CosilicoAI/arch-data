"""Manifests are read strictly and every entry field the sweeps rely on is validated.

A manifest is the record the byte boundary is decided from, so the reader
refuses what it cannot represent faithfully (a document whose duplicate keys
YAML would silently collapse), the comparison key for filenames covers every
spelling a filesystem folds together (case and Unicode normalisation), and
``validate_file_entry`` reports the shapes the commands used to act on
silently: a nameless or malformed entry, a misspelled field, a hash-only
entry carrying a storage block or a ``chronicle_fetch`` attestation, a
consumer attestation signed by Chronicle, and evidence whose URL has no host.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import pytest
import yaml

from chronicle.artifacts import (
    MalformedManifestError,
    _read_manifest,
    inventory_source_artifacts,
    publish_source_artifacts,
)
from chronicle.licences import licence_evidence_errors
from chronicle.registration import (
    HashOnlyRegistrationError,
    ManifestAccessError,
    filename_key,
    validate_file_entry,
    validate_manifest_files,
)
from chronicle.source_package import SourceArtifactSpec
from tests.test_chronicle_microdata_registration import (
    ATTESTED,
    EVIDENCE,
    FIXTURE_SHA,
    LICENSED_BYTES,
    OTHER_SHA,
    PINNED,
    PUBLIC_SHA,
    _attested_entry,
    _fetch_table,
    _forbid_uploads,
    _public_release_entry,
    _refuse_read,
    _register,
)


NFC_NAME = unicodedata.normalize("NFC", "adúlt.tab")
NFD_NAME = unicodedata.normalize("NFD", "adúlt.tab")
LICENSED_SHA = hashlib.sha256(LICENSED_BYTES).hexdigest()


def _write(path: Path, text: str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path.read_bytes()


def _release_manifest(**fields: object) -> dict:
    payload = {
        "source_id": "dwp",
        "package_id": "dwp-frs-2023-24",
        "kind": "microdata_release",
        "files": {},
    }
    payload.update(fields)
    return payload


# --------------------------------------------------------------------------
# Filename keys fold Unicode normalisation as well as case
# --------------------------------------------------------------------------


def test_filename_key_folds_unicode_normalisation():
    assert NFC_NAME != NFD_NAME
    assert filename_key(NFC_NAME) == filename_key(NFD_NAME)
    assert filename_key(NFD_NAME.upper()) == filename_key(NFC_NAME)


def test_a_normalisation_alias_is_a_filename_collision():
    manifest = _release_manifest(
        files={
            2023: [
                _attested_entry(filename=NFC_NAME),
                {**_public_release_entry(filename=NFD_NAME)},
            ]
        }
    )
    codes = validate_manifest_files(manifest)
    assert any(code.startswith("filename_collision:") for code in codes), codes


def test_fetch_refuses_a_normalisation_alias_of_a_hash_only_name(tmp_path, monkeypatch):
    package = tmp_path / "db" / "data" / "dwp" / "frs_2023_24"
    _register(package, filename=NFC_NAME)
    original = (package / "manifest.yaml").read_bytes()
    reads = _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)

    with pytest.raises(ManifestAccessError, match="access='licensed'"):
        _fetch_table(
            tmp_path / NFD_NAME,
            package,
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            filename=NFD_NAME,
            upload_r2=True,
        )

    assert reads == []
    assert (package / "manifest.yaml").read_bytes() == original
    assert sorted(path.name for path in package.iterdir()) == ["manifest.yaml"]


# --------------------------------------------------------------------------
# Entry validation vocabulary
# --------------------------------------------------------------------------


def test_a_nameless_or_malformed_entry_is_an_error_everywhere():
    nameless = _attested_entry(filename=None)
    assert "missing_filename" in validate_file_entry(
        nameless, kind="microdata_release", manifest={}, local_file_exists=False
    )
    assert "missing_filename" in validate_file_entry(
        {"source_url": "x"},
        kind="publisher_table",
        manifest={},
        local_file_exists=False,
    )
    for malformed in ("adult.tab", 3, None, ["adult.tab"]):
        assert validate_file_entry(
            malformed, kind="publisher_table", manifest={}, local_file_exists=False
        ) == ("malformed_file_spec",)


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("Access", "misspelled_field:Access"),
        ("ACCESS", "misspelled_field:ACCESS"),
        ("Sha256", "misspelled_field:Sha256"),
        ("Licence", "misspelled_field:Licence"),
        ("Storage", "misspelled_field:Storage"),
        ("Hash_Source", "misspelled_field:Hash_Source"),
        (" access", "misspelled_field: access"),
    ],
)
def test_a_misspelled_field_is_reported_not_ignored(field, code):
    entry = _attested_entry()
    entry[field] = entry.pop("access") if field.strip().lower() == "access" else "x"
    codes = validate_file_entry(
        entry, kind="microdata_release", manifest={}, local_file_exists=False
    )
    assert code in codes, codes


def test_an_unknown_entry_field_is_reported_not_ignored():
    """An ordinary typo must not turn gated bytes into inferred public bytes."""
    codes = validate_file_entry(
        {
            "filename": "adult.tab",
            "acess": "licensed",
        },
        kind="publisher_table",
        manifest={},
        local_file_exists=False,
    )

    assert "unknown_field:acess" in codes, codes


@pytest.mark.parametrize(
    "storage",
    [
        {"r2": "r2://ledger-raw/raw/x"},
        {"previous_r2": "r2://ledger-raw/raw/x"},
        {"r2": {"uri": "r2://ledger-raw/raw/x"}},
        {"previous_r2": [{"uri": "r2://ledger-raw/raw/x"}]},
        {"anything": 1},
        "r2://ledger-raw/raw/x",
        ["r2://ledger-raw/raw/x"],
    ],
)
def test_a_hash_only_entry_may_carry_no_storage_block_of_any_shape(storage):
    codes = validate_file_entry(
        _attested_entry(storage=storage),
        kind="microdata_release",
        manifest={},
        local_file_exists=False,
    )
    assert "storage_for_hash_only_entry" in codes, codes


def test_a_hash_only_entry_is_never_a_chronicle_fetch():
    entry = _attested_entry(
        hash_source="chronicle_fetch",
        attested_by="chronicle",
        verified_at="2026-09-03",
        attestation_evidence=None,
    )
    codes = validate_file_entry(
        entry, kind="microdata_release", manifest={}, local_file_exists=False
    )
    assert "hash_source_not_allowed_for_hash_only_entry:chronicle_fetch" in codes


@pytest.mark.parametrize("attestation", [ATTESTED, PINNED])
def test_a_consumer_attestation_is_not_signed_by_chronicle(attestation):
    entry = _attested_entry(**{**attestation, "attested_by": "chronicle"})
    codes = validate_file_entry(
        entry, kind="microdata_release", manifest={}, local_file_exists=False
    )
    assert "attested_by_chronicle_for_consumer_hash_source" in codes, codes


def test_register_refuses_a_consumer_attestation_signed_by_chronicle(tmp_path):
    package = tmp_path / "pkg"
    with pytest.raises(HashOnlyRegistrationError, match="chronicle"):
        _register(package, attested_by="chronicle")
    with pytest.raises(HashOnlyRegistrationError, match="chronicle"):
        _register(package, **{**PINNED, "attested_by": "Chronicle"})
    assert not package.exists()


def test_release_public_digests_are_keyed_per_vintage():
    """Public release bytes are staged content-addressed outside the tree, so
    one filename may hold different bytes under different vintages; under one
    vintage a revision is history, not a second entry."""
    per_vintage = _release_manifest(
        files={
            2022: [_public_release_entry()],
            2023: [
                _public_release_entry(
                    sha256=OTHER_SHA,
                    vintage="2023",
                    licence_evidence={
                        **EVIDENCE,
                        "licence": "US-Government-Work",
                        "sha256": OTHER_SHA,
                    },
                )
            ],
        }
    )
    assert validate_manifest_files(per_vintage) == ()

    same_vintage = _release_manifest(
        files={
            2022: [
                _public_release_entry(),
                _public_release_entry(
                    sha256=OTHER_SHA,
                    licence_evidence={
                        **EVIDENCE,
                        "licence": "US-Government-Work",
                        "sha256": OTHER_SHA,
                    },
                ),
            ]
        }
    )
    assert "filename_collision:csv_hus.zip" in validate_manifest_files(same_vintage)

    table = {
        "source_id": "irs_soi",
        "package_id": "soi-table-1-2",
        "kind": "publisher_table",
        "files": {
            2022: {"filename": "t.xlsx", "sha256": PUBLIC_SHA},
            2023: {"filename": "t.xlsx", "sha256": OTHER_SHA},
        },
    }
    assert "filename_collision:t.xlsx" in validate_manifest_files(table)


@pytest.mark.parametrize(
    "url",
    [
        "https://",
        "http://",
        "https:// evidence.example",
        " https://evidence.example",
        "https://evidence.example/a b",
        "ftp://evidence.example/x",
        "evidence.example/x",
    ],
)
def test_an_evidence_url_needs_a_scheme_and_a_host_and_no_whitespace(url):
    evidence = {
        **EVIDENCE,
        "licence": "US-Government-Work",
        "sha256": PUBLIC_SHA,
        "url": url,
    }
    assert "licence_evidence_url_not_durable" in licence_evidence_errors(
        evidence, licence="US-Government-Work", sha256=PUBLIC_SHA
    )


def test_an_evidence_url_with_a_host_is_durable():
    evidence = {**EVIDENCE, "licence": "US-Government-Work", "sha256": PUBLIC_SHA}
    assert (
        licence_evidence_errors(
            evidence, licence="US-Government-Work", sha256=PUBLIC_SHA
        )
        == []
    )


# --------------------------------------------------------------------------
# register-artifact validates the manifest it records into
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "existing",
    [
        {"filename": "other.tab", "access": "Public", "sha256": OTHER_SHA},
        {"filename": "adult.tab", "access": "open", "sha256": FIXTURE_SHA},
        {**_attested_entry(filename="other.tab", storage="r2://ledger-raw/x")},
        {**_attested_entry(filename="other.tab", hash_source="chronicle_fetch")},
    ],
    ids=[
        "title-case-access",
        "unknown-access-same-name",
        "string-storage",
        "chronicle-fetch",
    ],
)
def test_register_refuses_an_invalid_existing_entry_instead_of_replacing_it(
    tmp_path, existing
):
    package = tmp_path / "pkg"
    original = _write(
        package / "manifest.yaml",
        yaml.safe_dump(_release_manifest(files={2023: [existing]}), sort_keys=False),
    )

    with pytest.raises(HashOnlyRegistrationError, match="not a valid"):
        _register(package)

    assert (package / "manifest.yaml").read_bytes() == original


def test_register_reports_unreadable_yaml_as_a_refusal(tmp_path):
    package = tmp_path / "pkg"
    original = _write(package / "manifest.yaml", "files: [\n")

    with pytest.raises(HashOnlyRegistrationError, match="not valid YAML"):
        _register(package)

    assert (package / "manifest.yaml").read_bytes() == original


# --------------------------------------------------------------------------
# Duplicate keys are a malformed manifest, never silently collapsed
# --------------------------------------------------------------------------


DUPLICATE_DOCUMENTS = {
    "duplicate-vintage": (
        "source_id: irs_soi\npackage_id: soi-table-1-2\nkind: publisher_table\n"
        "files:\n  2023:\n    filename: a.xlsx\n  2023:\n    filename: b.xlsx\n"
    ),
    "int-alias-vintage": (
        "source_id: irs_soi\npackage_id: soi-table-1-2\nkind: publisher_table\n"
        "files:\n  2023:\n    filename: a.xlsx\n  2_023:\n    filename: b.xlsx\n"
    ),
    "float-alias-vintage": (
        "source_id: irs_soi\npackage_id: soi-table-1-2\nkind: publisher_table\n"
        "files:\n  2023:\n    filename: a.xlsx\n  2023.0:\n    filename: b.xlsx\n"
    ),
    "duplicate-files-block": (
        "source_id: irs_soi\npackage_id: soi-table-1-2\nkind: publisher_table\n"
        "files:\n  2023:\n    filename: a.xlsx\nfiles:\n  2024:\n    filename: b.xlsx\n"
    ),
    "duplicate-entry-field": (
        "source_id: irs_soi\npackage_id: soi-table-1-2\nkind: publisher_table\n"
        "files:\n  2023:\n    filename: a.xlsx\n    access: public\n    access: licensed\n"
    ),
}


@pytest.mark.parametrize(
    "document", DUPLICATE_DOCUMENTS.values(), ids=DUPLICATE_DOCUMENTS
)
def test_a_manifest_with_duplicate_keys_is_refused_by_every_reader(
    tmp_path, monkeypatch, document
):
    package = tmp_path / "db" / "data" / "irs_soi" / "table_1_2"
    original = _write(package / "manifest.yaml", document)
    (package / "a.xlsx").write_bytes(b"a")
    (package / "b.xlsx").write_bytes(b"b")

    with pytest.raises(MalformedManifestError, match="duplicate key"):
        _read_manifest(package / "manifest.yaml")

    reads = _refuse_read(monkeypatch)
    _forbid_uploads(monkeypatch)
    with pytest.raises(MalformedManifestError, match="duplicate key"):
        _fetch_table(tmp_path / "c.xlsx", package, year=2025, upload_r2=True)
    assert reads == []

    inventory = inventory_source_artifacts(tmp_path / "db" / "data")
    assert not inventory.valid
    assert inventory.entries == ()
    assert any("duplicate key" in error for error in inventory.errors)

    published = publish_source_artifacts(tmp_path / "db" / "data")
    assert not published.valid
    assert published.entries == ()
    assert any("duplicate key" in error for error in published.errors)

    with pytest.raises(HashOnlyRegistrationError, match="duplicate key"):
        _register(package, source_id="irs_soi", package_id="soi-table-1-2")

    assert (package / "manifest.yaml").read_bytes() == original


def test_the_byte_reader_refuses_a_manifest_with_duplicate_keys(tmp_path, monkeypatch):
    import sys
    import uuid

    package_name = f"chronicle_test_{uuid.uuid4().hex}"
    resource_dir = tmp_path / "pkgroot" / package_name / "data" / "irs_soi" / "t"
    _write(resource_dir / "manifest.yaml", DUPLICATE_DOCUMENTS["duplicate-vintage"])
    (resource_dir / "a.xlsx").write_bytes(b"a")
    (resource_dir / "b.xlsx").write_bytes(b"b")
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    spec = SourceArtifactSpec(
        source_name="irs_soi",
        source_table="Table 1.2",
        resource_package=package_name,
        resource_directory="data/irs_soi/t",
        manifest="manifest.yaml",
        vintage="2023",
        extracted_at="2026-09-02",
        extraction_method="none",
        parser="delimited_text_full_rows",
        artifact_year=2023,
    )

    with pytest.raises(ValueError, match="duplicate key"):
        spec.assert_parseable(2023)


def test_strict_loader_keeps_yaml_merge_overrides_and_refuses_explicit_duplicates():
    """A ``<<: *defaults`` merge followed by an explicit override is valid YAML
    (the explicit key wins); only two explicit spellings of one key are a
    duplicate the loader must refuse."""
    from chronicle.registration import load_manifest_document

    merged = load_manifest_document(
        "defaults: &defaults\n"
        "  source_url: https://publisher.test/a\n"
        "  filename: table.csv\n"
        "files:\n"
        "  2024:\n"
        "    <<: *defaults\n"
        "    source_url: https://publisher.test/b\n"
        "    sha256: " + "ab" * 32 + "\n"
    )

    assert merged["files"][2024]["source_url"] == "https://publisher.test/b"
    assert merged["files"][2024]["filename"] == "table.csv"

    with pytest.raises(yaml.YAMLError, match="duplicate key 'source_url'"):
        load_manifest_document(
            "defaults: &defaults\n"
            "  filename: table.csv\n"
            "files:\n"
            "  2024:\n"
            "    <<: *defaults\n"
            "    source_url: https://publisher.test/a\n"
            "    source_url: https://publisher.test/b\n"
        )


def test_strict_loader_keeps_merge_sequence_precedence_for_artifact_selection():
    """``<<: [first, second]`` selects ``first``'s values: earlier mappings in a
    merge sequence take precedence over later ones (YAML merge-key semantics,
    and what ``yaml.safe_load`` does), while an explicit key in the entry still
    overrides every merged source. The selected ``filename``/``sha256`` pair is
    what fetch, publish, and the source-package reader use to pick bytes, so
    reversing the precedence silently selects a different artifact."""
    from chronicle.registration import load_manifest_document

    document = (
        "first: &first\n"
        "  filename: first.csv\n"
        "  sha256: " + "aa" * 32 + "\n"
        "  source_url: https://publisher.test/first\n"
        "second: &second\n"
        "  filename: second.csv\n"
        "  sha256: " + "bb" * 32 + "\n"
        "  source_url: https://publisher.test/second\n"
        "  licence: second-only\n"
        "files:\n"
        "  2024:\n"
        "    <<: [*first, *second]\n"
        "  2023:\n"
        "    <<: [*first, *second]\n"
        "    filename: explicit.csv\n"
        "  2022:\n"
        "    <<: [{filename: inline-first.csv}, {filename: inline-second.csv}]\n"
        "  2021:\n"
        "    <<:\n"
        "      - <<: [*second, *first]\n"
        "        source_url: https://publisher.test/nested\n"
        "      - *first\n"
    )

    strict = load_manifest_document(document)
    reference = yaml.safe_load(document)
    assert strict["files"] == reference["files"]

    selected = strict["files"][2024]
    assert selected["filename"] == "first.csv"
    assert selected["sha256"] == "aa" * 32
    assert selected["source_url"] == "https://publisher.test/first"
    # Keys only the later source carries are still merged in.
    assert selected["licence"] == "second-only"
    # An explicit key beats every merged source; the rest still follow first.
    assert strict["files"][2023]["filename"] == "explicit.csv"
    assert strict["files"][2023]["sha256"] == "aa" * 32
    assert strict["files"][2022]["filename"] == "inline-first.csv"
    # Nested: the first sequence entry is itself a merge whose own explicit
    # key wins inside it, and whose [second, first] order selects second.
    nested = strict["files"][2021]
    assert nested["filename"] == "second.csv"
    assert nested["source_url"] == "https://publisher.test/nested"


def test_strict_loader_refuses_recursive_merges_as_yaml_errors():
    """A mapping that merges itself (directly or through a nested merge) has
    no expansion. The loader must refuse it with a ``yaml.YAMLError`` that the
    manifest-reading commands already handle, never a ``RecursionError``."""
    from chronicle.registration import load_manifest_document

    for document in (
        # Direct self-merge.
        "files: &f\n  2024:\n    filename: table.csv\n  <<: *f\n",
        # Self-merge through a nested merge sequence.
        "files: &f\n  2024:\n    filename: table.csv\n  <<:\n    - {filename: other.csv}\n    - *f\n",
        # Self-merge through a merged mapping's own merge.
        "a: &a\n  filename: table.csv\n  <<: {<<: *a}\n",
    ):
        with pytest.raises(yaml.YAMLError, match="recursive"):
            load_manifest_document(document)

    # A merge that only *repeats* a source is not a cycle.
    repeated = load_manifest_document(
        "d: &d\n  filename: table.csv\nfiles:\n  2024:\n    <<: [*d, *d]\n"
    )
    assert repeated["files"][2024] == {"filename": "table.csv"}


def test_strict_loader_does_not_mutate_anchored_entries_when_merged_later():
    """Constructing a later merge of an anchored entry must not turn that
    entry's inherited keys into 'explicit' ones: PyYAML constructs lazily and
    flattens merged nodes in place."""
    from chronicle.registration import load_manifest_document

    document = load_manifest_document(
        "defaults: &d\n"
        "  source_url: old\n"
        "files:\n"
        "  2024: &e\n"
        "    <<: *d\n"
        "    source_url: new\n"
        "latest:\n"
        "  <<: *e\n"
    )

    assert document["files"][2024]["source_url"] == "new"
    assert document["latest"]["source_url"] == "new"


def test_strict_loader_validates_duplicate_keys_inside_inline_merges():
    """A mapping reached through ``<<`` is still a mapping the document
    spells out; duplicate keys inside it must be refused, not collapsed."""
    from chronicle.registration import load_manifest_document

    with pytest.raises(yaml.YAMLError, match="duplicate key 'files'"):
        load_manifest_document(
            "<<: {files: {2023: {filename: old.csv}}, files: {2024: {filename: new.csv}}}\n"
        )
