"""Manifest reading contracts shared by every consumer on this branch."""

from __future__ import annotations

import pytest
import yaml


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
