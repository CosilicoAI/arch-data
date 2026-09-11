"""Tests for the packaged consumer-fact row schema and its validator.

The packaged schema is the single source of truth used by artifact builds and
loads. These tests pin it byte-for-byte to ``docs/schemas`` so the two copies
cannot drift, and exercise the validator's precise error reporting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chronicle.epoch import Epoch, HASH_DOMAINS, SCHEMA_IDS
from policyengine_chronicle.schema import (
    CONSUMER_FACT_SCHEMA_SHA256,
    consumer_fact_schema,
    normalize_consumer_fact_row_epochs,
    validate_consumer_fact_row_epochs,
    validate_consumer_fact_row,
)

_REPO_ROOT = Path(__file__).parents[1]
_DOCS_SCHEMA_PATH = _REPO_ROOT / "docs" / "schemas" / "consumer_fact.v1.schema.json"
_PACKAGED_SCHEMA_PATH = (
    _REPO_ROOT / "policyengine_chronicle" / "schemas" / "consumer_fact.v1.schema.json"
)
_SAMPLE_PATH = _REPO_ROOT / "chronicle" / "fixtures" / "consumer_facts.jsonl"
_FROZEN_SCHEMA_SHA256 = (
    "72ad3149564f8aab3e9bb6de5ea25950c8e19a3e8402e2cdcfc52d250bc8ee82"
)

_TOP_LEVEL_KEY_DOMAINS = {
    "aggregate_fact_key": "aggregate_fact",
    "semantic_fact_key": "semantic_fact",
    "legacy_fact_key": "fact",
    "source_release_key": "source_release",
    "source_series_key": "source_series",
    "observed_measure_key": "observed_measure",
    "dimension_set_key": "dimension_set",
    "universe_constraint_set_key": "universe_constraint_set",
}


def _fixture_row(index=0):
    return json.loads(_SAMPLE_PATH.read_text().splitlines()[index])


def _chronicle_epoch_row(row):
    row["schema_version"] = SCHEMA_IDS["consumer_fact"].chronicle
    for field_name, domain_name in _TOP_LEVEL_KEY_DOMAINS.items():
        row[field_name] = HASH_DOMAINS[domain_name].key_for_epoch(
            row[field_name], Epoch.CHRONICLE
        )
    alignment = row.get("concept_alignment")
    if alignment is not None:
        alignment["concept_alignment_key"] = HASH_DOMAINS[
            "concept_alignment"
        ].key_for_epoch(alignment["concept_alignment_key"], Epoch.CHRONICLE)
    lineage = row["lineage"]
    for field_name, domain_name in (
        ("source_cell_keys", "source_cell"),
        ("source_row_keys", "source_row"),
    ):
        lineage[field_name] = [
            HASH_DOMAINS[domain_name].key_for_epoch(key, Epoch.CHRONICLE)
            for key in lineage.get(field_name, [])
        ]
    return row


def test_packaged_schema_is_byte_identical_to_docs_schema():
    docs_bytes = _DOCS_SCHEMA_PATH.read_bytes()
    packaged_bytes = _PACKAGED_SCHEMA_PATH.read_bytes()

    assert packaged_bytes == docs_bytes
    assert hashlib.sha256(docs_bytes).hexdigest() == _FROZEN_SCHEMA_SHA256
    assert hashlib.sha256(packaged_bytes).hexdigest() == CONSUMER_FACT_SCHEMA_SHA256
    assert CONSUMER_FACT_SCHEMA_SHA256 == _FROZEN_SCHEMA_SHA256


def test_consumer_fact_schema_is_the_v1_contract_row():
    schema = consumer_fact_schema()

    assert schema["title"] == "Ledger consumer fact contract row"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == "ledger.consumer_fact.v1"


def test_valid_fixture_rows_pass_validation():
    rows = [
        json.loads(line)
        for line in _SAMPLE_PATH.read_text().splitlines()
        if line.strip()
    ]

    assert len(rows) == 3
    for line_number, row in enumerate(rows, start=1):
        validate_consumer_fact_row(row, line_number, _SAMPLE_PATH)


def test_all_chronicle_epoch_identifiers_pass_without_mutating_row():
    row = _chronicle_epoch_row(_fixture_row(1))
    row["lineage"]["source_row_keys"] = [
        HASH_DOMAINS["source_row"].chronicle + ":" + "a" * 24
    ]
    original = json.loads(json.dumps(row))

    validate_consumer_fact_row(row, 2, _SAMPLE_PATH)

    assert row == original
    normalized = normalize_consumer_fact_row_epochs(row, 2, _SAMPLE_PATH)
    assert normalized["schema_version"] == SCHEMA_IDS["consumer_fact"].ledger
    for field_name, domain_name in _TOP_LEVEL_KEY_DOMAINS.items():
        assert normalized[field_name].startswith(HASH_DOMAINS[domain_name].ledger + ":")
    assert normalized["concept_alignment"]["concept_alignment_key"].startswith(
        HASH_DOMAINS["concept_alignment"].ledger + ":"
    )
    assert normalized["lineage"]["source_cell_keys"][0].startswith(
        HASH_DOMAINS["source_cell"].ledger + ":"
    )
    assert normalized["lineage"]["source_row_keys"][0].startswith(
        HASH_DOMAINS["source_row"].ledger + ":"
    )


def test_mixed_epoch_identifiers_pass_validation():
    row = _fixture_row(1)
    row["schema_version"] = SCHEMA_IDS["consumer_fact"].chronicle
    row["aggregate_fact_key"] = HASH_DOMAINS["aggregate_fact"].key_for_epoch(
        row["aggregate_fact_key"], Epoch.CHRONICLE
    )
    row["concept_alignment"]["concept_alignment_key"] = HASH_DOMAINS[
        "concept_alignment"
    ].key_for_epoch(row["concept_alignment"]["concept_alignment_key"], Epoch.CHRONICLE)
    row["lineage"]["source_cell_keys"][0] = HASH_DOMAINS["source_cell"].key_for_epoch(
        row["lineage"]["source_cell_keys"][0], Epoch.CHRONICLE
    )
    row["lineage"]["source_row_keys"] = [
        HASH_DOMAINS["source_row"].ledger + ":" + "b" * 24
    ]

    validate_consumer_fact_row(row, 2, _SAMPLE_PATH)


@pytest.mark.parametrize(
    ("mutate", "ledger_form", "chronicle_form"),
    [
        (
            lambda row: row.__setitem__("schema_version", "future.consumer_fact.v9"),
            SCHEMA_IDS["consumer_fact"].ledger,
            SCHEMA_IDS["consumer_fact"].chronicle,
        ),
        (
            lambda row: row.__setitem__(
                "aggregate_fact_key", "future.aggregate_fact.v9:" + "0" * 24
            ),
            HASH_DOMAINS["aggregate_fact"].ledger,
            HASH_DOMAINS["aggregate_fact"].chronicle,
        ),
        (
            lambda row: row["concept_alignment"].__setitem__(
                "concept_alignment_key",
                "future.concept_alignment.v9:" + "0" * 24,
            ),
            HASH_DOMAINS["concept_alignment"].ledger,
            HASH_DOMAINS["concept_alignment"].chronicle,
        ),
        (
            lambda row: row["lineage"]["source_cell_keys"].__setitem__(
                0, "future.source_cell.v9:" + "0" * 24
            ),
            HASH_DOMAINS["source_cell"].ledger,
            HASH_DOMAINS["source_cell"].chronicle,
        ),
        (
            lambda row: row["lineage"].__setitem__(
                "source_row_keys", ["future.source_row.v9:" + "0" * 24]
            ),
            HASH_DOMAINS["source_row"].ledger,
            HASH_DOMAINS["source_row"].chronicle,
        ),
    ],
)
def test_unknown_epoch_identifier_names_both_accepted_forms(
    mutate,
    ledger_form,
    chronicle_form,
):
    row = _fixture_row(1)
    mutate(row)

    with pytest.raises(ValueError) as excinfo:
        validate_consumer_fact_row(row, 7, "mixed.jsonl")

    message = str(excinfo.value)
    assert "row 7 of mixed.jsonl" in message
    assert ledger_form in message
    assert chronicle_form in message


def test_missing_nested_required_field_names_field_and_location():
    row = json.loads(_SAMPLE_PATH.read_text().splitlines()[0])
    del row["observed_measure"]["unit"]

    with pytest.raises(ValueError) as excinfo:
        validate_consumer_fact_row(row, 4, "sample.jsonl")

    message = str(excinfo.value)
    assert "row 4 of sample.jsonl" in message
    assert "observed_measure" in message
    assert "unit" in message


def test_unknown_extra_field_is_rejected():
    row = json.loads(_SAMPLE_PATH.read_text().splitlines()[0])
    row["surprise_field"] = "unexpected"

    with pytest.raises(ValueError, match="surprise_field"):
        validate_consumer_fact_row(row, 1, "sample.jsonl")


def test_validate_only_epoch_check_matches_normalizer_and_never_copies():
    row = _chronicle_epoch_row(_fixture_row(1))
    original = json.loads(json.dumps(row))
    assert validate_consumer_fact_row_epochs(row, 2, _SAMPLE_PATH) is None
    assert row == original
    bad = _fixture_row(1)
    bad["lineage"]["source_cell_keys"] = ["bogus.domain.v9:" + "a" * 24]
    with pytest.raises(ValueError) as normalizer_error:
        normalize_consumer_fact_row_epochs(bad, 3, _SAMPLE_PATH)
    with pytest.raises(ValueError) as validator_error:
        validate_consumer_fact_row_epochs(bad, 3, _SAMPLE_PATH)
    assert str(validator_error.value) == str(normalizer_error.value)
    assert bad["lineage"]["source_cell_keys"] == ["bogus.domain.v9:" + "a" * 24]
