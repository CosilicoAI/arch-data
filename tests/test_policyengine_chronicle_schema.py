"""Tests for the packaged consumer-fact row schemas and their validator.

The packaged schemas are the single source of truth used by artifact builds and
loads. These tests pin them byte-for-byte to ``docs/schemas`` so the two copies
cannot drift, and exercise the validator's precise error reporting.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from chronicle.epoch import Epoch, HASH_DOMAINS, SCHEMA_IDS
from policyengine_chronicle.schema import (
    CONSUMER_FACT_SCHEMA_SHA256,
    CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION,
    consumer_fact_schema,
    normalize_consumer_fact_row_epochs,
    validate_consumer_fact_row_epochs,
    validate_consumer_fact_row,
)

_REPO_ROOT = Path(__file__).parents[1]
_SAMPLE_PATH = _REPO_ROOT / "chronicle" / "fixtures" / "consumer_facts.jsonl"
_V1 = SCHEMA_IDS["consumer_fact"].ledger
_V2 = SCHEMA_IDS["consumer_fact"].chronicle
_SCHEMA_FILES = {
    _V1: "consumer_fact.v1.schema.json",
    _V2: "consumer_fact.v2.schema.json",
}
_FROZEN_SCHEMA_SHA256 = {
    _V1: "72ad3149564f8aab3e9bb6de5ea25950c8e19a3e8402e2cdcfc52d250bc8ee82",
    _V2: "6a42e4a54b9758eaa1219489c318131429a3200fef6205e6700651d46bde068d",
}

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


def _v1_row(index=0):
    """A fixture row restated as a label-free ``ledger.consumer_fact.v1`` row."""
    row = _fixture_row(index)
    row["schema_version"] = _V1
    row.pop("dimension_labels", None)
    row.pop("dimension_value_labels", None)
    row.get("layout", {}).pop("groupby_dimension_label", None)
    return row


@pytest.mark.parametrize("schema_version", [_V1, _V2])
def test_packaged_schema_is_byte_identical_to_docs_schema(schema_version):
    filename = _SCHEMA_FILES[schema_version]
    docs_bytes = (_REPO_ROOT / "docs" / "schemas" / filename).read_bytes()
    packaged_bytes = (
        _REPO_ROOT / "policyengine_chronicle" / "schemas" / filename
    ).read_bytes()

    assert packaged_bytes == docs_bytes
    assert (
        hashlib.sha256(docs_bytes).hexdigest()
        == (_FROZEN_SCHEMA_SHA256[schema_version])
    )
    assert (
        CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION[schema_version]
        == (_FROZEN_SCHEMA_SHA256[schema_version])
    )


def test_emitted_schema_is_the_v2_contract():
    assert CONSUMER_FACT_SCHEMA_SHA256 == _FROZEN_SCHEMA_SHA256[_V2]
    assert dict(CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION) == _FROZEN_SCHEMA_SHA256
    assert consumer_fact_schema() == consumer_fact_schema(_V2)


def test_consumer_fact_schema_is_the_v1_contract_row():
    schema = consumer_fact_schema(_V1)

    assert schema["title"] == "Ledger consumer fact contract row"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == "ledger.consumer_fact.v1"


def test_v2_schema_is_v1_plus_optional_labels():
    v1 = copy.deepcopy(consumer_fact_schema(_V1))
    v2 = copy.deepcopy(consumer_fact_schema(_V2))

    assert v2["title"] == "Chronicle consumer fact contract row"
    assert v2["properties"]["schema_version"] == {"const": "chronicle.consumer_fact.v2"}
    assert v2["required"] == v1["required"]
    added = set(v2["properties"]) - set(v1["properties"])
    assert added == {"dimension_labels", "dimension_value_labels"}
    assert v2["properties"]["layout"]["properties"] == {
        "groupby_dimension_label": {
            "$ref": "#/$defs/label",
            "description": (
                "Label of layout.groupby_dimension; equals dimension_labels for "
                "that id."
            ),
        }
    }
    for schema in (v1, v2):
        for key in ("$id", "title", "description"):
            schema.pop(key, None)
        schema["properties"].pop("schema_version")
        schema["properties"].pop("dimension_labels", None)
        schema["properties"].pop("dimension_value_labels", None)
        schema["properties"]["layout"].pop("properties", None)
    assert v2["$defs"].pop("label") == {"type": "string", "pattern": ".*\\S.*"}
    assert v2 == v1


def test_valid_fixture_rows_pass_validation():
    rows = [
        json.loads(line)
        for line in _SAMPLE_PATH.read_text().splitlines()
        if line.strip()
    ]

    assert len(rows) == 3
    assert {row["schema_version"] for row in rows} == {_V2}
    for line_number, row in enumerate(rows, start=1):
        validate_consumer_fact_row(row, line_number, _SAMPLE_PATH)


def test_a_v1_row_validates_as_v1_and_restamped_as_v2():
    row = _v1_row()

    validate_consumer_fact_row(row, 1, _SAMPLE_PATH)
    validate_consumer_fact_row({**row, "schema_version": _V2}, 1, _SAMPLE_PATH)


def test_v2_rows_carry_labels_and_v1_rows_may_not():
    labelled = {
        **_v1_row(),
        "schema_version": _V2,
        "dimension_labels": {"filing_status": "Filing status"},
        "dimension_value_labels": {"filing_status": {"all": "All returns"}},
    }
    labelled["layout"] = {**labelled["layout"], "groupby_dimension_label": "AGI"}

    validate_consumer_fact_row(labelled, 1, "labelled.jsonl")
    with pytest.raises(ValueError, match="dimension_labels"):
        validate_consumer_fact_row({**labelled, "schema_version": _V1}, 1, "v1.jsonl")


@pytest.mark.parametrize(
    ("field_name", "value", "location"),
    [
        ("dimension_labels", {"filing_status": " "}, "dimension_labels/filing_status"),
        ("dimension_labels", {"filing_status": 1}, "dimension_labels/filing_status"),
        (
            "dimension_value_labels",
            {"filing_status": {"all": ""}},
            "dimension_value_labels/filing_status/all",
        ),
        ("dimension_value_labels", {"filing_status": "all"}, "dimension_value_labels"),
    ],
)
def test_v2_labels_must_be_nonempty_strings(field_name, value, location):
    row = {**_v1_row(), "schema_version": _V2, field_name: value}

    with pytest.raises(ValueError) as excinfo:
        validate_consumer_fact_row(row, 3, "bad.jsonl")

    assert location in str(excinfo.value)


def test_all_chronicle_epoch_identifiers_pass_without_mutating_row():
    row = _chronicle_epoch_row(_fixture_row(1))
    row["lineage"]["source_row_keys"] = [
        HASH_DOMAINS["source_row"].chronicle + ":" + "a" * 24
    ]
    original = json.loads(json.dumps(row))

    validate_consumer_fact_row(row, 2, _SAMPLE_PATH)

    assert row == original
    normalized = normalize_consumer_fact_row_epochs(row, 2, _SAMPLE_PATH)
    # The row contract is not an epoch alias any more: v2 stays v2.
    assert normalized["schema_version"] == _V2
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
