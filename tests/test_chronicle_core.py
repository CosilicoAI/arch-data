"""Tests for canonical Chronicle aggregate facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chronicle.core import (
    Aggregation,
    EntityDimension,
    GeographyDimension,
    Measure,
    PeriodDimension,
    SourceProvenance,
    SourceRecordLayout,
    AggregateFact,
    build_label,
    build_fact_key,
    validate_fact,
    validate_facts,
)
from chronicle.epoch import EMIT_EPOCH, HASH_DOMAINS, SCHEMA_IDS, Epoch
from chronicle.harness import main as harness_main
from chronicle.sources.cells import (
    SourceArtifactMetadata,
    SourceCell,
    build_source_cell_key,
)
from chronicle.sources.rows import (
    SourceColumn,
    SourceRow,
    SourceRowValue,
    build_source_column_key,
    build_source_row_key,
    build_source_row_value_key,
)
from chronicle.store import save_facts_jsonl


def _fact(**overrides):
    fact = AggregateFact(
        value=1000,
        period=PeriodDimension(type="tax_year", value=2023),
        geography=GeographyDimension(
            level="country",
            id="0100000US",
            vintage="2020_census",
            name="United States",
        ),
        entity=EntityDimension(name="tax_unit", role="filing_unit"),
        measure=Measure(concept="irs_soi.adjusted_gross_income", unit="usd"),
        aggregation=Aggregation(method="sum"),
        provenance_class="administrative",
        filters={"filing_status": "all"},
        source=SourceProvenance(
            source_name="irs_soi",
            source_table="Publication 1304 Table 1.1",
            source_file="23in11si.xls",
            url="https://www.irs.gov/statistics/soi-tax-stats",
            vintage="tax_year_2023",
            extracted_at="2026-05-04",
            extraction_method="fixture hand entry",
            method_notes="Fixture value for schema tests.",
        ),
        label="United States tax year 2023 sum adjusted gross income",
    )
    return AggregateFact(**{**fact.__dict__, **overrides})


def _source_artifact() -> SourceArtifactMetadata:
    return SourceArtifactMetadata(
        source_name="test_publisher",
        source_table="table",
        source_file="table.csv",
        url="https://example.test/table.csv",
        vintage="2026",
        sha256="a" * 64,
        size_bytes=10,
        extracted_at="2026-09-02",
        extraction_method="test",
    )


def test_valid_fact_passes_validation():
    assert validate_fact(_fact()) == ()
    assert validate_facts([_fact()]).valid


def test_academic_year_period_type_is_valid():
    fact = _fact(period=PeriodDimension(type="academic_year", value=2024))
    assert validate_fact(fact) == ()


@pytest.mark.parametrize(
    ("period_type", "value"),
    (("quarter", "2024-Q1"), ("week", "2026-W36")),
)
def test_subannual_period_types_are_valid(period_type, value):
    fact = _fact(period=PeriodDimension(type=period_type, value=value))
    assert validate_fact(fact) == ()


def test_unknown_period_type_is_rejected():
    fact = _fact(period=PeriodDimension(type="school_year", value=2024))
    assert any(issue.code == "malformed_period" for issue in validate_fact(fact))


def test_quantile_aggregation_passes_validation():
    fact = _fact(
        measure=Measure(concept="income_quantile_cut_point", unit="eur"),
        aggregation=Aggregation(method="quantile"),
        filters={"eurostat.quant_inc": "D1"},
    )

    assert validate_fact(fact) == ()


def test_stable_key_ignores_human_label():
    fact = _fact()
    relabeled = _fact(label="A different display label")

    assert build_fact_key(fact) == build_fact_key(relabeled)


def test_stable_key_ignores_source_table_layout():
    fact = _fact()
    with_layout = _fact(
        layout=SourceRecordLayout(
            record_set_id="irs_soi.ty2023.table_1_1",
            groupby_value_id="all",
            groupby_ordinal=0,
            measure_id="adjusted_gross_income",
            measure_ordinal=1,
        )
    )

    assert build_fact_key(fact) == build_fact_key(with_layout)


def test_duplicate_key_is_reported():
    report = validate_facts([_fact(), _fact(label="Different label")])

    assert not report.valid
    assert [error.code for error in report.errors] == ["duplicate_key"]


def test_missing_provenance_is_reported():
    fact = _fact(
        source=SourceProvenance(
            source_name=None,
            vintage=None,
            extracted_at=None,
            extraction_method=None,
        )
    )

    error_codes = {error.code for error in validate_fact(fact)}

    assert "missing_field" in error_codes
    assert "missing_provenance" in error_codes


def test_malformed_geography_entity_and_aggregation_are_reported():
    fact = _fact(
        geography=GeographyDimension(level="planet", id="earth"),
        entity=EntityDimension(name="simulator_row"),
        aggregation=Aggregation(method="magic"),
    )

    errors = validate_fact(fact)
    error_codes = {error.code for error in errors}

    assert "malformed_geography" in error_codes
    assert "malformed_entity" in error_codes
    assert "malformed_aggregation" in error_codes


def test_source_concept_requires_relation():
    fact = _fact(
        measure=Measure(
            concept="us:statutes/26/62#adjusted_gross_income",
            unit="usd",
            source_concept="irs_soi.adjusted_gross_income",
        )
    )

    errors = validate_fact(fact)

    assert "missing_field" in {error.code for error in errors}


def test_label_generation_uses_metadata_not_key_path():
    fact = _fact(label=None)

    assert build_label(fact) == (
        "United States 2023 tax year sum irs soi adjusted gross income "
        "for tax unit (filing status=all) "
        "[irs_soi Publication 1304 Table 1.1 23in11si.xls tax_year_2023]"
    )


def test_epoch_registry_covers_frozen_domains_and_schema_ids():
    expected_hash_domains = {
        "source_release": ("ledger.source_release.v2", "chronicle.source_release.v3"),
        "source_series": ("ledger.source_series.v2", "chronicle.source_series.v3"),
        "observed_measure": (
            "ledger.observed_measure.v2",
            "chronicle.observed_measure.v3",
        ),
        "dimension_set": ("ledger.dimension_set.v2", "chronicle.dimension_set.v3"),
        "universe_constraint_set": (
            "ledger.universe_constraint_set.v2",
            "chronicle.universe_constraint_set.v3",
        ),
        "aggregate_fact": ("ledger.aggregate_fact.v2", "chronicle.aggregate_fact.v3"),
        "semantic_fact": ("ledger.semantic_fact.v2", "chronicle.semantic_fact.v3"),
        "concept_alignment": (
            "ledger.concept_alignment.v2",
            "chronicle.concept_alignment.v3",
        ),
        "fact": ("ledger.fact.v1", "chronicle.fact.v2"),
        "source_cell": ("ledger.source_cell.v1", "chronicle.source_cell.v2"),
        "source_row": ("ledger.source_row.v1", "chronicle.source_row.v2"),
        "source_column": ("ledger.source_column.v1", "chronicle.source_column.v2"),
        "source_row_value": (
            "ledger.source_row_value.v1",
            "chronicle.source_row_value.v2",
        ),
        "build": ("ledger.build.v1", "chronicle.build.v2"),
        "build_artifact": (
            "ledger.build_artifact.v1",
            "chronicle.build_artifact.v2",
        ),
    }
    expected_schema_ids = {
        "bundle": ("ledger.bundle.v1", "chronicle.bundle.v2"),
        "bundle_coverage": (
            "ledger.bundle_coverage.v1",
            "chronicle.bundle_coverage.v2",
        ),
        "bundle_sources": ("ledger.bundle_sources.v1", "chronicle.bundle_sources.v2"),
        "consumer_fact": ("ledger.consumer_fact.v1", "chronicle.consumer_fact.v2"),
        "relational": ("ledger.relational.v1", "chronicle.relational.v2"),
        "source_package": ("ledger.source_package.v1", "chronicle.source_package.v2"),
        "offline_fetch_manifest": (
            "ledger.offline_fetch_manifest.v1",
            "chronicle.offline_fetch_manifest.v2",
        ),
        "fetch_manifest": ("ledger.fetch_manifest.v1", "chronicle.fetch_manifest.v2"),
        "consumer_artifact": (
            "policyengine_ledger.consumer_artifact.v2",
            "policyengine_chronicle.consumer_artifact.v3",
        ),
        "approved_agents": (
            "policyengine_ledger.approved_agents.v1",
            "policyengine_chronicle.approved_agents.v2",
        ),
    }

    assert EMIT_EPOCH == Epoch.LEDGER
    assert {name: pair.accepted for name, pair in HASH_DOMAINS.items()} == (
        expected_hash_domains
    )
    assert {name: pair.accepted for name, pair in SCHEMA_IDS.items()} == (
        expected_schema_ids
    )


def test_epoch_registry_unknown_key_names_both_accepted_forms():
    pair = HASH_DOMAINS["fact"]

    try:
        pair.infer_key_epoch("future.fact.v9:abc")
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("unknown key domain was accepted")

    assert pair.ledger in message
    assert pair.chronicle in message


def test_fact_key_epochs_hash_the_same_canonical_payload():
    ledger_key = build_fact_key(_fact(), epoch=Epoch.LEDGER)
    chronicle_key = build_fact_key(_fact(), epoch=Epoch.CHRONICLE)

    assert build_fact_key(_fact()) == ledger_key
    assert ledger_key.partition(":")[0] == "ledger.fact.v1"
    assert chronicle_key.partition(":")[0] == "chronicle.fact.v2"
    assert ledger_key.partition(":")[2] == chronicle_key.partition(":")[2]


def test_source_key_epochs_hash_the_same_canonical_payload():
    artifact = _source_artifact()
    cell = SourceCell(
        artifact=artifact,
        sheet_name="Sheet1",
        row_number=2,
        column_number=3,
        address="C2",
        cell_type="number",
        raw_value=42,
        display_value="42",
    )
    row = SourceRow(
        artifact=artifact,
        sheet_name="Sheet1",
        row_number=2,
        values={"amount": 42},
    )
    column = SourceColumn(
        artifact=artifact,
        sheet_name="Sheet1",
        column_number=1,
        raw_name="amount",
        normalized_name="amount",
    )
    cases = (
        (build_source_cell_key, cell, "source_cell"),
        (build_source_row_key, row, "source_row"),
        (build_source_column_key, column, "source_column"),
    )

    for builder, record, domain in cases:
        ledger_key = builder(record, epoch=Epoch.LEDGER)
        chronicle_key = builder(record, epoch=Epoch.CHRONICLE)
        assert builder(record) == ledger_key
        assert ledger_key.partition(":")[0] == HASH_DOMAINS[domain].ledger
        assert chronicle_key.partition(":")[0] == HASH_DOMAINS[domain].chronicle
        assert ledger_key.partition(":")[2] == chronicle_key.partition(":")[2]


def test_source_row_value_hash_canonicalizes_nested_key_epochs():
    artifact = _source_artifact()
    row = SourceRow(
        artifact=artifact,
        sheet_name="Sheet1",
        row_number=2,
        values={"amount": 42},
    )
    column = SourceColumn(
        artifact=artifact,
        sheet_name="Sheet1",
        column_number=1,
        raw_name="amount",
        normalized_name="amount",
    )
    ledger_value = SourceRowValue(
        source_row_key=build_source_row_key(row, epoch=Epoch.LEDGER),
        source_column_key=build_source_column_key(column, epoch=Epoch.LEDGER),
        row_number=2,
        column_number=1,
        raw_column_name="amount",
        normalized_column_name="amount",
        value=42,
    )
    chronicle_value = SourceRowValue(
        source_row_key=build_source_row_key(row, epoch=Epoch.CHRONICLE),
        source_column_key=build_source_column_key(column, epoch=Epoch.CHRONICLE),
        row_number=2,
        column_number=1,
        raw_column_name="amount",
        normalized_column_name="amount",
        value=42,
    )

    ledger_key = build_source_row_value_key(ledger_value, epoch=Epoch.LEDGER)
    chronicle_key = build_source_row_value_key(
        chronicle_value,
        epoch=Epoch.CHRONICLE,
    )

    assert build_source_row_value_key(ledger_value) == ledger_key
    assert ledger_key.partition(":")[2] == chronicle_key.partition(":")[2]
    assert ledger_key.startswith("ledger.source_row_value.v1:")
    assert chronicle_key.startswith("chronicle.source_row_value.v2:")


def test_fact_validation_accepts_mixed_lineage_epochs_with_distinct_digests():
    fact = _fact(
        source_cell_keys=(
            "ledger.source_cell.v1:ledger-cell",
            "chronicle.source_cell.v2:chronicle-cell",
        ),
        source_row_keys=(
            "ledger.source_row.v1:ledger-row",
            "chronicle.source_row.v2:chronicle-row",
        ),
    )

    assert validate_fact(fact) == ()


def test_fact_validation_rejects_canonical_duplicate_lineage_keys():
    fact = _fact(
        source_cell_keys=(
            "ledger.source_cell.v1:same-cell",
            "chronicle.source_cell.v2:same-cell",
        ),
        source_row_keys=(
            "ledger.source_row.v1:same-row",
            "chronicle.source_row.v2:same-row",
        ),
    )

    errors = [(issue.code, issue.field, issue.message) for issue in validate_fact(fact)]

    assert [(code, field) for code, field, _message in errors] == [
        ("duplicate_lineage_key", "source_cell_keys"),
        ("duplicate_lineage_key", "source_row_keys"),
    ]
    assert all("Ledger and Chronicle aliases" in message for _, _, message in errors)


def test_fact_validation_rejects_unknown_lineage_prefix_with_both_forms():
    errors = validate_fact(_fact(source_cell_keys=("future.source_cell.v9:1234",)))

    error = next(issue for issue in errors if issue.code == "malformed_lineage_key")
    assert error.field == "source_cell_keys"
    assert "ledger.source_cell.v1" in error.message
    assert "chronicle.source_cell.v2" in error.message


def test_validate_facts_cli_accepts_chronicle_lineage_end_to_end(tmp_path, capsys):
    fact = _fact(
        source_cell_keys=("chronicle.source_cell.v2:accepted",),
        source_row_keys=("chronicle.source_row.v2:accepted",),
    )
    path = tmp_path / "chronicle-facts.jsonl"
    save_facts_jsonl([fact], path)

    exit_code = harness_main(["validate-facts", "--input", str(path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["valid"]


def test_validate_facts_cli_unknown_lineage_names_both_forms(tmp_path, capsys):
    path = tmp_path / "unknown-facts.jsonl"
    save_facts_jsonl(
        [_fact(source_cell_keys=("future.source_cell.v9:rejected",))],
        path,
    )

    exit_code = harness_main(["validate-facts", "--input", str(path)])
    payload = json.loads(capsys.readouterr().out)
    message = json.dumps(payload)

    assert exit_code == 1
    assert "ledger.source_cell.v1" in message
    assert "chronicle.source_cell.v2" in message


def test_frozen_fixture_bytes_are_unchanged():
    fixture_root = Path(__file__).parents[1] / "chronicle" / "fixtures"
    expected_sha256 = {
        fixture_root / "facts.jsonl": (
            "b0dd06765db7932c16a678b1ab321a7d908af26e2f2014d7da99c0eb5127e401"
        ),
        fixture_root / "consumer_facts.jsonl": (
            "6123f1cca28ccc72c053b105b8d50b5c25a72a5f5b92e73e7219f32de152a96a"
        ),
        fixture_root / "source_cells" / "soi_table_1_1_2023_cells.jsonl": (
            "615639f21ee63e54595c677e24c3eddff484c00a795a2f91b45a8575f021c7e2"
        ),
    }

    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected_sha256
    } == expected_sha256
