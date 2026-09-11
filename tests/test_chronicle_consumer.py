"""Tests for facts-only Chronicle consumer artifacts."""

from __future__ import annotations

import hashlib
import json

import pytest

from chronicle.consumer_contract import consumer_fact_rows
from chronicle.core import (
    AggregateFact,
    Aggregation,
    EntityDimension,
    GeographyDimension,
    Measure,
    PeriodDimension,
    SourceProvenance,
    SourceRecordLayout,
)
from chronicle.harness import main
from chronicle.epoch import Epoch
from policyengine_chronicle.consumer import (
    build_consumer_artifact,
    load_consumer_artifact,
)
from jsonschema import Draft202012Validator
from policyengine_chronicle.schema import (
    CONSUMER_FACT_SCHEMA_SHA256,
    CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION,
    consumer_fact_schema,
)

SHA = "ab" * 32
_KEY_FIELDS = (
    "aggregate_fact_key",
    "semantic_fact_key",
    "legacy_fact_key",
    "source_release_key",
    "source_series_key",
    "observed_measure_key",
    "dimension_set_key",
    "universe_constraint_set_key",
)


def _fact(*, value, period_value):
    return AggregateFact(
        value=value,
        period=PeriodDimension(type="tax_year", value=period_value),
        geography=GeographyDimension(
            level="country",
            id="0100000US",
            vintage="2020_census",
        ),
        entity=EntityDimension(name="tax_unit", role="filing_unit"),
        measure=Measure(concept="irs_soi.adjusted_gross_income", unit="usd"),
        aggregation=Aggregation(method="sum"),
        provenance_class="administrative",
        source=SourceProvenance(
            source_name="irs_soi",
            source_table="Table T",
            source_file="t.xls",
            url="https://example.gov/t.xls",
            vintage=f"tax_year_{period_value}",
            extracted_at="2026-05-01",
            extraction_method="test",
            source_sha256=SHA,
            source_size_bytes=10,
            raw_r2_bucket="ledger-raw",
            raw_r2_key=f"raw/irs_soi/t/{period_value}/{SHA}/t.xls",
            raw_r2_uri=f"r2://ledger-raw/raw/irs_soi/t/{period_value}/{SHA}/t.xls",
        ),
        domain="all_returns",
        source_record_id=f"irs_soi.{period_value}.t.all.agi",
        source_cell_keys=(f"ledger.source_cell.v1:{period_value}agi",),
        layout=SourceRecordLayout(
            record_set_id=f"irs_soi.{period_value}.t",
            record_set_spec_id="irs_soi.t.v1",
            measure_id="agi",
            groupby_dimension="us.agi",
        ),
    )


def _rows():
    return consumer_fact_rows(
        [_fact(value=100, period_value=2021), _fact(value=110, period_value=2022)]
    )


def _write_facts(tmp_path):
    facts_path = tmp_path / "consumer_facts.jsonl"
    with facts_path.open("w") as file:
        for row in _rows():
            file.write(json.dumps(row, sort_keys=True) + "\n")
    return facts_path


def _write_rows(path, rows):
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _rewrite_manifest_hash(out_dir):
    facts_file = out_dir / "consumer_facts.jsonl"
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["facts_sha256"] = hashlib.sha256(facts_file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def _v1_rows():
    return [{**row, "schema_version": "ledger.consumer_fact.v1"} for row in _rows()]


def test_artifact_build_load_round_trip_is_facts_only(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"

    report = build_consumer_artifact(out_dir, facts_path=facts_path)
    artifact = load_consumer_artifact(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())

    assert report.to_dict() == {
        "schema_version": "policyengine_ledger.consumer_artifact.v2",
        "output_dir": str(out_dir),
        "fact_row_count": 2,
    }
    assert manifest == {
        "schema_version": "policyengine_ledger.consumer_artifact.v2",
        "consumer_fact_schema_versions": ["chronicle.consumer_fact.v2"],
        "consumer_fact_schema_sha256": CONSUMER_FACT_SCHEMA_SHA256,
        "fact_row_count": 2,
        "facts_sha256": hashlib.sha256(
            (out_dir / "consumer_facts.jsonl").read_bytes()
        ).hexdigest(),
    }
    assert (
        CONSUMER_FACT_SCHEMA_SHA256
        == (CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION["chronicle.consumer_fact.v2"])
    )
    assert {path.name for path in out_dir.iterdir()} == {
        "consumer_facts.jsonl",
        "manifest.json",
    }
    assert artifact.path == out_dir
    assert len(artifact.rows) == 2


def test_artifact_is_reproducible(tmp_path):
    facts_path = _write_facts(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_consumer_artifact(first, facts_path=facts_path)
    build_consumer_artifact(second, facts_path=facts_path)

    for name in ("manifest.json", "consumer_facts.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_artifact_build_restamps_v1_rows_as_v2(tmp_path):
    """v2 is v1 plus optional labels, so a v1 row is a valid v2 row: an artifact
    built from v1 rows is the artifact built from the same rows emitted as v2."""
    v1_path = tmp_path / "v1-consumer-facts.jsonl"
    _write_rows(v1_path, _v1_rows())

    build_consumer_artifact(tmp_path / "from-v1", facts_path=v1_path)
    build_consumer_artifact(tmp_path / "from-v2", facts_path=_write_facts(tmp_path))

    assert _artifact_bytes(tmp_path / "from-v1") == _artifact_bytes(
        tmp_path / "from-v2"
    )


def test_artifact_load_accepts_a_v1_artifact(tmp_path):
    """Artifacts built before consumer_fact v2 pin the v1 schema and still load."""
    out_dir = tmp_path / "v1-artifact"
    out_dir.mkdir()
    _write_rows(out_dir / "consumer_facts.jsonl", _v1_rows())
    manifest = {
        "schema_version": "policyengine_ledger.consumer_artifact.v2",
        "consumer_fact_schema_versions": ["ledger.consumer_fact.v1"],
        "consumer_fact_schema_sha256": CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION[
            "ledger.consumer_fact.v1"
        ],
        "fact_row_count": 2,
        "facts_sha256": hashlib.sha256(
            (out_dir / "consumer_facts.jsonl").read_bytes()
        ).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    artifact = load_consumer_artifact(out_dir)

    assert {row["schema_version"] for row in artifact.rows} == {
        "ledger.consumer_fact.v1"
    }


def test_artifact_load_rejects_a_sha_pinned_to_the_other_contract(tmp_path):
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=_write_facts(tmp_path))
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["consumer_fact_schema_sha256"] = CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION[
        "ledger.consumer_fact.v1"
    ]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="but pins the schema sha256"):
        load_consumer_artifact(out_dir)


def test_chronicle_epoch_artifact_emission_is_refused_until_the_key_cutover(
    tmp_path,
):
    """Mechanism 1 emits unchanged: artifacts keep Ledger-named keys until the
    separate, consumer-gated key cutover."""
    facts_path = tmp_path / "chronicle-consumer-facts.jsonl"
    rows = consumer_fact_rows(
        [_fact(value=100, period_value=2021)], emit_epoch=Epoch.CHRONICLE
    )
    _write_rows(facts_path, rows)
    out_dir = tmp_path / "chronicle-artifact"

    with pytest.raises(ValueError, match="consumer-gated key cutover"):
        build_consumer_artifact(
            out_dir,
            facts_path=facts_path,
            emit_epoch=Epoch.CHRONICLE,
        )
    assert not (out_dir / "manifest.json").exists()


def _artifact_bytes(out_dir):
    return {
        name: (out_dir / name).read_bytes()
        for name in ("manifest.json", "consumer_facts.jsonl")
    }


def test_mixed_epoch_rows_are_emitted_ledger_keyed_in_one_artifact(tmp_path):
    """Dual-domain acceptance on read; the artifact emits Ledger-named keys under
    the v2 row contract, so its rows validate against the pinned v2 bytes alone."""
    ledger_row = consumer_fact_rows([_fact(value=100, period_value=2021)])[0]
    chronicle_row = consumer_fact_rows(
        [_fact(value=110, period_value=2022)], emit_epoch=Epoch.CHRONICLE
    )[0]
    facts_path = tmp_path / "mixed-consumer-facts.jsonl"
    _write_rows(facts_path, [ledger_row, chronicle_row])
    out_dir = tmp_path / "mixed-artifact"

    build_consumer_artifact(out_dir, facts_path=facts_path)
    artifact = load_consumer_artifact(out_dir)

    assert artifact.manifest["consumer_fact_schema_versions"] == [
        "chronicle.consumer_fact.v2"
    ]
    assert len(artifact.rows) == 2
    validator = Draft202012Validator(consumer_fact_schema())
    for line in (out_dir / "consumer_facts.jsonl").read_text().splitlines():
        row = json.loads(line)
        validator.validate(row)
        assert row["schema_version"] == "chronicle.consumer_fact.v2"
        for field_name in _KEY_FIELDS:
            assert row[field_name].startswith("ledger.")
        assert all(
            key.startswith("ledger.") for key in row["lineage"]["source_cell_keys"]
        )

    # Canonicalizing on emit is the same artifact as emitting Ledger-named rows.
    ledger_only_path = tmp_path / "ledger-consumer-facts.jsonl"
    _write_rows(
        ledger_only_path,
        [ledger_row, consumer_fact_rows([_fact(value=110, period_value=2022)])[0]],
    )
    ledger_dir = tmp_path / "ledger-artifact"
    build_consumer_artifact(ledger_dir, facts_path=ledger_only_path)
    assert _artifact_bytes(out_dir) == _artifact_bytes(ledger_dir)


@pytest.mark.parametrize(
    ("emit_epoch", "message"),
    [
        (Epoch.CHRONICLE, "consumer-gated key cutover"),
        ("chronicle", "consumer-gated key cutover"),
        ("bogus", "unknown emit epoch"),
    ],
)
def test_refused_emit_leaves_an_existing_artifact_untouched(
    tmp_path, emit_epoch, message
):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    before = _artifact_bytes(out_dir)

    with pytest.raises(ValueError, match=message):
        build_consumer_artifact(
            out_dir, facts_path=facts_path, replace=True, emit_epoch=emit_epoch
        )

    assert _artifact_bytes(out_dir) == before


def test_emit_epoch_accepts_its_string_value(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"

    report = build_consumer_artifact(
        out_dir, facts_path=facts_path, emit_epoch="ledger"
    )

    assert report.schema_version == "policyengine_ledger.consumer_artifact.v2"
    assert load_consumer_artifact(out_dir).manifest["schema_version"] == (
        "policyengine_ledger.consumer_artifact.v2"
    )


def test_artifact_load_rejects_tampered_facts(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)

    facts_file = out_dir / "consumer_facts.jsonl"
    rows = facts_file.read_text().splitlines()
    tampered = json.loads(rows[0])
    tampered["value"] = 999
    rows[0] = json.dumps(tampered, sort_keys=True)
    facts_file.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="manifest hash"):
        load_consumer_artifact(out_dir)


def test_artifact_load_rejects_profile_metadata(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"] = {"legacy": {"sha256": "00" * 32, "target_count": 1}}
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="target profiles are consumer-owned"):
        load_consumer_artifact(out_dir)


def test_artifact_load_rejects_legacy_v1_schema(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "policyengine_ledger.consumer_artifact.v1"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(
        ValueError,
        match="Unsupported consumer artifact schema_version",
    ):
        load_consumer_artifact(out_dir)


def test_artifact_load_unknown_schema_names_both_accepted_forms(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "future.consumer_artifact.v9"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError) as error:
        load_consumer_artifact(out_dir)

    message = str(error.value)
    assert "policyengine_ledger.consumer_artifact.v2" in message
    assert "policyengine_chronicle.consumer_artifact.v3" in message


def test_artifact_load_unknown_row_schema_names_both_accepted_forms(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["consumer_fact_schema_versions"] = ["future.consumer_fact.v9"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError) as error:
        load_consumer_artifact(out_dir)

    message = str(error.value)
    assert "ledger.consumer_fact.v1" in message
    assert "chronicle.consumer_fact.v2" in message


def test_artifact_load_rejects_declared_row_schema_drift(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["consumer_fact_schema_versions"] = ["ledger.consumer_fact.v1"]
    manifest["consumer_fact_schema_sha256"] = CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION[
        "ledger.consumer_fact.v1"
    ]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="but its rows use"):
        load_consumer_artifact(out_dir)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "provenance_class"),
        ("unknown", "provenance_class"),
        ("wrong_type", "provenance_class"),
        ("survey_missing_instrument", "survey_instrument"),
        ("misplaced_instrument", "survey_instrument"),
    ],
)
def test_artifact_build_rejects_malformed_provenance(tmp_path, case, message):
    facts_path = _write_facts(tmp_path)
    rows = facts_path.read_text().splitlines()
    first = json.loads(rows[0])
    if case == "missing":
        first.pop("provenance_class")
    elif case == "unknown":
        first["provenance_class"] = "unknown"
    elif case == "wrong_type":
        first["provenance_class"] = 1
    elif case == "survey_missing_instrument":
        first["provenance_class"] = "survey_aggregate"
    elif case == "misplaced_instrument":
        first["survey_instrument"] = "ACS 1-year"
    rows[0] = json.dumps(first, sort_keys=True)
    facts_path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match=message):
        build_consumer_artifact(tmp_path / "artifact", facts_path=facts_path)


def test_artifact_load_rejects_row_missing_required_field(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    facts_file = out_dir / "consumer_facts.jsonl"
    rows = facts_file.read_text().splitlines()
    first = json.loads(rows[0])
    first.pop("entity")
    rows[0] = json.dumps(first, sort_keys=True)
    facts_file.write_text("\n".join(rows) + "\n")
    _rewrite_manifest_hash(out_dir)

    with pytest.raises(ValueError, match="schema validation"):
        load_consumer_artifact(out_dir)


def test_artifact_load_rejects_unknown_schema_sha256(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["consumer_fact_schema_sha256"] = "00" * 32
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="does not match the packaged"):
        load_consumer_artifact(out_dir)


def test_artifact_build_rejects_duplicate_aggregate_fact_key(tmp_path):
    facts_path = _write_facts(tmp_path)
    first = facts_path.read_text().splitlines()[0]
    facts_path.write_text(first + "\n" + first + "\n")

    with pytest.raises(ValueError, match="must be unique"):
        build_consumer_artifact(tmp_path / "artifact", facts_path=facts_path)


def test_load_rejects_a_forged_identity_key(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    facts_file = out_dir / "consumer_facts.jsonl"
    rows = facts_file.read_text().splitlines()
    first = json.loads(rows[0])
    first["aggregate_fact_key"] = "ledger.aggregate_fact.v2:" + "0" * 24
    rows[0] = json.dumps(first, sort_keys=True)
    facts_file.write_text("\n".join(rows) + "\n")
    _rewrite_manifest_hash(out_dir)

    with pytest.raises(ValueError, match="identity key does not match the row"):
        load_consumer_artifact(out_dir)


def test_load_rejects_a_false_manifest_row_count(tmp_path):
    facts_path = _write_facts(tmp_path)
    out_dir = tmp_path / "artifact"
    build_consumer_artifact(out_dir, facts_path=facts_path)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fact_row_count"] = 999
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="declares fact_row_count"):
        load_consumer_artifact(out_dir)


def test_build_consumer_artifact_help_has_no_profile_options(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["build-consumer-artifact", "--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--profile" not in help_text
    assert "facts-only" in help_text
