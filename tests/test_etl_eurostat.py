"""Offline ETL coverage for the Eurostat BE/DE/FR pilot packages."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from chronicle.consumer_contract import validate_consumer_fact_contract
from chronicle.core import validate_facts
from chronicle.source_package import (
    SOURCE_PACKAGE_ALIASES,
    load_source_package,
    validate_source_package,
)
from chronicle.sources.cells import validate_source_cells
from chronicle.sources.rows import build_source_row_key, validate_source_rows
from chronicle.suite import build_source_suite


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ONLY_COMMENT = (
    "# EVALUATION-ONLY by class: Ledger ingests everything; "
    "the Populace gate decides use."
)
EUROSTAT_PACKAGES = {
    "eurostat-gov-10a-taxag": ("gov_10a_taxag", 2024, 24, 24),
    "eurostat-gov-10a-taxag-2025": ("gov_10a_taxag", 2025, 12, 24),
    "eurostat-spr-exp-func": ("spr_exp_func", 2023, 27, 27),
    "eurostat-spr-exp-func-2024": ("spr_exp_func", 2024, 9, 18),
    "eurostat-nasa-10-nf-tr": ("nasa_10_nf_tr", 2024, 78, 84),
    "eurostat-ilc-li02": ("ilc_li02", 2024, 3, 3),
    "eurostat-ilc-di01": ("ilc_di01", 2024, 54, 54),
}
EUROSTAT_DATASET_IDS = {values[0] for values in EUROSTAT_PACKAGES.values()}
NASA_10_NF_TR_ALIAS = "eurostat-nasa-10-nf-tr"
NASA_10_NF_TR_YEAR = 2024
NASA_10_NF_TR_SHA256 = (
    "30d3f5bf3d1414c78633d13f505558b837b84e92c24b17050249ab19cec20a6d"
)
EUROSTAT_ARTIFACT_FILENAMES = {
    "eurostat-gov-10a-taxag": "gov_10a_taxag.json",
    "eurostat-gov-10a-taxag-2025": "gov_10a_taxag_2022_2025.json",
    "eurostat-spr-exp-func": "spr_exp_func.json",
    "eurostat-spr-exp-func-2024": "spr_exp_func_2023_2024.json",
    "eurostat-nasa-10-nf-tr": "nasa_10_nf_tr.json",
    "eurostat-ilc-li02": "ilc_li02.json",
    "eurostat-ilc-di01": "ilc_di01.json",
}
# chronicle#261 added dimension and value labels to every fact; with those three
# fields stripped, these facts still hash to 9db298bc... and 51768608... .
PRIOR_VINTAGE_FACT_DIGESTS = {
    "eurostat-gov-10a-taxag": (
        "af578be022bf0f8808851936edccdf5c4fd6e0282fcd1a241137ec796144a160"
    ),
    "eurostat-spr-exp-func": (
        "1a5569be1c59c4102576cc8e29d783e4795cb290368a8494f580f58401985398"
    ),
}
EXPECTED_QUERY_FILTERS = {
    "gov_10a_taxag": {
        "freq": ["A"],
        "unit": ["MIO_EUR"],
        "sector": ["S13"],
        "na_item": ["D2", "D5", "D51", "D61"],
        "geo": ["BE", "DE", "FR"],
        "time": ["2023", "2024"],
    },
    "spr_exp_func": {
        "freq": ["A"],
        "spdeps": ["SPR"],
        "spfunc": [
            "TOTAL",
            "SICK",
            "DIS",
            "OLD",
            "SRV",
            "FAM",
            "UNE",
            "HOU",
            "EXCL",
        ],
        "unit": ["MIO_EUR"],
        "geo": ["BE", "DE", "FR"],
        "time": ["2023"],
    },
    "nasa_10_nf_tr": {
        "freq": ["A"],
        "unit": ["CP_MEUR"],
        "sector": ["S14_S15"],
        "na_item": [
            "B2G",
            "B3G",
            "B3N",
            "B6G",
            "D1",
            "D11",
            "D12",
            "D4",
            "D41",
            "D42",
            "D44",
            "D45",
            "D5",
            "D61",
            "D62",
        ],
        "direct": ["RECV", "PAID"],
        "geo": ["BE"],
        "time": ["2022", "2023", "2024"],
    },
    "ilc_li02": {
        "freq": ["A"],
        "statinfo": ["MED_EI"],
        "unit": ["PC"],
        "rskpovth": ["B_60"],
        "sex": ["T"],
        "age": ["TOTAL"],
        "geo": ["BE", "DE", "FR"],
        "time": ["2024"],
    },
    "ilc_di01": {
        "freq": ["A"],
        "quant_inc": [f"D{index}" for index in range(1, 10)],
        "statinfo": ["TC", "SHARE"],
        "unit": ["EUR"],
        "geo": ["BE", "DE", "FR"],
        "time": ["2024"],
    },
}
EXPECTED_VINTAGE_QUERY_FILTERS = {
    ("gov_10a_taxag", "gov_10a_taxag_2022_2025.json"): {
        "freq": ["A"],
        "unit": ["MIO_EUR"],
        "sector": ["S13"],
        "na_item": ["D51", "D51A_C1", "D51B_C2", "D5", "D61", "D2"],
        "geo": ["BE"],
        "time": ["2022", "2023", "2024", "2025"],
    },
    ("spr_exp_func", "spr_exp_func_2023_2024.json"): {
        **EXPECTED_QUERY_FILTERS["spr_exp_func"],
        "geo": ["BE"],
        "time": ["2023", "2024"],
    },
}


@lru_cache
def _package_outputs(alias: str):
    _dataset_id, year, _expected_count, _expected_row_count = EUROSTAT_PACKAGES[alias]
    package = load_source_package(alias)
    rows = package.build_source_rows(year)
    cells = package.build_source_cells(year, source_rows=rows)
    facts = package.build_facts(year, cells=cells, source_rows=rows)
    return package, rows, cells, facts


@lru_cache
def _nasa_10_nf_tr_outputs():
    return _package_outputs(NASA_10_NF_TR_ALIAS)


def test_nasa_10_nf_tr_preserves_full_cube_and_selected_fact_lineage():
    package, rows, cells, facts = _nasa_10_nf_tr_outputs()
    package_report = validate_source_package(
        NASA_10_NF_TR_ALIAS,
        year=NASA_10_NF_TR_YEAR,
    )
    rows_by_key = {build_source_row_key(row): row for row in rows}

    assert NASA_10_NF_TR_ALIAS in SOURCE_PACKAGE_ALIASES
    assert package.artifact.parser == "json_stat_2_full_rows"
    assert package_report.valid, package_report.to_dict()
    assert package_report.counts == {
        "record_set_count": 6,
        "row_count": 78,
        "measure_count": 6,
        "source_record_count": 78,
        "source_region_count": 6,
    }
    assert len(package.record_sets) == 6
    assert len(rows) == 84
    assert len(facts) == 78
    assert validate_source_rows(rows).valid
    assert validate_source_cells(cells).valid
    assert validate_facts(facts).valid
    contract_report = validate_consumer_fact_contract(facts)
    assert contract_report.valid, contract_report.to_dict()

    assert Counter((fact.period.type, fact.period.value) for fact in facts) == {
        ("calendar_year", 2022): 26,
        ("calendar_year", 2023): 26,
        ("calendar_year", 2024): 26,
    }
    assert Counter(fact.filters["direct"] for fact in facts) == {
        "PAID": 39,
        "RECV": 39,
    }
    assert {fact.provenance_class for fact in facts} == {"administrative"}
    assert {fact.survey_instrument for fact in facts} == {None}
    assert {fact.entity.name for fact in facts} == {"household"}
    assert {fact.entity.role for fact in facts} == {"resident_households_and_npish"}
    assert {fact.measure.unit for fact in facts} == {"eur"}
    assert {fact.measure.concept_relation for fact in facts} == {"exact"}
    assert {fact.measure.concept_authority for fact in facts} == {"ledger-be"}
    assert {fact.measure.concept for fact in facts} == {
        "belgium_household_national_accounts_non_financial_transaction_amount"
    }

    groupby_value_ids = {fact.layout.groupby_value_id for fact in facts}
    assert {
        "belgium_household_mixed_income_gross",
        "belgium_household_disposable_income_gross",
        "belgium_household_interest_received",
        "belgium_household_distributed_income_of_corporations_received",
        "belgium_household_wages_and_salaries_received",
        "belgium_household_current_taxes_on_income_paid",
        "belgium_household_net_social_contributions_paid",
        "belgium_household_social_benefits_received",
    } <= groupby_value_ids

    null_coordinates = {
        (
            row.values["direct"],
            row.values["na_item"],
            str(row.values["time"]),
        )
        for row in rows
        if row.values["value"] is None
    }
    assert null_coordinates == {
        (direction, item, str(year))
        for direction, item in (("PAID", "D42"), ("RECV", "D5"))
        for year in (2022, 2023, 2024)
    }

    assert len({fact.source_record_id for fact in facts}) == 78
    for fact in facts:
        assert len(fact.source_row_keys) == 1
        assert len(fact.source_cell_keys) == 8
        source_row = rows_by_key[fact.source_row_keys[0]]
        assert source_row.values["value"] is not None
        assert fact.value == source_row.values["value"] * 1_000_000
        assert fact.filters["direct"] == source_row.values["direct"]
        assert fact.filters["na_item"] == source_row.values["na_item"]
        assert str(fact.period.value) == str(source_row.values["time"])
        assert fact.source.source_sha256 == NASA_10_NF_TR_SHA256
        assert NASA_10_NF_TR_SHA256 in fact.source.raw_r2_key
        assert fact.source.raw_r2_uri == (
            f"r2://{fact.source.raw_r2_bucket}/{fact.source.raw_r2_key}"
        )

    manifest_path = (
        REPO_ROOT / "db" / "data" / "eurostat" / "nasa_10_nf_tr" / "manifest.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text())
    file_spec = manifest["files"][NASA_10_NF_TR_YEAR]
    artifact_path = manifest_path.parent / file_spec["filename"]
    artifact_bytes = artifact_path.read_bytes()
    expected_r2_key = (
        "raw/eurostat/eurostat-nasa-10-nf-tr/2024/"
        f"{NASA_10_NF_TR_SHA256}/nasa_10_nf_tr.json"
    )

    assert hashlib.sha256(artifact_bytes).hexdigest() == NASA_10_NF_TR_SHA256
    assert len(artifact_bytes) == 5098
    assert file_spec["sha256"] == NASA_10_NF_TR_SHA256
    assert file_spec["size_bytes"] == 5098
    assert file_spec["storage"]["r2"]["key"] == expected_r2_key
    assert file_spec["storage"]["r2"]["uri"] == (f"r2://ledger-raw/{expected_r2_key}")


def test_eurostat_aliases_and_packages_validate_end_to_end():
    assert set(EUROSTAT_PACKAGES) <= set(SOURCE_PACKAGE_ALIASES)

    for alias, (
        _dataset_id,
        year,
        expected_count,
        expected_row_count,
    ) in EUROSTAT_PACKAGES.items():
        package, rows, cells, facts = _package_outputs(alias)
        package_report = validate_source_package(alias, year=year)

        assert package.artifact.parser == "json_stat_2_full_rows"
        assert package_report.valid, package_report.to_dict()
        assert len(rows) == expected_row_count
        assert len(facts) == expected_count
        assert validate_source_rows(rows).valid
        assert validate_source_cells(cells).valid
        assert validate_facts(facts).valid
        contract_report = validate_consumer_fact_contract(facts)
        assert contract_report.valid, contract_report.to_dict()


@pytest.mark.parametrize("alias", EUROSTAT_PACKAGES)
def test_eurostat_packages_pass_full_agent_acceptance(alias, tmp_path):
    _dataset_id, year, expected_count, _expected_row_count = EUROSTAT_PACKAGES[alias]

    report = build_source_suite(alias, tmp_path / alias, year=year)

    assert report.valid, report.to_dict()
    assert report.agent_acceptance.valid, report.agent_acceptance.to_dict()
    assert report.agent_acceptance.counts["fact_count"] == expected_count
    assert report.agent_acceptance.counts["row_semantic_error_count"] == 0


def test_eurostat_facts_preserve_raw_cube_dimensions_and_scaling():
    unit_mapping = {
        "MIO_EUR": ("eur", 1_000_000),
        "CP_MEUR": ("eur", 1_000_000),
        "PC": ("percent", 1),
        "EUR": ("eur", 1),
    }

    for alias in EUROSTAT_PACKAGES:
        _package, rows, _cells, facts = _package_outputs(alias)
        rows_by_key = {build_source_row_key(row): row for row in rows}

        for fact in facts:
            assert len(fact.source_row_keys) == 1
            source_row = rows_by_key[fact.source_row_keys[0]]
            raw_unit = str(source_row.values["unit"])
            if alias == "eurostat-ilc-di01":
                expected_unit = {
                    "SHARE": "percent",
                    "TC": "eur",
                }[str(source_row.values["statinfo"])]
                scale = 1
            else:
                expected_unit, scale = unit_mapping[raw_unit]

            assert source_row.values["geo"] == fact.geography.id
            assert str(source_row.values["time"]) == str(fact.period.value)
            assert fact.geography.level == "country"
            assert fact.geography.vintage == "current"
            assert fact.measure.unit == expected_unit
            assert fact.value == pytest.approx(source_row.values["value"] * scale)


def test_eurostat_provenance_and_evaluation_only_boundary():
    administrative_aliases = {
        "eurostat-gov-10a-taxag",
        "eurostat-gov-10a-taxag-2025",
        "eurostat-spr-exp-func",
        "eurostat-spr-exp-func-2024",
        "eurostat-nasa-10-nf-tr",
    }
    survey_aliases = {"eurostat-ilc-li02", "eurostat-ilc-di01"}

    for alias in administrative_aliases:
        facts = _package_outputs(alias)[3]
        assert {fact.provenance_class for fact in facts} == {"administrative"}
        assert {fact.survey_instrument for fact in facts} == {None}

    for alias in survey_aliases:
        package, _rows, _cells, facts = _package_outputs(alias)
        package_text = (package.package_path / "source_package.yaml").read_text()
        assert EVALUATION_ONLY_COMMENT in package_text
        assert {fact.provenance_class for fact in facts} == {"survey_aggregate"}
        assert {fact.survey_instrument for fact in facts} == {"EU-SILC"}

    di01_facts = _package_outputs("eurostat-ilc-di01")[3]
    assert Counter(
        (fact.measure.unit, fact.aggregation.method) for fact in di01_facts
    ) == {
        ("percent", "share"): 27,
        ("eur", "quantile"): 27,
    }


def test_eurostat_production_values_match_publisher_bytes():
    # Decoded independently from the raw JSON-stat cube (row-major index over
    # freq/unit/sector/na_item/geo/time): D2 taxes on production and imports,
    # calendar year 2023, MIO_EUR scaled by 1,000,000. Exact equality, not
    # approx: publisher lexemes must survive scaling bit-for-bit.
    _package, rows, _cells, facts = _package_outputs("eurostat-gov-10a-taxag")
    rows_by_key = {build_source_row_key(row): row for row in rows}
    expected = {"BE": 72_825_500_000, "DE": 428_710_000_000, "FR": 446_580_000_000}

    d2_2023 = {
        source_row.values["geo"]: fact
        for fact in facts
        for source_row in (rows_by_key[fact.source_row_keys[0]],)
        if source_row.values["na_item"] == "D2"
        and str(source_row.values["time"]) == "2023"
    }

    assert set(d2_2023) == set(expected)
    for geo, value in expected.items():
        assert d2_2023[geo].value == value
        assert isinstance(d2_2023[geo].value, int)
        assert d2_2023[geo].measure.unit == "eur"
        assert d2_2023[geo].geography.id == geo

    # The decimal-scaling regression case: 16448.06 MIO_EUR must scale to
    # exactly 16,448,060,000 (binary-float multiply emitted ...000.000002).
    _package, spr_rows, _cells, spr_facts = _package_outputs("eurostat-spr-exp-func")
    spr_rows_by_key = {build_source_row_key(row): row for row in spr_rows}
    be_disability = [
        fact
        for fact in spr_facts
        for source_row in (spr_rows_by_key[fact.source_row_keys[0]],)
        if source_row.values["geo"] == "BE" and source_row.values["spfunc"] == "DIS"
    ]
    assert len(be_disability) == 1
    assert be_disability[0].value == 16_448_060_000
    assert isinstance(be_disability[0].value, int)


@pytest.mark.parametrize("alias", PRIOR_VINTAGE_FACT_DIGESTS)
def test_eurostat_prior_vintage_facts_remain_byte_stable(alias):
    facts = _package_outputs(alias)[3]
    canonical = json.dumps(
        [asdict(fact) for fact in facts],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert hashlib.sha256(canonical).hexdigest() == PRIOR_VINTAGE_FACT_DIGESTS[alias]


def test_eurostat_2025_tax_vintage_adds_only_new_coordinates():
    old_package, old_rows, _old_cells, old_facts = _package_outputs(
        "eurostat-gov-10a-taxag"
    )
    new_package, new_rows, _new_cells, new_facts = _package_outputs(
        "eurostat-gov-10a-taxag-2025"
    )
    old_rows_by_key = {build_source_row_key(row): row for row in old_rows}
    new_rows_by_key = {build_source_row_key(row): row for row in new_rows}

    def coordinates(facts, rows_by_key):
        return {
            (
                str(rows_by_key[fact.source_row_keys[0]].values["time"]),
                str(rows_by_key[fact.source_row_keys[0]].values["na_item"]),
                str(rows_by_key[fact.source_row_keys[0]].values["geo"]),
            )
            for fact in facts
        }

    old_coordinates = coordinates(old_facts, old_rows_by_key)
    new_coordinates = coordinates(new_facts, new_rows_by_key)
    expected_values = {
        ("2022", "D51A_C1"): 65_901_300_000,
        ("2022", "D51B_C2"): 21_681_500_000,
        ("2023", "D51A_C1"): 69_524_900_000,
        ("2023", "D51B_C2"): 23_264_100_000,
        ("2024", "D51A_C1"): 73_712_500_000,
        ("2024", "D51B_C2"): 27_026_800_000,
        ("2025", "D2"): 75_518_700_000,
        ("2025", "D5"): 106_046_800_000,
        ("2025", "D51"): 103_324_300_000,
        ("2025", "D51A_C1"): 75_791_100_000,
        ("2025", "D51B_C2"): 26_308_000_000,
        ("2025", "D61"): 97_512_600_000,
    }
    compiled_values = {
        (
            str(new_rows_by_key[fact.source_row_keys[0]].values["time"]),
            str(new_rows_by_key[fact.source_row_keys[0]].values["na_item"]),
        ): fact.value
        for fact in new_facts
    }
    linked_statuses = {
        str(new_rows_by_key[fact.source_row_keys[0]].values["time"]): new_rows_by_key[
            fact.source_row_keys[0]
        ].values["status"]
        for fact in new_facts
    }

    assert old_package.package_id == "eurostat-gov-10a-taxag"
    assert new_package.package_id == "eurostat-gov-10a-taxag-2025"
    assert old_coordinates.isdisjoint(new_coordinates)
    assert new_coordinates == {(year, item, "BE") for year, item in expected_values}
    assert compiled_values == expected_values
    assert linked_statuses == {
        "2022": None,
        "2023": None,
        "2024": None,
        "2025": "p",
    }
    assert {fact.assertion for fact in new_facts} == {"observation"}


def test_eurostat_2024_esspros_vintage_excludes_overlapping_2023_facts():
    _old_package, old_rows, _old_cells, old_facts = _package_outputs(
        "eurostat-spr-exp-func"
    )
    _new_package, new_rows, _new_cells, new_facts = _package_outputs(
        "eurostat-spr-exp-func-2024"
    )
    old_rows_by_key = {build_source_row_key(row): row for row in old_rows}
    new_rows_by_key = {build_source_row_key(row): row for row in new_rows}

    def coordinates(facts, rows_by_key):
        return {
            (
                str(rows_by_key[fact.source_row_keys[0]].values["time"]),
                str(rows_by_key[fact.source_row_keys[0]].values["spfunc"]),
                str(rows_by_key[fact.source_row_keys[0]].values["geo"]),
            )
            for fact in facts
        }

    expected_values = {
        "TOTAL": 177_883_870_000,
        "SICK": 50_457_080_000,
        "DIS": 17_863_640_000,
        "OLD": 74_545_990_000,
        "SRV": 9_262_130_000,
        "FAM": 13_162_100_000,
        "UNE": 5_734_360_000,
        "HOU": 1_306_850_000,
        "EXCL": 5_551_710_000,
    }
    compiled_values = {
        str(new_rows_by_key[fact.source_row_keys[0]].values["spfunc"]): fact.value
        for fact in new_facts
    }
    linked_statuses = {
        new_rows_by_key[fact.source_row_keys[0]].values["status"] for fact in new_facts
    }

    assert coordinates(old_facts, old_rows_by_key).isdisjoint(
        coordinates(new_facts, new_rows_by_key)
    )
    assert coordinates(new_facts, new_rows_by_key) == {
        ("2024", function, "BE") for function in expected_values
    }
    assert compiled_values == expected_values
    assert linked_statuses == {"e"}
    assert {fact.assertion for fact in new_facts} == {"observation"}


def test_eurostat_production_surfaces_carry_no_fixture_identity():
    # Fixture identity must be absent everywhere it could reach a fact:
    # package specs (vintage, sheet names, notes), db manifests, and the
    # artifacts themselves. Case-insensitive, marker-word based.
    forbidden = re.compile(r"fixture|sentinel|synthetic", re.IGNORECASE)
    surfaces = sorted(
        list((REPO_ROOT / "packages" / "eurostat").rglob("source_package.yaml"))
        + list((REPO_ROOT / "db" / "data" / "eurostat").rglob("manifest.yaml"))
        + list((REPO_ROOT / "db" / "data" / "eurostat").rglob("*.json"))
    )
    assert len(surfaces) >= 12
    for surface in surfaces:
        match = forbidden.search(surface.read_text())
        assert match is None, f"{surface}: {match.group(0)!r}"


def test_eurostat_artifacts_match_verified_live_dimension_shapes_and_labels():
    expected_shapes = {
        "nasa_10_nf_tr": (
            2024,
            ["freq", "unit", "direct", "na_item", "sector", "geo", "time"],
            [1, 1, 2, 14, 1, 1, 3],
        ),
        "spr_exp_func": (
            2023,
            ["freq", "spdeps", "spfunc", "unit", "geo", "time"],
            [1, 1, 9, 1, 3, 1],
        ),
        "ilc_li02": (
            2024,
            ["freq", "statinfo", "unit", "rskpovth", "sex", "age", "geo", "time"],
            [1, 1, 1, 1, 1, 1, 3, 1],
        ),
        "ilc_di01": (
            2024,
            ["freq", "quant_inc", "statinfo", "unit", "geo", "time"],
            [1, 9, 2, 1, 3, 1],
        ),
    }

    fixtures = {}
    for dataset_id, (year, expected_id, expected_size) in expected_shapes.items():
        package_dir = REPO_ROOT / "db" / "data" / "eurostat" / dataset_id
        manifest = yaml.safe_load((package_dir / "manifest.yaml").read_text())
        filename = manifest["files"][year]["filename"]
        fixture = json.loads((package_dir / filename).read_text())
        fixtures[dataset_id] = fixture

        assert fixture["id"] == expected_id
        assert fixture["size"] == expected_size

    assert fixtures["spr_exp_func"]["dimension"]["spfunc"]["category"]["index"] == {
        "TOTAL": 0,
        "SICK": 1,
        "DIS": 2,
        "OLD": 3,
        "SRV": 4,
        "FAM": 5,
        "UNE": 6,
        "HOU": 7,
        "EXCL": 8,
    }
    assert fixtures["ilc_li02"]["dimension"]["statinfo"]["category"]["label"] == {
        "MED_EI": "Median equivalised income"
    }
    assert fixtures["ilc_li02"]["dimension"]["rskpovth"]["category"]["label"] == {
        "B_60": "Below 60%"
    }
    assert fixtures["ilc_di01"]["dimension"]["statinfo"]["category"]["label"] == {
        "TC": "Top cut-off point",
        "SHARE": "Share of national equivalised income",
    }
    assert (
        "D10" not in fixtures["ilc_di01"]["dimension"]["quant_inc"]["category"]["index"]
    )

    li02_package_text = (
        REPO_ROOT / "packages" / "eurostat" / "ilc_li02" / "source_package.yaml"
    ).read_text()
    assert 'B_60 is "Below 60%"' in li02_package_text
    assert 'A_60 is "Above 60%"' in li02_package_text


def test_eurostat_manifests_pin_real_publisher_artifacts():
    for alias, (
        dataset_id,
        year,
        _expected_count,
        _expected_row_count,
    ) in EUROSTAT_PACKAGES.items():
        package_dir = REPO_ROOT / "db" / "data" / "eurostat" / dataset_id
        manifest = yaml.safe_load((package_dir / "manifest.yaml").read_text())
        file_spec = manifest["files"][year]
        artifact_path = package_dir / file_spec["filename"]
        artifact = json.loads(artifact_path.read_text())
        content = artifact_path.read_bytes()
        markers = (
            " ".join(
                str(artifact.get(field) or "") for field in ("label", "source", "note")
            )
            + " "
            + str(manifest.get("source_name", ""))
            + " "
            + str(file_spec.get("source_table", ""))
        )

        assert file_spec["filename"] == EUROSTAT_ARTIFACT_FILENAMES[alias]
        assert file_spec["sha256"] == hashlib.sha256(content).hexdigest()
        assert file_spec["size_bytes"] == len(content)
        assert file_spec["storage"]["r2"]["uri"].startswith("r2://ledger-raw/")
        assert file_spec["sha256"] in file_spec["storage"]["r2"]["uri"]
        assert artifact["source"] == "ESTAT"
        assert "TEST FIXTURE" not in markers
        assert "sentinel" not in markers.lower()


def test_fetch_manifest_has_exact_filtered_eurostat_requests():
    fetch_manifest = json.loads(
        (REPO_ROOT / "FETCH-MANIFEST-EUROSTAT.json").read_text()
    )
    fetches = fetch_manifest["fetches"]

    assert fetch_manifest["schema_version"] == "ledger.fetch_manifest.v1"
    assert {fetch["dataset_id"] for fetch in fetches} == EUROSTAT_DATASET_IDS
    assert len(fetches) == 7
    for fetch in fetches:
        dataset_id = fetch["dataset_id"]
        parsed = urlsplit(fetch["source_url"])
        query = parse_qs(parsed.query)
        filename = Path(fetch["destination"]).name
        package_manifest = yaml.safe_load(
            (
                REPO_ROOT / "db" / "data" / "eurostat" / dataset_id / "manifest.yaml"
            ).read_text()
        )
        matching_specs = [
            spec
            for spec in package_manifest["files"].values()
            if spec["filename"] == filename
        ]
        assert len(matching_specs) == 1
        artifact_spec = matching_specs[0]

        assert parsed.scheme == "https"
        assert parsed.netloc == "ec.europa.eu"
        assert parsed.path.endswith(f"/data/{dataset_id}")
        assert query.pop("format") == ["JSON"]
        assert query.pop("lang") == ["en"]
        assert query == EXPECTED_VINTAGE_QUERY_FILTERS.get(
            (dataset_id, filename), EXPECTED_QUERY_FILTERS[dataset_id]
        )
        assert fetch["destination"] == (
            f"db/data/eurostat/{dataset_id}/{artifact_spec['filename']}"
        )
        sha = fetch["sha256"]
        dest = REPO_ROOT / fetch["destination"]
        assert re.fullmatch(r"[0-9a-f]{64}", sha)
        assert dest.exists()
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == sha
        assert artifact_spec["sha256"] == sha
        assert artifact_spec["source_url"] == fetch["source_url"]
