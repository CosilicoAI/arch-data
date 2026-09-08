"""Tests for merged Chronicle consumer bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronicle.bundle import (
    BUNDLE_COVERAGE_SCHEMA_VERSION,
    BUNDLE_SCHEMA_VERSION,
    BUNDLE_SOURCES_SCHEMA_VERSION,
    UK_BUNDLE_SOURCES,
    _load_jsonl as load_bundle_jsonl,
    build_bundle,
    build_bundle_coverage,
)
from chronicle.epoch import HASH_DOMAINS, SCHEMA_IDS, Epoch
from chronicle.harness import build_bundle_dir
from chronicle.harness import main as harness_main


def _load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _fixture_consumer_rows():
    path = Path(__file__).parents[1] / "chronicle" / "fixtures" / "consumer_facts.jsonl"
    return _load_jsonl(path)


def _row_for_epoch(row, epoch):
    transformed = json.loads(json.dumps(row))
    transformed["schema_version"] = SCHEMA_IDS["consumer_fact"].for_epoch(epoch)
    for field_name, domain_name in (
        ("aggregate_fact_key", "aggregate_fact"),
        ("semantic_fact_key", "semantic_fact"),
        ("legacy_fact_key", "fact"),
        ("source_release_key", "source_release"),
        ("source_series_key", "source_series"),
        ("observed_measure_key", "observed_measure"),
        ("dimension_set_key", "dimension_set"),
        ("universe_constraint_set_key", "universe_constraint_set"),
    ):
        transformed[field_name] = HASH_DOMAINS[domain_name].key_for_epoch(
            transformed[field_name], epoch
        )
    alignment = transformed.get("concept_alignment")
    if alignment:
        alignment["concept_alignment_key"] = HASH_DOMAINS[
            "concept_alignment"
        ].key_for_epoch(alignment["concept_alignment_key"], epoch)
    lineage = transformed["lineage"]
    for field_name, domain_name in (
        ("source_cell_keys", "source_cell"),
        ("source_row_keys", "source_row"),
    ):
        lineage[field_name] = [
            HASH_DOMAINS[domain_name].key_for_epoch(key, epoch)
            for key in lineage[field_name]
        ]
    return transformed


def test_build_bundle_dir_uk_suite_uses_curated_sources(tmp_path, monkeypatch):
    captured = {}

    class FakeReport:
        valid = True

        def to_dict(self):
            return {"valid": True}

    def fake_build_bundle(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return FakeReport()

    monkeypatch.setattr("chronicle.harness.build_bundle", fake_build_bundle)

    report = build_bundle_dir(tmp_path / "bundle", year=2023, suite="uk")

    assert report.valid
    assert "dfc-ni-uc-statistics-may-2026" in UK_BUNDLE_SOURCES
    assert "dfc-ni-uc-statistics-may-2025" not in UK_BUNDLE_SOURCES
    assert "hmrc-cgt-statistics-2026" in UK_BUNDLE_SOURCES
    assert "hmrc-cgt-statistics-2025" not in UK_BUNDLE_SOURCES
    assert "hmrc-cgt-size-of-gain-2026" in UK_BUNDLE_SOURCES
    assert "hmrc-cgt-size-of-gain-2025" not in UK_BUNDLE_SOURCES
    assert tuple(captured["sources"]) == UK_BUNDLE_SOURCES
    assert captured["output_dir"] == tmp_path / "bundle"


def test_build_bundle_cli_accepts_uk_suite(tmp_path, monkeypatch):
    captured = {}

    class FakeReport:
        valid = True

        def to_dict(self):
            return {"valid": True}

    def fake_build_bundle_dir(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return FakeReport()

    monkeypatch.setattr("chronicle.harness.build_bundle_dir", fake_build_bundle_dir)

    status = harness_main(["build-bundle", "--suite", "uk", "--out", str(tmp_path)])

    assert status == 0
    assert captured["suite"] == "uk"


def test_build_bundle_writes_merged_consumer_contract(tmp_path):
    output_dir = tmp_path / "bundle"

    report = build_bundle(output_dir, year=2023)
    summary = json.loads((output_dir / "reports" / "build_bundle.json").read_text())
    rows = _load_jsonl(output_dir / "consumer_facts.jsonl")
    source_packages = json.loads((output_dir / "source_packages.json").read_text())
    coverage = json.loads((output_dir / "coverage.json").read_text())

    assert report.valid
    assert summary["valid"]
    assert summary["counts"] == {
        "aggregate_duplicate_key_count": 0,
        "entity_count": 12,
        "error_count": 0,
        "fact_count": 195193,
        "geography_count": 12539,
        "period_count": 269,
        "semantic_duplicate_key_count": 177,
        "skipped_source_count": 10,
        "source_count": 45,
        "source_package_count": 169,
        "warning_count": 1,
    }
    assert len(rows) == 195193
    assert {row["provenance_class"] for row in rows} <= {
        "administrative",
        "census",
        "model_output",
        "survey_aggregate",
    }
    assert all(
        (
            isinstance(row.get("survey_instrument"), str)
            and row["survey_instrument"].strip()
        )
        if row["provenance_class"] == "survey_aggregate"
        else "survey_instrument" not in row
        for row in rows
    )
    assert rows[0]["aggregate_fact_key"].startswith("ledger.aggregate_fact.v2:")
    assert rows[0]["semantic_fact_key"].startswith("ledger.semantic_fact.v2:")
    assert source_packages["source_package_count"] == 169
    assert source_packages["skipped_source_count"] == 10
    assert sorted(item["source"] for item in source_packages["skipped_sources"]) == [
        "census-acs-s0101-congressional-district-age-2024",
        "census-acs-s0101-national-age-2024",
        "census-acs-s0101-state-age-2024",
        "census-acs-s2201-congressional-district-snap-2024",
        "cms-aca-effectuated-enrollment-2022",
        "cms-aca-oep-state-level",
        "cms-aca-oep-state-level-2022",
        "cms-aca-oep-state-level-2025",
        "jct-obbba-revenue-estimates-2025",
        "jct-tax-expenditures-2024",
    ]
    assert coverage["fact_count"] == 195193
    assert coverage["counts"]["by_source"] == {
        "bea": 445,
        "bfp_economic_outlook": 5,
        "cbo": 7,
        "census_acs": 468,
        "census_pep": 4132,
        "census_population_projections": 86,
        "census_stc": 46,
        "cms_medicaid": 515,
        "cms_medicare": 1,
        "cms_nhe": 3,
        "dfe": 770,
        "dfc_ni": 1189,
        "dft": 233,
        "dwp": 10831,
        "eurostat": 207,
        "federal_reserve": 1,
        "fpb_economic_outlook": 1000,
        "hhs_acf_liheap": 2,
        "hhs_acf_tanf": 110,
        "hmrc": 22319,
        "ici": 12,
        "irs_soi": 40063,
        "isc": 2,
        "jrc_euromod_be": 90,
        "kff": 52,
        "mhclg": 2712,
        "nbb_national_accounts": 1,
        "nisra": 533,
        "nrs": 5647,
        "obr": 270,
        "onem_rva_unemployment": 1,
        "ons": 80741,
        "onss_contributions": 1,
        "opgroeien_groeipakket": 11,
        "scotgov": 2787,
        "sfpd_pensions": 4,
        "slc": 199,
        "spf_finances_pit": 1,
        "ssa": 426,
        "statbel_fiscal_income": 565,
        "statbel_fiscal_income_distribution": 14600,
        "statbel_population_structure": 36,
        "usda_snap": 852,
        "voa": 3001,
        "welshgov": 216,
    }
    table_counts = coverage["counts"]["by_source_table"]
    assert len(table_counts) == 164
    assert (
        table_counts[
            "dwp:Households on Universal Credit by family type, "
            "payment indicator and child entitlement, April 2023 to May 2026"
        ]
        == 760
    )
    assert (
        table_counts[
            "dwp:Households on Universal Credit by number of children, "
            "payment indicator and child entitlement, April 2023 to May 2026"
        ]
        == 1824
    )
    assert (
        table_counts["dfe:Funded early education and childcare 2026, Headline figures"]
        == 770
    )
    assert (
        table_counts[
            "hmrc:Tax-Free Childcare Statistics March 2026, Table 2: Numbers "
            "of Children with Open and Used Tax-Free Childcare Accounts and "
            "Government Top-up"
        ]
        == 126
    )
    assert (
        table_counts[
            "dft:BUS05i estimated operating revenue and net support for local bus services"
        ]
        == 24
    )
    assert (
        table_counts[
            "dft:NTS0705a average trips by household income quintile and main mode"
        ]
        == 24
    )
    assert (
        table_counts[
            "dft:BUS0415a local bus fares index by metropolitan area status and country"
        ]
        == 104
    )
    assert (
        table_counts[
            "hmrc:Capital Gains Tax statistics Table 3: individual taxpayers and "
            "gains by size of gain and taxable income"
        ]
        == 290
    )
    assert (
        table_counts[
            "hmrc:Capital Gains Tax statistics Table 5: taxpayers, gains and "
            "liabilities by UK country and region"
        ]
        == 84
    )
    assert (
        table_counts[
            "hmrc:Capital Gains Tax statistics Table 6: individual taxpayers, gains "
            "and liabilities by age"
        ]
        == 60
    )
    assert (
        table_counts[
            "dfc_ni:Universal Credit Statistics supplementary tables, May 2026"
        ]
        == 1189
    )
    assert (
        table_counts[
            "ons:Price Index of Private Rents, UK: monthly price statistics, "
            "July 2026 edition (data to June 2026)"
        ]
        == 14656
    )
    assert (
        table_counts[
            "dwp:Universal Credit childcare element statistics to August 2025, Table 1"
        ]
        == 54
    )
    assert (
        table_counts[
            "dwp:Households on Universal Credit with carer entitlement, "
            "April to December 2025"
        ]
        == 9
    )
    assert (
        table_counts[
            "dwp:Households on Universal Credit by number of children, "
            "April to December 2025"
        ]
        == 72
    )
    assert (
        table_counts[
            "dwp:Households on Universal Credit by family type, "
            "April to December 2023 through 2025"
        ]
        == 162
    )
    assert (
        table_counts[
            "dwp:Households on Universal Credit with housing entitlement, "
            "April to December 2025"
        ]
        == 9
    )
    assert (
        table_counts[
            "dwp:Households on Universal Credit with LCWRA entitlement, "
            "April to December 2025"
        ]
        == 9
    )
    assert table_counts["usda_snap:SNAP FY2025 Monthly State Participation"] == 636
    assert (
        table_counts[
            "welshgov:Council Tax Reduction Scheme: annual report 2024 to 2025, "
            "Table 2 total value of CTRS awards in Wales"
        ]
        == 2
    )
    assert (
        table_counts[
            "welshgov:Council Tax Reduction Scheme: annual report 2025 to 2026, "
            "Table 2 total value of CTRS awards in Wales"
        ]
        == 2
    )
    assert (
        table_counts[
            "mhclg:Collection rates for council tax and non-domestic rates in "
            "England 2025 to 2026, Table 5 amount collected in year"
        ]
        == 10
    )
    assert (
        table_counts[
            "mhclg:Council Tax levels set by local authorities in England 2025 "
            "to 2026, Tables 1 to 9 (revised), Table 1 England summary"
        ]
        == 30
    )
    assert (
        table_counts[
            "scotgov:Council Tax Collection Statistics, 2024-25: publication "
            "tables (CTRR), Figure 1 Council Tax billed and received"
        ]
        == 104
    )
    assert (
        table_counts[
            "scotgov:Council Tax Collection Statistics, 2025-26: publication "
            "tables (CTRR), Figure 1 Council Tax billed and received"
        ]
        == 108
    )
    assert (
        table_counts[
            "scotgov:Scottish Local Government Finance Statistics (SLGFS) "
            "2024-25: publication tables, council tax tables 2.6, 2.8, 2.10, 2.11 "
            "and chart 2.7"
        ]
        == 71
    )
    assert (
        table_counts[
            "welshgov:Council tax collection rates in Wales, April 2024 to March "
            "2025, Table 2 amounts outstanding in respect of 2024-25 bills and "
            "arrears"
        ]
        == 7
    )
    assert (
        table_counts[
            "welshgov:Council tax collection rates in Wales, April 2025 to March "
            "2026, Table 2 amounts outstanding in respect of 2025-26 bills and "
            "arrears"
        ]
        == 7
    )
    assert table_counts["irs_soi:Congressional District Data 2022"] == 26880
    assert table_counts["irs_soi:IRS SOI County Data 2022"] == 6286
    assert table_counts["census_pep:Vintage 2024 County Population Totals"] == 3144
    assert (
        table_counts[
            "dwp:Universal Credit deductions statistics March 2025 to February 2026, Table 1"
        ]
        == 35
    )
    assert (
        table_counts[
            "cms_nhe:Employer-Sponsored Private Health Insurance: "
            "Calendar Years 1987-2024"
        ]
        == 2
    )
    assert table_counts["ssa:SSI Monthly Statistics, December 2024, Table 1"] == 4
    assert table_counts["irs_soi:Publication 1304 Table 1.1"] == 80
    assert (
        table_counts[
            "irs_soi:Publication 1304 Table 2.5 EITC by AGI and qualifying children"
        ]
        == 464
    )
    assert table_counts["bea:BEA Regional annual state personal income CSV ZIP"] == 416
    assert (
        table_counts["cbo:CBO budget and economic data, individual income tax receipts"]
        == 1
    )
    assert (
        table_counts[
            "cbo:Revenue Projections, by Category, February 2026, "
            "sheet 3.Individual Income Tax Details"
        ]
        == 6
    )
    assert (
        table_counts[
            "census_acs:ACS 2023 1-year detailed table B01001 female age bands by state"
        ]
        == 468
    )
    assert (
        table_counts[
            "cms_medicaid:State Medicaid and CHIP Applications, Eligibility "
            "Determinations, and Enrollment Data"
        ]
        == 515
    )
    assert table_counts["ssa:SSA Annual Statistical Supplement 2025 Table 7.B1"] == 416
    assert (
        table_counts[
            "ons:UK Business, Activity, Size and Location 2025 enterprise "
            "counts by SIC division, turnover band, and employment size band"
        ]
        == 1232
    )
    assert (
        table_counts[
            "ons:UK Business, Activity, Size and Location 2025 enterprise "
            "turnover and employment size bands"
        ]
        == 14
    )
    assert (
        table_counts[
            "ons:Households by type of household and family, regions of England "
            "and GB constituent countries: Scotland worksheet"
        ]
        == 1
    )
    assert (
        table_counts[
            "hmrc:Annual UK VAT Statistics 2024 to 2025 VAT trader "
            "population and net VAT liability by trade sector"
        ]
        == 176
    )
    assert (
        table_counts[
            "hmrc:Annual UK VAT Statistics 2024 to 2025 VAT trader "
            "population and net VAT liability by turnover band"
        ]
        == 17
    )
    assert (
        table_counts[
            "hmrc:Capital Gains Tax statistics Table 2: estimated number of taxpayers, "
            "amounts of gains and tax liabilities by size of gain"
        ]
        == 51
    )
    assert (
        table_counts[
            "hmrc:Capital Gains Tax statistics Table 1: taxpayer numbers, gains and "
            "tax liabilities by year of disposal"
        ]
        == 342
    )
    assert (
        table_counts[
            "statbel_fiscal_income:Personal income tax statistics by municipality, "
            "income year 2023, 2025 NIS geography"
        ]
        == 565
    )
    assert (
        table_counts[
            "statbel_fiscal_income_distribution:Statbel fiscal income distribution "
            "tables A.1, B.1, B.3, B.4, and B.5, income year 2023"
        ]
        == 14600
    )
    assert (
        table_counts[
            "spf_finances_pit:Personal income tax statistics total taxes, income year "
            "2023"
        ]
        == 1
    )
    assert (
        table_counts[
            "statbel_population_structure:Population by place of residence, nationality, "
            "marital status, age and sex, 2025"
        ]
        == 18
    )
    assert (
        table_counts[
            "statbel_population_structure:Population by place of residence, nationality, "
            "marital status, age and sex, 2026"
        ]
        == 18
    )
    assert (
        table_counts[
            "onss_contributions:Declared contributions 2024, Table 6 by sector, "
            "status and sex"
        ]
        == 1
    )
    assert (
        table_counts[
            "onem_rva_unemployment:Annual report complete unemployment benefit "
            "recipients, 2024"
        ]
        == 1
    )
    assert (
        table_counts[
            "nbb_national_accounts:Household income accounts, gross disposable income, "
            "Belgium"
        ]
        == 1
    )
    assert (
        table_counts[
            "jrc_euromod_be:EUROMOD Country Report Belgium 2025 validation tables"
        ]
        == 90
    )
    assert (
        table_counts[
            "eurostat:Eurostat nasa_10_nf_tr Non-financial transactions - "
            "annual data for Belgian households and NPISH"
        ]
        == 78
    )
    assert (
        table_counts[
            "eurostat:Eurostat gov_10a_taxag Main national accounts tax "
            "aggregates for Belgium, Germany, and France"
        ]
        == 24
    )
    assert (
        table_counts[
            "eurostat:Eurostat gov_10a_taxag Main national accounts tax "
            "aggregates for Belgium"
        ]
        == 12
    )
    assert (
        table_counts[
            "eurostat:Eurostat spr_exp_func Expenditure on social benefits by "
            "function for Belgium, Germany, and France"
        ]
        == 27
    )
    assert (
        table_counts[
            "eurostat:Eurostat spr_exp_func Expenditure on social benefits by "
            "function for Belgium"
        ]
        == 9
    )
    assert (
        table_counts[
            "fpb_economic_outlook:Economic Outlook 2026-2031, June 2026 statistical "
            "annex (T01, T06, T07, T11, T17, T24)"
        ]
        == 1000
    )
    assert (
        table_counts[
            "eurostat:Eurostat ilc_li02 At-risk-of-poverty rate by poverty "
            "threshold, age and sex - EU-SILC and ECHP surveys for Belgium, "
            "Germany, and France"
        ]
        == 3
    )
    assert (
        table_counts[
            "eurostat:Eurostat ilc_di01 Distribution of income by quantiles "
            "for Belgium, Germany, and France"
        ]
        == 54
    )
    expected_period_counts = {
        "academic_year:2013": 6,
        "academic_year:2014": 6,
        "academic_year:2015": 6,
        "academic_year:2016": 6,
        "academic_year:2017": 6,
        "academic_year:2018": 6,
        "academic_year:2019": 6,
        "academic_year:2020": 6,
        "academic_year:2021": 6,
        "academic_year:2022": 6,
        "academic_year:2023": 6,
        "academic_year:2024": 26,
        "academic_year:2025": 20,
        "academic_year:2026": 20,
        "academic_year:2027": 20,
        "academic_year:2028": 20,
        "academic_year:2029": 16,
        "calendar_year:1951": 3,
        "calendar_year:1961": 3,
        "calendar_year:1971": 3,
        "calendar_year:1981": 3,
        "calendar_year:1995": 3,
        "calendar_year:1996": 36,
        "calendar_year:1997": 36,
        "calendar_year:1998": 36,
        "calendar_year:1999": 36,
        "calendar_year:2000": 36,
        "calendar_year:2001": 36,
        "calendar_year:2002": 39,
        "calendar_year:2003": 39,
        "calendar_year:2004": 39,
        "calendar_year:2005": 39,
        "calendar_year:2006": 39,
        "calendar_year:2007": 39,
        "calendar_year:2008": 39,
        "calendar_year:2009": 39,
        "calendar_year:2010": 39,
        "calendar_year:2011": 39,
        "calendar_year:2012": 39,
        "calendar_year:2013": 72,
        "calendar_year:2014": 72,
        "calendar_year:2015": 74,
        "calendar_year:2016": 72,
        "calendar_year:2017": 72,
        "calendar_year:2018": 176,
        "calendar_year:2019": 175,
        "calendar_year:2020": 175,
        "calendar_year:2021": 4196,
        "calendar_year:2022": 2223,
        "calendar_year:2023": 6445,
        "calendar_year:2024": 34038,
        "calendar_year:2025": 4661,
        "calendar_year:2026": 341,
        "calendar_year:2027": 320,
        "calendar_year:2028": 320,
        "calendar_year:2029": 320,
        "calendar_year:2030": 100,
        "calendar_year:2031": 102,
        "fiscal_year:1996": 33,
        "fiscal_year:1997": 33,
        "fiscal_year:1998": 33,
        "fiscal_year:1999": 41,
        "fiscal_year:2000": 41,
        "fiscal_year:2001": 41,
        "fiscal_year:2002": 41,
        "fiscal_year:2003": 41,
        "fiscal_year:2004": 41,
        "fiscal_year:2005": 41,
        "fiscal_year:2006": 41,
        "fiscal_year:2007": 41,
        "fiscal_year:2008": 41,
        "fiscal_year:2009": 41,
        "fiscal_year:2010": 41,
        "fiscal_year:2011": 41,
        "fiscal_year:2012": 41,
        "fiscal_year:2013": 41,
        "fiscal_year:2014": 41,
        "fiscal_year:2015": 41,
        "fiscal_year:2016": 41,
        "fiscal_year:2017": 41,
        "fiscal_year:2018": 41,
        "fiscal_year:2019": 41,
        "fiscal_year:2020": 44,
        "fiscal_year:2021": 52,
        "fiscal_year:2022": 52,
        "fiscal_year:2023": 413,
        "fiscal_year:2024": 705,
        "fiscal_year:2025": 1318,
        "fiscal_year:2026": 1445,
        "fiscal_year:2027": 34,
        "fiscal_year:2028": 35,
        "fiscal_year:2029": 35,
        "fiscal_year:2030": 31,
        "month:2003-08": 7,
        "month:2004-08": 7,
        "month:2005-08": 7,
        "month:2006-08": 7,
        "month:2007-08": 7,
        "month:2008-08": 7,
        "month:2009-08": 7,
        "month:2010-08": 7,
        "month:2011-05": 21,
        "month:2011-08": 18,
        "month:2012-05": 21,
        "month:2012-08": 18,
        "month:2013-05": 21,
        "month:2013-08": 22,
        "month:2014-05": 21,
        "month:2014-08": 22,
        "month:2015-05": 21,
        "month:2015-08": 22,
        "month:2016-05": 21,
        "month:2016-08": 22,
        "month:2017-05": 21,
        "month:2017-08": 22,
        "month:2018-05": 21,
        "month:2018-08": 22,
        "month:2019-05": 21,
        "month:2019-08": 23,
        "month:2020-05": 21,
        "month:2020-08": 23,
        "month:2021-03": 1,
        "month:2021-04": 1,
        "month:2021-05": 22,
        "month:2021-06": 1,
        "month:2021-07": 1,
        "month:2021-08": 24,
        "month:2021-09": 1,
        "month:2021-10": 1,
        "month:2021-11": 1,
        "month:2021-12": 1,
        "month:2022-01": 1,
        "month:2022-02": 1,
        "month:2022-03": 1,
        "month:2022-04": 1,
        "month:2022-05": 22,
        "month:2022-06": 1,
        "month:2022-07": 1,
        "month:2022-08": 24,
        "month:2022-09": 1,
        "month:2022-10": 1,
        "month:2022-11": 1,
        "month:2022-12": 1,
        "month:2023-01": 380,
        "month:2023-02": 379,
        "month:2023-03": 387,
        "month:2023-04": 453,
        "month:2023-05": 474,
        "month:2023-06": 461,
        "month:2023-07": 453,
        "month:2023-08": 476,
        "month:2023-09": 461,
        "month:2023-10": 453,
        "month:2023-11": 453,
        "month:2023-12": 467,
        "month:2024-01": 448,
        "month:2024-02": 447,
        "month:2024-03": 455,
        "month:2024-04": 453,
        "month:2024-05": 474,
        "month:2024-06": 461,
        "month:2024-07": 453,
        "month:2024-08": 476,
        "month:2024-09": 461,
        "month:2024-10": 559,
        "month:2024-11": 559,
        "month:2024-12": 837,
        "month:2025-01": 555,
        "month:2025-02": 553,
        "month:2025-03": 683,
        "month:2025-04": 753,
        "month:2025-05": 6720,
        "month:2025-06": 669,
        "month:2025-07": 661,
        "month:2025-08": 1098,
        "month:2025-09": 677,
        "month:2025-10": 660,
        "month:2025-11": 675,
        "month:2025-12": 924,
        "month:2026-01": 449,
        "month:2026-02": 453,
        "month:2026-03": 454,
        "month:2026-04": 446,
        "month:2026-05": 445,
        "month:2026-06": 348,
        "tax_year:1987": 9,
        "tax_year:1988": 9,
        "tax_year:1989": 9,
        "tax_year:1990": 9,
        "tax_year:1991": 9,
        "tax_year:1992": 9,
        "tax_year:1993": 9,
        "tax_year:1994": 9,
        "tax_year:1995": 9,
        "tax_year:1996": 9,
        "tax_year:1997": 9,
        "tax_year:1998": 9,
        "tax_year:1999": 9,
        "tax_year:2000": 9,
        "tax_year:2001": 9,
        "tax_year:2002": 9,
        "tax_year:2003": 9,
        "tax_year:2004": 9,
        "tax_year:2005": 9,
        "tax_year:2006": 9,
        "tax_year:2007": 9,
        "tax_year:2008": 9,
        "tax_year:2009": 9,
        "tax_year:2010": 9,
        "tax_year:2011": 9,
        "tax_year:2012": 11,
        "tax_year:2013": 11,
        "tax_year:2014": 11,
        "tax_year:2015": 11,
        "tax_year:2016": 11,
        "tax_year:2017": 9,
        "tax_year:2018": 11,
        "tax_year:2019": 11,
        "tax_year:2020": 11,
        "tax_year:2021": 11,
        "tax_year:2022": 41239,
        "tax_year:2023": 63310,
        "tax_year:2024": 296,
    }
    for fiscal_year in range(2017, 2026):
        key = f"fiscal_year:{fiscal_year}"
        expected_period_counts[key] += 2
    for year, count in {
        2011: 4,
        2012: 18,
        2013: 18,
        2014: 18,
        2015: 32,
        2016: 32,
        2017: 32,
        2018: 64,
        2019: 64,
        2020: 64,
        2021: 64,
        2022: 64,
        2023: 64,
        2024: 64,
        2025: 84,
        2026: 84,
    }.items():
        key = f"month:{year}-01"
        expected_period_counts[key] = expected_period_counts.get(key, 0) + count
    year, month = 2017, 4
    while (year, month) <= (2026, 3):
        key = f"month:{year}-{month:02d}"
        expected_period_counts[key] = expected_period_counts.get(key, 0) + 1
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    assert coverage["counts"]["by_period"] == expected_period_counts
    assert coverage["counts"]["by_geography"]["country:BE"] == 4888
    assert coverage["counts"]["by_geography"]["country:DE"] == 36
    assert coverage["counts"]["by_geography"]["country:FR"] == 36
    assert coverage["counts"]["by_geography"]["nuts1:BE1"] == 3662
    assert coverage["counts"]["by_geography"]["nuts1:BE2"] == 3673
    assert coverage["counts"]["by_geography"]["nuts1:BE3"] == 3662
    assert coverage["counts"]["by_geography"]["commune:11002"] == 1
    assert coverage["counts"]["by_geography"]["country:0100000US"] == 2109
    assert coverage["counts"]["by_geography"]["state:0400000US06"] == 229
    assert (
        coverage["counts"]["by_geography"]["congressional_district:5001700US0601"] == 56
    )
    assert coverage["counts"]["by_geography"]["country:K02000001"] == 6373
    assert coverage["counts"]["by_geography"]["country:E92000001"] == 1437
    assert coverage["counts"]["by_geography"]["country:K03000001"] == 4676
    assert len(coverage["counts"]["by_geography"]) == 12539
    assert coverage["counts"]["by_entity"] == {
        "benefit_unit": 4680,
        "dwelling": 27041,
        "family": 1299,
        "firm": 1439,
        "government": 1350,
        "household": 40708,
        "institutional_sector": 261,
        "pension_plan": 2,
        "person": 63282,
        "return": 14600,
        "social_protection_scheme": 36,
        "tax_unit": 40495,
    }
    assert not coverage["duplicates"]["aggregate_fact_keys"]
    assert len(coverage["duplicates"]["semantic_fact_keys"]) == 177
    assert summary["warnings"] == [
        {
            "code": "duplicate_semantic_fact_key",
            "message": (
                "One or more semantic facts appear in multiple rows; downstream "
                "consumers should reconcile or select sources."
            ),
        }
    ]
    for source in (
        "dfe-funded-early-education-childcare-2026",
        "dft-bus0415-fares-index-2026",
        "dft-bus05i-revenue-support-2025",
        "dft-nts0705-local-bus-trips-2024",
        "dwp-uc-childcare-element-march-2021-august-2025",
        "dwp-uc-households-carer-entitlement-april-december-2025",
        "dwp-uc-households-children-april-december-2025",
        "dwp-uc-households-family-type-april-december-2025",
        "dwp-uc-households-housing-entitlement-april-december-2025",
        "dwp-uc-households-lcwra-entitlement-april-december-2025",
        "hmrc-tax-free-childcare-march-2026",
        "dfc-ni-uc-statistics-may-2026",
        "hmrc-cgt-age-2026",
        "hmrc-cgt-country-region-2026",
        "hmrc-cgt-gain-by-income-2026",
        "hmrc-cgt-size-of-gain-2026",
        "hmrc-cgt-statistics-2026",
    ):
        assert (output_dir / "sources" / source / "consumer_facts.jsonl").exists()
    for source in (
        "eurostat-gov-10a-taxag-2025",
        "eurostat-spr-exp-func-2024",
        "fpb-economic-outlook-2026-2031-june-2026",
        "statbel-fiscal-income-distribution-2023",
        "statbel-population-structure-2025",
    ):
        assert (output_dir / "sources" / source / "consumer_facts.jsonl").exists()
    assert (output_dir / "sources" / "soi-table-1-1" / "consumer_facts.jsonl").exists()
    assert (
        output_dir / "sources" / "soi-table-1-4" / "reports" / "build_summary.json"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "hhs-acf-liheap-fy2024-national-profile"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "soi-ira-roth-contributions-2022"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "census-stc-individual-income-tax"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "cms-medicare-trustees-report-2025-part-b-premium-income"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "cms-nhe-historical-service-source"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir / "sources" / "cms-nhe-table-24" / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "federal-reserve-z1-household-net-worth"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir / "sources" / "usda-snap-fy69-to-current" / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "usda-snap-fy2025-monthly-state-caseloads"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir / "sources" / "soi-historic-table-2" / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir / "sources" / "hhs-acf-tanf-caseload-2024" / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir / "sources" / "hhs-acf-tanf-financial-2024" / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "soi-congressional-district-2022"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "cbo-revenue-projections-income-by-source-2026-02"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "bea-regional-state-personal-income-components-2024"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir / "sources" / "ssa-ssi-table-7b1-2024" / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "ssa-ssi-monthly-statistics-2024-12"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "cms-medicaid-chip-monthly-enrollment-december-2024"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "soi-historic-table-2-state-broad-2022"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "kff-marketplace-effectuated-enrollment"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "ons-uk-business-firm-targets-2025"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "ons-uk-business-firm-sector-targets-2025"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "ons-households-by-type-country-2025"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir / "sources" / "hmrc-cgt-size-of-gain-2026" / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "hmrc-vat-firm-targets-2024-25"
        / "consumer_facts.jsonl"
    ).exists()
    assert (
        output_dir
        / "sources"
        / "hmrc-vat-firm-sector-targets-2024-25"
        / "consumer_facts.jsonl"
    ).exists()
    for source in (
        "eurostat-gov-10a-taxag",
        "eurostat-spr-exp-func",
        "eurostat-nasa-10-nf-tr",
        "eurostat-ilc-li02",
        "eurostat-ilc-di01",
    ):
        assert (output_dir / "sources" / source / "consumer_facts.jsonl").exists()


def test_build_bundle_cli_supports_explicit_sources(tmp_path, capsys):
    output_dir = tmp_path / "bundle"

    exit_code = harness_main(
        [
            "build-bundle",
            "--year",
            "2023",
            "--source",
            "soi-table-1-1",
            "--out",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["valid"]
    assert payload["counts"]["source_package_count"] == 1
    assert payload["counts"]["fact_count"] == 80
    assert payload["outputs"]["consumer_facts"] == str(
        output_dir / "consumer_facts.jsonl"
    )
    assert payload["coverage"]["counts"]["by_source_table"] == {
        "irs_soi:Publication 1304 Table 1.1": 80
    }


def test_build_bundle_cli_supports_historic_table_2_source(tmp_path, capsys):
    output_dir = tmp_path / "bundle"

    exit_code = harness_main(
        [
            "build-bundle",
            "--year",
            "2023",
            "--source",
            "soi-historic-table-2",
            "--out",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["valid"]
    assert payload["counts"]["source_package_count"] == 1
    assert payload["counts"]["fact_count"] == 605
    assert payload["coverage"]["counts"]["by_source_table"] == {
        "irs_soi:Historic Table 2": 605
    }
    assert (
        output_dir / "sources" / "soi-historic-table-2" / "source_rows.jsonl"
    ).exists()


def test_build_bundle_cli_supports_ssa_supplement_source(tmp_path, capsys):
    output_dir = tmp_path / "bundle"

    exit_code = harness_main(
        [
            "build-bundle",
            "--year",
            "2024",
            "--source",
            "ssa-annual-statistical-supplement-2025",
            "--out",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    rows = _load_jsonl(output_dir / "consumer_facts.jsonl")

    assert exit_code == 0
    assert payload["valid"]
    assert payload["counts"]["source_package_count"] == 1
    assert payload["counts"]["fact_count"] == 6
    assert payload["coverage"]["counts"]["by_source"] == {"ssa": 6}
    assert payload["coverage"]["counts"]["by_entity"] == {"person": 6}
    assert {row["universe_constraints"]["constraints"][0]["value"] for row in rows} == {
        "social_security_benefits",
        "social_security_retirement_benefits",
        "social_security_survivors_benefits",
        "social_security_disability_benefits",
        "social_security_dependents_benefits",
        "ssi_payments",
    }


def test_build_bundle_cli_supports_jct_tax_expenditure_source(tmp_path, capsys):
    output_dir = tmp_path / "bundle"

    exit_code = harness_main(
        [
            "build-bundle",
            "--year",
            "2024",
            "--source",
            "jct-tax-expenditures-2024",
            "--out",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    rows = _load_jsonl(output_dir / "consumer_facts.jsonl")

    assert exit_code == 0
    assert payload["valid"]
    assert payload["counts"]["source_package_count"] == 1
    assert payload["counts"]["fact_count"] == 11
    assert payload["coverage"]["counts"]["by_source"] == {"jct": 11}
    assert payload["coverage"]["counts"]["by_entity"] == {"tax_unit": 11}
    assert {row["lineage"]["source_record_id"] for row in rows} == {
        "jct.tax_expenditures.cy2024.salt_deduction.revenue_loss",
        "jct.tax_expenditures.cy2024.medical_expense_deduction.revenue_loss",
        "jct.tax_expenditures.cy2024.charitable_deduction.revenue_loss",
        "jct.tax_expenditures.cy2024.deductible_mortgage_interest.revenue_loss",
        "jct.tax_expenditures.cy2024.qualified_business_income_deduction.revenue_loss",
        # JCX-48-24 extension (microcosm#514 anchors):
        "jct.tax_expenditures.cy2024.self_employed_health_insurance_deduction.revenue_loss",
        "jct.tax_expenditures.cy2024.health_savings_account_deduction.revenue_loss",
        "jct.tax_expenditures.cy2024.student_loan_interest_deduction.revenue_loss",
        "jct.tax_expenditures.cy2024.self_employed_pension_contribution_deduction.revenue_loss",
        "jct.tax_expenditures.cy2024.traditional_ira_deduction.revenue_loss",
        "jct.tax_expenditures.cy2024.cdcc_and_employer_child_care_exclusion.revenue_loss",
    }


def test_build_bundle_cli_supports_jct_obbba_source(tmp_path, capsys):
    output_dir = tmp_path / "bundle"

    exit_code = harness_main(
        [
            "build-bundle",
            "--year",
            "2026",
            "--source",
            "jct-obbba-revenue-estimates-2025",
            "--out",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    rows = _load_jsonl(output_dir / "consumer_facts.jsonl")

    assert exit_code == 0
    assert payload["valid"]
    assert payload["counts"]["source_package_count"] == 1
    assert payload["counts"]["fact_count"] == 2
    assert payload["coverage"]["counts"]["by_source"] == {"jct": 2}
    assert payload["coverage"]["counts"]["by_entity"] == {"tax_unit": 2}
    assert {row["lineage"]["source_record_id"] for row in rows} == {
        "jct.obbba_title_vii.fy2026.no_tax_on_tips.revenue_effect",
        "jct.obbba_title_vii.fy2026.no_tax_on_overtime.revenue_effect",
    }
    values = {row["lineage"]["source_record_id"]: row["value"] for row in rows}
    # JCX-35-25 FY2026: tips -$10,121M, overtime -$32,806M.
    assert (
        values["jct.obbba_title_vii.fy2026.no_tax_on_tips.revenue_effect"]
        == -10_121_000_000
    )
    assert (
        values["jct.obbba_title_vii.fy2026.no_tax_on_overtime.revenue_effect"]
        == -32_806_000_000
    )


def test_build_bundle_coverage_reports_duplicate_keys():
    rows = [
        {
            "aggregate_fact_key": "ledger.aggregate_fact.v2:a",
            "semantic_fact_key": "ledger.semantic_fact.v2:s",
            "legacy_fact_key": "ledger.fact.v1:one",
            "source": {
                "source_name": "irs_soi",
                "source_table": "Publication 1304 Table 1.1",
            },
            "period": {"type": "tax_year", "value": 2023},
            "geography": {"level": "country", "id": "0100000US"},
            "entity": {"name": "tax_unit"},
            "observed_measure": {
                "source_name": "irs_soi",
                "source_measure_id": "return_count",
                "source_concept": "irs_soi.individual_income_tax_returns",
            },
        },
        {
            "aggregate_fact_key": "ledger.aggregate_fact.v2:a",
            "semantic_fact_key": "ledger.semantic_fact.v2:s",
            "legacy_fact_key": "ledger.fact.v1:two",
            "source": {
                "source_name": "irs_soi",
                "source_table": "Publication 1304 Table 1.1",
            },
            "period": {"type": "tax_year", "value": 2023},
            "geography": {"level": "country", "id": "0100000US"},
            "entity": {"name": "tax_unit"},
            "observed_measure": {
                "source_name": "irs_soi",
                "source_measure_id": "return_count",
                "source_concept": "irs_soi.individual_income_tax_returns",
            },
        },
    ]

    coverage = build_bundle_coverage(
        rows,
        aggregate_duplicates=[
            {
                "key": "ledger.aggregate_fact.v2:a",
                "count": 2,
                "sources": ["irs_soi:Publication 1304 Table 1.1"],
                "legacy_fact_keys": ["ledger.fact.v1:one", "ledger.fact.v1:two"],
            }
        ],
        semantic_duplicates=[
            {
                "key": "ledger.semantic_fact.v2:s",
                "count": 2,
                "sources": ["irs_soi:Publication 1304 Table 1.1"],
                "legacy_fact_keys": ["ledger.fact.v1:one", "ledger.fact.v1:two"],
            }
        ],
    )

    assert coverage["counts"]["by_source"] == {"irs_soi": 2}
    assert coverage["duplicates"]["aggregate_fact_keys"][0]["count"] == 2
    assert coverage["duplicates"]["semantic_fact_keys"][0]["count"] == 2


def test_bundle_jsonl_ingestion_accepts_chronicle_only_rows(tmp_path):
    row = _row_for_epoch(_fixture_consumer_rows()[0], Epoch.CHRONICLE)
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text(json.dumps(row, sort_keys=True) + "\n")

    loaded = load_bundle_jsonl(path)

    assert loaded == [row]
    assert loaded[0]["schema_version"] == "chronicle.consumer_fact.v2"
    assert loaded[0]["aggregate_fact_key"].startswith("chronicle.aggregate_fact.v3:")


def test_bundle_jsonl_ingestion_accepts_mixed_epoch_rows(tmp_path):
    ledger_row, chronicle_source = _fixture_consumer_rows()[:2]
    chronicle_row = _row_for_epoch(chronicle_source, Epoch.CHRONICLE)
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (ledger_row, chronicle_row)
        )
    )

    loaded = load_bundle_jsonl(path)

    assert loaded == [ledger_row, chronicle_row]
    assert {row["schema_version"] for row in loaded} == {
        "ledger.consumer_fact.v1",
        "chronicle.consumer_fact.v2",
    }


def test_bundle_jsonl_ingestion_rejects_unknown_key_domain(tmp_path):
    row = _row_for_epoch(_fixture_consumer_rows()[0], Epoch.CHRONICLE)
    digest = row["aggregate_fact_key"].partition(":")[2]
    row["aggregate_fact_key"] = f"future.aggregate_fact.v4:{digest}"
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text(json.dumps(row, sort_keys=True) + "\n")

    with pytest.raises(ValueError) as error:
        load_bundle_jsonl(path)

    message = str(error.value)
    assert "ledger.aggregate_fact.v2" in message
    assert "chronicle.aggregate_fact.v3" in message


def test_bundle_coverage_canonicalizes_cross_epoch_identities():
    ledger_row = _fixture_consumer_rows()[0]
    chronicle_row = _row_for_epoch(ledger_row, Epoch.CHRONICLE)

    coverage = build_bundle_coverage([ledger_row, chronicle_row])

    assert BUNDLE_SCHEMA_VERSION == "ledger.bundle.v1"
    assert BUNDLE_COVERAGE_SCHEMA_VERSION == "ledger.bundle_coverage.v1"
    assert BUNDLE_SOURCES_SCHEMA_VERSION == "ledger.bundle_sources.v1"
    assert coverage["unique_counts"] == {
        "aggregate_fact_key": 1,
        "semantic_fact_key": 1,
        "source_release_key": 1,
        "source_series_key": 1,
        "observed_measure_key": 1,
        "dimension_set_key": 1,
        "universe_constraint_set_key": 1,
    }
    assert coverage["duplicates"]["aggregate_fact_keys"] == [
        {
            "key": ledger_row["aggregate_fact_key"],
            "count": 2,
            "sources": ["irs_soi:Publication 1304 Table 1.1"],
            "legacy_fact_keys": [ledger_row["legacy_fact_key"]],
        }
    ]


def test_bundle_coverage_preserves_non_string_identity_scalars(tmp_path):
    identity_fields = (
        "aggregate_fact_key",
        "semantic_fact_key",
        "legacy_fact_key",
        "source_release_key",
        "source_series_key",
        "observed_measure_key",
        "dimension_set_key",
        "universe_constraint_set_key",
    )
    rows = [_fixture_consumer_rows()[0]]
    for value in (None, 7):
        row = json.loads(json.dumps(rows[0]))
        for field_name in identity_fields:
            row[field_name] = value
        rows.append(row)
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    loaded = load_bundle_jsonl(path)
    coverage = build_bundle_coverage(loaded)

    for field_name in identity_fields:
        assert [row[field_name] for row in loaded] == [
            rows[0][field_name],
            None,
            7,
        ]
    assert coverage["unique_counts"] == {
        "aggregate_fact_key": 3,
        "semantic_fact_key": 3,
        "source_release_key": 3,
        "source_series_key": 3,
        "observed_measure_key": 3,
        "dimension_set_key": 3,
        "universe_constraint_set_key": 3,
    }
    assert coverage["duplicates"] == {
        "aggregate_fact_keys": [],
        "semantic_fact_keys": [],
    }
