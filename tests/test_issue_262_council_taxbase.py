"""chronicle#262: council tax stock by band from the councils' own taxbase returns.

England: MHCLG Council Taxbase (CTB) local authority level data, October 2023 to
October 2025. Wales: StatsWales CT1 council tax dwellings, financial years
2023-24 to 2026-27. Scotland: CTAXBASE chargeable dwellings, September 2023 to
September 2025, for Scotland and the 32 council areas. Every oracle below is a
publisher cell; the England and Wales totals are the figures the releases quote.
"""

from __future__ import annotations

import functools
import json

import pytest
import yaml

from chronicle.source_package import load_source_package, resolve_source_package_path
from chronicle.suite import build_source_suite

ENGLAND = {
    2023: "mhclg-council-taxbase-england-2023",
    2024: "mhclg-council-taxbase-england-2024",
    2025: "mhclg-council-taxbase-england-2025",
}
SCOTLAND = {
    2023: "scotgov-council-tax-bands-2023",
    2024: "scotgov-council-tax-bands-2024",
    2025: "scotgov-council-tax-bands-2025",
}
WALES = "welshgov-council-tax-dwellings-2023-24-to-2026-27"
ALL_262 = [
    *((alias, year) for year, alias in ENGLAND.items()),
    *((alias, year) for year, alias in SCOTLAND.items()),
    (WALES, 2026),
]

CTB = "mhclg.council_taxbase."
CT1 = "welshgov.council_tax_dwellings."
CTB_LINES = {
    "dwellings_on_valuation_list": 1,
    "exempt_dwellings": 2,
    "chargeable_dwellings": 4,
    "chargeable_dwellings_adjusted_for_disabled_relief": 7,
    "single_adult_discount_dwellings": 8,
    "second_homes": 11,
    "empty_dwellings_no_discount_or_premium": 12,
    "empty_dwellings_with_discount": 13,
    "empty_dwellings_with_premium": 14,
    "empty_dwellings": 15,
    "empty_dwellings_over_six_months": 16,
    "empty_dwellings_over_six_months_class_d": 17,
    "dwelling_equivalents": 22,
    "band_d_equivalents": 24,
}
A_MINUS_LINES = {
    "chargeable_dwellings_adjusted_for_disabled_relief",
    "single_adult_discount_dwellings",
    "dwelling_equivalents",
    "band_d_equivalents",
}


@functools.cache
def _facts(alias: str, year: int) -> tuple:
    package = load_source_package(alias)
    rows = package.build_source_rows(year)
    cells = package.build_source_cells(year, source_rows=rows)
    return tuple(package.build_facts(year, cells=cells, source_rows=rows))


@functools.cache
def _index(alias: str, year: int) -> dict:
    index = {}
    for fact in _facts(alias, year):
        key = (
            fact.measure.concept,
            fact.geography.id,
            fact.period.value,
            fact.filters.get("council_tax_band"),
        )
        assert key not in index, key
        index[key] = fact.value
    return index


def _england(year: int, line: str, geography: str = "E92000001", band=None):
    return _index(ENGLAND[year], year)[(CTB + line, geography, f"{year}-10", band)]


@pytest.mark.parametrize("year", sorted(ENGLAND))
def test_england_packages_carry_fourteen_ctb_lines_by_band(year):
    facts = _facts(ENGLAND[year], year)

    assert len(facts) == 38_610
    assert {fact.period.type for fact in facts} == {"month"}
    assert {fact.period.value for fact in facts} == {f"{year}-10"}
    assert {fact.measure.concept for fact in facts} == {
        CTB + line for line in CTB_LINES
    }
    assert {fact.provenance_class for fact in facts} == {"administrative"}
    assert all(fact.source.raw_r2_uri for fact in facts)
    geographies = {(fact.geography.id, fact.geography.level) for fact in facts}
    assert len(geographies) == 297
    assert ("E92000001", "country") in geographies
    assert {level for _id, level in geographies} == {"country", "local_authority"}
    assert {key for fact in facts for key in fact.filters} == {"council_tax_band"}
    bands_by_line = {}
    for fact in facts:
        bands_by_line.setdefault(fact.measure.concept, set()).add(
            fact.filters.get("council_tax_band")
        )
    for line in CTB_LINES:
        expected = {None, *"ABCDEFGH"}
        if line in A_MINUS_LINES:
            expected.add("A-")
        assert bands_by_line[CTB + line] == expected, line
    roles = {fact.measure.concept: fact.entity.role for fact in facts}
    assert roles[CTB + "dwellings_on_valuation_list"] == "banded_property"
    assert roles[CTB + "exempt_dwellings"] == "exempt_dwelling"
    assert roles[CTB + "chargeable_dwellings"] == "chargeable_dwelling"


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        # listed, exempt, chargeable, single adult, empty, second homes, line 7 'A-'
        (2023, (25_462_055, 741_398, 24_718_212, 8_388_189, 480_845, 263_318, 15_947)),
        (2024, (25_675_421, 747_958, 24_925_917, 8_409_217, 502_263, 279_870, 16_501)),
        (2025, (25_817_220, 758_040, 25_056_421, 8_496_178, 542_260, 267_894, 17_102)),
    ],
)
def test_england_row_matches_the_release(year, expected):
    # The 2025 row is the release's '25.8m listed, 758k exempt, 25.1m chargeable,
    # 8.5m single adult, 542k empty, 268k second homes' quoted in chronicle#262.
    assert (
        _england(year, "dwellings_on_valuation_list"),
        _england(year, "exempt_dwellings"),
        _england(year, "chargeable_dwellings"),
        _england(year, "single_adult_discount_dwellings"),
        _england(year, "empty_dwellings"),
        _england(year, "second_homes"),
        _england(year, "chargeable_dwellings_adjusted_for_disabled_relief", band="A-"),
    ) == expected


@pytest.mark.parametrize("year", sorted(ENGLAND))
def test_england_publisher_identities(year):
    index = _index(ENGLAND[year], year)
    period = f"{year}-10"
    authorities = sorted({geo for (_c, geo, _p, _b) in index if geo != "E92000001"})
    assert len(authorities) == 296

    for line in (
        "dwellings_on_valuation_list",
        "chargeable_dwellings",
        "second_homes",
        "empty_dwellings",
    ):
        assert (
            sum(index[(CTB + line, geo, period, None)] for geo in authorities)
            == (index[(CTB + line, "E92000001", period, None)])
        ), line

    for geo in ["E92000001", *authorities]:
        for band in (None, *"ABCDEFGH"):
            assert index[(CTB + "empty_dwellings", geo, period, band)] == sum(
                index[(CTB + part, geo, period, band)]
                for part in (
                    "empty_dwellings_no_discount_or_premium",
                    "empty_dwellings_with_discount",
                    "empty_dwellings_with_premium",
                )
            )

    mismatched = {
        geo: (
            index[(CTB + "chargeable_dwellings", geo, period, None)],
            index[
                (
                    CTB + "chargeable_dwellings_adjusted_for_disabled_relief",
                    geo,
                    period,
                    None,
                )
            ],
        )
        for geo in ["E92000001", *authorities]
        if index[(CTB + "chargeable_dwellings", geo, period, None)]
        != index[
            (
                CTB + "chargeable_dwellings_adjusted_for_disabled_relief",
                geo,
                period,
                None,
            )
        ]
    }
    if year == 2023:
        # Carried as published: Buckinghamshire's line 7 total is 206 above its line 4.
        assert mismatched == {
            "E06000060": (230_534, 230_740),
            "E92000001": (24_718_212, 24_718_418),
        }
        single = [
            index[(CTB + "single_adult_discount_dwellings", "E08000012", period, band)]
            for band in ("A-", *"ABCDEFGH")
        ]
        assert (
            sum(single),
            index[(CTB + "single_adult_discount_dwellings", "E08000012", period, None)],
        ) == (
            93_656,
            93_657,
        )
    else:
        assert mismatched == {}


def test_england_codes_follow_the_publisher_and_the_voa_roster():
    voa = yaml.safe_load(
        resolve_source_package_path("voa-council-tax-stock-by-lad-2025").read_text()
    )
    voa_english = {
        row["geography_id"]
        for record_set in voa["record_sets"]
        for row in record_set.get("rows") or ()
        if row.get("geography_id", "").startswith("E0")
    }
    assert len(voa_english) == 296

    codes = {}
    for year, alias in ENGLAND.items():
        codes[year] = {
            fact.geography.id
            for fact in _facts(alias, year)
            if fact.geography.level == "local_authority"
        }
    assert codes[2023] == codes[2024] == voa_english
    renamed = {"E08000016": "E08000038", "E08000019": "E08000039"}
    assert codes[2025] == {renamed.get(code, code) for code in voa_english}


def test_wales_package_carries_ct1_lines_for_four_financial_years():
    facts = _facts(WALES, 2026)

    assert len(facts) == 8_646
    assert {fact.period.type for fact in facts} == {"fiscal_year"}
    per_year = {}
    for fact in facts:
        per_year[fact.period.value] = per_year.get(fact.period.value, 0) + 1
    assert per_year == {2023: 2_146, 2024: 2_157, 2025: 2_168, 2026: 2_175}
    assert all(fact.source.raw_r2_uri for fact in facts)
    geographies = {(fact.geography.id, fact.geography.level) for fact in facts}
    assert len(geographies) == 23
    assert ("W92000004", "country") in geographies
    assert {geo[:3] for geo, _level in geographies} == {"W92", "W06"}
    bands = {}
    for fact in facts:
        bands.setdefault(fact.measure.concept, set()).add(
            fact.filters.get("council_tax_band")
        )
    assert bands[CT1 + "all_chargeable_dwellings"] == {None, *"ABCDEFGHI"}
    assert bands[CT1 + "adjusted_chargeable_dwellings"] == {None, "A-", *"ABCDEFGHI"}
    assert bands[CT1 + "total_chargeable_second_homes"] == {None, *"ABCDEFGHI"}
    assert bands[CT1 + "exempt_dwellings_class_o"] == {None}
    assert bands[CT1 + "long_term_empty_premium_total"] == {None}
    roles = {fact.measure.concept: fact.entity.role for fact in facts}
    assert (
        roles[CT1 + "exempt_dwellings_classes_a_to_n_and_p_to_w"] == "exempt_dwelling"
    )
    assert roles[CT1 + "all_chargeable_dwellings"] == "chargeable_dwelling"


@pytest.mark.parametrize(
    ("year", "chargeable", "exempt", "empty", "second_homes"),
    [
        # Release Table 3 figures; StatsWales rounds to 2 decimal places.
        (2023, 1_411_081.68, 60_223.98, 22_457, 24_170),
        (2024, 1_416_847.41, 64_308.88, 22_633.6, 21_930.5),
        (2025, 1_424_158.03, 64_273.71, 22_558, 23_967),
        (2026, 1_433_016.47, 64_323.47, 23_033, 26_174),
    ],
)
def test_wales_totals_match_the_releases(year, chargeable, exempt, empty, second_homes):
    index = _index(WALES, 2026)

    def value(line, geography="W92000004", band=None):
        return index[(CT1 + line, geography, year, band)]

    assert value("all_chargeable_dwellings") == pytest.approx(chargeable, abs=1e-6)
    assert value("exempt_dwellings_classes_a_to_n_and_p_to_w") + value(
        "exempt_dwellings_class_o"
    ) == pytest.approx(exempt, abs=0.011)
    assert value("total_chargeable_empty_properties") == pytest.approx(empty, abs=1e-6)
    assert value("total_chargeable_second_homes") == pytest.approx(
        second_homes, abs=1e-6
    )

    authorities = sorted(
        {
            geo
            for (concept, geo, period, _b) in index
            if period == year and geo != "W92000004"
        }
    )
    assert len(authorities) == 22
    assert sum(
        value("all_chargeable_dwellings", geo) for geo in authorities
    ) == pytest.approx(chargeable, abs=0.02)
    assert sum(
        value("all_chargeable_dwellings", band=band) for band in "ABCDEFGHI"
    ) == pytest.approx(chargeable, abs=0.05)


def test_wales_band_i_and_disability_band_follow_the_release():
    index = _index(WALES, 2026)
    # Release Table 1, 2026-27: Cardiff band I 1,453.
    assert index[(CT1 + "all_chargeable_dwellings", "W06000015", 2026, "I")] == 1_453
    # A3 moves disability-reduced band A dwellings to 'A-' (charged at 5/9).
    assert index[(CT1 + "adjusted_chargeable_dwellings", "W06000001", 2026, "A-")] == 10
    assert (
        index[(CT1 + "adjusted_chargeable_dwellings", "W06000001", 2026, None)]
        == 35_033
    )


@pytest.mark.parametrize(
    ("year", "scotland_total"),
    [(2023, 2_581_728), (2024, 2_602_545), (2025, 2_623_149)],
)
def test_scotland_council_rows_parent_under_the_scotland_row(year, scotland_total):
    facts = _facts(SCOTLAND[year], year)

    assert len(facts) == 297
    assert {fact.period.value for fact in facts} == {f"{year}-09"}
    scotland = {
        fact.measure.concept: fact.value
        for fact in facts
        if fact.geography.id == "S92000003"
    }
    councils = [fact for fact in facts if fact.geography.level == "local_authority"]
    assert len({fact.geography.id for fact in councils}) == 32
    assert {fact.geography.id[:3] for fact in councils} == {"S12"}
    assert {fact.measure.concept for fact in councils} == set(scotland)
    for concept, total in scotland.items():
        assert (
            sum(f.value for f in councils if f.measure.concept == concept) == total
        ), concept
    assert scotland["scotgov.chargeable_dwellings_total"] == scotland_total
    assert all(not fact.filters for fact in facts)


def test_scotland_2025_country_rows_are_unchanged():
    facts = _facts(SCOTLAND[2025], 2025)
    scotland = {
        fact.layout.measure_id: fact.value
        for fact in facts
        if fact.geography.id == "S92000003"
    }
    assert scotland == {
        "band_a": 498_707,
        "band_b": 583_705,
        "band_c": 426_388,
        "band_d": 369_621,
        "band_e": 366_881,
        "band_f": 221_282,
        "band_g": 142_084,
        "band_h": 14_481,
        "total": 2_623_149,
    }
    record_sets = {fact.source_record_id.rsplit(".", 2)[0] for fact in facts}
    assert any(
        r.startswith("scotgov.ctaxbase2025.chargeable_dwellings.scotland")
        for r in record_sets
    )


@pytest.mark.parametrize(("alias", "year"), ALL_262)
def test_issue_262_packages_pass_agent_acceptance(alias, year, tmp_path):
    output_dir = tmp_path / alias
    report = build_source_suite(alias, output_dir, year=year)
    acceptance = json.loads(
        (output_dir / "reports" / "agent_acceptance.json").read_text()
    )

    assert report.valid
    assert acceptance["valid"]
    assert acceptance["counts"]["row_semantic_error_count"] == 0
