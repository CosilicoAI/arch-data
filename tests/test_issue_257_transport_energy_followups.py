"""Regression coverage for the UK transport and energy follow-ups in issue 257."""

from __future__ import annotations

import pytest

from chronicle.bundle import UK_BUNDLE_SOURCES
from chronicle.consumer_contract import validate_consumer_fact_contract
from chronicle.core import validate_facts
from chronicle.source_package import (
    SOURCE_PACKAGE_ALIASES,
    load_source_package,
    validate_source_package,
)
from chronicle.sources.cells import validate_source_cells


ISSUE_257_ARTIFACT_YEARS = {
    "dft-nts0313-mode-use-frequency-2025": 2025,
    "dft-nts0621-local-bus-use-frequency-2025": 2025,
    "nithc-annual-report-accounts-2024-25": 2025,
    "ofgem-energy-price-cap-levels-2024-2026": 2026,
    "scotgov-bus-coach-statistics-2024-25": 2025,
    "welshgov-transport-revenue-outturn-2024-25": 2025,
}

EXPECTED_FACT_COUNTS = {
    "dft-nts0313-mode-use-frequency-2025": 1332,
    "dft-nts0621-local-bus-use-frequency-2025": 198,
    "nithc-annual-report-accounts-2024-25": 8,
    "ofgem-energy-price-cap-levels-2024-2026": 3636,
    "scotgov-bus-coach-statistics-2024-25": 10,
    "welshgov-transport-revenue-outturn-2024-25": 828,
}

REPRESENTATIVE_PUBLISHER_FACTS = {
    "dft-nts0313-mode-use-frequency-2025": (
        "dft.nts0313.once_or_twice_a_week.cy2025.local_bus.once_or_twice_a_week",
        9.23299228690409,
    ),
    "dft-nts0621-local-bus-use-frequency-2025": (
        "dft.nts0621.3_or_more_times_a_week.cy2025.cy2025.3_or_more_times_a_week",
        12.9723580412945,
    ),
    "nithc-annual-report-accounts-2024-25": (
        "nithc.dfi_transactions.fy2024.concessionary_fare_compensation.transaction_value",
        49_900_000,
    ),
    "ofgem-energy-price-cap-levels-2024-2026": (
        "ofgem.price_cap.direct_debit.electricity_single_rate."
        "benchmark_consumption.2024_q2.london.cap_level",
        901.1677982175162,
    ),
    "scotgov-bus-coach-statistics-2024-25": (
        "scotgov.bus.table_2_9.fy2024.concessionary_fares.support",
        392_000_000,
    ),
    "welshgov-transport-revenue-outturn-2024-25": (
        "welshgov.local_transport.concessionary_fares.gross_expenditure."
        "fy2024.wales.gross_expenditure",
        62_443_460,
    ),
}

OFGEM_CHARGE_RESTRICTION_REGIONS = {
    "ofgem:north_west",
    "ofgem:northern",
    "ofgem:yorkshire",
    "ofgem:northern_scotland",
    "ofgem:southern",
    "ofgem:southern_scotland",
    "ofgem:n_wales_and_mersey",
    "ofgem:london",
    "ofgem:south_east",
    "ofgem:eastern",
    "ofgem:east_midlands",
    "ofgem:midlands",
    "ofgem:southern_western",
    "ofgem:south_wales",
}

OFGEM_CAP_PERIODS = {
    f"{year}-Q{quarter}" for year in (2024, 2025, 2026) for quarter in (1, 2, 3, 4)
}

NTS_FREQUENCY_BANDS = {
    "3 or more times a week (%)",
    "Once or twice a week (%)",
    "Less than once a week, more than once or twice a month (%)",
    "Once or twice a month (%)",
    "Less than once a month, more than once or twice a year (%)",
    "Once or twice a year (%)",
    "Less than once a year or never (%)",
}


def _facts(alias):
    return load_source_package(alias).build_facts(ISSUE_257_ARTIFACT_YEARS[alias])


def test_issue_257_packages_are_registered():
    assert set(ISSUE_257_ARTIFACT_YEARS) <= set(SOURCE_PACKAGE_ALIASES)
    assert set(ISSUE_257_ARTIFACT_YEARS) <= set(UK_BUNDLE_SOURCES)


@pytest.mark.parametrize(
    ("alias", "year"),
    sorted(ISSUE_257_ARTIFACT_YEARS.items()),
)
def test_issue_257_packages_build_valid_consumer_facts(alias, year):
    package = load_source_package(alias)
    report = validate_source_package(package.package_path, year=year)
    cells = package.build_source_cells(year)
    facts = package.build_facts(year, cells=cells)

    assert report.valid, report.to_dict()
    assert cells
    assert len(facts) == EXPECTED_FACT_COUNTS[alias]
    assert validate_source_cells(cells).valid
    assert validate_facts(facts).valid
    assert validate_consumer_fact_contract(facts).valid


@pytest.mark.parametrize(
    ("alias", "expected"),
    sorted(REPRESENTATIVE_PUBLISHER_FACTS.items()),
)
def test_issue_257_packages_preserve_representative_publisher_values(alias, expected):
    source_record_id, expected_value = expected
    facts = {fact.source_record_id: fact for fact in _facts(alias)}

    assert facts[source_record_id].value == pytest.approx(expected_value)


def test_issue_257_facts_declare_exact_period_coverage():
    for alias in ISSUE_257_ARTIFACT_YEARS:
        facts = _facts(alias)

        assert all(fact.period_coverage is not None for fact in facts)
        assert all(fact.period_coverage.start_date for fact in facts)
        assert all(fact.period_coverage.end_date for fact in facts)
        assert all(fact.period_coverage.basis for fact in facts)


def _ofgem_cap_levels():
    return [
        fact
        for fact in _facts("ofgem-energy-price-cap-levels-2024-2026")
        if fact.measure.concept == "ofgem.price_cap.cap_level"
    ]


def test_ofgem_cap_levels_cover_every_period_region_and_payment_method():
    facts = _ofgem_cap_levels()
    regional = [fact for fact in facts if fact.geography.level == "statistical_scope"]

    assert {fact.period.type for fact in facts} == {"quarter"}
    assert {fact.period.value for fact in facts} == OFGEM_CAP_PERIODS
    assert {fact.geography.id for fact in regional} == OFGEM_CHARGE_RESTRICTION_REGIONS
    assert {fact.filters["payment_method"] for fact in facts} == {
        "direct_debit",
        "standard_credit",
        "prepayment",
    }
    assert {fact.filters["consumption_level"] for fact in facts} == {
        "nil_consumption",
        "benchmark_consumption",
    }
    # 12 cap periods x 3 payment methods x 3 fuel and metering arrangements
    # x 2 consumption levels x 14 regions.
    assert len(regional) == 3024


def test_ofgem_cap_levels_are_published_annual_levels_not_derived_rates():
    facts = _ofgem_cap_levels()

    assert {fact.measure.unit for fact in facts} == {"gbp_per_year"}
    assert all(fact.aggregation.method == "mean" for fact in facts)


def test_ofgem_publishes_the_benchmark_consumption_each_cap_level_is_set_at():
    facts = [
        fact
        for fact in _facts("ofgem-energy-price-cap-levels-2024-2026")
        if fact.measure.concept == "ofgem.price_cap.benchmark_consumption"
    ]
    by_period_and_fuel = {
        (fact.period.value, fact.filters["fuel"]): fact.value for fact in facts
    }

    assert {fact.measure.unit for fact in facts} == {"kwh"}
    assert {fact.period.value for fact in facts} == OFGEM_CAP_PERIODS
    # Ofgem moved the benchmark twice inside the ported window, so a consumer
    # deriving a unit rate cannot divide by one constant.
    assert by_period_and_fuel[("2024-Q1", "electricity_single_rate")] == 3100
    assert by_period_and_fuel[("2026-Q1", "electricity_single_rate")] == 2700
    assert by_period_and_fuel[("2026-Q4", "electricity_single_rate")] == 2500
    assert by_period_and_fuel[("2024-Q1", "gas")] == 12000
    assert by_period_and_fuel[("2026-Q4", "gas")] == 9500


def test_ofgem_regional_levels_exclude_vat_and_gb_averages_carry_both():
    facts = _ofgem_cap_levels()
    regional = [fact for fact in facts if fact.geography.level == "statistical_scope"]
    gb_average = [fact for fact in facts if fact.geography.id == "K03000001"]

    assert {fact.filters["vat_treatment"] for fact in regional} == {"excluding_vat"}
    assert {fact.filters["vat_treatment"] for fact in gb_average} == {
        "excluding_vat",
        "including_vat",
    }


def test_ofgem_regional_levels_match_the_published_cap_tables():
    facts = {fact.source_record_id: fact for fact in _ofgem_cap_levels()}
    # "Energy price cap level: 1 January to 31 March 2024", charge restriction
    # period 11b, and "Energy price cap levels: 1 October to 31 December 2026",
    # period 17a.
    published = {
        "ofgem.price_cap.direct_debit.electricity_single_rate."
        "nil_consumption.2024_q1.london.cap_level": 133.84,
        "ofgem.price_cap.standard_credit.electricity_multi_register."
        "benchmark_consumption.2024_q1.south_wales.cap_level": 1377.41,
        "ofgem.price_cap.prepayment.gas.nil_consumption."
        "2024_q1.east_midlands.cap_level": 140.51,
        "ofgem.price_cap.direct_debit.gas.benchmark_consumption."
        "2026_q4.north_west.cap_level": 815.02,
    }

    for source_record_id, expected in published.items():
        assert facts[source_record_id].value == pytest.approx(expected, abs=0.01)


def test_ons_consumer_trends_sheets_share_one_coicop_constraint_key():
    facts = load_source_package("ons-consumer-trends-current-price-2026").build_facts(
        2026
    )
    by_sheet = {"04cn": set(), "07cn": set()}
    for fact in facts:
        record_set_id = fact.layout.record_set_id
        sheet = "04cn" if ".04cn." in record_set_id else "07cn"
        by_sheet[sheet].add(fact.filters.get("coicop"))

    assert by_sheet["04cn"] == {"04.5"}
    assert by_sheet["07cn"] == {"07.2.2", "07.3.2"}
    assert all(fact.filters.get("coicop") for fact in facts)
    assert {fact.layout.groupby_value_id for fact in facts} == {
        "coicop_04_5",
        "coicop_07_2_2",
        "coicop_07_3_2",
    }
    assert {fact.layout.table_record_kind for fact in facts} == {"detail"}


def test_northern_ireland_support_is_recorded_as_all_public_transport():
    facts = _facts("nithc-annual-report-accounts-2024-25")

    assert {fact.filters["mode_scope"] for fact in facts} == {"all_public_transport"}
    assert {fact.geography.id for fact in facts} == {"N92000002"}
    assert {fact.period.value for fact in facts} == {2023, 2024}
    assert {fact.filters["funding_component"] for fact in facts} == {
        "Capital grants",
        "Public Service Obligation compensation",
        "Concessionary fare compensation for a range of groups",
        "Other revenue funding",
    }


def test_wales_bus_finance_covers_every_unitary_authority_and_the_wales_total():
    facts = _facts("welshgov-transport-revenue-outturn-2024-25")
    fy2024 = [fact for fact in facts if fact.period.value == 2024]

    assert {fact.period.value for fact in facts} == {2022, 2023, 2024}
    assert {fact.filters["service_line"] for fact in facts} == {
        "Concessionary fares",
        "Support to operators",
        "Total public transport",
    }
    assert len({fact.geography.id for fact in fy2024}) == 23
    assert "W92000004" in {fact.geography.id for fact in fy2024}
    assert (
        len({fact.geography.id for fact in fy2024 if fact.geography.id.startswith("W06")})
        == 22
    )


def test_scotland_bus_statistics_reach_the_england_base_data_year():
    facts = _facts("scotgov-bus-coach-statistics-2024-25")
    fiscal = [fact for fact in facts if fact.period.type == "fiscal_year"]

    assert {fact.period.value for fact in fiscal} == {2024}
    assert {fact.geography.id for fact in facts} == {"S92000003"}
    assert {
        fact.layout.groupby_value_id
        for fact in facts
        if fact.layout.record_set_id.endswith("table_2_9.fy2024")
    } == {
        "local_authority_bus_support",
        "concessionary_fares",
        "network_support_grant",
        "all_government_support",
    }


def test_nts_local_bus_use_frequency_carries_every_publisher_band():
    modal = _facts("dft-nts0313-mode-use-frequency-2025")
    older = _facts("dft-nts0621-local-bus-use-frequency-2025")
    local_bus_2025 = [
        fact
        for fact in modal
        if fact.period.value == 2025
        and str(fact.filters.get("transport_mode")).startswith("Local bus")
        and fact.filters.get("use_frequency_band")
    ]

    assert {fact.filters["use_frequency_band"] for fact in local_bus_2025} == (
        NTS_FREQUENCY_BANDS
    )
    assert {fact.geography.id for fact in modal} == {"E92000001"}
    assert {fact.geography.id for fact in older} == {"E92000001"}
    assert {fact.filters["transport_mode"] for fact in older} == {"Local bus"}
    assert {fact.filters["age_band"] for fact in older} == {"60 and over"}
    assert 2025 in {fact.period.value for fact in older}


def test_nts_frequency_shares_are_publisher_percentages_not_household_counts():
    facts = _facts("dft-nts0313-mode-use-frequency-2025")
    shares = [
        fact
        for fact in facts
        if fact.measure.concept == "dft.nts.mode_use_frequency_share"
    ]
    sample_sizes = [
        fact
        for fact in facts
        if fact.measure.concept == "dft.nts.unweighted_sample_size"
    ]

    assert {fact.measure.unit for fact in shares} == {"percent"}
    assert {fact.measure.unit for fact in sample_sizes} == {"count"}
    assert {fact.provenance_class for fact in facts} == {"survey_aggregate"}
    assert {fact.survey_instrument for fact in facts} == {"National Travel Survey"}
