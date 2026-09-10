"""Regression coverage for the UK transport and energy facts in issue 254."""

from __future__ import annotations

import pytest

from chronicle.consumer_contract import validate_consumer_fact_contract
from chronicle.core import validate_facts
from chronicle.source_package import (
    SOURCE_PACKAGE_ALIASES,
    load_source_package,
    validate_source_package,
)
from chronicle.sources.cells import validate_source_cells


ISSUE_254_ARTIFACT_YEARS = {
    "desnz-monthly-annual-road-fuel-prices-august-2026": 2026,
    "desnz-need-england-wales-2023": 2023,
    "desnz-need-scotland-2023": 2023,
    "desnz-weekly-road-fuel-prices-september-2026": 2026,
    "dfi-ni-public-transport-statistics-2024-25": 2025,
    "dft-veh1103-cars-fuel-type-2025": 2025,
    "hmrc-hydrocarbon-oils-quantities-june-2026": 2026,
    "obr-fuel-duty-receipts-by-vehicle-april-2024": 2024,
    "ofgem-energy-price-cap-q1-2024": 2024,
    "ons-consumer-trends-current-price-2026": 2026,
    "orr-government-support-7270-2024-25": 2025,
    "orr-government-support-7271-2024-25": 2025,
    "orr-rail-fares-7180-2026": 2026,
    "orr-rail-finance-7223-2024-25": 2025,
    "scotgov-bus-coach-statistics-2023-24": 2024,
    "welshgov-bus-statistics-2024-25": 2025,
}

EXPECTED_FACT_COUNTS = {
    "desnz-monthly-annual-road-fuel-prices-august-2026": 94,
    "desnz-need-england-wales-2023": 144,
    "desnz-need-scotland-2023": 84,
    "desnz-weekly-road-fuel-prices-september-2026": 579,
    "dfi-ni-public-transport-statistics-2024-25": 24,
    "dft-veh1103-cars-fuel-type-2025": 36,
    "hmrc-hydrocarbon-oils-quantities-june-2026": 168,
    "obr-fuel-duty-receipts-by-vehicle-april-2024": 49,
    "ofgem-energy-price-cap-q1-2024": 4,
    "ons-consumer-trends-current-price-2026": 93,
    "orr-government-support-7270-2024-25": 10,
    "orr-government-support-7271-2024-25": 60,
    "orr-rail-fares-7180-2026": 28,
    "orr-rail-finance-7223-2024-25": 1,
    "scotgov-bus-coach-statistics-2023-24": 8,
    "welshgov-bus-statistics-2024-25": 3,
}

REPRESENTATIVE_PUBLISHER_FACTS = {
    "desnz-monthly-annual-road-fuel-prices-august-2026": (
        "desnz.monthly_road_fuel_prices.2023_01.united_kingdom.ulsp_price",
        148.45071213621574,
    ),
    "desnz-need-england-wales-2023": (
        "desnz.need_2023.table_5.property_type.detached.detached.mean_consumption",
        15518.4544493478,
    ),
    "desnz-need-scotland-2023": (
        "desnz.need_2023.table_3.property_type.detached.detached.mean_consumption",
        16974.4827866605,
    ),
    "desnz-weekly-road-fuel-prices-september-2026": (
        "desnz.weekly_road_fuel_prices.2023_01_02.2023_01_02.ulsp_price",
        150.89909733333337,
    ),
    "dfi-ni-public-transport-statistics-2024-25": (
        "dfi_ni.public_transport.figure_3.fy2019.ulsterbus.passenger_journeys",
        37_890_000,
    ),
    "dft-veh1103-cars-fuel-type-2025": (
        "dft.veh1103a.cars.cy2023.cars.cars_petrol",
        18_740_091,
    ),
    "hmrc-hydrocarbon-oils-quantities-june-2026": (
        "hmrc.hydrocarbon_oils.table_2a.fy2020.united_kingdom.petrol_quantity",
        11670759532.8041,
    ),
    "obr-fuel-duty-receipts-by-vehicle-april-2024": (
        "obr.fuel_duty_receipts_by_vehicle.fy2022.total.receipts",
        25_100_000_000,
    ),
    "ofgem-energy-price-cap-q1-2024": (
        "ofgem.price_cap.q1_2024.electricity_unit_rate.electricity_unit_rate.electricity_unit_rate",
        28.62,
    ),
    "ons-consumer-trends-current-price-2026": (
        "ons.consumer_trends.04cn.cy2020.coicop_04_5.expenditure",
        31_805_000_000,
    ),
    "orr-government-support-7270-2024-25": (
        "orr.government_support.table_7270.fy2015.great_britain.total_government_support",
        10852252009.315622,
    ),
    "orr-government-support-7271-2024-25": (
        "orr.government_support.table_7271.fy2015.all_sources.support",
        4553029127.862197,
    ),
    "orr-rail-fares-7180-2026": (
        "orr.rail_fares.table_7180.cy2020.regulated_standard.fare_index",
        208.716987746566,
    ),
    "orr-rail-finance-7223-2024-25": (
        "orr.rail_finance.table_7223.fy2024.fare_income.fare_income",
        11251872733.02,
    ),
    "scotgov-bus-coach-statistics-2023-24": (
        "scotgov.bus.table_2_2a.fy2023.all_passenger_journeys.passenger_journeys",
        334329795.89247,
    ),
    "welshgov-bus-statistics-2024-25": (
        "welshgov.bus_statistics.passenger_journeys.fy2024.wales.passenger_journeys",
        71_700_000,
    ),
}


def test_issue_254_packages_are_registered():
    assert set(ISSUE_254_ARTIFACT_YEARS) <= set(SOURCE_PACKAGE_ALIASES)


@pytest.mark.parametrize(
    ("alias", "year"),
    sorted(ISSUE_254_ARTIFACT_YEARS.items()),
)
def test_issue_254_packages_build_valid_consumer_facts(alias, year):
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


def test_issue_254_packages_do_not_publish_computed_share_facts():
    aliases = (
        "dft-veh1103-cars-fuel-type-2025",
        "scotgov-bus-coach-statistics-2023-24",
        "welshgov-bus-statistics-2024-25",
    )
    concepts = {
        fact.measure.concept
        for alias in aliases
        for fact in load_source_package(alias).build_facts(2025)
    }

    assert all("share" not in concept for concept in concepts)
    assert all(
        fact.aggregation.method != "share"
        for alias in aliases
        for fact in load_source_package(alias).build_facts(2025)
    )


def test_issue_254_facts_declare_exact_period_coverage():
    for alias, year in ISSUE_254_ARTIFACT_YEARS.items():
        facts = load_source_package(alias).build_facts(year)

        assert all(fact.period_coverage is not None for fact in facts)
        assert all(fact.period_coverage.start_date for fact in facts)
        assert all(fact.period_coverage.end_date for fact in facts)
        assert all(fact.period_coverage.basis for fact in facts)


def test_issue_254_subannual_facts_use_distinct_period_types():
    weekly = load_source_package(
        "desnz-weekly-road-fuel-prices-september-2026"
    ).build_facts(2026)
    ons = load_source_package("ons-consumer-trends-current-price-2026").build_facts(
        2026
    )
    ofgem = load_source_package("ofgem-energy-price-cap-q1-2024").build_facts(2024)

    assert {fact.period.type for fact in weekly} == {"week"}
    assert {fact.period.type for fact in ofgem} == {"quarter"}
    assert {fact.period.type for fact in ons} == {"calendar_year", "quarter"}
    assert {fact.period.value for fact in ofgem} == {"2024-Q1"}


def test_issue_254_weekly_fuel_duty_rate_is_emitted():
    facts = load_source_package(
        "desnz-weekly-road-fuel-prices-september-2026"
    ).build_facts(2026)
    duty_rates = [
        fact
        for fact in facts
        if fact.measure.concept == "desnz.road_fuel.ulsp_duty_rate"
    ]

    assert len(duty_rates) == 193
    assert {fact.value for fact in duty_rates} == {52.95}
    assert {fact.period.type for fact in duty_rates} == {"week"}


def test_issue_254_need_uses_standard_england_wales_geography():
    facts = load_source_package("desnz-need-england-wales-2023").build_facts(2023)

    assert {fact.geography.id for fact in facts} == {"K04000001"}
    assert {fact.geography.level for fact in facts} == {"country"}


@pytest.mark.parametrize(
    ("alias", "expected"),
    sorted(REPRESENTATIVE_PUBLISHER_FACTS.items()),
)
def test_issue_254_packages_preserve_representative_publisher_values(alias, expected):
    source_record_id, expected_value = expected
    facts = {
        fact.source_record_id: fact
        for fact in load_source_package(alias).build_facts(
            ISSUE_254_ARTIFACT_YEARS[alias]
        )
    }

    assert facts[source_record_id].value == pytest.approx(expected_value)
