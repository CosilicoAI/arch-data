"""UK local household-count legs on the shared per-level record-set specs (chronicle#253).

England and Wales (Census 2021 TS041), Northern Ireland (Census 2021 HOUSEHOLD)
and Scotland (Census 2022 UV404 all occupied households) publish occupied
household counts at their own grains. Scotland's council-area cells are filed
twice from one artifact: under the tenure spec with their tenure detail, and
under the local-authority households spec beside the other two legs, so one
consumer selector spans all 361 local authorities as it already does for the
650 constituencies. Nothing is summed or reconciled across legs or grains.
"""

from __future__ import annotations

from chronicle.consumer_contract import (
    consumer_fact_rows,
    validate_consumer_fact_contract,
)
from chronicle.core import build_fact_key
from chronicle.source_package import load_source_package

LOCAL_AUTHORITY_SPEC = "uk.local_geography.households.by_local_authority.v1"
CONSTITUENCY_SPEC = "uk.local_geography.households.by_constituency.v1"
SCOTLAND_TENURE_SPEC = "uk.local_geography.tenure.all_households.v1"


def _spec_facts(alias, year, spec_id):
    facts = load_source_package(alias).build_facts(year)
    return [fact for fact in facts if fact.layout.record_set_spec_id == spec_id]


def _assert_household_count_leg(facts):
    assert facts
    assert all(fact.entity.name == "household" for fact in facts)
    assert all(fact.entity.role == "occupied_household" for fact in facts)
    assert all(fact.domain == "household_count" for fact in facts)
    assert all(fact.measure.unit == "count" for fact in facts)
    assert all(fact.provenance_class == "census" for fact in facts)
    assert all(fact.layout.table_record_kind == "total" for fact in facts)
    assert all(fact.source.raw_r2_uri for fact in facts)
    assert all(fact.source.source_sha256 for fact in facts)
    assert all(fact.source_row_keys and fact.source_cell_keys for fact in facts)
    assert validate_consumer_fact_contract(facts).valid
    assert len(consumer_fact_rows(facts)) == len(facts)


def test_local_authority_household_legs_cover_all_361_areas_on_one_spec():
    legs = {
        "ons-census2021-ts041-households-lad": (2021, "ons.census2021_households", 318),
        "nisra-census2021-households-lgd": (2021, "nisra.census2021_households", 11),
        "nrs-census2022-uv404-tenure-council-area": (
            2022,
            "nrs.census2022_households",
            32,
        ),
    }
    areas: dict[str, int] = {}
    for alias, (year, concept, expected_count) in legs.items():
        facts = _spec_facts(alias, year, LOCAL_AUTHORITY_SPEC)
        assert len(facts) == expected_count, alias
        assert all(fact.geography.level == "local_authority" for fact in facts)
        assert all(fact.measure.concept == concept for fact in facts)
        _assert_household_count_leg(facts)
        for fact in facts:
            assert fact.geography.id not in areas, fact.geography.id
            areas[fact.geography.id] = fact.value
    assert len(areas) == 361
    assert sum(code.startswith(("E06", "E07", "E08", "E09")) for code in areas) == 296
    assert sum(code.startswith("W06") for code in areas) == 22
    assert sum(code.startswith("N09") for code in areas) == 11
    assert sum(code.startswith("S12") for code in areas) == 32


def test_scotland_council_household_total_matches_its_tenure_filing_cell_for_cell():
    package_facts = load_source_package(
        "nrs-census2022-uv404-tenure-council-area"
    ).build_facts(2022)
    households = [
        fact
        for fact in package_facts
        if fact.layout.record_set_spec_id == LOCAL_AUTHORITY_SPEC
    ]
    tenure_total = [
        fact
        for fact in package_facts
        if fact.layout.record_set_spec_id == SCOTLAND_TENURE_SPEC
    ]
    assert len(households) == len(tenure_total) == 32
    by_area = {fact.geography.id: fact for fact in households}
    assert set(by_area) == {fact.geography.id for fact in tenure_total}
    for published in tenure_total:
        filed = by_area[published.geography.id]
        assert filed.value == published.value
        assert filed.source_row_keys == published.source_row_keys
        assert (
            filed.filters
            == published.filters
            == {"household_tenure": "All occupied households"}
        )
        assert filed.period.value == published.period.value == 2022
        assert filed.geography.vintage == published.geography.vintage == "ca_2019"
    # The Scotland-wide total the package documents; nothing sums it into a fact.
    assert sum(fact.value for fact in households) == 2_509_275
    assert all(fact.domain == "housing_tenure" for fact in tenure_total)
    assert all(fact.domain == "household_count" for fact in households)
    assert all(fact.layout.groupby_dimension == "geography" for fact in households)
    # Filing the same cells on both specs must not collide on fact identity.
    assert len({build_fact_key(fact) for fact in households + tenure_total}) == 64
    assert len({fact.source_record_id for fact in households + tenure_total}) == 64


def test_constituency_household_legs_cover_all_650_areas_on_one_spec():
    legs = {
        "ons-census2021-ts041-households-pcon24": (2021, 575),
        "nrs-census2022-households-ukpc24": (2022, 57),
        "nisra-census2021-households-pcon24": (2021, 18),
    }
    areas: set[str] = set()
    for alias, (year, expected_count) in legs.items():
        facts = _spec_facts(alias, year, CONSTITUENCY_SPEC)
        assert len(facts) == expected_count, alias
        assert all(fact.geography.level == "constituency" for fact in facts)
        _assert_household_count_leg(facts)
        areas.update(fact.geography.id for fact in facts)
    assert len(areas) == 650
