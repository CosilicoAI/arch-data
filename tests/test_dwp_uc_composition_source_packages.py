"""Regression tests for Stat-Xplore UC composition histories (#233, #234, #251).

The three chronicle#233 crosses (family type x child entitlement, number of
children x child entitlement, family type x payment indicator) and the two
chronicle#234 histories (monthly award band x family type, Scotland age of
youngest child) share one shape: nine monthly record sets, April to December
2025, benefit-unit entity, publisher labels as categorical identity.
The chronicle#251 payment-status x child-entitlement crosses cover April 2023
through May 2026, including a separate publisher Total over child entitlement.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from importlib.resources import files

import pytest
import yaml

from chronicle.consumer_contract import (
    consumer_fact_rows,
    validate_consumer_fact_contract,
)
from chronicle.core import validate_facts
from chronicle.source_package import load_source_package, validate_source_package
from chronicle.suite import build_source_suite

MONTHS = [f"2025-{month:02d}" for month in range(4, 13)]
FAMILY_CROSS_251 = (
    "dwp-uc-households-family-type-payment-indicator-child-entitlement-"
    "april-2023-may-2026"
)
CHILDREN_CROSS_251 = (
    "dwp-uc-households-children-payment-indicator-child-entitlement-april-2023-may-2026"
)


def _cross_archive_251(alias):
    directory = files("db").joinpath(
        "data", "dwp", alias.removeprefix("dwp-").replace("-", "_")
    )
    manifest = yaml.safe_load(directory.joinpath("manifest.yaml").read_text())
    entry = manifest["files"][2025]
    raw = directory.joinpath(entry["filename"]).read_bytes()
    return entry, raw, json.loads(raw)


def _month_period_251(item):
    code = item["uris"][0].rsplit(":", 1)[1]
    return f"{code[:4]}-{code[4:]}"


MONTHS_251 = [
    _month_period_251(item)
    for item in _cross_archive_251(FAMILY_CROSS_251)[2]["fields"][0]["items"]
]


@pytest.fixture(scope="module")
def composition_cross_facts_251():
    return {
        alias: load_source_package(alias).build_facts(2025)
        for alias in (FAMILY_CROSS_251, CHILDREN_CROSS_251)
    }


def _fact(facts, *, period, **filters):
    matches = [
        fact
        for fact in facts
        if fact.period.value == period
        and all(fact.filters.get(key) == value for key, value in filters.items())
    ]
    assert len(matches) == 1, (period, filters, len(matches))
    return matches[0]


@pytest.mark.parametrize(
    ("alias", "rows_per_month", "geography_id", "groupby_dimension"),
    [
        (
            "dwp-uc-households-family-type-child-entitlement-april-december-2025",
            10,
            "K03000001",
            "dwp.uc_family_type",
        ),
        (
            "dwp-uc-households-children-child-entitlement-april-december-2025",
            16,
            "K03000001",
            "dwp.uc_number_of_children",
        ),
        (
            "dwp-uc-households-family-type-payment-indicator-april-december-2025",
            10,
            "K03000001",
            "dwp.uc_family_type",
        ),
        (
            "dwp-uc-payment-distribution-april-december-2025",
            135,
            "K03000001",
            "dwp.uc_monthly_award_band",
        ),
        (
            "dwp-uc-scotland-youngest-child-april-december-2025",
            23,
            "S92000003",
            "dwp.uc_youngest_child_age",
        ),
    ],
)
def test_uc_monthly_packages_cover_april_to_december_2025(
    alias,
    rows_per_month,
    geography_id,
    groupby_dimension,
):
    package = load_source_package(alias)
    report = validate_source_package(package.package_path, year=2025)
    facts = package.build_facts(2025)

    assert report.valid
    assert report.counts["record_set_count"] == 9
    assert report.counts["row_count"] == rows_per_month * 9
    assert report.counts["source_record_count"] == rows_per_month * 9
    assert len(facts) == rows_per_month * 9
    assert {fact.period.value for fact in facts} == set(MONTHS)
    assert all(fact.period.type == "month" for fact in facts)
    assert all(fact.measure.concept == "dwp.uc_benefit_units" for fact in facts)
    # The #188 histories' source concept is kept so consumers bound to
    # dwp.uc_households keep resolving after the May snapshots retire.
    assert all(fact.measure.source_concept == "dwp.uc_households" for fact in facts)
    assert all(fact.entity.name == "benefit_unit" for fact in facts)
    assert all(fact.geography.id == geography_id for fact in facts)
    assert all(fact.assertion == "observation" for fact in facts)
    assert all(fact.provenance_class == "administrative" for fact in facts)
    assert all(fact.layout.groupby_dimension == groupby_dimension for fact in facts)
    assert all(fact.source_row_keys for fact in facts)
    assert validate_facts(facts).valid
    assert validate_consumer_fact_contract(facts).valid
    assert len(consumer_fact_rows(facts)) == len(facts)


def test_family_type_child_entitlement_cross_keeps_both_publisher_dimensions():
    facts = load_source_package(
        "dwp-uc-households-family-type-child-entitlement-april-december-2025"
    ).build_facts(2025)

    with_element = _fact(
        facts,
        period="2025-04",
        family_type="Single, with children",
        child_entitlement="Yes",
    )
    without_element = _fact(
        facts,
        period="2025-04",
        family_type="Single, with children",
        child_entitlement="No",
    )
    assert with_element.value == 2_093_294
    assert without_element.value == 114_267
    assert {constraint.variable for constraint in with_element.constraints} == {
        "family_type",
        "child_entitlement",
    }
    assert (
        _fact(
            facts,
            period="2025-12",
            family_type="Couple, with children",
            child_entitlement="Yes",
        ).value
        == 851_844
    )
    # Childless family types never carry a child element.
    assert all(
        fact.value == 0
        for fact in facts
        if fact.filters["child_entitlement"] == "Yes"
        and fact.filters["family_type"]
        in {"Single, no children", "Couple, no children"}
    )


def test_number_of_children_child_entitlement_cross_preserves_publisher_categories():
    facts = load_source_package(
        "dwp-uc-households-children-child-entitlement-april-december-2025"
    ).build_facts(2025)

    # Digit-only publisher labels land as integers, as in the number-of-children
    # package this cross extends.
    assert {fact.filters["number_of_children"] for fact in facts} == {
        0,
        1,
        2,
        3,
        4,
        "5 or more",
        "Unknown or missing",
        "Not available prior to April 2019",
    }
    assert (
        _fact(
            facts, period="2025-04", number_of_children=2, child_entitlement="Yes"
        ).value
        == 1_084_955
    )
    assert (
        _fact(
            facts, period="2025-04", number_of_children=2, child_entitlement="No"
        ).value
        == 18_473
    )
    assert (
        _fact(
            facts, period="2025-12", number_of_children=1, child_entitlement="Yes"
        ).value
        == 1_122_012
    )


def test_payment_indicator_no_matches_the_no_payment_award_band_cell_for_cell():
    """DWP's Payment Indicator 'No' is the nil-award household; the award-band
    cube publishes the same households under its 'No payment' band."""
    indicator = load_source_package(
        "dwp-uc-households-family-type-payment-indicator-april-december-2025"
    ).build_facts(2025)
    bands = load_source_package(
        "dwp-uc-payment-distribution-april-december-2025"
    ).build_facts(2025)

    assert (
        _fact(
            indicator,
            period="2025-04",
            family_type="Single, with children",
            payment_indicator="No",
        ).value
        == 74_530
    )
    assert (
        _fact(
            indicator,
            period="2025-05",
            family_type="Single, with children",
            payment_indicator="No",
        ).value
        == 67_972
    )
    for period in MONTHS:
        for family_type in (
            "Single, no children",
            "Single, with children",
            "Couple, no children",
            "Couple, with children",
            "Unknown or missing family type",
        ):
            nil_award = _fact(
                indicator,
                period=period,
                family_type=family_type,
                payment_indicator="No",
            )
            no_payment_band = _fact(
                bands,
                period=period,
                family_type=family_type,
                monthly_award_amount_bands="No payment",
            )
            assert nil_award.value == no_payment_band.value, (period, family_type)


def test_payment_distribution_history_omits_the_pre_september_2022_band():
    """DWP's '£1500.01 or over' band applies to months up to August 2022 only
    (Stat-Xplore metadata); it is zero in every April to December 2025 cell and
    is not ported, so nothing can bind it by mistake."""
    facts = load_source_package(
        "dwp-uc-payment-distribution-april-december-2025"
    ).build_facts(2025)
    bands = {fact.filters["monthly_award_amount_bands"] for fact in facts}

    assert len(facts) == 27 * 5 * 9
    assert len(bands) == 27
    assert "£1500.01 or over" not in bands
    assert {
        "No payment",
        "£1400.01 to £1500.00",
        "£1500.01 to £1600.00",
        "£2500.01 or over",
    } <= bands
    assert all(fact.layout.table_record_kind == "detail" for fact in facts)
    assert (
        _fact(
            facts,
            period="2025-04",
            family_type="Single, no children",
            monthly_award_amount_bands="No payment",
        ).value
        == 345_734
    )
    assert (
        _fact(
            facts,
            period="2025-04",
            family_type="Single, with children",
            monthly_award_amount_bands="£1500.01 to £1600.00",
        ).value
        > 0
    )
    assert (
        _fact(
            facts,
            period="2025-12",
            family_type="Single, with children",
            monthly_award_amount_bands="£2500.01 or over",
        ).value
        == 93_095
    )


def test_scotland_youngest_child_history_carries_single_years_of_age():
    facts = load_source_package(
        "dwp-uc-scotland-youngest-child-april-december-2025"
    ).build_facts(2025)
    ages = {
        fact.filters["age_of_youngest_child_bands_and_single_year"] for fact in facts
    }

    assert ages == set(range(20)) | {
        "No children",
        "Unknown or missing",
        "Not available prior to April 2019",
    }
    assert all(fact.geography.name == "Scotland" for fact in facts)
    assert (
        _fact(
            facts, period="2025-04", age_of_youngest_child_bands_and_single_year=0
        ).value
        == 14_451
    )
    assert (
        _fact(
            facts, period="2025-12", age_of_youngest_child_bands_and_single_year=0
        ).value
        == 14_170
    )
    assert (
        _fact(
            facts,
            period="2025-12",
            age_of_youngest_child_bands_and_single_year="No children",
        ).value
        == 378_281
    )


def _assert_cross_fact_contract_251(facts, groupby_dimension):
    assert len(MONTHS_251) == 38
    assert MONTHS_251[0] == "2023-04"
    assert MONTHS_251[-1] == "2026-05"
    month_indices = [int(period[:4]) * 12 + int(period[5:]) for period in MONTHS_251]
    assert month_indices == list(range(month_indices[0], month_indices[-1] + 1))
    assert {fact.period.value for fact in facts} == set(MONTHS_251)
    assert all(fact.period.type == "month" for fact in facts)
    assert all(fact.measure.concept == "dwp.uc_benefit_units" for fact in facts)
    assert all(fact.measure.source_concept == "dwp.uc_households" for fact in facts)
    assert all(fact.entity.name == "benefit_unit" for fact in facts)
    assert all(
        fact.entity.role == "universal_credit_unit_of_assessment" for fact in facts
    )
    assert all(fact.geography.id == "K03000001" for fact in facts)
    assert all(fact.geography.name == "Great Britain" for fact in facts)
    assert all(fact.assertion == "observation" for fact in facts)
    assert all(fact.provenance_class == "administrative" for fact in facts)
    assert all(fact.layout.groupby_dimension == groupby_dimension for fact in facts)
    assert all(fact.source_row_keys and fact.source_cell_keys for fact in facts)
    assert validate_facts(facts).valid
    assert validate_consumer_fact_contract(facts).valid
    assert len(consumer_fact_rows(facts)) == len(facts)


def test_family_type_payment_indicator_child_entitlement_package_covers_april_2023_onward(
    composition_cross_facts_251,
):
    package = load_source_package(FAMILY_CROSS_251)
    report = validate_source_package(package.package_path, year=2025)
    facts = composition_cross_facts_251[FAMILY_CROSS_251]

    assert report.valid
    assert report.counts["record_set_count"] == len(MONTHS_251)
    assert (
        report.counts["row_count"]
        == report.counts["source_record_count"]
        == len(facts)
        == 20 * len(MONTHS_251)
        == 760
    )
    assert all(fact.layout.measure_id == "benefit_units" for fact in facts)
    assert all(fact.layout.table_record_kind == "detail" for fact in facts)
    _assert_cross_fact_contract_251(facts, "dwp.uc_family_type")
    assert {
        row["observed_measure"]["source_measure_id"]
        for row in consumer_fact_rows(facts)
    } == {"benefit_units"}


def test_children_payment_indicator_child_entitlement_package_keeps_detail_and_total_record_sets(
    composition_cross_facts_251,
):
    package = load_source_package(CHILDREN_CROSS_251)
    report = validate_source_package(package.package_path, year=2025)
    facts = composition_cross_facts_251[CHILDREN_CROSS_251]
    detail = [fact for fact in facts if fact.layout.measure_id == "benefit_units"]
    totals = [fact for fact in facts if fact.layout.measure_id == "total_benefit_units"]

    assert report.valid
    assert report.counts["record_set_count"] == 2 * len(MONTHS_251) == 76
    assert (
        report.counts["row_count"]
        == report.counts["source_record_count"]
        == len(facts)
        == 48 * len(MONTHS_251)
        == 1_824
    )
    assert len(detail) == 32 * len(MONTHS_251) == 1_216
    assert len(totals) == 16 * len(MONTHS_251) == 608
    assert all(fact.layout.table_record_kind == "detail" for fact in detail)
    assert all(fact.layout.table_record_kind == "total" for fact in totals)
    assert all(fact.filters["child_entitlement"] == "all" for fact in totals)
    assert all(
        {constraint.variable for constraint in fact.constraints}
        == {"number_of_children", "payment_indicator", "child_entitlement"}
        for fact in detail
    )
    # Publisher Total is guarded by its label; an `== all` constraint would
    # incorrectly compare the consumer filter with the publisher label Total.
    assert all(
        {constraint.variable for constraint in fact.constraints}
        == {"number_of_children", "payment_indicator"}
        for fact in totals
    )
    _assert_cross_fact_contract_251(facts, "dwp.uc_number_of_children")
    assert {
        row["observed_measure"]["source_measure_id"]
        for row in consumer_fact_rows(facts)
    } == {"benefit_units", "total_benefit_units"}


def test_family_type_payment_indicator_child_entitlement_cross_keeps_three_publisher_dimensions(
    composition_cross_facts_251,
):
    facts = composition_cross_facts_251[FAMILY_CROSS_251]
    assert {fact.filters["family_type"] for fact in facts} == {
        "Single, no children",
        "Single, with children",
        "Couple, no children",
        "Couple, with children",
        "Unknown or missing family type",
    }
    assert {fact.filters["payment_indicator"] for fact in facts} == {"No", "Yes"}
    assert {fact.filters["child_entitlement"] for fact in facts} == {"No", "Yes"}
    for fact in facts:
        assert set(fact.filters) == {
            "family_type",
            "payment_indicator",
            "child_entitlement",
        }
        assert {constraint.variable for constraint in fact.constraints} == set(
            fact.filters
        )
        assert all(
            constraint.operator == "=="
            and constraint.value == fact.filters[constraint.variable]
            for constraint in fact.constraints
        )
        if (
            fact.filters["family_type"]
            in {"Single, no children", "Couple, no children"}
            and fact.filters["child_entitlement"] == "Yes"
        ) or (
            fact.filters["family_type"] == "Unknown or missing family type"
            and fact.filters["child_entitlement"] == "No"
        ):
            assert fact.value == 0

    # Publisher cells observed on the 18 August 2026 release. Matrix order is
    # payment No/Yes x child entitlement No/Yes.
    samples = {
        "2023-04": [[20_042, 47_260], [56_744, 1_609_816]],
        "2025-04": [[25_860, 48_669], [88_403, 2_044_622]],
        "2026-05": [[17_219, 37_804], [123_529, 2_044_209]],
    }
    for period, expected in samples.items():
        assert [
            [
                _fact(
                    facts,
                    period=period,
                    family_type="Single, with children",
                    payment_indicator=payment,
                    child_entitlement=entitlement,
                ).value
                for entitlement in ("No", "Yes")
            ]
            for payment in ("No", "Yes")
        ] == expected


def test_children_payment_indicator_child_entitlement_cross_preserves_publisher_categories(
    composition_cross_facts_251,
):
    facts = composition_cross_facts_251[CHILDREN_CROSS_251]
    categories = {
        0,
        1,
        2,
        3,
        4,
        "5 or more",
        "Unknown or missing",
        "Not available prior to April 2019",
    }
    for kind in ("detail", "total"):
        subset = [fact for fact in facts if fact.layout.table_record_kind == kind]
        assert {fact.filters["number_of_children"] for fact in subset} == categories
        assert {fact.filters["payment_indicator"] for fact in subset} == {"No", "Yes"}
        assert all(
            type(fact.filters["number_of_children"]) is int
            for fact in subset
            if fact.filters["number_of_children"] in range(5)
        )
    assert {
        fact.filters["child_entitlement"]
        for fact in facts
        if fact.layout.table_record_kind == "detail"
    } == {"No", "Yes"}
    for fact in facts:
        children = fact.filters["number_of_children"]
        entitlement = fact.filters["child_entitlement"]
        if (
            children == "Not available prior to April 2019"
            or (children == "Unknown or missing" and entitlement == "No")
            or (
                children == 0
                and entitlement == "Yes"
                and fact.filters["payment_indicator"] == "No"
            )
        ):
            assert fact.value == 0
        elif (
            children == 0
            and entitlement == "Yes"
            and fact.filters["payment_indicator"] == "Yes"
        ):
            # These three publisher cells are 5; the other 35 months are zero.
            assert fact.value == (
                5 if fact.period.value in {"2024-03", "2024-04", "2024-05"} else 0
            )

    # Publisher cells observed on the 18 August 2026 release. April 2025 was
    # read directly from the archived response to complete the three-month pin.
    # Matrix order is payment No/Yes x child entitlement No/Yes/Total.
    samples = {
        "2023-04": [[4_991, 41_512, 46_499], [9_346, 797_141, 806_488]],
        "2025-04": [[6_701, 44_716, 51_420], [11_773, 1_040_238, 1_052_011]],
        "2026-05": [[3_241, 35_271, 38_516], [19_894, 1_026_680, 1_046_574]],
    }
    for period, expected in samples.items():
        assert [
            [
                _fact(
                    facts,
                    period=period,
                    number_of_children=2,
                    payment_indicator=payment,
                    child_entitlement=entitlement,
                ).value
                for entitlement in ("No", "Yes", "all")
            ]
            for payment in ("No", "Yes")
        ] == expected


@pytest.mark.parametrize(
    ("alias", "category_field", "category_valueset", "category_codes", "shape"),
    [
        (
            FAMILY_CROSS_251,
            "hnfamily_type",
            "C_UC_FAMILY_TYPE",
            [1, 2, 3, 4, 99],
            [38, 5, 2, 2],
        ),
        (
            CHILDREN_CROSS_251,
            "NUMBER_OF_CHILDREN",
            "C_UC_NUMBER_OF_CHILDREN",
            [0, 1, 2, 3, 4, 5, 9998, 9999],
            [38, 8, 2, 3],
        ),
    ],
)
def test_uc_composition_cross_responses_echo_the_requested_query(
    alias, category_field, category_valueset, category_codes, shape
):
    manifest_entry, raw, response = _cross_archive_251(alias)
    assert hashlib.sha256(raw).hexdigest() == manifest_entry["sha256"]
    assert len(raw) == manifest_entry["size_bytes"]
    with_total = alias == CHILDREN_CROSS_251
    measure_uri = "str:count:UC_Households:V_F_UC_HOUSEHOLDS"
    field_prefix = "str:field:UC_Households:"
    value_prefix = "str:value:UC_Households:"
    requested_axes = [
        (
            "F_UC_HH_DATE:DATE_NAME",
            "C_UC_HH_DATE",
            [period.replace("-", "") for period in MONTHS_251],
        ),
        (f"V_F_UC_HOUSEHOLDS:{category_field}", category_valueset, category_codes),
        ("V_F_UC_HOUSEHOLDS:HCPAYMENT", "C_UC_PAYMENT", [0, 1]),
        ("V_F_UC_HOUSEHOLDS:HCCHILD_ENTITLEMENT", "C_UC_CHILD_ENTITLEMENT", [0, 1]),
    ]
    expected_recodes = {
        f"{field_prefix}{field}": {
            "map": [[f"{value_prefix}{field}:{valueset}:{code}"] for code in codes],
            "total": with_total and field.endswith(":HCCHILD_ENTITLEMENT"),
        }
        for field, valueset, codes in requested_axes
    }
    assert response["query"] == {
        "database": "str:database:UC_Households",
        "measures": [measure_uri],
        "recodes": expected_recodes,
        "dimensions": [[f"{field_prefix}{field}"] for field, _, _ in requested_axes],
    }
    assert response["database"]["uri"] == response["query"]["database"]
    assert [measure["uri"] for measure in response["measures"]] == [measure_uri]
    assert [field["uri"] for field in response["fields"]] == list(expected_recodes)
    assert [len(field["items"]) for field in response["fields"]] == shape
    for field in response["fields"]:
        recode = expected_recodes[field["uri"]]
        assert [item["uris"] for item in field["items"] if "uris" in item] == recode[
            "map"
        ]
        total_items = [item for item in field["items"] if "uris" not in item]
        assert total_items == (
            [{"type": "Total", "labels": ["Total"]}] if recode["total"] else []
        )

    month_items = response["fields"][0]["items"]
    periods = [_month_period_251(item) for item in month_items]
    assert periods == MONTHS_251
    assert len(periods) == 38
    assert periods[0] == "2023-04"
    assert periods[-1] == "2026-05"
    month_indices = [int(period[:4]) * 12 + int(period[5:]) for period in periods]
    assert month_indices == list(range(month_indices[0], month_indices[-1] + 1))
    assert [
        datetime.strptime(item["labels"][0], "%B %Y").strftime("%Y-%m")
        for item in month_items
    ] == periods
    annotations = {
        _month_period_251(item): item.get("annotationKeys", []) for item in month_items
    }
    # Both responses in the 18 August 2026 release mark June 2024-February 2026
    # revised, March-May 2026 provisional, and April 2023-May 2024 unmarked.
    assert annotations == {
        period: (
            ["r"]
            if "2024-06" <= period <= "2026-02"
            else ["p"]
            if "2026-03" <= period <= "2026-05"
            else []
        )
        for period in periods
    }
    assert sum(keys == ["r"] for keys in annotations.values()) == 21
    assert sum(keys == ["p"] for keys in annotations.values()) == 3
    assert sum(keys == [] for keys in annotations.values()) == 14
    assert set(response["annotationMap"]) == {"I", "II", "III", "IV", "p", "r", "z"} | (
        {"XII"} if with_total else set()
    )

    def assert_cube_shape(values, axis=0):
        if axis == len(shape):
            assert type(values) is int
            return
        assert isinstance(values, list)
        assert len(values) == shape[axis]
        for value in values:
            assert_cube_shape(value, axis + 1)

    assert set(response["cubes"]) == {measure_uri}
    assert_cube_shape(response["cubes"][measure_uri]["values"])


def test_children_payment_indicator_child_entitlement_total_is_a_publisher_margin(
    composition_cross_facts_251,
):
    facts = composition_cross_facts_251[CHILDREN_CROSS_251]
    totals = [fact for fact in facts if fact.layout.measure_id == "total_benefit_units"]
    gaps = []
    for total in totals:
        detail_sum = sum(
            _fact(
                facts,
                period=total.period.value,
                number_of_children=total.filters["number_of_children"],
                payment_indicator=total.filters["payment_indicator"],
                child_entitlement=entitlement,
            ).value
            for entitlement in ("No", "Yes")
        )
        gaps.append(abs(total.value - detail_sum))
    # On the 18 August 2026 release: observed maximum 8, with 333/608 nonzero
    # gaps from disclosure control. This comparison never emits a summed fact.
    assert len(gaps) == 608
    assert max(gaps) == 8
    assert sum(gap != 0 for gap in gaps) == 333
    assert all(gap <= 10 for gap in gaps)


def test_composition_crosses_reconcile_with_the_pairwise_cubes_within_disclosure_tolerance(
    composition_cross_facts_251,
):
    # Both extracts come from the 18 August 2026 release (#239 fetched 3 September
    # and #251 fetched 8 September). Observed maxima are 6, 9, and 7 respectively;
    # the common tolerance is 15. Reconciliation is test-only, never a fact.
    comparisons = [
        (
            FAMILY_CROSS_251,
            "dwp-uc-households-family-type-payment-indicator-april-december-2025",
            "child_entitlement",
            90,
            6,
            31,
        ),
        (
            FAMILY_CROSS_251,
            "dwp-uc-households-family-type-child-entitlement-april-december-2025",
            "payment_indicator",
            90,
            9,
            58,
        ),
        (
            CHILDREN_CROSS_251,
            "dwp-uc-households-children-child-entitlement-april-december-2025",
            "payment_indicator",
            144,
            7,
            96,
        ),
    ]
    for (
        cross_alias,
        pairwise_alias,
        summed_axis,
        count,
        maximum,
        nonzero,
    ) in comparisons:
        cross = composition_cross_facts_251[cross_alias]
        pairwise = load_source_package(pairwise_alias).build_facts(2025)
        assert {fact.period.value for fact in pairwise} == set(MONTHS)
        gaps = []
        for published in pairwise:
            detail_sum = sum(
                _fact(
                    cross,
                    period=published.period.value,
                    **published.filters,
                    **{summed_axis: value},
                ).value
                for value in ("No", "Yes")
            )
            gaps.append(abs(published.value - detail_sum))
        assert len(gaps) == count
        assert max(gaps) == maximum
        assert sum(gap != 0 for gap in gaps) == nonzero
        assert all(gap <= 15 for gap in gaps)


@pytest.mark.parametrize("alias", [FAMILY_CROSS_251, CHILDREN_CROSS_251])
def test_uc_composition_cross_packages_pass_agent_acceptance(alias, tmp_path):
    output_dir = tmp_path / alias
    report = build_source_suite(alias, output_dir, year=2025)
    acceptance = json.loads(
        (output_dir / "reports" / "agent_acceptance.json").read_text()
    )

    assert report.valid
    assert acceptance["valid"]
    assert acceptance["counts"]["row_semantic_error_count"] == 0
