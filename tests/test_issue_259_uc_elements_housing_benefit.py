"""Regression tests for the chronicle#259 packages.

Five Stat-Xplore UC_Households crosses of a UC element with the payment indicator
(January 2023 to May 2026), the People on UC employment indicator, the Housing Benefit
caseload by client type and tenure (January 2023 to February 2026) and by accommodation
type (from September 2025), and the UC childcare element statistics to May 2026, which
supersede the August 2025 vintage. Observed values are from DWP's 18 August 2026 releases.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import pytest
import yaml

from chronicle.consumer_contract import (
    consumer_fact_rows,
    validate_consumer_fact_contract,
)
from chronicle.core import validate_facts
from chronicle.source_package import (
    SOURCE_PACKAGE_ALIASES,
    load_source_package,
    validate_source_package,
)
from chronicle.suite import build_source_suite

WINDOW = "january-2023-may-2026"
HOUSING_TENURE = f"dwp-uc-households-housing-tenure-payment-indicator-{WINDOW}"
HOUSING_ENTITLEMENT = (
    f"dwp-uc-households-housing-entitlement-payment-indicator-{WINDOW}"
)
LCW = f"dwp-uc-households-lcw-entitlement-payment-indicator-{WINDOW}"
LCW_GROUP = f"dwp-uc-households-lcw-entitlement-group-payment-indicator-{WINDOW}"
CARER = f"dwp-uc-households-carer-entitlement-payment-indicator-{WINDOW}"
EMPLOYMENT = f"dwp-uc-people-employment-indicator-{WINDOW}"
HB = "dwp-hb-claimants-client-type-tenure-january-2023-february-2026"
HB_ACCOMMODATION = "dwp-hb-claimants-client-type-tenure-accommodation-type-september-2025-february-2026"
CHILDCARE = "dwp-uc-childcare-element-march-2021-may-2026"
ALL_259 = (
    HOUSING_TENURE,
    HOUSING_ENTITLEMENT,
    LCW,
    LCW_GROUP,
    CARER,
    EMPLOYMENT,
    HB,
    HB_ACCOMMODATION,
    CHILDCARE,
)
TENURES = ("Social Rented Sector", "Private Rented Sector", "Other or unknown")
LCW_KEY = "limited_capability_for_work_entitlement"
# alias -> (element filter key, ported categories, groupby dimension, fact count)
UC_CROSSES = {
    HOUSING_TENURE: (
        "housing_entitlement_tenure",
        {"No Housing Entitlement", *TENURES},
        "dwp.housing_entitlement_tenure",
        492,
    ),
    HOUSING_ENTITLEMENT: (
        "housing_entitlement_tenure",
        {"Yes"},
        "dwp.housing_entitlement_tenure",
        123,
    ),
    LCW: (
        LCW_KEY,
        {"None", "LCW", "LCWRA - Higher", "LCWRA - Lower"},
        "dwp.limited_capability_for_work_entitlement",
        492,
    ),
    LCW_GROUP: (
        LCW_KEY,
        {"LCWRA"},
        "dwp.limited_capability_for_work_entitlement",
        123,
    ),
    CARER: ("carer_entitlement", {"No", "Yes"}, "dwp.carer_entitlement", 246),
}
# The April-December 2025 grouped tenure cell (codes 1, 2 and 3) of the retired
# dwp-uc-households-housing-entitlement-april-december-2025, as archived on 2026-08-21.
RETIRED_GROUPED_HOUSING_CELL = [
    4_097_119,
    4_149_985,
    4_219_939,
    4_268_742,
    4_321_601,
    4_364_070,
    4_404_489,
    4_442_383,
    4_464_277,
]


def _months(start, end):
    year, month = int(start[:4]), int(start[5:])
    out = []
    while f"{year}-{month:02d}" <= end:
        out.append(f"{year}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


UC_MONTHS = _months("2023-01", "2026-05")
HB_MONTHS = _months("2023-01", "2026-02")
ACCOMMODATION_MONTHS = _months("2025-09", "2026-02")
CHILDCARE_MONTHS = _months("2021-03", "2026-05")


def _archive(alias):
    directory = files("db").joinpath(
        "data", "dwp", alias.removeprefix("dwp-").replace("-", "_")
    )
    manifest = yaml.safe_load(directory.joinpath("manifest.yaml").read_text())
    entry = manifest["files"][2025]
    raw = directory.joinpath(entry["filename"]).read_bytes()
    return entry, raw


@pytest.fixture(scope="module")
def facts_259():
    return {alias: load_source_package(alias).build_facts(2025) for alias in ALL_259}


def _fact(facts, *, period, measure_id=None, **filters):
    matches = [
        fact
        for fact in facts
        if fact.period.value == period
        and (measure_id is None or fact.layout.measure_id == measure_id)
        and all(fact.filters.get(key) == value for key, value in filters.items())
    ]
    assert len(matches) == 1, (period, measure_id, filters, len(matches))
    return matches[0].value


def _assert_common_contract(facts, *, months, entity, role, concept, source_concept):
    assert {fact.period.value for fact in facts} == set(months)
    assert all(fact.period.type == "month" for fact in facts)
    assert all(fact.measure.concept == concept for fact in facts)
    assert all(fact.measure.source_concept == source_concept for fact in facts)
    assert all(fact.entity.name == entity for fact in facts)
    assert all(fact.entity.role == role for fact in facts)
    assert all(fact.geography.id == "K03000001" for fact in facts)
    assert all(fact.geography.name == "Great Britain" for fact in facts)
    assert all(fact.assertion == "observation" for fact in facts)
    assert all(fact.provenance_class == "administrative" for fact in facts)
    assert all(fact.source_row_keys and fact.source_cell_keys for fact in facts)
    assert validate_facts(facts).valid
    assert validate_consumer_fact_contract(facts).valid
    assert len(consumer_fact_rows(facts)) == len(facts)


def test_retired_packages_are_no_longer_registered():
    # #259 item 2: the grouped housing cell is replaced by DWP's Housing Entitlement
    # 'Yes' value; the May 2026 childcare tables supersede the August 2025 vintage.
    assert "dwp-uc-households-housing-entitlement-april-december-2025" not in (
        SOURCE_PACKAGE_ALIASES
    )
    assert "dwp-uc-childcare-element-march-2021-august-2025" not in (
        SOURCE_PACKAGE_ALIASES
    )
    assert set(ALL_259) <= set(SOURCE_PACKAGE_ALIASES)


@pytest.mark.parametrize("alias", list(UC_CROSSES))
def test_uc_element_crosses_cover_january_2023_to_may_2026(alias, facts_259):
    element_key, categories, groupby, count = UC_CROSSES[alias]
    report = validate_source_package(load_source_package(alias).package_path, year=2025)
    facts = facts_259[alias]
    detail = [fact for fact in facts if fact.layout.measure_id == "benefit_units"]
    totals = [fact for fact in facts if fact.layout.measure_id == "total_benefit_units"]

    assert report.valid
    assert report.counts["record_set_count"] == 2 * len(UC_MONTHS) == 82
    assert (
        report.counts["row_count"]
        == report.counts["source_record_count"]
        == len(facts)
        == 3 * len(categories) * len(UC_MONTHS)
        == count
    )
    assert len(detail) == 2 * len(totals)
    assert all(fact.layout.table_record_kind == "detail" for fact in detail)
    assert all(fact.layout.table_record_kind == "total" for fact in totals)
    assert all(fact.layout.groupby_dimension == groupby for fact in facts)
    assert {fact.filters[element_key] for fact in facts} == categories
    assert all(
        set(fact.filters) == {element_key, "payment_indicator"} for fact in facts
    )
    assert {fact.filters["payment_indicator"] for fact in detail} == {"No", "Yes"}
    assert {fact.filters["payment_indicator"] for fact in totals} == {"all"}
    assert all(
        {c.variable for c in fact.constraints} == {element_key, "payment_indicator"}
        for fact in detail
    )
    # The publisher Total is guarded by its label, never by an `== all` constraint.
    assert all(
        {c.variable for c in fact.constraints} == {element_key} for fact in totals
    )
    _assert_common_contract(
        facts,
        months=UC_MONTHS,
        entity="benefit_unit",
        role="universal_credit_unit_of_assessment",
        concept="dwp.uc_benefit_units",
        source_concept="dwp.uc_households",
    )


UC_DB = "str:database:UC_Households"
UC_MEASURE = "str:count:UC_Households:V_F_UC_HOUSEHOLDS"
UC_DATE = ("str:field:UC_Households:F_UC_HH_DATE:DATE_NAME", "C_UC_HH_DATE")
UC_PAYMENT = (
    "str:field:UC_Households:V_F_UC_HOUSEHOLDS:HCPAYMENT",
    "C_UC_PAYMENT",
    [0, 1],
    True,
)
HB_DB = "str:database:hb_new"
HB_MEASURE = "str:count:hb_new:V_F_HB_NEW"
HB_DATE = ("str:field:hb_new:F_HB_NEW_DATE:NEW_DATE_NAME", "C_HB_NEW_DATE")
HB_CLIENT = ("str:field:hb_new:V_F_HB_NEW:PENWORK", "C_PENWORK", [1, 2], True)
HB_TENURE = (
    "str:field:hb_new:V_F_HB_NEW:TENURE_PUB",
    "C_HOUSING_SECTOR",
    [1, 2, 99],
    True,
)
UC_BASE_NOTES = {"I", "II", "III", "IV", "p", "r", "z"}
QUERY_SPECS = {
    HOUSING_TENURE: (
        UC_DB, UC_MEASURE, UC_DATE, UC_MONTHS,
        [("str:field:UC_Households:V_F_UC_HOUSEHOLDS:TENURE", "C_UC_HOUSING_TENURE", [0, 1, 2, 3], False), UC_PAYMENT],
        [41, 4, 3], UC_BASE_NOTES,
    ),
    HOUSING_ENTITLEMENT: (
        UC_DB, UC_MEASURE, UC_DATE, UC_MONTHS,
        [("str:field:UC_Households:V_F_UC_HOUSEHOLDS:TENURE", "C_UC_HOUSING_ENTITLEMENT", [0, 1], False), UC_PAYMENT],
        [41, 2, 3], UC_BASE_NOTES,
    ),
    LCW: (
        UC_DB, UC_MEASURE, UC_DATE, UC_MONTHS,
        [("str:field:UC_Households:V_F_UC_HOUSEHOLDS:HCLCW_ENTITLEMENT", "C_UC_LCW_ENTITLEMENT", [0, 1, 2, 3, 4], False), UC_PAYMENT],
        [41, 5, 3], UC_BASE_NOTES | {"VIII", "XIII", "XIV"},
    ),
    LCW_GROUP: (
        UC_DB, UC_MEASURE, UC_DATE, UC_MONTHS,
        [("str:field:UC_Households:V_F_UC_HOUSEHOLDS:HCLCW_ENTITLEMENT", "C_UC_LCW_ENTITLEMENT_GROUP", [0, 1, 2, 3], False), UC_PAYMENT],
        [41, 4, 3], UC_BASE_NOTES | {"VIII", "IX"},
    ),
    CARER: (
        UC_DB, UC_MEASURE, UC_DATE, UC_MONTHS,
        [("str:field:UC_Households:V_F_UC_HOUSEHOLDS:HCCARER_ENTITLEMENT", "C_UC_CARER_ENTITLEMENT", [0, 1], False), UC_PAYMENT],
        [41, 2, 3], UC_BASE_NOTES,
    ),
    EMPLOYMENT: (
        "str:database:UC_Monthly", "str:count:UC_Monthly:V_F_UC_CASELOAD_FULL",
        ("str:field:UC_Monthly:F_UC_DATE:DATE_NAME", "C_UC_DATE"), UC_MONTHS,
        [("str:field:UC_Monthly:V_F_UC_CASELOAD_FULL:EMPLOYMENT_CODE", "C_UC_EMPLOYMENT_2023_BAND", [0, 1, 99], True)],
        [41, 4], {"I", "II", "V", "rev", "z"},
    ),
    HB: (
        HB_DB, HB_MEASURE, HB_DATE, HB_MONTHS, [HB_CLIENT, HB_TENURE],
        [38, 3, 4], {"XXI", "XXV", "XXVI"},
    ),
    HB_ACCOMMODATION: (
        HB_DB, HB_MEASURE, HB_DATE, ACCOMMODATION_MONTHS,
        [HB_CLIENT, HB_TENURE, ("str:field:hb_new:V_F_HB_NEW:SATA", "C_SATA", [1, 2, 9, 99], False)],
        [6, 3, 4, 4], {"XXI", "XXV", "XXVI"},
    ),
}  # fmt: skip


def _value_uri(field, valueset, code):
    return f"{field.replace('str:field:', 'str:value:')}:{valueset}:{code}"


@pytest.mark.parametrize("alias", list(QUERY_SPECS))
def test_issue_259_responses_echo_the_requested_query(alias):
    database, measure, (date_field, date_set), months, axes, shape, notes = QUERY_SPECS[
        alias
    ]
    entry, raw = _archive(alias)
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert len(raw) == entry["size_bytes"]
    assert entry["source_url"] == "https://stat-xplore.dwp.gov.uk/webapi/rest/v1/table"
    response = json.loads(raw)
    recodes = {
        date_field: {
            "map": [
                [_value_uri(date_field, date_set, period.replace("-", ""))]
                for period in months
            ],
            "total": False,
        }
    }
    for field, valueset, codes, total in axes:
        recodes[field] = {
            "map": [[_value_uri(field, valueset, code)] for code in codes],
            "total": total,
        }
    assert response["query"] == {
        "database": database,
        "measures": [measure],
        "recodes": recodes,
        "dimensions": [[field] for field in recodes],
    }
    assert [field["uri"] for field in response["fields"]] == list(recodes)
    assert [len(field["items"]) for field in response["fields"]] == shape
    for field in response["fields"]:
        recode = recodes[field["uri"]]
        assert [item["uris"] for item in field["items"] if "uris" in item] == recode[
            "map"
        ]
        assert [item for item in field["items"] if "uris" not in item] == (
            [{"type": "Total", "labels": ["Total"]}] if recode["total"] else []
        )
    month_items = response["fields"][0]["items"]
    annotations = {
        item["uris"][0].rsplit(":", 1)[1]: item.get("annotationKeys", [])
        for item in month_items
    }
    if database == UC_DB:
        # 18 August 2026 release: June 2024-February 2026 revised, March-May 2026
        # provisional, January 2023-May 2024 unmarked.
        expected = {
            code: ["r"]
            if "202406" <= code <= "202602"
            else ["p"]
            if code >= "202603"
            else []
            for code in annotations
        }
    elif database == "str:database:UC_Monthly":
        # 'rev': revised, and not previously on Stat-Xplore as the data was suspended.
        expected = {
            code: ["rev"] if code in {"202504", "202505"} else []
            for code in annotations
        }
    else:
        expected = {code: [] for code in annotations}
    assert annotations == expected
    assert set(response["annotationMap"]) == notes

    def assert_cube_shape(values, axis=0):
        if axis == len(shape):
            assert type(values) is int
            return
        assert len(values) == shape[axis]
        for value in values:
            assert_cube_shape(value, axis + 1)

    assert set(response["cubes"]) == {measure}
    assert_cube_shape(response["cubes"][measure]["values"])


@pytest.mark.parametrize(
    ("alias", "count", "maximum", "nonzero"),
    [
        (HOUSING_TENURE, 164, 8, 149),
        (HOUSING_ENTITLEMENT, 41, 6, 34),
        (LCW, 164, 8, 103),
        (LCW_GROUP, 41, 8, 33),
        (CARER, 82, 8, 71),
    ],
)
def test_uc_element_total_is_a_publisher_payment_indicator_margin(
    alias, count, maximum, nonzero, facts_259
):
    element_key = UC_CROSSES[alias][0]
    facts = facts_259[alias]
    gaps = []
    for total in [fact for fact in facts if fact.filters["payment_indicator"] == "all"]:
        detail_sum = sum(
            _fact(
                facts,
                period=total.period.value,
                **{
                    element_key: total.filters[element_key],
                    "payment_indicator": payment,
                },
            )
            for payment in ("No", "Yes")
        )
        gaps.append(abs(total.value - detail_sum))
    # Observed on the 18 August 2026 release; the gaps are disclosure control, and
    # this comparison never emits a summed fact.
    assert len(gaps) == count
    assert max(gaps) == maximum
    assert sum(gap != 0 for gap in gaps) == nonzero
    assert all(gap <= 10 for gap in gaps)


def test_uc_element_crosses_pin_publisher_cells(facts_259):
    # Order per category: payment No, Yes, all (the publisher Total).
    samples = {
        HOUSING_TENURE: {
            "2023-01": {"Social Rented Sector": [93_300, 1_549_425, 1_642_728], "No Housing Entitlement": [379_428, 1_425_541, 1_804_970]},
            "2025-04": {"Social Rented Sector": [81_071, 2_253_181, 2_334_247], "Private Rented Sector": [96_396, 1_551_868, 1_648_265]},
            "2026-05": {"Private Rented Sector": [84_330, 1_619_930, 1_704_264], "Other or unknown": [7_727, 123_224, 130_950]},
        },
        HOUSING_ENTITLEMENT: {
            "2023-01": {"Yes": [215_132, 2_951_190, 3_166_328]},
            "2025-04": {"Yes": [185_824, 3_911_297, 4_097_119]},
            "2026-05": {"Yes": [160_795, 4_323_921, 4_484_716]},
        },
        LCW: {
            "2023-01": {"LCWRA - Higher": [16_280, 1_103_516, 1_119_796], "LCW": [407, 23_658, 24_060]},
            "2025-04": {"LCWRA - Higher": [20_964, 2_050_162, 2_071_127], "None": [513_435, 3_896_244, 4_409_680]},
            "2026-05": {"LCWRA - Higher": [18_096, 2_715_434, 2_733_533], "LCWRA - Lower": [65, 672, 735]},
        },
        LCW_GROUP: {
            "2023-01": {"LCWRA": [16_280, 1_103_516, 1_119_796]},
            "2026-05": {"LCWRA": [18_159, 2_716_106, 2_734_269]},
        },
        CARER: {
            "2023-01": {"Yes": [23_997, 524_385, 548_386]},
            "2025-04": {"Yes": [34_498, 1_047_219, 1_081_717], "No": [500_227, 4_937_249, 5_437_476]},
            "2026-05": {"Yes": [33_503, 1_182_372, 1_215_880]},
        },
    }  # fmt: skip
    for alias, by_period in samples.items():
        element_key = UC_CROSSES[alias][0]
        for period, by_category in by_period.items():
            for category, expected in by_category.items():
                assert [
                    _fact(
                        facts_259[alias],
                        period=period,
                        **{element_key: category, "payment_indicator": payment},
                    )
                    for payment in ("No", "Yes", "all")
                ] == expected, (alias, period, category)


def test_housing_entitlement_yes_carries_the_retired_grouped_tenure_cell(facts_259):
    tenure = facts_259[HOUSING_TENURE]
    entitlement = facts_259[HOUSING_ENTITLEMENT]
    # DWP's Housing Entitlement 'Yes' value is the same publisher cell the retired
    # April-December 2025 package requested as one grouped recode of tenure codes
    # 1, 2 and 3 and mislabelled 'Social Rented Sector' (#259 item 2).
    assert [
        _fact(
            entitlement,
            period=period,
            housing_entitlement_tenure="Yes",
            payment_indicator="all",
        )
        for period in _months("2025-04", "2025-12")
    ] == RETIRED_GROUPED_HOUSING_CELL
    # Summing the three tenure rows reproduces 'Yes' only within disclosure control:
    # observed maxima 9, 9 and 10 (39, 38 and 40 of 41 months nonzero).
    for payment, maximum, nonzero in (("No", 9, 39), ("Yes", 9, 38), ("all", 10, 40)):
        gaps = [
            abs(
                _fact(
                    entitlement,
                    period=period,
                    housing_entitlement_tenure="Yes",
                    payment_indicator=payment,
                )
                - sum(
                    _fact(
                        tenure,
                        period=period,
                        housing_entitlement_tenure=category,
                        payment_indicator=payment,
                    )
                    for category in TENURES
                )
            )
            for period in UC_MONTHS
        ]
        assert max(gaps) == maximum
        assert sum(gap != 0 for gap in gaps) == nonzero
        assert all(gap <= 15 for gap in gaps)

    # 'No' is the same publisher code as 'No Housing Entitlement' (identical cells),
    # which is why only the tenure package emits it.
    tenure_cube = json.loads(_archive(HOUSING_TENURE)[1])["cubes"][UC_MEASURE]["values"]
    entitlement_cube = json.loads(_archive(HOUSING_ENTITLEMENT)[1])["cubes"][
        UC_MEASURE
    ]["values"]
    assert [month[0] for month in entitlement_cube] == [
        month[0] for month in tenure_cube
    ]


def test_lcw_categories_and_the_two_rate_lcwra_group(facts_259):
    detail = facts_259[LCW]
    group = facts_259[LCW_GROUP]
    # 'LCW or LCWRA' is zero in every archived cell of both cubes, so it is not ported.
    for alias in (LCW, LCW_GROUP):
        response = json.loads(_archive(alias)[1])
        labels = [item["labels"][0] for item in response["fields"][1]["items"]]
        position = labels.index("LCW or LCWRA")
        cube = response["cubes"][UC_MEASURE]["values"]
        assert all(cell == 0 for month in cube for cell in month[position])
    # 'LCWRA - Lower' is DWP's second LCWRA rate from April 2026.
    lower = {
        (period, payment): _fact(
            detail,
            period=period,
            limited_capability_for_work_entitlement="LCWRA - Lower",
            payment_indicator=payment,
        )
        for period in UC_MONTHS
        for payment in ("No", "Yes", "all")
    }
    assert {key: value for key, value in lower.items() if value} == {
        ("2026-04", "No"): 7,
        ("2026-04", "Yes"): 80,
        ("2026-04", "all"): 88,
        ("2026-05", "No"): 65,
        ("2026-05", "Yes"): 672,
        ("2026-05", "all"): 735,
    }
    for payment in ("No", "Yes", "all"):
        for period in UC_MONTHS:
            group_value = _fact(
                group,
                period=period,
                limited_capability_for_work_entitlement="LCWRA",
                payment_indicator=payment,
            )
            rates = [
                _fact(
                    detail,
                    period=period,
                    limited_capability_for_work_entitlement=rate,
                    payment_indicator=payment,
                )
                for rate in ("LCWRA - Higher", "LCWRA - Lower")
            ]
            if period < "2026-04":
                # One LCWRA rate only: the group cell is the same publisher cell.
                assert group_value == rates[0]
            else:
                # Observed gaps at most 6 against the two-rate sum (disclosure control).
                assert abs(group_value - sum(rates)) <= 6


def test_element_crosses_reproduce_the_april_december_2025_all_claims_packages(
    facts_259,
):
    # Same 18 August 2026 release: the publisher Totals equal the older single-category
    # extracts cell for cell.
    for old_alias, new_alias, element_key, category in (
        (
            "dwp-uc-households-lcwra-entitlement-april-december-2025",
            LCW,
            LCW_KEY,
            "LCWRA - Higher",
        ),
        (
            "dwp-uc-households-carer-entitlement-april-december-2025",
            CARER,
            "carer_entitlement",
            "Yes",
        ),
    ):
        old = load_source_package(old_alias).build_facts(2025)
        assert len(old) == 9
        for fact in old:
            assert fact.value == _fact(
                facts_259[new_alias],
                period=fact.period.value,
                **{element_key: category, "payment_indicator": "all"},
            )


def test_employment_indicator_package_counts_people(facts_259):
    report = validate_source_package(
        load_source_package(EMPLOYMENT).package_path, year=2025
    )
    facts = facts_259[EMPLOYMENT]
    detail = [fact for fact in facts if fact.layout.measure_id == "people"]
    totals = [fact for fact in facts if fact.layout.measure_id == "total_people"]
    categories = [
        "Not in employment (PAYE) or self-employment",
        "In employment (PAYE) or self-employment",
        "Not available",
    ]

    assert report.valid
    assert report.counts["record_set_count"] == 82
    assert len(facts) == report.counts["row_count"] == 162
    assert len(detail) == 121
    assert len(totals) == 41
    assert {fact.filters["employment_indicator"] for fact in detail} == set(categories)
    assert {fact.filters["employment_indicator"] for fact in totals} == {"all"}
    assert all(
        fact.layout.groupby_dimension == "dwp.uc_employment_indicator" for fact in facts
    )
    _assert_common_contract(
        facts,
        months=UC_MONTHS,
        entity="person",
        role="universal_credit_claimant",
        concept="dwp.uc_people",
        source_concept="dwp.people_on_universal_credit",
    )
    # DWP holds the latest month's indicator back until the next release: in May 2026
    # everyone is 'Not available', so only that row and the Total are ported.
    may = {
        fact.filters["employment_indicator"]: fact.value
        for fact in facts
        if fact.period.value == "2026-05"
    }
    assert may == {"Not available": 8_359_744, "all": 8_359_744}
    response = json.loads(_archive(EMPLOYMENT)[1])
    assert response["cubes"]["str:count:UC_Monthly:V_F_UC_CASELOAD_FULL"]["values"][
        -1
    ] == [
        0,
        0,
        8_359_744,
        8_359_744,
    ]
    gaps = [
        abs(
            _fact(facts, period=period, employment_indicator="all")
            - sum(
                _fact(facts, period=period, employment_indicator=c) for c in categories
            )
        )
        for period in UC_MONTHS[:-1]
    ]
    assert max(gaps) == 7
    assert sum(gap != 0 for gap in gaps) == 34
    samples = {
        "2023-01": [3_153_329, 2_646_274, 0, 5_799_603],
        "2025-04": [4_461_078, 3_218_378, 0, 7_679_452],
        "2026-04": [5_177_584, 3_138_941, 0, 8_316_522],
    }
    for period, expected in samples.items():
        assert [
            _fact(facts, period=period, employment_indicator=c)
            for c in [*categories, "all"]
        ] == expected


HB_CLIENTS = ("Working age", "Pension age")
HB_TENURES = (
    "Social Rented Sector",
    "Private Rented Sector",
    "Unknown or missing housing sector",
)


def test_housing_benefit_client_type_tenure_package_keeps_publisher_margins(facts_259):
    report = validate_source_package(load_source_package(HB).package_path, year=2025)
    facts = facts_259[HB]
    detail = [fact for fact in facts if fact.layout.measure_id == "claimants"]
    totals = [fact for fact in facts if fact.layout.measure_id == "total_claimants"]

    assert report.valid
    assert report.counts["record_set_count"] == 2 * len(HB_MONTHS) == 76
    assert len(facts) == report.counts["row_count"] == 12 * len(HB_MONTHS) == 456
    assert len(detail) == len(totals) == 228
    assert {fact.filters["client_type"] for fact in detail} == set(HB_CLIENTS)
    assert {fact.filters["private_or_social_rented"] for fact in detail} == set(
        HB_TENURES
    )
    assert all("all" in fact.filters.values() for fact in totals)
    assert all(fact.layout.groupby_dimension == "dwp.hb_client_type" for fact in facts)
    _assert_common_contract(
        facts,
        months=HB_MONTHS,
        entity="benefit_unit",
        role="housing_benefit_claim",
        concept="dwp.hb_claimants",
        source_concept="dwp.housing_benefit_claimants",
    )
    # DWP-emitted margins against the detail cells (observed on the August 2026 data):
    # over tenure max 8 (65/76 nonzero), over client type max 9 (102/114), grand
    # total max 11 (36/38). Nothing here emits a summed fact.
    over_tenure = [
        abs(
            _fact(facts, period=m, client_type=a, private_or_social_rented="all")
            - sum(
                _fact(facts, period=m, client_type=a, private_or_social_rented=t)
                for t in HB_TENURES
            )
        )
        for m in HB_MONTHS
        for a in HB_CLIENTS
    ]
    over_client = [
        abs(
            _fact(facts, period=m, client_type="all", private_or_social_rented=t)
            - sum(
                _fact(facts, period=m, client_type=a, private_or_social_rented=t)
                for a in HB_CLIENTS
            )
        )
        for m in HB_MONTHS
        for t in HB_TENURES
    ]
    grand = [
        abs(
            _fact(facts, period=m, client_type="all", private_or_social_rented="all")
            - sum(
                _fact(facts, period=m, client_type=a, private_or_social_rented=t)
                for a in HB_CLIENTS
                for t in HB_TENURES
            )
        )
        for m in HB_MONTHS
    ]
    for gaps, maximum, nonzero, count in (
        (over_tenure, 8, 65, 76),
        (over_client, 9, 102, 114),
        (grand, 11, 36, 38),
    ):
        assert len(gaps) == count
        assert max(gaps) == maximum
        assert sum(gap != 0 for gap in gaps) == nonzero
        assert all(gap <= 15 for gap in gaps)
    # Rows: working age, pension age, all; columns: social, private, unknown, all.
    samples = {
        "2023-01": [[1_024_911, 319_623, 108, 1_344_641], [895_371, 213_779, 22, 1_109_174], [1_920_278, 533_399, 132, 2_453_812]],
        "2025-04": [[590_106, 109_311, 248, 699_668], [889_759, 216_026, 28, 1_105_816], [1_479_869, 325_342, 274, 1_805_482]],
        "2026-02": [[300_913, 44_154, 85, 345_159], [887_890, 218_908, 20, 1_106_819], [1_188_808, 263_064, 104, 1_451_978]],
    }  # fmt: skip
    for period, expected in samples.items():
        assert [
            [
                _fact(facts, period=period, client_type=a, private_or_social_rented=t)
                for t in [*HB_TENURES, "all"]
            ]
            for a in [*HB_CLIENTS, "all"]
        ] == expected


def test_housing_benefit_accommodation_type_starts_september_2025(facts_259):
    report = validate_source_package(
        load_source_package(HB_ACCOMMODATION).package_path, year=2025
    )
    facts = facts_259[HB_ACCOMMODATION]
    accommodation = (
        "Specified Accommodation",
        "Temporary Accommodation",
        "Other",
        "Unknown / Missing",
    )

    assert report.valid
    assert report.counts["record_set_count"] == 12
    assert len(facts) == report.counts["row_count"] == 48 * 6 == 288
    assert {fact.filters["accommodation_type"] for fact in facts} == set(accommodation)
    assert all(
        fact.layout.groupby_dimension == "dwp.hb_accommodation_type" for fact in facts
    )
    # DWP publishes the breakdown from September 2025; no earlier month is ported.
    _assert_common_contract(
        facts,
        months=ACCOMMODATION_MONTHS,
        entity="benefit_unit",
        role="housing_benefit_claim",
        concept="dwp.hb_claimants",
        source_concept="dwp.housing_benefit_claimants",
    )
    # Accommodation types summed at the grand margin land on the client type x tenure
    # package's grand total within disclosure control (observed 2, 3, 4, 6, 5, 6).
    gaps = [
        abs(
            _fact(
                facts_259[HB],
                period=m,
                client_type="all",
                private_or_social_rented="all",
            )
            - sum(
                _fact(
                    facts,
                    period=m,
                    client_type="all",
                    private_or_social_rented="all",
                    accommodation_type=s,
                )
                for s in accommodation
            )
        )
        for m in ACCOMMODATION_MONTHS
    ]
    assert gaps == [2, 3, 4, 6, 5, 6]
    # Columns: working age, pension age, all; tenure all.
    samples = {
        "2025-09": {"Specified Accommodation": [190_000, 54_221, 244_218], "Temporary Accommodation": [123_438, 4_527, 127_968], "Other": [140_728, 1_050_003, 1_190_731]},
        "2026-02": {"Specified Accommodation": [196_021, 54_417, 250_434], "Temporary Accommodation": [127_506, 4_384, 131_892], "Unknown / Missing": [31, 8, 38]},
    }  # fmt: skip
    for period, by_type in samples.items():
        for accommodation_type, expected in by_type.items():
            assert [
                _fact(
                    facts,
                    period=period,
                    client_type=a,
                    private_or_social_rented="all",
                    accommodation_type=accommodation_type,
                )
                for a in [*HB_CLIENTS, "all"]
            ] == expected


def test_childcare_element_package_supersedes_the_august_2025_vintage(facts_259):
    report = validate_source_package(
        load_source_package(CHILDCARE).package_path, year=2025
    )
    facts = facts_259[CHILDCARE]
    by_measure = {}
    for fact in facts:
        by_measure.setdefault(fact.layout.measure_id, []).append(fact)

    assert report.valid
    assert report.counts["record_set_count"] == 2 * len(CHILDCARE_MONTHS) == 126
    assert len(facts) == 4 * len(CHILDCARE_MONTHS) == 252
    assert {key: len(value) for key, value in by_measure.items()} == {
        "benefit_units": 63,
        "single_benefit_units": 63,
        "couple_benefit_units": 63,
        "mean_amount": 63,
    }
    assert {fact.period.value for fact in facts} == set(CHILDCARE_MONTHS)
    for measure_id, family_type in (
        ("benefit_units", None),
        ("single_benefit_units", "Single"),
        ("couple_benefit_units", "Couple"),
    ):
        for fact in by_measure[measure_id]:
            assert fact.measure.concept == "dwp.uc_benefit_units_with_childcare_element"
            assert (
                fact.measure.source_concept
                == "dwp.uc_households_receiving_childcare_element"
            )
            assert fact.filters == (
                {"uc_element": "childcare"}
                | ({"family_type": family_type} if family_type else {})
            )
    for fact in by_measure["mean_amount"]:
        assert fact.measure.concept == "dwp.uc_childcare_element_average_award"
        assert fact.filters == {"uc_element": "childcare"}
    # DWP publishes no total amount paid; Chronicle does not multiply count by mean.
    assert not [
        fact
        for fact in facts
        if fact.measure.concept == "dwp.uc_childcare_element_amount"
    ]
    assert all(fact.entity.name == "benefit_unit" for fact in facts)
    assert all(fact.geography.id == "K03000001" for fact in facts)
    assert validate_facts(facts).valid
    assert validate_consumer_fact_contract(facts).valid
    # Table 1 is rounded to the nearest 1,000: single + couple is within 1,000 of the
    # total (observed 11 of 63 months off by exactly 1,000).
    gaps = [
        abs(
            _fact(facts, period=m, measure_id="benefit_units")
            - _fact(facts, period=m, measure_id="single_benefit_units")
            - _fact(facts, period=m, measure_id="couple_benefit_units")
        )
        for m in CHILDCARE_MONTHS
    ]
    assert max(gaps) == 1_000
    assert sum(gap != 0 for gap in gaps) == 11
    # Order: total, single, couple, mean amount (GBP).
    samples = {
        "2021-03": [88_000, 70_000, 18_000, 330],
        "2025-01": [187_000, 151_000, 37_000, 400],
        "2025-08": [165_000, 134_000, 31_000, 420],
        "2026-05": [164_000, 134_000, 30_000, 400],
    }
    for period, expected in samples.items():
        assert [
            _fact(facts, period=period, measure_id=measure_id)
            for measure_id in (
                "benefit_units",
                "single_benefit_units",
                "couple_benefit_units",
                "mean_amount",
            )
        ] == expected


@pytest.mark.parametrize("alias", ALL_259)
def test_issue_259_packages_pass_agent_acceptance(alias, tmp_path):
    output_dir = tmp_path / alias
    report = build_source_suite(alias, output_dir, year=2025)
    acceptance = json.loads(
        (output_dir / "reports" / "agent_acceptance.json").read_text()
    )

    assert report.valid
    assert acceptance["valid"]
    assert acceptance["counts"]["row_semantic_error_count"] == 0
