"""Chronicle-owned dimension and value labels on consumer facts (chronicle#261).

Microcosm's calibration hierarchy names every dimension and value a target
inherits from the facts it selects and takes those names only from Chronicle.
These tests pin how Chronicle resolves the labels, the check that mirrors
Microcosm's reader, and that labels never move a fact key or value.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from chronicle.bundle import (
    UK_BUNDLE_SOURCES,
    _cross_package_dimension_label_errors,
    _dimension_label_reports,
)
from chronicle.consumer_contract import consumer_fact_row
from chronicle.core import (
    AggregateConstraint,
    AggregateFact,
    Aggregation,
    EntityDimension,
    GeographyDimension,
    Measure,
    PeriodDimension,
    SourceProvenance,
    SourceRecordLayout,
)
from chronicle.dimension_labels import (
    dimension_label_issues,
    dimension_value_id,
    label_facts,
    unused_label_declarations,
)
from chronicle.source_package import (
    _dimension_labels_from_mapping,
    _dimension_value_labels_from_mapping,
)
from chronicle.sources.cells import SourceArtifactMetadata
from chronicle.sources.rows import SourceRow, build_source_row_key
from policyengine_chronicle.schema import validate_consumer_fact_row

SHA = "cd" * 32
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


def _fact(
    filters=None,
    *,
    constraint_labels=None,
    groupby="dwp.carer_entitlement",
    value_id="carer_entitlement_yes",
    value_label="Yes",
    source_row_keys=(),
):
    filters = dict(filters or {})
    constraint_labels = constraint_labels or {}
    return AggregateFact(
        value=100,
        period=PeriodDimension(type="month", value="2025-01"),
        geography=GeographyDimension(
            level="country", id="K03000001", name="Great Britain"
        ),
        entity=EntityDimension(name="benefit_unit"),
        measure=Measure(concept="dwp.uc_benefit_units", unit="count"),
        aggregation=Aggregation(method="sum"),
        provenance_class="administrative",
        source=SourceProvenance(
            source_name="dwp",
            source_table="Households on Universal Credit",
            source_file="uc.json",
            url="https://stat-xplore.dwp.gov.uk/webapi/rest/v1/table",
            vintage="stat_xplore_2026_09_11",
            extracted_at="2026-09-11",
            extraction_method="test",
            source_sha256=SHA,
            source_size_bytes=10,
            raw_r2_bucket="ledger-raw",
            raw_r2_key=f"raw/uk/dwp/uc/{SHA}/uc.json",
            raw_r2_uri=f"r2://ledger-raw/raw/uk/dwp/uc/{SHA}/uc.json",
        ),
        filters=filters,
        domain="universal_credit",
        source_record_id=f"dwp.uc.{value_id}.benefit_units",
        source_cell_keys=("ledger.source_cell.v1:" + "a" * 24,),
        source_row_keys=tuple(source_row_keys),
        constraints=tuple(
            AggregateConstraint(
                variable=key,
                operator="==",
                value=value,
                label=constraint_labels.get(key),
            )
            for key, value in filters.items()
            if value != "all"
        ),
        layout=SourceRecordLayout(
            record_set_id="dwp.uc.month2025_01",
            record_set_spec_id="dwp.uc.month2025_01.v1",
            groupby_dimension=groupby,
            groupby_value_id=value_id,
            groupby_value_label=value_label,
            measure_id="benefit_units",
        ),
    )


def _source_row(values):
    return SourceRow(
        artifact=SourceArtifactMetadata(
            source_name="dwp",
            source_table="Households on Universal Credit",
            source_file="uc.json",
            url=None,
            vintage="stat_xplore_2026_09_11",
            sha256=SHA,
            size_bytes=10,
            extracted_at="2026-09-11",
            extraction_method="test",
        ),
        sheet_name="api_response",
        row_number=2,
        values=values,
    )


def _row(**overrides):
    """A consumer row labelled the way Microcosm's hierarchy reads it."""
    row = {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:" + "0" * 24,
        "dimensions": {"payment_indicator": "Yes"},
        "dimension_labels": {
            "payment_indicator": "Payment Indicator",
            "dwp.carer_entitlement": "Carer Entitlement",
        },
        "dimension_value_labels": {"payment_indicator": {"Yes": "Yes"}},
        "layout": {
            "groupby_dimension": "dwp.carer_entitlement",
            "groupby_dimension_label": "Carer Entitlement",
            "groupby_value_id": "carer_entitlement_yes",
            "groupby_value_label": "Yes",
        },
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "true"),
        (False, "false"),
        (2, "2"),
        (2.5, "2.5"),
        ("No", "No"),
        (Decimal("1.50"), "1.50"),
        (["b", "a"], '["b","a"]'),
        ({"b": 1, "a": 2}, '{"a":2,"b":1}'),
    ],
)
def test_dimension_value_id_is_the_form_microcosm_looks_labels_up_by(value, expected):
    assert dimension_value_id(value) == expected


def test_declared_labels_win_and_label_the_groupby_dimension():
    fact = _fact(
        {"payment_indicator": "Yes"},
        constraint_labels={"payment_indicator": "payment indicator"},
    )

    [labelled] = label_facts(
        [fact],
        dimension_labels={
            "dwp.carer_entitlement": "Carer Entitlement",
            "payment_indicator": "Payment Indicator",
        },
        dimension_value_labels={"payment_indicator": {"Yes": "Yes"}},
    )

    assert labelled.dimension_labels == {
        "dwp.carer_entitlement": "Carer Entitlement",
        "payment_indicator": "Payment Indicator",
    }
    assert labelled.layout.groupby_dimension_label == "Carer Entitlement"
    assert labelled.dimension_value_labels == {
        "payment_indicator": {"Yes": "Yes"},
        "dwp.carer_entitlement": {"carer_entitlement_yes": "Yes"},
    }


def test_stat_xplore_field_labels_name_the_columns_they_head():
    [labelled] = label_facts(
        [_fact({"carer_entitlement": "Yes"})],
        field_labels={"carer_entitlement": "Carer Entitlement"},
    )

    assert labelled.dimension_labels == {"carer_entitlement": "Carer Entitlement"}


def test_a_constraint_label_names_its_filter_unless_it_repeats_the_identifier():
    [labelled] = label_facts(
        [
            _fact(
                {"payment_method": "direct_debit", "fuel": "gas"},
                constraint_labels={"payment_method": "Payment method", "fuel": "fuel"},
            )
        ]
    )

    assert labelled.dimension_labels == {"payment_method": "Payment method"}


def test_constraint_labels_that_disagree_label_nothing():
    facts = label_facts(
        [
            _fact({"fuel": "gas"}, constraint_labels={"fuel": "Fuel"}),
            _fact(
                {"fuel": "gas"},
                constraint_labels={"fuel": "Fuel type"},
                value_id="carer_entitlement_no",
            ),
        ]
    )

    assert all("fuel" not in fact.dimension_labels for fact in facts)


def test_the_groupby_row_label_labels_a_filter_on_the_same_axis():
    [labelled] = label_facts(
        [
            _fact(
                {
                    "household_income_band": "gbp15_000_gbp19_999",
                    "region": "gbp15_000_gbp19_999",
                },
                groupby="desnz.need.household_income_band",
                value_id="gbp15_000_gbp19_999",
                value_label="£15,000 to £19,999 ",
            )
        ]
    )

    assert labelled.dimension_value_labels["household_income_band"] == {
        "gbp15_000_gbp19_999": "£15,000 to £19,999"
    }
    assert "region" not in labelled.dimension_value_labels


def test_publisher_row_text_labels_the_filters_it_evidences():
    row = _source_row({"carer_entitlement": "Yes", "payment_indicator": "Total"})
    key = build_source_row_key(row)

    evidenced, contradicted = label_facts(
        [
            _fact(
                {"carer_entitlement": "Yes", "payment_indicator": "all"},
                source_row_keys=(key,),
            ),
            _fact(
                {"carer_entitlement": "No"},
                source_row_keys=(key,),
                value_id="carer_entitlement_no",
            ),
        ],
        source_rows=[row],
    )

    assert evidenced.dimension_value_labels["carer_entitlement"] == {"Yes": "Yes"}
    assert evidenced.dimension_value_labels["payment_indicator"] == {"all": "Total"}
    assert "carer_entitlement" not in contradicted.dimension_value_labels


def test_numbers_and_iso_dates_label_themselves():
    [labelled] = label_facts(
        [
            _fact(
                {
                    "age": 5,
                    "observation_date": "2023-01-02",
                    "flag": True,
                    "band": "5_to_10",
                }
            )
        ]
    )

    assert labelled.dimension_value_labels == {
        "age": {"5": "5"},
        "observation_date": {"2023-01-02": "2023-01-02"},
        "dwp.carer_entitlement": {"carer_entitlement_yes": "Yes"},
    }


def test_labels_never_move_a_fact_key_or_value():
    fact = _fact({"carer_entitlement": "Yes"})
    [labelled] = label_facts(
        [fact],
        dimension_labels={
            "dwp.carer_entitlement": "Carer Entitlement",
            "carer_entitlement": "Carer Entitlement",
        },
        dimension_value_labels={"carer_entitlement": {"Yes": "Yes"}},
    )

    plain = consumer_fact_row(fact)
    rich = consumer_fact_row(labelled)

    for field_name in (*_KEY_FIELDS, "value", "dimensions", "universe_constraints"):
        assert rich[field_name] == plain[field_name]
    assert set(rich) - set(plain) == {"dimension_labels", "dimension_value_labels"}
    assert rich["layout"] == {
        **plain["layout"],
        "groupby_dimension_label": "Carer Entitlement",
    }
    assert rich["schema_version"] == "chronicle.consumer_fact.v2"
    validate_consumer_fact_row(rich, 1, "labelled.jsonl")


def test_unused_declarations_are_named():
    unused = unused_label_declarations(
        [_fact({"payment_indicator": "Yes"})],
        dimension_labels={"payment_indicator": "Payment Indicator", "fuel": "Fuel"},
        dimension_value_labels={
            "payment_indicator": {"Yes": "Yes", "No": "No"},
            "dwp.carer_entitlement": {"carer_entitlement_yes": "Yes"},
        },
    )

    assert unused == [
        "dimension_labels.fuel",
        "dimension_value_labels.payment_indicator.No",
    ]


def test_a_complete_row_has_no_label_issues():
    assert dimension_label_issues([_row(), _row()]) == []


def test_unlabelled_dimensions_and_values_are_counted():
    issues = dimension_label_issues(
        [
            _row(dimension_labels={"dwp.carer_entitlement": "Carer Entitlement"}),
            _row(dimension_value_labels={}),
        ]
    )

    assert [(issue.code, issue.dimension_id, issue.value_id) for issue in issues] == [
        ("missing_dimension_label", "payment_indicator", None),
        ("missing_dimension_value_label", "payment_indicator", "Yes"),
    ]
    assert {issue.fact_count for issue in issues} == {1}


def test_a_groupby_label_that_disagrees_with_dimension_labels_conflicts():
    row = _row()
    row["layout"] = {**row["layout"], "groupby_dimension_label": "Carer"}

    [issue] = dimension_label_issues([row])

    assert issue.code == "conflicting_dimension_label"
    assert issue.labels == ("Carer", "Carer Entitlement")


def test_a_dimension_or_value_labelled_two_ways_across_rows_conflicts():
    issues = dimension_label_issues(
        [
            _row(),
            _row(
                dimension_labels={
                    "payment_indicator": "Payment indicator",
                    "dwp.carer_entitlement": "Carer Entitlement",
                },
                dimension_value_labels={"payment_indicator": {"Yes": "In payment"}},
            ),
        ]
    )

    assert [(issue.code, issue.dimension_id, issue.labels) for issue in issues] == [
        (
            "conflicting_dimension_label",
            "payment_indicator",
            ("Payment Indicator", "Payment indicator"),
        ),
        (
            "conflicting_dimension_value_label",
            "payment_indicator",
            ("In payment", "Yes"),
        ),
    ]


def test_publisher_row_labels_that_drift_across_periods_are_their_own_issue():
    april, may = _row(), _row()
    april["layout"] = {**april["layout"], "groupby_value_label": "April 2021"}
    may["layout"] = {**may["layout"], "groupby_value_label": "May 2021"}

    [issue] = dimension_label_issues([april, may])

    assert issue.code == "conflicting_groupby_value_label"
    assert issue.labels == ("April 2021", "May 2021")


def test_bundle_requires_labels_for_uk_packages_only():
    unlabelled = _row(dimension_labels={}, dimension_value_labels={})
    unlabelled["layout"] = {
        key: value
        for key, value in unlabelled["layout"].items()
        if key != "groupby_dimension_label"
    }
    drifting = _row()
    drifting["layout"] = {**drifting["layout"], "groupby_value_label": "April 2021"}

    assert _dimension_label_reports("soi-table-1-1", [unlabelled]) == ([], [])
    errors, warnings = _dimension_label_reports(
        UK_BUNDLE_SOURCES[0], [unlabelled, _row(), drifting]
    )

    assert {error.code for error in errors} == {
        "missing_dimension_label",
        "missing_dimension_value_label",
    }
    assert [warning.code for warning in warnings] == ["conflicting_groupby_value_label"]
    assert all(issue.source == UK_BUNDLE_SOURCES[0] for issue in errors + warnings)


def test_one_uk_dimension_id_needs_one_label_across_packages():
    errors = _cross_package_dimension_label_errors(
        {
            "geography": {"Geography": ["ons-a"], "Area": ["nrs-b"]},
            "age": {"Age": ["ons-a", "nrs-b"]},
        }
    )

    assert [(error.code, error.key) for error in errors] == [
        ("conflicting_dimension_label_across_packages", "geography")
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (["payment_indicator"], "must be a mapping"),
        ({False: "No"}, "quoted, non-empty string"),
        ({"payment_indicator": " "}, "non-empty string"),
        ({"payment_indicator": 3}, "non-empty string"),
    ],
)
def test_package_label_declarations_are_quoted_ids_and_nonempty_labels(
    payload, message
):
    with pytest.raises(TypeError, match=message):
        _dimension_labels_from_mapping(payload, context="packages/x: dimension_labels")


def test_package_value_label_declarations_load_trimmed():
    assert _dimension_value_labels_from_mapping(
        {"payment_indicator": {"Yes": " Yes ", "all": "Total"}},
        context="packages/x: dimension_value_labels",
    ) == {"payment_indicator": {"Yes": "Yes", "all": "Total"}}
    with pytest.raises(TypeError, match="quoted, non-empty string"):
        _dimension_value_labels_from_mapping(
            {True: {"Yes": "Yes"}},
            context="packages/x: dimension_value_labels",
        )
