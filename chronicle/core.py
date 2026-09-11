"""Canonical Chronicle aggregate facts and validation.

This module defines source-backed aggregate facts, deterministic fact keys,
human-readable labels, and schema-level validation for Chronicle fact sets.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

from chronicle.epoch import EMIT_EPOCH, HASH_DOMAINS, Epoch, hash_domain

Scalar = str | int | float | bool | None

# Year-typed periods store one integer. Split-label years store the opening
# year of the publisher's label: UK fiscal year FY2024-25 is fiscal_year 2024
# (packages/hmrc/vat_firm_targets_2024_25) and academic year AY 2024/25 is
# academic_year 2024 (#131). Publishers that label a fiscal year with a single
# year keep that label year (US federal FY2024 -> 2024,
# packages/cbo/individual_income_tax_receipts_2026_02).
# New Zealand labels its April-March income tax year by the ending year, so the
# publisher's "2024 tax year" is tax_year 2024 with PeriodCoverage 2023-04-01
# through 2024-03-31 and the publisher label recorded in source_period_label.
# Quarter values use YYYY-QN and ISO week values use YYYY-Www; exact coverage
# dates remain explicit in PeriodCoverage so source observation dates are not
# inferred from the compact period identity.
# Caution: the EES helper _academic_year_end (chronicle/sources/rows.py) names
# value COLUMNS
# by the academic year's END year ("2024/25" -> 2025); that is source-layout
# naming only and must not leak into fact periods.
ALLOWED_PERIOD_TYPES = {
    "calendar_year",
    "tax_year",
    "fiscal_year",
    "academic_year",
    "month",
    "quarter",
    "week",
}
ALLOWED_GEOGRAPHY_LEVELS = {
    "country",
    "region",
    "nuts1",
    "state",
    "county",
    "commune",
    "congressional_district",
    "state_legislative_district_upper",
    "state_legislative_district_lower",
    "constituency",
    "local_authority",
    "msoa",
    "metro_area",
    "zip_code",
    "statistical_scope",
}
ALLOWED_ENTITIES = {
    "person",
    "household",
    "tax_unit",
    "family",
    "benefit_unit",
    "return",
    "pension_plan",
    "government",
    "dwelling",
    "institutional_sector",
    "social_protection_scheme",
    "firm",
}
ALLOWED_AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "quantile",
    "rate",
    "ratio",
    "share",
}
ALLOWED_CONSTRAINT_OPERATORS = {"==", "!=", ">", ">=", "<", "<=", "in"}
ALLOWED_CONCEPT_RELATIONS = {
    "exact",
    "broad_match",
    "narrow_match",
    "approximate",
    "source_label",
}
ALLOWED_ASSERTIONS = {"observation", "source_projection"}
DEFAULT_ASSERTION = "observation"
ALLOWED_PROVENANCE_CLASSES = {
    "administrative",
    "census",
    "model_output",
    "survey_aggregate",
}
ALLOWED_PERIOD_BASES = {
    "calendar",
    "tax",
    "fiscal",
    "survey_reference",
    "projection_horizon",
}
ALLOWED_ACCOUNTING_BASES = {"cash", "accrual"}
FACT_KEY_PREFIX = hash_domain("fact")


@dataclass(frozen=True)
class PeriodDimension:
    """Fact period identity.

    ``value`` identifies the period the fact's value refers to (its reference
    period), which may differ from the release it was published under; release
    labeling belongs in ``SourceProvenance.vintage`` and
    ``PeriodCoverage.source_period_label``.
    """

    type: str
    value: int | str


@dataclass(frozen=True)
class PeriodCoverage:
    """Non-identity period provenance for a fact's reference period.

    These fields disambiguate what span of time a published value covers when
    the coarse ``PeriodDimension`` is not enough: lagged survey reference
    periods (SILC incomes reference the year before the survey label),
    fiscal-versus-calendar accounting, and publisher projection horizons.
    They are provenance and audit metadata, excluded from stable fact keys.
    """

    start_date: str | None = None
    end_date: str | None = None
    basis: str | None = None
    source_period_label: str | None = None
    accounting_basis: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class GeographyDimension:
    """Fact geography identity.

    ``name`` is human-readable metadata and is excluded from stable keys.
    """

    level: str
    id: str
    vintage: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class EntityDimension:
    """Unit/entity measured by the fact."""

    name: str
    role: str | None = None


@dataclass(frozen=True)
class Measure:
    """Simulator-neutral fact measure."""

    concept: str
    unit: str
    source_concept: str | None = None
    concept_relation: str | None = None
    concept_authority: str | None = None
    concept_evidence_url: str | None = None
    concept_evidence_notes: str | None = None
    legal_vintage: str | None = None


@dataclass(frozen=True)
class Aggregation:
    """How source observations are aggregated into the fact value."""

    method: str
    denominator: str | None = None


@dataclass(frozen=True)
class SourceProvenance:
    """Source identity and extraction provenance for a fact."""

    source_name: str | None
    source_table: str | None = None
    source_file: str | None = None
    url: str | None = None
    vintage: str | None = None
    extracted_at: str | None = None
    extraction_method: str | None = None
    method_notes: str | None = None
    source_sha256: str | None = None
    source_size_bytes: int | None = None
    raw_r2_bucket: str | None = None
    raw_r2_key: str | None = None
    raw_r2_uri: str | None = None


@dataclass(frozen=True)
class AggregateConstraint:
    """One semantic filter that scopes an aggregate fact."""

    variable: str
    operator: str
    value: Scalar
    unit: str | None = None
    role: str = "filter"
    label: str | None = None


@dataclass(frozen=True)
class SourceRecordLayout:
    """Non-semantic layout metadata for rebuilding source tables.

    These fields help humans and tools reconstitute compact source tables from
    atomic facts. They are intentionally excluded from stable fact keys.
    """

    record_set_id: str | None = None
    record_set_spec_id: str | None = None
    record_set_spec_hash: str | None = None
    groupby_dimension: str | None = None
    groupby_value_id: str | None = None
    groupby_value_label: str | None = None
    groupby_ordinal: int | None = None
    measure_id: str | None = None
    measure_label: str | None = None
    measure_ordinal: int | None = None
    source_row_id: str | None = None
    source_column_id: str | None = None
    table_record_kind: str | None = None
    parent_record_set_id: str | None = None
    total_record_id: str | None = None
    groupby_dimension_label: str | None = None


@dataclass(frozen=True)
class AggregateFact:
    """Canonical Chronicle published aggregate fact.

    Every fact is a source-backed claim. ``assertion`` records what kind of
    claim the publisher made: ``observation`` for measured or administered
    outcomes as published, ``source_projection`` for the publisher's own
    forward-looking estimate (CBO baselines, BFP outlooks, SSA trustees
    tables, TPC/JCT scores). PolicyEngine-computed values are never facts;
    aged, uprated, or reconciled levels belong in downstream builds.

    ``provenance_class`` records the publisher's measurement basis and has no
    default. Survey aggregates additionally require ``survey_instrument``;
    that field is forbidden for administrative, census, and model outputs.
    """

    value: int | float | str | Decimal
    period: PeriodDimension
    geography: GeographyDimension
    entity: EntityDimension
    measure: Measure
    aggregation: Aggregation
    source: SourceProvenance
    provenance_class: str
    survey_instrument: str | None = None
    filters: dict[str, Scalar] = field(default_factory=dict)
    domain: str = "all"
    label: str | None = None
    source_record_id: str | None = None
    source_cell_keys: tuple[str, ...] = ()
    source_row_keys: tuple[str, ...] = ()
    constraints: tuple[AggregateConstraint, ...] = ()
    layout: SourceRecordLayout | None = None
    assertion: str = DEFAULT_ASSERTION
    period_coverage: PeriodCoverage | None = None
    # Display metadata, never part of a fact key (chronicle#261): a label for
    # each filter and layout groupby dimension id, and for each of their values
    # keyed by ``consumer_contract.dimension_value_id``.
    dimension_labels: dict[str, str] = field(default_factory=dict)
    dimension_value_labels: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationIssue:
    """One fact validation issue."""

    code: str
    message: str
    field: str | None = None
    fact_key: str | None = None
    fact_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable issue."""
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class ValidationReport:
    """Validation and QA summary for a fact set."""

    fact_count: int
    counts: dict[str, dict[str, int]]
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether the fact set has no validation errors."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "valid": self.valid,
            "fact_count": self.fact_count,
            "counts": self.counts,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def build_fact_key(
    fact: AggregateFact,
    *,
    epoch: Epoch = EMIT_EPOCH,
) -> str:
    """Build a stable key from fact schema fields, not human labels."""
    payload = _canonical_key_payload(fact)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{hash_domain('fact', epoch)}:{digest}"


def build_label(fact: AggregateFact) -> str:
    """Build a human-readable label from fact metadata."""
    concept = _humanize(fact.measure.concept)
    aggregation = _aggregation_label(fact)
    entity = _humanize(fact.entity.name)
    period = f"{fact.period.value} {_humanize(fact.period.type)}"
    geography = fact.geography.name or fact.geography.id
    source = _source_label(fact.source)

    label = f"{geography} {period} {aggregation} {concept} for {entity}"
    if fact.filters:
        label = f"{label} ({_format_filters(fact.filters)})"
    if source:
        label = f"{label} [{source}]"
    return label


def _aggregation_label(fact: AggregateFact) -> str:
    if fact.aggregation.method == "sum" and fact.measure.unit == "count":
        label = "count"
    else:
        label = _humanize(fact.aggregation.method)
    if fact.assertion == "source_projection":
        return f"projected {label}"
    return label


def build_aggregate_constraints(
    fact: AggregateFact,
) -> tuple[AggregateConstraint, ...]:
    """Build first-class aggregate constraints from fact metadata.

    Source loaders can provide explicit constraints. For the current fixture
    slice, older fact records carry SOI bracket boundaries in ``filters``;
    this function lifts those into queryable semantic constraints.
    """
    if fact.constraints:
        return fact.constraints

    constraints: list[AggregateConstraint] = []
    filing_status = fact.filters.get("filing_status")
    if filing_status not in (None, "all"):
        constraints.append(
            AggregateConstraint(
                variable="irs_soi.filing_status",
                operator="==",
                value=filing_status,
                label="Filing status",
            )
        )

    lower = fact.filters.get("agi_lower_usd")
    if lower is not None:
        constraints.append(
            AggregateConstraint(
                variable="irs_soi.adjusted_gross_income",
                operator=">=",
                value=lower,
                unit="usd",
                label="Adjusted gross income lower bound",
            )
        )

    upper = fact.filters.get("agi_upper_usd")
    if upper is not None:
        constraints.append(
            AggregateConstraint(
                variable="irs_soi.adjusted_gross_income",
                operator="<",
                value=upper,
                unit="usd",
                label="Adjusted gross income upper bound",
            )
        )

    handled_filters = {
        "agi_lower_usd",
        "agi_upper_usd",
        "filing_status",
        "income_range",
    }
    child_count = fact.filters.get("eitc_child_count")
    if child_count not in (None, "all"):
        if child_count == "3plus":
            constraints.append(
                AggregateConstraint(
                    variable="us.tax.earned_income_credit_qualifying_children",
                    operator=">=",
                    value=3,
                    unit="count",
                    label="EITC qualifying children",
                )
            )
        else:
            constraints.append(
                AggregateConstraint(
                    variable="us.tax.earned_income_credit_qualifying_children",
                    operator="==",
                    value=child_count,
                    unit="count",
                    label="EITC qualifying children",
                )
            )
        handled_filters.add("eitc_child_count")

    for key, value in sorted(fact.filters.items()):
        if key in handled_filters or value in (None, "all"):
            continue
        constraints.append(
            AggregateConstraint(
                variable=key,
                operator="==",
                value=value,
                label=_humanize(key),
            )
        )

    return tuple(constraints)


def validate_fact(fact: AggregateFact) -> tuple[ValidationIssue, ...]:
    """Validate one aggregate fact."""
    errors: list[ValidationIssue] = []

    _require_nonempty(errors, fact.measure.concept, "measure.concept")
    _require_nonempty(errors, fact.measure.unit, "measure.unit")
    _validate_measure_concept_alignment(errors, fact.measure)
    _require_nonempty(errors, fact.domain, "domain")

    if fact.period.type not in ALLOWED_PERIOD_TYPES:
        errors.append(
            _issue(
                "malformed_period",
                f"Unsupported period type: {fact.period.type!r}",
                "period.type",
            )
        )
    if isinstance(fact.period.value, str) and not fact.period.value.strip():
        errors.append(
            _issue("missing_period", "Period value is required", "period.value")
        )

    if fact.geography.level not in ALLOWED_GEOGRAPHY_LEVELS:
        errors.append(
            _issue(
                "malformed_geography",
                f"Unsupported geography level: {fact.geography.level!r}",
                "geography.level",
            )
        )
    _require_nonempty(errors, fact.geography.id, "geography.id")

    if fact.entity.name not in ALLOWED_ENTITIES:
        errors.append(
            _issue(
                "malformed_entity",
                f"Unsupported entity: {fact.entity.name!r}",
                "entity.name",
            )
        )

    if fact.aggregation.method not in ALLOWED_AGGREGATIONS:
        errors.append(
            _issue(
                "malformed_aggregation",
                f"Unsupported aggregation method: {fact.aggregation.method!r}",
                "aggregation.method",
            )
        )

    if fact.assertion not in ALLOWED_ASSERTIONS:
        errors.append(
            _issue(
                "malformed_assertion",
                f"Unsupported assertion: {fact.assertion!r}",
                "assertion",
            )
        )

    _validate_provenance_class(errors, fact)

    _validate_value(errors, fact.value)
    _validate_filters(errors, fact.filters)
    _validate_constraints(errors, fact.constraints)
    _validate_provenance(errors, fact.source)
    _validate_lineage_keys(
        errors,
        fact.source_cell_keys,
        field="source_cell_keys",
        domain="source_cell",
    )
    _validate_lineage_keys(
        errors,
        fact.source_row_keys,
        field="source_row_keys",
        domain="source_row",
    )
    if fact.period_coverage is not None:
        _validate_period_coverage(errors, fact.period_coverage)

    return tuple(errors)


def validate_facts(facts: list[AggregateFact]) -> ValidationReport:
    """Validate a fact set and return QA counts plus issues."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    key_indices: dict[str, list[int]] = {}

    for index, fact in enumerate(facts):
        key = build_fact_key(fact)
        key_indices.setdefault(key, []).append(index)
        for issue in validate_fact(fact):
            errors.append(
                ValidationIssue(
                    code=issue.code,
                    message=issue.message,
                    field=issue.field,
                    fact_key=key,
                    fact_index=index,
                )
            )
        if not fact.label:
            warnings.append(
                ValidationIssue(
                    code="missing_label",
                    message="Fact has no human-readable label metadata",
                    field="label",
                    fact_key=key,
                    fact_index=index,
                )
            )

    for key, indices in key_indices.items():
        if len(indices) > 1:
            errors.append(
                ValidationIssue(
                    code="duplicate_key",
                    message=f"Duplicate fact key appears at indices {indices}",
                    fact_key=key,
                    fact_index=indices[0],
                )
            )

    return ValidationReport(
        fact_count=len(facts),
        counts=fact_counts(facts),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def fact_counts(facts: list[AggregateFact]) -> dict[str, dict[str, int]]:
    """Count facts across the main QA dimensions."""
    return {
        "by_source": _counter_dict(
            fact.source.source_name or "missing" for fact in facts
        ),
        "by_geography": _counter_dict(
            f"{fact.geography.level}:{fact.geography.id}" for fact in facts
        ),
        "by_entity": _counter_dict(fact.entity.name for fact in facts),
        "by_period": _counter_dict(
            f"{fact.period.type}:{fact.period.value}" for fact in facts
        ),
        "by_assertion": _counter_dict(fact.assertion for fact in facts),
        "by_provenance_class": _counter_dict(fact.provenance_class for fact in facts),
        "missing_labels": {"count": sum(1 for fact in facts if not fact.label)},
        "missing_provenance": {
            "count": sum(1 for fact in facts if _has_missing_provenance(fact))
        },
        "missing_lineage": {
            "count": sum(
                1
                for fact in facts
                if not fact.source_cell_keys and not fact.source_row_keys
            )
        },
    }


def _canonical_key_payload(fact: AggregateFact) -> dict[str, Any]:
    payload = {
        "period": asdict(fact.period),
        "geography": {
            "level": fact.geography.level,
            "id": fact.geography.id,
            "vintage": fact.geography.vintage,
        },
        "entity": asdict(fact.entity),
        "measure": asdict(fact.measure),
        "aggregation": asdict(fact.aggregation),
        "domain": fact.domain,
        "filters": {
            key: fact.filters[key]
            for key in sorted(fact.filters)
            if fact.filters[key] is not None
        },
        "source": {
            "source_name": fact.source.source_name,
            "source_table": fact.source.source_table,
            "source_file": fact.source.source_file,
            "vintage": fact.source.vintage,
        },
    }
    if fact.constraints:
        payload["constraints"] = [asdict(constraint) for constraint in fact.constraints]
    if fact.assertion != DEFAULT_ASSERTION:
        payload["assertion"] = fact.assertion
    return payload


def _validate_value(errors: list[ValidationIssue], value: Any) -> None:
    if isinstance(value, Decimal):
        return
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        errors.append(_issue("malformed_value", "Fact value must be scalar", "value"))
        return
    if isinstance(value, str) and not value.strip():
        errors.append(_issue("missing_value", "Fact value is required", "value"))


def _validate_lineage_keys(
    errors: list[ValidationIssue],
    keys: tuple[str, ...],
    *,
    field: str,
    domain: str,
) -> None:
    pair = HASH_DOMAINS[domain]
    seen: set[str] = set()
    for key in keys:
        if not isinstance(key, str):
            errors.append(
                _issue(
                    "malformed_lineage_key",
                    f"Unsupported lineage key {key!r}; accepted prefixes are "
                    f"{pair.ledger!r} and {pair.chronicle!r}",
                    field,
                )
            )
            continue
        try:
            canonical_key = pair.key_for_epoch(key, Epoch.LEDGER)
        except ValueError as error:
            errors.append(
                _issue(
                    "malformed_lineage_key",
                    str(error),
                    field,
                )
            )
            continue
        if canonical_key in seen:
            errors.append(
                _issue(
                    "duplicate_lineage_key",
                    f"Duplicate canonical lineage key {canonical_key!r}; each "
                    "lineage identity may appear only once across accepted "
                    "Ledger and Chronicle aliases",
                    field,
                )
            )
            continue
        seen.add(canonical_key)


def _validate_provenance_class(
    errors: list[ValidationIssue],
    fact: AggregateFact,
) -> None:
    provenance_class = fact.provenance_class
    if type(provenance_class) is not str or (
        provenance_class not in ALLOWED_PROVENANCE_CLASSES
    ):
        errors.append(
            _issue(
                "malformed_provenance_class",
                f"Unsupported provenance class: {provenance_class!r}",
                "provenance_class",
            )
        )
        return

    if provenance_class == "survey_aggregate":
        if (
            type(fact.survey_instrument) is not str
            or not fact.survey_instrument.strip()
        ):
            errors.append(
                _issue(
                    "missing_survey_instrument",
                    "Survey-aggregate facts need a non-empty survey instrument.",
                    "survey_instrument",
                )
            )
    elif fact.survey_instrument is not None:
        errors.append(
            _issue(
                "misplaced_survey_instrument",
                "survey_instrument is only valid for survey_aggregate facts.",
                "survey_instrument",
            )
        )


def _validate_filters(
    errors: list[ValidationIssue],
    filters: dict[str, Scalar],
) -> None:
    for key, value in filters.items():
        if not key.strip():
            errors.append(
                _issue("malformed_filter", "Filter keys must be nonempty", "filters")
            )
        if isinstance(value, Decimal):
            continue
        if isinstance(value, list | dict | tuple | set):
            errors.append(
                _issue(
                    "malformed_filter",
                    f"Filter {key!r} has non-scalar value",
                    f"filters.{key}",
                )
            )


def _validate_measure_concept_alignment(
    errors: list[ValidationIssue],
    measure: Measure,
) -> None:
    relation = measure.concept_relation
    if relation is not None and relation not in ALLOWED_CONCEPT_RELATIONS:
        errors.append(
            _issue(
                "malformed_concept_relation",
                f"Unsupported concept relation: {relation!r}",
                "measure.concept_relation",
            )
        )
    if measure.source_concept is not None and not measure.source_concept.strip():
        errors.append(
            _issue(
                "missing_field",
                "Source concept must be nonempty when provided",
                "measure.source_concept",
            )
        )
    if measure.source_concept and relation is None:
        errors.append(
            _issue(
                "missing_field",
                "Concept relation is required when source_concept is provided",
                "measure.concept_relation",
            )
        )


def _validate_constraints(
    errors: list[ValidationIssue],
    constraints: tuple[AggregateConstraint, ...],
) -> None:
    for index, constraint in enumerate(constraints):
        if not constraint.variable.strip():
            errors.append(
                _issue(
                    "malformed_constraint",
                    "Constraint variable must be nonempty",
                    f"constraints.{index}.variable",
                )
            )
        if constraint.operator not in ALLOWED_CONSTRAINT_OPERATORS:
            errors.append(
                _issue(
                    "malformed_constraint",
                    f"Unsupported constraint operator: {constraint.operator!r}",
                    f"constraints.{index}.operator",
                )
            )
        value = constraint.value
        if isinstance(value, Decimal):
            continue
        if isinstance(value, list | dict | tuple | set):
            errors.append(
                _issue(
                    "malformed_constraint",
                    f"Constraint {constraint.variable!r} has non-scalar value",
                    f"constraints.{index}.value",
                )
            )


def _validate_period_coverage(
    errors: list[ValidationIssue],
    coverage: PeriodCoverage,
) -> None:
    from datetime import date

    parsed: dict[str, date] = {}
    for field_name in ("start_date", "end_date"):
        raw = getattr(coverage, field_name)
        if raw is None:
            continue
        try:
            parsed[field_name] = date.fromisoformat(str(raw))
        except ValueError:
            errors.append(
                _issue(
                    "malformed_period_coverage",
                    f"Period coverage {field_name} must be an ISO date: {raw!r}",
                    f"period_coverage.{field_name}",
                )
            )
    if (
        "start_date" in parsed
        and "end_date" in parsed
        and parsed["start_date"] > parsed["end_date"]
    ):
        errors.append(
            _issue(
                "malformed_period_coverage",
                "Period coverage start_date must not be after end_date",
                "period_coverage.start_date",
            )
        )
    if coverage.basis is not None and coverage.basis not in ALLOWED_PERIOD_BASES:
        errors.append(
            _issue(
                "malformed_period_coverage",
                f"Unsupported period coverage basis: {coverage.basis!r}",
                "period_coverage.basis",
            )
        )
    if (
        coverage.accounting_basis is not None
        and coverage.accounting_basis not in ALLOWED_ACCOUNTING_BASES
    ):
        errors.append(
            _issue(
                "malformed_period_coverage",
                "Unsupported period coverage accounting basis: "
                f"{coverage.accounting_basis!r}",
                "period_coverage.accounting_basis",
            )
        )


def _validate_provenance(
    errors: list[ValidationIssue],
    source: SourceProvenance,
) -> None:
    _require_nonempty(errors, source.source_name, "source.source_name")
    if not (source.source_table or source.source_file):
        errors.append(
            _issue(
                "missing_provenance",
                "Source provenance must include source_table or source_file",
                "source.source_table",
            )
        )
    _require_nonempty(errors, source.vintage, "source.vintage")
    _require_nonempty(errors, source.extracted_at, "source.extracted_at")
    _require_nonempty(errors, source.extraction_method, "source.extraction_method")


def _has_missing_provenance(fact: AggregateFact) -> bool:
    return any(
        issue.field is not None
        and issue.field.startswith("source.")
        and issue.code in {"missing_field", "missing_provenance"}
        for issue in validate_fact(fact)
    )


def _require_nonempty(
    errors: list[ValidationIssue],
    value: str | None,
    field_name: str,
) -> None:
    if value is None or not str(value).strip():
        errors.append(
            _issue(
                "missing_field",
                f"Required field is missing: {field_name}",
                field_name,
            )
        )


def _issue(code: str, message: str, field_name: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, field=field_name)


def _counter_dict(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _humanize(value: object) -> str:
    text = str(value)
    text = re.sub(r"[_\-.]+", " ", text)
    return " ".join(text.split())


def _format_filters(filters: dict[str, Scalar]) -> str:
    return ", ".join(
        f"{_humanize(key)}={_humanize(value)}"
        for key, value in sorted(filters.items())
        if value is not None
    )


def _source_label(source: SourceProvenance) -> str:
    parts = [part for part in (source.source_name, source.source_table) if part]
    if source.source_file and source.source_file not in parts:
        parts.append(source.source_file)
    if source.vintage:
        parts.append(str(source.vintage))
    return " ".join(parts)
