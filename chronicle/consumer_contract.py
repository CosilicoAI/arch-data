"""Chronicle-owned downstream consumer contract exports.

The rows produced here are source-fact contract rows. They expose stable Chronicle
identity and audit fields that downstream adapters can consume without importing
Chronicle internals or depending on source table layout.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from chronicle import artifacts
from chronicle.core import (
    ALLOWED_PROVENANCE_CLASSES,
    DEFAULT_ASSERTION,
    AggregateConstraint,
    AggregateFact,
    build_aggregate_constraints,
    build_fact_key,
)
from chronicle.epoch import EMIT_EPOCH, HASH_DOMAINS, Epoch, hash_domain, schema_id
from chronicle.store import fact_to_mapping

CONSUMER_FACT_SCHEMA_VERSION = schema_id("consumer_fact")


@dataclass(frozen=True)
class ConsumerFactExportReport:
    """Counts from exporting Chronicle facts to consumer-contract JSONL."""

    schema_version: str
    fact_count: int
    output: str

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-serializable report."""
        return asdict(self)


@dataclass(frozen=True)
class ConsumerFactContractIssue:
    """One consumer-contract validation issue."""

    code: str
    message: str
    fact_index: int | None = None
    fact_key: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable issue."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class ConsumerFactContractReport:
    """Validation report for consumer-contract rows."""

    schema_version: str
    fact_count: int
    errors: tuple[ConsumerFactContractIssue, ...]

    @property
    def valid(self) -> bool:
        """Whether every fact can be exported as a valid contract row."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "schema_version": self.schema_version,
            "valid": self.valid,
            "fact_count": self.fact_count,
            "errors": [issue.to_dict() for issue in self.errors],
        }


def build_source_release_key(
    fact: AggregateFact,
    *,
    epoch: Epoch = EMIT_EPOCH,
) -> str:
    """Build a source-release key from fact provenance."""
    return _hash_key(
        hash_domain("source_release", epoch),
        {
            "source_name": fact.source.source_name,
            "source_table": fact.source.source_table,
            "source_file": fact.source.source_file,
            "url": fact.source.url,
            "vintage": fact.source.vintage,
            "source_sha256": fact.source.source_sha256,
            "source_size_bytes": fact.source.source_size_bytes,
            "raw_r2_uri": fact.source.raw_r2_uri,
        },
    )


def build_source_series_key(
    fact: AggregateFact,
    *,
    epoch: Epoch = EMIT_EPOCH,
) -> str:
    """Build a source-series key for a logical publisher table or series."""
    return _hash_key(
        hash_domain("source_series", epoch),
        {
            "source_name": fact.source.source_name,
            "source_table": fact.source.source_table,
            "record_set_spec_id": (
                fact.layout.record_set_spec_id if fact.layout else None
            ),
        },
    )


def build_observed_measure_key(
    fact: AggregateFact,
    *,
    epoch: Epoch = EMIT_EPOCH,
) -> str:
    """Build a source-observed measure key, separate from concept alignment."""
    return _hash_key(
        hash_domain("observed_measure", epoch),
        _observed_measure_payload(fact),
    )


def build_dimension_set_key(
    fact: AggregateFact,
    *,
    epoch: Epoch = EMIT_EPOCH,
) -> str:
    """Build a canonical key for fact dimensions represented as filters."""
    return _hash_key(
        hash_domain("dimension_set", epoch),
        _dimension_set_payload(fact),
    )


def build_universe_constraint_set_key(
    fact: AggregateFact,
    *,
    epoch: Epoch = EMIT_EPOCH,
) -> str:
    """Build a canonical key for semantic universe constraints."""
    return _hash_key(
        hash_domain("universe_constraint_set", epoch),
        _universe_constraint_set_payload(fact),
    )


def build_aggregate_fact_key(
    fact: AggregateFact,
    *,
    epoch: Epoch = EMIT_EPOCH,
) -> str:
    """Build source-specific aggregate fact identity."""
    return _hash_key(
        hash_domain("aggregate_fact", epoch),
        {
            # Nested identities stay in their frozen Ledger form so changing
            # only the outer domain cannot change the canonical payload.
            "source_release_key": build_source_release_key(fact, epoch=Epoch.LEDGER),
            "source_series_key": build_source_series_key(fact, epoch=Epoch.LEDGER),
            "observed_measure_key": build_observed_measure_key(
                fact, epoch=Epoch.LEDGER
            ),
            "aggregation": _aggregation_payload(fact),
            "period": asdict(fact.period),
            "geography": _geography_payload(fact),
            "entity": asdict(fact.entity),
            "dimension_set_key": build_dimension_set_key(fact, epoch=Epoch.LEDGER),
            "universe_constraint_set_key": build_universe_constraint_set_key(
                fact, epoch=Epoch.LEDGER
            ),
            "assertion": _assertion_key_value(fact),
        },
    )


def build_semantic_fact_key(
    fact: AggregateFact,
    *,
    epoch: Epoch = EMIT_EPOCH,
) -> str:
    """Build source-agnostic fact identity for downstream reconciliation."""
    return _hash_key(
        hash_domain("semantic_fact", epoch),
        {
            "canonical_measure": {
                "concept": fact.measure.concept,
                "unit": fact.measure.unit,
            },
            "aggregation": _aggregation_payload(fact),
            "period": asdict(fact.period),
            "geography": _geography_payload(fact),
            "entity": asdict(fact.entity),
            "universe_constraint_set_key": build_universe_constraint_set_key(
                fact, epoch=Epoch.LEDGER
            ),
            "assertion": _assertion_key_value(fact),
        },
    )


def _assertion_key_value(fact: AggregateFact) -> str | None:
    """Return the assertion for key payloads, omitting the default.

    ``_hash_key`` drops ``None`` values, so observation facts keep the exact
    keys they had before ``assertion`` existed while publisher projections
    get distinct identities.
    """
    if fact.assertion == DEFAULT_ASSERTION:
        return None
    return fact.assertion


def build_concept_alignment_key(
    fact: AggregateFact,
    *,
    epoch: Epoch = EMIT_EPOCH,
) -> str | None:
    """Build a concept-alignment key when source alignment metadata exists."""
    if not (
        fact.measure.source_concept
        or fact.measure.concept_relation
        or fact.measure.concept_authority
        or fact.measure.legal_vintage
    ):
        return None
    return _hash_key(
        hash_domain("concept_alignment", epoch),
        {
            "observed_measure_key": build_observed_measure_key(
                fact, epoch=Epoch.LEDGER
            ),
            "canonical_concept": fact.measure.concept,
            "relation": fact.measure.concept_relation,
            "authority": fact.measure.concept_authority,
            "legal_vintage": fact.measure.legal_vintage,
        },
    )


def consumer_fact_rows(
    facts: list[AggregateFact],
    *,
    emit_epoch: Epoch = EMIT_EPOCH,
) -> list[dict[str, Any]]:
    """Build JSON-compatible consumer-contract rows for facts."""
    contract_report = validate_consumer_fact_contract(facts)
    if not contract_report.valid:
        raise ValueError("Cannot export invalid Chronicle consumer-contract facts.")
    return [_consumer_fact_row(fact, emit_epoch=emit_epoch) for fact in facts]


def validate_consumer_fact_contract(
    facts: list[AggregateFact],
) -> ConsumerFactContractReport:
    """Validate that facts can be emitted as consumer-contract rows."""
    errors: list[ConsumerFactContractIssue] = []
    for index, fact in enumerate(facts):
        fact_key = build_fact_key(fact)
        for lineage_field, domain_name in (
            ("source_cell_keys", "source_cell"),
            ("source_row_keys", "source_row"),
        ):
            pair = HASH_DOMAINS[domain_name]
            for key in getattr(fact, lineage_field):
                try:
                    pair.infer_key_epoch(key)
                except ValueError as error:
                    errors.append(
                        ConsumerFactContractIssue(
                            code="malformed_lineage_key",
                            message=str(error),
                            fact_index=index,
                            fact_key=fact_key,
                            field=f"lineage/{lineage_field}",
                        )
                    )
        if any(
            issue.code == "malformed_lineage_key" and issue.fact_index == index
            for issue in errors
        ):
            # The remaining checks build the row, which canonicalizes lineage
            # keys and would raise on the same malformed key.
            continue
        filter_constraints = _filter_derived_constraints(fact)
        source_filter_variables = _source_filter_variables(fact, filter_constraints)
        if filter_constraints and not fact.constraints:
            errors.append(
                ConsumerFactContractIssue(
                    code="implicit_constraints_from_filters",
                    message=(
                        "Consumer-contract facts must carry semantic "
                        "constraints explicitly; source-layout filters are "
                        "metadata only."
                    ),
                    fact_index=index,
                    fact_key=fact_key,
                    field="constraints",
                )
            )
        elif filter_constraints:
            expected_constraint_payloads = Counter(
                _constraint_compare_payload(
                    _canonical_filter_constraint(
                        fact,
                        constraint,
                        source_filter_variables=source_filter_variables,
                    )
                )
                for constraint in filter_constraints
            )
            explicit_constraint_payloads = Counter(
                _constraint_compare_payload(constraint)
                for constraint in fact.constraints
            )
            for (
                constraint_payload,
                expected_count,
            ) in expected_constraint_payloads.items():
                if explicit_constraint_payloads[constraint_payload] >= expected_count:
                    continue
                errors.append(
                    ConsumerFactContractIssue(
                        code="constraint_filter_mismatch",
                        message=(
                            "Source-layout filters imply a semantic constraint "
                            "that is not present in explicit constraints."
                        ),
                        fact_index=index,
                        fact_key=fact_key,
                        field="constraints",
                    )
                )

        for constraint in fact.constraints:
            if _is_source_specific_variable(fact, constraint.variable):
                errors.append(
                    ConsumerFactContractIssue(
                        code="source_specific_constraint_variable",
                        message=(
                            "Consumer-contract universe constraints must use "
                            "canonical variables; source-specific variables "
                            "belong in observed-measure metadata."
                        ),
                        fact_index=index,
                        fact_key=fact_key,
                        field="constraints",
                    )
                )
                continue
            canonical_constraint = _canonical_filter_constraint(
                fact,
                constraint,
                source_filter_variables=source_filter_variables,
            )
            if canonical_constraint.variable == constraint.variable:
                continue
            errors.append(
                ConsumerFactContractIssue(
                    code="source_specific_constraint_variable",
                    message=(
                        "Consumer-contract universe constraints must use the "
                        "canonical constraint variable when one is known."
                    ),
                    fact_index=index,
                    fact_key=fact_key,
                    field="constraints",
                )
            )

        for field_name in (
            "source_file",
            "source_sha256",
            "source_size_bytes",
            "raw_r2_uri",
        ):
            value = getattr(fact.source, field_name)
            if value in (None, ""):
                errors.append(
                    ConsumerFactContractIssue(
                        code="missing_contract_provenance",
                        message=(
                            "Consumer-contract source provenance must include "
                            f"{field_name}."
                        ),
                        fact_index=index,
                        fact_key=fact_key,
                        field=f"source.{field_name}",
                    )
                )

        derived_source_issue = _derived_source_provenance_issue(fact)
        if derived_source_issue is not None:
            errors.append(
                ConsumerFactContractIssue(
                    code="derived_fact_provenance",
                    message=derived_source_issue,
                    fact_index=index,
                    fact_key=fact_key,
                    field="source",
                )
            )

        provenance_issue = _fact_provenance_class_issue(fact)
        if provenance_issue is not None:
            code, message, field = provenance_issue
            errors.append(
                ConsumerFactContractIssue(
                    code=code,
                    message=message,
                    fact_index=index,
                    fact_key=fact_key,
                    field=field,
                )
            )

        if not fact.source_record_id:
            errors.append(
                ConsumerFactContractIssue(
                    code="missing_contract_lineage",
                    message="Consumer-contract facts must carry source_record_id.",
                    fact_index=index,
                    fact_key=fact_key,
                    field="source_record_id",
                )
            )
        if not fact.source_cell_keys:
            errors.append(
                ConsumerFactContractIssue(
                    code="missing_contract_lineage",
                    message="Consumer-contract facts must carry source_cell_keys.",
                    fact_index=index,
                    fact_key=fact_key,
                    field="source_cell_keys",
                )
            )

        row = _consumer_fact_row(fact)
        for field_name in _CONSUMER_REQUIRED_FIELDS:
            if field_name not in row:
                errors.append(
                    ConsumerFactContractIssue(
                        code="missing_contract_field",
                        message=(
                            "Consumer-contract row is missing required field "
                            f"{field_name}."
                        ),
                        fact_index=index,
                        fact_key=fact_key,
                        field=field_name,
                    )
                )

    return ConsumerFactContractReport(
        schema_version=CONSUMER_FACT_SCHEMA_VERSION,
        fact_count=len(facts),
        errors=tuple(errors),
    )


def _r2_uri_parts(uri: str) -> tuple[str, str]:
    """Split an ``r2://bucket/key`` URI into its bucket and key."""
    if uri[: len("r2://")].lower() != "r2://":
        return "", ""
    bucket, _, key = uri[len("r2://") :].partition("/")
    return bucket, key


# The marker a downstream target row carries in its source_record_id. It moves
# with the rename window: producers write `ledger_derived` today and
# `chronicle_derived` once they migrate (PolicyEngine/chronicle#143, mechanism
# 3), so the boundary has to reject both spellings identically or the guard
# stops firing the moment a producer renames.
DERIVED_SOURCE_RECORD_SUFFIXES = frozenset({"ledger_derived", "chronicle_derived"})


def _is_derived_source_record_id(source_record_id: str) -> bool:
    """Whether a source_record_id marks a downstream derived target row."""
    _, separator, suffix = source_record_id.rpartition(".")
    return bool(separator) and suffix in DERIVED_SOURCE_RECORD_SUFFIXES


def _points_at_derived(bucket: str, key: str) -> bool:
    """Use the same derived routes publication is permitted to address."""
    return artifacts.is_derived_r2_route(bucket, key)


def _derived_source_provenance_issue(fact: AggregateFact) -> str | None:
    """Return a boundary error if a fact is a downstream target derivation."""
    source = fact.source
    source_name = (source.source_name or "").lower()
    source_file = source.source_file or ""
    raw_r2_bucket = source.raw_r2_bucket or ""
    raw_r2_key = source.raw_r2_key or ""
    raw_r2_uri = source.raw_r2_uri or ""
    source_record_id = fact.source_record_id or ""

    if source_name in {
        "chronicle",
        "policyengine_chronicle",
        "ledger",
        "policyengine_ledger",
    }:
        return (
            "Chronicle consumer facts must cite publisher sources, not Chronicle "
            "itself. Target construction, aging, and reconciliation belong in "
            "Microcosm."
        )
    if source_file[: len("r2://")].lower() == "r2://":
        source_file_bucket, source_file_key = _r2_uri_parts(source_file)
    else:
        source_file_bucket, bucket_separator, source_file_key = source_file.partition(
            ":"
        )
        if not bucket_separator:
            source_file_bucket = source_file_key = ""
    if _points_at_derived(source_file_bucket, source_file_key):
        return (
            "Chronicle consumer facts must cite raw publisher artifacts. Derived "
            "target-construction artifacts belong in Microcosm."
        )
    if (
        _points_at_derived(raw_r2_bucket, raw_r2_key)
        or _points_at_derived(*_r2_uri_parts(raw_r2_uri))
        or _points_at_derived(*_r2_uri_parts(source.url or ""))
    ):
        return (
            "Chronicle consumer facts must point at raw source artifacts, not "
            "derived build artifacts."
        )
    if _is_derived_source_record_id(source_record_id):
        return (
            "Chronicle source_record_id must identify a publisher-backed row, not "
            "a downstream derived target row."
        )
    return None


def _fact_provenance_class_issue(
    fact: AggregateFact,
) -> tuple[str, str, str] | None:
    if type(fact.provenance_class) is not str or (
        fact.provenance_class not in ALLOWED_PROVENANCE_CLASSES
    ):
        return (
            "malformed_provenance_class",
            f"Unsupported provenance class: {fact.provenance_class!r}.",
            "provenance_class",
        )
    if fact.provenance_class == "survey_aggregate":
        if (
            type(fact.survey_instrument) is not str
            or not fact.survey_instrument.strip()
        ):
            return (
                "missing_survey_instrument",
                "Survey aggregates need a non-empty survey instrument.",
                "survey_instrument",
            )
    elif fact.survey_instrument is not None:
        return (
            "misplaced_survey_instrument",
            "survey_instrument is only valid for survey_aggregate facts.",
            "survey_instrument",
        )
    return None


def _filter_derived_constraints(
    fact: AggregateFact,
) -> tuple[AggregateConstraint, ...]:
    """Return the legacy filter-derived constraint fallback for comparison."""
    return build_aggregate_constraints(replace(fact, constraints=()))


def _source_filter_variables(
    fact: AggregateFact,
    filter_constraints: tuple[AggregateConstraint, ...],
) -> frozenset[str]:
    """Return source-specific variables that Chronicle can map from filters."""
    return frozenset(
        constraint.variable
        for constraint in filter_constraints
        if _is_source_specific_variable(fact, constraint.variable)
    )


def _canonical_filter_constraint(
    fact: AggregateFact,
    constraint: AggregateConstraint,
    *,
    source_filter_variables: frozenset[str],
) -> AggregateConstraint:
    """Canonicalize a source-derived filter constraint when metadata permits."""
    variable = _canonical_filter_variable(
        fact,
        constraint,
        source_filter_variables=source_filter_variables,
    )
    if variable == constraint.variable:
        return constraint
    return replace(constraint, variable=variable)


def _canonical_filter_variable(
    fact: AggregateFact,
    constraint: AggregateConstraint,
    *,
    source_filter_variables: frozenset[str],
) -> str:
    """Resolve source-layout filter variables to canonical constraint variables."""
    if (
        constraint.variable in source_filter_variables
        and fact.layout
        and fact.layout.groupby_dimension
        and _concept_leaf(constraint.variable)
        == _concept_leaf(fact.layout.groupby_dimension)
    ):
        return fact.layout.groupby_dimension
    return constraint.variable


def _is_source_specific_variable(fact: AggregateFact, variable: str) -> bool:
    """Return whether a constraint variable uses the current source namespace."""
    source_name = fact.source.source_name
    return bool(
        source_name
        and (
            variable.startswith(f"{source_name}.")
            or variable.startswith(f"{source_name}:")
        )
    )


def _concept_leaf(concept: str) -> str:
    """Return the local concept name for source-to-canonical comparisons."""
    return (
        concept.rsplit("#", maxsplit=1)[-1]
        .rsplit(
            "/",
            maxsplit=1,
        )[-1]
        .rsplit(".", maxsplit=1)[-1]
        .rsplit(":", maxsplit=1)[-1]
    )


def _constraint_compare_payload(constraint: AggregateConstraint) -> str:
    """Return a stable comparable representation of an aggregate constraint."""
    payload = _constraint_payload(constraint)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def consumer_fact_row(
    fact: AggregateFact,
    *,
    emit_epoch: Epoch = EMIT_EPOCH,
) -> dict[str, Any]:
    """Build one consumer-contract row from an aggregate fact."""
    return consumer_fact_rows([fact], emit_epoch=emit_epoch)[0]


def _consumer_fact_row(
    fact: AggregateFact,
    *,
    emit_epoch: Epoch = EMIT_EPOCH,
) -> dict[str, Any]:
    """Build one consumer-contract row without recursive validation."""
    aggregate_fact_key = build_aggregate_fact_key(fact, epoch=emit_epoch)
    semantic_fact_key = build_semantic_fact_key(fact, epoch=emit_epoch)
    concept_alignment_key = build_concept_alignment_key(fact, epoch=emit_epoch)

    row: dict[str, Any] = {
        "schema_version": schema_id("consumer_fact", emit_epoch),
        "aggregate_fact_key": aggregate_fact_key,
        "semantic_fact_key": semantic_fact_key,
        "legacy_fact_key": build_fact_key(fact, epoch=emit_epoch),
        "source_release_key": build_source_release_key(fact, epoch=emit_epoch),
        "source_series_key": build_source_series_key(fact, epoch=emit_epoch),
        "observed_measure_key": build_observed_measure_key(fact, epoch=emit_epoch),
        "dimension_set_key": build_dimension_set_key(fact, epoch=emit_epoch),
        "universe_constraint_set_key": build_universe_constraint_set_key(
            fact, epoch=emit_epoch
        ),
        "value": _json_value(fact.value),
        "value_type": _value_type(fact.value),
        "assertion": fact.assertion,
        "provenance_class": fact.provenance_class,
        "survey_instrument": fact.survey_instrument,
        "period": asdict(fact.period),
        "geography": _geography_payload(fact),
        "entity": asdict(fact.entity),
        "aggregation": _aggregation_payload(fact),
        "observed_measure": _observed_measure_payload(fact),
        "dimensions": _dimension_set_payload(fact),
        "universe_constraints": _universe_constraint_set_payload(fact),
        "source": _clean(fact_to_mapping(fact)["source"]),
        "lineage": {
            "source_record_id": fact.source_record_id,
            "source_cell_keys": _lineage_keys_for_epoch(
                "source_cell", fact.source_cell_keys, emit_epoch
            ),
            "source_row_keys": _lineage_keys_for_epoch(
                "source_row", fact.source_row_keys, emit_epoch
            ),
        },
        "layout": _clean(asdict(fact.layout)) if fact.layout else {},
        "label": fact.label,
    }
    if fact.period_coverage is not None:
        row["period_coverage"] = _clean(asdict(fact.period_coverage))
    if concept_alignment_key:
        row["concept_alignment"] = {
            "concept_alignment_key": concept_alignment_key,
            "source_concept": fact.measure.source_concept,
            "canonical_concept": fact.measure.concept,
            "relation": fact.measure.concept_relation,
            "authority": fact.measure.concept_authority,
            "evidence_url": fact.measure.concept_evidence_url,
            "evidence_notes": fact.measure.concept_evidence_notes,
            "legal_vintage": fact.measure.legal_vintage,
        }
    return _clean_consumer_row(row)


def _require_ledger_emit(emit_epoch: Epoch | str, boundary: str) -> Epoch:
    """Refuse a non-Ledger emit at an artifact boundary; return the epoch.

    Mechanism 1 of the rename accepts both epochs on read and emits unchanged:
    the only packaged, sha-pinned consumer-fact schema is the Ledger one, so
    every artifact boundary emits Ledger-named rows (the artifact builder
    canonicalizes whatever epoch it read) and refuses a Chronicle emit until a
    successor schema is pinned by a separate, consumer-gated cutover.

    ``emit_epoch`` may be the enum member or its string value; anything else is
    refused with :class:`ValueError` naming the boundary. Callers must check
    this before touching the filesystem so a refused call leaves existing
    outputs intact.
    """

    try:
        epoch = Epoch(emit_epoch)
    except ValueError as error:
        raise ValueError(
            f"{boundary}: unknown emit epoch {emit_epoch!r}; expected "
            f"{Epoch.LEDGER.value!r} or {Epoch.CHRONICLE.value!r}"
        ) from error
    if epoch != Epoch.LEDGER:
        raise ValueError(
            f"{boundary}: emitting {epoch.value!r}-epoch identifiers needs a "
            "packaged successor consumer-fact schema; until one is pinned, "
            "artifacts are emitted Ledger-named (mechanism 1: emit unchanged)."
        )
    return epoch


def write_consumer_facts_jsonl(
    facts: list[AggregateFact],
    path: str | Path,
    *,
    emit_epoch: Epoch | str = EMIT_EPOCH,
) -> ConsumerFactExportReport:
    """Write consumer-contract fact rows to JSON Lines."""
    emit_epoch = _require_ledger_emit(emit_epoch, "write_consumer_facts_jsonl")
    rows = consumer_fact_rows(facts, emit_epoch=emit_epoch)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")
    return ConsumerFactExportReport(
        schema_version=schema_id("consumer_fact", emit_epoch),
        fact_count=len(rows),
        output=str(output_path),
    )


def _lineage_keys_for_epoch(
    domain_name: str,
    keys: tuple[str, ...],
    epoch: Epoch,
) -> list[str]:
    """Return accepted lineage identities under the requested emit epoch."""

    pair = HASH_DOMAINS[domain_name]
    return list(dict.fromkeys(pair.key_for_epoch(key, epoch) for key in keys))


def _hash_key(namespace: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(
        _clean(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{namespace}:{digest}"


def _observed_measure_payload(fact: AggregateFact) -> dict[str, Any]:
    return {
        "source_name": fact.source.source_name,
        "source_table": fact.source.source_table,
        "source_measure_id": _source_measure_id(fact),
        "source_concept": fact.measure.source_concept or fact.measure.concept,
        "unit": fact.measure.unit,
    }


def _source_measure_id(fact: AggregateFact) -> str:
    if fact.layout and fact.layout.measure_id:
        return fact.layout.measure_id
    return fact.measure.source_concept or fact.measure.concept


def _dimension_set_payload(fact: AggregateFact) -> dict[str, Any]:
    return {
        key: _json_value(value)
        for key, value in sorted(fact.filters.items())
        if value is not None
    }


def _universe_constraint_set_payload(fact: AggregateFact) -> dict[str, Any]:
    constraints = [_constraint_payload(constraint) for constraint in fact.constraints]
    return {
        "domain": fact.domain,
        "constraints": sorted(
            constraints,
            key=lambda constraint: json.dumps(
                constraint,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    }


def _constraint_payload(constraint: AggregateConstraint) -> dict[str, Any]:
    return _clean(
        {
            "variable": constraint.variable,
            "operator": constraint.operator,
            "value": _json_value(constraint.value),
            "unit": constraint.unit,
            "role": constraint.role,
        }
    )


def _geography_payload(fact: AggregateFact) -> dict[str, Any]:
    return _clean(
        {
            "level": fact.geography.level,
            "id": fact.geography.id,
            "vintage": fact.geography.vintage,
        }
    )


def _aggregation_payload(fact: AggregateFact) -> dict[str, Any]:
    return _clean(asdict(fact.aggregation))


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Decimal):
        return "decimal"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return value


_CONSUMER_REQUIRED_FIELDS = (
    "schema_version",
    "aggregate_fact_key",
    "semantic_fact_key",
    "legacy_fact_key",
    "source_release_key",
    "source_series_key",
    "observed_measure_key",
    "dimension_set_key",
    "universe_constraint_set_key",
    "value",
    "value_type",
    "assertion",
    "provenance_class",
    "period",
    "geography",
    "entity",
    "aggregation",
    "observed_measure",
    "dimensions",
    "universe_constraints",
    "source",
    "lineage",
)


def _clean_consumer_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = _clean(row)
    cleaned["dimensions"] = _clean(row["dimensions"])
    cleaned["lineage"] = {
        "source_record_id": row["lineage"]["source_record_id"],
        "source_cell_keys": row["lineage"]["source_cell_keys"],
        "source_row_keys": row["lineage"]["source_row_keys"],
    }
    cleaned["layout"] = _clean(row["layout"])
    return cleaned


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in sorted(value.items())
            if (cleaned := _clean(item)) not in (None, {}, [])
        }
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    return value
