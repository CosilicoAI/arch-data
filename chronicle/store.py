"""Load and save Chronicle aggregate facts."""

from __future__ import annotations

import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from chronicle.core import (
    DEFAULT_ASSERTION,
    Aggregation,
    AggregateConstraint,
    EntityDimension,
    GeographyDimension,
    Measure,
    PeriodCoverage,
    PeriodDimension,
    SourceProvenance,
    SourceRecordLayout,
    AggregateFact,
)


def load_facts_jsonl(path: str | Path) -> list[AggregateFact]:
    """Load Chronicle aggregate facts from JSON Lines."""
    fact_path = Path(path)
    facts: list[AggregateFact] = []
    with fact_path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {fact_path}"
                ) from exc
            facts.append(fact_from_mapping(payload))
    return facts


def save_facts_jsonl(facts: list[AggregateFact], path: str | Path) -> None:
    """Save Chronicle aggregate facts to JSON Lines."""
    fact_path = Path(path)
    fact_path.parent.mkdir(parents=True, exist_ok=True)
    with fact_path.open("w") as file:
        for fact in facts:
            file.write(json.dumps(fact_to_mapping(fact), sort_keys=True))
            file.write("\n")


def fact_from_mapping(payload: dict[str, Any]) -> AggregateFact:
    """Build an aggregate fact from a JSON-compatible mapping."""
    value = payload["value"]
    if payload.get("value_type") == "decimal":
        value = Decimal(str(value))
    return AggregateFact(
        value=value,
        period=PeriodDimension(**payload["period"]),
        geography=GeographyDimension(**payload["geography"]),
        entity=EntityDimension(**payload["entity"]),
        measure=Measure(**payload["measure"]),
        aggregation=Aggregation(**payload["aggregation"]),
        source=SourceProvenance(**payload["source"]),
        provenance_class=payload["provenance_class"],
        survey_instrument=payload.get("survey_instrument"),
        filters=dict(payload.get("filters", {})),
        domain=payload.get("domain", "all"),
        label=payload.get("label"),
        source_record_id=payload.get("source_record_id"),
        source_cell_keys=tuple(payload.get("source_cell_keys", ())),
        source_row_keys=tuple(payload.get("source_row_keys", ())),
        constraints=tuple(
            AggregateConstraint(**constraint)
            for constraint in payload.get("constraints", ())
        ),
        layout=(
            SourceRecordLayout(**payload["layout"]) if payload.get("layout") else None
        ),
        assertion=payload.get("assertion", DEFAULT_ASSERTION),
        period_coverage=(
            PeriodCoverage(**payload["period_coverage"])
            if payload.get("period_coverage")
            else None
        ),
        dimension_labels=dict(payload.get("dimension_labels", {})),
        dimension_value_labels={
            dimension_id: dict(value_labels)
            for dimension_id, value_labels in payload.get(
                "dimension_value_labels", {}
            ).items()
        },
    )


def fact_to_mapping(fact: AggregateFact) -> dict[str, Any]:
    """Convert an aggregate fact to a JSON-compatible mapping."""
    payload = asdict(fact)
    if isinstance(fact.value, Decimal):
        payload["value"] = str(fact.value)
        payload["value_type"] = "decimal"
    for label_field in ("dimension_labels", "dimension_value_labels"):
        if not payload[label_field]:
            payload.pop(label_field)
    payload["measure"] = {
        key: value for key, value in payload["measure"].items() if value is not None
    }
    if not fact.constraints:
        payload.pop("constraints", None)
    if not fact.source_row_keys:
        payload.pop("source_row_keys", None)
    if fact.layout is None:
        payload.pop("layout", None)
    else:
        payload["layout"] = {
            key: value for key, value in payload["layout"].items() if value is not None
        }
    if fact.assertion == DEFAULT_ASSERTION:
        payload.pop("assertion", None)
    if fact.survey_instrument is None:
        payload.pop("survey_instrument", None)
    if fact.period_coverage is None:
        payload.pop("period_coverage", None)
    else:
        payload["period_coverage"] = {
            key: value
            for key, value in payload["period_coverage"].items()
            if value is not None
        }
    return payload
