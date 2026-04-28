"""Helpers for source facts and target inputs.

These helpers are for low-assumption structural normalization: unit/scale
standardization and same-source arithmetic with explicit lineage. Modeling
choices such as inflation, aging, target reconciliation, or simulator-specific
allocation belong in Microplex calibration recipes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arch.facts import DerivationStep, SourceFact
from arch.targets import DataSource, GeographicLevel, Jurisdiction, TargetType

Constraint = tuple[str, str, str]


@dataclass(frozen=True)
class TargetBlueprint:
    """A target input ready for a loader to materialize."""

    variable: str
    value: float
    period: int
    target_type: TargetType
    source: DataSource
    jurisdiction: Jurisdiction
    stratum_name: str
    constraints: tuple[Constraint, ...] = ()
    geographic_level: GeographicLevel | None = None
    source_table: str | None = None
    source_url: str | None = None
    notes: str | None = None
    is_preliminary: bool = False
    margin_of_error: float | None = None
    derivation: tuple[DerivationStep, ...] = ()


def scale_value(
    fact: SourceFact,
    factor: float,
    *,
    name: str | None = None,
    unit: str | None = None,
    operation: str = "scale_value",
    note: str | None = None,
) -> SourceFact:
    """Scale a published fact value while recording normalization lineage."""
    parameters: dict[str, Any] = {
        "factor": factor,
        "input_unit": fact.unit,
        "output_unit": unit if unit is not None else fact.unit,
    }
    if note:
        parameters["note"] = note

    return fact.with_step(
        DerivationStep(operation=operation, parameters=parameters),
        name=name,
        value=fact.value * factor,
        unit=unit,
    )


def convert_units(
    fact: SourceFact,
    factor: float,
    to_unit: str,
    *,
    name: str | None = None,
    note: str | None = None,
) -> SourceFact:
    """Normalize a published unit or scale with a multiplicative factor."""
    return scale_value(
        fact,
        factor,
        name=name,
        unit=to_unit,
        operation="convert_units",
        note=note,
    )


def apply_share(
    total_fact: SourceFact,
    share: float,
    *,
    name: str,
    share_name: str | None = None,
    unit: str | None = None,
) -> SourceFact:
    """Materialize a same-source total/share relationship.

    Use this only when the source publishes both the total and the share. Do
    not use it for model-driven allocation or target reconciliation.
    """
    return total_fact.with_step(
        DerivationStep(
            operation="apply_share",
            parameters={
                "share": share,
                "share_name": share_name,
                "total_fact": total_fact.name,
            },
        ),
        name=name,
        value=total_fact.value * share,
        unit=unit,
    )


def as_target(
    fact: SourceFact,
    *,
    variable: str | None = None,
    target_type: TargetType = TargetType.COUNT,
    source: DataSource | str | None = None,
    jurisdiction: Jurisdiction | str | None = None,
    stratum_name: str,
    constraints: tuple[Constraint, ...] = (),
    geographic_level: GeographicLevel | None = None,
    notes: str | None = None,
) -> TargetBlueprint:
    """Convert a source fact into a target input blueprint."""
    target_source = _coerce_data_source(source if source is not None else fact.source)
    target_jurisdiction = _coerce_jurisdiction(
        jurisdiction if jurisdiction is not None else fact.jurisdiction
    )

    derivation_note = format_derivation(fact.derivation)
    if notes and derivation_note:
        notes = f"{notes} Derived via: {derivation_note}."
    elif derivation_note:
        notes = f"Derived via: {derivation_note}."

    return TargetBlueprint(
        variable=variable if variable is not None else fact.name,
        value=fact.value,
        period=fact.period,
        target_type=target_type,
        source=target_source,
        jurisdiction=target_jurisdiction,
        stratum_name=stratum_name,
        constraints=constraints,
        geographic_level=geographic_level,
        source_table=fact.source_table,
        source_url=fact.source_url,
        notes=notes,
        is_preliminary=fact.is_preliminary,
        margin_of_error=fact.margin_of_error,
        derivation=fact.derivation,
    )


def target_kwargs(blueprint: TargetBlueprint, stratum_id: int) -> dict[str, Any]:
    """Return keyword arguments for constructing a target DB row."""
    return {
        "stratum_id": stratum_id,
        "variable": blueprint.variable,
        "period": blueprint.period,
        "value": blueprint.value,
        "target_type": blueprint.target_type,
        "geographic_level": blueprint.geographic_level,
        "source": blueprint.source,
        "source_table": blueprint.source_table,
        "source_url": blueprint.source_url,
        "notes": blueprint.notes,
        "is_preliminary": blueprint.is_preliminary,
        "margin_of_error": blueprint.margin_of_error,
    }


def format_derivation(steps: tuple[DerivationStep, ...]) -> str:
    """Format derivation steps for target notes or diagnostics."""
    parts = []
    for step in steps:
        params = ", ".join(
            f"{key}={value}" for key, value in step.parameters.items()
        )
        parts.append(f"{step.operation}({params})" if params else step.operation)
    return "; ".join(parts)


def _coerce_data_source(source: DataSource | str | None) -> DataSource:
    if source is None:
        raise ValueError("Target source must be supplied by the fact or caller")
    if isinstance(source, DataSource):
        return source
    return DataSource(source)


def _coerce_jurisdiction(jurisdiction: Jurisdiction | str | None) -> Jurisdiction:
    if jurisdiction is None:
        raise ValueError("Target jurisdiction must be supplied by the fact or caller")
    if isinstance(jurisdiction, Jurisdiction):
        return jurisdiction
    return Jurisdiction(jurisdiction)
