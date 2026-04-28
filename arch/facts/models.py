"""Models for source-backed facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from arch.sources.models import SourceFile
from db.schema import DataSource, Jurisdiction


@dataclass(frozen=True)
class DerivationStep:
    """One auditable structural normalization step applied to a source fact."""

    operation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceFact:
    """A source-backed fact before target-input materialization."""

    name: str
    value: float
    period: int
    unit: str | None = None
    source: DataSource | str | None = None
    jurisdiction: Jurisdiction | str | None = None
    dimensions: Mapping[str, str] = field(default_factory=dict)
    source_table: str | None = None
    source_url: str | None = None
    source_file: SourceFile | None = None
    is_preliminary: bool = False
    margin_of_error: float | None = None
    derivation: tuple[DerivationStep, ...] = ()

    def with_step(
        self,
        step: DerivationStep,
        *,
        name: str | None = None,
        value: float | None = None,
        period: int | None = None,
        unit: str | None = None,
    ) -> "SourceFact":
        """Return a fact updated by one derivation step."""
        return replace(
            self,
            name=name if name is not None else self.name,
            value=value if value is not None else self.value,
            period=period if period is not None else self.period,
            unit=unit if unit is not None else self.unit,
            derivation=self.derivation + (step,),
        )
