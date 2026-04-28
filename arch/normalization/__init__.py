"""Low-assumption structural normalization helpers for source facts.

This package handles representation changes such as unit/scale conversion and
same-source arithmetic with explicit lineage. It does not own modeling choices
such as inflation, aging, cross-source reconciliation, or active calibration
target selection.
"""

from arch.facts import DerivationStep, SourceFact
from .targets import (
    TargetBlueprint,
    apply_share,
    as_target,
    convert_units,
    format_derivation,
    scale_value,
    target_kwargs,
)

__all__ = [
    "DerivationStep",
    "SourceFact",
    "TargetBlueprint",
    "apply_share",
    "as_target",
    "convert_units",
    "format_derivation",
    "scale_value",
    "target_kwargs",
]
