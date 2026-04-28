"""Models for external source lineage.

These models let Arch facts retain source/file provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from db.schema import DataSource, Jurisdiction


@dataclass(frozen=True)
class SourceReference:
    """An external source registry entry."""

    source: DataSource | str
    institution: str
    dataset: str
    jurisdiction: Jurisdiction | str | None = None
    url: str | None = None
    update_frequency: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceFile:
    """A versioned source file or artifact reference."""

    source: SourceReference
    r2_key: str
    fetched_at: str | None = None
    checksum: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
