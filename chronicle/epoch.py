"""Epoch registry for frozen Ledger and successor Chronicle identifiers.

The migration is additive: readers accept both identifiers in each pair, while
emitters use :data:`EMIT_EPOCH` and therefore remain Ledger-named until a later
cutover changes that single default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class Epoch(StrEnum):
    """Fact-identity naming epochs."""

    LEDGER = "ledger"
    CHRONICLE = "chronicle"


# Changing this default is a separate, consumer-gated migration step.
EMIT_EPOCH = Epoch.LEDGER


@dataclass(frozen=True)
class EpochPair:
    """A frozen Ledger identifier and its Chronicle-era successor."""

    ledger: str
    chronicle: str

    @property
    def accepted(self) -> tuple[str, str]:
        return (self.ledger, self.chronicle)

    def for_epoch(self, epoch: Epoch = EMIT_EPOCH) -> str:
        if epoch == Epoch.LEDGER:
            return self.ledger
        if epoch == Epoch.CHRONICLE:
            return self.chronicle
        raise ValueError(f"unknown emit epoch {epoch!r}; expected ledger or chronicle")

    def infer_identifier_epoch(self, identifier: str) -> Epoch:
        if identifier == self.ledger:
            return Epoch.LEDGER
        if identifier == self.chronicle:
            return Epoch.CHRONICLE
        raise ValueError(
            f"unsupported identifier {identifier!r}; accepted forms are "
            f"{self.ledger!r} and {self.chronicle!r}"
        )

    def infer_key_epoch(self, key: str) -> Epoch:
        if not isinstance(key, str):
            raise ValueError(
                f"unsupported key {key!r}; keys are strings with prefix "
                f"{self.ledger!r} or {self.chronicle!r}"
            )
        prefix, separator, _digest = key.partition(":")
        if not separator:
            raise ValueError(
                f"unsupported key {key!r}; accepted prefixes are "
                f"{self.ledger!r} and {self.chronicle!r}"
            )
        try:
            return self.infer_identifier_epoch(prefix)
        except ValueError as error:
            raise ValueError(
                f"unsupported key {key!r}; accepted prefixes are "
                f"{self.ledger!r} and {self.chronicle!r}"
            ) from error

    def key_for_epoch(self, key: str, epoch: Epoch) -> str:
        """Return *key* under *epoch* without changing its payload digest."""

        self.infer_key_epoch(key)
        _prefix, _separator, digest = key.partition(":")
        return f"{self.for_epoch(epoch)}:{digest}"


HASH_DOMAINS: Mapping[str, EpochPair] = MappingProxyType(
    {
        "source_release": EpochPair(
            "ledger.source_release.v2", "chronicle.source_release.v3"
        ),
        "source_series": EpochPair(
            "ledger.source_series.v2", "chronicle.source_series.v3"
        ),
        "observed_measure": EpochPair(
            "ledger.observed_measure.v2", "chronicle.observed_measure.v3"
        ),
        "dimension_set": EpochPair(
            "ledger.dimension_set.v2", "chronicle.dimension_set.v3"
        ),
        "universe_constraint_set": EpochPair(
            "ledger.universe_constraint_set.v2",
            "chronicle.universe_constraint_set.v3",
        ),
        "aggregate_fact": EpochPair(
            "ledger.aggregate_fact.v2", "chronicle.aggregate_fact.v3"
        ),
        "semantic_fact": EpochPair(
            "ledger.semantic_fact.v2", "chronicle.semantic_fact.v3"
        ),
        "concept_alignment": EpochPair(
            "ledger.concept_alignment.v2", "chronicle.concept_alignment.v3"
        ),
        "fact": EpochPair("ledger.fact.v1", "chronicle.fact.v2"),
        "source_cell": EpochPair("ledger.source_cell.v1", "chronicle.source_cell.v2"),
        "source_row": EpochPair("ledger.source_row.v1", "chronicle.source_row.v2"),
        "source_column": EpochPair(
            "ledger.source_column.v1", "chronicle.source_column.v2"
        ),
        "source_row_value": EpochPair(
            "ledger.source_row_value.v1", "chronicle.source_row_value.v2"
        ),
        "build": EpochPair("ledger.build.v1", "chronicle.build.v2"),
        "build_artifact": EpochPair(
            "ledger.build_artifact.v1", "chronicle.build_artifact.v2"
        ),
    }
)


SCHEMA_IDS: Mapping[str, EpochPair] = MappingProxyType(
    {
        "bundle": EpochPair("ledger.bundle.v1", "chronicle.bundle.v2"),
        "bundle_coverage": EpochPair(
            "ledger.bundle_coverage.v1", "chronicle.bundle_coverage.v2"
        ),
        "bundle_sources": EpochPair(
            "ledger.bundle_sources.v1", "chronicle.bundle_sources.v2"
        ),
        "consumer_fact": EpochPair(
            "ledger.consumer_fact.v1", "chronicle.consumer_fact.v2"
        ),
        "relational": EpochPair("ledger.relational.v1", "chronicle.relational.v2"),
        "source_package": EpochPair(
            "ledger.source_package.v1", "chronicle.source_package.v2"
        ),
        "offline_fetch_manifest": EpochPair(
            "ledger.offline_fetch_manifest.v1",
            "chronicle.offline_fetch_manifest.v2",
        ),
        "fetch_manifest": EpochPair(
            "ledger.fetch_manifest.v1", "chronicle.fetch_manifest.v2"
        ),
        # Facts-only v2 is the live artifact contract. The retired target-profile
        # and resolved-target v1 contracts intentionally have no successors.
        "consumer_artifact": EpochPair(
            "policyengine_ledger.consumer_artifact.v2",
            "policyengine_chronicle.consumer_artifact.v3",
        ),
        "approved_agents": EpochPair(
            "policyengine_ledger.approved_agents.v1",
            "policyengine_chronicle.approved_agents.v2",
        ),
    }
)


# The consumer-fact row contract Chronicle emits. consumer_fact v2 (chronicle#261)
# is v1 plus optional dimension and value labels, so a v1 row restamped with the
# v2 id is a valid v2 row; its id is the pair's Chronicle-era name. The row
# contract and the hash-key domains move independently: EMIT_EPOCH still keeps
# every key Ledger-named.
CONSUMER_FACT_EMIT_SCHEMA_VERSION = SCHEMA_IDS["consumer_fact"].chronicle


def hash_domain(name: str, epoch: Epoch = EMIT_EPOCH) -> str:
    """Return the hash domain registered for *name* and *epoch*."""

    return HASH_DOMAINS[name].for_epoch(epoch)


def schema_id(name: str, epoch: Epoch = EMIT_EPOCH) -> str:
    """Return the schema id registered for *name* and *epoch*."""

    return SCHEMA_IDS[name].for_epoch(epoch)


def canonicalize_key(name: str, key: str) -> str:
    """Normalize either accepted key form to its immutable Ledger form."""

    return HASH_DOMAINS[name].key_for_epoch(key, Epoch.LEDGER)
