"""Chronicle source-data foundation.

Chronicle owns government-statistics releases: source artifacts (and, once
chronicle#221 lands, registrations of raw microdata releases plus custody of
public-use bytes with redistribution evidence), source-backed facts,
constraints, and provenance. Microdata content (records, rows, columns, row
values, cells, and facts computed from raw microdata by Chronicle or a
PolicyEngine-side consumer), licensed or restricted microdata bytes, selection
contracts, source reconciliation, aging, imputation, target activation, and
calibration belong in downstream systems such as Microcosm.
"""

__all__ = [
    "bundle",
    "client",
    "concepts",
    "consumer_contract",
    "core",
    "database",
    "facts",
    "harness",
    "jurisdictions",
    "mirror",
    "normalization",
    "source_package",
    "store",
    "sources",
    "suite",
    "targets",
]
