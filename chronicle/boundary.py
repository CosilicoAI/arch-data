"""The microdata boundary, stated once so docs, the judge contract, and tests agree.

``docs/adr-chronicle-raw-microdata-identity.md`` decides that Chronicle registers
raw microdata releases by identity and stores none of their content. These are
the three negative clauses that decision reduces to. ``.github/chronicle-agents.yml``
carries them verbatim in the ``ledger-boundary`` judge verdict, and
``tests/test_chronicle_governance.py`` asserts that it does, so the wording cannot
regress or invert without failing the suite.
"""

from __future__ import annotations

MICRODATA_BOUNDARY_CLAUSES: tuple[str, ...] = (
    "no microdata records, rows, columns, row values, or cells enter any "
    "Chronicle parsed-source surface, registry, derived artifact, or journal",
    "no fact computed from raw microdata by Chronicle or by a PolicyEngine-side "
    "consumer (Microcosm, PolicyEngine, Thesis, or any system that builds from a "
    "Chronicle registration) enters Chronicle, however many intermediate artifacts "
    "stand between them, while a value that a third party asserted and published, "
    "whether the microdata's own publisher or another, is an ordinary fact with "
    "ordinary provenance",
    "no licensed or restricted microdata bytes enter any Chronicle store, and "
    "public microdata bytes enter only with artifact-bound redistribution evidence",
)

__all__ = ["MICRODATA_BOUNDARY_CLAUSES"]
