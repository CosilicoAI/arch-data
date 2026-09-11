# Chronicle Agent Rules

Chronicle is a source-backed fact store. It may parse publisher artifacts, normalize
representation, and preserve provenance. Selection and measurement contracts live
in consumers such as Microcosm.

Every fact value must trace to a publisher. The boundary is who asserted the
value, not level versus projection: a publisher's own projection (CBO baseline,
BFP outlook, SSA trustees, TPC/JCT score) is a fact typed
`assertion: source_projection`. PolicyEngine-computed values — aged, uprated,
forecast, or reconciled levels — are never Chronicle facts.

Do not put Microcosm work in Chronicle:

- no cross-source reconciliation
- no aging to a build year
- no imputation
- no support-aware target activation
- no solver-ready target construction
- no target profiles or model-measurement bindings
- no PolicyEngine-computed values stored as facts
- no microdata records, rows, columns, row values, or cells, no facts
  computed from raw microdata by Chronicle or by a PolicyEngine-side consumer
  (Microcosm, PolicyEngine, Thesis), and no licensed or restricted microdata
  bytes in any Chronicle store (registering a release's identity is allowed
  once chronicle#221 lands its access-aware refusals, and until then no
  microdata release may be pointed at any Chronicle command; any value a
  third party asserted and published remains an ordinary fact; see
  `docs/adr-chronicle-raw-microdata-identity.md`)

Chronicle records every fact's publisher reference period. Consumers own and
enforce any declaration that aligns those facts to another period; Chronicle
returns the published level and never an aligned number.

Only approved Chronicle agent roles in `.github/chronicle-agents.yml` should add or
modify source packages or contract schemas. Source-data PRs
need deterministic validation plus the listed Chronicle judge reviews before merge.
