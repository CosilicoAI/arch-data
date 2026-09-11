# Chronicle Contribution Governance

Chronicle is the source-of-truth layer for PolicyEngine government-statistics
release facts. Its job is to preserve publisher-backed facts with provenance.
Microcosm turns those facts into active, aged, reconciled calibration targets.

## Boundary

The store is facts-only, and the line is who asserted the value, not level
versus projection. A publisher's own forward-looking estimate — a CBO
baseline, a BFP outlook, an SSA trustees table, a TPC or JCT score — is a
source-backed claim like any other and enters Chronicle as a fact typed
`assertion: source_projection`. A value PolicyEngine computed — an aged,
uprated, forecast, or reconciled level — is never a fact, whatever object it
hides in. This keeps three properties intact: every value in Chronicle traces to
a publisher; facts never churn when models update, so append-only audit stays
meaningful; and Thesis can resolve forecasts against Chronicle observations
without scoring model output against model output.

Chronicle may:

- register raw publisher artifacts and checksums
- register raw microdata releases as source artifacts (publisher, access
  route, vintage, checksum, licence, hash source, verification date) once
  chronicle#221 lands its access-aware refusals, archiving bytes only when
  the release is `public`, its licence is on the redistributable allowlist,
  and the entry carries artifact-bound redistribution evidence naming the
  file; until then no microdata release may be pointed at any Chronicle
  command
- parse source rows and cells
- emit source-backed aggregate facts, including publisher projections typed
  `assertion: source_projection`
- record period-coverage provenance (reference period start/end, basis,
  source period label, accounting basis) for facts whose reference period
  needs disambiguation
- normalize representation, such as units, scales, dates, geography IDs, and
  same-source total/share arithmetic when the publisher defines that relation

Chronicle must not:

- reconcile across sources
- age facts to a build year
- store PolicyEngine-computed values (aged, uprated, forecast, or reconciled
  levels) as facts or in any other store object
- compute aligned values; it publishes the source period and value unchanged
- impute missing values
- parse survey or administrative microdata into records, rows, columns, row
  values, or cells, compute facts from raw microdata (by Chronicle or a
  PolicyEngine-side consumer), or hold licensed or restricted microdata bytes
  in any Chronicle store
- own selection, measurement, period-alignment, or model-binding contracts
- choose a support-aware active target subset
- build solver-ready calibration targets
- invent derived facts whose source is Chronicle itself

## Approval Model

The repository uses `.github/CODEOWNERS` to route all changes through
`@PolicyEngine/core-developers`. Branch protection should require code-owner
review.

Approved agent roles live in `.github/chronicle-agents.yml`. Contributions that
touch source packages or consumer contracts should name the
agent role used and attach the required deterministic checks and judge verdicts.

## Judge Model

Chronicle follows the Axiom pattern: deterministic checks run first, then specialist
reviewers judge the source-data boundary with that evidence. The required judge
types are:

- `ledger-source-fidelity`
- `ledger-contract`
- `ledger-boundary`

The overall verdict fails if any required judge fails. A judge must fail if a
change moves reconciliation, aging, imputation, active target selection, or
solver construction from Microcosm into Chronicle, stores a
PolicyEngine-computed value as a fact, or breaks any clause of the microdata
boundary in `chronicle/boundary.py`, quoted here verbatim: (1) no microdata records, rows, columns, row values, or cells enter any Chronicle parsed-source surface, registry, derived artifact, or journal; (2) no fact computed from raw microdata by Chronicle or by a PolicyEngine-side consumer (Microcosm, PolicyEngine, Thesis, or any system that builds from a Chronicle registration) enters Chronicle, however many intermediate artifacts stand between them, while a value that a third party asserted and published, whether the microdata's own publisher or another, is an ordinary fact with ordinary provenance; (3) no licensed or restricted microdata bytes enter any Chronicle store, and public microdata bytes enter only with artifact-bound redistribution evidence.
