# ADR: Chronicle is a facts-only store

Status: accepted 2026-07-02 (supersedes the facts-plus-projections split
originally proposed in issue #71)

## Decision

Chronicle stores source-published values only. The boundary is **who asserted
the value**, not level versus projection:

1. Anything a publisher asserted is a fact — including the publisher's own
   projections — with one grain exception added by
   `docs/adr-chronicle-raw-microdata-identity.md`: a publisher-authored value
   at microdata grain (a record, row, column, row value, or cell of a microdata
   release) is content, not a fact; only the values a publisher asserted and
   published over that microdata are facts. "CBO's January 2026 baseline
   projects individual income tax receipts of $X in 2027" is a source-backed
   claim with lineage, exactly like an SOI observation. These facts carry
   `assertion: source_projection`; measured or administered outcomes carry the
   default `assertion: observation`.
2. Anything PolicyEngine computed — an aged, uprated, forecast, or
   reconciled level — is never a Chronicle object. Such values are regenerable
   build artifacts and live in the consumer (Microcosm calibration owns
   aging), implemented as named, versioned models that consume growth-factor
   facts from Chronicle and emit their own lineage.

Instead of projection objects, Chronicle contributes two guarantees:

- **Reference-period semantics.** `PeriodDimension` identifies the period a
  value refers to; `PeriodCoverage` records non-identity provenance (start
  and end dates, basis, the publisher's period label, accounting basis) for
  cases like BE-SILC incomes that reference the year before the survey
  label.
- **Facts-only consumer artifacts.** Chronicle publishes schema-validated fact
  rows with manifest hashes. Consumers own the selection, measurement,
  period-alignment, and model-binding contracts that interpret those rows. The
  facts-only artifact is `policyengine_ledger.consumer_artifact.v2`; the version
  bump makes the removal of v1's embedded profiles and resolution surface an
  explicit incompatible transition.

## Why not facts plus projections in one schema

- **Thesis stays clean.** Thesis resolves forecasts against Chronicle facts as
  official observations. If the store held PolicyEngine-computed
  projections, a forecast could be scored against partly-model output —
  circular. A facts-only store is a model-free resolution substrate.
- **Append-only stays meaningful.** Facts never churn. PolicyEngine-computed
  projections would churn on every CBO update and every aging-model version
  bump, turning an auditable chronicle into a store of volatile derivations.
- **The consumer set does not need it.** Thesis does not want
  PolicyEngine-computed aging; validation comparators (TPC, JCT scores) are
  source-published and therefore facts; the only consumer of
  PolicyEngine-computed aged values is Microcosm calibration, so that is
  where the code belongs (PolicyEngine/microcosm#116).
- **The microcosm#212 lesson is a contract lesson.** The failure was not
  where aging lived; it was that un-aged consumption was silent — SOI
  TY2022/23 dollar levels calibrated exactly at 2024 while simulated 2025
  aggregates ran ~6–10% under current-year projections. The fix is making
  silent period mismatch impossible, which is a consumption-contract
  property, not a projection-object property.

## Consequences

- The `assertion` field enters canonical key payloads only when it is not
  the default, so every pre-existing observation fact keeps byte-identical
  v1 and v2 keys and byte-identical JSONL serialization.
- Source packages declare `assertion` and `period_coverage` per record set;
  values other than `observation` and `source_projection` fail validation
  with an error explaining that PolicyEngine-computed values are not facts.
- Consumer-contract rows always carry `assertion` explicitly, and the
  consumer artifact (`chronicle build-consumer-artifact`) contains only fact
  rows and the manifest hashes needed to verify them. Microcosm packages its
  own selection contracts and builds its target registry without Chronicle
  profiles (issues #166 and #172).
- The retired `policyengine_ledger.target_profile.v1` and
  `policyengine_ledger.resolved_target.v1` schema IDs have no v2 successor in
  issue #143. The Chronicle-side Belgian profile plan in issue #70 is
  superseded; Belgian contracts also live consumer-side.
- Geography vintage translation (microcosm#205) follows the same pattern: a
  declared consumer-side transform over facts, never an edit to them.
