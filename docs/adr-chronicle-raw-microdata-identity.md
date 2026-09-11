# ADR: Raw microdata in Chronicle is identity, not content

Status: accepted 2026-09-02. Enforcement lands in two steps and this ADR's
guarantees are prospective until they merge: chronicle#221 (PR #227: manifest
classification, the access-aware refusals in `validate-package`,
`fetch-artifact`, `publish-raw`, and the source-package byte reader, the
redistribution allowlist, artifact-bound licence evidence, untracked staging
for public microdata bytes) and chronicle#238 (asserting principal and
root-artifact lineage on fact provenance). Until #227 merges, no microdata
release may be pointed at any Chronicle command: today they materialize,
upload, and parse every input. Until #238 merges, the derived-fact prohibition
is a review obligation, not a validator.

## Decision

Chronicle registers every raw microdata release its consumers build from, and
stores none of their content. "Content" means any parsed representation:
microdata records, rows, columns, row values, or cells, and any fact computed
from them by Chronicle or by a PolicyEngine-side consumer (Microcosm,
PolicyEngine, Thesis, or any system that builds from a Chronicle registration).
A value that a third party asserted and published, whether the microdata's own
publisher or another, is an ordinary fact. Custody of a public-use file's bytes
under artifact-bound redistribution evidence is not content; custody of
licensed or restricted bytes is never taken at all.

1. **Classification is explicit.** Every manifest created or modified after
   this ADR declares `kind`, either `publisher_table` or `microdata_release`.
   `validate-package`, `fetch-artifact`, `publish-raw`, and the source-package
   byte reader refuse a new or modified manifest without it, at every entry
   point. The manifests that exist at #227's merge commit are grandfathered as
   publisher tables by an explicit frozen list checked into the repository; a
   kindless manifest outside that list is an error, never a publisher table by
   default. `microdata_release` is what lets every parser refuse the file, so a
   public microdata release can never be mistaken for a public aggregate
   workbook.
2. **Registration.** A microdata release (CPS ASEC, ACS PUMS, SCF, SIPP, FRS,
   BE-SILC, the IRS PUF, and their successors in every jurisdiction) is a
   source artifact registered by identity: publisher, source URL or access
   route, vintage, SHA-256, size, licence, an access class from a closed set
   (`public`, `licensed`, `restricted`), and the attestation fields in
   decision 5. One package per publisher release, its files listed under the
   vintage year; a registration is identified by
   `{source_id, package_id, year, sha256, filename}`. Keys follow the
   convention in `docs/storage-architecture.md`:
   `raw/{country}/{source_id}/{package_id}/{year}/{sha256}/{filename}` for
   publishers mapped to a country segment (UK and New Zealand today) and the
   legacy `raw/{source_id}/{package_id}/{year}/{sha256}/{filename}` shape for
   every unmapped publisher, US sources included. No source package parses
   the file.
3. **Bytes only with artifact-bound redistribution evidence.** Being
   downloadable is not a licence, and an allowlisted licence name is not
   evidence that this file was issued under it. Chronicle archives a release's
   bytes only when `access` is `public`, the recorded `licence` is on
   Chronicle's allowlist of redistributable terms (maintained in code with the
   evidence for each term: a U.S. Government work under 17 U.S.C. §105, the
   Open Government Licence v3, CC0, CC BY), and the entry carries
   `licence_evidence` binding the artifact to the term: issuer, licence
   identifier and version, the scope statement, a durable evidence URL, and
   the covered SHA-256. The `ledger-boundary` judge verifies that the evidence
   names the artifact. Any public-download file without that evidence is
   classed `licensed`. Licensed or restricted files (FRS under the UKDS
   agreement, BE-SILC scientific-use files, the IRS PUF) are registered
   hash-only: no bytes in any Chronicle store, no Chronicle credential grants
   access to them, and their bytes stay in the licensed environments
   consumers already operate. Public microdata bytes are staged in an
   untracked, transient directory outside `db/data/**` during acquisition and
   uploaded from there; a repository guard refuses tracked microdata bytes.
   Git custody of a microdata release is always manifest-only.
4. **No content.** No microdata record, row, column, row value, or cell enters
   `source_records`, `source_rows`, `source_columns`, `source_row_values`,
   `source_cells`, the relational registry, the derived-artifact bucket, or the
   journal, whether the release is public or gated. No fact computed from raw
   microdata by Chronicle or by a PolicyEngine-side consumer enters Chronicle,
   however many intermediate artifacts stand between the microdata and the
   value. The test is who asserted the value: a value that a third party
   asserted and published, whether the microdata's own publisher (Census over
   the ASEC) or another publisher (JCT or TPC over the IRS PUF, JRC over
   EU-SILC), is an ordinary fact with ordinary provenance; a value Chronicle or
   a PolicyEngine-side consumer computed is not. This is enforceable only once
   facts carry an asserting principal and root-artifact lineage (`asserted_by`,
   `root_artifacts`; chronicle#238): a fact rooted in a `microdata_release`
   registration is then rejected unless `asserted_by` is a third-party
   publisher, never Chronicle or a PolicyEngine-side consumer. Until then
   reviewers enforce it by hand.
5. **What a registration attests, and who.** Each registration records
   `hash_source` and its attester:
   - `chronicle_fetch`: Chronicle fetched the bytes and hashed them;
     `attested_by: chronicle`, `verified_at` is the fetch date. The
     registration attests the bytes.
   - `consumer_attested`: a consumer that holds the bytes recomputed the hash
     and recorded evidence; `attested_by` is that consumer,
     `attestation_evidence` points at its record, `verified_at` is its
     comparison date. The registration attests the bytes on that consumer's
     word.
   - `consumer_pin`: transcribed from a consumer's reviewed pin without
     recomputation; `attested_by` is that consumer, `pinned_from` names the
     repository path and commit, and there is no `verified_at`. The
     registration attests the pin, not the bytes.
   The witnessed journal (the release manifests on `codex/thesis-ledger-facts`,
   each committing to the journal state that first covers a registration's
   manifest hash) bounds when the registration existed. It bounds when the
   bytes existed only for `chronicle_fetch` and `consumer_attested` entries.
6. **Consumers point at the registration.** A Microcosm source-stage manifest
   that names a microdata artifact carries the Chronicle artifact reference
   and the same SHA-256, so every root of a build graph resolves to one
   witnessed registration and a build fails closed when its local bytes
   differ from the registered ones.

## Why

- **The publisher record is the thing to witness.** Publishers revise and
  withdraw microdata files: the IRS withdrew the public-use file in 2026, and
  Census reissues ASEC files under the same vintage label. A registered hash
  with a witnessed time is the only durable statement that a given release
  existed with those bytes. This is the same transparency property Chronicle
  already provides for published tables, applied to the files calibration
  actually starts from.
- **Pins today are scattered and unwitnessed.** Microcosm pins raw inputs in
  per-country manifests, checkpoint metadata, code constants, and command-line
  arguments, with no shared registry, no licence record, and no timestamp
  anyone outside the build can check. A build's root inputs deserve the same
  declared identity as every other node.
- **Content would break what makes the store useful.** Row-level microdata
  would grow the relational registry and journal by orders of magnitude,
  churn on every reissue, and put access-controlled bytes inside the one
  system whose value is that anyone can verify it. Thesis resolves forecasts
  against Chronicle observations; microdata are not observations of anything
  Thesis scores.
- **This keeps the 2026-06-30 ruling.** PR #68 removed microdata parsers,
  adapters, and tracked raw storage from the package. Nothing here brings any
  of that back. The code the enforcement PRs add refuses to read microdata
  rather than reading it.

## Consequences

- `docs/adr-chronicle-facts-only.md` is amended: "anything a publisher
  asserted is a fact" now excludes publisher-authored values at microdata
  grain, which are content; only the values a publisher asserted and
  published over that microdata are facts, whether the microdata's own
  publisher or another. README's boundary block says the same.
- Manifests gain `kind`, `licence`, `access`, `licence_evidence`,
  `hash_source`, `attested_by`, and per-source `verified_at` /
  `attestation_evidence` / `pinned_from`; validators refuse the combinations
  decisions 1, 3, and 5 forbid (#227).
- Fact provenance gains `asserted_by` and `root_artifacts` (#238).
- `docs/storage-architecture.md` narrows its non-goal to the three clauses in
  `chronicle/boundary.py`; its ownership matrix gains a row for microdata
  releases and states that Git custody is manifest-only with untracked
  staging. `docs/target-construction-harness-plan.md` is scoped to publisher
  aggregate artifacts.
- `chronicle/boundary.py` states the three negative clauses once;
  `.github/chronicle-agents.yml` carries them verbatim in the
  `ledger-boundary` judge verdict; `tests/test_chronicle_governance.py`
  asserts each complete clause, so the wording cannot regress or invert.
- Microcosm's raw-input entries reference Chronicle registrations
  (PolicyEngine/microcosm#848).
- Flip conditions for revisiting content: a consumer other than Microcosm
  needs row-level evidence from Chronicle, or a publisher grants
  redistribution of a currently licensed file and Chronicle has an
  access-controlled tier designed for it. Neither holds today.
