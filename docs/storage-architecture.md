# Chronicle Storage Architecture

This note is the canonical storage plan for Chronicle while the source-package
harness stabilizes. The detailed agent workflow remains in
`docs/agent-source-package-harness.md`; this document only defines where each
class of Chronicle data belongs.

## Decision Summary

Chronicle uses three storage layers with different jobs.

`ledger-raw` is the immutable source-byte archive. It stores exact publisher
artifacts as fetched: workbooks, CSVs, PDFs, ZIPs, HTML snapshots, and similar
government-statistics release files. Raw objects are
content-addressed by checksum and should never be overwritten in place.

Raw microdata releases follow the same rule at the artifact level only. The
bytes of a `public` release whose licence is on Chronicle's redistributable
allowlist, and whose registration carries artifact-bound redistribution
evidence naming the file (Census public-use files are the model case), live in
the raw bucket under the same content-addressed key shape. Every other
microdata release (FRS, BE-SILC, the IRS PUF, and any publicly downloadable
file under unstated terms) is registered by manifest, with checksum, vintage,
licence, access route, hash source, and verification date, and no bytes in any
Chronicle store. No microdata is parsed into records, rows, columns, row
values, or cells, and no fact is derived from it. Until chronicle#221 adds
registry columns, a registration exists only in its manifest: `ledger.db` and
Supabase carry nothing for it. See
`docs/adr-chronicle-raw-microdata-identity.md`.

`ledger-derived` is the reproducible artifact archive. It stores build outputs
that Chronicle can regenerate from raw bytes, package specs, parser code, and build
configuration. Examples include parsed-cell or parsed-row Parquet/JSONL files,
source record outputs, `ledger.db`, mirror JSONL exports, QA reports, Data
Package metadata, and RO-Crate metadata.

Supabase/Postgres is the queryable relational registry for accepted Chronicle
builds. It stores rows that applications, agents, and downstream systems need
to search and join: source artifacts, source rows/cells, source records,
aggregate facts, aggregate constraints, concept alignments, lineage edges, build
metadata, validation status, and object-location pointers. Supabase is not the
source of raw bytes and should not be hand-edited as the ingestion authority.

The deterministic local build remains the authority for source-backed facts.
Hosted tables mirror accepted build outputs and provide a shared query surface.

## Ownership Matrix

| Data class | Git/local package | `ledger-raw` R2 | `ledger-derived` R2 | SQLite `ledger.db` | Supabase/Postgres |
|------------|-------------------|---------------|-------------------|------------------|-------------------|
| Source package specs | Authoritative YAML and parser code | No | Optional packaged snapshot | No | Metadata only |
| Raw publisher files | Tiny fixtures only | Authoritative bytes | No | Metadata only | Metadata plus R2 pointer |
| Raw microdata releases | Manifest only (`kind: microdata_release`); public bytes staged in an untracked transient directory, never tracked | Bytes only for `public` releases with artifact-bound redistribution evidence; hash-only otherwise | No | Not represented until chronicle#221 | Not represented until chronicle#221 |
| Source manifests | Authoritative checked metadata | No | Optional snapshot | Metadata loaded into tables | Queryable artifact registry |
| Parsed source rows/cells | Generated local output | No | Snapshot artifact | Queryable table | Queryable mirror |
| Source records/facts | Generated local output | No | Snapshot artifact | Queryable table | Queryable mirror |
| Aggregate constraints | Generated local output | No | Snapshot artifact | Queryable table | Queryable mirror |
| Build reports and QA | Generated local output | No | Snapshot artifact | Build summary rows | Queryable validation status |
| Mirror JSONL exports | Generated local output | No | Snapshot artifact | Export source | Bulk-load input |
| Microcosm active targets | No | No | No | No | Future adapter output outside Chronicle core |

## Object Key Conventions

New UK and New Zealand source artifacts use country-organized,
content-addressed keys:

```text
raw/{country}/{source_id}/{package_id}/{year}/{sha256}/{filename}
```

For example, an IRD artifact uses:

```text
raw/nz/ird/ird-working-for-families-statistics-sept-2025/2024/{sha256}/working-for-families-statistics---sept-2025.xlsx
```

The implemented country segments are `nz` and `uk`. US objects deliberately
retain the legacy shape `raw/{source_id}/...`; migrating those keys requires a
separate consumer audit. The fetch and raw-publish commands infer the country
from the package publisher directory. Raw publication refuses to replace a
manifest-recorded key that disagrees with the inferred country path.

New UK and New Zealand derived build artifacts use the same country segment
and build-scoped keys so different builds can coexist and be audited:

```text
derived/{country}/{source_id}/{package_id}/{year}/{build_id}/{artifact_name}
```

Examples:

```text
derived/uk/ons/ons-mye-2024-uk/2024/{build_id}/source_cells.jsonl
derived/nz/ird/ird-working-for-families-statistics-sept-2025/2024/{build_id}/ledger.db
```

Legacy US derived keys likewise remain `derived/{source_id}/...`.

Derived artifacts are reproducible and may be replaced by a new build, but a
specific `{build_id}` path should be immutable once published.

## Relational Registry Contract

The hosted `chronicle` schema should be the lookup surface for Chronicle, not the place
where agents invent source facts. Rows should be bulk-loaded from deterministic
build outputs.

The registry should expose:

- source artifact identity: source name, table/file, URL, vintage, extraction
  date, extraction method, checksum, size, and raw R2 bucket/key/URI;
- source rows/cells and source records, including exact source-row and
  source-cell lineage;
- source columns and source-row values, so row-oriented artifacts are queryable
  by raw or normalized column names without JSON scans;
- aggregate facts and aggregate constraints with stable keys, dimensions,
  filters, units, aggregation semantics, labels, and source provenance;
- concept alignments, including source concept, canonical concept, relation,
  authority, legal vintage, and evidence;
- build metadata, validation status, and derived artifact R2 bucket/key/URI.

The current Supabase migration mirrors the core relational tables and includes
R2 location fields for raw source artifacts and derived build artifacts, so the
registry can serve as the shared index over both R2 buckets.

## Build And Publish Flow

The intended flow is:

1. Register raw source artifacts with `uv run chronicle fetch-artifact`, which
   writes local bytes, records checksums in `manifest.yaml`, and can upload the
   exact bytes to `ledger-raw`. Existing manifest-declared artifacts can be
   checksum-validated, uploaded, and linked with `uv run chronicle publish-raw`.
   Production package specs may omit raw bytes from Git as long as the manifest
   keeps `source_url` and SHA-256 metadata; builds can fill
   `LEDGER_SOURCE_ARTIFACT_CACHE_DIR` by setting
   `LEDGER_SOURCE_ARTIFACT_FETCH=1`. The old `CHRONICLE_`-prefixed environment
   variables remain accepted only as migration fallbacks.
2. Validate and build a source package with `uv run chronicle validate-package` and
   `uv run chronicle build-suite`.
3. Produce local deterministic outputs: parsed rows/cells, source records,
   aggregate facts, `ledger.db`, QA reports, Data Package metadata, and RO-Crate
   metadata.
4. Export relational mirror files with `uv run chronicle export-db-tables`.
5. Publish derived build outputs to `ledger-derived`:

   ```bash
   uv run chronicle publish-derived \
     --dir /tmp/chronicle-suite \
     --source-id irs_soi \
     --package-id soi-table-1-1 \
     --year 2023 \
     --build-artifacts-out /tmp/chronicle-build-artifacts.jsonl
   ```

6. Bulk-load or upsert accepted relational rows into Supabase/Postgres:

   ```bash
   uv run chronicle load-supabase-mirror \
     --dir /tmp/chronicle-mirror \
     --build-artifacts /tmp/chronicle-build-artifacts.jsonl
   ```

The Supabase project must have the checked migration applied and the `chronicle`
schema exposed in PostgREST/Data API settings before the REST loader can write
to it. Use `--dry-run` to verify local JSONL files without writing.

## Non-Goals

Supabase should not store large raw binary artifacts. It should point to R2.

Chronicle should not parse survey or administrative microdata into records,
rows, columns, row values, or cells, derive facts from it, or hold licensed or
restricted microdata bytes in any store. It registers microdata releases as
source artifacts and archives bytes only for public releases on its
redistributable allowlist that carry artifact-bound redistribution evidence
(see `docs/adr-chronicle-raw-microdata-identity.md`). It reflects government
statistics releases and the provenance needed to audit those published facts.

R2 should not be the schema authority. It stores bytes and reproducible build
files, while Chronicle code and checked specs define semantics.

Chronicle should not own Microcosm source selection, aging, reconciliation,
activation profiles, or simulator-specific target mappings. Those belong in
Microcosm or thin downstream adapters.
