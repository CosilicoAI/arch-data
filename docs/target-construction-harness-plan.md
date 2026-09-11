# Fact Construction Harness Plan

Scope: this plan covers publisher aggregate artifacts (workbooks, CSVs, API
extracts of published statistics). Raw microdata releases are out of its
scope: Chronicle registers them by identity only and never preserves or
parses their content (`docs/adr-chronicle-raw-microdata-identity.md`).

## Goal

Build a harness that verifies source fact construction end to end:

1. Chronicle preserves the full source artifact behind each PolicyEngine calibration source.
2. Chronicle preserves parsed source cells, including values PE omits.
3. Chronicle emits simulator-neutral source records for published statistical values.
4. Microcosm can later compose active target values from Chronicle facts under declared modeling choices.
5. Each active target is bound downstream to the correct model measure: variable, count basis, filters, aggregation, units, weights, entity semantics, geography, period, and universe.

This is separate from parser tests. Parser tests answer whether a spreadsheet was read correctly. The harness answers whether the parsed value became a source-backed fact with the right dimensions, constraints, provenance, and lineage.

## Boundary

Load-bearing rule:

> Chronicle can change representation. Microcosm can change meaning.

Chronicle owns source preservation and simulator-neutral source records.

Chronicle may:

- Store immutable artifacts, checksums, retrieval metadata, parsed sheets/tables/rows/cells.
- Preserve raw value, raw units, source table, source row, source column, formula/display metadata, notes, and extraction method.
- Re-express published values in canonical units and IDs when the conversion is mechanical and reversible.
- Emit aggregate facts: source-published values that Microcosm may later use.
- Run source-side integrity checks, such as declared component-to-total checks.

Chronicle must not:

- Store PolicyEngine, Microcosm, or Axiom model variable IDs in source records.
- Age values to a different model year.
- Reconcile inconsistent national/state/district values.
- Select active targets.
- Choose a deflator, growth factor, or source preference.
- Apply simulator-specific calibration weighting.

Microcosm owns active targets and model measures.

Microcosm may:

- Select records from Chronicle.
- Age source records to the model year.
- Reconcile across source granularities.
- Decide which calibration profile is active.
- Compile model measures against PolicyEngine or future Axiom variables.
- Bind source-derived target values to model-side measures through explicit target contracts.

## ChatGPT Pro Review

I asked ChatGPT Pro to critique the plan as a senior data/simulation architecture reviewer. The useful changes incorporated here are:

- Remove model variable identity from Chronicle records. Chronicle records should use simulator-neutral source concepts, statistics, universes, periods, geographies, and dimensions.
- Split cell selection from semantic interpretation. A selector failure and a wrong AGI-band/universe mapping should fail different tests.
- Add a first-class target contract layer. Correct source value plus correct model expression is not enough if their units, universes, entities, periods, or geographies do not match.
- Treat PE parity as a pinned differential harness, not the truth oracle.
- Preserve every parsed source cell, but do not pretend every spreadsheet cell is a semantic source record.

## Core Artifacts

### Source Artifact Manifest

The source manifest is the top-level checklist of documents/files PE uses to construct national and local targets.

Current files:

- `docs/pe-us-source-manifest.csv`
- `docs/pe-us-source-manifest.md`

Required fields:

- `origin_project`
- `pipeline`
- `jurisdiction`
- `source_id`
- `artifact_kind`
- `artifact`
- `filename`
- `format`
- `ledger_source_status`
- `source_cell_status`
- `target_construction_status`
- `value_capture_policy`
- `notes`

Acceptance criteria:

- Every PE source document/file used by national or local calibration appears in the manifest.
- Every row has an explicit policy to preserve the full source artifact, including omitted PE rows.
- `ledger_source_status` is row-parse inventory status only: `not_loaded`, `inventory_error`, `identity_mismatch`, `fetch_error`, `fetched_unparsed`, `parsed_no_rows`, or `row_parsed`.
- `row_parsed` does not imply source-cell completeness, selector readiness, or target-construction readiness. Those are represented by separate `source_cell_status` and `target_construction_status` fields.
- A row is row-parsed only when the expected artifact instance matches the stored artifact identity. URL-backed artifacts must match the expected URL; local artifacts must match the expected local path. Hash/vintage checks should be added as expected checksums and publisher vintages become available.
- The manifest is a migration checklist, not the permanent ontology. The long-term ontology should distinguish source series, artifact instances, and PE usage crosswalks.

Long-term split:

```text
source_series
  IRS SOI Publication 1304 Table 1.4 workbooks

artifact_instance
  TY2023 workbook, exact checksum, retrieval timestamp

pe_usage_crosswalk
  PE used this source file/derived CSV/column for this target
```

### Parsed Source Cells

Chronicle should preserve every parsed source cell as a generic cell record.

```text
source_cell = generic parsed cell, including headers, notes, omitted cells
source_record = semantic statistical record selected from cells
```

This gives full preservation without claiming every spreadsheet cell is calibration-relevant.

Parsed cells should distinguish:

- empty cell
- zero
- suppressed value
- not applicable
- withheld or confidential
- formula cell
- displayed rounded value
- stored Excel numeric value
- footnoted value

### Cell Selector Specs

A selector spec maps a source artifact/table region to one or more parsed cells. It should use both coordinates and semantic guards when possible.

Example:

```yaml
selector_spec_id: irs_soi.ty2023.table_1_4.taxable_interest.amount.all.selector
artifact_slug: policyengine-us-data/national-soi-workbooks/irs_soi_ty2023_table_1_4.xls
table: Table 1.4
sheet: Table 1.4
cell:
  expected_address: T12
row_selector:
  header_path:
    - All returns
    - Total
column_selector:
  header_path:
    - Taxable interest
    - Amount
semantic_guards:
  expected_left_header: All returns
  expected_column_label_contains: Taxable interest
```

Rules:

- Coordinate-only selectors are allowed for first drafts, but semantic guards should be added before declaring the source family production-ready.
- A workbook layout shift should fail a selector test.
- A same-cell semantic change should fail a guard test.

### Source Record Specs

A source record spec interprets selected source cells as simulator-neutral statistical records.

Example:

```yaml
source_record_id: irs_soi.ty2023.table_1_4.taxable_interest.amount.all
selector_spec_id: irs_soi.ty2023.table_1_4.taxable_interest.amount.all.selector
source:
  raw_unit: thousand_usd
canonical:
  concept_id: irs_soi.taxable_interest_income
  statistic_id: aggregate_amount
  universe_id: irs_soi.all_individual_income_tax_returns
  value_unit: usd
  period_type: tax_year
  period: 2023
  geography_id: 0100000US
  geography_vintage: census_current
  dimensions:
    filing_status: all
    agi_lower: -inf
    agi_upper: inf
    taxable_only: true
lineage:
  preserve_raw_value: true
  preserve_source_cell: true
```

Rules:

- `scale` should not be a calibration-time target option. Mechanical scale conversion belongs to source-record construction.
- Raw values and canonical values should both be stored or reconstructible.
- Representation conversions are allowed only when they are mechanical and documented, e.g. thousands of dollars to dollars.
- Canonical values should use decimal/fixed precision, not binary float.
- `all`, `unknown`, and missing must be distinct. `all` is an explicit dimension value; `unknown` is explicit and rare; missing dimensions are invalid unless the concept schema permits them.
- Chronicle source records should not contain simulator variable IDs. They may contain source concepts, source-domain variables, statistics, universes, and dimensions.

### Target Value Recipes

A target value recipe is Microcosm-owned. It resolves Chronicle source records and transforms them into an active target value.

Specs should normally use source queries, not hard-coded source record IDs. Compiled target values should store the resolved source record IDs.

Example:

```yaml
recipe_id: us_2024.irs_soi.taxable_interest.amount.all.value
source_query:
  publisher: irs_soi
  table: individual_income_tax_returns_table_1_4
  concept_id: irs_soi.taxable_interest_income
  statistic_id: aggregate_amount
  universe_id: irs_soi.all_individual_income_tax_returns
  period: 2023
  geography_id: 0100000US
  dimensions:
    filing_status: all
    agi_lower: -inf
    agi_upper: inf
    taxable_only: true
source_selection_policy: unique_required
transform_dag:
  - method: age_dollar_value
    base_period: 2023
    target_period: 2024
    growth_factor_source: soi_total_agi_growth
output_unit: usd
```

Rules:

- Aging and reconciliation methods must be named and versioned.
- Identity/no-op recipes should be implemented first so the value side can be tested before aging.
- The compiled value records the resolved source IDs, transform fingerprint, Chronicle snapshot, and code/spec hashes.

### Model Measure Specs

A model measure spec maps a target contract to a quantity computed on a model
population.

Example:

```yaml
measure_id: policyengine_us.taxable_interest_income.taxable_filers
model: policyengine_us
entity: tax_unit
aggregate: sum
variable: taxable_interest_income
weight_variable: household_weight
weight_entity: household
output_unit: usd
filters:
  - variable: tax_unit_is_filer
    operator: "=="
    value: true
  - variable: income_tax_before_credits
    operator: ">"
    value: 0
```

For count targets:

```yaml
measure_id: policyengine_us.tax_unit_count.filers_by_agi
model: policyengine_us
entity: tax_unit
aggregate: sum
variable: tax_unit_count
weight_variable: household_weight
weight_entity: household
filters:
  - variable: tax_unit_is_filer
    operator: "=="
    value: true
  - variable: adjusted_gross_income
    operator: ">="
    value: 50000
  - variable: adjusted_gross_income
    operator: "<"
    value: 75000
```

Rules:

- Entity can be inferred from the model variable registry as a linting convenience, but the compiled production spec should store it explicitly.
- Counts should be represented as sums of count-valued measures, not as a separate aggregation method.
- The harness should reject non-sum count specs.
- Aggregation should be explicit, and count targets should use `sum`.
- Weight variable, weight entity, and entity joins must be explicit and validated. Microcosm should default to household weights for final dataset validation.

### Target Contracts

A target contract binds a source-derived target value to a model measure and states the compatibility expectations.

Example:

```yaml
target_contract_id: us_2024.irs_soi.taxable_interest.amount.all.contract
recipe_id: us_2024.irs_soi.taxable_interest.amount.all.value
measure_id: policyengine_us.taxable_interest_income.taxable_filers
source_period:
  period_type: tax_year
  period: 2023
  period_basis: source_tax_year
target_value_period:
  period_type: tax_year
  period: 2024
  period_basis: target_tax_year
model_evaluation_period:
  period_type: tax_year
  period: 2024
  period_basis: model_tax_year
expected_statistic_id: aggregate_amount
expected_value_definition: total_nominal_dollars
expected_unit: usd
currency_basis:
  currency_year: nominal_current
  scale: units
  annualization: annual
expected_entity: tax_unit
expected_universe_id: tax_units_with_taxable_interest_income_and_positive_income_tax
expected_weighting: household_weighted_estimate
denominator: none
geography_id: 0100000US
geography_basis:
  code_system: geoid
  boundary_vintage: census_current
  assignment_basis: filing_residence
dimensions:
  filing_status: all
  agi_lower: -inf
  agi_upper: inf
tolerance:
  absolute: 1
  relative: 0.000001
priority: primary
```

Rules:

- A correctly parsed source value and a correctly compiled model measure can still be wrong together if their universes differ.
- Contracts should fail closed on statistic, value definition, unit, currency basis, denominator, weighting, entity, universe, geography basis, source period, transformed target-value period, model evaluation period, and dimension mismatch.
- Aged targets must keep source period and target-value period separate so valid aging does not look like an accidental period mismatch.
- Shares, rates, means, ratios, medians, percentiles, and bin counts need explicit denominator or statistic semantics. They must not be inferred from labels alone.
- Administrative totals, weighted survey estimates, and unweighted microdata counts are different value definitions even when their period, entity, geography, and unit match.
- Duplicate active targets in the same profile should be rejected unless marked as alternatives or competing definitions.

### Active Target Specs

An active target spec places a target contract into a Microcosm calibration profile.

Example:

```yaml
active_target_id: us_2024.irs_soi.taxable_interest.amount.all
target_contract_id: us_2024.irs_soi.taxable_interest.amount.all.contract
calibration_profile: populace_us_2024_national_v1
status: active
replacement_for:
  project: policyengine-us-data
  target_id: soi.taxable_interest_income.amount.all
replacement_mode: improved_source_lineage
```

Rules:

- Chronicle should not own active target specs.
- Excluded source records and excluded candidate targets need explicit exclusion reasons.
- Active target values should be reproducible from Chronicle snapshot + Microcosm specs + transform DAG.

## Harness Layers

### Layer 1: Source Preservation Harness

Purpose:

Ensure every manifest row is represented in Chronicle source tables.

Checks:

- Artifact exists in `source_artifacts`.
- Stored artifact identity matches the expected manifest artifact instance: source URL for URL-backed artifacts, local path for local artifacts, and later expected checksum/vintage when available.
- Checksum is present.
- Retrieval status is explicit.
- Storage URI is present.
- Publisher/source URL or documented manual source is present.
- Retrieved/accessed date is present.
- Format is present.
- Parser version is present where parsing is attempted.
- Parsed table count is nonzero unless binary-only.
- Parsed row/cell count is nonzero for spreadsheet/CSV/JSON/text files.
- Source pages that fetch with errors are explicit `.fetch_error.yaml` artifacts, not silently absent.
- Parsed tables store row count, column count, parse-region checksum, sheet/page/table lineage, skipped-region explanations, and parse status.
- Row-parsed artifacts are not treated as source-cell-complete, selector-ready, or target-construction-ready until the `source_cell`, selector, and target layers say so explicitly.

Initial acceptance:

- PE-US manifest parsed coverage increases as new publisher-source documents are loaded, and publisher-source coverage is tracked separately from PE intermediate/support coverage.
- National SOI workbook rows move from todo to done.

### Layer 2: Selector Harness

Purpose:

Ensure source specs select the intended cells robustly.

Checks:

- Every selector resolves to exactly one source artifact unless a multi-cell selector is explicit.
- Row and column selectors match at least one cell.
- Expected coordinate and semantic guards agree.
- Header paths match.
- Neighbor guards match when supplied.
- Empty, zero, suppressed, not applicable, formula, displayed value, and stored numeric value are represented distinctly.

Failure modes this should catch:

- Cell moved but meaning stayed the same.
- Cell stayed the same but meaning changed.
- Parser shifted a merged header or footnote into the wrong column.

### Layer 3: Source Record Construction Harness

Purpose:

Ensure selected source cells are transformed into canonical Chronicle source records without value loss or simulator leakage.

Checks:

- Source records contain no simulator variable IDs.
- Raw value is captured before canonical conversion.
- Canonical value equals raw value converted by declared unit rules.
- Unit conversion is not applied twice.
- All required dimensions are present: period, geography, statistic, universe, AGI band, filing status, taxable-only, count-vs-amount.
- `all`, `unknown`, and missing are distinct.
- Omitted PE rows are still present as source records when they appear in the source artifact and pass semantic selection.
- Source integrity checks pass where declared.

Initial SOI focus:

- Publication 1304 Table 1.1: all rows/columns used for AGI, return counts, filing status.
- Publication 1304 Table 1.4: all income-source columns and AGI rows, not only PE-targeted variables.
- Publication 1304 Table 2.1: itemized deductions and AGI rows.
- IRS `in55cmcsv`: all state AGI-stub variables, not only `N1`, `A00100`, and income tax.
- IRS `incd`: all congressional-district AGI-stub variables.

Optional source integrity checks:

- Sum of declared mutually exclusive components approximately equals source total.
- Amount is nonnegative unless the concept permits negative values.
- Recipient count is less than or equal to universe count where both are published.
- Published mean equals amount/count when the source publishes all three.
- State totals sum to national only when the publisher states that the components are exhaustive.

### Layer 4: Target Value Recipe Harness

Purpose:

Ensure Microcosm target values are reproducible from Chronicle records and declared transformations.

Checks:

- Source query resolves to the expected number of records.
- Source selection policy is satisfied.
- Transform DAG is declared and hashable.
- Identity/no-op recipes reproduce the source record value exactly.
- Aging/reconciliation/source-selection methods are named and versioned.
- Target value is reproducible from Chronicle snapshot + source records + transform metadata.
- Excluded records have explicit exclusion reasons.

### Layer 5: Model Measure Harness

Purpose:

Ensure each target contract has the right model-side expression.

Checks:

- All referenced variables exist in the selected model backend.
- Entity is explicit in the compiled spec.
- Weight variable and weight entity are explicit and compatible.
- Filters compile to backend expressions.
- Filter entity compatibility is checked.
- Counts have a count basis.
- Ratio numerator and denominator are compatible.
- NaN, missing, and negative value handling are explicit.
- Output unit is explicit.
- Aggregates produce finite values on fixture datasets.
- Tiny synthetic fixtures produce hand-checkable results.
- Matrix-builder results match direct simulation results for the same variable, geography, weights, and filters.

Tiny fixture examples:

- Two tax units, one filer and one non-filer.
- One filer with positive taxable interest and no income tax.
- One filer with positive income tax.
- Multiple persons in a single household to catch person-vs-tax-unit-vs-household count confusion.
- Multiple tax units in a household to catch household mapping errors.
- A ratio target with zero denominator handling.

### Layer 6: PE Differential Harness

Purpose:

For targets inherited from PE, compare our source values, active values, measures, and profile decisions against pinned PE outputs.

PE is a migration reference, not the source of truth. This matters because PE has multiple historical/current target-construction paths, and we already intend to supersede omissions and some modeling choices.

Comparison modes:

```text
exact
expected_difference
replacement
not_applicable
not_yet_classified
```

Checks:

- Pin PE version/commit, dataset, and target files.
- Compare Chronicle source records against PE source CSV/workbook-derived values where PE has equivalents.
- Compare Microcosm active target values against PE target databases/CSVs where PE has equivalents.
- Compile equivalent Microcosm measure specs on the same dataset.
- Compare each PE-equivalent measure vector where an equivalence mapping exists.
- Compare included/excluded target profile decisions where PE has an equivalent profile.
- Fail on unclassified differences in entity mapping, filters, signs, units, weights, aggregation, or profile inclusion.

Scope:

- Start with national SOI rows because PE has useful legacy national target machinery there.
- Add PE local/geography-aware parity after national slices work.
- Do not require parity for improved or intentionally changed Microcosm targets; those should have explicit `replacement_for` metadata and a reason.

### Layer 7: Active Target Profile Harness

Purpose:

Ensure Microcosm's active calibration profiles are explicit and reproducible.

Checks:

- Every active target points to one target contract.
- Every contract points to one target value recipe and one model measure.
- Every compiled active target value records resolved Chronicle source records.
- Aging/reconciliation/source-selection methods are declared.
- Target value is reproducible from source records and method metadata.
- No duplicate active target occupies the same measure/year/geography/dimension domain unless explicitly marked as alternative.
- Profile hash is reproducible from Chronicle snapshot, model adapter, measure specs, target specs, and code versions.

## Database Implications

Chronicle source tables already store artifacts, parsed tables, columns, and rows. However, the current row-oriented storage is not sufficient for the selector harness: it does not preserve stable spreadsheet coordinates, empty-cell distinctions, formulas, displayed values, merged-header lineage, or footnotes. The next implementation step must add a real `source_cell` layer before, or alongside, source-record specs.

After `source_cell` exists, the next tables or artifacts should be source-record oriented, not active-target oriented.

Candidate Chronicle concepts:

```text
source_series
source_artifact
source_table
source_cell
cell_selector_specs
source_record_specs
source_records
source_record_lineage
chronicle_snapshots
```

`source_records` should include:

- stable source record ID
- source record spec ID and spec hash
- source artifact/table/cell lineage
- publisher
- source table label
- source vintage
- raw text
- raw value
- raw unit
- canonical value
- canonical unit
- period type
- period
- geography ID
- geography vintage
- source concept ID
- statistic ID
- universe ID
- dimension JSON
- dimension hash
- missingness code
- footnotes JSON
- lineage hash

`source_records` should not include PolicyEngine, Microcosm, or Axiom model variable IDs.

Microcosm-side artifacts should include:

```text
model_adapters
measure_specs
target_value_recipes
target_contracts
active_target_specs
active_target_values
target_profile_manifests
target_profile_snapshots
```

These can start as YAML/JSON files before being promoted into database tables.

Authoring and materialization pattern:

```text
YAML spec -> schema validation -> compiled DB rows -> emitted source records/target values
```

The DB should store spec hashes and compiled representations. It should not be the normal hand-editing surface.

## Implementation Sequence

### Phase 0: Boundary And Schema Contract

Status: started through this plan.

Tasks:

- Freeze the minimum boundary rules:
  - no model variable IDs in Chronicle source records
  - source specs are versioned and hashable
  - active targets are reproducible from Chronicle snapshot + Microcosm specs
  - every target has a value side, measure side, and binding contract
- Keep `docs/pe-us-source-manifest.csv` current.
- Add missing publisher source documents as they are discovered.
- Add manifest generation to CI as a smoke test.
- Fail CI if a known PE source disappears from the manifest unintentionally.

### Phase 1: Artifact Vault And Parsed-Cell Pilot

Tasks:

- Load full IRS SOI TY2023 Table 1.4 workbook into Chronicle source artifacts.
- Capture artifact checksum, retrieval metadata, and storage URI.
- Parse sheets/tables/cells.
- Preserve headers, footnotes, raw cell values, displayed values, and omitted adjacent columns.
- Add parsed-cell coverage checks.

Acceptance:

- The Table 1.4 artifact is immutable and checksum-addressed.
- Parsed cell coverage is stable and tested.
- Empty/zero/suppressed/not-applicable/formula/displayed-value distinctions are represented.

### Phase 2: Selector And Source Record Specs

Tasks:

- Define Pydantic/dataclass models for:
  - `CellSelectorSpec`
  - `SourceRecordSpec`
- Add YAML loader with clear validation errors.
- Add unit conversion registry:
  - `usd`
  - `thousand_usd`
  - `count`
  - `thousand_count`
  - `person_months` if needed later
- Add source cell selector primitives:
  - Excel column letter
  - zero/one-based row index with explicit convention
  - label lookup
  - header path lookup
  - table/sheet selection
- Add source-record specs for all Table 1.4 columns PE currently maps.
- Extend specs to all adjacent source columns in the workbook, even if PE omits them.
- Emit canonical simulator-neutral source records.

Acceptance:

- Chronicle can reproduce all PE TY2023 Table 1.4 rows in `soi_targets.csv` at the source-value layer.
- Chronicle also exposes source records for non-targeted Table 1.4 columns/rows.
- Source records have concept/statistic/universe metadata and no simulator model variable IDs.

### Phase 3: Microcosm Target Contract Handoff

Tasks:

- Publish stable source-record IDs, periods, units, geography, and value
  definitions in Chronicle facts.
- Declare selectors and model-measurement contracts in Microcosm.
- Validate consumer contracts against a hash-pinned Chronicle facts artifact.

Acceptance:

- Microcosm can resolve every contract row back to Chronicle source records and
  source artifacts.
- Chronicle remains unaware of model variables and active calibration decisions.
- Microcosm owns target contracts, active values, aging, model-measure
  compilation, scoring, and differential tests.

### Phase 4: Expand SOI National Family

Tasks:

- Extend source-record specs to SOI Table 1.1, Table 1.4, Table 2.1, top-tail tables, filing-status variants, and AGI-band variants.
- Add source integrity checks for totals/components where valid.

Acceptance:

- Source records exist for every relevant value in the national SOI source files.
- Missing and intentionally omitted values are visible in coverage reports.

### Phase 5: CI And Dashboard

Tasks:

- Add CI job for source manifest smoke tests.
- Add selector/source-record spec validation tests.
- Surface Chronicle-owned results in the source observatory:
  - source coverage
  - parsed-cell coverage
  - source-record coverage
  - known source coverage gaps

Acceptance:

- A developer can see whether a source-package version preserves more official
  statistics release content, regresses source fidelity, or intentionally omits
  source rows.
- CI catches accidental source omission, selector breakage, and source coverage
  regressions. Microcosm CI covers its target-contract schema and fact resolution.

### Phase 8: State And Local Expansion

Tasks:

- Extend source-record specs to `in55cmcsv`, `incd`, ACA, SNAP, Medicaid, ACS age, and Census support files.
- Add local PE matrix/database parity where available.
- Keep PE-local replacement scope separate from national replacement scope.

Acceptance:

- Source records exist for every value in PE target source files.
- Active Microcosm local/geography-aware targets can be benchmarked against PE local outputs where PE local parity data exists.
- Local-readiness report shows which source families are not yet Microcosm-active.

## Open Design Answers

1. Source-record specs should live in Chronicle as YAML source of truth plus compiled DB rows.
2. Selection contracts and active target specs live in Microcosm.
3. Count targets should be sums of count-valued measures, never a separate count aggregation.
4. PE parity should be required enough to classify differences, not enough to force exact reproduction of PE omissions or legacy shortcuts.
5. Chronicle should preserve every parsed source cell, then layer semantic source records on top. Do not automatically turn every cell into a semantic record.

## Recommendation

Start with YAML specs plus generated database records.

Use:

- Chronicle for source artifacts, parsed cells, selectors, and simulator-neutral source records.
- Microcosm for target value recipes, model measure specs, target contracts,
  active target profiles, and PE differential metadata.
- PE parity as a pinned differential harness, not as the source of truth.

The first implementation should target IRS SOI TY2023 Table 1.4 because it exercises the main failure modes:

- workbook parsing
- raw-cell preservation
- selector semantic guards
- thousands-to-dollars conversion
- taxable-only interpretation
- source concept/statistic/universe mapping
- AGI-bin filters
- count-vs-amount distinction

First-slice acceptance criteria:

- The Table 1.4 artifact is stored with checksum and retrieval metadata.
- Parsed cells preserve raw/displayed/formula/missingness distinctions.
- Selector specs cover all PE-used cells and adjacent omitted source columns.
- Source records are simulator-neutral and contain concept/statistic/universe metadata.
- Downstream Microcosm contracts select Chronicle source records without moving
  model measures, runtime code, active support decisions, or compiler
  configuration into Chronicle.
