# Chronicle Data Architecture

## Overview

Chronicle is PolicyEngine's source-data foundation for social simulation. It captures
source publications, preserves provenance, and represents published values as
structured, queryable facts. Microcosm consumes Chronicle facts to produce final
calibrated simulation inputs.

This document describes the source-publication pipeline from government
statistics releases to source-backed facts. Chronicle is global at the schema,
validation, and build-harness layer. Jurisdiction source packages such as
`ledger-us` and `chronicle-uk` emit records into that shared contract.

## Repository Boundaries

| Layer | Owns | Does not own |
|-------|------|--------------|
| Chronicle | Source artifacts (including microdata release registrations), provenance, aggregate facts, constraints | Selection and measurement contracts, microdata records, rows, columns, row values, or cells; facts computed from raw microdata by Chronicle or a PolicyEngine-side consumer; licensed or restricted microdata bytes; source reconciliation, aging, imputation, active target selection |
| Microcosm Targets | Selection and measurement contracts, reconciliation, aging, imputation, active target sets | Source artifact storage and provenance |
| Microcosm | Entity model, weights, calibration interfaces, calibrated output | Source ETL and source provenance |
| Jurisdiction source packages | Source-specific parsers and specs that emit Chronicle records | Forked fact or constraint schemas |
| Jurisdiction simulation packages | Model-specific adapters, variable mappings, target recipes | Source facts |
| PolicyEngine | Policy-facing workflows and analysis tools | Source ETL or calibrated dataset generation |

## Storage Layers

### Object Storage

Source files are immutable and versioned.

```text
sources/
  irs/soi/2023/table_1_2.xlsx          # IRS SOI individual returns
  census/acs/2023/table_b01001.csv     # ACS published table
  bls/cpi/2024/monthly.csv             # CPI monthly series
  usda/snap/2023/qc_data.xlsx          # SNAP QC data
```

### Supabase Schemas

| Schema | Purpose | Example Tables |
|--------|---------|----------------|
| `chronicle` | Source metadata and lineage | sources, files, content, fetch_log |
| `indices` | Source time series | series, values (CPI, wage growth) |
| `targets` | Target inputs | strata, constraints, targets |
| `microcosm` | Final calibrated data | households, persons, tax_units |

## Python Namespaces

New code should use the `policyengine_chronicle` namespace:

```python
from policyengine_chronicle.sources import SourceFile, SourceReference, query_sources
from policyengine_chronicle.facts import SourceFact
from policyengine_chronicle.targets import Target, query_targets
from policyengine_chronicle.normalization import convert_units
```

The `db` package contains the current SQLModel persistence and loader
implementation behind the public `policyengine_chronicle` namespace.

Jurisdiction source packages should use short import namespaces and published
distribution names with a PolicyEngine prefix:

```text
repo: PolicyEngine/ledger-us
distribution: policyengine-ledger-us
import: policyengine_chronicle_us
```

They should depend on `policyengine-chronicle` and emit shared `chronicle` objects
rather than redefining source rows/cells, source-row values, aggregate facts,
aggregate constraints, stable keys, or DB tables.

## Data Flow

```text
source publications
(files, manifests,
 parsed-as-published cells)
      |
      v
policyengine_chronicle.sources
(source lineage references)
      |
      v
policyengine_chronicle.facts
(structured source claims)
      |
      |
      v
policyengine_chronicle.normalization
(units, scales, IDs, source-published arithmetic)
      |
      v
policyengine_chronicle.aggregate_facts
(published aggregate facts)
      |
      v
        Microcosm Targets
   (consumer-owned contracts;
    selected, reconciled,
    aged active target sets)
                  |
                  v
             microcosm.*
          (final calibrated
              datasets)
```

## Source Facts And Microcosm Targets

Source ETL should separate Chronicle aggregate facts from Microcosm target composition:

1. Load or parse source publications into source lineage and published cells.
2. Materialize source-backed facts in Chronicle.
3. Apply representation-only normalization such as unit scale conversion or
   source-published total/share arithmetic.
4. Keep the fact queryable with source and derivation metadata.
5. Let Microcosm select, reconcile, age, and activate calibration target sets.

Chronicle source facts can align source-published concepts to canonical vocabulary
terms. When a legal concept is available from Axiom, Chronicle should use the Axiom
term as the canonical concept key and keep the publisher's column/series concept
as `source_concept`. For example, SOI adjusted gross income is represented as:

```text
canonical concept: us:statutes/26/62#adjusted_gross_income
source concept:    irs_soi.adjusted_gross_income
relation:          exact
authority:         ledger-us
```

This alignment is evidence-bearing metadata, not a Chronicle dependency on Axiom
runtime behavior. Nonlegal empirical inputs can use shared Chronicle/common concepts
and later align to Axiom or Microcosm where appropriate.

Consumer-facing source vocabulary is a compatibility contract. Keep
`source_concept` stable across package vintages when the publisher's statistical
concept has not changed; put publication-specific wording in `source_table` and
labels instead. Within a `source_name`, `measure_id` must distinguish different
statistical measures even when their publisher columns share labels such as
`band_a` or `total`. A deliberate rename requires a migration note and a
regression test for the affected consumer selector.

The `policyengine_chronicle.normalization` package owns low-assumption representation helpers:

```python
from policyengine_chronicle.facts import SourceFact
from policyengine_chronicle.normalization import convert_units

snap_households = SourceFact(
    name="snap_households",
    value=22_323,
    period=2023,
    unit="thousands",
    source="usda_snap",
    jurisdiction="us",
)

normalized_fact = convert_units(snap_households, 1000, "count")
```

Projection facts from official sources such as CBO, OBR, and ONS can be loaded
as source facts directly. PolicyEngine-owned inflation, aging, projection, or
cross-source reconciliation assumptions belong in Microcosm Targets, not Chronicle.

### Downstream Adapter Aliases

Chronicle variables should describe source-backed facts, not downstream simulator
variables. If a Microcosm or PolicyEngine target cell names the same empirical
quantity differently, the alias belongs in the downstream adapter.

For example, IRS SOI publishes nonnegative income tax liability aggregates.
Chronicle should preserve that as an SOI liability fact, while a Microcosm adapter
may use it to satisfy a model target named `income_tax_positive`. Chronicle should
not create a duplicate source fact solely to match the model variable name.

This rule also applies in reverse: if a Microcosm target cell is really a
survey input, imputed model feature, or source-selection decision rather than a
publisher aggregate, the cell should stay out of Chronicle until a primary source
fact and its provenance are identified.

## Downstream Target Composition

### 1. Target Inputs (from `targets.*` schema)

Target inputs define source-backed aggregates that Microcosm may use:

```sql
-- targets.strata: Population subgroups
INSERT INTO targets.strata (name, jurisdiction, constraints)
VALUES ('CA adults 18-64', 'us', '[{"variable": "age", "operator": ">=", "value": "18"}, ...]');

-- targets.targets: Source-backed aggregate values
INSERT INTO targets.targets (stratum_id, variable, value, period)
VALUES (1, 'eitc_recipients', 2500000, 2023);
```

### 2. Variable Mapping

Chronicle fact concepts are source-linked or canonical vocabulary IDs. They should not
depend on a simulator implementation. Microcosm jurisdiction packages map those
target IDs to model variables and entities.

### 3. Target Composition

Microcosm Targets owns composition from source-backed inputs to active target
sets:

```python
target_set = microcosm.targets.compose(
    inputs=chronicle_targets,
    target_year=2024,
    reconciliation="scale_states_to_national",
    aging="apply_published_growth_factor",
)
```

Every source choice, reconciliation rule, aging method, and activation rule is
declared and versioned in Microcosm, not Chronicle.

### 4. Hierarchical Constraint Building

Since all weights are at the household level, person-level targets must be
aggregated:

```python
# What we want: count of people aged 18-64 in California
# What we compute: for each household, count matching persons

build_hierarchical_constraint_matrix(
    hh_df=households,      # 18,825 rows
    person_df=persons,     # 48,292 rows
    targets=targets,       # Microcosm active target set
)

# Returns: Constraint objects with indicators at household level
# indicator[i] = count of matching persons in household i
```

The key insight: since all persons in a household share the household weight:

```text
sum over HH(hh_weight * count_matching_in_hh) = total_matching_persons
```

### 5. IPF Calibration

Iterative Proportional Fitting adjusts household weights to match all targets:

```python
for iteration in range(max_iter):
    for constraint in constraints:
        current = sum(hh_weight * constraint.indicator)
        ratio = constraint.target_value / current
        hh_weight *= clip(ratio, 0.9, 1.1)

    hh_weight = clip(hh_weight, min_weight, max_weight)
```

### 6. Output (`microcosm.*` schema)

```sql
-- microcosm.households: Calibrated household weights
SELECT household_id, state_fips, weight, ...
FROM microcosm.households;
-- 18,825 rows

-- microcosm.persons: Linked to households
SELECT person_id, household_id, age, employment_income, ...
FROM microcosm.persons;
-- 48,292 rows
```

## Entity Hierarchy

```text
Household (weight lives here)
├── Tax Unit 1
│   ├── Person A (head)
│   └── Person B (spouse)
└── Tax Unit 2
    └── Person C (dependent filing separately)
```

- Weights are always at household level.
- Person-level targets aggregate count/sum per household.
- Tax-unit-level targets aggregate count/sum per household.
- Household-level targets use a direct indicator.

## Schema Details

### chronicle.* (Source Lineage)

```sql
chronicle.sources        -- institution, dataset, url, update_frequency
chronicle.files          -- r2_key, checksum, source_id, fetched_at
chronicle.content        -- parsed-as-published text/tables
chronicle.fetch_log      -- change detection, version history
```

### indices.* (Source Time Series)

```sql
indices.series      -- series_id, name, source, frequency
indices.values      -- series_id, date, value
```

Microcosm recipes can reference these source series when they choose an
indexing rule:

```yaml
indexing_rule eitc_inflation:
  series: indices/bls_chained_cpi_u
  base_year: 2015
  rounding: 10
```

### targets.* (Target Inputs)

```sql
targets.strata           -- population subgroups with constraints
targets.constraints      -- variable, operator, value per stratum
targets.targets          -- stratum_id, variable, value, period, source
```

Targets in Chronicle are source-backed inputs. Active, aged, reconciled calibration
target sets belong to Microcosm Targets.

### microcosm.* (Final Output)

```sql
microcosm.households     -- calibrated household records with weights
microcosm.persons        -- person records linked to households
microcosm.tax_units      -- tax unit records (future)
```
