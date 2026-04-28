# Arch Data Architecture

## Overview

Arch is Cosilico's source-data foundation for social simulation. It captures
source publications, preserves provenance, and represents published values as
structured, queryable facts. Microplex consumes Arch facts to produce final
calibrated simulation inputs.

This document describes the full data pipeline from source publications to
Microplex target sets and calibrated output.

## Repository Boundaries

| Layer | Owns | Does not own |
|-------|------|--------------|
| Arch | Source artifacts, provenance, source facts, target inputs, microdata ingestion | Source reconciliation, aging, imputation, active target selection |
| Microplex Targets | Source selection, reconciliation, aging, imputation, active target sets | Source artifact storage and provenance |
| Microplex | Entity model, weights, calibration interfaces, calibrated output | Source ETL and source provenance |
| Jurisdiction packages | Model-specific adapters, variable mappings, target recipes | Source facts |
| PolicyEngine | Policy-facing workflows and analysis tools | Source ETL or calibrated microdata generation |

## Storage Layers

### Object Storage

Source files are immutable and versioned.

```text
sources/
  irs/soi/2023/table_1_2.xlsx          # IRS SOI individual returns
  census/acs/2023/pums_hh.csv          # ACS PUMS households
  census/cps/2024/asec_raw.zip         # CPS ASEC microdata
  bls/cpi/2024/monthly.csv             # CPI monthly series
  usda/snap/2023/qc_data.xlsx          # SNAP QC data
```

### Supabase Schemas

| Schema | Purpose | Example Tables |
|--------|---------|----------------|
| `arch` | Source metadata and lineage | sources, files, content, fetch_log |
| `indices` | Source time series | series, values (CPI, wage growth) |
| `targets` | Target inputs | strata, constraints, targets |
| `microdata` | Intermediate microdata | cps_asec, acs_pums, synthetic |
| `microplex` | Final calibrated data | households, persons, tax_units |

## Python Namespaces

New code should use the `arch` namespace:

```python
from arch.sources import SourceFile, SourceReference, query_sources
from arch.facts import SourceFact
from arch.targets import Target, query_targets
from arch.microdata import query_cps_asec
from arch.normalization import convert_units
```

The `db` package contains the current SQLModel persistence and loader
implementation behind the public `arch` namespace.

## Data Flow

```text
source publications
(files, manifests,
 parsed-as-published cells)
      |
      v
arch.sources
(source lineage references)
      |
      v
arch.facts
(structured source claims)
      |
      +------------------------+
      v                        v
arch.normalization        microdata.*
(units, scales,           (typed source
 IDs, source-published     microdata)
 arithmetic)
      |                        |
      v                        |
targets.*                     |
(target input                 |
 facts)                       |
      |                        |
      +-----------+------------+
                  v
        Microplex Targets
   (selected, reconciled,
    aged active target sets)
                  |
                  v
             microplex.*
          (final calibrated
              microdata)
```

## Source Facts And Target Inputs

Source ETL should separate source facts from Microplex target composition:

1. Load or parse source publications into source lineage and published cells.
2. Materialize source-backed facts in Arch.
3. Apply representation-only normalization such as unit scale conversion or
   source-published total/share arithmetic.
4. Materialize the fact as a target input with source and derivation metadata.
5. Let Microplex Targets select, reconcile, age, and activate target sets.

The `arch.normalization` package owns low-assumption representation helpers:

```python
from arch.facts import SourceFact
from arch.targets import DataSource, Jurisdiction, TargetType
from arch.normalization import as_target, convert_units

snap_households = SourceFact(
    name="snap_households",
    value=22_323,
    period=2023,
    unit="thousands",
    source=DataSource.USDA_SNAP,
    jurisdiction=Jurisdiction.US,
)

target_input = as_target(
    convert_units(snap_households, 1000, "count"),
    target_type=TargetType.COUNT,
    stratum_name="US SNAP Households",
)
```

Projection facts from official sources such as CBO, OBR, and ONS can be loaded
as source facts directly. Cosilico-owned inflation, aging, projection, or
cross-source reconciliation assumptions belong in Microplex Targets, not Arch.

## Calibration Pipeline

### 1. Target Inputs (from `targets.*` schema)

Target inputs define source-backed aggregates that Microplex may use:

```sql
-- targets.strata: Population subgroups
INSERT INTO targets.strata (name, jurisdiction, constraints)
VALUES ('CA adults 18-64', 'us', '[{"variable": "age", "operator": ">=", "value": "18"}, ...]');

-- targets.targets: Source-backed aggregate values
INSERT INTO targets.targets (stratum_id, variable, value, period)
VALUES (1, 'eitc_recipients', 2500000, 2023);
```

### 2. Variable Mapping

Arch target input variables are source-linked variable IDs. They should not
depend on a simulator implementation. Microplex jurisdiction packages map those
target IDs to model variables and entities.

### 3. Target Composition

Microplex Targets owns composition from source-backed inputs to active target
sets:

```python
target_set = microplex.targets.compose(
    inputs=arch_targets,
    target_year=2024,
    reconciliation="scale_states_to_national",
    aging="apply_published_growth_factor",
)
```

Every source choice, reconciliation rule, aging method, and activation rule is
declared and versioned in Microplex, not Arch.

### 4. Hierarchical Constraint Building

Since all weights are at the household level, person-level targets must be
aggregated:

```python
# What we want: count of people aged 18-64 in California
# What we compute: for each household, count matching persons

build_hierarchical_constraint_matrix(
    hh_df=households,      # 18,825 rows
    person_df=persons,     # 48,292 rows
    targets=targets,       # Microplex active target set
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

### 6. Output (`microplex.*` schema)

```sql
-- microplex.households: Calibrated household weights
SELECT household_id, state_fips, weight, ...
FROM microplex.households;
-- 18,825 rows

-- microplex.persons: Linked to households
SELECT person_id, household_id, age, employment_income, ...
FROM microplex.persons;
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

### arch.* (Source Lineage)

```sql
arch.sources        -- institution, dataset, url, update_frequency
arch.files          -- r2_key, checksum, source_id, fetched_at
arch.content        -- parsed-as-published text/tables
arch.fetch_log      -- change detection, version history
```

### indices.* (Source Time Series)

```sql
indices.series      -- series_id, name, source, frequency
indices.values      -- series_id, date, value
```

Microplex recipes can reference these source series when they choose an
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

Targets in Arch are source-backed inputs. Active, aged, reconciled calibration
target sets belong to Microplex Targets.

### microdata.* (Intermediate)

```sql
microdata.cps_asec       -- processed CPS (cleaned, typed)
microdata.acs_pums       -- processed ACS
microdata.synthetic      -- generated synthetic records
```

### microplex.* (Final Output)

```sql
microplex.households     -- calibrated household records with weights
microplex.persons        -- person records linked to households
microplex.tax_units      -- tax unit records (future)
```
