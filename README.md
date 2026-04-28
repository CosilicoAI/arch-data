# Arch

Arch is Cosilico's source-data foundation for social simulation. It captures
source publications, preserves provenance, and represents published values as
structured, queryable facts.

Arch may normalize structure: parse files, type values, declare units and
scales, assign geography and period identifiers, and preserve lineage back to
source artifacts. Arch does not choose among sources, reconcile inconsistent
sources, age values, impute missing data, select active calibration targets, or
apply simulator-specific mappings.

Microplex consumes Arch facts to build simulation datasets and Microplex
Targets. Modeling choices live in Microplex, not Arch.

## Purpose

This repository provides:

- **Sources**: Source file references, retrieval metadata, manifests, checksums,
  and provenance.
- **Facts**: Source-backed claims represented with typed values, units,
  geography, period, source table, and lineage.
- **Normalization**: Low-assumption representation changes such as unit/scale
  conversion and source-published total/share arithmetic.
- **Target inputs**: Source-published aggregates, projections, rates, counts,
  and metadata that Microplex may use to compose calibration targets.
- **Microdata**: Survey and administrative microdata ingestion for CPS, PUF,
  FRS, and related datasets.
- **Jurisdiction loaders**: US and UK source-specific ETL.

Arch facts are not Cosilico's assertion that a source claim is ultimately true.
They are source-backed claims with provenance.

## Boundary

The load-bearing rule:

> Arch may re-express a published value, but may not choose among, reconcile,
> age, impute, or transform published values in ways that change their meaning.

| Layer | Owns | Examples |
|-------|------|----------|
| Arch Sources | Source artifacts and provenance | URLs, checksums, source files, parsed tables/cells |
| Arch Facts | Structured source claims | SOI cells, ACS estimates, CPI values, CBO-published projections |
| Arch Normalization | Representation changes | Unit scales, typed values, geography/date identifiers |
| Arch Target Inputs | Source facts shaped for calibration | SOI EITC totals, CBO baselines, source-published growth factors |
| Microplex Targets | Model-ready target sets | Source selection, reconciliation, aging, activation profiles |

## Structure

```text
arch/
├── arch/                    # Public Arch namespace
│   ├── sources/             # Source lineage helpers
│   ├── facts/               # Source-backed facts
│   ├── normalization/       # Low-assumption representation helpers
│   ├── targets/             # Target input schema, client, loaders
│   ├── microdata/           # Microdata registry, ingestion, queries
│   └── jurisdictions/       # Jurisdiction-specific loaders
├── db/                      # SQLModel persistence and source loaders
│   ├── schema.py            # SQLModel: Target, Stratum, StratumConstraint
│   ├── supabase_client.py   # Supabase client helpers
│   └── etl_*.py             # Source-specific ETL pipelines
├── micro/                   # Microdata source adapters
├── calibration/             # Calibration target adapters and constraints
├── microplex/               # Microplex output pipelines
├── data/                    # Cached data files
└── docs/                    # Architecture and source documentation
```

New code should prefer `arch.sources`, `arch.facts`, `arch.normalization`,
`arch.targets`, and `arch.microdata`.

## Quick Start

### 1. Install

```bash
pip install cosilico-arch
# Or for development:
git clone https://github.com/CosilicoAI/arch-data arch
cd arch
pip install -e ".[dev]"
```

### 2. Initialize and Load Target Inputs

```bash
arch init
arch load soi --years 2021
arch stats
```

### 3. Query Target Inputs in Python

```python
from arch.targets import DataSource, Target, TargetType
from calibration.targets import get_targets

target_inputs = get_targets(
    jurisdiction="us",
    year=2021,
    sources=["irs-soi"],
)
```

### 4. Query Microdata

```python
from arch.microdata import query_cps_asec

persons = query_cps_asec(year=2024, table_type="person", limit=10_000)
```

## Target Input Schema

Target inputs use a three-table schema:

- **strata**: Population subgroups, such as California filers with AGI between
  $50k and $75k.
- **stratum_constraints**: Rules defining each stratum.
- **targets**: Source-published aggregate values linked to strata.

These are inputs to Microplex target composition. Microplex owns the active,
reconciled, aged target sets used for calibration.

## Arch Facts And Target Inputs

Source facts should be structurally normalized before becoming target inputs.
Normalization is about representation, not modeling: units, scales, typed
values, geography IDs, period IDs, and same-source arithmetic where the source
publishes the total/share relationship.

Inflation, aging, cross-source reconciliation, source selection, and target
activation belong in Microplex Targets unless the source itself publishes the
adjusted or projected series.

```python
from arch.facts import SourceFact
from arch.targets import DataSource, Jurisdiction, TargetType
from arch.normalization import as_target, convert_units

fact = SourceFact(
    name="snap_households",
    value=22_323,
    period=2023,
    unit="thousands",
    source=DataSource.USDA_SNAP,
    jurisdiction=Jurisdiction.US,
)

target_input = as_target(
    convert_units(fact, 1000, "count"),
    target_type=TargetType.COUNT,
    stratum_name="US SNAP Households",
)
```

## Current Coverage

### Microdata

| Source | Variables | Description |
|--------|-----------|-------------|
| US CPS ASEC | 78 | Census household survey |
| US IRS PUF | 33 | Tax return sample |
| UK FRS | 29 | DWP household survey |

### Aggregate Facts And Target Inputs

| Source | Coverage | Description |
|--------|----------|-------------|
| IRS SOI | National, state, AGI brackets | Tax return aggregates |
| Census | Demographics, poverty, districts | Population statistics |
| BLS | Labor market and price data | Employment and index series |
| CBO | Federal projections | Budget and economic projections |
| SSA/SSI | National and state programs | Social Security data |
| SNAP | State-level | Food assistance |
| CMS | Medicaid and ACA enrollment | Health coverage |
| HMRC/ONS/OBR | UK tax, population, projections | UK official statistics |

## Boundaries

- **Arch** owns source data, provenance, source facts, target inputs, and
  microdata ingestion.
- **Microplex Targets** owns source selection, reconciliation, aging, imputation,
  active target sets, and calibration profiles.
- **Microplex** owns simulation interfaces, entity modeling, weights, and
  calibration execution.
- **Jurisdiction packages** such as `microplex-us` own simulation-specific
  variable mappings and target recipes.
- **PolicyEngine** owns policy-facing tools and analysis workflows.

## Related Repositories

- [microplex](https://github.com/CosilicoAI/microplex) - Core microsimulation
  abstractions and calibration interfaces.
- [microplex-us](https://github.com/CosilicoAI/microplex-us) - US-specific
  simulation adapters and calibration profiles.
