# Calibration Pipeline Design

## Overview

The calibration pipeline connects administrative targets to microdata, producing reweighted samples that match official statistics.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Microdata     │     │  Targets DB     │     │  Calibrated     │
│   (CPS/FRS)     │ ──▶ │  (SQLite)       │ ──▶ │  Weights        │
│   ~200k records │     │  ~1000 targets  │     │  weights.parquet│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Pipeline Steps

### 1. Load Microdata

```python
from calibration import load_microdata

# Load raw CPS with original weights
microdata = load_microdata(
    source="cps",           # or "frs" for UK
    year=2023,
    variables=["age", "employment_income", "is_tax_filer", ...]
)
# Returns: DataFrame with ~200k rows, original weights in 'weight' column
```

### 2. Query Targets

```python
from calibration import get_targets

targets = get_targets(
    jurisdiction="us",      # or "uk"
    year=2023,
    sources=["irs-soi", "census-acs", "bls"],  # which admin sources
    variables=["tax_returns", "total_agi", "employment"],  # optional filter
)
# Returns: List[TargetSpec] with strata constraints and values
```

### 3. Build Constraint Matrix

Map targets to microdata aggregations:

```python
from calibration import build_constraint_matrix

# Each target becomes a constraint: sum(weights * indicator) = target_value
constraints = build_constraint_matrix(
    microdata=microdata,
    targets=targets,
    tolerance=0.01,  # Allow 1% deviation
)
```

**Constraint types:**

| Target Variable | Microdata Aggregation |
|----------------|----------------------|
| `tax_returns` | `sum(w * is_tax_filer)` |
| `total_agi` | `sum(w * agi * is_tax_filer)` |
| `snap_households` | `sum(w * receives_snap)` |
| `population_65_plus` | `sum(w * (age >= 65))` |

### 4. Calibrate Weights

```python
from calibration import calibrate

new_weights = calibrate(
    microdata=microdata,
    constraints=constraints,
    method="entropy",       # or "raking", "linear"
    bounds=(0.1, 10.0),     # Weight adjustment bounds
    max_iterations=100,
)
```

**Methods:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| `entropy` | Minimize KL divergence from original weights | Default, smooth adjustments |
| `raking` | Iterative proportional fitting | Many margin constraints |
| `linear` | Linear regression adjustment | Few constraints, fast |

### 5. Validate & Export

```python
from calibration import validate, export_weights

# Check calibration quality
report = validate(
    microdata=microdata,
    new_weights=new_weights,
    targets=targets,
)
# Returns: ValidationReport with residuals, diagnostics

# Export for use in PolicyEngine
export_weights(
    weights=new_weights,
    output_path="calibrated_weights_2023.parquet",
    metadata={
        "source": "cps",
        "year": 2023,
        "targets_hash": targets.hash(),
        "method": "entropy",
    }
)
```

## API Design

```python
# calibration/__init__.py

from .loader import load_microdata
from .targets import get_targets, TargetSpec
from .constraints import build_constraint_matrix, Constraint
from .methods import calibrate, EntropyCalibrator, RakingCalibrator
from .validation import validate, ValidationReport
from .export import export_weights


# High-level convenience function
def calibrate_microdata(
    source: str,
    year: int,
    jurisdiction: str = "us",
    method: str = "entropy",
    output_path: str | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Full calibration pipeline in one call.

    Returns calibrated microdata and validation report.
    """
    microdata = load_microdata(source, year)
    targets = get_targets(jurisdiction, year)
    constraints = build_constraint_matrix(microdata, targets)
    new_weights = calibrate(microdata, constraints, method=method)
    report = validate(microdata, new_weights, targets)

    if output_path:
        export_weights(new_weights, output_path)

    microdata["weight"] = new_weights
    return microdata, report
```

## Directory Structure

```
arch/
├── calibration/
│   ├── __init__.py
│   ├── loader.py           # load_microdata()
│   ├── targets.py          # get_targets(), TargetSpec
│   ├── constraints.py      # build_constraint_matrix()
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── entropy.py      # EntropyCalibrator
│   │   ├── raking.py       # RakingCalibrator
│   │   └── linear.py       # LinearCalibrator
│   ├── validation.py       # validate()
│   └── export.py           # export_weights()
├── tests/
│   └── test_calibration.py
└── docs/
    └── calibration-pipeline.md  # This file
```

## Variable Mapping

The key challenge is mapping target variables to microdata variables:

```python
# calibration/variable_map.py

VARIABLE_MAP = {
    # Target variable -> (microdata_variable, aggregation_type)
    "tax_returns": ("is_tax_filer", "count"),
    "total_agi": ("adjusted_gross_income", "sum"),
    "snap_households": ("receives_snap", "count"),
    "snap_benefits": ("snap_amount", "sum"),
    "employment": ("is_employed", "count"),
    "unemployment": ("is_unemployed", "count"),
    "population": (None, "count"),  # Just sum weights
    "population_65_plus": ("age >= 65", "count"),

    # UK variables
    "income_tax_payers": ("pays_income_tax", "count"),
    "total_income_tax": ("income_tax", "sum"),
    "uc_households": ("receives_uc", "count"),
}
```

## Stratum Handling

Targets come with stratum constraints that must be applied:

```python
def apply_stratum_constraints(
    microdata: pd.DataFrame,
    stratum: Stratum,
) -> pd.Series:
    """Return boolean mask for records matching stratum."""
    mask = pd.Series(True, index=microdata.index)

    for constraint in stratum.constraints:
        if constraint.operator == "==":
            mask &= microdata[constraint.variable] == constraint.value
        elif constraint.operator == ">=":
            mask &= microdata[constraint.variable] >= float(constraint.value)
        # ... etc

    return mask
```

## Performance Considerations

| Dataset | Records | Targets | Calibration Time |
|---------|---------|---------|------------------|
| CPS | 200k | 100 | ~10s |
| CPS | 200k | 500 | ~30s |
| Full Census | 3M | 1000 | ~5 min |

Optimizations:
- Vectorized constraint evaluation (NumPy/Pandas)
- Sparse constraint matrices (scipy.sparse)
- Parallel target aggregation
- Caching stratum masks

## Next Steps

1. [ ] Implement `calibration/loader.py` (CPS/FRS loading)
2. [ ] Implement `calibration/targets.py` (query targets DB)
3. [ ] Implement `calibration/methods/entropy.py` (core algorithm)
4. [ ] Add tests with mock data
5. [ ] Benchmark on real CPS
6. [ ] Integrate with Arch microdata loaders
