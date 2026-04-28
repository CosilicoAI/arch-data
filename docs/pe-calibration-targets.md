# PolicyEngine-US-Data Calibration Targets

This document catalogs the calibration targets used in PolicyEngine-US-Data and compares them to Arch.

## Overview

PolicyEngine-US-Data calibrates the Enhanced CPS dataset to **2,813 targets** from multiple authoritative sources. The calibration uses a dropout-regularized gradient descent optimization algorithm to reweight survey records to match administrative benchmarks.

## Data Sources Used by PolicyEngine

| Source | Description | Target Categories |
|--------|-------------|-------------------|
| IRS Statistics of Income (SOI) | Tax return data | Income, deductions, credits, filing patterns |
| Census Population Projections | Demographic data | Age, geography, household composition |
| Congressional Budget Office (CBO) | Program estimates | Benefit program costs, economic projections |
| Treasury Expenditure Data | Government spending | Tax expenditure estimates |
| Joint Committee on Taxation (JCT) | Tax expenditures | Credit and deduction costs |
| Healthcare Spending Data | Medical costs | Insurance patterns, healthcare expenditures |
| Social Security Administration (SSA) | Retirement/disability | OASDI, SSI participation and costs |

## Target Categories

### 1. IRS SOI Tax Targets

From `etl_irs_soi.py` in policyengine-us-data:

#### Income Variables
| Variable Code | PolicyEngine Variable | Description |
|---------------|----------------------|-------------|
| 00100 | `adjusted_gross_income` | Total AGI |
| 00300 | `taxable_interest_income` | Taxable interest |
| 00400 | `tax_exempt_interest_income` | Tax-exempt interest |
| 00600 | `dividend_income` | Dividend income |
| 00650 | `qualified_dividend_income` | Qualified dividends |
| 01000 | `net_capital_gain` | Net capital gains |
| 01400 | `taxable_ira_distributions` | Taxable IRA distributions |
| 01700 | `taxable_pension_income` | Taxable pension income |
| 02300 | `unemployment_compensation` | Unemployment benefits |
| 02500 | `taxable_social_security` | Taxable Social Security |
| 26270 | `partnership_s_corp_income` | Pass-through income |

#### Tax Variables
| Variable Code | PolicyEngine Variable | Description |
|---------------|----------------------|-------------|
| 06500 | `income_tax` | Total income tax |
| 11070 | `refundable_ctc` | Refundable CTC |
| 59661-59664 | `eitc` | EITC by number of children (0, 1, 2, 3+) |

#### Deduction Variables
| Variable Code | PolicyEngine Variable | Description |
|---------------|----------------------|-------------|
| 04475 | `qualified_business_income_deduction` | QBI deduction |
| 17000 | `medical_expense_deduction` | Medical expenses |
| 18425 | `salt_deduction` | SALT deduction |
| 18500 | `real_estate_taxes` | Real estate taxes |

#### Stratification
- **Geographic**: National, State, Congressional District
- **Income**: 9 AGI brackets (from under $1 to over $500,000)
- **Filing Status**: Single, Married Joint, Married Separate, Head of Household

### 2. Census Demographic Targets

From `etl_age.py` in policyengine-us-data:

#### Age Distribution
| Age Range | Variable |
|-----------|----------|
| 0-4 | `person_count` |
| 5-9 | `person_count` |
| 10-14 | `person_count` |
| 15-19 | `person_count` |
| 20-24 | `person_count` |
| 25-29 | `person_count` |
| 30-34 | `person_count` |
| 35-39 | `person_count` |
| 40-44 | `person_count` |
| 45-49 | `person_count` |
| 50-54 | `person_count` |
| 55-59 | `person_count` |
| 60-64 | `person_count` |
| 65-69 | `person_count` |
| 70-74 | `person_count` |
| 75-79 | `person_count` |
| 80-84 | `person_count` |
| 85+ | `person_count` |

**Source**: Census Table S0101 (Age and Sex)

**Geographic Levels**:
- National (1)
- State (51)
- Congressional District (436)

### 3. Benefit Program Targets

#### SNAP (etl_snap.py)
| Variable | Description | Geographic Level |
|----------|-------------|------------------|
| `household_count` | SNAP households | State, Congressional District |
| `snap` | Total benefit cost | State |

**Sources**:
- Administrative data (Source ID 3): State totals
- ACS Survey data (Source ID 4): Congressional district via Census Table S2201

#### Medicaid (etl_medicaid.py)
| Variable | Description | Geographic Level |
|----------|-------------|------------------|
| `person_count` | Medicaid enrollment | State, Congressional District |

**Sources**:
- State Medicaid administrative reports (Source ID 2)
- Census ACS variable S2704_C02_006E

### 4. Tax Expenditure Targets

Referenced but specific structure not visible in code:

- **Treasury**: Total tax expenditure estimates
- **JCT**: Joint Committee on Taxation estimates for specific provisions

## Database Schema

PolicyEngine-US-Data uses a three-table schema:

### strata
| Column | Type | Description |
|--------|------|-------------|
| stratum_id | int | Primary key |
| definition_hash | str(64) | SHA-256 hash of constraints |
| parent_stratum_id | int | Hierarchical parent |
| stratum_group_id | int | Groups related strata |
| notes | str | Description |

### stratum_constraints
| Column | Type | Description |
|--------|------|-------------|
| stratum_id | int | Foreign key to strata |
| constraint_variable | USVariable | PolicyEngine variable name |
| operation | str | Comparison operator |
| value | str | Threshold value |

### targets
| Column | Type | Description |
|--------|------|-------------|
| target_id | int | Primary key |
| variable | USVariable | PolicyEngine variable |
| period | int | Year |
| stratum_id | int | Foreign key to strata |
| reform_id | int | Policy scenario (0=baseline) |
| value | float | Target amount |
| source_id | int | Data source reference |
| active | bool | Whether target is used |
| tolerance | float | Acceptable error percentage |

## Comparison: Arch vs PolicyEngine

### Currently Implemented in Arch

| Category | Source | ETL File | Status |
|----------|--------|----------|--------|
| **IRS SOI** | IRS Table 1.1 | `etl_soi.py` | Basic (national totals, AGI brackets, filing status) |
| **IRS SOI State** | IRS Historic Table 2 | `etl_soi_state.py` | Partial (5 states, limited years) |
| **SNAP** | USDA FNS | `etl_snap.py` | Basic (national + 10 states) |
| **Census** | Population Estimates | `etl_census.py` | Basic (population, households, age groups) |
| **CBO** | Budget Outlook | `etl_cbo.py` | Macro projections only |
| **SSA** | Trustee Report | `etl_ssa.py` | Basic |
| **BLS** | Employment | `etl_bls.py` | Basic |

### Gaps vs PolicyEngine

#### High Priority - Missing Target Categories

1. **IRS SOI Income by Source** (Priority: High)
   - Interest income, dividends, capital gains, pensions
   - Partnership/S-Corp income
   - Not in our SOI ETL

2. **IRS SOI Credits** (Priority: High)
   - EITC by number of children
   - Refundable CTC
   - We have docs but no ETL implementation

3. **IRS SOI Deductions** (Priority: High)
   - Medical expenses, SALT, real estate taxes
   - QBI deduction
   - Not implemented

4. **Medicaid Enrollment** (Priority: High)
   - State-level enrollment counts
   - Congressional district estimates
   - Missing entirely

5. **Congressional District Geography** (Priority: Medium)
   - PE has 436 district-level targets
   - We only have state-level

6. **Age Distribution Granularity** (Priority: Medium)
   - PE has 18 age brackets
   - We have 5 broad groups

#### Medium Priority - Coverage Gaps

7. **All 50 States + DC** (Priority: Medium)
   - PE covers all states
   - We only have top 10 by population

8. **Healthcare Spending Patterns** (Priority: Medium)
   - Insurance enrollment
   - ACA marketplace data
   - Missing entirely

9. **SSI Targets** (Priority: Medium)
   - Supplemental Security Income
   - Not in SSA ETL

#### Lower Priority - Enhancement Opportunities

10. **Treasury Tax Expenditures** (Priority: Low)
    - Annual tax expenditure estimates
    - Would validate credit/deduction totals

11. **JCT Estimates** (Priority: Low)
    - Tax provision cost estimates
    - Cross-validation source

## Target Count Comparison

| Category | PolicyEngine (est.) | cosilico (current) | Gap |
|----------|---------------------|-------------------|-----|
| IRS SOI Tax | ~500 | ~80 | ~420 |
| Demographics | ~900 | ~50 | ~850 |
| Benefit Programs | ~800 | ~75 | ~725 |
| Healthcare | ~300 | 0 | ~300 |
| Tax Expenditures | ~300 | 0 | ~300 |
| **Total** | **~2,813** | **~205** | **~2,608** |

## Implementation Recommendations

### Phase 1: Core Tax Targets (Est. +400 targets)
1. Expand IRS SOI to all 50 states + DC
2. Add income by source (interest, dividends, capital gains)
3. Add EITC targets by child count
4. Add CTC/ACTC targets

### Phase 2: Benefit Programs (Est. +500 targets)
1. Add Medicaid enrollment (state + district)
2. Expand SNAP to all states
3. Add SSI participation and benefits
4. Add TANF (if modeled)

### Phase 3: Demographics (Est. +800 targets)
1. Expand age brackets to 18 groups
2. Add congressional district level
3. Add race/ethnicity distributions

### Phase 4: Healthcare & Advanced (Est. +400 targets)
1. Add healthcare insurance enrollment
2. Add ACA marketplace data
3. Add tax expenditure cross-validation

## Sources

- [PolicyEngine US Data Documentation](https://policyengine.github.io/policyengine-us-data/)
- [PolicyEngine US Data GitHub Repository](https://github.com/PolicyEngine/policyengine-us-data)
- [PolicyEngine Microcalibrate Package](https://github.com/PolicyEngine/microcalibrate)
- [IRS Statistics of Income](https://www.irs.gov/statistics/soi-tax-stats-statistics-of-income)
- [Census Bureau Population Estimates](https://www.census.gov/programs-surveys/popest.html)
- [CBO Budget and Economic Data](https://www.cbo.gov/data/budget-economic-data)
- [USDA SNAP Data](https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap)

---

*Document created: 2024-12-22*
*Based on: policyengine-us-data repository analysis*
