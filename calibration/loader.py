"""
Microdata loading for calibration.

Loads raw microdata (CPS, FRS) with original survey weights.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Valid data sources
VALID_SOURCES = {"synthetic", "cps", "frs"}

# US State FIPS codes (valid ones)
US_STATE_FIPS = [
    1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51,
    53, 54, 55, 56
]


def _validate_year(year: int, source: str) -> None:
    """Validate year is reasonable."""
    current_year = datetime.now().year

    # Don't allow future years
    if year > current_year:
        raise ValueError(
            f"Year {year} is in the future. Maximum year is {current_year}."
        )

    # Don't allow ancient years
    if year < 1900:
        raise ValueError(
            f"Year {year} is too far in the past. Minimum year is 1900."
        )


def _generate_synthetic_cps(
    year: int,
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic microdata matching CPS structure.

    Creates realistic demographic and income distributions for testing
    calibration algorithms.

    Args:
        year: Year for synthetic data
        n_samples: Number of sample records to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic microdata and weights
    """
    if seed is not None:
        np.random.seed(seed)

    # Age distribution (roughly matching US population)
    # More children and working-age adults, fewer elderly
    age_groups = [
        (0, 18, 0.22),    # Children
        (18, 35, 0.22),   # Young adults
        (35, 55, 0.26),   # Middle-aged
        (55, 70, 0.18),   # Near retirement
        (70, 100, 0.12),  # Elderly
    ]

    ages = []
    for min_age, max_age, proportion in age_groups:
        group_size = int(n_samples * proportion)
        group_ages = np.random.randint(min_age, max_age + 1, group_size)
        ages.extend(group_ages)

    # Adjust to exact n_samples
    while len(ages) < n_samples:
        ages.append(np.random.randint(0, 100))
    ages = np.array(ages[:n_samples])

    # Employment status: 0=not in labor force, 1=employed, 2=unemployed
    # Children and elderly mostly not in labor force
    employment_status = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        if ages[i] < 16:
            employment_status[i] = 0  # Not in labor force
        elif ages[i] >= 65:
            # 20% still working, 75% retired, 5% unemployed
            employment_status[i] = np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05])
        else:
            # Working age: 62% employed, 34% not in labor force, 4% unemployed
            employment_status[i] = np.random.choice([0, 1, 2], p=[0.34, 0.62, 0.04])

    # Income (based on employment status and age)
    income = np.zeros(n_samples)
    for i in range(n_samples):
        if employment_status[i] == 1:  # Employed
            # Log-normal income distribution
            base_income = np.random.lognormal(10.5, 0.8)
            # Adjust by age (peak around 45-55)
            age_factor = 1.0 + 0.3 * np.sin(np.pi * (ages[i] - 18) / 47)
            income[i] = base_income * max(0.5, age_factor)
        elif employment_status[i] == 0:  # Not in labor force
            if ages[i] >= 65:
                # Retirement income (Social Security, pensions)
                income[i] = np.random.lognormal(9.5, 0.7)
            elif ages[i] < 18:
                # Children have no income
                income[i] = 0
            else:
                # Adults not in labor force may have some income
                income[i] = np.random.lognormal(8.0, 1.0) * np.random.choice([0, 1], p=[0.7, 0.3])
        else:  # Unemployed
            # Unemployment benefits or no income
            income[i] = np.random.lognormal(8.5, 0.5) * np.random.choice([0, 1], p=[0.4, 0.6])

    # has_children: probability depends on age
    has_children = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if 25 <= ages[i] <= 55:
            has_children[i] = np.random.choice([0, 1], p=[0.4, 0.6])
        elif 18 <= ages[i] < 25 or 55 < ages[i] <= 65:
            has_children[i] = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            has_children[i] = 0

    # State FIPS (weighted by population)
    # Approximate state population weights (simplified)
    state_weights = {
        6: 0.12,   # California
        48: 0.09,  # Texas
        12: 0.07,  # Florida
        36: 0.06,  # New York
        17: 0.04,  # Illinois
        42: 0.04,  # Pennsylvania
        39: 0.04,  # Ohio
        13: 0.03,  # Georgia
        37: 0.03,  # North Carolina
        26: 0.03,  # Michigan
    }
    other_weight = 1.0 - sum(state_weights.values())
    other_states = [s for s in US_STATE_FIPS if s not in state_weights]

    states = []
    for _ in range(n_samples):
        if np.random.random() < other_weight:
            states.append(np.random.choice(other_states))
        else:
            # Pick from weighted states
            state_probs = np.array(list(state_weights.values()))
            state_probs = state_probs / state_probs.sum()
            states.append(
                np.random.choice(list(state_weights.keys()), p=state_probs)
            )

    state_fips = np.array(states)

    # Survey weights (roughly matching CPS scale)
    # Total US population ~330M, sample of 10k means avg weight ~33k
    base_weight = 330_000_000 / n_samples
    # Add some variance
    weights = np.random.gamma(shape=4, scale=base_weight / 4, size=n_samples)

    return pd.DataFrame({
        "weight": weights,
        "age": ages,
        "income": income,
        "employment_status": employment_status,
        "has_children": has_children,
        "state_fips": state_fips,
    })


def _load_cps_from_file(year: int, auto_download: bool = True) -> Optional[pd.DataFrame]:
    """
    Attempt to load CPS data from parquet files, downloading if needed.

    Looks for data in micro/us/ directory. If not found and auto_download
    is True, downloads from Census Bureau.

    Args:
        year: Year of data to load
        auto_download: If True, download from Census if file not found

    Returns:
        DataFrame if file exists or downloaded, None otherwise
    """
    # Check standard locations
    data_dir = Path(__file__).parent.parent / "micro" / "us"

    potential_paths = [
        data_dir / f"cps_{year}.parquet",
        data_dir / f"cps_asec_{year}.parquet",
        data_dir / f"asec_{year}.parquet",
    ]

    for path in potential_paths:
        if path.exists():
            df = pd.read_parquet(path)
            # Ensure weight column exists
            if "weight" not in df.columns:
                # Try common weight column names
                weight_candidates = [
                    "ASECWT", "asecwt", "WTFINL", "wtfinl",
                    "PWGTP", "pwgtp", "survey_weight"
                ]
                for col in weight_candidates:
                    if col in df.columns:
                        df["weight"] = df[col]
                        break
                else:
                    raise ValueError(
                        f"Could not find weight column in {path}. "
                        f"Columns: {df.columns.tolist()}"
                    )
            return df

    # Try to download if not found
    if auto_download:
        try:
            from micro.us.census.download_cps import download_and_process_cps, CPS_URL_BY_YEAR

            if year in CPS_URL_BY_YEAR:
                print(f"CPS {year} not found locally, downloading from Census Bureau...")
                output_path = data_dir / f"cps_{year}.parquet"
                return download_and_process_cps(year, output_path)
        except ImportError:
            pass  # download_cps module not available
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to download CPS {year}: {e}")

    return None


def load_microdata(
    source: str,
    year: int,
    variables: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load microdata for calibration.

    Args:
        source: Data source ("synthetic" for testing, "cps" for US, "frs" for UK)
        year: Year of microdata
        variables: Optional list of variables to load (weight always included)
        seed: Random seed for synthetic data reproducibility

    Returns:
        DataFrame with microdata and original weights in 'weight' column

    Raises:
        ValueError: If source or year is invalid
    """
    # Validate source
    if source not in VALID_SOURCES:
        raise ValueError(
            f"Invalid source: {source}. Valid sources: {VALID_SOURCES}"
        )

    # Validate year
    _validate_year(year, source)

    # Load data based on source
    if source == "synthetic":
        df = _generate_synthetic_cps(year, seed=seed)

    elif source == "cps":
        # Try to load real CPS data
        df = _load_cps_from_file(year)
        if df is None:
            # Fall back to synthetic with a warning
            import warnings
            warnings.warn(
                f"CPS data for {year} not found, falling back to synthetic data. "
                "To use real CPS data, add parquet files to micro/us/."
            )
            df = _generate_synthetic_cps(year, seed=seed)

    elif source == "frs":
        # UK Family Resources Survey - not yet implemented
        import warnings
        warnings.warn(
            "FRS data loading not yet implemented, using synthetic data."
        )
        df = _generate_synthetic_cps(year, seed=seed)

    else:
        raise ValueError(f"Unknown source: {source}")

    # Filter to requested variables (weight always included)
    if variables is not None:
        # Ensure weight is always included
        cols_to_keep = set(variables) | {"weight"}
        available_cols = set(df.columns)
        missing = cols_to_keep - available_cols
        if missing:
            raise ValueError(
                f"Requested variables not in data: {missing}. "
                f"Available: {available_cols}"
            )
        df = df[list(cols_to_keep)]

    return df
