"""Compare flat tax-unit and household-weight Microplex calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arch.targets import TargetSpec
from micro.us.entities import MicroplexEntityFrames, build_microplex_entities
from micro.us.pipeline import (
    CalibrationResult,
    calibrate_household_weights,
    calibrate_weights,
    load_cps_from_local_file,
    load_cps_from_supabase,
    load_targets_from_supabase,
    maybe_add_policyengine_income_tax,
    print_target_composition_diagnostics,
)
from micro.us.targets import MicroplexTargetProfile, compose_microplex_targets
from micro.us.validation_dashboard import (
    build_microplex_version_id,
    current_git_sha,
    write_calibration_dashboard,
)


@dataclass(frozen=True)
class HierarchyComparisonResult:
    """Outputs from flat-vs-household calibration comparison."""

    entities: MicroplexEntityFrames
    flat_tax_units: pd.DataFrame
    flat_result: CalibrationResult
    household_result: CalibrationResult
    target_comparison: pd.DataFrame
    summary: pd.DataFrame


def compare_flat_vs_household_calibration(
    *,
    year: int = 2024,
    limit: int | None = None,
    microdata_source: str = "local",
    cps_path: Path | None = None,
    target_source: str = "db",
    db_path: Path | None = None,
    age_soi: bool = True,
    include_amount_targets: bool = True,
    min_target_obs: int = 100,
    calibration_method: str = "auto",
    min_weight_factor: float = 0.1,
    max_weight_factor: float = 20.0,
    add_policyengine_tax: bool = True,
    output_dir: Path | None = None,
    reports_root: Path | None = None,
    version_id: str | None = None,
    verbose: bool = True,
) -> HierarchyComparisonResult:
    """Run flat and household calibration on the same Microplex entity build."""
    report_config = _report_config(
        year=year,
        limit=limit,
        microdata_source=microdata_source,
        cps_path=cps_path,
        target_source=target_source,
        db_path=db_path,
        age_soi=age_soi,
        include_amount_targets=include_amount_targets,
        min_target_obs=min_target_obs,
        calibration_method=calibration_method,
        min_weight_factor=min_weight_factor,
        max_weight_factor=max_weight_factor,
        add_policyengine_tax=add_policyengine_tax,
    )
    git_sha = current_git_sha()
    version_id = version_id or build_microplex_version_id(
        year=year,
        config=report_config,
        git_sha=git_sha,
    )
    output_dir = _resolve_output_dir(
        output_dir=output_dir,
        reports_root=reports_root,
        version_id=version_id,
    )

    if microdata_source == "local":
        persons = load_cps_from_local_file(year, path=cps_path, limit=limit)
    elif microdata_source == "supabase":
        persons = load_cps_from_supabase(year, limit=limit or 200_000)
    else:
        raise ValueError(f"Unknown microdata_source: {microdata_source}")

    targets, holdout_variables = _load_targets_and_holdouts(
        year=year,
        target_source=target_source,
        db_path=db_path,
        age_soi=age_soi,
        verbose=verbose,
    )

    entities = build_microplex_entities(persons)
    if add_policyengine_tax:
        entities = replace(
            entities,
            tax_units=maybe_add_policyengine_income_tax(
                entities.tax_units,
                targets,
                year=year,
                persons=entities.persons,
            ),
        )

    flat_tax_units = entities.tax_units[
        entities.tax_units["is_tax_filer"] == 1
    ].copy()
    if flat_tax_units.empty:
        raise ValueError("No filing tax units available for flat comparison.")

    if verbose:
        print("\nRunning flat tax-unit calibration...")
    flat_result = calibrate_weights(
        flat_tax_units,
        targets,
        include_amounts=include_amount_targets,
        min_obs=min_target_obs,
        calibration_method=calibration_method,
        weight_bounds=(min_weight_factor, max_weight_factor),
        holdout_variables=holdout_variables,
        verbose=verbose,
    )

    if verbose:
        print("\nRunning household-weight calibration...")
    household_result = calibrate_household_weights(
        entities.households,
        entities.tax_units,
        targets,
        include_amounts=include_amount_targets,
        min_obs=min_target_obs,
        calibration_method=calibration_method,
        weight_bounds=(min_weight_factor, max_weight_factor),
        holdout_variables=holdout_variables,
        verbose=verbose,
    )

    target_comparison = compare_target_diagnostics(
        flat_result.diagnostics,
        household_result.diagnostics,
    )
    summary = summarize_comparison(
        entities=entities,
        flat_tax_units=flat_tax_units,
        flat_result=flat_result,
        household_result=household_result,
        target_comparison=target_comparison,
    )

    if output_dir is not None:
        write_hierarchy_comparison(
            output_dir,
            summary=summary,
            target_comparison=target_comparison,
            flat_diagnostics=flat_result.diagnostics,
            household_diagnostics=household_result.diagnostics,
            metadata={
                "microplex_version": version_id,
                "git_sha": git_sha,
                "jurisdiction": "us",
                "year": year,
                "comparison": "flat_vs_household",
                "config": report_config,
            },
        )

    if verbose:
        print("\nFlat vs household summary:")
        _print_summary(summary)

    return HierarchyComparisonResult(
        entities=entities,
        flat_tax_units=flat_tax_units,
        flat_result=flat_result,
        household_result=household_result,
        target_comparison=target_comparison,
        summary=summary,
    )


def compare_target_diagnostics(
    flat_diagnostics: pd.DataFrame,
    household_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per target with flat and household diagnostic values."""
    key_columns = [
        "target_index",
        "source",
        "variable",
        "target_type",
        "period",
        "source_period",
        "stratum",
        "constraints",
        "target_value",
    ]
    flat = flat_diagnostics.add_prefix("flat_")
    household = household_diagnostics.add_prefix("household_")
    merged = flat.merge(
        household,
        left_on=[f"flat_{column}" for column in key_columns],
        right_on=[f"household_{column}" for column in key_columns],
        how="outer",
    )
    for column in key_columns:
        flat_column = f"flat_{column}"
        household_column = f"household_{column}"
        merged[column] = merged[flat_column].combine_first(merged[household_column])

    merged["flat_abs_post_error"] = merged["flat_post_error"].abs()
    merged["household_abs_post_error"] = merged["household_post_error"].abs()
    merged["abs_post_error_delta"] = (
        merged["household_abs_post_error"] - merged["flat_abs_post_error"]
    )
    leading = key_columns + [
        "flat_role",
        "household_role",
        "flat_status",
        "household_status",
        "flat_drop_reason",
        "household_drop_reason",
        "flat_n_obs",
        "household_n_obs",
        "flat_pre_error",
        "household_pre_error",
        "flat_post_error",
        "household_post_error",
        "flat_abs_post_error",
        "household_abs_post_error",
        "abs_post_error_delta",
    ]
    trailing = [column for column in merged.columns if column not in leading]
    return merged[leading + trailing].sort_values(
        ["target_index", "variable", "target_type"],
        na_position="last",
    )


def summarize_comparison(
    *,
    entities: MicroplexEntityFrames,
    flat_tax_units: pd.DataFrame,
    flat_result: CalibrationResult,
    household_result: CalibrationResult,
    target_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact scalar metrics for flat-vs-household comparison."""
    rows: list[dict[str, Any]] = [
        _metric("households", "count", len(entities.households)),
        _metric("persons", "count", len(entities.persons)),
        _metric("tax_units", "count", len(entities.tax_units)),
        _metric("flat_filing_tax_units", "count", len(flat_tax_units)),
    ]
    rows.extend(_result_metrics("flat", flat_result))
    rows.extend(_result_metrics("household", household_result))
    rows.extend(_diagnostic_error_metrics("flat", flat_result.diagnostics))
    rows.extend(_diagnostic_error_metrics("household", household_result.diagnostics))
    rows.extend(_delta_metrics(target_comparison))
    return pd.DataFrame(rows)


def write_hierarchy_comparison(
    output_dir: Path,
    *,
    summary: pd.DataFrame,
    target_comparison: pd.DataFrame,
    flat_diagnostics: pd.DataFrame,
    household_diagnostics: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write comparison artifacts to a directory."""
    dashboard_metadata = {
        "microplex_version": output_dir.name,
        "jurisdiction": "us",
        "comparison": "flat_vs_household",
        **(metadata or {}),
    }
    write_calibration_dashboard(
        output_dir,
        summary=summary,
        target_comparison=target_comparison,
        flat_diagnostics=flat_diagnostics,
        household_diagnostics=household_diagnostics,
        metadata=dashboard_metadata,
    )


def _report_config(
    *,
    year: int,
    limit: int | None,
    microdata_source: str,
    cps_path: Path | None,
    target_source: str,
    db_path: Path | None,
    age_soi: bool,
    include_amount_targets: bool,
    min_target_obs: int,
    calibration_method: str,
    min_weight_factor: float,
    max_weight_factor: float,
    add_policyengine_tax: bool,
) -> dict[str, Any]:
    return {
        "year": year,
        "limit": limit,
        "microdata_source": microdata_source,
        "cps_path": str(cps_path) if cps_path is not None else None,
        "target_source": target_source,
        "db_path": str(db_path) if db_path is not None else None,
        "age_soi": age_soi,
        "include_amount_targets": include_amount_targets,
        "min_target_obs": min_target_obs,
        "calibration_method": calibration_method,
        "min_weight_factor": min_weight_factor,
        "max_weight_factor": max_weight_factor,
        "add_policyengine_tax": add_policyengine_tax,
    }


def _resolve_output_dir(
    *,
    output_dir: Path | None,
    reports_root: Path | None,
    version_id: str,
) -> Path | None:
    if output_dir is not None:
        return output_dir
    if reports_root is not None:
        return reports_root / version_id
    return None


def _load_targets_and_holdouts(
    *,
    year: int,
    target_source: str,
    db_path: Path | None,
    age_soi: bool,
    verbose: bool,
) -> tuple[list[TargetSpec] | list[dict[str, Any]], tuple[str, ...]]:
    if target_source == "db":
        profile = MicroplexTargetProfile(age_soi=age_soi)
        composition = compose_microplex_targets(
            target_year=year,
            db_path=db_path,
            profile=profile,
        )
        if verbose:
            print_target_composition_diagnostics(composition)
        return composition.targets, profile.holdout_variables
    if target_source == "supabase":
        targets = load_targets_from_supabase(year)
        return targets, ()
    raise ValueError(f"Unknown target_source: {target_source}")


def _result_metrics(prefix: str, result: CalibrationResult) -> list[dict[str, Any]]:
    adjustments = result.adjustment_factors
    return [
        _metric(prefix, "calibration_unit", result.calibration_unit),
        _metric(prefix, "success", result.success),
        _metric(prefix, "method", result.method),
        _metric(prefix, "l2_loss", result.l2_loss),
        _metric(prefix, "original_weight_sum", result.original_weights.sum()),
        _metric(prefix, "calibrated_weight_sum", result.calibrated_weights.sum()),
        _metric(prefix, "adjustment_mean", adjustments.mean()),
        _metric(prefix, "adjustment_min", adjustments.min()),
        _metric(prefix, "adjustment_p01", np.quantile(adjustments, 0.01)),
        _metric(prefix, "adjustment_p50", np.quantile(adjustments, 0.50)),
        _metric(prefix, "adjustment_p99", np.quantile(adjustments, 0.99)),
        _metric(prefix, "adjustment_max", adjustments.max()),
    ]


def _diagnostic_error_metrics(
    prefix: str,
    diagnostics: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for role in ["active", "holdout", "diagnostic", "unsupported"]:
        role_diagnostics = diagnostics[diagnostics["role"] == role]
        rows.append(_metric(prefix, f"{role}_targets", len(role_diagnostics)))
        usable = role_diagnostics[role_diagnostics["post_error"].notna()]
        if usable.empty:
            continue
        abs_error = usable["post_error"].abs()
        rows.extend(
            [
                _metric(prefix, f"{role}_mean_abs_post_error", abs_error.mean()),
                _metric(prefix, f"{role}_median_abs_post_error", abs_error.median()),
                _metric(prefix, f"{role}_max_abs_post_error", abs_error.max()),
            ]
        )
    return rows


def _delta_metrics(target_comparison: pd.DataFrame) -> list[dict[str, Any]]:
    active = target_comparison[
        (target_comparison["flat_role"] == "active")
        & (target_comparison["household_role"] == "active")
    ]
    if active.empty:
        return []
    delta = active["abs_post_error_delta"].dropna()
    if delta.empty:
        return []
    return [
        _metric("delta", "active_mean_abs_post_error", delta.mean()),
        _metric("delta", "active_median_abs_post_error", delta.median()),
        _metric("delta", "active_household_better_targets", int((delta < 0).sum())),
        _metric("delta", "active_flat_better_targets", int((delta > 0).sum())),
    ]


def _metric(group: str, metric: str, value: Any) -> dict[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    return {"group": group, "metric": metric, "value": value}


def _print_summary(summary: pd.DataFrame) -> None:
    for _, row in summary.iterrows():
        print(f"  {row['group']}.{row['metric']}: {row['value']}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare flat and household-weight Microplex calibration",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--microdata-source",
        choices=["local", "supabase"],
        default="local",
    )
    parser.add_argument("--cps-path", type=Path, default=None)
    parser.add_argument("--target-source", choices=["db", "supabase"], default="db")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument(
        "--age-soi-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-amount-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-target-obs", type=int, default=100)
    parser.add_argument(
        "--calibration-method",
        choices=["auto", "ipf", "generalized-rake"],
        default="auto",
    )
    parser.add_argument("--min-weight-factor", type=float, default=0.1)
    parser.add_argument("--max-weight-factor", type=float, default=20.0)
    parser.add_argument(
        "--policyengine-tax",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=None,
        help="Directory where a version-named dashboard directory is created",
    )
    parser.add_argument(
        "--version-id",
        default=None,
        help="Microplex version identifier for the validation dashboard",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    compare_flat_vs_household_calibration(
        year=args.year,
        limit=args.limit,
        microdata_source=args.microdata_source,
        cps_path=args.cps_path,
        target_source=args.target_source,
        db_path=args.db_path,
        age_soi=args.age_soi_targets,
        include_amount_targets=args.include_amount_targets,
        min_target_obs=args.min_target_obs,
        calibration_method=args.calibration_method,
        min_weight_factor=args.min_weight_factor,
        max_weight_factor=args.max_weight_factor,
        add_policyengine_tax=args.policyengine_tax,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
        version_id=args.version_id,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
