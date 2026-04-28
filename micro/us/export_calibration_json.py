"""
Export calibration results to JSON for the cosilico.ai dashboard.
"""

import json
from datetime import datetime
from gradient_calibrate import calibrate_weights
from tax_unit_builder import load_and_build_tax_units


def export_calibration_results(year: int = 2024, output_path: str = None) -> dict:
    """
    Run calibration and export results as JSON.

    Args:
        year: CPS data year
        output_path: Where to save JSON (if None, returns dict only)

    Returns:
        Results dict matching CalibrationResults TypeScript interface
    """
    print("Loading tax unit data...")
    df = load_and_build_tax_units(year)
    print(f"Loaded {len(df):,} tax units")

    # Filter to likely filers
    filer_mask = (
        (df['total_income'] > 13850) |
        (df['wage_income'] > 0) |
        (df['self_employment_income'] > 0)
    )
    df = df[filer_mask].copy()
    print(f"Filtered to {len(df):,} likely filers")

    print("Running calibration...")
    result = calibrate_weights(df, include_states=True, epochs=500)

    # Build metadata
    metadata = {
        "date": datetime.now().isoformat(),
        "data_year": year,
        "tax_year": 2021,  # IRS SOI targets are for 2021
        "n_records": len(df),
        "n_targets": len(result.targets_df),
        "initial_loss": float(result.initial_loss),
        "final_loss": float(result.final_loss),
        "optimizer": "torch" if result.epochs == 500 else "scipy",
        "epochs": result.epochs,
    }

    # Build summary
    weights = result.weights
    original_weights = result.original_weights
    adj = weights / original_weights

    summary = {
        "total_population_original": float(original_weights.sum()),
        "total_population_calibrated": float(weights.sum()),
        "total_agi_calibrated": float((df['adjusted_gross_income'] * weights).sum()),
        "mean_weight": float(adj.mean()),
        "std_weight": float(adj.std()),
        "min_weight": float(adj.min()),
        "max_weight": float(adj.max()),
    }

    # Build targets list
    targets = []
    for _, row in result.targets_df.iterrows():
        # Determine group (national, state, county)
        geo_id = row['geographic_id']
        if geo_id == "US":
            group = "national"
        elif len(str(geo_id)) == 2:
            group = "state"
        else:
            group = "county"

        # Format name nicely
        name = row['name']
        if row['variable'] == 'returns':
            category = f"Returns ({row['bracket']})"
        else:
            category = f"AGI ({row['bracket']})"

        rel_error = float(row['rel_error'])
        targets.append({
            "name": name,
            "group": group,
            "category": category,
            "target_value": float(row['target']),
            "estimated_value": float(row['estimate']),
            "relative_error": rel_error,
            "absolute_error": abs(rel_error),
        })

    results = {
        "metadata": metadata,
        "summary": summary,
        "targets": targets,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_path}")

    return results


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else "calibration-results.json"
    results = export_calibration_results(output_path=output)

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Initial loss: {results['metadata']['initial_loss']:.6f}")
    print(f"Final loss: {results['metadata']['final_loss']:.6f}")
    print(f"Reduction: {(1 - results['metadata']['final_loss'] / results['metadata']['initial_loss']) * 100:.1f}%")
    print(f"Calibrated population: {results['summary']['total_population_calibrated']:,.0f}")
    print(f"Calibrated AGI: ${results['summary']['total_agi_calibrated']:,.0f}")

    # Count by error level
    good = sum(1 for t in results['targets'] if t['absolute_error'] < 0.02)
    medium = sum(1 for t in results['targets'] if 0.02 <= t['absolute_error'] < 0.10)
    bad = sum(1 for t in results['targets'] if t['absolute_error'] >= 0.10)
    print(f"\nTarget accuracy: {good} good (<2%), {medium} medium (2-10%), {bad} bad (>10%)")
