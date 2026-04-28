"""Versioned calibration dashboard artifacts for Microplex validation."""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPORT_SCHEMA_VERSION = 1


def build_microplex_version_id(
    *,
    year: int,
    config: dict[str, Any],
    git_sha: str | None = None,
) -> str:
    """Return a stable default version id for a Microplex validation run."""
    sha = git_sha or current_git_sha()
    short_sha = sha[:7] if sha else "dev"
    fingerprint = hashlib.sha256(
        json.dumps(_json_safe(config), allow_nan=False, sort_keys=True).encode(
            "utf-8",
        ),
    ).hexdigest()[:8]
    return f"microplex-us-{year}-{short_sha}-{fingerprint}"


def current_git_sha() -> str | None:
    """Return the current git sha when running inside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_calibration_dashboard(
    output_dir: Path,
    *,
    summary: pd.DataFrame,
    target_comparison: pd.DataFrame,
    flat_diagnostics: pd.DataFrame,
    household_diagnostics: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Write structured calibration report artifacts for a Microplex version."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / "summary.csv", index=False)
    target_comparison.to_csv(output_dir / "target_comparison.csv", index=False)
    flat_diagnostics.to_csv(output_dir / "flat_diagnostics.csv", index=False)
    household_diagnostics.to_csv(
        output_dir / "household_diagnostics.csv",
        index=False,
    )

    active_targets = _active_target_comparison(target_comparison)
    variable_summary = _variable_summary(active_targets)
    worst_targets = _worst_targets(active_targets)
    largest_regressions = _largest_delta_targets(active_targets, ascending=False)
    largest_improvements = _largest_delta_targets(active_targets, ascending=True)

    variable_summary.to_csv(output_dir / "variable_summary.csv", index=False)
    worst_targets.to_csv(output_dir / "worst_targets.csv", index=False)
    largest_regressions.to_csv(
        output_dir / "largest_household_regressions.csv",
        index=False,
    )
    largest_improvements.to_csv(
        output_dir / "largest_household_improvements.csv",
        index=False,
    )

    metrics = build_dashboard_metrics(
        summary=summary,
        target_comparison=target_comparison,
        variable_summary=variable_summary,
    )
    manifest = build_dashboard_manifest(metadata=metadata, metrics=metrics)

    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "dashboard.html").write_text(
        render_calibration_dashboard(
            manifest=manifest,
            metrics=metrics,
            variable_summary=variable_summary,
            worst_targets=worst_targets,
            largest_regressions=largest_regressions,
            largest_improvements=largest_improvements,
        ),
        encoding="utf-8",
    )


def build_dashboard_metrics(
    *,
    summary: pd.DataFrame,
    target_comparison: pd.DataFrame,
    variable_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Build dashboard metrics from report data frames."""
    summary_metrics = _summary_to_nested_dict(summary)
    active = _active_target_comparison(target_comparison)
    delta = active["abs_post_error_delta"].dropna()

    return {
        "summary": summary_metrics,
        "target_roles": {
            "flat": _role_counts(target_comparison, "flat_role"),
            "household": _role_counts(target_comparison, "household_role"),
        },
        "active_delta": {
            "target_count": int(len(active)),
            "household_better_targets": int((delta < 0).sum()),
            "flat_better_targets": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "mean_abs_post_error_delta": float(delta.mean())
            if not delta.empty
            else None,
            "median_abs_post_error_delta": float(delta.median())
            if not delta.empty
            else None,
        },
        "by_variable": variable_summary.to_dict(orient="records"),
    }


def build_dashboard_manifest(
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a machine-readable report manifest."""
    version_id = metadata.get("microplex_version") or metadata.get("version_id")
    if not version_id:
        raise ValueError("Dashboard metadata must include microplex_version.")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "microplex_calibration_dashboard",
        "microplex_version": version_id,
        "generated_at": metadata.get("generated_at")
        or datetime.now(timezone.utc).isoformat(),
        "git_sha": metadata.get("git_sha"),
        "jurisdiction": metadata.get("jurisdiction", "us"),
        "year": metadata.get("year"),
        "comparison": metadata.get("comparison", "flat_vs_household"),
        "config": metadata.get("config", {}),
        "headline": {
            "flat_active_mean_abs_post_error": _nested_get(
                metrics,
                "summary",
                "flat",
                "active_mean_abs_post_error",
            ),
            "household_active_mean_abs_post_error": _nested_get(
                metrics,
                "summary",
                "household",
                "active_mean_abs_post_error",
            ),
            "active_targets": _nested_get(
                metrics,
                "active_delta",
                "target_count",
            ),
        },
        "artifacts": {
            "dashboard": "dashboard.html",
            "manifest": "manifest.json",
            "metrics": "metrics.json",
            "summary": "summary.csv",
            "target_comparison": "target_comparison.csv",
            "flat_diagnostics": "flat_diagnostics.csv",
            "household_diagnostics": "household_diagnostics.csv",
            "variable_summary": "variable_summary.csv",
            "worst_targets": "worst_targets.csv",
            "largest_household_regressions": "largest_household_regressions.csv",
            "largest_household_improvements": "largest_household_improvements.csv",
        },
    }


def render_calibration_dashboard(
    *,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    variable_summary: pd.DataFrame,
    worst_targets: pd.DataFrame,
    largest_regressions: pd.DataFrame,
    largest_improvements: pd.DataFrame,
) -> str:
    """Render a static HTML calibration dashboard."""
    version = html.escape(str(manifest["microplex_version"]))
    year = html.escape(str(manifest.get("year", "")))
    generated_at = html.escape(str(manifest.get("generated_at", "")))
    git_sha = html.escape(str(manifest.get("git_sha") or "unknown"))
    summary = metrics["summary"]
    active_delta = metrics["active_delta"]

    cards = [
        _metric_card(
            "Flat active MAE",
            _format_percent(_nested_get(summary, "flat", "active_mean_abs_post_error")),
        ),
        _metric_card(
            "Household active MAE",
            _format_percent(
                _nested_get(summary, "household", "active_mean_abs_post_error"),
            ),
        ),
        _metric_card(
            "Household vs flat delta",
            _format_percent(active_delta.get("mean_abs_post_error_delta")),
        ),
        _metric_card("Active targets", _format_number(active_delta["target_count"])),
        _metric_card(
            "Household better",
            _format_number(active_delta["household_better_targets"]),
        ),
        _metric_card("Flat better", _format_number(active_delta["flat_better_targets"])),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Microplex Calibration Dashboard - {version}</title>
  <style>
    body {{
      margin: 0;
      color: #1f2933;
      background: #f7f8fa;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid #d9dee6;
      padding: 24px 32px;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px 32px 48px;
    }}
    h1, h2 {{
      margin: 0;
      font-weight: 650;
    }}
    h1 {{
      font-size: 26px;
    }}
    h2 {{
      font-size: 18px;
      margin-top: 28px;
      margin-bottom: 12px;
    }}
    .meta {{
      margin-top: 8px;
      color: #52606d;
      font-size: 13px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d9dee6;
      border-radius: 8px;
      padding: 16px;
    }}
    .card-label {{
      color: #52606d;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card-value {{
      margin-top: 6px;
      font-size: 25px;
      font-weight: 650;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #d9dee6;
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e6e9ef;
      text-align: right;
      vertical-align: top;
    }}
    th {{
      background: #eef1f5;
      color: #323f4b;
      font-weight: 650;
    }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2) {{
      text-align: left;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Microplex Calibration Dashboard</h1>
    <div class="meta">
      Version {version} | Year {year} | Git {git_sha} | Generated {generated_at}
    </div>
  </header>
  <main>
    <section class="cards">
      {''.join(cards)}
    </section>
    {_table_section("Active Target Summary By Variable", variable_summary)}
    {_table_section("Worst Household Active Target Errors", worst_targets)}
    {_table_section("Largest Household Regressions Vs Flat", largest_regressions)}
    {_table_section("Largest Household Improvements Vs Flat", largest_improvements)}
  </main>
</body>
</html>
"""


def _active_target_comparison(target_comparison: pd.DataFrame) -> pd.DataFrame:
    if target_comparison.empty:
        return target_comparison.copy()
    return target_comparison[
        (target_comparison["flat_role"] == "active")
        & (target_comparison["household_role"] == "active")
    ].copy()


def _variable_summary(active_targets: pd.DataFrame) -> pd.DataFrame:
    if active_targets.empty:
        return pd.DataFrame(
            columns=[
                "variable",
                "target_count",
                "flat_mean_abs_post_error",
                "household_mean_abs_post_error",
                "mean_abs_post_error_delta",
                "household_better_targets",
                "flat_better_targets",
            ],
        )

    grouped = active_targets.groupby("variable", dropna=False)
    return grouped.agg(
        target_count=("target_index", "count"),
        flat_mean_abs_post_error=("flat_abs_post_error", "mean"),
        household_mean_abs_post_error=("household_abs_post_error", "mean"),
        mean_abs_post_error_delta=("abs_post_error_delta", "mean"),
        household_better_targets=("abs_post_error_delta", lambda s: int((s < 0).sum())),
        flat_better_targets=("abs_post_error_delta", lambda s: int((s > 0).sum())),
    ).reset_index()


def _worst_targets(active_targets: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    columns = _target_table_columns()
    if active_targets.empty:
        return pd.DataFrame(columns=columns)
    output = active_targets.copy()
    output["household_abs_post_error"] = output["household_abs_post_error"].abs()
    return output.sort_values("household_abs_post_error", ascending=False)[
        columns
    ].head(n)


def _largest_delta_targets(
    active_targets: pd.DataFrame,
    *,
    ascending: bool,
    n: int = 25,
) -> pd.DataFrame:
    columns = _target_table_columns()
    if active_targets.empty:
        return pd.DataFrame(columns=columns)
    return active_targets.sort_values(
        "abs_post_error_delta",
        ascending=ascending,
    )[columns].head(n)


def _target_table_columns() -> list[str]:
    return [
        "variable",
        "target_type",
        "stratum",
        "target_value",
        "flat_post_error",
        "household_post_error",
        "flat_abs_post_error",
        "household_abs_post_error",
        "abs_post_error_delta",
    ]


def _summary_to_nested_dict(summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    nested: dict[str, dict[str, Any]] = {}
    for _, row in summary.iterrows():
        group = str(row["group"])
        metric = str(row["metric"])
        nested.setdefault(group, {})[metric] = _json_safe(row["value"])
    return nested


def _role_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df:
        return {}
    return {
        str(role): int(count)
        for role, count in df[column].fillna("missing").value_counts().items()
    }


def _table_section(title: str, df: pd.DataFrame) -> str:
    if df.empty:
        table = "<p>No rows.</p>"
    else:
        table = df.to_html(
            index=False,
            classes="data-table",
            border=0,
            na_rep="",
            float_format=lambda value: f"{value:,.4f}",
        )
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<div class=\"table-wrap\">{table}</div></section>"
    )


def _metric_card(label: str, value: str) -> str:
    return (
        "<div class=\"card\">"
        f"<div class=\"card-label\">{html.escape(label)}</div>"
        f"<div class=\"card-value\">{html.escape(value)}</div>"
        "</div>"
    )


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.0f}"


def _nested_get(values: dict[str, Any], *keys: str) -> Any:
    current: Any = values
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value
