"""Bundle-level Chronicle build artifacts for downstream consumers."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from chronicle.dimension_labels import dimension_label_issues, dimension_labels_by_id
from chronicle.epoch import canonicalize_key, schema_id
from chronicle.source_package import (
    SOURCE_PACKAGE_ALIASES,
    assert_alias_map_covers_packages,
    validate_source_package,
)
from chronicle.suite import BuildSuiteReport, build_source_suite
from policyengine_chronicle.schema import (
    validate_consumer_fact_row_epochs,
)

BUNDLE_SCHEMA_VERSION = schema_id("bundle")
BUNDLE_COVERAGE_SCHEMA_VERSION = schema_id("bundle_coverage")
BUNDLE_SOURCES_SCHEMA_VERSION = schema_id("bundle_sources")
DEFAULT_BUNDLE_SOURCES = tuple(sorted(SOURCE_PACKAGE_ALIASES))
UK_BUNDLE_SOURCE_PREFIXES = (
    "desnz",
    "dfc_ni",
    "dfe",
    "dfi_ni",
    "dft",
    "dwp",
    "hmrc",
    "isc",
    "mhclg",
    "nisra",
    "nithc",
    "nrs",
    "obr",
    "ofgem",
    "ons",
    "orr",
    "scotgov",
    "slc",
    "voa",
    "welshgov",
)
UK_BUNDLE_SOURCES = (
    "desnz-monthly-annual-road-fuel-prices-august-2026",
    "desnz-need-england-wales-2023",
    "desnz-need-scotland-2023",
    "desnz-weekly-road-fuel-prices-september-2026",
    "dfc-ni-uc-statistics-may-2026",
    "dfe-funded-early-education-childcare-2026",
    "dfi-ni-public-transport-statistics-2024-25",
    "dft-bus0415-fares-index-2026",
    "dft-bus05i-revenue-support-2025",
    "dft-nts-vehicle-ownership-2024",
    "dft-nts0313-mode-use-frequency-2025",
    "dft-nts0621-local-bus-use-frequency-2025",
    "dft-nts0705-local-bus-trips-2024",
    "dft-veh1103-cars-fuel-type-2025",
    "dwp-benefit-cap-november-2025",
    "dwp-benefit-statistics-february-2026",
    "dwp-hb-claimants-client-type-tenure-accommodation-type-september-2025-february-2026",
    "dwp-hb-claimants-client-type-tenure-january-2023-february-2026",
    "dwp-pip-daily-living-foi-2025",
    "dwp-uc-childcare-element-march-2021-may-2026",
    "dwp-uc-deductions-march-2025-february-2026",
    "dwp-uc-households-by-constituency-children-may-2025",
    "dwp-uc-households-by-constituency-may-2025",
    "dwp-uc-households-by-local-authority-may-2025",
    "dwp-uc-households-carer-entitlement-payment-indicator-january-2023-may-2026",
    "dwp-uc-households-children-april-december-2025",
    "dwp-uc-households-children-child-entitlement-april-december-2025",
    "dwp-uc-households-children-payment-indicator-child-entitlement-april-2023-may-2026",
    "dwp-uc-households-family-type-april-december-2025",
    "dwp-uc-households-family-type-child-entitlement-april-december-2025",
    "dwp-uc-households-family-type-payment-indicator-april-december-2025",
    "dwp-uc-households-family-type-payment-indicator-child-entitlement-april-2023-may-2026",
    "dwp-uc-households-housing-entitlement-payment-indicator-january-2023-may-2026",
    "dwp-uc-households-housing-tenure-payment-indicator-january-2023-may-2026",
    "dwp-uc-households-lcw-entitlement-group-payment-indicator-january-2023-may-2026",
    "dwp-uc-households-lcw-entitlement-payment-indicator-january-2023-may-2026",
    "dwp-uc-payment-distribution-april-december-2025",
    "dwp-uc-people-employment-indicator-january-2023-may-2026",
    "dwp-uc-scotland-youngest-child-april-december-2025",
    "dwp-uc-two-child-limit-2025",
    "hmrc-cgt-age-2026",
    "hmrc-cgt-country-region-2026",
    "hmrc-cgt-gain-by-income-2026",
    "hmrc-cgt-size-of-gain-2026",
    "hmrc-cgt-statistics-2026",
    "hmrc-child-benefit-august-2025",
    "hmrc-hydrocarbon-oils-quantities-june-2026",
    "hmrc-salary-sacrifice-reform-2029-headcounts",
    "hmrc-salary-sacrifice-relief-2024-25",
    "hmrc-spi-income-bands-2023-24",
    "hmrc-spi-income-by-area-2023-24",
    "hmrc-tax-free-childcare-march-2026",
    "hmrc-vat-firm-sector-targets-2024-25",
    "hmrc-vat-firm-targets-2024-25",
    "isc-annual-census-2023",
    "isc-annual-census-2024",
    "mhclg-council-tax-collection-england-2025-26",
    "mhclg-council-tax-levels-england-2026-27",
    "mhclg-council-tax-levels-england-summary-2025-26",
    "mhclg-ehs-weekly-housing-costs-2023-24",
    "nisra-census2021-household-composition-country",
    "nisra-census2021-households-lgd",
    "nisra-census2021-households-pcon24",
    "nisra-census2021-tenure-lgd",
    "nisra-pcon24-population-by-age-2024",
    "nithc-annual-report-accounts-2024-25",
    "nrs-census2022-households-ukpc24",
    "nrs-census2022-uv113-household-composition-country",
    "nrs-census2022-uv404-tenure-council-area",
    "nrs-pcon24-population-by-age-2024",
    "obr-efo-aggregates-march-2026",
    "obr-efo-economy-march-2026",
    "obr-efo-expenditure-march-2026",
    "obr-efo-receipts-march-2026",
    "obr-fuel-duty-receipts-by-vehicle-april-2024",
    "ofgem-energy-price-cap-levels-2024-2026",
    "ofgem-energy-price-cap-q1-2024",
    "ons-census2021-ts003-household-composition-country",
    "ons-census2021-ts041-households-lad",
    "ons-census2021-ts041-households-pcon24",
    "ons-census2021-ts054-tenure-lad",
    "ons-consumer-trends-current-price-2026",
    "ons-families-households-2025",
    "ons-households-by-type-country-2025",
    "ons-lad-population-by-age-2024",
    "ons-mye-2023-england-regions",
    "ons-mye-2023-uk-countries",
    "ons-mye-2024-uk",
    "ons-national-balance-sheet-land-2025",
    "ons-pcon24-population-by-age-2024",
    "ons-pipr-private-rent-march-2026",
    "ons-pipr-rents-by-area-june-2026",
    "ons-public-sector-employment-2026",
    "ons-savings-interest-income",
    "ons-small-area-income-msoa-fye2023",
    "ons-subnational-dwellings-by-tenure-2024",
    "ons-uk-business-firm-sector-targets-2025",
    "ons-uk-business-firm-targets-2025",
    "ons-uk-population-projections-2024",
    "orr-government-support-7270-2024-25",
    "orr-government-support-7271-2024-25",
    "orr-rail-fares-7180-2026",
    "orr-rail-finance-7223-2024-25",
    "scotgov-band-d-council-tax-rates-2026-27",
    "scotgov-band-d-equivalents-2025",
    "scotgov-bus-coach-statistics-2023-24",
    "scotgov-bus-coach-statistics-2024-25",
    "scotgov-council-tax-bands-2025",
    "scotgov-council-tax-collection-2024-25",
    "scotgov-council-tax-collection-2025-26",
    "scotgov-scottish-budget-social-security-assistance-2026",
    "scotgov-slgfs-council-tax-2024-25",
    "slc-student-loan-borrower-forecasts-england-2025",
    "slc-student-loan-repayments-england-2025",
    "slc-student-loan-repayments-northern-ireland-2025",
    "slc-student-loan-repayments-scotland-2025",
    "slc-student-loan-repayments-wales-2025",
    "slc-student-support-england-2025",
    "voa-council-tax-bands-2025",
    "voa-council-tax-stock-by-lad-2025",
    "welshgov-bus-statistics-2024-25",
    "welshgov-council-tax-collection-2024-25",
    "welshgov-council-tax-collection-2025-26",
    "welshgov-council-tax-levels-2026-27",
    "welshgov-ctrs-annual-report-2024-25",
    "welshgov-ctrs-annual-report-2025-26",
    "welshgov-transport-revenue-outturn-2024-25",
)

_KEY_DOMAINS = {
    "aggregate_fact_key": "aggregate_fact",
    "semantic_fact_key": "semantic_fact",
    "legacy_fact_key": "fact",
    "source_release_key": "source_release",
    "source_series_key": "source_series",
    "observed_measure_key": "observed_measure",
    "dimension_set_key": "dimension_set",
    "universe_constraint_set_key": "universe_constraint_set",
}


def uk_bundle_sources_from_aliases() -> tuple[str, ...]:
    """Return UK-package aliases implied by the source-package directory prefixes."""
    return tuple(
        sorted(
            alias
            for alias, path in SOURCE_PACKAGE_ALIASES.items()
            if path.parts and path.parts[0] in UK_BUNDLE_SOURCE_PREFIXES
        )
    )


def assert_uk_bundle_sources_match_aliases() -> None:
    """Fail loudly if the curated UK suite omits a UK-prefixed alias."""
    expected = uk_bundle_sources_from_aliases()
    if UK_BUNDLE_SOURCES != expected:
        missing = sorted(set(expected) - set(UK_BUNDLE_SOURCES))
        extra = sorted(set(UK_BUNDLE_SOURCES) - set(expected))
        raise ValueError(
            "UK_BUNDLE_SOURCES drifted from UK-prefixed SOURCE_PACKAGE_ALIASES: "
            f"missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class BuildBundleIssue:
    """One bundle-level build issue."""

    code: str
    message: str
    source: str | None = None
    key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable issue."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class SkippedSourceReport:
    """A source package omitted from a default bundle for this year."""

    source: str
    reason: str
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


@dataclass(frozen=True)
class BundleSourceReport:
    """Summary for one source-package suite inside a bundle."""

    source: str
    suite_source: str
    output_dir: str
    valid: bool
    counts: dict[str, Any]
    outputs: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


@dataclass(frozen=True)
class BuildBundleReport:
    """End-to-end report for a merged Chronicle consumer bundle."""

    schema_version: str
    year: int
    output_dir: str
    outputs: dict[str, str]
    source_packages: tuple[BundleSourceReport, ...]
    skipped_sources: tuple[SkippedSourceReport, ...]
    coverage: dict[str, Any]
    errors: tuple[BuildBundleIssue, ...]
    warnings: tuple[BuildBundleIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether the bundle passed source-suite and duplicate checks."""
        return not self.errors and all(source.valid for source in self.source_packages)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        duplicates = self.coverage["duplicates"]
        return {
            "schema_version": self.schema_version,
            "valid": self.valid,
            "year": self.year,
            "output_dir": self.output_dir,
            "outputs": self.outputs,
            "counts": {
                "source_package_count": len(self.source_packages),
                "skipped_source_count": len(self.skipped_sources),
                "fact_count": self.coverage["fact_count"],
                "source_count": len(self.coverage["counts"]["by_source"]),
                "period_count": len(self.coverage["counts"]["by_period"]),
                "geography_count": len(self.coverage["counts"]["by_geography"]),
                "entity_count": len(self.coverage["counts"]["by_entity"]),
                "aggregate_duplicate_key_count": len(duplicates["aggregate_fact_keys"]),
                "semantic_duplicate_key_count": len(duplicates["semantic_fact_keys"]),
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
            },
            "source_packages": [source.to_dict() for source in self.source_packages],
            "skipped_sources": [skipped.to_dict() for skipped in self.skipped_sources],
            "coverage": self.coverage,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def build_bundle(
    output_dir: str | Path,
    *,
    year: int = 2023,
    sources: Sequence[str | Path] | None = None,
    axiom_command: Sequence[str] | None = None,
    axiom_roots: Sequence[str | Path] = (),
    require_axiom_validation: bool = False,
    replace: bool = False,
) -> BuildBundleReport:
    """Build source-package suites and merge their consumer-contract facts."""
    output_path = Path(output_dir)
    _prepare_output_dir(output_path, replace=replace)
    reports_path = output_path / "reports"
    sources_path = output_path / "sources"
    reports_path.mkdir(parents=True, exist_ok=True)
    sources_path.mkdir(parents=True, exist_ok=True)

    explicit_sources = sources is not None
    if not explicit_sources:
        # A default bundle must cover every packages/* directory. Fail loudly
        # if the alias map has drifted from the layout rather than silently
        # dropping unmapped packages (see PolicyEngine/chronicle#78).
        assert_alias_map_covers_packages()
    requested_sources = tuple(
        str(source) for source in (sources or DEFAULT_BUNDLE_SOURCES)
    )
    build_sources, skipped_sources = _resolve_bundle_sources(
        requested_sources,
        year=year,
        explicit=explicit_sources,
    )

    errors: list[BuildBundleIssue] = []
    warnings: list[BuildBundleIssue] = []
    source_reports: list[BundleSourceReport] = []
    consumer_rows: list[dict[str, Any]] = []
    uk_dimension_labels: dict[str, dict[str, list[str]]] = {}

    for source in build_sources:
        suite_dir = sources_path / _safe_source_dir_name(source)
        try:
            suite_report = build_source_suite(
                source,
                suite_dir,
                year=year,
                axiom_command=axiom_command,
                axiom_roots=axiom_roots,
                require_axiom_validation=require_axiom_validation,
                replace=True,
            )
        except Exception as exc:
            errors.append(
                BuildBundleIssue(
                    code="source_suite_build_failed",
                    message=str(exc),
                    source=source,
                )
            )
            continue

        if not suite_report.valid:
            errors.append(
                BuildBundleIssue(
                    code="source_suite_invalid",
                    message="Nested source-package suite did not pass validation.",
                    source=source,
                )
            )
        rows = _load_jsonl(Path(suite_report.outputs["consumer_facts"]))
        consumer_rows.extend(rows)
        source_reports.append(_bundle_source_report(source, suite_report))
        label_errors, label_warnings = _dimension_label_reports(source, rows)
        errors.extend(label_errors)
        warnings.extend(label_warnings)
        if source in UK_BUNDLE_SOURCES:
            for dimension_id, labels in dimension_labels_by_id(rows).items():
                for label in labels:
                    uk_dimension_labels.setdefault(dimension_id, {}).setdefault(
                        label, []
                    ).append(source)

    errors.extend(_cross_package_dimension_label_errors(uk_dimension_labels))
    aggregate_duplicates = _duplicate_key_reports(
        consumer_rows,
        "aggregate_fact_key",
    )
    semantic_duplicates = _duplicate_key_reports(
        consumer_rows,
        "semantic_fact_key",
    )
    for duplicate in aggregate_duplicates:
        errors.append(
            BuildBundleIssue(
                code="duplicate_aggregate_fact_key",
                message="Aggregate fact keys must be unique within a bundle.",
                key=duplicate["key"],
            )
        )
    if semantic_duplicates:
        warnings.append(
            BuildBundleIssue(
                code="duplicate_semantic_fact_key",
                message=(
                    "One or more semantic facts appear in multiple rows; "
                    "downstream consumers should reconcile or select sources."
                ),
            )
        )

    consumer_facts_path = output_path / "consumer_facts.jsonl"
    source_packages_path = output_path / "source_packages.json"
    coverage_path = output_path / "coverage.json"
    report_path = reports_path / "build_bundle.json"

    _write_jsonl(consumer_facts_path, consumer_rows)
    coverage = build_bundle_coverage(
        consumer_rows,
        aggregate_duplicates=aggregate_duplicates,
        semantic_duplicates=semantic_duplicates,
    )
    _write_report(coverage_path, coverage)
    _write_report(
        source_packages_path,
        {
            "schema_version": BUNDLE_SOURCES_SCHEMA_VERSION,
            "year": year,
            "source_package_count": len(source_reports),
            "skipped_source_count": len(skipped_sources),
            "source_packages": [
                source_report.to_dict() for source_report in source_reports
            ],
            "skipped_sources": [
                skipped_source.to_dict() for skipped_source in skipped_sources
            ],
        },
    )
    report = BuildBundleReport(
        schema_version=BUNDLE_SCHEMA_VERSION,
        year=year,
        output_dir=str(output_path),
        outputs={
            "consumer_facts": str(consumer_facts_path),
            "source_packages": str(source_packages_path),
            "coverage": str(coverage_path),
            "reports": str(reports_path),
            "build_bundle": str(report_path),
            "sources": str(sources_path),
        },
        source_packages=tuple(source_reports),
        skipped_sources=tuple(skipped_sources),
        coverage=coverage,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
    _write_report(report_path, report.to_dict())
    return report


def _dimension_label_reports(
    source: str,
    rows: list[dict[str, Any]],
) -> tuple[list[BuildBundleIssue], list[BuildBundleIssue]]:
    """Report one source's dimension-label issues (chronicle#261).

    UK packages must label every dimension and value, so their issues are
    errors, except publisher row labels that differ across the rows of one
    groupby value: Chronicle keeps that text as published and warns. Other
    packages carry labels wherever they declare or evidence them, unchecked.
    """
    if source not in UK_BUNDLE_SOURCES:
        return [], []
    errors: list[BuildBundleIssue] = []
    warnings: list[BuildBundleIssue] = []
    for issue in dimension_label_issues(rows):
        report = BuildBundleIssue(
            code=issue.code,
            message=issue.message,
            source=source,
            key=(
                issue.dimension_id
                if issue.value_id is None
                else f"{issue.dimension_id}={issue.value_id}"
            ),
        )
        if issue.code == "conflicting_groupby_value_label":
            warnings.append(report)
        else:
            errors.append(report)
    return errors, warnings


def _cross_package_dimension_label_errors(
    labels_by_dimension: dict[str, dict[str, list[str]]],
) -> list[BuildBundleIssue]:
    """Require one label per UK dimension id across packages.

    A consumer target can select facts from several packages that share a
    dimension id, such as ``geography``, and must see one name for it.
    """
    return [
        BuildBundleIssue(
            code="conflicting_dimension_label_across_packages",
            message=(
                f"UK packages label dimension {dimension_id!r} differently: "
                + "; ".join(
                    f"{label!r} in {sorted(sources)}"
                    for label, sources in sorted(labels.items())
                )
                + "."
            ),
            key=dimension_id,
        )
        for dimension_id, labels in sorted(labels_by_dimension.items())
        if len(labels) > 1
    ]


def build_bundle_coverage(
    rows: list[dict[str, Any]],
    *,
    aggregate_duplicates: list[dict[str, Any]] | None = None,
    semantic_duplicates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build source/concept/geography coverage diagnostics for consumer rows."""
    aggregate_duplicates = (
        aggregate_duplicates
        if aggregate_duplicates is not None
        else _duplicate_key_reports(rows, "aggregate_fact_key")
    )
    semantic_duplicates = (
        semantic_duplicates
        if semantic_duplicates is not None
        else _duplicate_key_reports(rows, "semantic_fact_key")
    )
    return {
        "schema_version": BUNDLE_COVERAGE_SCHEMA_VERSION,
        "fact_count": len(rows),
        "counts": {
            "by_source": _counts_by(rows, _source_name),
            "by_source_table": _counts_by(rows, _source_table_key),
            "by_period": _counts_by(rows, _period_key),
            "by_geography": _counts_by(rows, _geography_key),
            "by_entity": _counts_by(rows, _entity_key),
            "by_observed_measure": _counts_by(rows, _observed_measure_key),
            "by_observed_concept": _counts_by(rows, _observed_concept_key),
            "by_canonical_concept": _counts_by(rows, _canonical_concept_key),
        },
        "unique_counts": {
            "aggregate_fact_key": _unique_count(rows, "aggregate_fact_key"),
            "semantic_fact_key": _unique_count(rows, "semantic_fact_key"),
            "source_release_key": _unique_count(rows, "source_release_key"),
            "source_series_key": _unique_count(rows, "source_series_key"),
            "observed_measure_key": _unique_count(rows, "observed_measure_key"),
            "dimension_set_key": _unique_count(rows, "dimension_set_key"),
            "universe_constraint_set_key": _unique_count(
                rows,
                "universe_constraint_set_key",
            ),
        },
        "duplicates": {
            "aggregate_fact_keys": aggregate_duplicates,
            "semantic_fact_keys": semantic_duplicates,
        },
    }


def _resolve_bundle_sources(
    requested_sources: tuple[str, ...],
    *,
    year: int,
    explicit: bool,
) -> tuple[list[str], list[SkippedSourceReport]]:
    build_sources: list[str] = []
    skipped_sources: list[SkippedSourceReport] = []
    for source in requested_sources:
        report = validate_source_package(source, year=year)
        if report.valid or explicit or not _source_unavailable_for_year(report, year):
            build_sources.append(source)
            continue
        skipped_sources.append(
            SkippedSourceReport(
                source=source,
                reason="source package is not available for requested year",
                validation=report.to_dict(),
            )
        )
    return build_sources, skipped_sources


def _source_unavailable_for_year(report: Any, year: int) -> bool:
    unavailable_messages = {repr(str(year)), f"No source artifact for year {year}"}
    unavailable_codes = {
        "record_set_compile_failed",
        "source_artifact_unavailable",
    }
    return bool(report.errors) and all(
        error.code in unavailable_codes and error.message in unavailable_messages
        for error in report.errors
    )


def _bundle_source_report(
    source: str,
    suite_report: BuildSuiteReport,
) -> BundleSourceReport:
    payload = suite_report.to_dict()
    return BundleSourceReport(
        source=source,
        suite_source=suite_report.source,
        output_dir=suite_report.output_dir,
        valid=suite_report.valid,
        counts=payload["counts"],
        outputs=suite_report.outputs,
    )


def _duplicate_key_reports(
    rows: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_canonical_key(row[key], key), []).append(row)
    return [
        {
            "key": key_value,
            "count": len(key_rows),
            "sources": sorted({_source_table_key(row) for row in key_rows}),
            "legacy_fact_keys": sorted(
                {
                    _canonical_key(legacy_key, "legacy_fact_key")
                    for row in key_rows
                    if (legacy_key := row.get("legacy_fact_key"))
                },
                key=_identity_sort_key,
            ),
        }
        for key_value, key_rows in sorted(
            grouped.items(),
            key=lambda item: _identity_sort_key(item[0]),
        )
        if len(key_rows) > 1
    ]


def _counts_by(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str | None],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = key_fn(row)
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _unique_count(rows: list[dict[str, Any]], key: str) -> int:
    return len({_canonical_key(row[key], key) for row in rows if key in row})


def _canonical_key(value: Any, field_name: str) -> Any:
    """Return a stable identity for either accepted naming epoch."""

    domain_name = _KEY_DOMAINS.get(field_name)
    if domain_name is None or not isinstance(value, str):
        return value
    return canonicalize_key(domain_name, value)


def _identity_sort_key(value: Any) -> tuple[int, str]:
    """Sort identity scalars deterministically without comparing their types."""

    if isinstance(value, str):
        return (0, value)
    return (1, f"{type(value).__name__}:{value!r}")


def _source_name(row: dict[str, Any]) -> str | None:
    return row.get("source", {}).get("source_name")


def _source_table_key(row: dict[str, Any]) -> str | None:
    source = row.get("source", {})
    source_name = source.get("source_name")
    source_table = source.get("source_table")
    if not source_name or not source_table:
        return source_name or source_table
    return f"{source_name}:{source_table}"


def _period_key(row: dict[str, Any]) -> str | None:
    period = row.get("period", {})
    period_type = period.get("type")
    period_value = period.get("value")
    if period_type is None or period_value is None:
        return None
    return f"{period_type}:{period_value}"


def _geography_key(row: dict[str, Any]) -> str | None:
    geography = row.get("geography", {})
    level = geography.get("level")
    geography_id = geography.get("id")
    if level is None or geography_id is None:
        return None
    return f"{level}:{geography_id}"


def _entity_key(row: dict[str, Any]) -> str | None:
    return row.get("entity", {}).get("name")


def _observed_measure_key(row: dict[str, Any]) -> str | None:
    measure = row.get("observed_measure", {})
    source_name = measure.get("source_name")
    source_measure_id = measure.get("source_measure_id")
    if not source_name or not source_measure_id:
        return source_measure_id or source_name
    return f"{source_name}:{source_measure_id}"


def _observed_concept_key(row: dict[str, Any]) -> str | None:
    return row.get("observed_measure", {}).get("source_concept")


def _canonical_concept_key(row: dict[str, Any]) -> str | None:
    return row.get("concept_alignment", {}).get("canonical_concept")


def _safe_source_dir_name(source: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", source).strip("-")
    return safe or "source"


def _prepare_output_dir(output_path: Path, *, replace: bool) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        if not replace:
            raise FileExistsError(
                f"Build bundle output directory is not empty: {output_path}"
            )
        if output_path.resolve() in {Path("/").resolve(), Path.home().resolve()}:
            raise ValueError(
                f"Refusing to replace unsafe output directory: {output_path}"
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        row = json.loads(line)
        # Bundle assembly historically consumes suite output without applying
        # the stricter consumer-artifact schema. Keep that boundary intact,
        # while still rejecting identifiers outside the two accepted epochs.
        validate_consumer_fact_row_epochs(row, line_number, path)
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
