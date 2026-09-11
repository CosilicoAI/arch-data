"""Build and verify facts-only Chronicle consumer artifacts.

Chronicle publishes source-backed fact rows and the hashes needed to verify
them. Selection, measurement, period-alignment, and model-binding contracts
belong to consumers such as Microcosm.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chronicle.consumer_contract import _hash_key, _require_ledger_emit
from chronicle.core import (
    ALLOWED_ASSERTIONS,
    ALLOWED_PROVENANCE_CLASSES,
    DEFAULT_ASSERTION,
)
from chronicle.epoch import (
    CONSUMER_FACT_EMIT_SCHEMA_VERSION,
    EMIT_EPOCH,
    HASH_DOMAINS,
    SCHEMA_IDS,
    Epoch,
    canonicalize_key,
    hash_domain,
    schema_id,
)
from policyengine_chronicle.schema import (
    CONSUMER_FACT_SCHEMA_SHA256,
    CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION,
    normalize_consumer_fact_row_epochs,
    validate_consumer_fact_row,
)

CONSUMER_ARTIFACT_SCHEMA_VERSION = schema_id("consumer_artifact")


@dataclass(frozen=True)
class ConsumerArtifact:
    """A verified facts-only Chronicle consumer artifact."""

    path: Path
    manifest: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ConsumerArtifactBuildReport:
    """Build summary for one facts-only consumer artifact."""

    schema_version: str
    output_dir: str
    fact_row_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


def build_consumer_artifact(
    output_dir: str | Path,
    *,
    facts_path: str | Path,
    replace: bool = False,
    emit_epoch: Epoch | str = EMIT_EPOCH,
) -> ConsumerArtifactBuildReport:
    """Build a reproducible facts-only artifact from consumer fact rows.

    ``facts_path`` is a ``consumer_facts.jsonl`` file or a bundle directory
    containing one. Rows may carry either accepted naming epoch on each
    identifier (mechanism 1: dual-domain acceptance) and either row contract;
    the artifact writes every row with its keys canonicalized to
    ``emit_epoch`` and stamped with the emitted ``chronicle.consumer_fact.v2``
    contract, a superset of v1, so every row conforms to the schema whose
    sha256 the manifest pins. Target contracts are packaged by the consumer,
    not Chronicle.

    A refused call (unknown or non-Ledger ``emit_epoch``) raises before the
    output directory is touched, so an existing artifact survives it.
    """
    emit_epoch = _require_ledger_emit(emit_epoch, "build_consumer_artifact")
    resolved_facts_path = _resolve_facts_path(facts_path)
    rows = [
        {
            **normalize_consumer_fact_row_epochs(row, line_number, resolved_facts_path),
            "schema_version": CONSUMER_FACT_EMIT_SCHEMA_VERSION,
        }
        for line_number, row in enumerate(
            _load_consumer_rows(resolved_facts_path, validate_schema=True), start=1
        )
    ]

    output_path = Path(output_dir)
    if output_path.exists():
        if not replace:
            raise FileExistsError(
                f"Output directory exists: {output_path}. Pass replace=True."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    facts_out = output_path / "consumer_facts.jsonl"
    with facts_out.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")

    artifact_schema_version = schema_id("consumer_artifact", emit_epoch)
    manifest = {
        "schema_version": artifact_schema_version,
        "consumer_fact_schema_versions": sorted(
            {row.get("schema_version") for row in rows}
        ),
        "consumer_fact_schema_sha256": CONSUMER_FACT_SCHEMA_SHA256,
        "fact_row_count": len(rows),
        "facts_sha256": _sha256_file(facts_out),
    }
    _write_json(output_path / "manifest.json", manifest)

    return ConsumerArtifactBuildReport(
        schema_version=artifact_schema_version,
        output_dir=str(output_path),
        fact_row_count=len(rows),
    )


def load_consumer_artifact(path: str | Path) -> ConsumerArtifact:
    """Load a facts-only consumer artifact and verify its manifest hashes."""
    artifact_path = Path(path)
    manifest = json.loads((artifact_path / "manifest.json").read_text())
    manifest_schema_version = manifest.get("schema_version")
    try:
        SCHEMA_IDS["consumer_artifact"].infer_identifier_epoch(manifest_schema_version)
    except ValueError as error:
        raise ValueError(
            "Unsupported consumer artifact schema_version: "
            f"{manifest_schema_version!r}; accepted forms are "
            f"{SCHEMA_IDS['consumer_artifact'].ledger!r} and "
            f"{SCHEMA_IDS['consumer_artifact'].chronicle!r}."
        ) from error
    if "profiles" in manifest:
        raise ValueError(
            "Consumer artifact manifests must not contain profiles; target profiles "
            "are consumer-owned contracts and must be loaded by Microcosm."
        )
    declared_fact_schema_versions = manifest.get("consumer_fact_schema_versions")
    if declared_fact_schema_versions is not None:
        if not isinstance(declared_fact_schema_versions, list):
            raise ValueError(
                "Consumer artifact consumer_fact_schema_versions must be a list."
            )
        for row_schema_version in declared_fact_schema_versions:
            try:
                SCHEMA_IDS["consumer_fact"].infer_identifier_epoch(row_schema_version)
            except ValueError as error:
                raise ValueError(
                    "Unsupported consumer fact schema_version in artifact "
                    f"manifest: {row_schema_version!r}; accepted forms are "
                    f"{SCHEMA_IDS['consumer_fact'].ledger!r} and "
                    f"{SCHEMA_IDS['consumer_fact'].chronicle!r}."
                ) from error
    manifest_schema_sha256 = manifest.get("consumer_fact_schema_sha256")
    if manifest_schema_sha256 is not None:
        pinned_versions = sorted(
            version
            for version, sha256 in CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION.items()
            if sha256 == manifest_schema_sha256
        )
        if not pinned_versions:
            raise ValueError(
                "Consumer artifact declares consumer_fact_schema_sha256 "
                f"{manifest_schema_sha256!r}, which does not match the packaged "
                "consumer-fact schema of either contract "
                f"{dict(CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION)!r}."
            )
        if (
            declared_fact_schema_versions is not None
            and declared_fact_schema_versions != pinned_versions
        ):
            raise ValueError(
                "Consumer artifact declares consumer_fact_schema_versions "
                f"{declared_fact_schema_versions!r} but pins the schema sha256 "
                f"of {pinned_versions[0]!r}."
            )

    facts_file = artifact_path / "consumer_facts.jsonl"
    actual_sha256 = _sha256_file(facts_file)
    if actual_sha256 != manifest["facts_sha256"]:
        raise ValueError(
            "Consumer artifact fact rows do not match the manifest hash: "
            f"{actual_sha256} != {manifest['facts_sha256']}."
        )
    rows = _load_consumer_rows(facts_file, validate_schema=True)
    actual_fact_schema_versions = sorted({row.get("schema_version") for row in rows})
    if (
        declared_fact_schema_versions is not None
        and declared_fact_schema_versions != actual_fact_schema_versions
    ):
        raise ValueError(
            "Consumer artifact manifest declares consumer_fact_schema_versions "
            f"{declared_fact_schema_versions!r} but its rows use "
            f"{actual_fact_schema_versions!r}."
        )
    declared_row_count = manifest.get("fact_row_count")
    if declared_row_count is not None and declared_row_count != len(rows):
        raise ValueError(
            "Consumer artifact manifest declares fact_row_count "
            f"{declared_row_count} but the feed carries {len(rows)} rows."
        )
    return ConsumerArtifact(
        path=artifact_path,
        manifest=manifest,
        rows=tuple(rows),
    )


def _resolve_facts_path(facts_path: str | Path) -> Path:
    path = Path(facts_path)
    if path.is_dir():
        candidate = path / "consumer_facts.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(
                f"No consumer_facts.jsonl in bundle directory {path}."
            )
        return candidate
    if not path.exists():
        raise FileNotFoundError(f"No consumer facts file at {path}.")
    return path


def _reject_non_finite(value: Any) -> Any:
    raise ValueError(f"Consumer fact contains a non-finite JSON number: {value!r}.")


def _assert_finite_numbers(value: Any, *, line_number: int, path: Path) -> None:
    """Reject non-finite numbers, including those in nested structures."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"Row {line_number} of {path} contains a non-finite number: {value!r}."
        )
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_numbers(item, line_number=line_number, path=path)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_numbers(item, line_number=line_number, path=path)


def _recompute_aggregate_fact_key(row: dict[str, Any]) -> str:
    """Recompute the aggregate fact key from the row's content."""
    declared_key = row.get("aggregate_fact_key")
    epoch = HASH_DOMAINS["aggregate_fact"].infer_key_epoch(declared_key)
    assertion = row.get("assertion")
    payload = {
        "source_release_key": canonicalize_key(
            "source_release", row.get("source_release_key")
        ),
        "source_series_key": canonicalize_key(
            "source_series", row.get("source_series_key")
        ),
        "observed_measure_key": canonicalize_key(
            "observed_measure", row.get("observed_measure_key")
        ),
        "aggregation": row.get("aggregation"),
        "period": row.get("period"),
        "geography": row.get("geography"),
        "entity": row.get("entity"),
        "dimension_set_key": canonicalize_key(
            "dimension_set", row.get("dimension_set_key")
        ),
        "universe_constraint_set_key": canonicalize_key(
            "universe_constraint_set", row.get("universe_constraint_set_key")
        ),
        "assertion": None if assertion == DEFAULT_ASSERTION else assertion,
    }
    return _hash_key(hash_domain("aggregate_fact", epoch), payload)


def _load_consumer_rows(
    path: Path,
    *,
    validate_schema: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=_reject_non_finite)
            _assert_finite_numbers(row, line_number=line_number, path=path)
            if validate_schema:
                validate_consumer_fact_row(row, line_number, path)
            _validate_consumer_row_provenance(row, line_number=line_number, path=path)
            assertion = row.setdefault("assertion", DEFAULT_ASSERTION)
            if assertion not in ALLOWED_ASSERTIONS:
                raise ValueError(
                    f"Row {line_number} of {path} has unsupported assertion "
                    f"{assertion!r}."
                )
            key = row.get("aggregate_fact_key")
            if validate_schema:
                recomputed = _recompute_aggregate_fact_key(row)
                if key != recomputed:
                    raise ValueError(
                        f"Row {line_number} of {path} declares aggregate_fact_key "
                        f"{key!r} but its content hashes to {recomputed!r}; the "
                        "identity key does not match the row."
                    )
            canonical_key = canonicalize_key("aggregate_fact", key)
            if canonical_key in seen_keys:
                raise ValueError(
                    f"Row {line_number} of {path} repeats aggregate_fact_key "
                    f"{key!r}; consumer artifact fact rows must be unique."
                )
            seen_keys.add(canonical_key)
            rows.append(row)
    return rows


def load_consumer_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load and verify consumer-fact JSONL rows from either accepted epoch."""

    return tuple(_load_consumer_rows(Path(path), validate_schema=True))


def _validate_consumer_row_provenance(
    row: Mapping[str, Any],
    *,
    line_number: int,
    path: Path,
) -> None:
    if "provenance_class" not in row:
        raise ValueError(
            f"Row {line_number} of {path} is missing required provenance_class."
        )
    provenance_class = row["provenance_class"]
    if type(provenance_class) is not str or (
        provenance_class not in ALLOWED_PROVENANCE_CLASSES
    ):
        raise ValueError(
            f"Row {line_number} of {path} has unsupported provenance_class "
            f"{provenance_class!r}."
        )
    has_survey_instrument = "survey_instrument" in row
    survey_instrument = row.get("survey_instrument")
    if provenance_class == "survey_aggregate":
        if type(survey_instrument) is not str or not survey_instrument.strip():
            raise ValueError(
                f"Row {line_number} of {path} needs a non-empty "
                "survey_instrument for survey_aggregate provenance."
            )
    elif has_survey_instrument:
        raise ValueError(
            f"Row {line_number} of {path} has survey_instrument outside "
            "survey_aggregate provenance."
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CONSUMER_ARTIFACT_SCHEMA_VERSION",
    "ConsumerArtifact",
    "ConsumerArtifactBuildReport",
    "build_consumer_artifact",
    "load_consumer_artifact",
    "load_consumer_rows",
]
