"""Consumer-fact row schema validation for Chronicle artifacts.

Both consumer-fact row schemas are packaged with the wheel so builds and loads
validate every fact row against the exact contract it claims: the frozen
``consumer_fact.v1`` schema for ``ledger.consumer_fact.v1`` rows, and the
``consumer_fact.v2`` schema, v1 plus optional dimension and value labels
(chronicle#261), for ``chronicle.consumer_fact.v2`` rows. The packaged schema
bytes are the single source of truth: each artifact manifest records the sha256
of the schema its rows use, and a load rejects any manifest that claims a
different one.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from functools import lru_cache
from importlib.resources import files as _resource_files
from types import MappingProxyType
from typing import Any

from chronicle.epoch import (
    CONSUMER_FACT_EMIT_SCHEMA_VERSION,
    SCHEMA_IDS,
    canonicalize_key,
)
from jsonschema import Draft202012Validator

_SCHEMA_PACKAGE = "policyengine_chronicle.schemas"
_SCHEMA_RESOURCES = MappingProxyType(
    {
        SCHEMA_IDS["consumer_fact"].ledger: "consumer_fact.v1.schema.json",
        SCHEMA_IDS["consumer_fact"].chronicle: "consumer_fact.v2.schema.json",
    }
)

_TOP_LEVEL_KEY_DOMAINS = {
    "aggregate_fact_key": "aggregate_fact",
    "semantic_fact_key": "semantic_fact",
    "legacy_fact_key": "fact",
    "source_release_key": "source_release",
    "source_series_key": "source_series",
    "observed_measure_key": "observed_measure",
    "dimension_set_key": "dimension_set",
    "universe_constraint_set_key": "universe_constraint_set",
}


def _packaged_schema_bytes(schema_version: str) -> bytes:
    resource = _SCHEMA_RESOURCES[schema_version]
    return _resource_files(_SCHEMA_PACKAGE).joinpath(resource).read_bytes()


CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION = MappingProxyType(
    {
        schema_version: hashlib.sha256(
            _packaged_schema_bytes(schema_version)
        ).hexdigest()
        for schema_version in _SCHEMA_RESOURCES
    }
)
#: The schema Chronicle emits consumer-fact rows under and artifacts pin.
CONSUMER_FACT_SCHEMA_SHA256 = CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION[
    CONSUMER_FACT_EMIT_SCHEMA_VERSION
]


@lru_cache(maxsize=None)
def consumer_fact_schema(
    schema_version: str = CONSUMER_FACT_EMIT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return the parsed, cached row schema of one consumer-fact contract."""
    return json.loads(_packaged_schema_bytes(schema_version))


@lru_cache(maxsize=None)
def _validator(schema_version: str) -> Draft202012Validator:
    return Draft202012Validator(consumer_fact_schema(schema_version))


def _epoch_validation_error(
    *,
    line_number: int,
    path: Any,
    location: str,
    error: ValueError,
) -> ValueError:
    return ValueError(
        f"Consumer fact row {line_number} of {path} failed epoch validation "
        f"at {location!r}: {error}"
    )


def _normalize_key(
    key: Any,
    *,
    domain_name: str,
    line_number: int,
    path: Any,
    location: str,
) -> Any:
    # Leave type errors to the frozen JSON schema so its established messages
    # stay stable. String identifiers receive the additive epoch check first.
    if not isinstance(key, str):
        return key
    try:
        return canonicalize_key(domain_name, key)
    except ValueError as error:
        raise _epoch_validation_error(
            line_number=line_number,
            path=path,
            location=location,
            error=error,
        ) from error


def _check_schema_version(row: dict[str, Any], line_number: int, path: Any) -> None:
    schema_version = row.get("schema_version")
    if not isinstance(schema_version, str):
        return
    try:
        SCHEMA_IDS["consumer_fact"].infer_identifier_epoch(schema_version)
    except ValueError as error:
        raise _epoch_validation_error(
            line_number=line_number,
            path=path,
            location="schema_version",
            error=error,
        ) from error


def normalize_consumer_fact_row_epochs(
    row: Any,
    line_number: int,
    path: Any,
) -> Any:
    """Return a copy of a consumer-fact row with every key in its Ledger form.

    This adapter accepts either registered naming epoch on each identifier,
    independently, and normalizes only the copy it returns. The caller's row is
    never mutated, and mixed-epoch rows remain valid. ``schema_version`` must be
    an accepted row contract but is kept as it is: ``ledger.consumer_fact.v1``
    and ``chronicle.consumer_fact.v2`` are distinct contracts (v2 adds optional
    labels), so each row validates against its own schema.
    """

    normalized = deepcopy(row)
    if not isinstance(normalized, dict):
        return normalized

    _check_schema_version(normalized, line_number, path)

    for field_name, domain_name in _TOP_LEVEL_KEY_DOMAINS.items():
        if field_name in normalized:
            normalized[field_name] = _normalize_key(
                normalized[field_name],
                domain_name=domain_name,
                line_number=line_number,
                path=path,
                location=field_name,
            )

    concept_alignment = normalized.get("concept_alignment")
    if (
        isinstance(concept_alignment, dict)
        and "concept_alignment_key" in concept_alignment
    ):
        concept_alignment["concept_alignment_key"] = _normalize_key(
            concept_alignment["concept_alignment_key"],
            domain_name="concept_alignment",
            line_number=line_number,
            path=path,
            location="concept_alignment/concept_alignment_key",
        )

    lineage = normalized.get("lineage")
    if isinstance(lineage, dict):
        for field_name, domain_name in (
            ("source_cell_keys", "source_cell"),
            ("source_row_keys", "source_row"),
        ):
            keys = lineage.get(field_name)
            if not isinstance(keys, list):
                continue
            for index, key in enumerate(keys):
                keys[index] = _normalize_key(
                    key,
                    domain_name=domain_name,
                    line_number=line_number,
                    path=path,
                    location=f"lineage/{field_name}/{index}",
                )

    return normalized


def validate_consumer_fact_row_epochs(
    row: Any,
    line_number: int,
    path: Any,
) -> None:
    """Check every epoch-bearing identifier of a row without copying it.

    This is :func:`normalize_consumer_fact_row_epochs` minus the deep copy and
    the rewrite: it raises the same error for the same identifier and leaves
    the caller's row untouched, so a bundle build can validate ~150k rows
    without materializing a canonical copy of each.
    """

    if not isinstance(row, dict):
        return
    _check_schema_version(row, line_number, path)
    for field_name, domain_name in _TOP_LEVEL_KEY_DOMAINS.items():
        if field_name in row:
            _normalize_key(
                row[field_name],
                domain_name=domain_name,
                line_number=line_number,
                path=path,
                location=field_name,
            )
    concept_alignment = row.get("concept_alignment")
    if (
        isinstance(concept_alignment, dict)
        and "concept_alignment_key" in concept_alignment
    ):
        _normalize_key(
            concept_alignment["concept_alignment_key"],
            domain_name="concept_alignment",
            line_number=line_number,
            path=path,
            location="concept_alignment/concept_alignment_key",
        )
    lineage = row.get("lineage")
    if isinstance(lineage, dict):
        for field_name, domain_name in (
            ("source_cell_keys", "source_cell"),
            ("source_row_keys", "source_row"),
        ):
            keys = lineage.get(field_name)
            if not isinstance(keys, list):
                continue
            for index, key in enumerate(keys):
                _normalize_key(
                    key,
                    domain_name=domain_name,
                    line_number=line_number,
                    path=path,
                    location=f"lineage/{field_name}/{index}",
                )


def validate_consumer_fact_row(
    row: Any,
    line_number: int,
    path: Any,
) -> None:
    """Validate one consumer-fact row against the schema of its contract.

    Raises :class:`ValueError` naming the source ``path``, the 1-based
    ``line_number``, the failing JSON location, and the schema reason. The
    first error by schema location is reported so the message is stable. A row
    without a recognizable ``schema_version`` is checked against the emitted
    contract, which reports the missing or malformed id.
    """
    normalized = normalize_consumer_fact_row_epochs(row, line_number, path)
    schema_version = (
        normalized.get("schema_version") if isinstance(normalized, dict) else None
    )
    if not isinstance(schema_version, str) or schema_version not in _SCHEMA_RESOURCES:
        schema_version = CONSUMER_FACT_EMIT_SCHEMA_VERSION
    errors = sorted(
        _validator(schema_version).iter_errors(normalized),
        key=lambda error: (
            [str(part) for part in error.absolute_path],
            error.message,
        ),
    )
    if not errors:
        return
    error = errors[0]
    location = "/".join(str(part) for part in error.absolute_path) or "<root>"
    raise ValueError(
        f"Consumer fact row {line_number} of {path} failed schema validation "
        f"at {location!r}: {error.message}"
    )


__all__ = [
    "CONSUMER_FACT_SCHEMA_SHA256",
    "CONSUMER_FACT_SCHEMA_SHA256_BY_VERSION",
    "consumer_fact_schema",
    "normalize_consumer_fact_row_epochs",
    "validate_consumer_fact_row",
    "validate_consumer_fact_row_epochs",
]
