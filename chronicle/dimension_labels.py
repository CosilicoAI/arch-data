"""Chronicle-owned labels for fact dimensions and dimension values.

Microcosm's calibration hierarchy (microcosm#855) names every dimension and
dimension value a target inherits from the Chronicle facts it selects, and it
takes those names only from Chronicle (chronicle#261). Every fact therefore
carries ``dimension_labels`` (dimension id to label) and
``dimension_value_labels`` (dimension id to value id to label) covering its
``filters`` and, when it has one, its ``layout.groupby_dimension``, whose label
is also stamped on ``layout.groupby_dimension_label``.

Labels resolve per package; the first source that has one wins.

Dimension ids:

1. the package's ``dimension_labels`` declaration;
2. the publisher's field label, for a Stat-Xplore field the parser read;
3. the label on the package's ``==`` constraints for that filter, when they
   agree on one label that is not the identifier itself.

Dimension values:

1. the package's ``dimension_value_labels`` declaration;
2. the fact's publisher row label (``layout.groupby_value_label``) when the
   dimension is the record set's groupby axis (the same id, or the same leaf
   name source-filter canonicalization matches on) and the value is the row's
   value id;
3. the publisher's text in the matching column of the fact's source row, when
   the row-semantic acceptance matches the filter to it (for ``all``, the
   publisher's aggregate item, such as Stat-Xplore's ``Total``);
4. a number or an ISO date, which labels itself.

Everything else must be declared. :func:`dimension_label_issues` checks rows
the way Microcosm reads them and reports what is missing or labelled two ways.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from typing import Any

from chronicle.consumer_contract import _concept_leaf
from chronicle.core import AggregateFact, SourceRecordLayout
from chronicle.epoch import Epoch, canonicalize_key
from chronicle.sources.rows import SourceRow, build_source_row_key

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def dimension_value_id(value: Any) -> str:
    """Return the ``dimension_value_labels`` key of one dimension value.

    Booleans are lower-cased, lists and mappings JSON-serialized with sorted
    keys and no spaces, and anything else is written as ``str()`` of its JSON
    value: the form Microcosm's calibration hierarchy looks labels up by.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def label_facts(
    facts: Iterable[AggregateFact],
    *,
    dimension_labels: Mapping[str, str] | None = None,
    dimension_value_labels: Mapping[str, Mapping[str, str]] | None = None,
    field_labels: Mapping[str, str] | None = None,
    source_rows: Iterable[SourceRow] = (),
) -> list[AggregateFact]:
    """Return one package's facts carrying the labels it declares or evidences.

    ``dimension_labels`` and ``dimension_value_labels`` are the package's
    declarations, ``field_labels`` the publisher's labels for parsed source-row
    columns, and ``source_rows`` the parsed rows the facts' lineage points at.
    A dimension or value no source labels is left unlabelled for
    :func:`dimension_label_issues` to report.
    """
    facts = list(facts)
    declared = dict(dimension_labels or {})
    declared_values = {
        dimension_id: dict(value_labels)
        for dimension_id, value_labels in (dimension_value_labels or {}).items()
    }
    publisher_field_labels = dict(field_labels or {})
    constraint_labels = _constraint_labels(facts)
    rows_by_key = {
        build_source_row_key(row, epoch=Epoch.LEDGER): row for row in source_rows
    }

    labelled = []
    for fact in facts:
        dimensions = _fact_dimensions(fact)
        layout = fact.layout
        groupby = layout.groupby_dimension if layout else None
        labels: dict[str, str] = {}
        for dimension_id in (*dimensions, *([groupby] if groupby else [])):
            label = (
                declared.get(dimension_id)
                or publisher_field_labels.get(dimension_id)
                or constraint_labels.get(dimension_id)
            )
            if label:
                labels[dimension_id] = label

        rows = [
            rows_by_key[key]
            for key in (
                canonicalize_key("source_row", key) for key in fact.source_row_keys
            )
            if key in rows_by_key
        ]
        value_labels: dict[str, dict[str, str]] = defaultdict(dict)
        for dimension_id, value in dimensions.items():
            value_id = dimension_value_id(value)
            label = (
                declared_values.get(dimension_id, {}).get(value_id)
                or _groupby_row_label(layout, dimension_id, value_id)
                or _source_row_label(rows, dimension_id, value)
                or _self_label(value)
            )
            if label:
                value_labels[dimension_id][value_id] = label
        if (
            groupby
            and groupby not in dimensions
            and layout.groupby_value_id is not None
        ):
            value_id = dimension_value_id(layout.groupby_value_id)
            label = declared_values.get(groupby, {}).get(value_id) or (
                _groupby_row_label(layout, groupby, value_id)
            )
            if label:
                value_labels[groupby][value_id] = label

        labelled.append(
            replace(
                fact,
                dimension_labels=labels,
                dimension_value_labels=dict(value_labels),
                layout=(
                    replace(layout, groupby_dimension_label=labels.get(groupby))
                    if layout
                    else layout
                ),
            )
        )
    return labelled


def unused_label_declarations(
    facts: Iterable[AggregateFact],
    *,
    dimension_labels: Mapping[str, str] | None = None,
    dimension_value_labels: Mapping[str, Mapping[str, str]] | None = None,
) -> list[str]:
    """Return the declarations no fact of the package uses, as dotted paths."""
    used_dimensions: set[str] = set()
    used_values: set[tuple[str, str]] = set()
    for fact in facts:
        for dimension_id, value in _fact_dimensions(fact).items():
            used_dimensions.add(dimension_id)
            used_values.add((dimension_id, dimension_value_id(value)))
        layout = fact.layout
        if layout and layout.groupby_dimension:
            used_dimensions.add(layout.groupby_dimension)
            if layout.groupby_value_id is not None:
                used_values.add(
                    (
                        layout.groupby_dimension,
                        dimension_value_id(layout.groupby_value_id),
                    )
                )
    unused = [
        f"dimension_labels.{dimension_id}"
        for dimension_id in (dimension_labels or {})
        if dimension_id not in used_dimensions
    ]
    unused.extend(
        f"dimension_value_labels.{dimension_id}.{value_id}"
        for dimension_id, value_labels in (dimension_value_labels or {}).items()
        for value_id in value_labels
        if (dimension_id, value_id) not in used_values
    )
    return unused


@dataclass(frozen=True)
class DimensionLabelIssue:
    """A dimension or value that is unlabelled, or labelled two ways."""

    code: str
    dimension_id: str
    value_id: str | None = None
    labels: tuple[str, ...] = ()
    fact_count: int = 0
    example_fact_key: str | None = None

    @property
    def message(self) -> str:
        target = f"dimension {self.dimension_id!r}"
        if self.value_id is not None:
            target = f"{target} value {self.value_id!r}"
        if self.code == "conflicting_groupby_value_label":
            return (
                f"{target} carries publisher row labels that differ across "
                f"facts: {list(self.labels)!r}; a consumer selecting facts "
                "from more than one of those rows sees conflicting labels."
            )
        if self.code.startswith("conflicting"):
            return f"{target} carries more than one label: {list(self.labels)!r}."
        return (
            f"{target} has no Chronicle label on {self.fact_count} fact(s); "
            "declare it in the package's dimension_labels or "
            "dimension_value_labels."
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable issue."""
        payload = {
            key: value
            for key, value in asdict(self).items()
            if value not in (None, (), [], 0)
        }
        if self.labels:
            payload["labels"] = list(self.labels)
        payload["message"] = self.message
        return payload


def dimension_label_issues(
    rows: Iterable[Mapping[str, Any]],
) -> list[DimensionLabelIssue]:
    """Check consumer-fact rows the way Microcosm's calibration hierarchy reads them.

    Each row's dimensions (its ``dimensions`` plus ``layout.groupby_dimension``
    folded in at ``layout.groupby_value_id`` when absent) each need exactly one
    non-empty label, read from ``dimension_labels`` or, for the groupby
    dimension, ``layout.groupby_dimension_label``; each value likewise, from
    ``dimension_value_labels`` or, for the groupby value,
    ``layout.groupby_value_label``. Across *rows*, every dimension id and every
    value must carry a single label. Values that only ever appear as a groupby
    value, whose publisher row labels differ across rows, are reported
    separately as ``conflicting_groupby_value_label``: that text is the
    publisher's, kept as published. Issues are aggregated per dimension or
    value, with a fact count and an example fact key.
    """
    missing: dict[tuple[str, str, str | None], list[Any]] = {}
    row_conflicts: dict[tuple[str, str | None], set[str]] = defaultdict(set)
    dimension_labels_seen: dict[str, set[str]] = defaultdict(set)
    value_labels_seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    plain_values: set[tuple[str, str]] = set()

    for row in rows:
        fact_key = row.get("aggregate_fact_key")
        layout = row.get("layout") or {}
        groupby = str(layout.get("groupby_dimension") or "").strip()
        groupby_value_id = str(layout.get("groupby_value_id") or "").strip()
        dimensions = {
            str(key): value for key, value in (row.get("dimensions") or {}).items()
        }
        if groupby and groupby_value_id:
            dimensions.setdefault(groupby, groupby_value_id)
        row_dimension_labels = row.get("dimension_labels") or {}
        row_value_labels = row.get("dimension_value_labels") or {}

        for dimension_id, value in dimensions.items():
            labels = _nonempty_labels(
                row_dimension_labels.get(dimension_id),
                (
                    layout.get("groupby_dimension_label")
                    if dimension_id == groupby
                    else None
                ),
            )
            if not labels:
                _count(
                    missing, ("missing_dimension_label", dimension_id, None), fact_key
                )
            elif len(labels) > 1:
                row_conflicts[(dimension_id, None)].update(labels)
            else:
                dimension_labels_seen[dimension_id].update(labels)

            value_id = dimension_value_id(value)
            groupby_value = dimension_id == groupby and value_id == groupby_value_id
            if not groupby_value:
                plain_values.add((dimension_id, value_id))
            value_labels = _nonempty_labels(
                (row_value_labels.get(dimension_id) or {}).get(value_id),
                layout.get("groupby_value_label") if groupby_value else None,
            )
            if not value_labels:
                _count(
                    missing,
                    ("missing_dimension_value_label", dimension_id, value_id),
                    fact_key,
                )
            elif len(value_labels) > 1:
                row_conflicts[(dimension_id, value_id)].update(value_labels)
            else:
                value_labels_seen[(dimension_id, value_id)].update(value_labels)

    issues = [
        DimensionLabelIssue(
            code=code,
            dimension_id=dimension_id,
            value_id=value_id,
            fact_count=count,
            example_fact_key=example,
        )
        for (code, dimension_id, value_id), (count, example) in missing.items()
    ]
    for (dimension_id, value_id), labels in row_conflicts.items():
        issues.append(
            DimensionLabelIssue(
                code=(
                    "conflicting_dimension_label"
                    if value_id is None
                    else "conflicting_dimension_value_label"
                ),
                dimension_id=dimension_id,
                value_id=value_id,
                labels=tuple(sorted(labels)),
            )
        )
    for dimension_id, labels in dimension_labels_seen.items():
        if len(labels) > 1 and (dimension_id, None) not in row_conflicts:
            issues.append(
                DimensionLabelIssue(
                    code="conflicting_dimension_label",
                    dimension_id=dimension_id,
                    labels=tuple(sorted(labels)),
                )
            )
    for (dimension_id, value_id), labels in value_labels_seen.items():
        if len(labels) > 1 and (dimension_id, value_id) not in row_conflicts:
            issues.append(
                DimensionLabelIssue(
                    code=(
                        "conflicting_dimension_value_label"
                        if (dimension_id, value_id) in plain_values
                        else "conflicting_groupby_value_label"
                    ),
                    dimension_id=dimension_id,
                    value_id=value_id,
                    labels=tuple(sorted(labels)),
                )
            )
    return sorted(
        issues,
        key=lambda issue: (issue.code, issue.dimension_id, issue.value_id or ""),
    )


def dimension_labels_by_id(rows: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Return the labels each dimension id carries across consumer-fact rows."""
    found: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for dimension_id, label in (row.get("dimension_labels") or {}).items():
            found[str(dimension_id)].update(_nonempty_labels(label))
        layout = row.get("layout") or {}
        groupby = str(layout.get("groupby_dimension") or "").strip()
        if groupby:
            found[groupby].update(
                _nonempty_labels(layout.get("groupby_dimension_label"))
            )
    return {dimension_id: labels for dimension_id, labels in found.items() if labels}


def _fact_dimensions(fact: AggregateFact) -> dict[str, Any]:
    return {key: value for key, value in fact.filters.items() if value is not None}


def _constraint_labels(facts: list[AggregateFact]) -> dict[str, str]:
    """Return the one label a package's ``==`` constraints give each filter."""
    found: dict[str, set[str]] = defaultdict(set)
    for fact in facts:
        for constraint in fact.constraints:
            if constraint.operator != "==" or constraint.variable not in fact.filters:
                continue
            label = (constraint.label or "").strip()
            if label and label not in {
                constraint.variable,
                _concept_leaf(constraint.variable),
            }:
                found[constraint.variable].add(label)
    return {
        variable: next(iter(labels))
        for variable, labels in found.items()
        if len(labels) == 1
    }


def _groupby_row_label(
    layout: SourceRecordLayout | None,
    dimension_id: str,
    value_id: str,
) -> str | None:
    if layout is None or not layout.groupby_dimension:
        return None
    if layout.groupby_value_id is None:
        return None
    if dimension_value_id(layout.groupby_value_id) != value_id:
        return None
    groupby = layout.groupby_dimension
    if dimension_id != groupby and _concept_leaf(dimension_id) != _concept_leaf(
        groupby
    ):
        return None
    return (layout.groupby_value_label or "").strip() or None


def _source_row_label(
    rows: list[SourceRow],
    dimension_id: str,
    value: Any,
) -> str | None:
    if not rows:
        return None
    # Imported here: chronicle.suite imports chronicle.source_package, which
    # labels its facts through this module.
    from chronicle.suite import _filter_value_matches_source_value, _source_row_value

    matched = []
    for row in rows:
        found, source_value = _source_row_value(row, dimension_id)
        if found:
            matched.append(source_value)
    if not matched or any(source_value is None for source_value in matched):
        return None
    texts = {str(source_value).strip() for source_value in matched}
    if len(texts) != 1 or "" in texts:
        return None
    if value != "all" and not all(
        _filter_value_matches_source_value(dimension_id, value, source_value)
        for source_value in matched
    ):
        return None
    return texts.pop()


def _self_label(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | Decimal):
        return dimension_value_id(value)
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        return value
    return None


def _nonempty_labels(*candidates: Any) -> set[str]:
    return {
        str(candidate).strip()
        for candidate in candidates
        if candidate is not None and str(candidate).strip()
    }


def _count(
    counts: dict[tuple[str, str, str | None], list[Any]],
    key: tuple[str, str, str | None],
    fact_key: Any,
) -> None:
    entry = counts.setdefault(key, [0, fact_key])
    entry[0] += 1


__all__ = [
    "DimensionLabelIssue",
    "dimension_label_issues",
    "dimension_labels_by_id",
    "dimension_value_id",
    "label_facts",
    "unused_label_declarations",
]
