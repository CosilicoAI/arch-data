"""Shared source-artifact registration primitives.

This module holds manifest parsing and filename identity rules used at every
artifact boundary.  PR #227 extends the same surface with access-specific
registration; keeping the common functions here lets that stacked work rebase
without inventing parallel helpers.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Mapping
import unicodedata

import yaml


class ArtifactFilenameError(ValueError):
    """Raised when a filename is not a bare name inside a package directory."""


def is_bare_filename(value: Any) -> bool:
    """Whether ``value`` names a file inside a directory, with no path."""
    if value is None:
        return False
    text = str(value).strip()
    if not text or text != str(value) or text in (".", ".."):
        return False
    if "/" in text or "\\" in text or "\x00" in text:
        return False
    return Path(text).name == text


def bare_filename(value: Any, *, what: str = "filename") -> str:
    """Return ``value`` as a bare filename, refusing any other spelling.

    ``./adult.tab``, ``sub/../adult.tab``, ``adult.tab/`` and an absolute path
    all resolve to the same file as ``adult.tab`` once joined under the package
    directory, so the manifest and every guard use one spelling.
    """
    if not is_bare_filename(value):
        raise ArtifactFilenameError(
            f"{what} must be a bare filename inside the package directory, not "
            f"{value!r}; it may not carry a directory, '.', '..', a trailing "
            "slash, surrounding whitespace, or an absolute path."
        )
    return str(value)


def filename_key(value: Any) -> str:
    """Return the case-folded, Unicode-normalized comparison key for a name."""
    return unicodedata.normalize("NFC", Path(str(value)).name).casefold()


_MANIFEST_FILENAME_RE = re.compile(
    r"^manifest(?:_[^/\\]+)?\.ya?ml$",
    re.IGNORECASE,
)


def is_manifest_filename(value: Any) -> bool:
    """Whether ``value`` is a package-manifest filename."""
    return is_bare_filename(value) and bool(_MANIFEST_FILENAME_RE.fullmatch(str(value)))


def package_manifest_paths(package_dir: Path) -> list[Path]:
    """Return every manifest file a package directory keeps, sorted by name."""
    directory = Path(package_dir)
    if not directory.is_dir():
        return []
    manifests = []
    for path in sorted(directory.iterdir()):
        if not is_manifest_filename(path.name):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"{path} carries a manifest name but is not a regular file; "
                "Chronicle will not register beside it or sweep past it."
            )
        manifests.append(path)
    return manifests


def validate_package_directory(
    manifests: Mapping[str, Mapping[str, Any] | None],
    *,
    entry_digest: Callable[[str, Any, Mapping[str, Any]], str | None] | None = None,
) -> tuple[str, ...]:
    """Return filename-identity collisions across a package's manifests.

    Two manifests may name one physical file only when every identity they
    record for it agrees. A differing digest means the same package-local
    bytes have two incompatible identities, so no command may act through
    either record. An entry that records no identity yet -- no ``sha256`` and
    no locator, as a manifest looks before its first fetch or publication --
    cannot contradict an identified owner and is not a collision: the command
    that eventually identifies it hashes the shared bytes, so a selected or
    partially failed publication never strands its siblings behind a false
    collision.

    ``entry_digest(manifest_name, vintage, entry)`` resolves an entry's
    *effective* recorded digest -- for example the one its content-addressed
    R2 key encodes when the entry declares no ``sha256`` -- and returns
    ``None`` to fall back to the declared ``sha256`` field. Without it only the
    declared field is compared.
    """
    by_name: dict[str, list[tuple[str, str]]] = {}
    for name, manifest in manifests.items():
        files = manifest.get("files") if isinstance(manifest, Mapping) else None
        if not isinstance(files, Mapping):
            continue
        for vintage, entry in files.items():
            if not isinstance(entry, Mapping):
                continue
            filename = entry.get("filename")
            if filename is None:
                continue
            digest = entry_digest(name, vintage, entry) if entry_digest else None
            if digest is None:
                digest = entry.get("sha256")
            digest = digest.strip() if isinstance(digest, str) else ""
            if not digest:
                # No recorded identity yet: nothing to contradict.
                continue
            by_name.setdefault(filename_key(filename), []).append((name, digest))

    errors: list[str] = []
    for key, records in by_name.items():
        if len({name for name, _digest in records}) < 2:
            continue
        if len({digest for _name, digest in records}) > 1:
            errors.append(f"filename_collision_across_manifests:{key}")
    return tuple(dict.fromkeys(errors))


def matching_directory_entry(directory: Any, filename: Any) -> Any | None:
    """Return the actual directory entry matching a bare filename's safe key.

    Scanning real entries makes the identity rule the same on case-sensitive
    and case-folding filesystems, including Unicode-normalized aliases.
    """
    if not is_bare_filename(filename) or not directory.is_dir():
        return None
    wanted = filename_key(filename)
    matches = [
        path
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if filename_key(path.name) == wanted
    ]
    if len(matches) > 1:
        names = ", ".join(repr(path.name) for path in matches)
        raise ValueError(
            f"{filename!r} matches more than one physical entry ({names}); "
            "the package holds conflicting spellings of one artifact identity "
            "and must be repaired by hand."
        )
    return matches[0] if matches else None


class StrictManifestLoader(yaml.SafeLoader):
    """A YAML loader that refuses a mapping with duplicate keys.

    PyYAML keeps the last of two equal keys, so ``files:`` recorded twice, or
    a vintage recorded as ``2023`` and again as ``2_023`` (the same integer),
    would read as one entry and the shadowed entry would be dropped by the
    next write. A manifest is the record the byte boundary is decided from,
    so a document the loader cannot represent faithfully is malformed.

    ``<<`` merges are honoured with YAML precedence (an explicit key overrides
    a merged one) but never by mutating a node: PyYAML's ``flatten_mapping``
    rewrites the node in place, and because construction is lazy a later
    merge of an anchored entry would turn that entry's inherited keys into
    apparent explicit duplicates. Merge sources are expanded into a fresh
    pair list instead, and every mapping reached through a merge gets the
    same duplicate check as a mapping the document spells out directly.
    """

    _MERGE_TAG = "tag:yaml.org,2002:merge"

    def _explicit_pairs(self, node: Any, deep: bool) -> tuple[list[Any], list[Any]]:
        """Split a mapping node into merge sources and its explicit pairs.

        Refuses duplicate explicit keys on the node's own pair list, before any
        merge is consulted.
        """
        merge_sources: list[Any] = []
        explicit_pairs: list[Any] = []
        seen: set[Any] = set()
        for key_node, value_node in node.value:
            if key_node.tag == self._MERGE_TAG:
                merge_sources.append(value_node)
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found unhashable key ({exc})",
                    key_node.start_mark,
                ) from exc
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
            explicit_pairs.append((key_node, value_node))
        return merge_sources, explicit_pairs

    def _merged_pairs(
        self, source: Any, deep: bool, active: tuple[int, ...]
    ) -> list[Any]:
        """Return the pairs a ``<<`` source contributes, validated, unmutated.

        ``active`` holds the mappings whose merges are being expanded on the
        way to ``source``. A source that is already active merges itself,
        directly or through nested merges, which has no expansion: refuse it
        as a ``ConstructorError`` (a ``yaml.YAMLError`` every manifest reader
        handles) rather than recursing until the interpreter gives up.
        """
        if isinstance(source, yaml.MappingNode):
            if id(source) in active:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    source.start_mark,
                    "found a recursive merge: a mapping merges itself",
                    source.start_mark,
                )
            nested_sources, explicit = self._explicit_pairs(source, deep)
            nested_active = (*active, id(source))
            pairs: list[Any] = []
            for nested in nested_sources:
                pairs.extend(self._merged_pairs(nested, deep, nested_active))
            pairs.extend(explicit)
            return pairs
        if isinstance(source, yaml.SequenceNode):
            for subnode in source.value:
                if not isinstance(subnode, yaml.MappingNode):
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        source.start_mark,
                        f"expected a mapping for merging, but found {subnode.id}",
                        subnode.start_mark,
                    )
            # YAML merge-key precedence: earlier mappings in a ``<<`` sequence
            # win over later ones. Pairs are assigned last-wins, so contribute
            # the later mappings first and the first mapping last.
            pairs = []
            for subnode in reversed(source.value):
                pairs.extend(self._merged_pairs(subnode, deep, active))
            return pairs
        raise yaml.constructor.ConstructorError(
            "while constructing a mapping",
            source.start_mark,
            f"expected a mapping or list of mappings for merging, but found {source.id}",
            source.start_mark,
        )

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        merge_sources, explicit_pairs = self._explicit_pairs(node, deep)
        merged_pairs: list[Any] = []
        for source in merge_sources:
            merged_pairs.extend(self._merged_pairs(source, deep, (id(node),)))
        mapping: dict[Any, Any] = {}
        # Merged pairs first, explicit pairs last: YAML precedence, last wins.
        for key_node, value_node in [*merged_pairs, *explicit_pairs]:
            key = self.construct_object(key_node, deep=deep)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def validate_manifest_vintages(payload: Any) -> None:
    """Refuse different keys that identify one logical ``files`` vintage.

    YAML distinguishes integer ``2024`` from quoted ``"2024"``, but manifest
    consumers select or report them as the same vintage. Validate the entire
    manifest, including vintages other than the one a caller requested, before
    any consumer can read artifact bytes or construct publication routes.

    Leave non-mapping documents and ``files`` blocks to the consumers' existing
    shape checks. Labels retain their spelling, including leading zeroes.
    """
    files = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(files, Mapping):
        return
    seen: dict[str, Any] = {}
    for vintage in files:
        identity = str(vintage)
        if identity in seen:
            raise yaml.YAMLError(
                f"Vintage {identity!r} is recorded under both keys "
                f"{seen[identity]!r} and {vintage!r}; one vintage has one key. "
                "Merge the entries by hand first. Chronicle will not choose "
                "which entry is the record."
            )
        seen[identity] = vintage


def load_manifest_document(text: str) -> Any:
    """Parse a manifest document, refusing duplicate keys and vintages.

    Raises :class:`yaml.YAMLError` for keys YAML would silently collapse or
    for distinct YAML keys that manifest consumers treat as one vintage.
    """
    payload = yaml.load(text, Loader=StrictManifestLoader)  # noqa: S506
    validate_manifest_vintages(payload)
    return payload


__all__ = [
    "ArtifactFilenameError",
    "StrictManifestLoader",
    "bare_filename",
    "filename_key",
    "is_bare_filename",
    "is_manifest_filename",
    "load_manifest_document",
    "matching_directory_entry",
    "package_manifest_paths",
    "validate_manifest_vintages",
    "validate_package_directory",
]
