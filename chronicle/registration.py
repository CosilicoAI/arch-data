"""Access classification and hash-only registration for Chronicle artifacts.

Chronicle registers every raw artifact its consumers build from and stores the
bytes of only those the publisher permits it to redistribute. Manifest fields
carry that split:

``kind``
    Manifest-level: ``publisher_table`` or ``microdata_release``. Every
    manifest created or modified after
    ``docs/adr-chronicle-raw-microdata-identity.md`` declares it. Manifests
    that predate the rule are read as publisher tables only while they match
    the frozen list in :mod:`chronicle.grandfathered_manifests`; any other
    kindless manifest is an error, never a publisher table by default.

``licence``
    The publisher's terms. For a public microdata release this is an
    identifier from the allowlist in :mod:`chronicle.licences`, and the entry
    also carries ``licence_evidence`` binding this artifact to that term.

``access``
    A closed class: ``public``, ``licensed``, or ``restricted``.

``hash_source`` and its attester
    Who asserts the checksum: ``chronicle_fetch`` (``attested_by: chronicle``,
    ``verified_at`` = fetch date), ``consumer_attested`` (``attested_by`` = the
    consumer, ``attestation_evidence``, ``verified_at``), or ``consumer_pin``
    (``attested_by`` = the consumer, ``pinned_from`` = repository, path and
    commit, no ``verified_at``).

``public`` artifacts keep the fetch/publish path: bytes are archived in the raw
R2 bucket under the content-addressed key
``raw/{source_id}/{package_id}/{year}/{sha256}/{filename}``. ``licensed`` and
``restricted`` artifacts are registered *hash-only*: the manifest records the
checksum, vintage, licence, access route and attestation, and no Chronicle
store ever holds the bytes. That key exists only for ``public`` artifacts.

A registration is identified by ``{source_id, package_id, year, sha256,
filename}``. Consumers reference a registration by exactly that tuple. The
filename is a bare name inside the package directory, compared case-folded,
and ``2023`` and ``'2023'`` are one vintage key.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
import unicodedata

import yaml

from chronicle.grandfathered_manifests import is_grandfathered_manifest
from chronicle.licences import is_redistributable_licence, licence_evidence_errors


ACCESS_PUBLIC = "public"
ACCESS_LICENSED = "licensed"
ACCESS_RESTRICTED = "restricted"
#: The closed set of access classes a manifest file entry may declare.
ACCESS_CLASSES: tuple[str, ...] = (ACCESS_PUBLIC, ACCESS_LICENSED, ACCESS_RESTRICTED)
#: Access class inferred for a publisher-table entry that does not declare one.
DEFAULT_ACCESS = ACCESS_PUBLIC

#: The manifest a package directory keeps unless it feeds several packages.
DEFAULT_MANIFEST_FILENAME = "manifest.yaml"

PUBLISHER_TABLE_KIND = "publisher_table"
MICRODATA_RELEASE_KIND = "microdata_release"
#: The closed set of manifest kinds.
MANIFEST_KINDS: tuple[str, ...] = (PUBLISHER_TABLE_KIND, MICRODATA_RELEASE_KIND)
#: Kind of a manifest that does not exist yet, and of a frozen kindless one.
DEFAULT_MANIFEST_KIND = PUBLISHER_TABLE_KIND

HASH_SOURCE_CHRONICLE_FETCH = "chronicle_fetch"
HASH_SOURCE_CONSUMER_ATTESTED = "consumer_attested"
HASH_SOURCE_CONSUMER_PIN = "consumer_pin"
#: The closed set of checksum provenances a registration may declare.
HASH_SOURCES: tuple[str, ...] = (
    HASH_SOURCE_CHRONICLE_FETCH,
    HASH_SOURCE_CONSUMER_ATTESTED,
    HASH_SOURCE_CONSUMER_PIN,
)
#: Provenances a hash-only registration may declare: Chronicle never fetched
#: the bytes, so the checksum is always the consumer's.
HASH_ONLY_HASH_SOURCES: tuple[str, ...] = (
    HASH_SOURCE_CONSUMER_ATTESTED,
    HASH_SOURCE_CONSUMER_PIN,
)
#: The attester of a ``chronicle_fetch`` checksum.
CHRONICLE_ATTESTER = "chronicle"
#: Fields a ``pinned_from`` block carries: where the consumer's pin was read.
PINNED_FROM_FIELDS: tuple[str, ...] = ("repository", "path", "commit")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# Registration entry key order, so emitted manifests are byte-stable.
_REGISTRATION_FIELD_ORDER: tuple[str, ...] = (
    "filename",
    "access",
    "licence",
    "licence_evidence",
    "vintage",
    "sha256",
    "size_bytes",
    "source_url",
    "access_route",
    "doi",
    "study",
    "fetched_at",
    "verified_at",
    "hash_source",
    "attested_by",
    "attestation_evidence",
    "pinned_from",
    "notes",
)


#: Legacy publisher-table extensions that remain part of the accepted entry
#: schema. They are read by table-specific consumers rather than the generic
#: artifact commands, but are explicit so an ordinary typo is never silently
#: treated as an extension.
_ENTRY_EXTENSION_FIELDS: frozenset[str] = frozenset(
    {
        "archive_member",
        "csv_member",
        "download_url",
        "source_table",
        "source_urls",
        "year",
        "years",
    }
)

#: Every field a manifest file entry may carry. This is a closed top-level
#: schema: writers may add a field only by adding it to the registration
#: contract or the explicit legacy extension allowlist above.
_ENTRY_FIELDS: frozenset[str] = frozenset(
    {
        *_REGISTRATION_FIELD_ORDER,
        *_ENTRY_EXTENSION_FIELDS,
        "storage",
        "size_bytes",
        "source_url",
    }
)


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
                f"duplicate_vintage_key:{identity}: Vintage {identity!r} is recorded under both keys "
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


class ManifestAccessError(ValueError):
    """Raised when a manifest declares an unusable access class or kind."""


class ManifestKindError(ManifestAccessError):
    """Raised when a manifest declares no kind and is not frozen kindless."""


class ArtifactFilenameError(ManifestAccessError):
    """Raised when a filename is not a bare name inside the package directory."""


class AmbiguousVintageKeyError(ManifestAccessError):
    """Raised when a manifest records one vintage under both key spellings."""


class HashOnlyRegistrationError(ValueError):
    """Raised when a hash-only registration is malformed or would store bytes."""


class MicrodataReleaseNotParseableError(ValueError):
    """Raised when a source package points at a registered microdata release.

    Registration is manifest-level: no source package parses a microdata
    release, and no microdata row, cell, or fact enters Chronicle.
    """


@dataclass(frozen=True)
class ListSpecRejected:
    """Marker for a list ``files[year]`` value outside a microdata release."""

    spec: Any


# --------------------------------------------------------------------------
# Manifest kind
# --------------------------------------------------------------------------


def manifest_kind(
    manifest: Mapping[str, Any] | None,
    *,
    manifest_path: Any = None,
) -> str:
    """Return a manifest's declared kind.

    An absent or empty manifest has the default kind: the command creating it
    declares one. So does a manifest that declares no file entry (a bare
    ``files:`` line or an empty mapping): there is nothing in it that could be
    read as a publisher table, and the command writing its first entry
    declares the kind. A manifest with entries must declare ``kind`` itself,
    unless ``manifest_path`` names a file frozen kindless before the rule and
    its bytes still match the freeze.
    """
    if not isinstance(manifest, Mapping) or not manifest:
        return DEFAULT_MANIFEST_KIND
    declared = manifest.get("kind")
    if declared is None:
        if not has_file_entries(manifest):
            return DEFAULT_MANIFEST_KIND
        if manifest_path is not None and is_grandfathered_manifest(manifest_path):
            return PUBLISHER_TABLE_KIND
        where = str(manifest_path) if manifest_path is not None else "Manifest"
        raise ManifestKindError(
            f"{where} declares no kind. Every manifest created or modified "
            f"after the microdata-identity ADR declares kind: one of "
            f"{list(MANIFEST_KINDS)}; a kindless manifest is read as a "
            "publisher table only while it matches the frozen list in "
            "chronicle/grandfathered_manifests.py byte for byte."
        )
    kind = str(declared)
    if kind not in MANIFEST_KINDS:
        raise ManifestAccessError(
            f"Unknown manifest kind {kind!r}; expected one of {list(MANIFEST_KINDS)}."
        )
    return kind


def safe_manifest_kind(
    manifest: Mapping[str, Any] | None,
    *,
    manifest_path: Any = None,
) -> tuple[str, str | None]:
    """Return ``(kind, error_code)`` without raising.

    The reporting commands use this so a manifest with a missing or unknown
    kind is still walked and reported. The returned kind is only what the
    entries are read *as* for that report; the error code says the manifest
    itself is invalid.
    """
    try:
        return manifest_kind(manifest, manifest_path=manifest_path), None
    except ManifestKindError:
        return DEFAULT_MANIFEST_KIND, "manifest_kind_missing"
    except ManifestAccessError:
        declared = manifest.get("kind") if isinstance(manifest, Mapping) else None
        return DEFAULT_MANIFEST_KIND, f"unknown_manifest_kind:{declared}"


def is_microdata_release(
    manifest: Mapping[str, Any] | None,
    *,
    manifest_path: Any = None,
) -> bool:
    """Whether a manifest registers a microdata release rather than a table."""
    return manifest_kind(manifest, manifest_path=manifest_path) == (
        MICRODATA_RELEASE_KIND
    )


# --------------------------------------------------------------------------
# Access
# --------------------------------------------------------------------------


def normalize_access(access: str | None) -> str:
    """Return a validated access class, defaulting to ``public``."""
    if access is None:
        return DEFAULT_ACCESS
    value = str(access)
    if value not in ACCESS_CLASSES:
        raise ManifestAccessError(
            f"Unknown access class {value!r}; expected one of {list(ACCESS_CLASSES)}."
        )
    return value


def entry_access(spec: Any) -> str:
    """Return the access class a manifest file entry declares or inherits."""
    if not isinstance(spec, Mapping):
        return DEFAULT_ACCESS
    return normalize_access(spec.get("access"))


def safe_entry_access(spec: Any) -> str:
    """Return an entry's access class, falling back to ``public`` if unknown.

    An unparseable class is reported by :func:`validate_file_entry`; treating it
    as ``public`` here would be unsafe, so it is treated as ``restricted`` and
    therefore never uploaded.
    """
    if not isinstance(spec, Mapping):
        return DEFAULT_ACCESS
    try:
        return normalize_access(spec.get("access"))
    except ManifestAccessError:
        return ACCESS_RESTRICTED


def strict_entry_access(spec: Any, *, kind: str) -> str:
    """Return an entry's access class, refusing to infer one for a release.

    A publisher-table entry that omits ``access`` is public; a microdata
    release entry must say what it is, and an unknown class is refused with
    the value the manifest actually declares.
    """
    if not isinstance(spec, Mapping):
        return DEFAULT_ACCESS
    declared = spec.get("access")
    if declared is None:
        if kind == MICRODATA_RELEASE_KIND:
            raise ManifestAccessError(
                f"Entry {spec.get('filename')!r} declares no access class. A "
                "microdata release entry must declare access; Chronicle will "
                "not infer public for it."
            )
        return DEFAULT_ACCESS
    return normalize_access(declared)


def stores_bytes(access: str) -> bool:
    """Whether Chronicle may hold this access class's bytes."""
    return normalize_access(access) == ACCESS_PUBLIC


def is_hash_only(access: str) -> bool:
    """Whether this access class must be registered without bytes."""
    return not stores_bytes(access)


# --------------------------------------------------------------------------
# Filenames and vintage keys
# --------------------------------------------------------------------------


def is_bare_filename(value: Any) -> bool:
    """Whether ``value`` names a file inside a directory, with no path."""
    text = _text(value)
    if text is None or text != str(value) or text in (".", ".."):
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
    """Return the comparison key for a filename.

    Case-folded and Unicode-normalised (NFC), because the package directories
    these commands run in are as often as not on a filesystem that folds
    both: ``ADULT.TAB`` and ``adult.tab`` are one file, and so are the
    composed and decomposed spellings of an accented name. Treating them as
    one artifact path is the safe rule everywhere.
    """
    return unicodedata.normalize("NFC", Path(str(value)).name).casefold()


#: The names a package directory's manifests may carry: ``manifest.yaml`` or
#: ``manifest_<package>.yaml`` (``.yml`` accepted), matched case-insensitively
#: because the directory is as often as not on a case-insensitive filesystem.
_MANIFEST_FILENAME_RE = re.compile(r"^manifest(?:_[^/\\]+)?\.ya?ml$", re.IGNORECASE)


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


def iter_directory_entries(
    manifests: Mapping[str, Mapping[str, Any] | None],
) -> Iterator[tuple[str, Any, int | None, Any]]:
    """Yield ``(manifest_name, key, index, entry)`` across a directory's manifests."""
    for name, manifest in manifests.items():
        for key, index, entry in iter_manifest_entries(manifest):
            yield name, key, index, entry


def hash_only_registrations(
    manifests: Mapping[str, Mapping[str, Any] | None],
    *,
    filename: Any = None,
    sha256: Any = None,
) -> list[tuple[str, Any, Mapping[str, Any]]]:
    """Return the hash-only entries across ``manifests`` matching a name or digest.

    A match by case-folded bare filename is the same path in the package
    directory; a match by digest is the same bytes under another name. Both
    identify the gated artifact, whichever manifest registers it. An entry
    whose access class cannot be read is treated as hash-only (never public).
    """
    wanted_name = filename_key(filename) if filename else None
    wanted_digest = _text(sha256)
    matches: list[tuple[str, Any, Mapping[str, Any]]] = []
    for name, key, _index, entry in iter_directory_entries(manifests):
        if not isinstance(entry, Mapping):
            continue
        if not is_hash_only(safe_entry_access(entry)):
            continue
        entry_name = entry.get("filename")
        same_name = (
            wanted_name is not None
            and entry_name is not None
            and filename_key(entry_name) == wanted_name
        )
        same_bytes = wanted_digest is not None and _text(entry.get("sha256")) == (
            wanted_digest
        )
        if same_name or same_bytes:
            matches.append((name, key, entry))
    return matches


def validate_package_directory(
    manifests: Mapping[str, Mapping[str, Any] | None],
    *,
    entry_digest: Callable[[str, Any, Mapping[str, Any]], str | None] | None = None,
) -> tuple[str, ...]:
    """Return the collision codes across every manifest a directory keeps.

    Two manifests may record one public file only as the same bytes (the
    tracked shape: ``manifest.yaml`` beside ``manifest_<package>.yaml`` both
    recording one publisher zip with one digest). A name held public in one
    manifest and hash-only in another, public under two digests, or a digest
    held public in one and hash-only in another, is one file that two records
    disagree about, and no command may act through either record.

    A public entry without a recorded digest cannot contradict an identified
    public owner: the command that first identifies it checks the shared bytes.
    Its filename still participates in public/hash-only access conflicts.

    ``entry_digest(manifest_name, vintage, entry)`` resolves an entry's effective
    recorded digest, such as the digest encoded in its content-addressed R2 key.
    Returning ``None`` falls back to the declared ``sha256`` field; without a
    resolver only that field is compared. Archived identities are checked
    independently, including historical locators whose entry metadata changed.
    """
    by_name: dict[str, list[tuple[str, bool, str]]] = {}
    by_digest: dict[str, list[tuple[str, bool]]] = {}
    archived_names: set[str] = set()
    archived_digests: set[str] = set()
    for name, vintage, _index, entry in iter_directory_entries(manifests):
        if not isinstance(entry, Mapping):
            continue
        hash_only = is_hash_only(safe_entry_access(entry))
        digest = entry_digest(name, vintage, entry) if entry_digest else None
        if digest is None:
            digest = entry.get("sha256")
        digest = _text(digest) or ""
        filename = entry.get("filename")
        if filename is not None:
            by_name.setdefault(filename_key(filename), []).append(
                (name, hash_only, digest)
            )
        if digest:
            by_digest.setdefault(digest, []).append((name, hash_only))
        for archived_name, archived_digest, _locator in _recorded_object_identities(
            entry
        ):
            if archived_name is not None:
                archived_names.add(archived_name)
            if archived_digest is not None:
                archived_digests.add(archived_digest)

    errors: list[str] = []
    for key, records in by_name.items():
        if len({name for name, _hash_only, _digest in records}) < 2:
            continue
        classes = {hash_only for _name, hash_only, _digest in records}
        digests = {digest for _name, _hash_only, digest in records if digest}
        if len(classes) > 1 or (not classes.pop() and len(digests) > 1):
            errors.append(f"filename_collision_across_manifests:{key}")
    for digest, records in by_digest.items():
        if len({name for name, _hash_only in records}) < 2:
            continue
        if len({hash_only for _name, hash_only in records}) > 1:
            errors.append(f"sha256_collision_across_manifests:{digest}")
    # An immutable object remains archived after its current entry changes
    # filename or checksum. Compare gated registrations with every recorded
    # object, including history in the same manifest as the registration.
    for key in sorted(archived_names.intersection(by_name)):
        if any(hash_only for _name, hash_only, _digest in by_name[key]):
            errors.append(f"archived_filename_collision:{key}")
    for digest in sorted(archived_digests.intersection(by_digest)):
        if any(hash_only for _name, hash_only in by_digest[digest]):
            errors.append(f"archived_sha256_collision:{digest}")
    return tuple(_dedupe(errors))


def vintage_key_forms(year: Any) -> tuple[Any, ...]:
    """Return the key spellings that address the same vintage as ``year``.

    ``2023`` and ``'2023'`` are one vintage: every identity Chronicle derives
    from a year (registration ids, R2 keys) renders it as text, and the
    source-package reader accepts both. Label keys such as ``'A_1'`` have no
    other spelling.
    """
    if isinstance(year, bool):
        return (year,)
    if isinstance(year, int):
        return (year, str(year))
    text = str(year)
    if text.isdecimal() and (text == "0" or not text.startswith("0")):
        return (text, int(text))
    return (year,)


def resolve_vintage_key(files: Mapping[Any, Any], year: Any) -> Any | None:
    """Return the key ``files`` already uses for ``year``'s vintage, or None.

    Refuses a mapping that records the vintage under both spellings: one
    vintage has one key, and Chronicle will not choose which entry is the
    record.
    """
    present = [form for form in vintage_key_forms(year) if form in files]
    if len(present) > 1:
        raise AmbiguousVintageKeyError(
            f"Vintage {year!r} is recorded under both keys {present!r}; one "
            "vintage has one key. Merge the entries by hand first."
        )
    return present[0] if present else None


def iter_manifest_entries(
    manifest: Mapping[str, Any] | None,
) -> Iterator[tuple[Any, int | None, Any]]:
    """Yield every ``(key, index, entry)`` a manifest declares, whatever shape.

    Deliberately not gated on the manifest ``kind``: a guard must see the
    entries a manifest actually holds, including a list under a manifest whose
    kind is absent or misspelled. ``index`` is the entry's position in a list
    value and None for a single mapping.
    """
    if not isinstance(manifest, Mapping):
        return
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return
    for key, spec in files.items():
        if isinstance(spec, list):
            for index, entry in enumerate(spec):
                yield key, index, entry
        else:
            yield key, None, spec


def has_file_entries(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a manifest declares any file entry at all.

    A ``files`` block that is absent, an explicit null (a bare ``files:``
    line) or an empty mapping declares nothing; a vintage key holding an empty
    list declares nothing either. A ``files`` value that is not a mapping is
    content Chronicle cannot read, and counts as entries so that the kind rule
    and the ``files_not_a_mapping`` refusal both fire on it.
    """
    if not isinstance(manifest, Mapping):
        return False
    files = manifest.get("files")
    if files is None:
        return False
    if not isinstance(files, Mapping):
        return True
    return any(True for _entry in iter_manifest_entries(manifest))


def registration_id(
    *,
    source_id: str,
    package_id: str,
    year: Any,
    sha256: str,
    filename: str,
) -> str:
    """Return the registration identity tuple as a stable string."""
    return f"{source_id}/{package_id}/{year}/{sha256}/{filename}"


def iter_file_specs(spec: Any, *, kind: str) -> tuple[Any, ...]:
    """Expand one ``files[year]`` value into individual file entries.

    A microdata release registers many files under one vintage — the 14 FRS
    2023-24 tabs share ``{source_id, package_id, year}`` and differ only by
    ``filename`` and ``sha256`` — so its ``files[year]`` value may be a list.
    Publisher-table manifests keep the single-mapping shape, and a list there is
    surfaced as a rejected entry rather than silently expanded.
    """
    if isinstance(spec, list):
        if kind != MICRODATA_RELEASE_KIND:
            return (ListSpecRejected(spec),)
        return tuple(spec)
    return (spec,)


# --------------------------------------------------------------------------
# Validation vocabulary
# --------------------------------------------------------------------------


def validate_manifest_files(manifest: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return manifest-level error codes: shape, key and filename collisions.

    These are properties of the ``files`` mapping as a whole, which no single
    entry can see: a vintage recorded under two key spellings, a filename that
    is not a bare name, and two entries that resolve to one file in the
    package directory while disagreeing about what it holds.
    """
    if not isinstance(manifest, Mapping):
        return ()
    files = manifest.get("files")
    if files is None:
        return ()
    if not isinstance(files, Mapping):
        return ("files_not_a_mapping",)

    errors: list[str] = []
    for key in files:
        for other in vintage_key_forms(key):
            if other != key and other in files:
                errors.append(f"duplicate_vintage_key:{key}")

    release = manifest.get("kind") == MICRODATA_RELEASE_KIND
    # Same file, one package directory: group every entry by its resolved name.
    by_name: dict[str, list[tuple[Any, Mapping[str, Any]]]] = {}
    for key, _index, entry in iter_manifest_entries(manifest):
        if not isinstance(entry, Mapping):
            continue
        filename = entry.get("filename")
        if filename is None:
            continue
        if not is_bare_filename(filename):
            errors.append(f"non_canonical_filename:{filename}")
        by_name.setdefault(filename_key(filename), []).append((key, entry))

    by_digest: dict[str, set[bool]] = {}
    for _key, _index, entry in iter_manifest_entries(manifest):
        if not isinstance(entry, Mapping):
            continue
        digest = _text(entry.get("sha256"))
        if digest:
            by_digest.setdefault(digest, set()).add(
                is_hash_only(safe_entry_access(entry))
            )
    for digest, classes in by_digest.items():
        if len(classes) > 1:
            # The same bytes are the same artifact whatever name they carry:
            # a digest is not both bytes Chronicle holds and bytes it must
            # never hold.
            errors.append(f"sha256_collision:{digest}")

    for name, entries in by_name.items():
        if len(entries) < 2:
            continue
        classes = {is_hash_only(safe_entry_access(entry)) for _key, entry in entries}
        if len(classes) > 1:
            # One path cannot be both bytes Chronicle holds and bytes it must
            # never hold.
            errors.append(f"filename_collision:{name}")
            continue
        hash_only = classes.pop()
        seen: dict[Any, set[str]] = {}
        for key, entry in entries:
            digest = _text(entry.get("sha256")) or ""
            vintage = seen.setdefault(key, set())
            if digest in vintage:
                errors.append(f"duplicate_filename_in_vintage:{name}")
            vintage.add(digest)
        if hash_only:
            # Several vintages, or an explicit reissue, may register the same
            # filename with different bytes: no file exists to collide.
            continue
        if release:
            # A public release's bytes are staged content-addressed outside
            # the tree, so one filename may hold different bytes under
            # different vintages; under one vintage a revision is recorded in
            # storage.previous_r2, not as a second entry.
            for digests in seen.values():
                if len(digests) > 1:
                    errors.append(f"filename_collision:{name}")
            continue
        digests = {_text(entry.get("sha256")) or "" for _key, entry in entries}
        if len(digests) > 1:
            # Public entries share one path in the tree and one current object
            # per name; a revision is recorded in storage.previous_r2, not as a
            # second entry.
            errors.append(f"filename_collision:{name}")
    return tuple(_dedupe(errors))


def validate_file_entry(
    spec: Any,
    *,
    kind: str,
    manifest: Mapping[str, Any] | None,
    local_file_exists: bool,
) -> tuple[str, ...]:
    """Return stable error codes for one manifest file entry.

    The codes are the refusal vocabulary shared by ``inventory-artifacts``,
    ``publish-raw``, ``fetch-artifact`` and ``register-artifact``.
    ``local_file_exists`` says whether the entry's filename exists beside the
    manifest, in the package directory.
    """
    if isinstance(spec, ListSpecRejected):
        return ("list_file_spec_requires_microdata_release_kind",)
    if not isinstance(spec, Mapping):
        return ("malformed_file_spec",)

    errors: list[str] = []
    for field in spec:
        if field in _ENTRY_FIELDS:
            continue
        if isinstance(field, str) and field.strip().casefold() in _ENTRY_FIELDS:
            # A field the writer meant but the reader would ignore: ``Access``
            # is not ``access``, and an entry whose access class sits under
            # the wrong key is not public by omission.
            errors.append(f"misspelled_field:{field}")
        else:
            # The schema is closed. Treating an arbitrary key as an extension
            # would make ordinary typos such as ``acess`` indistinguishable
            # from intentional metadata, allowing missing ``access`` to fall
            # through to the publisher-table public default.
            errors.append(f"unknown_field:{field}")
    filename = spec.get("filename")
    if not _text(filename):
        errors.append("missing_filename")
    elif not is_bare_filename(filename):
        errors.append(f"non_canonical_filename:{filename}")
    elif is_manifest_filename(filename):
        errors.append(f"manifest_named_filename:{filename}")

    declared_sha256 = spec.get("sha256")
    if declared_sha256 is not None and not (
        isinstance(declared_sha256, str) and _SHA256_RE.fullmatch(declared_sha256)
    ):
        errors.append("malformed_sha256")

    declared_access = spec.get("access")
    if declared_access is None:
        if kind == MICRODATA_RELEASE_KIND:
            errors.append("missing_access")
        access = DEFAULT_ACCESS
    else:
        try:
            access = normalize_access(declared_access)
        except ManifestAccessError:
            return (*errors, f"unknown_access_class:{declared_access}")

    if kind == MICRODATA_RELEASE_KIND and not _text(spec.get("licence")):
        errors.append("missing_licence")

    if is_hash_only(access):
        errors.extend(
            _hash_only_entry_errors(
                spec,
                manifest=manifest,
                local_file_exists=local_file_exists,
            )
        )
    elif kind == MICRODATA_RELEASE_KIND:
        errors.extend(
            _public_release_entry_errors(spec, local_file_exists=local_file_exists)
        )
    if is_hash_only(access) or kind == MICRODATA_RELEASE_KIND:
        errors.extend(_attestation_errors(spec))
    return tuple(_dedupe(errors))


def _checksum_errors(spec: Mapping[str, Any]) -> list[str]:
    sha256 = _text(spec.get("sha256"))
    if not sha256:
        return ["missing_sha256"]
    if not _SHA256_RE.match(sha256):
        return ["malformed_sha256"]
    return []


def _hash_only_entry_errors(
    spec: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None,
    local_file_exists: bool,
) -> list[str]:
    """Return refusal codes for a licensed or restricted registration."""
    errors: list[str] = []
    if not _text(spec.get("licence")):
        errors.append("missing_licence")
    errors.extend(_checksum_errors(spec))
    if not _text(spec.get("vintage")):
        errors.append("missing_vintage")
    if not _access_route(spec, manifest):
        errors.append("missing_access_route")
    if local_file_exists:
        errors.append("bytes_present_for_hash_only_entry")
    if recorded_r2(spec):
        errors.append("r2_location_for_hash_only_entry")
    if recorded_previous_r2(spec):
        errors.append("r2_history_for_hash_only_entry")
    storage = spec.get("storage")
    if storage is not None and storage != {}:
        # No Chronicle store holds these bytes, so there is nothing a storage
        # block could truthfully record, whatever its shape.
        errors.append("storage_for_hash_only_entry")
    hash_source = _text(spec.get("hash_source"))
    if hash_source in HASH_SOURCES and hash_source not in HASH_ONLY_HASH_SOURCES:
        # Chronicle never fetched a hash-only entry's bytes: its checksum is
        # always the consumer's.
        errors.append(f"hash_source_not_allowed_for_hash_only_entry:{hash_source}")
    return errors


def _public_release_entry_errors(
    spec: Mapping[str, Any],
    *,
    local_file_exists: bool,
) -> list[str]:
    """Return refusal codes for a public microdata release entry.

    Bytes are archived only under an allowlisted licence with evidence that
    binds this artifact to it, and never inside the package directory: public
    microdata is staged outside ``db/data`` and uploaded from there.
    """
    errors: list[str] = []
    errors.extend(_checksum_errors(spec))
    if not _text(spec.get("vintage")):
        errors.append("missing_vintage")
    licence = _text(spec.get("licence"))
    if licence and not is_redistributable_licence(licence):
        errors.append(f"licence_not_redistributable:{licence}")
    errors.extend(
        licence_evidence_errors(
            spec.get("licence_evidence"),
            licence=licence,
            sha256=_text(spec.get("sha256")),
        )
    )
    if local_file_exists:
        errors.append("bytes_present_for_microdata_release_entry")
    return errors


def _attestation_errors(spec: Mapping[str, Any]) -> list[str]:
    """Return refusal codes for an entry's ``hash_source`` and attester."""
    errors: list[str] = []
    hash_source = _text(spec.get("hash_source"))
    if not hash_source:
        return ["missing_hash_source"]
    if hash_source not in HASH_SOURCES:
        return [f"unknown_hash_source:{hash_source}"]
    attested_by = _text(spec.get("attested_by"))
    if not attested_by:
        errors.append("missing_attested_by")
    verified_at = _text(spec.get("verified_at"))
    if hash_source == HASH_SOURCE_CHRONICLE_FETCH:
        if attested_by and attested_by != CHRONICLE_ATTESTER:
            errors.append("attested_by_not_chronicle")
        if not verified_at:
            errors.append("missing_verified_at")
    elif hash_source == HASH_SOURCE_CONSUMER_ATTESTED:
        if not _text(spec.get("attestation_evidence")):
            errors.append("missing_attestation_evidence")
        if not verified_at:
            errors.append("missing_verified_at")
    else:
        errors.extend(_pinned_from_errors(spec.get("pinned_from")))
        if verified_at:
            errors.append("verified_at_forbidden_for_consumer_pin")
    if hash_source != HASH_SOURCE_CHRONICLE_FETCH and _is_chronicle(attested_by):
        # A consumer's checksum is attested by the consumer; Chronicle only
        # attests what it fetched and hashed itself.
        errors.append("attested_by_chronicle_for_consumer_hash_source")
    return errors


def _is_chronicle(attester: str | None) -> bool:
    return bool(attester) and attester.strip().casefold() == CHRONICLE_ATTESTER


def _pinned_from_errors(pinned_from: Any) -> list[str]:
    if pinned_from is None:
        return ["missing_pinned_from"]
    if not isinstance(pinned_from, Mapping):
        return ["malformed_pinned_from"]
    errors = [
        f"pinned_from_missing_field:{field}"
        for field in PINNED_FROM_FIELDS
        if not _text(pinned_from.get(field))
    ]
    commit = _text(pinned_from.get("commit"))
    if commit and not _COMMIT_RE.match(commit):
        errors.append("malformed_pinned_from_commit")
    return errors


def _access_route(
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> str | None:
    """Return the recorded route to the bytes, from the entry or the manifest."""
    for key in ("access_route", "source_url", "source_page", "doi"):
        value = _text(spec.get(key))
        if value:
            return value
    if isinstance(manifest, Mapping):
        for key in ("source_page", "access_route"):
            value = _text(manifest.get(key))
            if value:
                return value
    return None


def recorded_r2(spec: Any) -> Mapping[str, Any] | None:
    """Return a recorded ``storage.r2`` mapping, if the entry carries one."""
    if not isinstance(spec, Mapping):
        return None
    storage = spec.get("storage")
    if not isinstance(storage, Mapping):
        return None
    r2 = storage.get("r2")
    return r2 if isinstance(r2, Mapping) else None


def recorded_previous_r2(spec: Any) -> tuple[Any, ...]:
    """Return the entry's ``storage.previous_r2`` history, if it carries one."""
    if not isinstance(spec, Mapping):
        return ()
    storage = spec.get("storage")
    if not isinstance(storage, Mapping):
        return ()
    previous = storage.get("previous_r2")
    if isinstance(previous, list):
        return tuple(previous)
    return (previous,) if previous else ()


def split_r2_uri(uri: str) -> tuple[str, str, str] | None:
    """Split ``provider://bucket/key`` into its three parts, or None.

    One parser for both readers of a recorded locator. A raw key is a path
    with no query or fragment, so everything after the authority's first
    ``/`` is the key: reading the same URI with a URL parser would stop the
    key at a ``?`` or ``#`` and lose the identity the rest of it carries,
    while the validator kept reading.
    """
    provider, separator, remainder = uri.partition("://")
    if not separator or not provider:
        return None
    bucket, separator, key = remainder.partition("/")
    if not separator or not bucket or not key:
        return None
    return (provider, bucket, key)


def _recorded_object_identities(
    spec: Any,
) -> Iterator[tuple[str | None, str | None, str]]:
    """Yield filename/checksum identities from every current and archived object.

    Read both locator fields independently: a conflicting or incomplete block
    must not hide an archived identity simply because another field was chosen.
    Locator validity remains the artifact reader's separate responsibility.
    """
    for block in (recorded_r2(spec), *recorded_previous_r2(spec)):
        if not isinstance(block, Mapping):
            continue
        for field in ("key", "uri"):
            locator = block.get(field)
            if not isinstance(locator, str) or not locator:
                continue
            if field == "uri":
                parts = split_r2_uri(locator)
                if parts is None:
                    continue
                key = parts[2]
            else:
                key = locator
            segments = key.rsplit("/", 2)
            filename = (
                filename_key(segments[-1]) if is_bare_filename(segments[-1]) else None
            )
            digest = (
                segments[-2]
                if len(segments) >= 2 and _SHA256_RE.fullmatch(segments[-2])
                else None
            )
            if filename is not None or digest is not None:
                yield filename, digest, locator


def records_r2_object(spec: Any) -> bool:
    """Whether an entry names any object in the raw bucket, current or past."""
    return recorded_r2(spec) is not None or bool(recorded_previous_r2(spec))


def normalize_hash_source(value: Any, *, allowed: Iterable[str] = HASH_SOURCES) -> str:
    """Return a validated ``hash_source`` value."""
    text = _text(value)
    allowed = tuple(allowed)
    if text is None or text not in allowed:
        raise ManifestAccessError(
            f"Unknown hash_source {value!r}; expected one of {list(allowed)}."
        )
    return text


# --------------------------------------------------------------------------
# Hash-only registration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRegistrationReport:
    """Report from registering one hash-only source artifact."""

    manifest_path: str
    source_id: str
    package_id: str
    year: int
    filename: str
    sha256: str
    size_bytes: int | None
    vintage: str
    licence: str
    access: str
    registration: str
    replaced: bool
    hash_source: str
    attested_by: str
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether the registration was written without refusals."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "valid": self.valid,
            "manifest_path": self.manifest_path,
            "source_id": self.source_id,
            "package_id": self.package_id,
            "year": self.year,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "vintage": self.vintage,
            "licence": self.licence,
            "access": self.access,
            "registration": self.registration,
            "replaced": self.replaced,
            "hash_source": self.hash_source,
            "attested_by": self.attested_by,
            "r2_location": None,
            "errors": list(self.errors),
        }


def register_hash_only_artifact(
    *,
    source_id: str,
    package_id: str,
    year: int,
    output_dir: str | Path,
    filename: str,
    sha256: str,
    licence: str,
    access: str,
    vintage: str,
    hash_source: str,
    attested_by: str,
    attestation_evidence: str | None = None,
    pinned_from: Mapping[str, Any] | None = None,
    verified_at: str | None = None,
    size_bytes: int | None = None,
    source_page: str | None = None,
    source_url: str | None = None,
    access_route: str | None = None,
    doi: str | None = None,
    study: str | None = None,
    dataset: str | None = None,
    table: str | None = None,
    publisher: str | None = None,
    fetched_at: str | None = None,
    notes: str | None = None,
    allow_reissue: bool = False,
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
) -> ArtifactRegistrationReport:
    """Register a licensed or restricted artifact by identity, without bytes.

    Writes (or updates) a ``kind: microdata_release`` manifest entry carrying the
    checksum, size, vintage, licence, access route, and the attestation of who
    asserts the checksum. No bytes are read, written, or uploaded, and no R2
    key is recorded. Every refusal below happens before the manifest is
    touched.

    ``manifest_filename`` names the manifest inside ``output_dir`` the entry
    belongs to, exactly as ``fetch-artifact --manifest`` does: a directory
    that keeps ``manifest_<package>.yaml`` files and no ``manifest.yaml`` is
    refused the default name rather than given a stray third manifest.
    """
    source_id_text = _text(source_id)
    package_id_text = _text(package_id)
    if source_id_text is None or package_id_text is None:
        raise HashOnlyRegistrationError(
            "A registration needs non-empty source_id and package_id values; "
            f"got source_id={source_id!r}, package_id={package_id!r}."
        )
    access_class = normalize_access(access)
    if stores_bytes(access_class):
        raise HashOnlyRegistrationError(
            "register-artifact records identity without bytes and refuses "
            f"access={ACCESS_PUBLIC!r}. Register a public artifact with its "
            "bytes using fetch-artifact."
        )
    checksum = _text(sha256)
    if not checksum or not _SHA256_RE.match(checksum):
        raise HashOnlyRegistrationError(
            "A registration needs a lowercase 64-character SHA-256; refusing to "
            f"register {filename!r} with sha256={sha256!r}. Never invent a hash."
        )
    if not _text(licence):
        raise HashOnlyRegistrationError(
            f"A {access_class} registration must record the publisher licence."
        )
    if not _text(vintage):
        raise HashOnlyRegistrationError(
            f"A {access_class} registration must record the artifact vintage."
        )
    if not is_bare_filename(filename):
        raise HashOnlyRegistrationError(
            f"Registration filename must be a bare filename; got {filename!r}."
        )
    if is_manifest_filename(filename):
        raise HashOnlyRegistrationError(
            f"Registration filename {filename!r} is a manifest name; an "
            "artifact may not be named like a manifest."
        )
    artifact_name = str(filename)
    provenance = _hash_only_attestation(
        hash_source=hash_source,
        attested_by=attested_by,
        attestation_evidence=attestation_evidence,
        pinned_from=pinned_from,
        verified_at=verified_at,
        access_class=access_class,
    )

    entry = _registration_entry(
        filename=artifact_name,
        access=access_class,
        licence=str(licence),
        vintage=str(vintage),
        sha256=checksum,
        size_bytes=size_bytes,
        source_url=source_url,
        access_route=access_route,
        doi=doi,
        study=study,
        fetched_at=fetched_at,
        notes=notes,
        **provenance,
    )
    output = Path(output_dir)
    manifest_path = _registration_manifest_path(output, manifest_filename)
    preparation = {
        "output": output,
        "manifest_path": manifest_path,
        "source_id": source_id_text,
        "package_id": package_id_text,
        "year": year,
        "artifact_name": artifact_name,
        "checksum": checksum,
        "access_class": access_class,
        "entry": entry,
        "source_page": source_page,
        "dataset": dataset,
        "publisher": publisher,
        "table": table,
        "allow_reissue": allow_reissue,
    }
    # Complete the read/validate/mutate calculation once before creating the
    # output directory or lock file. Ordinary refusals therefore have no
    # filesystem side effect. Repeat it under the package-wide lock so a
    # concurrent registration cannot be lost between read and replacement.
    _prepare_registration_payload(**preparation)
    output.mkdir(parents=True, exist_ok=True)
    with _registration_lock(output):
        payload, replaced = _prepare_registration_payload(**preparation)
        document = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        _atomic_replace_manifest(manifest_path, document)

    return ArtifactRegistrationReport(
        manifest_path=str(manifest_path),
        source_id=source_id_text,
        package_id=package_id_text,
        year=year,
        filename=artifact_name,
        sha256=checksum,
        size_bytes=size_bytes,
        vintage=str(vintage),
        licence=str(licence),
        access=access_class,
        registration=registration_id(
            source_id=source_id_text,
            package_id=package_id_text,
            year=year,
            sha256=checksum,
            filename=artifact_name,
        ),
        replaced=replaced,
        hash_source=provenance["hash_source"],
        attested_by=provenance["attested_by"],
    )


def _prepare_registration_payload(
    *,
    output: Path,
    manifest_path: Path,
    source_id: str,
    package_id: str,
    year: int,
    artifact_name: str,
    checksum: str,
    access_class: str,
    entry: dict[str, Any],
    source_page: str | None,
    dataset: str | None,
    publisher: str | None,
    table: str | None,
    allow_reissue: bool,
) -> tuple[dict[str, Any], bool]:
    """Return the complete registration write after validating current state.

    This function has no filesystem side effects. The caller runs it once as a
    preflight and again while holding the package lock, so ordinary refusals
    precede even lock creation while concurrent registrations still serialize
    their read/modify/replace sequence.
    """
    _assert_registration_target_safe(output, manifest_path)
    _assert_no_local_artifact_bytes(output, artifact_name, access_class)
    payload = _load_manifest(manifest_path)
    siblings = _registration_sibling_manifests(output, manifest_path)
    try:
        existing_kind = manifest_kind(payload, manifest_path=manifest_path)
    except ManifestAccessError as exc:
        raise HashOnlyRegistrationError(str(exc)) from exc
    if existing_kind != MICRODATA_RELEASE_KIND and (
        payload.get("kind") is not None or has_file_entries(payload)
    ):
        raise HashOnlyRegistrationError(
            f"{manifest_path} is a {existing_kind} manifest; hash-only "
            "registrations belong in a kind: microdata_release manifest."
        )

    route_context = dict(payload)
    if _text(source_page):
        route_context["source_page"] = source_page
    if not _access_route(entry, route_context):
        raise HashOnlyRegistrationError(
            "A hash-only registration must record how the bytes are reached; "
            "pass --access-route, --source-url, --doi, or --source-page."
        )

    _assert_manifest_identity(payload, manifest_path, "source_id", source_id)
    _assert_manifest_identity(payload, manifest_path, "package_id", package_id)
    files = payload.get("files")
    if files is not None and not isinstance(files, dict):
        raise HashOnlyRegistrationError(
            f"{manifest_path} files must be a mapping; it is a "
            f"{type(files).__name__}. Chronicle will not write into a manifest "
            "it cannot read."
        )
    _assert_registration_manifest_valid(
        payload,
        manifest_path,
        output=output,
        kind=existing_kind,
    )
    for sibling_path, sibling in siblings.items():
        try:
            sibling_kind = manifest_kind(sibling, manifest_path=sibling_path)
        except ManifestAccessError as exc:
            raise HashOnlyRegistrationError(str(exc)) from exc
        _assert_registration_manifest_valid(
            sibling,
            sibling_path,
            output=output,
            kind=sibling_kind,
        )

    _assert_no_archived_identity(
        payload, manifest_path, artifact_name, access_class, sha256=checksum
    )
    for sibling_path, sibling in siblings.items():
        _assert_no_archived_identity(
            sibling, sibling_path, artifact_name, access_class, sha256=checksum
        )

    try:
        vintage_key = resolve_vintage_key(files or {}, year)
    except AmbiguousVintageKeyError as exc:
        raise HashOnlyRegistrationError(f"{manifest_path}: {exc}") from exc
    key = vintage_key if vintage_key is not None else year

    _replace_blank_manifest_text(payload, "source_id", source_id)
    _replace_blank_manifest_text(payload, "package_id", package_id)
    payload["kind"] = MICRODATA_RELEASE_KIND
    _replace_blank_manifest_text(
        payload, "dataset", _text(dataset) or f"{source_id}_{package_id}"
    )
    if _text(publisher):
        _replace_blank_manifest_text(payload, "publisher", publisher)
    if _text(source_page):
        _replace_blank_manifest_text(payload, "source_page", source_page)
    if _text(table):
        _replace_blank_manifest_text(payload, "table", table)
    if payload.get("files") is None:
        payload["files"] = {}

    entries = _existing_entries(payload["files"], key)
    wanted = filename_key(artifact_name)
    # Two passes keep re-registering an existing pin idempotent after a reissue
    # added another entry for the same filename.
    replaced = False
    for index, existing in enumerate(entries):
        if not isinstance(existing, Mapping):
            continue
        if filename_key(existing.get("filename")) != wanted:
            continue
        if _text(existing.get("sha256")) == checksum:
            entries[index] = entry
            replaced = True
            break
    if not replaced:
        superseded = [
            existing
            for existing in entries
            if isinstance(existing, Mapping)
            and filename_key(existing.get("filename")) == wanted
        ]
        if superseded and not allow_reissue:
            raise HashOnlyRegistrationError(
                f"{manifest_path} already registers {artifact_name!r} for "
                f"{key!r} with sha256={superseded[0].get('sha256')!r}. Different "
                "bytes are a new publisher release, not a pin replacement; "
                "pass --allow-reissue to register both."
            )
        entries.append(entry)

    payload["files"][key] = sorted(entries, key=_entry_sort_key)
    for field, expected in (("source_id", source_id), ("package_id", package_id)):
        if _text(payload.get(field)) != expected:
            raise HashOnlyRegistrationError(
                f"Refusing to persist {manifest_path}: final {field} is "
                f"{payload.get(field)!r}, expected {expected!r}."
            )
    _assert_registration_manifest_valid(
        payload,
        manifest_path,
        output=output,
        kind=MICRODATA_RELEASE_KIND,
        final=True,
    )
    # Import lazily: artifact commands also depend on this module. Their owner
    # validator compares effective identities both within and across manifests.
    from chronicle.artifacts import (
        RecordedR2LocatorError,
        SourceArtifactManifestError,
        _assert_no_package_microdata_identities,
        _assert_package_file_owner_identities_agree,
        _validated_recorded_r2,
    )

    proposed_manifests = {str(path): sibling for path, sibling in siblings.items()}
    proposed_manifests[str(manifest_path)] = payload
    try:
        # Effective-digest resolution deliberately leaves malformed locators
        # to per-entry validation. Validate each one explicitly so its fallback
        # cannot conceal contradictory or incomplete recorded provenance.
        for name, proposed in proposed_manifests.items():
            path = Path(name)
            kind, _error = safe_manifest_kind(proposed, manifest_path=path)
            for vintage, _index, spec in iter_manifest_entries(proposed):
                locator_arguments = {
                    "manifest_path": path,
                    "year": vintage,
                    "source_id": proposed.get("source_id"),
                    "package_id": proposed.get("package_id"),
                    "bind_registration_identity": kind == MICRODATA_RELEASE_KIND,
                }
                locator = _validated_recorded_r2(spec, **locator_arguments)
                if locator is not None and (
                    locator.filename != spec.get("filename")
                    or (
                        spec.get("sha256") is not None
                        and locator.sha256 != spec["sha256"]
                    )
                ):
                    raise RecordedR2LocatorError(
                        f"{path} entry {vintage!r}: recorded_r2_identity_mismatch"
                    )
                # Historical objects may have different filenames and digests,
                # but each archived block must still locate one valid object.
                for index, previous in enumerate(recorded_previous_r2(spec)):
                    try:
                        _validated_recorded_r2(
                            {"storage": {"r2": previous}}, **locator_arguments
                        )
                    except SourceArtifactManifestError as exc:
                        raise RecordedR2LocatorError(
                            f"{path} entry {vintage!r} "
                            f"storage.previous_r2[{index}]: {exc}"
                        ) from exc
        # Registration checks recorded identities without opening artifacts.
        _assert_package_file_owner_identities_agree(
            proposed_manifests, check_local_files=False
        )
        # The same package-wide sweep publish-raw, inventory-artifacts and
        # fetch-artifact run: a public table entry carrying a release's
        # filename, checksum or archived R2 identity is that release under a
        # second record, whichever manifest this registration writes.
        _assert_no_package_microdata_identities(proposed_manifests)
    except SourceArtifactManifestError as exc:
        raise HashOnlyRegistrationError(
            f"{output} is not a valid package directory; refusing to persist "
            f"{manifest_path}: {exc}. Fix the manifests "
            "before registering into that directory."
        ) from exc
    return payload, replaced


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


def _assert_no_local_artifact_bytes(
    output: Path,
    artifact_name: str,
    access_class: str,
) -> None:
    """Refuse any actual path alias of a hash-only artifact filename."""
    local_path = matching_directory_entry(output, artifact_name)
    if local_path is None:
        return
    requested = "" if local_path.name == artifact_name else f" ({artifact_name!r})"
    raise HashOnlyRegistrationError(
        f"Refusing to register {local_path.name!r}{requested} hash-only while "
        f"its bytes are present at {local_path}. A {access_class} artifact's "
        "bytes must not live in a Chronicle store."
    )


def _registration_manifest_errors(
    payload: Mapping[str, Any],
    *,
    output: Path,
    kind: str,
) -> list[str]:
    """Return complete entry and manifest-level validation errors."""
    errors = list(validate_manifest_files(payload))
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return errors
    for existing_key, spec in files.items():
        for existing in iter_file_specs(spec, kind=kind):
            existing_name = (
                existing.get("filename") if isinstance(existing, Mapping) else None
            )
            exists = (
                is_bare_filename(existing_name)
                and matching_directory_entry(output, existing_name) is not None
            )
            errors.extend(
                f"{existing_key!r}/{existing_name}: {code}"
                for code in validate_file_entry(
                    existing,
                    kind=kind,
                    manifest=payload,
                    local_file_exists=exists,
                )
            )
    return errors


def _assert_registration_manifest_valid(
    payload: Mapping[str, Any],
    manifest_path: Path,
    *,
    output: Path,
    kind: str,
    final: bool = False,
) -> None:
    errors = _registration_manifest_errors(payload, output=output, kind=kind)
    if not errors:
        return
    action = "persist" if final else "register into"
    raise HashOnlyRegistrationError(
        f"{manifest_path} is not a valid {kind} manifest; refusing to {action} "
        f"it: {'; '.join(errors)}. Fix it by hand before registering; "
        "inventory-artifacts reports the same codes."
    )


def _replace_blank_manifest_text(payload: dict[str, Any], key: str, value: Any) -> None:
    """Fill a missing, null, empty, or whitespace-only manifest field."""
    replacement = _text(value)
    if _text(payload.get(key)) is None and replacement is not None:
        payload[key] = replacement


def _registration_sibling_manifests(
    output: Path, manifest_path: Path
) -> dict[Path, dict[str, Any]]:
    """Load every physically distinct sibling after alias validation."""
    _assert_registration_target_safe(output, manifest_path)
    return {
        path: _load_manifest(path)
        for path in package_manifest_paths(output)
        if path != manifest_path
    }


def _assert_registration_target_safe(output: Path, manifest_path: Path) -> None:
    """Refuse symlinked targets and normalized aliases before reading them."""
    lexical_output = output if output.is_absolute() else Path.cwd() / output
    current = Path(lexical_output.anchor)
    symlink_component = None
    for component in lexical_output.parts[1:]:
        if component == "..":
            current = current.parent
            continue
        current /= component
        if current.is_symlink():
            symlink_component = current
            break
    if symlink_component is not None:
        raise HashOnlyRegistrationError(
            f"Refusing registration output {output}: path component "
            f"{symlink_component} is a symbolic link. Registration writes only "
            "through a physical package-directory path."
        )
    if manifest_path.is_symlink():
        raise HashOnlyRegistrationError(
            f"Refusing manifest target {manifest_path}: it is a symbolic link. "
            "Registration never follows a manifest target outside its package."
        )
    by_name: dict[str, Path] = {}
    for path in package_manifest_paths(output):
        key = filename_key(path.name)
        previous = by_name.get(key)
        if previous is not None and previous != path:
            raise HashOnlyRegistrationError(
                f"{previous} and {path} have the same normalized manifest "
                "name. Keep exactly one physical spelling before registering."
            )
        by_name[key] = path
    alias = by_name.get(filename_key(manifest_path.name))
    if alias is not None and alias != manifest_path:
        raise HashOnlyRegistrationError(
            f"Refusing manifest target {manifest_path}: existing {alias} has "
            "the same normalized manifest name. Selecting one spelling would "
            "hide the other."
        )


def _registration_lock_path(output: Path) -> Path:
    """Return the persistent package-wide lock file outside the package tree."""
    # Use one identity before and after the package directory exists. NFC plus
    # case-folding models the most restrictive supported filesystem, so paths
    # that may be one directory on macOS/Windows always serialize; distinct
    # case-sensitive paths may harmlessly share a lock. ``resolve`` still
    # collapses existing symlink/parent aliases to their physical route.
    canonical = unicodedata.normalize(
        "NFC", str(output.resolve(strict=False))
    ).casefold()
    identity = os.fsencode(canonical)
    digest = hashlib.sha256(identity).hexdigest()
    return (
        Path(tempfile.gettempdir())
        / "policyengine-chronicle-manifest-locks"
        / f"{digest}.lock"
    )


@contextmanager
def _registration_lock(output: Path) -> Iterator[None]:
    """Hold the package-wide manifest lock for one read/modify/replace."""
    lock_path = _registration_lock_path(output)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_replace_manifest(manifest_path: Path, document: str) -> None:
    """Durably replace a manifest from a same-directory temporary file."""
    _assert_registration_target_safe(manifest_path.parent, manifest_path)
    mode = (
        stat.S_IMODE(manifest_path.stat().st_mode) if manifest_path.exists() else 0o644
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=manifest_path.parent,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
    )
    temporary_exists = True
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as temporary:
            descriptor = -1
            temporary.write(document)
            temporary.flush()
            os.fsync(temporary.fileno())
        _assert_registration_target_safe(manifest_path.parent, manifest_path)
        os.replace(temporary_name, manifest_path)
        temporary_exists = False
        directory_fd = os.open(manifest_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _hash_only_attestation(
    *,
    hash_source: Any,
    attested_by: Any,
    attestation_evidence: Any,
    pinned_from: Any,
    verified_at: Any,
    access_class: str,
) -> dict[str, Any]:
    """Validate and return the attestation fields of a hash-only registration.

    Chronicle never fetched the bytes, so the checksum is always the
    consumer's: attested against bytes it holds (``consumer_attested``, with
    evidence and a date) or transcribed from a reviewed pin in its repository
    (``consumer_pin``, with the repository, path and commit and no
    verification date).
    """
    try:
        source = normalize_hash_source(hash_source, allowed=HASH_ONLY_HASH_SOURCES)
    except ManifestAccessError as exc:
        raise HashOnlyRegistrationError(
            f"A {access_class} registration must record how its checksum is "
            f"known: {exc} Chronicle holds no bytes for it, so "
            f"{HASH_SOURCE_CHRONICLE_FETCH!r} does not apply."
        ) from exc
    attester = _text(attested_by)
    if not attester:
        raise HashOnlyRegistrationError(
            f"A {source} registration must name the consumer that attests the "
            "checksum; pass --attested-by."
        )
    if _is_chronicle(attester):
        raise HashOnlyRegistrationError(
            f"A {source} registration is attested by the consumer whose pin it "
            f"transcribes, never by {CHRONICLE_ATTESTER!r}: Chronicle holds no "
            "bytes for it and verified nothing. Pass --attested-by with the "
            "consumer's name."
        )
    fields: dict[str, Any] = {"hash_source": source, "attested_by": attester}
    if source == HASH_SOURCE_CONSUMER_ATTESTED:
        if not _text(attestation_evidence):
            raise HashOnlyRegistrationError(
                "A consumer_attested registration must record the consumer's "
                "attestation evidence; pass --attestation-evidence."
            )
        if not _text(verified_at):
            raise HashOnlyRegistrationError(
                "A consumer_attested registration must record when the checksum "
                "was verified against the bytes; pass --verified-at."
            )
        fields["attestation_evidence"] = str(attestation_evidence)
        fields["verified_at"] = str(verified_at)
        return fields
    if pinned_from is None or not isinstance(pinned_from, Mapping):
        raise HashOnlyRegistrationError(
            "A consumer_pin registration must record where the pin was read: "
            "pass --pinned-from-repository, --pinned-from-path and "
            "--pinned-from-commit."
        )
    errors = _pinned_from_errors(pinned_from)
    if errors:
        raise HashOnlyRegistrationError(
            "A consumer_pin registration's pinned_from must carry the "
            f"repository, path and a 40-hex commit: {', '.join(errors)}."
        )
    if _text(verified_at):
        raise HashOnlyRegistrationError(
            "A consumer_pin registration carries no verified_at: Chronicle did "
            "not verify the checksum against bytes, it transcribed the "
            "consumer's pin. Record the pin's commit instead."
        )
    fields["pinned_from"] = {
        field: str(pinned_from[field]).strip() for field in PINNED_FROM_FIELDS
    }
    return fields


def _assert_no_archived_identity(
    payload: Mapping[str, Any],
    manifest_path: Path,
    artifact_name: str,
    access_class: str,
    *,
    sha256: str | None = None,
) -> None:
    """Refuse to register hash-only an identity the manifest holds as public.

    A public entry may have been archived: its object sits in the raw bucket
    under ``storage.r2`` (or in ``storage.previous_r2`` once revised).
    Replacing that entry with a hash-only one would leave the bytes in a
    Chronicle store with nothing recording them, and inventory would report
    the tree clean. The transition is refused until the public entry, and the
    object it names, have been explicitly removed. The identity is matched by
    case-folded bare filename and by digest: the same bytes archived under
    another name are the same artifact.
    """
    wanted = filename_key(artifact_name)
    for key, _index, existing in iter_manifest_entries(payload):
        if not isinstance(existing, Mapping):
            continue
        existing_name = existing.get("filename")
        same_name = existing_name is not None and filename_key(existing_name) == wanted
        same_bytes = sha256 is not None and _text(existing.get("sha256")) == sha256
        archived_matches = [
            locator
            for name, digest, locator in _recorded_object_identities(existing)
            if name == wanted or (sha256 is not None and digest == sha256)
        ]
        if not (same_name or same_bytes or archived_matches):
            continue
        recorded = [
            str(block.get("uri") or block.get("key") or block)
            for block in (recorded_r2(existing), *recorded_previous_r2(existing))
            if isinstance(block, Mapping)
        ]
        if same_name:
            how = ""
        elif same_bytes:
            how = f" (the same bytes as {artifact_name!r})"
        else:
            how = f" (whose archived filename or checksum matches {artifact_name!r})"
        if recorded:
            raise HashOnlyRegistrationError(
                f"{manifest_path} records the R2 object(s) {recorded} for "
                f"{existing_name!r}{how} ({key!r}, "
                f"access={safe_entry_access(existing)!r}). Registering it "
                f"{access_class} would leave those bytes in a Chronicle store "
                "with nothing recording them. Remove the object and its "
                "storage record explicitly first; Chronicle will not reclassify "
                "an archived release in place."
            )
        if not is_hash_only(safe_entry_access(existing)):
            raise HashOnlyRegistrationError(
                f"{manifest_path} already registers {existing_name!r}{how} "
                f"({key!r}) as access={safe_entry_access(existing)!r}. A change "
                f"of access class to {access_class!r} is an explicit decision: "
                "remove the public entry by hand, then register the release."
            )


def _registration_manifest_path(output: Path, manifest_filename: Any) -> Path:
    """Return the manifest a registration records into, refusing a stray one.

    Mirrors ``fetch-artifact --manifest``: the name is a bare manifest name
    inside the package directory, and the default name is refused beside a
    package's named manifests (PolicyEngine/chronicle#225), so a
    registration never creates a manifest no sweep or package reads.
    """
    name = str(manifest_filename).strip() if manifest_filename is not None else ""
    if not is_manifest_filename(name):
        raise HashOnlyRegistrationError(
            f"--manifest must name {DEFAULT_MANIFEST_FILENAME} or "
            f"manifest_<package>.yaml inside the package directory, not "
            f"{manifest_filename!r}."
        )
    manifest_path = output / name
    _assert_registration_target_safe(output, manifest_path)
    if name == DEFAULT_MANIFEST_FILENAME and not manifest_path.exists():
        siblings = [
            path.name for path in package_manifest_paths(output) if path.name != name
        ]
        if siblings:
            raise HashOnlyRegistrationError(
                f"{output} keeps {', '.join(siblings)} and no "
                f"{DEFAULT_MANIFEST_FILENAME}; pass --manifest to name the "
                "manifest this registration records into rather than creating "
                f"{DEFAULT_MANIFEST_FILENAME} beside them."
            )
    return manifest_path


def _assert_manifest_identity(
    payload: Mapping[str, Any],
    manifest_path: Path,
    key: str,
    value: str,
) -> None:
    """Refuse to register into a manifest that identifies a different source."""
    existing = _text(payload.get(key))
    if existing is not None and existing != value:
        raise HashOnlyRegistrationError(
            f"{manifest_path} declares {key}={existing!r}; refusing to register "
            f"{key}={value!r} into it."
        )


def _entry_sort_key(entry: Any) -> tuple[str, str]:
    """Return a deterministic sort key for a registration entry."""
    if not isinstance(entry, Mapping):
        return ("", "")
    return (_text(entry.get("filename")) or "", _text(entry.get("sha256")) or "")


def _registration_entry(**values: Any) -> dict[str, Any]:
    """Build a deterministic, field-ordered registration entry."""
    entry: dict[str, Any] = {}
    for key in _REGISTRATION_FIELD_ORDER:
        value = values.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        entry[key] = value
    return entry


def _existing_entries(files: Any, key: Any) -> list[Any]:
    """Return the existing file entries under a vintage key as a mutable list."""
    if not isinstance(files, dict):
        return []
    spec = files.get(key)
    if spec is None:
        return []
    if isinstance(spec, list):
        return list(spec)
    return [spec]


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a manifest mapping, or an empty mapping when absent.

    Read strictly: a document with duplicate keys, or one that is not a
    mapping, is a refusal rather than something to record into.
    """
    if not manifest_path.exists():
        return {}
    try:
        payload = load_manifest_document(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HashOnlyRegistrationError(
            f"{manifest_path} is not valid YAML: {exc}"
        ) from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise HashOnlyRegistrationError(f"Manifest must be a mapping: {manifest_path}")
    return payload


def _text(value: Any) -> str | None:
    """Return a non-empty stripped string, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _dedupe(values: Iterable[str]) -> list[str]:
    """Return values with duplicates removed, preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


__all__ = [
    "ACCESS_CLASSES",
    "ACCESS_LICENSED",
    "ACCESS_PUBLIC",
    "ACCESS_RESTRICTED",
    "AmbiguousVintageKeyError",
    "ArtifactFilenameError",
    "ArtifactRegistrationReport",
    "CHRONICLE_ATTESTER",
    "DEFAULT_ACCESS",
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_MANIFEST_KIND",
    "HASH_ONLY_HASH_SOURCES",
    "HASH_SOURCES",
    "HASH_SOURCE_CHRONICLE_FETCH",
    "HASH_SOURCE_CONSUMER_ATTESTED",
    "HASH_SOURCE_CONSUMER_PIN",
    "HashOnlyRegistrationError",
    "ListSpecRejected",
    "MANIFEST_KINDS",
    "StrictManifestLoader",
    "load_manifest_document",
    "MICRODATA_RELEASE_KIND",
    "ManifestAccessError",
    "ManifestKindError",
    "MicrodataReleaseNotParseableError",
    "PINNED_FROM_FIELDS",
    "PUBLISHER_TABLE_KIND",
    "bare_filename",
    "entry_access",
    "filename_key",
    "has_file_entries",
    "hash_only_registrations",
    "is_bare_filename",
    "is_hash_only",
    "is_manifest_filename",
    "is_microdata_release",
    "iter_directory_entries",
    "iter_file_specs",
    "iter_manifest_entries",
    "matching_directory_entry",
    "manifest_kind",
    "normalize_access",
    "normalize_hash_source",
    "package_manifest_paths",
    "recorded_previous_r2",
    "recorded_r2",
    "records_r2_object",
    "register_hash_only_artifact",
    "registration_id",
    "resolve_vintage_key",
    "safe_entry_access",
    "safe_manifest_kind",
    "stores_bytes",
    "strict_entry_access",
    "validate_file_entry",
    "validate_manifest_files",
    "validate_manifest_vintages",
    "validate_package_directory",
    "vintage_key_forms",
]
