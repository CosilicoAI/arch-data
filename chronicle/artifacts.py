"""Source artifact acquisition and storage helpers for Chronicle."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import posixpath
import re
import shlex
import sqlite3
import stat
import subprocess
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
import yaml

from chronicle.database import (
    CHRONICLE_DB_FILENAME,
    CHRONICLE_DB_FILENAMES,
    LEGACY_CHRONICLE_DB_FILENAME,
)
from chronicle.env import env_value
from chronicle.epoch import EMIT_EPOCH, Epoch, canonicalize_key, hash_domain
from chronicle.licences import (
    REDISTRIBUTABLE_LICENCES,
    is_redistributable_licence,
    licence_evidence_errors,
)
from chronicle.registration import (
    ACCESS_CLASSES,
    ACCESS_PUBLIC,
    CHRONICLE_ATTESTER,
    HASH_SOURCE_CHRONICLE_FETCH,
    MANIFEST_KINDS,
    MICRODATA_RELEASE_KIND,
    AmbiguousVintageKeyError,
    ArtifactFilenameError as RegistrationArtifactFilenameError,
    ListSpecRejected,
    ManifestAccessError,
    ManifestKindError,
    _recorded_object_identities,
    bare_filename as registration_bare_filename,
    filename_key,
    has_file_entries,
    hash_only_registrations,
    is_bare_filename,
    is_hash_only,
    is_manifest_filename,
    iter_directory_entries,
    iter_file_specs,
    iter_manifest_entries,
    load_manifest_document,
    manifest_kind as normalize_manifest_kind,
    matching_directory_entry,
    normalize_access,
    package_manifest_paths,
    _assert_registration_target_safe,
    _registration_lock,
    resolve_vintage_key,
    safe_entry_access,
    safe_manifest_kind,
    _split_r2_uri,
    validate_file_entry,
    validate_manifest_files,
    validate_package_directory,
)


R2_RAW_BUCKET_ENV = "CHRONICLE_R2_RAW_BUCKET"
R2_DERIVED_BUCKET_ENV = "CHRONICLE_R2_DERIVED_BUCKET"

# The bucket defaults stay at their ledger-era names. Archived witness records
# pin raw R2 URLs by hash, so ledger-raw and ledger-derived are preserved
# read-only forever and no recorded manifest URI is ever rewritten. The env
# vars exist so the cutover in docs/storage-architecture.md can be rehearsed,
# and so flipping to chronicle-raw/chronicle-derived is a default change rather
# than a code change (PolicyEngine/chronicle#143, mechanism 3).
DEFAULT_R2_RAW_BUCKET = "ledger-raw"
DEFAULT_R2_DERIVED_BUCKET = "ledger-derived"
DEFAULT_R2_PREFIX = "raw"
DEFAULT_R2_DERIVED_PREFIX = "derived"

#: ``RawArtifactPublishEntry.skipped`` prefix for a licensed or restricted
#: registration: nothing to upload, because no Chronicle store holds its bytes.
HASH_ONLY_SKIP_PREFIX = "hash_only_access:"

# Most packages keep one manifest.yaml. Publisher directories that feed several
# source packages keep one manifest each -- db/data/irs_soi/ira_contributions
# holds manifest_traditional_source_package.yaml beside the Roth one -- so the
# name is an input, not a constant, wherever a caller addresses a package.
DEFAULT_MANIFEST_FILENAME = "manifest.yaml"

#: Wrangler invocation used for R2 uploads unless a caller overrides it.
DEFAULT_WRANGLER_COMMAND = "npx wrangler"

MICRODATA_STAGING_DIR_ENV = "CHRONICLE_MICRODATA_STAGING_DIR"
#: Where a public microdata release's bytes are staged before upload. Outside
#: the repository by construction: public microdata is archived in R2, never
#: committed beside its manifest (docs/adr-chronicle-raw-microdata-identity.md).
DEFAULT_MICRODATA_STAGING_DIR = (
    Path.home() / ".cache" / "policyengine-chronicle" / "microdata-staging"
)


def default_microdata_staging_dir() -> Path:
    """Resolve the staging root: ``$CHRONICLE_MICRODATA_STAGING_DIR`` or default."""
    return Path(
        env_value(MICRODATA_STAGING_DIR_ENV, default=DEFAULT_MICRODATA_STAGING_DIR)
    )


def microdata_staging_path(
    *,
    staging_dir: str | Path | None,
    source_id: str,
    package_id: str,
    year: Any,
    sha256: str,
    filename: str,
) -> Path:
    """Return the transient, content-addressed staging path for release bytes.

    Mirrors the raw R2 key shape so a staged file is addressed by the same
    identity as the object it becomes.
    """
    root = (
        Path(staging_dir)
        if staging_dir is not None
        else (default_microdata_staging_dir())
    )
    return root / source_id / package_id / str(year) / sha256 / Path(filename).name


def _validated_microdata_staging_destination(
    destination: Path, *, package_dir: Path
) -> Path:
    """Resolve an external staging file before any lock or publisher access.

    The registration path guard walks the original spelling before resolving
    it, so a symlink followed by ``..`` cannot disappear from the check. Give
    it the full content-addressed destination to cover identity components and
    the artifact filename as well as the supplied staging directory.
    """
    try:
        _assert_registration_target_safe(destination.parent, destination)
        resolved = destination.resolve()
        package_root = package_dir.resolve()
        repository_root = Path(__file__).resolve().parents[1]
        for root in (package_root, repository_root):
            if resolved.is_relative_to(root):
                raise ManifestAccessError(
                    f"Microdata staging destination {destination} is inside "
                    f"the repository or package tree {root}. Choose an "
                    "external staging directory."
                )
        for ancestor in resolved.parents:
            git_entry = ancestor / ".git"
            bare_repository = (
                (ancestor / "HEAD").is_file()
                and (ancestor / "objects").is_dir()
                and (ancestor / "refs").is_dir()
            )
            if git_entry.exists() or git_entry.is_symlink() or bare_repository:
                raise ManifestAccessError(
                    f"Microdata staging destination {destination} is inside "
                    f"the Git repository {ancestor}. Choose an external "
                    "staging directory."
                )
            if package_manifest_paths(ancestor):
                raise ManifestAccessError(
                    f"Microdata staging destination {destination} is inside "
                    f"the package directory {ancestor}. Choose an external "
                    "staging directory."
                )
    except (OSError, ValueError, ManifestAccessError) as exc:
        raise ManifestAccessError(
            f"Microdata staging destination {destination} is unsafe: {exc}"
        ) from exc
    return resolved


def _manifest_path(output: Path, manifest_filename: str) -> Path:
    """Return the named manifest inside ``output``.

    The name is a filename, not a path: it selects among the manifests a
    package directory keeps, and must not reach outside it.
    """
    name = str(manifest_filename)
    if not is_bare_filename(name) or any(character in name for character in "*?[]"):
        raise ManifestNameError(
            "Manifest must name a file inside the package directory, not "
            f"{manifest_filename!r}."
        )
    if not is_manifest_filename(name):
        raise ManifestNameError(
            f"Manifest must be named {DEFAULT_MANIFEST_FILENAME} or "
            f"manifest_<package>.yaml, not {manifest_filename!r}: the sweeps "
            "address a package's manifests by those names, and a manifest "
            "under any other name is invisible to them."
        )
    return output / name


def _package_manifests(
    output: Path,
    manifest_path: Path,
    existing_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return every manifest the package directory keeps, by path.

    The byte boundary is the file in the directory, not whichever manifest a
    fetch selected. A malformed sibling is therefore a pre-I/O refusal: until
    Chronicle can read every owner, it cannot safely overwrite shared bytes.
    """
    try:
        paths = package_manifest_paths(output)
    except ValueError as error:
        # Explicit sweep selectors still inspect every sibling registry. Keep
        # discovery refusals in the shared exception family both CLIs report.
        raise MalformedManifestError(str(error)) from error
    by_name: dict[str, Path] = {}
    for path in paths:
        key = filename_key(path.name)
        previous = by_name.get(key)
        if previous is not None and previous != path:
            raise AmbiguousManifestError(
                f"{previous} and {path} have the same normalized manifest name "
                f"{key!r}. Physically distinct manifest aliases can hide one "
                "another's registrations; keep exactly one spelling."
            )
        by_name[key] = path
    selected_alias = by_name.get(filename_key(manifest_path.name))
    if selected_alias is not None and selected_alias != manifest_path:
        raise AmbiguousManifestError(
            f"{manifest_path} and existing {selected_alias} have the same "
            "normalized manifest name. Selecting one spelling would hide the "
            "other's registrations; address the existing manifest or remove "
            "the duplicate."
        )

    manifests: dict[str, dict[str, Any]] = {str(manifest_path): existing_manifest}
    for path in paths:
        if path == manifest_path:
            continue
        sibling = _read_manifest(path)
        _manifest_files(sibling, path)
        manifests[str(path)] = sibling
    return manifests


def _root_manifest_paths(root: Path, manifest_filename: str) -> list[Path]:
    """Return the manifests a root sweep addresses.

    The default is package discovery, not one literal filename: both YAML
    extensions and every ``manifest_<package>`` sibling participate. A caller
    that supplies another filename keeps the historical exact-name override.
    """
    selected_name = _manifest_path(Path(), manifest_filename).name
    if selected_name != DEFAULT_MANIFEST_FILENAME:
        candidates = [path for path in root.rglob("*") if path.name == selected_name]
    else:
        candidates = [
            path for path in root.rglob("*") if is_manifest_filename(path.name)
        ]
    for path in candidates:
        _require_regular_manifest_file(path)
    return sorted(candidates)


def _assert_no_hash_only_bytes(
    manifests: Mapping[str, dict[str, Any]],
    *,
    sha256: str,
    filename: str,
    what: str,
) -> None:
    """Refuse to archive bytes any manifest in the directory registers hash-only.

    The same bytes under another name are the same gated artifact.
    """
    for name, key, entry in hash_only_registrations(manifests, sha256=sha256):
        raise ManifestAccessError(
            f"{name} registers {entry.get('filename')!r} for {key!r} as "
            f"access={safe_entry_access(entry)!r} with sha256={sha256}; {what} "
            "are those exact bytes. A gated artifact is not archived under "
            f"another name ({filename!r}); keep the hash-only registration."
        )


def _assert_no_microdata_identity(
    manifests: Mapping[str, dict[str, Any]],
    *,
    package_dir: Any,
    filename: str,
    digests: Iterable[str | None],
    require_digest: bool = False,
) -> None:
    """Refuse table bytes identified as current or archived package microdata.

    Readers require a recorded digest before any artifact I/O. Fetch can check
    an unknown digest after downloading into memory, before persisting bytes.
    ``package_dir`` also accepts the source reader's ZIP resource directory.
    Callers validate complete manifests before using this identity classifier.
    """
    wanted_name = filename_key(filename)
    wanted_digests = {digest for digest in digests if digest}
    for name, key, _index, entry in iter_directory_entries(manifests):
        path = package_dir.joinpath(Path(name).name)
        kind, _error = safe_manifest_kind(manifests[name], manifest_path=path)
        if kind != MICRODATA_RELEASE_KIND or not isinstance(entry, dict):
            continue
        identities = [
            (filename_key(entry.get("filename")), entry.get("sha256")),
            *(
                (recorded_name, digest)
                for recorded_name, digest, _ in _recorded_object_identities(entry)
            ),
        ]
        if any(
            recorded_name == wanted_name or digest in wanted_digests
            for recorded_name, digest in identities
        ):
            raise ManifestAccessError(
                f"{path} registers a microdata release for {key!r} sharing "
                f"the filename or checksum of {filename!r}, including recorded "
                "R2 history. No source package reads, fetches, caches, or "
                "parses microdata through another manifest, even when public."
            )
        if require_digest and not wanted_digests:
            raise ManifestAccessError(
                f"{package_dir} entry {filename!r} has no recorded digest "
                f"beside microdata release {name}. Record its checksum before "
                "reading, fetching, or caching bytes whose identity "
                "cannot yet exclude that release."
            )


#: Refusal code for local bytes a sibling microdata release already identifies.
#: Publish and inventory report codes rather than raising, so a package sweep
#: names every defect before anything is uploaded or rewritten.
MICRODATA_IDENTITY_CODE = "bytes_identified_by_microdata_release"


def _microdata_identity_errors(
    manifests: Mapping[str, dict[str, Any]] | None,
    *,
    package_dir: Any,
    filename: Any,
    digests: Iterable[str | None],
) -> list[str]:
    """Return the refusal codes for bytes a release sibling identifies.

    The reporting form of :func:`_assert_no_microdata_identity`, for the
    publish and inventory entry passes that accumulate codes instead of
    raising. A filename Chronicle cannot address in the package directory is
    left to the entry-shape codes that already refuse it.
    """
    if manifests is None:
        return []
    try:
        _assert_no_microdata_identity(
            manifests,
            package_dir=package_dir,
            filename=filename if is_bare_filename(filename) else "",
            digests=digests,
        )
    except ManifestAccessError as error:
        return [f"{MICRODATA_IDENTITY_CODE}:{error}"]
    return []


def _recorded_identity_aliases(spec: Any) -> tuple[set[str], set[str]]:
    """Return the filename keys and digests an entry's R2 objects record.

    The current object and every archived one: an immutable object keeps
    identifying the bytes it holds after the entry that recorded it changes
    filename or checksum, so a registration's identity is its whole history.
    Callers that must refuse an unidentified entry keep their own digest
    requirement: these aliases never stand in for a declared checksum.
    """
    names: set[str] = set()
    digests: set[str] = set()
    for recorded_name, digest, _locator in _recorded_object_identities(spec):
        if recorded_name:
            names.add(recorded_name)
        if digest:
            digests.add(digest)
    return names, digests


def _assert_no_table_claims_release_identity(
    manifests: Mapping[str, dict[str, Any]],
    *,
    package_dir: Any,
    filename: str,
    digests: Iterable[str | None],
) -> None:
    """Refuse a release whose identity a public table sibling already claims.

    The mirror of :func:`_assert_no_microdata_identity`, for the direction that
    creates the contradiction rather than finding it: registering these bytes
    as a release while a publisher table registers them as its own would leave
    a directory publish and inventory refuse outright. One artifact has one
    registration, and the release fetch is not entitled to overrule the
    existing one silently. A hash-only sibling keeps the gated-bytes refusal
    :func:`_assert_no_hash_only_bytes` already reports.
    """
    wanted_name = filename_key(filename)
    wanted_digests = {digest for digest in digests if digest}
    for name, key, _index, entry in iter_directory_entries(manifests):
        sibling_path = package_dir.joinpath(Path(name).name)
        kind, _error = safe_manifest_kind(manifests[name], manifest_path=sibling_path)
        if kind == MICRODATA_RELEASE_KIND or not isinstance(entry, dict):
            continue
        if is_hash_only(safe_entry_access(entry)):
            continue
        identities = [
            (filename_key(entry.get("filename")), entry.get("sha256")),
            *(
                (recorded_name, digest)
                for recorded_name, digest, _ in _recorded_object_identities(entry)
            ),
        ]
        if any(
            recorded_name == wanted_name or digest in wanted_digests
            for recorded_name, digest in identities
        ):
            raise ManifestAccessError(
                f"{sibling_path} registers {entry.get('filename')!r} for "
                f"{key!r} sharing the filename or checksum of the microdata "
                f"release {filename!r}, including recorded R2 history. "
                "Reconcile that registration before registering these bytes as "
                "a release; no package keeps one artifact under two records."
            )


def _assert_no_package_microdata_identities(
    manifests: Mapping[str, dict[str, Any]],
) -> None:
    """Refuse a directory whose public table entry a release sibling identifies.

    Publish and inventory sweep one manifest at a time, but the physical byte
    belongs to the directory: a table registration carrying a release's
    filename, checksum, or archived R2 identity is that gated-by-provenance
    release under another record, whichever manifest declares it. Checked as a
    package boundary, before any entry's bytes are read, uploaded or rewritten,
    exactly as the source-package readers check it before returning bytes.

    Only public non-release entries are swept here. A hash-only registration
    colliding with a public release keeps its existing
    :func:`validate_package_directory` codes, which this never replaces.
    """
    for name, vintage, _index, entry in iter_directory_entries(manifests):
        if not isinstance(entry, dict):
            continue
        manifest_path = Path(name)
        kind, _kind_error = safe_manifest_kind(
            manifests[name], manifest_path=manifest_path
        )
        if kind == MICRODATA_RELEASE_KIND:
            continue
        if is_hash_only(safe_entry_access(entry)):
            continue
        filename = entry.get("filename")
        # An immutable object stays archived after the entry that recorded it
        # changes name or checksum, on the table's side as much as the
        # release's. Compare every identity this registration has ever held.
        archived_names, archived_digests = _recorded_identity_aliases(entry)
        names = {filename if is_bare_filename(filename) else "", *archived_names}
        digests = {
            entry.get("sha256"),
            _effective_recorded_digest(name, vintage, entry),
            *archived_digests,
        }
        for candidate in sorted(names):
            try:
                _assert_no_microdata_identity(
                    manifests,
                    package_dir=manifest_path.parent,
                    filename=candidate,
                    digests=digests,
                )
            except ManifestAccessError as error:
                # Name the registration as well as the release: publish and
                # inventory report a package's codes as vintage/filename, code
                # and manifest, and an operator reconciling the directory needs
                # both.
                raise SourceArtifactManifestError(
                    f"{vintage!r}/{filename}: {MICRODATA_IDENTITY_CODE}: "
                    f"{manifest_path}. {error}"
                ) from error


def _assert_siblings_record_these_bytes(
    manifests: Mapping[str, dict[str, Any]],
    *,
    manifest_path: Path,
    vintage: Any,
    filename: str,
    sha256: str,
) -> None:
    """Refuse a default fetch that would stale another manifest's owner."""
    for owner in _manifest_file_owners(
        manifests, filename=filename, initializing=(manifest_path, vintage)
    ):
        if owner.manifest_path == manifest_path:
            continue
        identity = owner.identity
        assert identity is not None
        if identity.sha256 == sha256:
            continue
        raise SourceArtifactRevisionError(
            f"{owner.manifest_path} entry {owner.vintage!r} records "
            f"{filename!r} as sha256={identity.sha256}; this fetch would write "
            f"sha256={sha256} to the same package-local file. Re-run with "
            "--record-revision to update every owner together."
        )


def _assert_manifest_identifies(
    existing_manifest: dict[str, Any],
    manifest_path: Path,
    *,
    source_id: str,
    package_id: str,
) -> None:
    """Refuse to fetch into a manifest that identifies another package.

    The R2 key and registration identity are built from the fetch arguments.
    Recording them in a manifest that declares different identifiers would
    leave one entry making two incompatible provenance claims.
    """
    for field, value in (("source_id", source_id), ("package_id", package_id)):
        if field not in existing_manifest:
            continue
        declared = _require_identity_segment(
            existing_manifest[field], what=f"{manifest_path} {field}"
        )
        if declared != value:
            raise SourceArtifactManifestError(
                f"{manifest_path} declares {field}={declared!r}; refusing to "
                f"fetch {field}={value!r} into it. Fetch into the package the "
                "manifest identifies, or into that package's own directory."
            )


def _refuse_a_stray_default_manifest(output: Path, manifest_path: Path) -> None:
    """Refuse to create any new manifest beside a package's registry.

    Every supported spelling participates: a missing ``manifest.yaml`` beside
    ``manifest.yml`` or ``Manifest.yaml`` is just as ambiguous as one beside a
    named manifest, and a mistyped named selector must not create a parallel
    registry. Operators may create an intentional empty sibling explicitly,
    then select that existing file.
    """
    paths = package_manifest_paths(output)
    if (
        any(path.name == manifest_path.name for path in paths)
        or manifest_path.is_symlink()
    ):
        return
    siblings = [path.name for path in paths]
    if not siblings:
        return
    raise AmbiguousManifestError(
        f"{output} already keeps {', '.join(siblings)}; refusing to create "
        f"{manifest_path.name} beside that registry. Pass --manifest to name "
        "an existing manifest, or create an intentional empty sibling "
        "explicitly before fetching into it."
    )


def _manifest_files(payload: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Return a manifest's ``files`` block, refusing one that is not a mapping.

    ``inventory-artifacts`` and ``publish-raw`` report the same document as
    ``files must be a mapping``; a fetch must refuse it before reading the
    publisher, or the write fails only after the local artifact has been
    overwritten and any upload has run.
    """
    files = payload.get("files")
    if files is None:
        # A bare ``files:`` line parses as None: no entries, like an absent
        # block. The writer normalizes it to a mapping before recording into it.
        return {}
    if not isinstance(files, dict):
        raise MalformedManifestError(
            f"{manifest_path} files must be a mapping; it parses as a "
            f"{type(files).__name__}. Chronicle will not overwrite a manifest "
            "it cannot read."
        )
    return files


def default_r2_raw_bucket() -> str:
    """Resolve the raw bucket: ``$CHRONICLE_R2_RAW_BUCKET`` or the default."""
    return env_value(R2_RAW_BUCKET_ENV, default=DEFAULT_R2_RAW_BUCKET)


def default_r2_derived_bucket() -> str:
    """Resolve the derived bucket: ``$CHRONICLE_R2_DERIVED_BUCKET`` or default."""
    return env_value(R2_DERIVED_BUCKET_ENV, default=DEFAULT_R2_DERIVED_BUCKET)


class SourceArtifactManifestError(RuntimeError, ManifestAccessError):
    """A manifest refuses the write a fetch is about to make.

    The checks that raise these run before the publisher is read, so an
    ordinary refusal costs nothing and leaves the package exactly as it was.
    They are repeated immediately before the manifest is rewritten, so no
    caller can reach a false-provenance write by another route.
    """


class SourceArtifactRevisionError(SourceArtifactManifestError):
    """Fetched bytes are not the bytes the manifest entry identifies.

    A manifest entry identifies specific bytes: by its declared ``sha256``, and
    -- once published -- by a content-addressed R2 key that repeats them. When
    a publisher re-publishes under the same URL and vintage, rewriting that
    entry would attach its provenance, and any recorded URI, to bytes it never
    described. Chronicle refuses instead: same vintage plus new bytes is a new
    release revision (docs/adr-chronicle-fact-identity-v2.md), registered with
    ``fetch-artifact --record-revision``.
    """


class ManifestNameError(SourceArtifactManifestError, ValueError):
    """A manifest name is not a bare filename inside the package directory.

    Also a :class:`ValueError` for callers that validated the name that way
    before the CLI learned to report it as an ordinary manifest refusal.
    """


class AmbiguousManifestError(SourceArtifactManifestError):
    """The default manifest name would create a manifest beside the ones a
    package already keeps (PolicyEngine/chronicle#225)."""


class MalformedManifestError(SourceArtifactManifestError):
    """A manifest document, or a block inside one, is not a mapping.

    Reading such a file as an absent manifest would let a fetch replace it with
    a single entry, dropping whatever the unreadable document recorded.
    """


class RecordedR2LocatorError(SourceArtifactManifestError):
    """A recorded ``storage.r2`` block does not locate exactly one object.

    The provider and URI must explicitly identify R2. ``provider``, ``bucket``,
    ``key`` and ``uri`` all describe the same object, so any additional fields
    have to agree, and the key has to carry the ``{sha256}/{filename}`` tail
    that says which bytes it holds. A block whose fields contradict each other
    has no single answer to "which bytes does this entry claim R2 holds", and
    preserving or publishing under it would ship whichever field the reader
    happened to consult.
    """


class ExpectedArtifactIdentityError(SourceArtifactManifestError):
    """Fetched bytes are not the bytes a reviewed pin said to expect.

    Distinct from :class:`SourceArtifactRevisionError`: a revision has an
    opt-in (``--record-revision`` supersedes what the manifest records), an
    expectation has none. The operator re-reviews the release and changes
    ``--expected-sha256``; Chronicle never archives an unreviewed reissue.
    """


# New UK and New Zealand uploads are namespaced by country. US objects predate
# the country segment and deliberately keep their legacy ``raw/{source_id}``
# and ``derived/{source_id}`` shapes. Publisher directories are the stable
# routing input because they are shared by packages/ and db/data/.
R2_COUNTRY_PUBLISHERS = {
    "nz": frozenset({"ird", "mbie", "msd", "stats_nz"}),
    "uk": frozenset(
        {
            "dft",
            "dwp",
            "hmrc",
            "isc",
            "mhclg",
            "nisra",
            "nrs",
            "obr",
            "ons",
            "scotgov",
            "slc",
            "voa",
            "welshgov",
        }
    ),
}


@dataclass(frozen=True)
class ArtifactStorageLocation:
    """Location metadata for a stored artifact."""

    provider: str
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        """Return a storage URI."""
        return f"{self.provider}://{self.bucket}/{self.key}"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable location."""
        return {
            "provider": self.provider,
            "bucket": self.bucket,
            "key": self.key,
            "uri": self.uri,
        }


# A raw key ends in {sha256}/{filename} (see build_r2_key), so the segment
# before the filename is what says which bytes the object holds.
_SHA256_KEY_SEGMENT = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RecordedR2Object:
    """The R2 object a manifest entry's ``storage.r2`` block claims exists.

    Built only by :func:`_validated_recorded_r2`, so every instance names one
    object whose locator fields agree with each other.
    """

    provider: str
    bucket: str
    key: str
    sha256: str
    filename: str

    @property
    def uri(self) -> str:
        """Return the storage URI the recorded fields spell out."""
        return f"{self.provider}://{self.bucket}/{self.key}"


@dataclass(frozen=True)
class RecordedIdentity:
    """The bytes a manifest entry says its vintage currently holds.

    From the recorded object's content-addressed key once the entry has been
    published, and from the entry's own declared ``sha256``/``filename`` before
    that. ``r2`` is None in the second case: protection does not wait for an
    upload to have happened.
    """

    sha256: str
    filename: str
    size_bytes: int | None
    declared_sha256: str | None
    r2: RecordedR2Object | None

    def holds(self, *, sha256: str, filename: str) -> bool:
        """Whether this identity is exactly the given bytes under that name.

        The filename participates only when the entry records one: a published
        key always carries it, an entry that declares bytes and no name does
        not, and inventing a mismatch there would refuse a re-fetch of the very
        bytes the entry describes.
        """
        if self.sha256 != sha256:
            return False
        return not self.filename or self.filename == Path(filename).name


@dataclass(frozen=True)
class ArtifactCommandResult:
    """Result from a storage command."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """Whether the command succeeded."""
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class ArtifactFetchReport:
    """Report from fetching and storing one source artifact."""

    source_id: str
    package_id: str
    year: int
    source_url: str
    filename: str
    local_path: str
    manifest_path: str
    sha256: str
    size_bytes: int
    fetched_at: str
    r2_location: ArtifactStorageLocation | None
    r2_upload: ArtifactCommandResult | None
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether acquisition and optional upload succeeded."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "valid": self.valid,
            "source_id": self.source_id,
            "package_id": self.package_id,
            "year": self.year,
            "source_url": self.source_url,
            "filename": self.filename,
            "local_path": self.local_path,
            "manifest_path": self.manifest_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "fetched_at": self.fetched_at,
            "r2_location": (self.r2_location.to_dict() if self.r2_location else None),
            "r2_upload": self.r2_upload.to_dict() if self.r2_upload else None,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ArtifactInventoryEntry:
    """One manifest-declared source artifact status."""

    manifest_path: str
    year: str
    filename: str
    local_path: str
    exists: bool
    sha256_expected: str | None
    sha256_actual: str | None
    size_bytes: int | None
    source_url: str | None
    r2: dict[str, Any] | None
    errors: tuple[str, ...]
    access: str = ACCESS_PUBLIC
    licence: str | None = None
    hash_only: bool = False

    @property
    def valid(self) -> bool:
        """Whether this registration is complete and, if public, available."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable inventory entry."""
        return {
            "valid": self.valid,
            "manifest_path": self.manifest_path,
            "year": self.year,
            "filename": self.filename,
            "local_path": self.local_path,
            "exists": self.exists,
            "access": self.access,
            "licence": self.licence,
            "hash_only": self.hash_only,
            "sha256_expected": self.sha256_expected,
            "sha256_actual": self.sha256_actual,
            "size_bytes": self.size_bytes,
            "source_url": self.source_url,
            "r2": self.r2,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ArtifactInventoryReport:
    """Manifest inventory report for source artifacts."""

    root: str
    counts: dict[str, int]
    entries: tuple[ArtifactInventoryEntry, ...]
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether every manifest entry is locally available and valid."""
        return not self.errors and all(entry.valid for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "valid": self.valid,
            "root": self.root,
            "counts": self.counts,
            "entries": [entry.to_dict() for entry in self.entries],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class RawArtifactPublishEntry:
    """One manifest-declared raw artifact upload status."""

    manifest_path: str
    source_id: str
    package_id: str
    year: str
    filename: str
    local_path: str
    sha256: str | None
    size_bytes: int | None
    r2_location: ArtifactStorageLocation | None
    upload: ArtifactCommandResult | None
    errors: tuple[str, ...] = ()
    skipped: str | None = None

    @property
    def uploaded(self) -> bool:
        """Whether this run uploaded the artifact."""
        return self.upload is not None and self.upload.ok

    @property
    def hash_only_refused(self) -> bool:
        """Whether this entry was left alone because its access is hash-only."""
        return self.skipped is not None and self.skipped.startswith(
            HASH_ONLY_SKIP_PREFIX
        )

    @property
    def valid(self) -> bool:
        """Whether this raw artifact is published or correctly not uploaded:
        uploaded now, already held by the recorded object in a preserved
        bucket, or skipped as a hash-only registration (``skipped``)."""
        return not self.errors and (self.skipped is not None or self.uploaded)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable entry."""
        return {
            "valid": self.valid,
            "skipped": self.skipped,
            "manifest_path": self.manifest_path,
            "source_id": self.source_id,
            "package_id": self.package_id,
            "year": self.year,
            "filename": self.filename,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "r2_location": (self.r2_location.to_dict() if self.r2_location else None),
            "upload": self.upload.to_dict() if self.upload else None,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class RawArtifactPublishReport:
    """Report from publishing local manifest-declared raw artifacts to R2."""

    root: str
    entries: tuple[RawArtifactPublishEntry, ...]
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether every raw artifact uploaded and manifest metadata was updated."""
        return not self.errors and all(entry.valid for entry in self.entries)

    @property
    def counts(self) -> dict[str, int]:
        """Return summary counts."""
        manifest_paths = {entry.manifest_path for entry in self.entries}
        return {
            "manifest_count": len(manifest_paths),
            "artifact_count": len(self.entries),
            "uploaded_count": sum(1 for entry in self.entries if entry.uploaded),
            "skipped_count": sum(
                1 for entry in self.entries if entry.skipped is not None
            ),
            "failed_count": sum(1 for entry in self.entries if not entry.valid),
            "hash_only_refused_count": sum(
                1 for entry in self.entries if entry.hash_only_refused
            ),
            "r2_link_count": sum(
                1 for entry in self.entries if entry.r2_location is not None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "valid": self.valid,
            "root": self.root,
            "counts": self.counts,
            "entries": [entry.to_dict() for entry in self.entries],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class R2BootstrapReport:
    """Report from bootstrapping Chronicle R2 buckets."""

    buckets: tuple[str, ...]
    commands: tuple[ArtifactCommandResult, ...]
    authenticated: bool
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Whether all requested buckets were created or already available."""
        return self.authenticated and not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "valid": self.valid,
            "authenticated": self.authenticated,
            "buckets": list(self.buckets),
            "commands": [command.to_dict() for command in self.commands],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class DerivedArtifactUploadEntry:
    """One derived build artifact upload status."""

    artifact_name: str
    local_path: str
    sha256: str
    size_bytes: int
    r2_location: ArtifactStorageLocation
    upload: ArtifactCommandResult

    @property
    def valid(self) -> bool:
        """Whether this derived artifact uploaded successfully."""
        return self.upload.ok

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable entry."""
        return {
            "valid": self.valid,
            "artifact_name": self.artifact_name,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "r2_location": self.r2_location.to_dict(),
            "upload": self.upload.to_dict(),
        }


@dataclass(frozen=True)
class DerivedArtifactPublishReport:
    """Report from publishing deterministic build outputs to R2."""

    input_dir: str
    source_id: str
    package_id: str
    year: int
    build_id: str
    entries: tuple[DerivedArtifactUploadEntry, ...]
    build_artifacts_path: str | None = None
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether every derived artifact uploaded successfully."""
        return not self.errors and all(entry.valid for entry in self.entries)

    @property
    def counts(self) -> dict[str, int]:
        """Return summary counts."""
        return {
            "artifact_count": len(self.entries),
            "uploaded_count": sum(1 for entry in self.entries if entry.valid),
            "failed_count": sum(1 for entry in self.entries if not entry.valid),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "valid": self.valid,
            "input_dir": self.input_dir,
            "source_id": self.source_id,
            "package_id": self.package_id,
            "year": self.year,
            "build_id": self.build_id,
            "build_artifacts_path": self.build_artifacts_path,
            "counts": self.counts,
            "entries": [entry.to_dict() for entry in self.entries],
            "errors": list(self.errors),
        }


def fetch_source_artifact(
    source_url: str,
    *,
    source_id: str,
    package_id: str,
    year: int,
    output_dir: str | Path,
    dataset: str | None = None,
    source_page: str | None = None,
    table: str | None = None,
    filename: str | None = None,
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
    access: str = ACCESS_PUBLIC,
    licence: str | None = None,
    kind: str | None = None,
    publisher: str | None = None,
    vintage: str | None = None,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    licence_evidence: Mapping[str, Any] | None = None,
    staging_dir: str | Path | None = None,
    upload_r2: bool = False,
    record_revision: bool = False,
    r2_bucket: str | None = None,
    r2_prefix: str | None = None,
    wrangler_command: str = DEFAULT_WRANGLER_COMMAND,
    _manifest_lock_held: bool = False,
) -> ArtifactFetchReport:
    """Fetch/register a source artifact and optionally upload it to R2.

    ``manifest_filename`` names the manifest inside ``output_dir`` the entry
    belongs to. Packages that split one publisher directory across several
    source packages keep one manifest each, so a fetch that always wrote
    ``manifest.yaml`` would write a fresh manifest beside the real ones and
    never see the entry it is revising.

    ``record_revision`` opts into registering a publisher revision: the fetched
    bytes get their own content-addressed key under the configured bucket and
    the superseded object moves to ``storage.previous_r2``. Without it, bytes
    that disagree with the entry's recorded identity raise
    :class:`SourceArtifactRevisionError` before anything is overwritten.

    ``expected_sha256`` (and ``expected_size_bytes``) pin the bytes a reviewed
    identity says the publisher serves: bytes that hash differently raise
    :class:`ExpectedArtifactIdentityError` before anything is written or
    uploaded, and ``record_revision`` does not override that.

    Only ``public`` artifacts travel this path: it writes bytes and can upload
    them to the raw bucket. A publisher table's bytes land beside its
    manifest; a public microdata release (``kind: microdata_release``) is
    archived only under an allowlisted ``licence`` with ``licence_evidence``
    bound to ``expected_sha256``, and its bytes are staged in a transient
    directory outside the package tree and uploaded from there. Licensed and
    restricted artifacts are registered hash-only with
    :func:`chronicle.registration.register_hash_only_artifact`.

    Every refusal happens before the publisher is read; the checks that need
    the bytes (expected identity, recorded identity) run before anything is
    written.
    """
    r2_bucket = r2_bucket or default_r2_raw_bucket()
    output = Path(output_dir)
    manifest_path = _manifest_path(output, manifest_filename)
    access_class = normalize_access(access)
    if is_hash_only(access_class):
        raise ManifestAccessError(
            f"fetch-artifact stores bytes and refuses access={access_class!r}. "
            "Register a licensed or restricted artifact by identity with "
            "`chronicle register-artifact`."
        )
    # The destination name is a pure function of the arguments, so it is
    # resolved -- and any alias of a registered name refused -- before the
    # publisher is read. Reading first and refusing afterwards would pull
    # gated bytes to decide their name.
    what = (
        "--filename" if filename is not None else "The filename inferred from the URL"
    )
    artifact_filename = bare_filename(
        filename if filename is not None else _infer_artifact_filename(source_url),
        what=what,
    )
    if is_manifest_filename(artifact_filename):
        raise ArtifactFilenameError(
            f"{what} {artifact_filename!r} is a manifest name. An artifact may "
            "not be named like a manifest, which it would overwrite; pass "
            "--filename with the publisher's name for the bytes."
        )
    _require_identity_segment(source_id, what="source_id")
    _require_identity_segment(package_id, what="package_id")
    _require_identity_segment(str(year), what="year")
    expected = _expected_identity(expected_sha256, expected_size_bytes)
    resolved_r2_prefix = resolve_r2_prefix(
        prefix=r2_prefix,
        default_prefix=DEFAULT_R2_PREFIX,
        source_id=source_id,
        package_path=output,
    )

    # Read and validate the manifest being written before anything is fetched:
    # a manifest Chronicle cannot read, a manifest name that would sit beside
    # the ones a package keeps, a recorded block that names two different
    # objects, or a registration the fetch would overwrite are all refusals
    # that need not touch the publisher.
    try:
        _refuse_a_stray_default_manifest(output, manifest_path)
    except SourceArtifactManifestError:
        raise
    except ValueError as error:
        raise MalformedManifestError(str(error)) from error
    existing_manifest = _read_manifest(manifest_path)
    _manifest_files(existing_manifest, manifest_path)
    # The byte boundary is checked first, across every manifest the directory
    # keeps: overwriting a hash-only registration with bytes is the more
    # serious refusal, and its message is the one the caller needs, not a
    # prompt about the manifest's kind or licence.
    manifests = _package_manifests(output, manifest_path, existing_manifest)
    manifest_kinds: dict[str, str] = {}
    for sibling_path, sibling in manifests.items():
        if sibling_path == str(manifest_path):
            continue
        try:
            sibling_kind = normalize_manifest_kind(
                sibling, manifest_path=Path(sibling_path)
            )
        except ManifestAccessError as exc:
            raise ManifestAccessError(
                f"{sibling_path} cannot be classified safely: {exc}"
            ) from exc
        manifest_kinds[sibling_path] = sibling_kind
        _assert_manifest_valid_for_fetch(
            sibling,
            Path(sibling_path),
            kind=sibling_kind,
            package_dir=output,
        )
    for sibling_path, sibling in manifests.items():
        _assert_no_hash_only_entry(sibling, Path(sibling_path), artifact_filename)
    if expected.sha256:
        _assert_no_hash_only_bytes(
            manifests,
            sha256=expected.sha256,
            filename=artifact_filename,
            what="the reviewed pin's bytes",
        )
    _assert_manifest_identifies(
        existing_manifest,
        manifest_path,
        source_id=source_id,
        package_id=package_id,
    )
    manifest_kind_value = _resolve_manifest_kind(
        existing_manifest,
        manifest_path=manifest_path,
        requested_kind=kind,
    )
    _assert_manifest_valid_for_fetch(
        existing_manifest,
        manifest_path,
        kind=manifest_kind_value,
        package_dir=output,
    )
    # Every manifest in the directory is valid on its own terms by now, which
    # is where publish-raw and inventory-artifacts sweep the package as a
    # whole. A fetch rewrites one of these manifests, so it refuses the same
    # directory they refuse -- including a collision in a vintage this fetch
    # never touches -- before the publisher is read and before the lock.
    _assert_no_package_microdata_identities(manifests)
    manifest_kinds[str(manifest_path)] = manifest_kind_value
    # Entry validation never opens ``storage``, and the identity resolution
    # below reads only the selected entry's locators, so this is where an
    # unreadable one anywhere else in the directory is refused -- exactly as
    # ``_prepare_registration_payload`` refuses it before its own lock, rather
    # than leaving it to _upsert_manifest after the publisher is read. It
    # follows the sweep rather than preceding it because the sweep resolves
    # locators non-raisingly: what it does find on unreadable provenance is
    # still true, and the more serious refusal's message is the one the caller
    # needs. What it may have missed is why this cannot be skipped.
    for validated_path, validated in manifests.items():
        _assert_recorded_locators_valid(
            validated, Path(validated_path), kind=manifest_kinds[validated_path]
        )
    licence_text = licence.strip() if isinstance(licence, str) else None
    release = manifest_kind_value == MICRODATA_RELEASE_KIND
    evidence: dict[str, str] | None = None
    if release:
        evidence = _release_fetch_evidence(
            manifest_path,
            existing_manifest,
            filename=artifact_filename,
            package_dir=output,
            licence=licence_text,
            vintage=vintage,
            publisher=publisher,
            expected=expected,
            licence_evidence=licence_evidence,
        )
        _validated_microdata_staging_destination(
            microdata_staging_path(
                staging_dir=staging_dir,
                source_id=source_id,
                package_id=package_id,
                year=year,
                sha256=expected.sha256,
                filename=artifact_filename,
            ),
            package_dir=output,
        )

    vintage_key, existing_value, selected_spec, _index = _select_vintage_entry(
        existing_manifest,
        manifest_path=manifest_path,
        year=year,
        filename=artifact_filename,
        kind=manifest_kind_value,
    )
    recorded_identity = _recorded_identity(
        selected_spec,
        manifest_path=manifest_path,
        year=vintage_key,
        source_id=source_id,
        package_id=package_id,
        bind_registration_identity=release,
    )
    identity_digests = (
        expected.sha256,
        selected_spec.get("sha256"),
        recorded_identity.sha256 if recorded_identity is not None else None,
    )
    if release:
        _assert_no_table_claims_release_identity(
            manifests,
            package_dir=output,
            filename=artifact_filename,
            digests=identity_digests,
        )
    else:
        # The selected registration's archived objects identify these bytes as
        # surely as its current one, so an alias survives a rename on either
        # side of the package.
        archived_names, archived_digests = _recorded_identity_aliases(selected_spec)
        for candidate in sorted({artifact_filename, *archived_names}):
            _assert_no_microdata_identity(
                manifests,
                package_dir=output,
                filename=candidate,
                digests=(*identity_digests, *archived_digests),
            )
    _assert_table_vintage_is_revisable(
        existing_value,
        recorded_identity,
        manifest_path=manifest_path,
        year=vintage_key,
        filename=artifact_filename,
        release=release,
    )
    if (
        expected.sha256
        and recorded_identity is not None
        and not record_revision
        and not recorded_identity.holds(
            sha256=expected.sha256, filename=artifact_filename
        )
    ):
        # The manifest already identifies other bytes: no download can satisfy
        # both the recorded identity and the expectation, so refuse before it.
        raise ExpectedArtifactIdentityError(
            f"{manifest_path} entry {vintage_key!r} already records "
            f"sha256={recorded_identity.sha256} "
            f"filename={recorded_identity.filename or 'unknown'}, and the "
            f"reviewed pin expects sha256={expected.sha256}. Re-review the pin, "
            "or pass --record-revision together with the reviewed "
            "--expected-sha256 to register the publisher revision."
        )

    if expected.sha256 and not record_revision:
        _assert_siblings_record_these_bytes(
            manifests,
            manifest_path=manifest_path,
            vintage=vintage_key,
            filename=artifact_filename,
            sha256=expected.sha256,
        )
    owners = _manifest_file_owners(
        manifests,
        filename=artifact_filename,
        initializing=(manifest_path, vintage_key),
    )
    _assert_shared_owner_identities_agree(owners, filename=artifact_filename)
    try:
        existing_target = matching_directory_entry(output, artifact_filename)
    except ValueError as error:
        raise ArtifactFilenameError(str(error)) from error
    if existing_target is not None:
        if existing_target.is_symlink():
            raise ArtifactFilenameError(
                f"{existing_target} is a symbolic link (symlink)."
            )
        if existing_target.name != artifact_filename:
            raise ArtifactFilenameError(
                f"{existing_target} has a different spelling from {artifact_filename!r}."
            )
        if not existing_target.is_file():
            raise ArtifactFilenameError(f"{existing_target} is not a regular file.")

    if not _manifest_lock_held:
        # The first pass above is side-effect-free. Repeat it after acquiring
        # the registration lock, then keep that lock through publisher access,
        # local/staging persistence, upload, and every manifest rewrite.
        with _registration_lock(output):
            return fetch_source_artifact(
                source_url,
                source_id=source_id,
                package_id=package_id,
                year=year,
                output_dir=output,
                dataset=dataset,
                source_page=source_page,
                table=table,
                filename=filename,
                manifest_filename=manifest_filename,
                access=access,
                licence=licence,
                kind=kind,
                publisher=publisher,
                vintage=vintage,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
                licence_evidence=licence_evidence,
                staging_dir=staging_dir,
                upload_r2=upload_r2,
                record_revision=record_revision,
                r2_bucket=r2_bucket,
                r2_prefix=r2_prefix,
                wrangler_command=wrangler_command,
                _manifest_lock_held=True,
            )

    fetched_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    content, _inferred_filename = _read_artifact(source_url)

    sha256 = hashlib.sha256(content).hexdigest()
    size_bytes = len(content)

    # Guard before the cached artifact is touched. A rejected fetch must leave
    # the recorded bytes and their manifest entry exactly as they were.
    _assert_expected_identity(
        expected,
        manifest_path=manifest_path,
        year=vintage_key,
        filename=artifact_filename,
        source_url=source_url,
        sha256=sha256,
        size_bytes=size_bytes,
    )
    _assert_recorded_identity_holds_these_bytes(
        recorded_identity,
        manifest_path=manifest_path,
        year=vintage_key,
        filename=artifact_filename,
        sha256=sha256,
        size_bytes=size_bytes,
        r2_bucket=r2_bucket,
        record_revision=record_revision,
    )
    _assert_no_hash_only_bytes(
        manifests,
        sha256=sha256,
        filename=artifact_filename,
        what=f"the bytes served by {source_url}",
    )
    if release:
        _assert_no_table_claims_release_identity(
            manifests,
            package_dir=output,
            filename=artifact_filename,
            digests=(sha256,),
        )
    else:
        _assert_no_microdata_identity(
            manifests,
            package_dir=output,
            filename=artifact_filename,
            digests=(sha256,),
        )
    if not record_revision:
        _assert_siblings_record_these_bytes(
            manifests,
            manifest_path=manifest_path,
            vintage=vintage_key,
            filename=artifact_filename,
            sha256=sha256,
        )

    r2_location = ArtifactStorageLocation(
        provider="r2",
        bucket=r2_bucket,
        key=build_r2_key(
            source_id=source_id,
            package_id=package_id,
            year=year,
            sha256=sha256,
            filename=artifact_filename,
            prefix=resolved_r2_prefix,
            package_path=output,
        ),
    )
    manifest_update = dict(
        source_id=source_id,
        package_id=package_id,
        dataset=dataset or f"{source_id}_{package_id}",
        source_page=source_page,
        table=table,
        publisher=publisher,
        year=year,
        filename=artifact_filename,
        source_url=source_url,
        sha256=sha256,
        size_bytes=size_bytes,
        fetched_at=fetched_at,
        access=access_class,
        licence=licence_text,
        kind=manifest_kind_value,
        vintage=vintage,
        licence_evidence=evidence,
        expected=expected,
        record_revision=record_revision,
    )
    _upsert_manifest(
        manifest_path,
        **manifest_update,
        r2_location=r2_location if upload_r2 else None,
        _preflight_only=True,
    )

    for owner in owners:
        # The selected owner is (manifest, vintage key): a YAML anchor can make
        # two vintages share one dict object, and object identity would skip
        # both.
        if owner.manifest_path == manifest_path and str(owner.vintage) == str(
            vintage_key
        ):
            continue
        _assert_recorded_identity_holds_these_bytes(
            owner.identity,
            manifest_path=owner.manifest_path,
            year=owner.vintage,
            filename=artifact_filename,
            sha256=sha256,
            size_bytes=size_bytes,
            r2_bucket=r2_bucket,
            record_revision=record_revision,
        )

    if release:
        # Public microdata never lands in the package tree: it is staged in an
        # untracked, transient directory and uploaded from there.
        local_path = _validated_microdata_staging_destination(
            microdata_staging_path(
                staging_dir=staging_dir,
                source_id=source_id,
                package_id=package_id,
                year=year,
                sha256=sha256,
                filename=artifact_filename,
            ),
            package_dir=output,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)
        local_path = output / artifact_filename
    local_path.write_bytes(content)

    r2_upload = None
    errors: list[str] = []
    if upload_r2:
        r2_upload = _upload_r2_object(
            r2_location,
            local_path,
            wrangler_command=wrangler_command,
        )
        if not r2_upload.ok:
            errors.append("r2_upload_failed")

    _upsert_manifest(
        manifest_path,
        **manifest_update,
        r2_location=(r2_location if upload_r2 and r2_upload and r2_upload.ok else None),
    )

    return ArtifactFetchReport(
        source_id=source_id,
        package_id=package_id,
        year=year,
        source_url=source_url,
        filename=artifact_filename,
        local_path=str(local_path),
        manifest_path=str(manifest_path),
        sha256=sha256,
        size_bytes=size_bytes,
        fetched_at=fetched_at,
        r2_location=r2_location if upload_r2 else None,
        r2_upload=r2_upload,
        errors=tuple(errors),
    )


def publish_derived_artifacts(
    input_dir: str | Path,
    *,
    source_id: str,
    package_id: str,
    year: int,
    build_id: str | None = None,
    r2_bucket: str | None = None,
    r2_prefix: str | None = None,
    wrangler_command: str = "npx wrangler",
    build_artifacts_output: str | Path | None = None,
) -> DerivedArtifactPublishReport:
    """Upload a deterministic build output directory to the derived R2 bucket."""
    r2_bucket = r2_bucket or default_r2_derived_bucket()
    input_path = Path(input_dir)
    try:
        _require_identity_segment(source_id, what="source_id")
        _require_identity_segment(package_id, what="package_id")
        _require_identity_segment(str(year), what="year")
    except SourceArtifactManifestError as error:
        return DerivedArtifactPublishReport(
            input_dir=str(input_path),
            source_id=source_id,
            package_id=package_id,
            year=year,
            build_id=build_id or "",
            entries=(),
            build_artifacts_path=str(build_artifacts_output)
            if build_artifacts_output
            else None,
            errors=(f"r2_identity_invalid:{error}",),
        )
    try:
        resolved_r2_prefix = resolve_r2_prefix(
            prefix=r2_prefix,
            default_prefix=default_r2_derived_prefix(),
            source_id=source_id,
        )
        if not is_derived_r2_route(r2_bucket, resolved_r2_prefix):
            raise ValueError(
                "Set CHRONICLE_R2_DERIVED_BUCKET or CHRONICLE_R2_DERIVED_PREFIX "
                "to identify a custom derived route before publishing to it."
            )
    except ValueError as error:
        return DerivedArtifactPublishReport(
            input_dir=str(input_path),
            source_id=source_id,
            package_id=package_id,
            year=year,
            build_id=build_id or "",
            entries=(),
            build_artifacts_path=str(build_artifacts_output)
            if build_artifacts_output
            else None,
            errors=(f"derived_route_invalid:{error}",),
        )
    if not input_path.exists():
        return DerivedArtifactPublishReport(
            input_dir=str(input_path),
            source_id=source_id,
            package_id=package_id,
            year=year,
            build_id=build_id or "",
            entries=(),
            build_artifacts_path=str(build_artifacts_output)
            if build_artifacts_output
            else None,
            errors=(f"input_dir_not_found:{input_path}",),
        )
    if not input_path.is_dir():
        return DerivedArtifactPublishReport(
            input_dir=str(input_path),
            source_id=source_id,
            package_id=package_id,
            year=year,
            build_id=build_id or "",
            entries=(),
            build_artifacts_path=str(build_artifacts_output)
            if build_artifacts_output
            else None,
            errors=(f"input_dir_is_not_directory:{input_path}",),
        )

    try:
        artifact_paths = _derived_artifact_paths(input_path)
    except (OSError, ValueError) as error:
        return DerivedArtifactPublishReport(
            input_dir=str(input_path),
            source_id=source_id,
            package_id=package_id,
            year=year,
            build_id=build_id or "",
            entries=(),
            build_artifacts_path=str(build_artifacts_output)
            if build_artifacts_output
            else None,
            errors=(str(error),),
        )

    resolved_build_id = build_id or infer_build_id(input_path)
    if not resolved_build_id:
        return DerivedArtifactPublishReport(
            input_dir=str(input_path),
            source_id=source_id,
            package_id=package_id,
            year=year,
            build_id="",
            entries=(),
            build_artifacts_path=str(build_artifacts_output)
            if build_artifacts_output
            else None,
            errors=("missing_build_id",),
        )

    # Validate the resolved identity before deriving object keys, invoking the
    # uploader, or opening the optional registry output; a malformed id is an
    # input failure like the ones above, reported rather than raised.
    try:
        canonicalize_key("build", resolved_build_id)
        _require_identity_segment(resolved_build_id, what="build_id")
    except ValueError:
        return DerivedArtifactPublishReport(
            input_dir=str(input_path),
            source_id=source_id,
            package_id=package_id,
            year=year,
            build_id=resolved_build_id,
            entries=(),
            build_artifacts_path=str(build_artifacts_output)
            if build_artifacts_output
            else None,
            errors=("malformed_build_id",),
        )

    entries: list[DerivedArtifactUploadEntry] = []
    errors: list[str] = []
    for artifact_path in artifact_paths:
        relative_path = artifact_path.relative_to(input_path).as_posix()
        if relative_path == "build_artifacts.jsonl":
            continue
        content = artifact_path.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        location = ArtifactStorageLocation(
            provider="r2",
            bucket=r2_bucket,
            key=build_derived_r2_key(
                source_id=source_id,
                package_id=package_id,
                year=year,
                build_id=resolved_build_id,
                artifact_name=relative_path,
                prefix=resolved_r2_prefix,
            ),
        )
        upload = _upload_r2_object(
            location,
            artifact_path,
            wrangler_command=wrangler_command,
        )
        if not upload.ok:
            errors.append(f"derived_upload_failed:{relative_path}")
        entries.append(
            DerivedArtifactUploadEntry(
                artifact_name=relative_path,
                local_path=str(artifact_path),
                sha256=sha256,
                size_bytes=len(content),
                r2_location=location,
                upload=upload,
            )
        )

    report = DerivedArtifactPublishReport(
        input_dir=str(input_path),
        source_id=source_id,
        package_id=package_id,
        year=year,
        build_id=resolved_build_id,
        entries=tuple(entries),
        build_artifacts_path=str(build_artifacts_output)
        if build_artifacts_output
        else None,
        errors=tuple(errors),
    )
    if build_artifacts_output is not None:
        write_build_artifacts_jsonl(report, build_artifacts_output)
    return report


def publish_source_artifacts(
    root: str | Path,
    *,
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
    source_id: str | None = None,
    package_id: str | None = None,
    r2_bucket: str | None = None,
    r2_prefix: str | None = None,
    wrangler_command: str = DEFAULT_WRANGLER_COMMAND,
    skip_hash_only: bool = False,
    staging_dir: str | Path | None = None,
    _selected_manifest_path: Path | None = None,
    _manifest_lock_held: bool = False,
) -> RawArtifactPublishReport:
    """Preflight a complete raw publication, then publish under package locks.

    Licensed and restricted registrations remain identity only. Public release
    bytes come from external staging, and every selected package and its sibling
    manifests are validated before the first upload or manifest rewrite.
    """
    arguments = {
        "manifest_filename": manifest_filename,
        "source_id": source_id,
        "package_id": package_id,
        "r2_bucket": r2_bucket,
        "r2_prefix": r2_prefix,
        "wrangler_command": wrangler_command,
        "skip_hash_only": skip_hash_only,
        "staging_dir": staging_dir,
        "_selected_manifest_path": _selected_manifest_path,
    }
    if _manifest_lock_held:
        return _publish_source_artifacts_unlocked(root, **arguments)
    preflight = _publish_source_artifacts_unlocked(
        root, **arguments, _preflight_only=True
    )
    if preflight.errors or any(entry.errors for entry in preflight.entries):
        return preflight
    root_path = Path(root)
    paths = (
        [_selected_manifest_path]
        if _selected_manifest_path is not None
        else _root_manifest_paths(root_path, manifest_filename)
    )
    # Keep every participating package locked while re-reading the complete
    # operation. A registration cannot change the access boundary between a
    # successful preflight and any of this operation's uploads.
    with ExitStack() as locks:
        for directory in sorted({path.parent for path in paths}):
            locks.enter_context(_registration_lock(directory))
        return _publish_source_artifacts_unlocked(root, **arguments)


def build_artifact_rows(
    report: DerivedArtifactPublishReport,
) -> tuple[dict[str, Any], ...]:
    """Build relational build_artifacts rows from a derived publish report."""
    rows: list[dict[str, Any]] = []
    for entry in report.entries:
        if not entry.valid:
            continue
        rows.append(
            {
                "build_artifact_key": build_artifact_key(
                    build_id=report.build_id,
                    artifact_name=entry.artifact_name,
                    sha256=entry.sha256,
                ),
                "build_id": report.build_id,
                "artifact_kind": _derived_artifact_kind(entry.artifact_name),
                "artifact_name": entry.artifact_name,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
                "r2_bucket": entry.r2_location.bucket,
                "r2_key": entry.r2_location.key,
                "r2_uri": entry.r2_location.uri,
            }
        )
    return tuple(rows)


def write_build_artifacts_jsonl(
    report: DerivedArtifactPublishReport,
    output_path: str | Path,
) -> None:
    """Write build_artifacts JSONL rows for a derived publish report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in build_artifact_rows(report):
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")


def inventory_source_artifacts(
    root: str | Path,
    *,
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
    staging_dir: str | Path | None = None,
) -> ArtifactInventoryReport:
    """Inventory manifest-declared source artifacts under a root directory.

    Reports every manifest-level defect (a missing or unknown kind, a vintage
    under two key spellings, filename collisions) alongside the per-entry
    codes, and treats a public microdata release's bytes as staged outside
    the tree: a copy beside the manifest is an error, not an artifact.
    """
    root_path = Path(root)
    _manifest_path(Path(), manifest_filename)
    errors: list[str] = []
    entries: list[ArtifactInventoryEntry] = []
    if not root_path.exists():
        return ArtifactInventoryReport(
            root=str(root_path),
            counts={
                "manifest_count": 0,
                "artifact_count": 0,
                "missing_count": 0,
                "checksum_mismatch_count": 0,
                "hash_only_count": 0,
                "r2_link_count": 0,
            },
            entries=(),
            errors=(f"Root does not exist: {root_path}",),
        )

    manifests = _root_manifest_paths(root_path, manifest_filename)
    for manifest_path in manifests:
        try:
            manifest = _read_manifest(manifest_path)
        except (OSError, SourceArtifactManifestError) as exc:
            errors.append(f"Could not read {manifest_path}: {exc}")
            continue
        try:
            files = _manifest_files(manifest, manifest_path)
        except SourceArtifactManifestError as exc:
            errors.append(f"Could not read {manifest_path}: {exc}")
            continue
        kind, _kind_error = safe_manifest_kind(manifest, manifest_path=manifest_path)
        package_errors: list[str] = []
        package_manifests = None
        try:
            package_manifests = _package_manifests(
                manifest_path.parent, manifest_path, manifest
            )
        except (OSError, SourceArtifactManifestError) as exc:
            package_errors.append(
                f"Could not read a manifest beside {manifest_path}: {exc}"
            )
        else:
            for package_manifest_name, package_manifest in package_manifests.items():
                package_manifest_path = Path(package_manifest_name)
                package_kind, package_kind_error = safe_manifest_kind(
                    package_manifest,
                    manifest_path=package_manifest_path,
                )
                if package_kind_error:
                    package_errors.append(
                        f"{package_kind_error}: {package_manifest_path}"
                    )
                package_errors.extend(
                    f"{code}: {package_manifest_path}"
                    for code in validate_manifest_files(package_manifest)
                )
                package_errors.extend(
                    f"{code}: {package_manifest_path}"
                    for code in _manifest_entry_validation_errors(
                        package_manifest,
                        kind=package_kind,
                        package_dir=package_manifest_path.parent,
                    )
                )
            package_errors.extend(
                f"{code}: {manifest_path}"
                for code in validate_package_directory(
                    package_manifests, entry_digest=_effective_recorded_digest
                )
            )
        if package_manifests is not None:
            try:
                _assert_package_file_owner_identities_agree(package_manifests)
            except SourceArtifactManifestError as exc:
                package_errors.append(str(exc))
            try:
                _assert_no_package_microdata_identities(package_manifests)
            except SourceArtifactManifestError as exc:
                package_errors.append(str(exc))
        errors.extend(package_errors)
        for year, spec in files.items():
            for file_spec in iter_file_specs(spec, kind=kind):
                entries.append(
                    _inventory_entry(
                        manifest_path,
                        year,
                        file_spec,
                        manifest=manifest,
                        kind=kind,
                        staging_dir=staging_dir,
                        inspect_bytes=not package_errors,
                        package_manifests=package_manifests,
                    )
                )

    counts = {
        "manifest_count": len(manifests),
        "artifact_count": len(entries),
        "missing_count": sum(1 for entry in entries if "missing_file" in entry.errors),
        "checksum_mismatch_count": sum(
            1 for entry in entries if "checksum_mismatch" in entry.errors
        ),
        "hash_only_count": sum(1 for entry in entries if entry.hash_only),
        "r2_link_count": sum(1 for entry in entries if entry.r2 is not None),
    }
    return ArtifactInventoryReport(
        root=str(root_path),
        counts=counts,
        entries=tuple(entries),
        errors=tuple(errors),
    )


def bootstrap_r2_buckets(
    *,
    raw_bucket: str | None = None,
    derived_bucket: str | None = None,
    wrangler_command: str = "npx wrangler",
) -> R2BootstrapReport:
    """Create the R2 buckets Chronicle expects, if Wrangler is authenticated."""
    buckets = (
        raw_bucket or default_r2_raw_bucket(),
        derived_bucket or default_r2_derived_bucket(),
    )
    commands: list[ArtifactCommandResult] = []
    errors: list[str] = []

    auth = _run_command([*shlex.split(wrangler_command), "whoami"])
    commands.append(auth)
    authenticated = (
        auth.ok and "not authenticated" not in (auth.stdout + auth.stderr).lower()
    )
    if not authenticated:
        return R2BootstrapReport(
            buckets=buckets,
            commands=tuple(commands),
            authenticated=False,
            errors=(
                "wrangler_not_authenticated: run `npx wrangler login` in the "
                "PolicyEngine Cloudflare account, then rerun this command.",
            ),
        )

    for bucket in buckets:
        command = _run_command(
            [*shlex.split(wrangler_command), "r2", "bucket", "create", bucket]
        )
        commands.append(command)
        combined_output = (command.stdout + command.stderr).lower()
        if not command.ok and "already exists" not in combined_output:
            errors.append(f"r2_bucket_create_failed:{bucket}")

    return R2BootstrapReport(
        buckets=buckets,
        commands=tuple(commands),
        authenticated=authenticated,
        errors=tuple(errors),
    )


def infer_r2_country(
    *,
    source_id: str | None = None,
    package_path: str | Path | None = None,
) -> str | None:
    """Infer an R2 country segment from a package publisher directory.

    ``package_path`` is authoritative when supplied. ``source_id`` is a
    fallback for low-level callers that do not have a package path, including
    derived-artifact publication.
    """
    publisher = None
    if package_path is not None:
        path_parts = tuple(part.lower() for part in Path(package_path).parts)
        # Match the innermost canonical root with both publisher and package
        # directories. A package itself may be named "packages"; the following
        # manifest filename must not then be mistaken for a publisher.
        for index in range(len(path_parts) - 1, -1, -1):
            if path_parts[index : index + 2] == ("db", "data"):
                if index + 3 < len(path_parts):
                    publisher = path_parts[index + 2]
                    break
            elif path_parts[index] == "packages" and index + 2 < len(path_parts):
                publisher = path_parts[index + 1]
                break
    path_country = next(
        (
            country
            for country, publishers in R2_COUNTRY_PUBLISHERS.items()
            if publisher in publishers
        ),
        None,
    )

    source_countries: set[str] = set()
    normalized_source_id = (source_id or "").lower()
    if normalized_source_id:
        source_countries = {
            country
            for country, publishers in R2_COUNTRY_PUBLISHERS.items()
            if any(
                normalized_source_id == publisher
                or normalized_source_id.startswith(f"{publisher}_")
                or normalized_source_id.startswith(f"{publisher}-")
                for publisher in publishers
            )
        }
        if len(source_countries) > 1:
            raise ValueError(f"source_id maps to multiple R2 countries: {source_id}")

    source_country = next(iter(source_countries)) if source_countries else None
    if publisher is not None and source_country and source_country != path_country:
        raise ValueError(
            "package publisher directory and source_id map to different "
            f"countries: publisher={publisher!r}, source_id={source_id!r}"
        )
    return path_country if publisher is not None else source_country


def resolve_r2_prefix(
    *,
    prefix: str | None,
    default_prefix: str,
    source_id: str | None = None,
    package_path: str | Path | None = None,
) -> str:
    """Return the country-aware R2 prefix for one source package."""
    resolved = _clean_key_part(prefix or default_prefix)
    country = infer_r2_country(source_id=source_id, package_path=package_path)
    if country is None:
        return resolved

    suffix = resolved.rsplit("/", maxsplit=1)[-1]
    if suffix in R2_COUNTRY_PUBLISHERS:
        if suffix != country:
            raise ValueError(
                f"R2 prefix country {suffix!r} disagrees with publisher "
                f"country {country!r}"
            )
        return resolved
    return posixpath.join(resolved, country)


def build_r2_key(
    *,
    source_id: str,
    package_id: str,
    year: int | str,
    sha256: str,
    filename: str,
    prefix: str | None = None,
    package_path: str | Path | None = None,
) -> str:
    """Build the canonical immutable R2 key for a raw source artifact.

    ``year`` is usually a calendar year but may be a label such as
    ``source_capture`` for non-year manifest file entries.
    """
    resolved_prefix = resolve_r2_prefix(
        prefix=prefix,
        default_prefix=DEFAULT_R2_PREFIX,
        source_id=source_id,
        package_path=package_path,
    )
    return posixpath.join(
        resolved_prefix,
        _clean_key_part(source_id),
        _clean_key_part(package_id),
        str(year),
        sha256,
        Path(filename).name,
    )


def build_derived_r2_key(
    *,
    source_id: str,
    package_id: str,
    year: int,
    build_id: str,
    artifact_name: str,
    prefix: str | None = None,
) -> str:
    """Build the canonical R2 key for a derived build artifact."""
    resolved_prefix = resolve_r2_prefix(
        prefix=prefix,
        default_prefix=default_r2_derived_prefix(),
        source_id=source_id,
    )
    return posixpath.join(
        resolved_prefix,
        _clean_key_part(source_id),
        _clean_key_part(package_id),
        str(year),
        _clean_key_part(build_id),
        *_clean_relative_key_parts(artifact_name),
    )


def build_artifact_key(
    *,
    build_id: str,
    artifact_name: str,
    sha256: str,
    epoch: Epoch = EMIT_EPOCH,
) -> str:
    """Build a stable key for a derived build artifact registry row."""
    payload = json.dumps(
        {
            "artifact_name": artifact_name,
            "build_id": canonicalize_key("build", build_id),
            "sha256": sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    domain = hash_domain("build_artifact", epoch)
    return f"{domain}:{hashlib.sha256(payload).hexdigest()[:32]}"


def infer_build_id(input_dir: str | Path) -> str | None:
    """Infer a build ID from standard Chronicle build-suite outputs."""
    input_path = Path(input_dir)
    summary_path = input_path / "reports" / "build_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        build_id = payload.get("reports", {}).get("database", {}).get("build_id")
        if build_id:
            return str(build_id)

    database_report_path = input_path / "reports" / "database.json"
    if database_report_path.exists():
        payload = json.loads(database_report_path.read_text(encoding="utf-8"))
        build_id = payload.get("build_id")
        if build_id:
            return str(build_id)

    db_path = input_path / CHRONICLE_DB_FILENAME
    if not db_path.exists():
        db_path = input_path / LEGACY_CHRONICLE_DB_FILENAME
    if db_path.exists():
        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT build_id FROM ledger_builds ORDER BY build_id LIMIT 1"
            ).fetchone()
            if row:
                return str(row[0])
    return None


def _read_artifact(source_url: str) -> tuple[bytes, str]:
    parsed = urlparse(source_url)
    if parsed.scheme in ("http", "https"):
        response = httpx.get(source_url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        return response.content, _filename_from_url(source_url)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        return path.read_bytes(), path.name
    if not parsed.scheme:
        path = Path(source_url)
        return path.read_bytes(), path.name
    raise ValueError(f"Unsupported source URL scheme: {parsed.scheme}")


def _filename_from_url(source_url: str) -> str:
    parsed = urlparse(source_url)
    return Path(unquote(parsed.path)).name


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    """Return a manifest's parsed payload, refusing a document it cannot read.

    An absent or empty manifest reads as an empty mapping: ``fetch-artifact``
    writes the first entry into a package that has none. A document that parses
    as anything else -- a list, a scalar, a truncated or half-merged file -- is
    not an absent manifest, and treating it as one would let the fetch replace
    it with a single entry and drop everything it recorded.
    """
    if manifest_path.is_symlink():
        _require_regular_manifest_file(manifest_path)
    if not manifest_path.exists():
        return {}
    _require_regular_manifest_file(manifest_path)
    try:
        payload = load_manifest_document(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MalformedManifestError(
            f"{manifest_path} is not valid YAML: {exc}"
        ) from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise MalformedManifestError(
            f"{manifest_path} must be a YAML mapping; it parses as a "
            f"{type(payload).__name__}. Chronicle will not overwrite a manifest "
            "it cannot read."
        )
    return payload


def _infer_artifact_filename(source_url: str) -> str:
    """Return the filename :func:`_read_artifact` would report, without I/O.

    The name is a pure function of the URL: the last path segment for http(s)
    and ``file://`` URLs, the basename for a bare path. Resolving it before
    the read lets every filename guard run before the publisher is touched.
    """
    parsed = urlparse(source_url)
    if parsed.scheme in ("http", "https"):
        return _filename_from_url(source_url)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).name
    if not parsed.scheme:
        return Path(source_url).name
    raise ValueError(f"Unsupported source URL scheme: {parsed.scheme}")


@dataclass(frozen=True)
class ExpectedIdentity:
    """The bytes a reviewed pin says the publisher serves, if any."""

    sha256: str | None
    size_bytes: int | None


def _expected_identity(
    expected_sha256: str | None,
    expected_size_bytes: int | None,
) -> ExpectedIdentity:
    """Validate the expected-identity arguments before any I/O."""
    sha256 = expected_sha256.strip() if isinstance(expected_sha256, str) else None
    if expected_sha256 is not None and (
        not sha256 or not _SHA256_KEY_SEGMENT.fullmatch(sha256)
    ):
        raise ExpectedArtifactIdentityError(
            "--expected-sha256 must be a lowercase 64-character SHA-256 taken "
            f"from a reviewed pin, not {expected_sha256!r}. Never invent a hash."
        )
    if expected_size_bytes is not None and (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes <= 0
    ):
        raise ExpectedArtifactIdentityError(
            "--expected-size-bytes must be a positive integer, not "
            f"{expected_size_bytes!r}."
        )
    return ExpectedIdentity(sha256=sha256, size_bytes=expected_size_bytes)


def _assert_expected_identity(
    expected: ExpectedIdentity,
    *,
    manifest_path: Path,
    year: Any,
    filename: str,
    source_url: str,
    sha256: str,
    size_bytes: int,
) -> None:
    """Refuse fetched bytes the reviewed pin does not cover.

    ``--record-revision`` does not override this: it governs what the manifest
    records about bytes Chronicle chose to archive, and an expectation is the
    statement that only reviewed bytes are archived at all.
    """
    if expected.sha256 is None and expected.size_bytes is None:
        return
    if (expected.sha256 is None or expected.sha256 == sha256) and (
        expected.size_bytes is None or expected.size_bytes == size_bytes
    ):
        return
    raise ExpectedArtifactIdentityError(
        f"{manifest_path} entry {year!r} {filename}: the bytes served by "
        f"{source_url} are not the bytes the reviewed pin covers. Expected "
        f"sha256={expected.sha256 or 'unspecified'} "
        f"size_bytes={expected.size_bytes or 'unspecified'}; fetched "
        f"sha256={sha256} size_bytes={size_bytes}. The publisher is serving "
        "bytes the pin does not describe. Chronicle will not archive an "
        "unreviewed reissue; re-review the release and re-run with the pin you "
        "reviewed."
    )


def _resolve_manifest_kind(
    existing_manifest: dict[str, Any],
    *,
    manifest_path: Path,
    requested_kind: str | None,
) -> str:
    """Return the kind a fetch writes, refusing a conflict with the manifest.

    A manifest's kind is fixed once declared: an explicit ``--kind`` that
    differs from it would reclassify every entry the manifest holds as a side
    effect of one fetch. A kindless manifest with content is refused unless it
    is frozen kindless byte for byte, and a stored kind Chronicle does not
    recognise is never masked by the command line.
    """
    requested = (
        normalize_manifest_kind({"kind": requested_kind})
        if requested_kind is not None
        else None
    )
    try:
        stored = normalize_manifest_kind(existing_manifest, manifest_path=manifest_path)
    except ManifestKindError as exc:
        raise ManifestAccessError(
            f"{exc} fetch-artifact will not add to a manifest whose kind it "
            "cannot read; declare the kind by editing the manifest deliberately."
        ) from exc
    except ManifestAccessError as exc:
        raise ManifestAccessError(
            f"{manifest_path} declares an unknown manifest kind "
            f"{existing_manifest.get('kind')!r}; expected one of "
            f"{list(MANIFEST_KINDS)}. Fix the manifest before fetching into it."
        ) from exc
    if requested is None:
        return stored
    if requested != stored and (
        existing_manifest.get("kind") is not None or has_file_entries(existing_manifest)
    ):
        # A declared kind is fixed, and a frozen kindless manifest with
        # entries is a publisher table. A manifest with neither is declared
        # by this fetch.
        raise ManifestAccessError(
            f"{manifest_path} is a {stored} manifest; refusing to fetch into it "
            f"as a {requested}. A manifest's kind is fixed once declared: "
            f"register a {requested} in its own package directory, or migrate "
            "this manifest deliberately."
        )
    return requested


def _complete_manifest_validation_errors(
    manifest: dict[str, Any],
    *,
    kind: str,
    package_dir: Path,
) -> list[str]:
    """Return all manifest- and entry-level validation errors."""
    return [
        *validate_manifest_files(manifest),
        *_manifest_entry_validation_errors(
            manifest,
            kind=kind,
            package_dir=package_dir,
        ),
    ]


def _manifest_entry_validation_errors(
    manifest: dict[str, Any],
    *,
    kind: str,
    package_dir: Path,
) -> list[str]:
    """Return entry-level validation errors for a complete manifest."""
    codes: list[str] = []
    files = manifest.get("files") or {}
    if isinstance(files, dict):
        for key, spec in files.items():
            for file_spec in iter_file_specs(spec, kind=kind):
                name = (
                    file_spec.get("filename") if isinstance(file_spec, dict) else None
                )
                try:
                    exists = matching_directory_entry(package_dir, name) is not None
                except ValueError:
                    codes.append(f"duplicate_artifact_spellings:{name}")
                    continue
                for code in validate_file_entry(
                    file_spec,
                    kind=kind,
                    manifest=manifest,
                    local_file_exists=exists,
                ):
                    codes.append(f"{key!r}/{name}: {code}")
    return codes


def _assert_manifest_valid_for_fetch(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    kind: str,
    package_dir: Path,
) -> None:
    """Refuse to fetch into a manifest inventory would report as invalid.

    Uses the exact vocabulary ``inventory-artifacts`` and ``publish-raw``
    report, so a fetch never carries an invalid registration forward -- or
    conceals one under a rewrite.
    """
    if kind != MICRODATA_RELEASE_KIND:
        for year, entry in _manifest_files(manifest, manifest_path).items():
            if not isinstance(entry, dict):
                raise MalformedManifestError(
                    f"{manifest_path} entry {year!r} must be a mapping; "
                    f"it is a {type(entry).__name__}."
                )
    codes = _complete_manifest_validation_errors(
        manifest,
        kind=kind,
        package_dir=package_dir,
    )
    if codes:
        raise ManifestAccessError(
            f"{manifest_path} is not a valid {kind} manifest: "
            f"{'; '.join(codes)}. fetch-artifact rewrites this manifest and "
            "will not carry an invalid registration forward; "
            "inventory-artifacts reports the same codes."
        )


def _assert_recorded_locators_valid(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    kind: str,
) -> None:
    """Refuse a manifest whose recorded R2 provenance Chronicle cannot read.

    Entry validation reports the vocabulary
    :func:`_assert_manifest_valid_for_fetch` carries, and that vocabulary
    never opens ``storage``; the package-wide readers resolve locators
    non-raisingly, so an unreadable one reads as absent provenance to every
    check but this. Run over every entry of every manifest in the directory by
    each command that writes into one -- the fetch preflight before the
    publisher is read, :func:`_upsert_manifest` over the proposed tree under
    the lock, and ``_prepare_registration_payload`` before it persists -- so
    one validator, :func:`_validated_recorded_r2`, decides in all three.
    """
    for vintage, _index, entry in iter_manifest_entries(manifest):
        locator = _validated_recorded_r2(
            entry,
            manifest_path=manifest_path,
            year=vintage,
            source_id=manifest.get("source_id"),
            package_id=manifest.get("package_id"),
            bind_registration_identity=kind == MICRODATA_RELEASE_KIND,
        )
        if locator is not None and (
            locator.filename != entry.get("filename")
            or (entry.get("sha256") is not None and locator.sha256 != entry["sha256"])
        ):
            raise RecordedR2LocatorError(
                f"{manifest_path} entry {vintage!r}: recorded_r2_identity_mismatch"
            )


def _release_fetch_evidence(
    manifest_path: Path,
    existing_manifest: dict[str, Any],
    *,
    filename: str,
    package_dir: Path,
    licence: str | None,
    vintage: str | None,
    publisher: str | None,
    expected: ExpectedIdentity,
    licence_evidence: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return the ``licence_evidence`` block a public release fetch records.

    Bytes only with artifact-bound redistribution evidence: the licence must
    be on the allowlist, the evidence must bind this artifact (by the reviewed
    SHA-256) to that term, and the release must carry its publisher and
    vintage. Public microdata is staged outside the package tree, so a file of
    that name beside the manifest is refused as tracked microdata bytes.
    """
    if not licence:
        raise ManifestAccessError(
            f"{manifest_path} registers a microdata release, so every entry "
            "must record its publisher licence; pass --licence."
        )
    if not (vintage and vintage.strip()):
        raise ManifestAccessError(
            f"{manifest_path} registers a microdata release, so every entry "
            "must record its publisher vintage; pass --vintage."
        )
    if not (publisher or existing_manifest.get("publisher")):
        raise ManifestAccessError(
            f"{manifest_path} registers a microdata release, so it must name "
            "the publisher; pass --publisher."
        )
    if expected.sha256 is None:
        raise ManifestAccessError(
            "A public microdata release is archived only against a reviewed "
            "checksum that its licence evidence covers; pass --expected-sha256."
        )
    if not is_redistributable_licence(licence):
        raise ManifestAccessError(
            f"licence {licence!r} is not on Chronicle's allowlist of "
            f"redistributable terms {sorted(REDISTRIBUTABLE_LICENCES)}. A "
            "public-download file without redistribution evidence is classed "
            "licensed: register it hash-only with `chronicle register-artifact`."
        )
    supplied = dict(licence_evidence or {})
    evidence = {
        "issuer": str(supplied.get("issuer") or "").strip(),
        "licence": licence,
        "scope": str(supplied.get("scope") or "").strip(),
        "url": str(supplied.get("url") or "").strip(),
        "sha256": expected.sha256,
    }
    codes = licence_evidence_errors(evidence, licence=licence, sha256=expected.sha256)
    if codes:
        raise ManifestAccessError(
            f"{manifest_path}: archiving {filename!r} needs licence evidence "
            f"binding it to {licence!r}: {', '.join(codes)}. Pass "
            "--licence-evidence-issuer, --licence-evidence-scope and a durable "
            "--licence-evidence-url; the evidence covers --expected-sha256."
        )
    local_entry = matching_directory_entry(package_dir, filename)
    if local_entry is not None:
        raise ManifestAccessError(
            f"{local_entry} exists beside the manifest. Public "
            "microdata bytes are staged outside the package tree and uploaded "
            "from there; a repository never holds them. Remove the file first."
        )
    return evidence


def _select_vintage_entry(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    year: Any,
    filename: str,
    kind: str,
) -> tuple[Any, Any, dict[str, Any], int | None]:
    """Locate the entry a fetch revises: ``(key, files[key], entry, index)``.

    The vintage key is whichever spelling the manifest already uses for
    ``year`` (``2023`` and ``'2023'`` are one vintage). A publisher table's
    vintage is one mapping, and that entry is its identity whatever filename
    it records. A microdata release lists several files under one vintage, and
    the entry is the one whose bare filename matches. ``index`` is the entry's
    position in that list, or None.
    """
    files = payload.get("files") if isinstance(payload, dict) else None
    if files is None:
        return year, None, {}, None
    if not isinstance(files, dict):
        raise MalformedManifestError(
            f"{manifest_path} files must be a mapping; it is a "
            f"{type(files).__name__}. Chronicle will not write into a manifest "
            "it cannot read."
        )
    try:
        key = resolve_vintage_key(files, year)
    except AmbiguousVintageKeyError as exc:
        raise MalformedManifestError(
            f"{manifest_path}: {exc} Chronicle will not choose which entry is "
            "the record."
        ) from exc
    if key is None:
        return year, None, {}, None
    existing = files[key]
    if isinstance(existing, dict):
        return key, existing, existing, None
    if isinstance(existing, list):
        if kind != MICRODATA_RELEASE_KIND:
            raise MalformedManifestError(
                f"{manifest_path} entry {key!r} lists {len(existing)} files, but "
                f"the manifest is a {kind} manifest. Only a kind: "
                "microdata_release manifest may list several files under one "
                "vintage (list_file_spec_requires_microdata_release_kind); "
                "Chronicle will not add to or revise a vintage it cannot "
                "validate."
            )
        wanted = filename_key(filename)
        matches = [
            (index, entry)
            for index, entry in enumerate(existing)
            if isinstance(entry, dict) and filename_key(entry.get("filename")) == wanted
        ]
        if len(matches) > 1:
            raise MalformedManifestError(
                f"{manifest_path} entry {key!r} lists {len(matches)} entries "
                f"named {filename!r}; a fetch can revise exactly one, and "
                "Chronicle will not guess which."
            )
        if matches:
            return key, existing, matches[0][1], matches[0][0]
        return key, existing, {}, None
    raise MalformedManifestError(
        f"{manifest_path} entry {key!r} must be a mapping or a list of "
        f"mappings; it is a {type(existing).__name__}."
    )


def _assert_table_vintage_is_revisable(
    existing_value: Any,
    recorded_identity: RecordedIdentity | None,
    *,
    manifest_path: Path,
    year: Any,
    filename: str,
    release: bool,
) -> None:
    """Refuse a second filename in a publisher-table vintage with no identity.

    A publisher table holds one file per vintage. When its entry identifies
    bytes, a fetch under another name is judged by the recorded identity (and
    ``--record-revision`` supersedes the entry). An entry that identifies
    nothing -- no ``sha256``, no recorded object -- cannot be superseded, and
    adding a second file would turn the mapping into a list every reader
    refuses.
    """
    if release or not isinstance(existing_value, dict) or recorded_identity:
        return
    recorded_name = existing_value.get("filename")
    if recorded_name is None or filename_key(recorded_name) == filename_key(filename):
        return
    raise MalformedManifestError(
        f"{manifest_path} entry {year!r} records {recorded_name!r} without a "
        f"sha256; a publisher table holds one file per vintage, so {filename!r} "
        "cannot be added and there is no recorded identity to supersede. "
        "Register the vintage's bytes first or fix the entry by hand."
    )


def _recorded_storage(spec: Any) -> dict[str, Any]:
    """Return a manifest file spec's recorded ``storage`` block, if any."""
    if not isinstance(spec, dict):
        return {}
    storage = spec.get("storage")
    return storage if isinstance(storage, dict) else {}


def _recorded_r2(spec: Any) -> dict[str, Any]:
    """Return the recorded ``storage.r2`` block verbatim, if any.

    Raw access, for callers that carry the block forward as history. Callers
    that reason about which object it names go through
    :func:`_validated_recorded_r2` instead.
    """
    recorded = _recorded_storage(spec).get("r2")
    return recorded if isinstance(recorded, dict) else {}


def _validated_recorded_storage(
    spec: Any,
    *,
    manifest_path: Path,
    year: Any,
    source_id: str = "",
    package_id: str = "",
    bind_registration_identity: bool = False,
) -> dict[str, Any]:
    """Return the entry's ``storage`` mapping, refusing a malformed one.

    Every archived object is a locator like the current one, and is held to the
    same rules: an element Chronicle cannot read is unreadable provenance, not
    absent provenance. :func:`_recorded_object_identities` skips a block it
    cannot address, so a malformed element would otherwise hide the identity
    its key carries from every archived-alias check while the entry still read
    as valid. ``_prepare_registration_payload`` already refuses these elements;
    validating them here is what makes the artifact commands agree.
    """
    if not isinstance(spec, dict) or "storage" not in spec:
        return {}
    storage = spec["storage"]
    if not isinstance(storage, dict):
        raise MalformedManifestError(
            f"{manifest_path} entry {year!r} storage must be a mapping; it is a "
            f"{type(storage).__name__}."
        )
    if "previous_r2" in storage and not isinstance(storage["previous_r2"], list):
        previous = storage["previous_r2"]
        raise MalformedManifestError(
            f"{manifest_path} entry {year!r} storage.previous_r2 must be a "
            f"list; it is a {type(previous).__name__}. Chronicle will not "
            "discard malformed archived provenance."
        )
    for index, archived in enumerate(storage.get("previous_r2") or ()):
        try:
            _validated_recorded_r2(
                {"storage": {"r2": archived}},
                manifest_path=manifest_path,
                year=year,
                source_id=source_id,
                package_id=package_id,
                bind_registration_identity=bind_registration_identity,
            )
        except SourceArtifactManifestError as error:
            raise RecordedR2LocatorError(
                f"{manifest_path} entry {year!r} storage.previous_r2[{index}]: {error}"
            ) from error
    return storage


def _validated_recorded_r2(
    spec: Any,
    *,
    manifest_path: Path,
    year: Any,
    source_id: str = "",
    package_id: str = "",
    bind_registration_identity: bool = False,
) -> RecordedR2Object | None:
    """Return the object a recorded ``storage.r2`` block names, or None.

    Every locator field the block supplies is cross-checked against every
    other: ``key`` against the URI's path, ``bucket`` against its authority,
    ``provider`` against its scheme, and the resulting key against the
    canonical content-addressed shape :func:`build_r2_key` writes. Reading one
    field and trusting the rest is what lets a block that says two different
    things survive a preserve or a publish.
    """
    storage = _validated_recorded_storage(
        spec,
        manifest_path=manifest_path,
        year=year,
        source_id=source_id,
        package_id=package_id,
        bind_registration_identity=bind_registration_identity,
    )
    if "r2" not in storage:
        return None
    block = storage["r2"]
    where = f"{manifest_path} entry {year!r} storage.r2"
    if not isinstance(block, dict):
        raise MalformedManifestError(
            f"{where} must be a mapping; it is a {type(block).__name__}."
        )

    supplied: dict[str, str] = {}
    for field in ("provider", "bucket", "key", "uri"):
        value = block.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise RecordedR2LocatorError(
                f"{where}: {field} must be a non-empty string, not {value!r}."
            )
        supplied[field] = value

    provider = supplied.get("provider")
    bucket = supplied.get("bucket")
    key = supplied.get("key")
    uri = supplied.get("uri")
    missing_required = [
        field for field in ("provider", "uri") if not supplied.get(field)
    ]
    if missing_required:
        raise RecordedR2LocatorError(
            f"{where}: records no {', '.join(missing_required)}. A block under "
            "storage.r2 must explicitly record provider='r2' and an r2:// URI."
        )
    if provider != "r2":
        raise RecordedR2LocatorError(
            f"{where}: provider={provider!r} does not identify R2. A block "
            "under storage.r2 must use provider='r2' and an r2:// URI, not "
            f"{provider}://."
        )
    if uri is not None:
        parts = _split_r2_uri(uri)
        if parts is None:
            raise RecordedR2LocatorError(
                f"{where}: uri {uri!r} is not provider://bucket/key."
            )
        for field, value, from_uri in zip(
            ("provider", "bucket", "key"), (provider, bucket, key), parts
        ):
            if value is not None and value != from_uri:
                raise RecordedR2LocatorError(
                    f"{where}: {field}={value!r} contradicts uri {uri!r}, which "
                    f"names {from_uri!r}. The block records two different "
                    "objects, so Chronicle cannot say which bytes it claims."
                )
        provider, bucket, key = (
            provider or parts[0],
            bucket or parts[1],
            key or parts[2],
        )

    missing = [
        field
        for field, value in (
            ("provider", provider),
            ("bucket", bucket),
            ("key", key),
        )
        if not value
    ]
    if missing:
        raise RecordedR2LocatorError(
            f"{where}: records no {', '.join(missing)}. A recorded block has to "
            "locate its object: provider, bucket and key, or a uri that "
            "supplies them."
        )
    if provider != "r2":
        raise RecordedR2LocatorError(
            f"{where}: provider must be 'r2', not {provider!r}. A block under "
            "storage.r2 cannot record another storage service."
        )

    segments = key.split("/")
    if (
        len(segments) < 2
        or not all(segments)
        or not _SHA256_KEY_SEGMENT.fullmatch(segments[-2])
    ):
        raise RecordedR2LocatorError(
            f"{where}: key {key!r} is not content-addressed. A raw key ends in "
            "{sha256}/{filename}, which is what says the object holds the "
            "entry's bytes; Chronicle will not guess for a key that does not."
        )
    if bind_registration_identity:
        try:
            expected_release = (
                _clean_key_part(source_id),
                _clean_key_part(package_id),
                str(year),
            )
        except ValueError as exc:
            raise RecordedR2LocatorError(
                f"{where}: cannot bind the locator to source_id={source_id!r}, "
                f"package_id={package_id!r}, year={year!r}: {exc}"
            ) from exc
        recorded_release = tuple(segments[-5:-2])
        if recorded_release != expected_release:
            raise RecordedR2LocatorError(
                f"{where}: key {key!r} is bound to source/package/year "
                f"{recorded_release!r}, not {expected_release!r}. A recorded "
                "release object must carry the complete registration identity."
            )
    return RecordedR2Object(
        provider=provider,
        bucket=bucket,
        key=key,
        sha256=segments[-2],
        filename=segments[-1],
    )


def _recorded_identity(
    spec: Any,
    *,
    manifest_path: Path,
    year: Any,
    source_id: str = "",
    package_id: str = "",
    bind_registration_identity: bool = False,
) -> RecordedIdentity | None:
    """Return what a manifest entry says its vintage holds, if anything.

    A published entry is identified by its recorded object's content-addressed
    key. An entry that has not been published yet -- registered without an
    upload, or left behind by a failed one -- is identified by its own declared
    ``sha256`` and ``filename``. Both are recorded identities, and a fetch of
    different bytes over either one is a publisher revision.
    """
    recorded_r2 = _validated_recorded_r2(
        spec,
        manifest_path=manifest_path,
        year=year,
        source_id=source_id,
        package_id=package_id,
        bind_registration_identity=bind_registration_identity,
    )
    declared_sha256 = spec.get("sha256") if isinstance(spec, dict) else None
    declared_sha256 = declared_sha256 if isinstance(declared_sha256, str) else None
    declared_filename = spec.get("filename") if isinstance(spec, dict) else None
    declared_filename = (
        declared_filename if isinstance(declared_filename, str) else None
    )
    size_bytes = spec.get("size_bytes") if isinstance(spec, dict) else None
    size_bytes = size_bytes if isinstance(size_bytes, int) else None
    if recorded_r2 is not None:
        return RecordedIdentity(
            sha256=recorded_r2.sha256,
            filename=recorded_r2.filename,
            # Only report a size the recorded key agrees with: an entry can
            # arrive here already describing the new bytes.
            size_bytes=size_bytes if declared_sha256 == recorded_r2.sha256 else None,
            declared_sha256=declared_sha256,
            r2=recorded_r2,
        )
    if not declared_sha256:
        return None
    return RecordedIdentity(
        sha256=declared_sha256,
        filename=Path(declared_filename).name if declared_filename else "",
        size_bytes=size_bytes,
        declared_sha256=declared_sha256,
        r2=None,
    )


def _revision_error_message(
    *,
    manifest_path: Path,
    year: Any,
    filename: str,
    identity: RecordedIdentity,
    sha256: str,
    size_bytes: int,
    r2_bucket: str,
) -> str:
    """Explain a refused fetch: recorded identity, fetched identity, next step."""
    records = (
        f"already records the R2 object {identity.r2.uri}, which holds"
        if identity.r2 is not None
        else "already records"
    )
    message = (
        f"{manifest_path} entry {year!r} {records} "
        f"sha256={identity.sha256} "
        f"filename={identity.filename or 'unknown'} "
        f"size_bytes="
        f"{identity.size_bytes if identity.size_bytes is not None else 'unknown'}. "
        f"The fetched bytes are sha256={sha256} filename={Path(filename).name} "
        f"size_bytes={size_bytes}. Chronicle will not rewrite a vintage that "
        "identifies specific bytes to describe bytes it never identified."
    )
    if identity.r2 is not None and identity.declared_sha256 not in (
        None,
        identity.sha256,
    ):
        message += (
            f" (The entry also declares sha256={identity.declared_sha256}, which "
            "its own R2 key contradicts: an earlier fetch rewrote the hash "
            "without moving the object.)"
        )
    return message + (
        " The same vintage with new bytes is a new release revision "
        "(docs/adr-chronicle-fact-identity-v2.md). Re-run with "
        "--record-revision to store the fetched bytes under their own "
        f"content-addressed key in {r2_bucket} and keep the superseded object "
        "in storage.previous_r2."
    )


def _assert_recorded_identity_holds_these_bytes(
    identity: RecordedIdentity | None,
    *,
    manifest_path: Path,
    year: Any,
    filename: str,
    sha256: str,
    size_bytes: int,
    r2_bucket: str,
    record_revision: bool,
) -> None:
    """Refuse a publisher revision that has not been opted into."""
    if identity is None or identity.holds(sha256=sha256, filename=filename):
        return
    if identity.sha256 == sha256:
        # The recorded object holds exactly these bytes under another name. A
        # rename is not a publisher revision, so --record-revision does not
        # apply, and silently adopting the new name would leave the entry's
        # filename disagreeing with the key its own storage block records.
        raise SourceArtifactRevisionError(
            f"{manifest_path} entry {year!r} already records these exact bytes "
            f"(sha256={sha256}) as filename={identity.filename}; this fetch "
            f"names them {Path(filename).name}. A rename is not a release "
            "revision, so --record-revision does not apply. Re-run with "
            f"--filename {identity.filename} to keep the recorded identity."
        )
    if record_revision:
        return
    raise SourceArtifactRevisionError(
        _revision_error_message(
            manifest_path=manifest_path,
            year=year,
            filename=filename,
            identity=identity,
            sha256=sha256,
            size_bytes=size_bytes,
            r2_bucket=r2_bucket,
        )
    )


def _superseding_storage(
    recorded_spec: dict[str, Any],
    *,
    recorded_r2: RecordedR2Object | None,
    new_r2: dict[str, Any] | None,
    superseded_at: str,
) -> dict[str, Any]:
    """Return a storage block in which the recorded object becomes history.

    ``storage.r2`` only ever names the object that holds the entry's current
    bytes. The superseded block is appended, oldest first, to
    ``storage.previous_r2`` so the earlier bytes stay addressable by the URI
    archived witness records already pin. An entry that was never published has
    no object to supersede, and gets no ``previous_r2`` key.
    """
    storage = dict(_recorded_storage(recorded_spec))
    previous = storage.get("previous_r2")
    entries = list(previous) if isinstance(previous, list) else []
    if recorded_r2 is not None:
        entry = dict(_recorded_r2(recorded_spec))
        entry["sha256"] = recorded_r2.sha256
        if recorded_spec.get("sha256") == recorded_r2.sha256:
            # Only carry metadata the superseded key agrees with: a manifest
            # can arrive here already describing the new bytes.
            for field in ("size_bytes", "fetched_at", "source_url"):
                value = recorded_spec.get(field)
                if value is not None:
                    entry[field] = value
        entry["superseded_at"] = superseded_at
        entries.append(entry)
    if entries:
        storage["previous_r2"] = entries
    if new_r2 is None:
        storage.pop("r2", None)
    else:
        storage["r2"] = new_r2
    return storage


#: Entry fields a fetch owns. Everything else an entry already records --
#: notes, doi, study, access_route, a vintage the fetch did not restate -- is
#: carried forward when the fetch replaces that entry, so a re-fetch is never a
#: silent de-registration.
_FETCH_OWNED_FIELDS: frozenset[str] = frozenset(
    {
        "filename",
        "source_url",
        "access",
        "licence",
        "licence_evidence",
        "sha256",
        "size_bytes",
        "fetched_at",
        "verified_at",
        "hash_source",
        "attested_by",
        "storage",
    }
)


def _upsert_manifest(
    manifest_path: Path,
    *,
    source_id: str,
    package_id: str,
    dataset: str,
    source_page: str | None,
    table: str | None,
    publisher: str | None,
    year: int,
    filename: str,
    source_url: str,
    sha256: str,
    size_bytes: int,
    fetched_at: str,
    access: str,
    licence: str | None,
    kind: str,
    vintage: str | None,
    licence_evidence: Mapping[str, Any] | None,
    expected: ExpectedIdentity,
    r2_location: ArtifactStorageLocation | None,
    record_revision: bool = False,
    _preflight_only: bool = False,
) -> None:
    """Prepare and validate a complete package update before persisting it.

    ``_preflight_only`` performs the identical proposal and validation without
    creating directories or writing manifests, so fetch can refuse the update
    before staging or uploading publisher bytes.

    The guards fetch_source_artifact ran are repeated against the freshly
    re-read manifest, so no caller can reach a false-provenance write by
    another route. The entry the fetch revises is located by vintage key and
    bare filename and replaced where it sits; a publisher-table vintage stays
    one mapping and a release vintage stays a list, and every manifest this
    command touches declares its kind.
    """
    payload = _read_manifest(manifest_path)
    kind = _resolve_manifest_kind(
        payload, manifest_path=manifest_path, requested_kind=kind
    )
    _assert_manifest_identifies(
        payload, manifest_path, source_id=source_id, package_id=package_id
    )
    manifests = _package_manifests(manifest_path.parent, manifest_path, payload)
    for sibling_path, sibling in manifests.items():
        _assert_no_hash_only_entry(sibling, Path(sibling_path), filename)
    _assert_no_hash_only_bytes(
        manifests, sha256=sha256, filename=filename, what="the fetched bytes"
    )
    owners = _manifest_file_owners(
        manifests, filename=filename, initializing=(manifest_path, year)
    )
    _assert_shared_owner_identities_agree(owners, filename=filename)
    if not record_revision:
        _assert_siblings_record_these_bytes(
            manifests,
            manifest_path=manifest_path,
            vintage=year,
            filename=filename,
            sha256=sha256,
        )
    payload.setdefault("source_id", source_id)
    payload.setdefault("package_id", package_id)
    payload = _with_declared_kind(payload, kind)
    payload.setdefault("dataset", dataset)
    if publisher:
        payload.setdefault("publisher", publisher)
    payload.setdefault("source_page", source_page)
    payload.setdefault("table", table)
    if payload.get("files") is None:
        # setdefault keeps an explicit null (a bare ``files:`` line); the
        # entry below needs a mapping to record into.
        payload["files"] = {}
    release = kind == MICRODATA_RELEASE_KIND

    key, existing_value, recorded_spec, index = _select_vintage_entry(
        payload,
        manifest_path=manifest_path,
        year=year,
        filename=filename,
        kind=kind,
    )
    recorded_storage = _recorded_storage(recorded_spec)
    identity = _recorded_identity(
        recorded_spec,
        manifest_path=manifest_path,
        year=key,
        source_id=source_id,
        package_id=package_id,
        bind_registration_identity=release,
    )
    _assert_expected_identity(
        expected,
        manifest_path=manifest_path,
        year=key,
        filename=filename,
        source_url=source_url,
        sha256=sha256,
        size_bytes=size_bytes,
    )
    new_r2 = r2_location.to_dict() if r2_location is not None else None
    # Different bytes under the same vintage, or the same bytes under another
    # name: fetch_source_artifact refuses both before the read; the guard is
    # repeated here so no caller can reach a false-provenance write.
    _assert_recorded_identity_holds_these_bytes(
        identity,
        manifest_path=manifest_path,
        year=key,
        filename=filename,
        sha256=sha256,
        size_bytes=size_bytes,
        r2_bucket=(new_r2 or {}).get("bucket") or default_r2_raw_bucket(),
        record_revision=record_revision,
    )
    holds = identity is not None and identity.holds(sha256=sha256, filename=filename)
    if holds and identity.r2 is not None:
        # A recorded storage.r2 block for these exact bytes is historical
        # truth: archived witness records pin raw R2 URLs by hash. Re-fetching
        # under a renamed bucket copies bytes; it does not restate where the
        # bytes were first published (PolicyEngine/chronicle#143, mechanism 3).
        storage = {**recorded_storage, "r2": _recorded_r2(recorded_spec)}
    elif identity is not None and not holds:
        storage = _superseding_storage(
            recorded_spec,
            recorded_r2=identity.r2,
            new_r2=new_r2,
            superseded_at=fetched_at,
        )
    elif new_r2 is not None:
        storage = {**recorded_storage, "r2": new_r2}
    else:
        storage = dict(recorded_storage)

    # Access is written explicitly on every entry this command touches, so a
    # manifest never relies on the inferred ``public`` default once rewritten.
    file_entry: dict[str, Any] = {"filename": filename}
    if release:
        file_entry["access"] = access
        file_entry["licence"] = licence
        if licence_evidence:
            file_entry["licence_evidence"] = dict(licence_evidence)
        file_entry["vintage"] = vintage or recorded_spec.get("vintage")
        file_entry["sha256"] = sha256
        file_entry["size_bytes"] = size_bytes
        file_entry["source_url"] = source_url
        file_entry["fetched_at"] = fetched_at
        # Chronicle fetched and hashed these bytes itself.
        file_entry["verified_at"] = fetched_at[:10]
        file_entry["hash_source"] = HASH_SOURCE_CHRONICLE_FETCH
        file_entry["attested_by"] = CHRONICLE_ATTESTER
    else:
        file_entry["source_url"] = source_url
        file_entry["access"] = access
        if licence:
            file_entry["licence"] = licence
        if vintage:
            file_entry["vintage"] = vintage
        file_entry["sha256"] = sha256
        file_entry["size_bytes"] = size_bytes
        file_entry["fetched_at"] = fetched_at
    for field, value in recorded_spec.items():
        if field not in _FETCH_OWNED_FIELDS and field not in file_entry:
            file_entry[field] = value
    # An entry that has no storage to record carries no empty block: a
    # revision over a never-published entry supersedes nothing.
    if storage:
        file_entry["storage"] = storage

    if existing_value is None:
        payload["files"][key] = [file_entry] if release else file_entry
    elif isinstance(existing_value, dict):
        if not release:
            # A publisher table holds one file per vintage; a rename under
            # --record-revision supersedes that one entry, and the superseded
            # key in storage.previous_r2 keeps the old name.
            payload["files"][key] = file_entry
        elif filename_key(existing_value.get("filename")) == filename_key(filename):
            payload["files"][key] = [file_entry]
        else:
            payload["files"][key] = [existing_value, file_entry]
    else:
        entries = list(existing_value)
        if index is not None:
            entries[index] = file_entry
        else:
            entries.append(file_entry)
        payload["files"][key] = entries
    manifests[str(manifest_path)] = payload
    revision = any(
        owner.identity is not None
        and not owner.identity.holds(sha256=sha256, filename=filename)
        for owner in owners
    ) or (identity is not None and not identity.holds(sha256=sha256, filename=filename))
    changed_paths = {manifest_path}
    if record_revision and revision:
        for owner in owners:
            if owner.manifest_path == manifest_path and str(owner.vintage) == str(key):
                continue
            revised_entry = dict(owner.spec)
            revised_entry.update(
                filename=filename,
                sha256=sha256,
                size_bytes=size_bytes,
                fetched_at=fetched_at,
            )
            owner_storage = _storage_for_fetched_identity(
                owner.spec,
                identity=owner.identity,
                filename=filename,
                sha256=sha256,
                new_r2=new_r2,
                fetched_at=fetched_at,
            )
            if owner_storage:
                revised_entry["storage"] = owner_storage
            else:
                revised_entry.pop("storage", None)
            owner_payload = manifests[str(owner.manifest_path)]
            owner_payload = _with_declared_kind(
                owner_payload,
                normalize_manifest_kind(
                    owner_payload, manifest_path=owner.manifest_path
                ),
            )
            manifests[str(owner.manifest_path)] = owner_payload
            owner_value = owner_payload["files"][owner.vintage]
            if isinstance(owner_value, list):
                owner_payload["files"][owner.vintage] = [
                    revised_entry if entry is owner.spec else entry
                    for entry in owner_value
                ]
            else:
                owner_payload["files"][owner.vintage] = revised_entry
            changed_paths.add(owner.manifest_path)
    for proposed_path, proposed in manifests.items():
        path = Path(proposed_path)
        proposed_kind = normalize_manifest_kind(proposed, manifest_path=path)
        _assert_manifest_valid_for_fetch(
            proposed, path, kind=proposed_kind, package_dir=path.parent
        )
        _assert_recorded_locators_valid(proposed, path, kind=proposed_kind)
    _assert_package_file_owner_identities_agree(manifests)
    # The rewrite revises owner entries in sibling manifests too, so the
    # proposed directory gets the same package-wide sweep as the one read.
    _assert_no_package_microdata_identities(manifests)

    for owner in owners:
        if owner.manifest_path == manifest_path and str(owner.vintage) == str(key):
            continue
        _assert_recorded_identity_holds_these_bytes(
            owner.identity,
            manifest_path=owner.manifest_path,
            year=owner.vintage,
            filename=filename,
            sha256=sha256,
            size_bytes=size_bytes,
            r2_bucket=(new_r2 or {}).get("bucket") or default_r2_raw_bucket(),
            record_revision=record_revision,
        )
    rendered = {
        path: yaml.safe_dump(manifests[str(path)], sort_keys=False, allow_unicode=True)
        for path in changed_paths
    }
    if _preflight_only:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path, text in sorted(rendered.items(), key=lambda item: str(item[0])):
        path.write_text(text, encoding="utf-8")


def _with_declared_kind(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    """Return ``payload`` declaring ``kind``, placed after its identity keys."""
    if "kind" in payload:
        payload["kind"] = kind
        return payload
    ordered: dict[str, Any] = {}
    inserted = False
    for field, value in payload.items():
        ordered[field] = value
        if field == "package_id" and not inserted:
            ordered["kind"] = kind
            inserted = True
    if not inserted:
        ordered["kind"] = kind
    return ordered


def _assert_no_hash_only_entry(
    manifest: dict[str, Any],
    manifest_path: Path,
    filename: str,
) -> None:
    """Refuse to fetch bytes over an existing hash-only registration.

    The write target is a path in the package directory, so the search spans
    every vintage rather than the requested one: a licensed release registered
    under one year must not be fetched into the tree under another. Names are
    compared as resolved, case-folded bare filenames, so no alias of a
    registered name -- ``./adult.tab``, ``ADULT.TAB`` -- slips past.
    """
    wanted = filename_key(filename)
    for key, _index, spec in iter_manifest_entries(manifest):
        if not isinstance(spec, dict) or spec.get("filename") is None:
            continue
        if filename_key(spec.get("filename")) != wanted:
            continue
        declared = spec.get("access")
        if declared is None:
            # A release entry without an access class is reported by the
            # strict manifest validation that follows; it is never read as
            # public here.
            continue
        try:
            access = normalize_access(declared)
        except ManifestAccessError:
            raise ManifestAccessError(
                f"{manifest_path} registers {spec.get('filename')!r} for "
                f"{key!r} with access={declared!r}, which is not one of "
                f"{list(ACCESS_CLASSES)}. Its bytes must not enter a Chronicle "
                "store until the registration is fixed."
            ) from None
        if is_hash_only(access):
            requested = (
                f" (requested as {filename!r})"
                if spec.get("filename") != filename
                else ""
            )
            raise ManifestAccessError(
                f"{manifest_path} registers {spec.get('filename')!r}{requested} "
                f"for {key!r} as access={access!r}. Its bytes must not enter a "
                "Chronicle store; keep the hash-only registration."
            )


def _upload_r2_object(
    location: ArtifactStorageLocation,
    local_path: Path,
    *,
    wrangler_command: str,
) -> ArtifactCommandResult:
    content_type, _ = mimetypes.guess_type(local_path.name)
    command = [
        *shlex.split(wrangler_command),
        "r2",
        "object",
        "put",
        f"{location.bucket}/{location.key}",
        "--file",
        str(local_path),
        "--remote",
    ]
    if content_type:
        command.extend(["--content-type", content_type])
    return _run_command(command)


def _legacy_raw_r2_key(
    *,
    source_id: str,
    package_id: str,
    year: Any,
    sha256: str,
    filename: str,
    resolved_prefix: str,
    package_path: Path,
) -> str | None:
    """Return the compatible pre-country raw key, when this route has one."""
    country = infer_r2_country(source_id=source_id, package_path=package_path)
    prefix, separator, suffix = resolved_prefix.rpartition("/")
    if country is None or not separator or suffix != country or not prefix:
        return None
    return posixpath.join(
        prefix,
        _clean_key_part(source_id),
        _clean_key_part(package_id),
        str(year),
        sha256,
        Path(filename).name,
    )


def _raw_r2_route_keys(
    *,
    source_id: str,
    package_id: str,
    year: Any,
    sha256: str,
    filename: str,
    resolved_prefix: str,
    package_path: Path,
) -> set[str]:
    """Return the exact current and pre-country keys for one declared route."""
    keys = {
        build_r2_key(
            source_id=source_id,
            package_id=package_id,
            year=year,
            sha256=sha256,
            filename=filename,
            prefix=resolved_prefix,
            package_path=package_path,
        )
    }
    legacy = _legacy_raw_r2_key(
        source_id=source_id,
        package_id=package_id,
        year=year,
        sha256=sha256,
        filename=filename,
        resolved_prefix=resolved_prefix,
        package_path=package_path,
    )
    if legacy is not None:
        keys.add(legacy)
    return keys


def _entry_records_one_of_raw_routes(
    spec: dict[str, Any],
    *,
    manifest_path: Path,
    year: Any,
    source_id: str,
    package_id: str,
    route_keys: set[str],
) -> bool:
    """Whether an entry witnesses one exact canonical or legacy route."""
    try:
        recorded = _validated_recorded_r2(
            spec,
            manifest_path=manifest_path,
            year=year,
            source_id=source_id,
            package_id=package_id,
        )
    except SourceArtifactManifestError:
        return False
    return recorded is not None and recorded.key in route_keys


def _compatible_recorded_raw_keys(
    *,
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    package_manifests: Mapping[str, dict[str, Any]] | None,
    source_id: str,
    package_id: str,
    year: Any,
    sha256: str,
    filename: str,
    resolved_prefix: str,
) -> set[str]:
    """Return exact raw routes established by current or package history.

    Besides the entry's current/pre-country route, two recorded legacy shapes
    are evidence-backed: another public sibling owns the same bytes at its own
    canonical route, or multiple nonnumeric table labels use the numeric
    release year anchored by this manifest and its package ID. Merely sharing
    a digest/name tail, or having one unrelated numeric anchor, is never enough.
    """
    allowed = _raw_r2_route_keys(
        source_id=source_id,
        package_id=package_id,
        year=year,
        sha256=sha256,
        filename=filename,
        resolved_prefix=resolved_prefix,
        package_path=manifest_path,
    )

    for owner_manifest_name, owner_manifest in (package_manifests or {}).items():
        owner_source_id = owner_manifest.get("source_id")
        owner_package_id = owner_manifest.get("package_id")
        if not isinstance(owner_source_id, str) or not owner_source_id.strip():
            continue
        if not isinstance(owner_package_id, str) or not owner_package_id.strip():
            continue
        owner_manifest_path = Path(owner_manifest_name)
        for owner_year, _index, owner_spec in iter_manifest_entries(owner_manifest):
            if not isinstance(owner_spec, dict):
                continue
            if is_hash_only(safe_entry_access(owner_spec)):
                continue
            if owner_spec.get("sha256") != sha256 or filename_key(
                owner_spec.get("filename")
            ) != filename_key(filename):
                continue
            try:
                owner_routes = _raw_r2_route_keys(
                    source_id=owner_source_id,
                    package_id=owner_package_id,
                    year=owner_year,
                    sha256=sha256,
                    filename=filename,
                    resolved_prefix=resolved_prefix,
                    package_path=owner_manifest_path,
                )
            except ValueError:
                continue
            if _entry_records_one_of_raw_routes(
                owner_spec,
                manifest_path=owner_manifest_path,
                year=owner_year,
                source_id=owner_source_id,
                package_id=owner_package_id,
                route_keys=owner_routes,
            ):
                allowed.update(owner_routes)

    year_text = str(year)
    release_match = re.search(r"(?:^|[-_])(\d{4})$", package_id)
    if manifest is None or year_text.isdecimal() or release_match is None:
        return allowed
    release_year = release_match.group(1)
    labelled_peer_witnessed = False
    for peer_year, _index, peer_spec in iter_manifest_entries(manifest):
        peer_year_text = str(peer_year)
        if (
            peer_year_text == year_text
            or peer_year_text.isdecimal()
            or not isinstance(peer_spec, dict)
            or is_hash_only(safe_entry_access(peer_spec))
        ):
            continue
        peer_sha256 = peer_spec.get("sha256")
        peer_filename = peer_spec.get("filename")
        if not isinstance(peer_sha256, str) or not isinstance(peer_filename, str):
            continue
        if (peer_sha256, filename_key(peer_filename)) == (
            sha256,
            filename_key(filename),
        ):
            continue
        try:
            peer_release_routes = _raw_r2_route_keys(
                source_id=source_id,
                package_id=package_id,
                year=release_year,
                sha256=peer_sha256,
                filename=peer_filename,
                resolved_prefix=resolved_prefix,
                package_path=manifest_path,
            )
        except ValueError:
            continue
        if _entry_records_one_of_raw_routes(
            peer_spec,
            manifest_path=manifest_path,
            year=peer_year,
            source_id=source_id,
            package_id=package_id,
            route_keys=peer_release_routes,
        ):
            labelled_peer_witnessed = True
            break
    if not labelled_peer_witnessed:
        return allowed

    for anchor_year, _index, anchor_spec in iter_manifest_entries(manifest):
        if str(anchor_year) != release_year or not isinstance(anchor_spec, dict):
            continue
        if is_hash_only(safe_entry_access(anchor_spec)):
            continue
        try:
            anchor_routes = _raw_r2_route_keys(
                source_id=source_id,
                package_id=package_id,
                year=anchor_year,
                sha256=str(anchor_spec.get("sha256") or ""),
                filename=str(anchor_spec.get("filename") or ""),
                resolved_prefix=resolved_prefix,
                package_path=manifest_path,
            )
        except ValueError:
            continue
        if not _entry_records_one_of_raw_routes(
            anchor_spec,
            manifest_path=manifest_path,
            year=anchor_year,
            source_id=source_id,
            package_id=package_id,
            route_keys=anchor_routes,
        ):
            continue
        allowed.update(
            _raw_r2_route_keys(
                source_id=source_id,
                package_id=package_id,
                year=release_year,
                sha256=sha256,
                filename=filename,
                resolved_prefix=resolved_prefix,
                package_path=manifest_path,
            )
        )
        break
    return allowed


def _publish_raw_manifest_entry(
    manifest_path: Path,
    source_id: Any,
    package_id: Any,
    year: Any,
    spec: Any,
    *,
    manifest: dict[str, Any] | None = None,
    kind: str | None = None,
    r2_bucket: str,
    r2_prefix: str | None,
    wrangler_command: str,
    skip_hash_only: bool = False,
    staging_dir: str | Path | None = None,
    package_manifests: Mapping[str, dict[str, Any]] | None = None,
    preflight_only: bool = False,
    manifest_identity: dict[str, Any] | None = None,
) -> tuple[RawArtifactPublishEntry, dict[str, Any] | None]:
    errors: list[str] = []
    reported_source_id = str(source_id) if source_id is not None else ""
    reported_package_id = str(package_id) if package_id is not None else ""
    artifact_path = manifest_path.parent
    filename = ""
    sha256_actual = None
    size_bytes = None

    def refuse(reason: str | None = None) -> tuple[RawArtifactPublishEntry, None]:
        if reason is not None:
            errors.append(reason)
        return RawArtifactPublishEntry(
            manifest_path=str(manifest_path),
            source_id=reported_source_id,
            package_id=reported_package_id,
            year=str(year),
            filename=filename,
            local_path=str(artifact_path),
            sha256=sha256_actual,
            size_bytes=size_bytes,
            r2_location=None,
            upload=None,
            errors=tuple(errors),
        ), None

    if isinstance(spec, ListSpecRejected):
        return refuse("list_file_spec_requires_microdata_release_kind")
    if not isinstance(spec, dict):
        spec = {}
        errors.append("malformed_file_spec")
    filename = str(spec.get("filename") or "")
    if filename and not is_bare_filename(filename):
        return refuse(f"non_canonical_filename:{filename}")
    if is_manifest_filename(filename):
        return refuse(f"manifest_named_filename:{filename}")
    kind = kind or safe_manifest_kind(manifest, manifest_path=manifest_path)[0]
    access = safe_entry_access(spec)
    release = kind == MICRODATA_RELEASE_KIND
    try:
        local_entry = matching_directory_entry(manifest_path.parent, filename)
    except ValueError:
        return refuse(f"duplicate_artifact_spellings:{filename}")
    if is_hash_only(access):
        hash_only_errors = list(
            validate_file_entry(
                spec,
                kind=kind,
                manifest=manifest,
                local_file_exists=local_entry is not None,
            )
        )
        if not skip_hash_only:
            hash_only_errors.insert(0, f"hash_only_access_refuses_bytes:{access}")
        return RawArtifactPublishEntry(
            manifest_path=str(manifest_path),
            source_id=reported_source_id,
            package_id=reported_package_id,
            year=str(year),
            filename=filename,
            local_path=str(local_entry or manifest_path.parent / filename),
            sha256=spec.get("sha256"),
            size_bytes=spec.get("size_bytes"),
            r2_location=None,
            upload=None,
            errors=tuple(dict.fromkeys(hash_only_errors)),
            skipped=f"{HASH_ONLY_SKIP_PREFIX}{access}",
        ), None
    if local_entry is not None and local_entry.name != filename:
        return refuse(f"artifact_spelling_mismatch:{filename}:{local_entry.name}")
    errors.extend(
        validate_file_entry(
            spec,
            kind=kind,
            manifest=manifest,
            local_file_exists=local_entry is not None,
        )
    )
    if errors:
        return refuse()
    try:
        recorded_object = _validated_recorded_r2(
            spec,
            manifest_path=manifest_path,
            year=year,
            source_id=source_id,
            package_id=package_id,
            bind_registration_identity=release,
        )
    except SourceArtifactManifestError as error:
        return refuse(f"recorded_r2_locator_invalid:{error}")
    # Recorded table objects retain their historical routes and declarations.
    # Validate uncoerced values before constructing any new publication key.
    if recorded_object is None:
        try:
            _require_identity_segment(source_id, what="source_id")
            _require_identity_segment(package_id, what="package_id")
            _require_identity_segment(str(year), what="year")
            _assert_manifest_identifies(
                manifest_identity or manifest or {},
                manifest_path,
                source_id=source_id,
                package_id=package_id,
            )
            resolved_r2_prefix = resolve_r2_prefix(
                prefix=r2_prefix,
                default_prefix=DEFAULT_R2_PREFIX,
                source_id=source_id,
                package_path=manifest_path,
            )
        except (SourceArtifactManifestError, ValueError) as error:
            return refuse(f"r2_identity_invalid:{error}")
    sha256_expected = spec.get("sha256")
    if recorded_object is not None and not release:
        # A standard raw route still states its publisher and country. Keep
        # numeric historical package/vintage routes and opaque archived paths,
        # while requiring evidence for remapping semantic table labels.
        segments = recorded_object.key.split("/")
        if segments[0] == "raw" and len(segments) in (6, 7):
            recorded_source, _recorded_package, recorded_year = segments[-5:-2]
            country = segments[1] if len(segments) == 7 else None
            declared_source = str(source_id) if source_id is not None else ""
            expected_country = infer_r2_country(
                source_id=declared_source, package_path=manifest_path
            )
            matching_publisher = (
                recorded_source == _clean_key_part(declared_source)
                if declared_source.strip()
                else True
            )
            # Some archived manifests used source_id for the table's package
            # identity and omitted package_id. Their publisher directory and
            # recorded package establish the older source/package split.
            if (
                not matching_publisher
                and package_id is None
                and _recorded_package == declared_source
            ):
                path_parts = manifest_path.parts
                for index in range(len(path_parts) - 3, -1, -1):
                    if path_parts[index : index + 2] == ("db", "data"):
                        matching_publisher = path_parts[index + 2] == recorded_source
                        break
                    if path_parts[index] == "packages":
                        matching_publisher = path_parts[index + 1] == recorded_source
                        break
            matching_country = country is None or country == expected_country
            compatible_route = matching_publisher and matching_country
            if compatible_route and not str(year).isdecimal():
                try:
                    historical_prefix = resolve_r2_prefix(
                        prefix=r2_prefix,
                        default_prefix=DEFAULT_R2_PREFIX,
                        source_id=declared_source,
                        package_path=manifest_path,
                    )
                    compatible_route = (
                        recorded_object.key
                        in _compatible_recorded_raw_keys(
                            manifest_path=manifest_path,
                            manifest=manifest,
                            package_manifests=package_manifests,
                            source_id=declared_source,
                            package_id=str(package_id or ""),
                            year=year,
                            sha256=recorded_object.sha256,
                            filename=filename,
                            resolved_prefix=historical_prefix,
                        )
                    )
                except ValueError:
                    compatible_route = False
            elif compatible_route:
                compatible_route = recorded_year.isdecimal()
            if not compatible_route:
                return refuse(
                    "recorded_r2_key_disagrees_with_country_prefix:"
                    f"recorded={recorded_object.key}"
                )
    if release and filename and sha256_expected:
        try:
            artifact_path = _validated_microdata_staging_destination(
                microdata_staging_path(
                    staging_dir=staging_dir,
                    source_id=source_id,
                    package_id=package_id,
                    year=year,
                    sha256=str(sha256_expected),
                    filename=filename,
                ),
                package_dir=manifest_path.parent,
            )
        except ManifestAccessError as error:
            return refuse(f"unsafe_microdata_staging:{error}")
    else:
        artifact_path = local_entry or manifest_path.parent / filename
    if not filename:
        return refuse("missing_filename")
    if artifact_path.is_symlink():
        return refuse(f"artifact_path_is_symlink:{filename}")
    if not artifact_path.is_file():
        return refuse("staged_bytes_missing" if release else "missing_file")
    if not release:
        # A table whose declared or recorded identity is a release sibling's is
        # that release under another record. Refuse before its bytes are read,
        # after every structural code the entry would otherwise report.
        declared_identity_errors = _microdata_identity_errors(
            package_manifests,
            package_dir=manifest_path.parent,
            filename=filename,
            digests=(
                sha256_expected,
                recorded_object.sha256 if recorded_object is not None else None,
            ),
        )
        if declared_identity_errors:
            return refuse(declared_identity_errors[0])
    content = artifact_path.read_bytes()
    sha256_actual = hashlib.sha256(content).hexdigest()
    size_bytes = len(content)
    if sha256_expected and sha256_actual != sha256_expected:
        errors.append("checksum_mismatch")
    if package_manifests is not None:
        try:
            _assert_no_hash_only_bytes(
                package_manifests,
                sha256=sha256_actual,
                filename=filename,
                what="the local artifact's bytes",
            )
        except ManifestAccessError:
            errors.append(f"sha256_collision_across_manifests:{sha256_actual}")
    if not release:
        # An undeclared table digest is only knowable from the bytes. Classify
        # it in memory, before the preflight return, the recorded-object skip,
        # the upload, and the manifest rewrite that records it.
        errors.extend(
            _microdata_identity_errors(
                package_manifests,
                package_dir=manifest_path.parent,
                filename=filename,
                digests=(sha256_actual,),
            )
        )
    if recorded_object is not None and (
        recorded_object.sha256 != sha256_actual or recorded_object.filename != filename
    ):
        errors.append(
            "recorded_r2_identity_mismatch:"
            f"recorded_sha256={recorded_object.sha256}:"
            f"recorded_filename={recorded_object.filename}:"
            f"local_sha256={sha256_actual}:local_filename={filename}"
        )
    if errors:
        return refuse()
    if recorded_object is not None:
        skipped = "recorded_r2_already_published"
        if recorded_object.bucket != r2_bucket:
            skipped = (
                "recorded_r2_bucket_is_preserved_history:"
                f"recorded={recorded_object.bucket}:requested={r2_bucket}"
            )
        return RawArtifactPublishEntry(
            manifest_path=str(manifest_path),
            source_id=reported_source_id,
            package_id=reported_package_id,
            year=str(year),
            filename=filename,
            local_path=str(artifact_path),
            sha256=sha256_actual,
            size_bytes=size_bytes,
            r2_location=ArtifactStorageLocation(
                provider="r2",
                bucket=recorded_object.bucket,
                key=recorded_object.key,
            ),
            upload=None,
            errors=(),
            skipped=skipped,
        ), None
    location = ArtifactStorageLocation(
        provider="r2",
        bucket=r2_bucket,
        key=build_r2_key(
            source_id=source_id,
            package_id=package_id,
            year=year,
            sha256=sha256_actual,
            filename=filename,
            prefix=resolved_r2_prefix,
            package_path=manifest_path,
        ),
    )
    if preflight_only:
        return RawArtifactPublishEntry(
            manifest_path=str(manifest_path),
            source_id=reported_source_id,
            package_id=reported_package_id,
            year=str(year),
            filename=filename,
            local_path=str(artifact_path),
            sha256=sha256_actual,
            size_bytes=size_bytes,
            r2_location=location,
            upload=None,
            errors=(),
        ), None
    upload = _upload_r2_object(
        location, artifact_path, wrangler_command=wrangler_command
    )
    updated_spec = None
    if upload.ok:
        updated_spec = {
            "sha256": sha256_actual,
            "size_bytes": size_bytes,
            "storage": {
                **(
                    spec.get("storage") if isinstance(spec.get("storage"), dict) else {}
                ),
                "r2": location.to_dict(),
            },
        }
    else:
        errors.append("r2_upload_failed")
    return RawArtifactPublishEntry(
        manifest_path=str(manifest_path),
        source_id=reported_source_id,
        package_id=reported_package_id,
        year=str(year),
        filename=filename,
        local_path=str(artifact_path),
        sha256=sha256_actual,
        size_bytes=size_bytes,
        r2_location=location if upload.ok else None,
        upload=upload,
        errors=tuple(errors),
    ), updated_spec


def _inventory_entry(
    manifest_path: Path,
    year: Any,
    spec: Any,
    *,
    manifest: dict[str, Any] | None = None,
    kind: str | None = None,
    staging_dir: str | Path | None = None,
    inspect_bytes: bool = True,
    package_manifests: Mapping[str, dict[str, Any]] | None = None,
) -> ArtifactInventoryEntry:
    errors: list[str] = []
    original_spec = spec
    if isinstance(spec, ListSpecRejected):
        return ArtifactInventoryEntry(
            manifest_path=str(manifest_path),
            year=str(year),
            filename="",
            local_path=str(manifest_path.parent),
            exists=False,
            sha256_expected=None,
            sha256_actual=None,
            size_bytes=None,
            source_url=None,
            r2=None,
            errors=("list_file_spec_requires_microdata_release_kind",),
        )
    if not isinstance(spec, dict):
        spec = {}
        errors.append("malformed_file_spec")
    filename = str(spec.get("filename") or "")
    kind = kind or safe_manifest_kind(manifest, manifest_path=manifest_path)[0]
    bare = bool(filename) and is_bare_filename(filename)
    local_entry = None
    if bare:
        try:
            local_entry = matching_directory_entry(manifest_path.parent, filename)
            if local_entry is not None and local_entry.name != filename:
                errors.append(
                    f"artifact_spelling_mismatch:{filename}:{local_entry.name}"
                )
        except ValueError:
            errors.append(f"duplicate_artifact_spellings:{filename}")
    in_tree = local_entry is not None
    access = safe_entry_access(spec)
    hash_only = is_hash_only(access)
    release = kind == MICRODATA_RELEASE_KIND
    errors.extend(
        validate_file_entry(
            original_spec,
            kind=kind,
            manifest=manifest,
            local_file_exists=in_tree,
        )
    )
    if is_manifest_filename(filename):
        errors.append(f"manifest_named_filename:{filename}")
    sha256_expected = spec.get("sha256")
    validated_r2 = None
    try:
        validated_r2 = _validated_recorded_r2(
            spec,
            manifest_path=manifest_path,
            year=year,
            source_id=(manifest or {}).get("source_id"),
            package_id=(manifest or {}).get("package_id"),
            bind_registration_identity=release,
        )
    except SourceArtifactManifestError as error:
        errors.append(f"recorded_r2_locator_invalid:{error}")
    if validated_r2 is not None and (
        validated_r2.filename != filename
        or (sha256_expected is not None and validated_r2.sha256 != sha256_expected)
    ):
        errors.append(
            "recorded_r2_identity_mismatch:"
            f"recorded_sha256={validated_r2.sha256}:"
            f"recorded_filename={validated_r2.filename}:"
            f"declared_sha256={sha256_expected}:declared_filename={filename}"
            if release
            else "recorded_r2_identity_mismatch"
        )
        validated_r2 = None
    if release and not hash_only and validated_r2 is None:
        errors.append("r2_object_not_recorded")
    if release and not hash_only and bare and sha256_expected:
        artifact_path = microdata_staging_path(
            staging_dir=staging_dir,
            source_id=str((manifest or {}).get("source_id") or ""),
            package_id=str((manifest or {}).get("package_id") or ""),
            year=year,
            sha256=str(sha256_expected),
            filename=filename,
        )
    else:
        artifact_path = local_entry or (
            manifest_path.parent / filename if bare else manifest_path.parent
        )
    symlink = bare and artifact_path.is_symlink()
    exists = bool(bare and not symlink and not errors and artifact_path.is_file())
    sha256_actual = None
    size_bytes = spec.get("size_bytes") if hash_only or release else None
    if not filename:
        errors.append("missing_filename")
    elif not bare or hash_only:
        pass
    elif symlink:
        errors.append(f"artifact_path_is_symlink:{filename}")
    elif errors:
        pass
    elif release:
        if validated_r2 is None:
            errors.append("r2_object_not_recorded")
        if exists and inspect_bytes and not errors:
            content = artifact_path.read_bytes()
            sha256_actual = hashlib.sha256(content).hexdigest()
            size_bytes = len(content)
            if sha256_expected and sha256_actual != sha256_expected:
                errors.append("checksum_mismatch")
    elif not exists:
        errors.append("missing_file")
    elif inspect_bytes:
        content = artifact_path.read_bytes()
        sha256_actual = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        if sha256_expected and sha256_actual != sha256_expected:
            errors.append("checksum_mismatch")
    if not release and not hash_only:
        # Report the release sibling's identity beside every other code the
        # entry earns. A metadata alias is already a package error, which left
        # these bytes uninspected; only an undeclared digest needs them.
        errors.extend(
            _microdata_identity_errors(
                package_manifests,
                package_dir=manifest_path.parent,
                filename=filename,
                digests=(
                    sha256_expected,
                    validated_r2.sha256 if validated_r2 is not None else None,
                    sha256_actual,
                ),
            )
        )
    r2 = None
    if validated_r2 is not None:
        if sha256_actual is not None and sha256_actual != validated_r2.sha256:
            errors.append("recorded_r2_identity_mismatch")
        if not errors:
            r2 = {
                "provider": validated_r2.provider,
                "bucket": validated_r2.bucket,
                "key": validated_r2.key,
                "uri": validated_r2.uri,
            }
    return ArtifactInventoryEntry(
        manifest_path=str(manifest_path),
        year=str(year),
        filename=filename,
        local_path=str(artifact_path),
        exists=exists,
        sha256_expected=sha256_expected,
        sha256_actual=sha256_actual,
        size_bytes=size_bytes,
        source_url=spec.get("source_url"),
        r2=r2,
        errors=tuple(dict.fromkeys(errors)),
        access=access,
        licence=spec.get("licence"),
        hash_only=hash_only,
    )


def _derived_artifact_kind(artifact_name: str) -> str:
    if artifact_name in CHRONICLE_DB_FILENAMES:
        return "sqlite_database"
    if artifact_name.endswith(".jsonl"):
        return "jsonl"
    if artifact_name.startswith("reports/"):
        return "report"
    if artifact_name.endswith(".json"):
        return "json"
    return "artifact"


def _run_command(command: list[str]) -> ArtifactCommandResult:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return ArtifactCommandResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _clean_key_part(value: str) -> str:
    if not isinstance(value, str):
        # Callers bind keys to manifest-supplied identities, which a manifest
        # may simply omit. Every one of them turns a ValueError into its own
        # refusal; an AttributeError on None would escape all of them.
        raise ValueError(f"R2 key parts must be strings, not {type(value).__name__}.")
    cleaned = value.strip().strip("/")
    if not cleaned:
        raise ValueError("R2 key parts cannot be empty.")
    return cleaned.replace(" ", "_")


def _clean_relative_key_parts(value: str) -> tuple[str, ...]:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("R2 artifact paths must be relative.")
    parts = tuple(_clean_key_part(part) for part in path.parts if part != ".")
    if not parts or any(part == ".." for part in parts):
        raise ValueError("R2 artifact paths cannot be empty or contain '..'.")
    return parts


def _require_regular_manifest_file(path: Path) -> None:
    """Refuse a manifest-named entry that is not a regular, non-symlink file.

    ``is_file`` follows symlinks, so a dangling symlink, a symlink to a
    directory, or any other non-regular entry would silently vanish from a
    sweep and from sibling-registry checks; a registry entry that cannot be
    read as a manifest is a defect to surface, never to skip.
    """
    if path.is_symlink() or not path.is_file():
        raise MalformedManifestError(
            f"{path} carries a manifest name but is not a regular file; "
            "Chronicle will not sweep past it or register beside it."
        )


def default_r2_derived_prefix() -> str:
    """Resolve the derived route shared by publication and fact refusals."""
    return env_value("CHRONICLE_R2_DERIVED_PREFIX", default=DEFAULT_R2_DERIVED_PREFIX)


def is_derived_r2_route(bucket: str, key: str) -> bool:
    """Whether an R2 bucket/key pair addresses derived build output.

    Resolve publication configuration at validation time: an operator may use
    a bucket or prefix with no ``derived`` marker in its spelling. Archived
    rename-window routes remain derived after the active destination changes.
    """
    derived_buckets = {
        "ledger-derived",
        "chronicle-derived",
        DEFAULT_R2_DERIVED_BUCKET,
        default_r2_derived_bucket(),
    }
    derived_prefixes = {
        "derived",
        resolve_r2_prefix(
            prefix=None,
            default_prefix=DEFAULT_R2_DERIVED_PREFIX,
        ),
        resolve_r2_prefix(
            prefix=None,
            default_prefix=default_r2_derived_prefix(),
        ),
    }
    # Configured routes extend the boundary; they never narrow it. A bucket
    # ending in ``-derived`` (any case) or a ``derived/`` key was derived
    # before routes became configurable, and archived facts still cite such
    # routes, so the legacy spelling rule stays alongside the configured set.
    return (
        bucket in derived_buckets
        or bucket.casefold().endswith("-derived")
        or any(
            key == prefix or key.startswith(f"{prefix}/") for prefix in derived_prefixes
        )
    )


class IdentitySegmentError(SourceArtifactManifestError, ValueError):
    """A registration identity (source_id / package_id) is not one canonical
    R2 key segment."""


@dataclass(frozen=True)
class _ManifestFileOwner:
    """One manifest entry that names a package-local artifact."""

    manifest_path: Path
    vintage: Any
    spec: dict[str, Any]
    identity: RecordedIdentity | None


def _derived_artifact_paths(input_path: Path) -> list[Path]:
    """Preflight every build-tree entry before reading or publishing any file."""
    if not stat.S_ISDIR(input_path.lstat().st_mode):
        raise ValueError(f"derived_root_not_regular_directory:{input_path}")
    artifacts: list[Path] = []

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                artifacts.append(path)
            else:
                raise ValueError(f"derived_entry_not_regular_file:{path}")

    visit(input_path)
    return sorted(artifacts)


def _manifest_file_owners(
    manifests: Mapping[str, dict[str, Any]],
    *,
    filename: str,
    initializing: tuple[Path, Any] | None = None,
) -> list[_ManifestFileOwner]:
    """Return every entry in a package directory that names ``filename``.

    ``initializing`` names the ``(manifest_path, vintage)`` entry the calling
    command is about to identify -- a predeclared entry (``filename`` and
    ``source_url`` only) on its first fetch. That entry has no identity yet
    and is not an owner; every other entry naming the file must already
    record one, because Chronicle cannot tell whether overwriting the shared
    bytes would change what an unidentified sibling means.
    """
    wanted = filename_key(filename)
    owners: list[_ManifestFileOwner] = []
    for name, payload in manifests.items():
        manifest_path = Path(name)
        if (
            safe_manifest_kind(payload, manifest_path=manifest_path)[0]
            == MICRODATA_RELEASE_KIND
        ):
            # Release vintages use distinct content-addressed staging paths.
            # Only publisher tables share a package-local artifact to rewrite.
            continue
        for vintage, _index, spec in iter_manifest_entries(payload):
            if not isinstance(spec, dict):
                raise MalformedManifestError(
                    f"{manifest_path} entry {vintage!r} must be a mapping; it "
                    f"is a {type(spec).__name__}. Chronicle cannot decide "
                    "whether it owns a shared package-local file."
                )
            recorded_name = spec.get("filename")
            if recorded_name is None:
                continue
            if not is_bare_filename(recorded_name):
                raise MalformedManifestError(
                    f"{manifest_path} entry {vintage!r} filename must be a "
                    f"bare package-local name, not {recorded_name!r}."
                )
            if filename_key(recorded_name) != wanted:
                continue
            identity = _recorded_identity(
                spec,
                manifest_path=manifest_path,
                year=vintage,
            )
            if identity is None:
                if (
                    initializing is not None
                    and manifest_path == initializing[0]
                    and str(vintage) == str(initializing[1])
                ):
                    # The entry this command identifies: not an owner yet.
                    continue
                raise MalformedManifestError(
                    f"{manifest_path} entry {vintage!r} names "
                    f"{recorded_name!r} but records no sha256 identity. "
                    "Chronicle cannot safely overwrite an unidentifiable "
                    "shared file."
                )
            owners.append(
                _ManifestFileOwner(
                    manifest_path=manifest_path,
                    vintage=vintage,
                    spec=spec,
                    identity=identity,
                )
            )
    return owners


def _assert_shared_owner_identities_agree(
    owners: list[_ManifestFileOwner],
    *,
    filename: str,
) -> None:
    """Refuse an already-contradictory set of owners before publisher I/O."""
    if not owners:
        return
    first = owners[0]
    first_identity = first.identity
    assert first_identity is not None
    for owner in owners[1:]:
        identity = owner.identity
        assert identity is not None
        if identity.sha256 == first_identity.sha256 and filename_key(
            identity.filename
        ) == filename_key(first_identity.filename):
            continue
        raise SourceArtifactManifestError(
            f"{first.manifest_path} entry {first.vintage!r} and "
            f"{owner.manifest_path} entry {owner.vintage!r} both name "
            f"{filename!r} but identify different bytes. One package-local "
            "file must have one recorded identity; reconcile the manifests "
            "before fetching it again."
        )


def _effective_recorded_digest(
    manifest_name: str, vintage: Any, entry: Mapping[str, Any]
) -> str | None:
    """Return the digest an entry's recorded R2 key encodes, if it has one.

    Two manifests may identify one package-local file through identical
    content-addressed R2 locators without declaring ``sha256``; refetching
    those bytes records ``sha256`` on the selected manifest only. The
    directory-level collision check must compare the identities the entries
    *effectively* record -- the same identity :func:`_recorded_identity`
    resolves -- so a sibling that only carries the locator does not read as
    an empty digest. A malformed locator is the per-entry preflight's error,
    not a collision: return None and let directory validation use the declared
    field. Only the current object is read here -- unreadable *archived*
    provenance is its own refusal, and letting it erase the digest this entry
    plainly records would drop the entry out of the directory comparison and
    stop naming a contradiction between two manifests.
    """
    try:
        recorded = _validated_recorded_r2(
            {"storage": {"r2": _recorded_r2(entry)}},
            manifest_path=Path(manifest_name),
            year=vintage,
        )
    except SourceArtifactManifestError:
        return None
    return None if recorded is None else recorded.sha256


def _assert_package_file_owner_identities_agree(
    manifests: Mapping[str, dict[str, Any]],
    *,
    observed_sha256: Mapping[str, str] | None = None,
    check_local_files: bool = True,
) -> None:
    """Refuse contradictory identities for any package-local filename.

    Publish and inventory sweep a manifest at a time, but the physical byte is
    shared by every manifest in its directory. Validate every identified owner
    as one package boundary before a selected manifest can upload anything.
    Entry-shape and local-file errors remain the per-entry preflight's job.

    Resource readers supply observed digests by filename and disable local
    filesystem reads: their bytes may come from ZIP resources, cache, or a
    publisher response, and must agree before they are returned or cached.
    """
    collision_codes = validate_package_directory(
        manifests, entry_digest=_effective_recorded_digest
    )
    if collision_codes:
        raise SourceArtifactManifestError(
            "Package manifests identify different bytes for one package-local "
            f"filename: {', '.join(collision_codes)}. Reconcile the manifests "
            "before publishing or inventorying that directory."
        )

    owners_by_filename: dict[str, list[_ManifestFileOwner]] = {}
    display_names: dict[str, str] = {}
    unidentified: dict[str, list[tuple[Path, Any, str]]] = {}
    for name, payload in manifests.items():
        manifest_path = Path(name)
        if (
            safe_manifest_kind(payload, manifest_path=manifest_path)[0]
            == MICRODATA_RELEASE_KIND
        ):
            # Release vintages use distinct content-addressed staging paths.
            # Only publisher tables share a package-local artifact to rewrite.
            continue
        for vintage, _index, spec in iter_manifest_entries(payload):
            if not isinstance(spec, dict):
                continue
            recorded_name = spec.get("filename")
            if not is_bare_filename(recorded_name):
                continue
            try:
                identity = _recorded_identity(
                    spec,
                    manifest_path=manifest_path,
                    year=vintage,
                )
            except SourceArtifactManifestError:
                # The complete per-entry preflight reports the precise locator
                # or history error without letting another entry upload first.
                continue
            key = filename_key(recorded_name)
            display_names.setdefault(key, str(recorded_name))
            if identity is None:
                unidentified.setdefault(key, []).append(
                    (manifest_path, vintage, str(recorded_name))
                )
                continue
            owners_by_filename.setdefault(key, []).append(
                _ManifestFileOwner(
                    manifest_path=manifest_path,
                    vintage=vintage,
                    spec=spec,
                    identity=identity,
                )
            )
    for key, owners in owners_by_filename.items():
        _assert_shared_owner_identities_agree(
            owners,
            filename=display_names[key],
        )
    # An entry that records no identity yet is not a collision (nothing to
    # contradict), but the command that identifies it will hash the shared
    # bytes. Those bytes must already be what the identified owners record,
    # otherwise identifying it would split one package-local file into two
    # identities. Check before any upload or manifest rewrite.
    observed_by_filename = {
        filename_key(name): digest for name, digest in (observed_sha256 or {}).items()
    }
    for key, pending in unidentified.items():
        owners = owners_by_filename.get(key)
        if not owners:
            continue
        expected = owners[0].identity
        assert expected is not None
        for manifest_path, vintage, recorded_name in pending:
            actual = observed_by_filename.get(key)
            if actual is None:
                if not check_local_files:
                    continue
                try:
                    local = matching_directory_entry(
                        manifest_path.parent, recorded_name
                    )
                except ValueError:
                    # Conflicting spellings are the per-entry preflight's refusal.
                    continue
                if local is None or local.is_symlink() or not local.is_file():
                    continue
                actual = hashlib.sha256(local.read_bytes()).hexdigest()
            if actual == expected.sha256:
                continue
            raise SourceArtifactManifestError(
                f"{manifest_path} entry {vintage!r} names {recorded_name!r} "
                f"without a recorded identity, and the package-local bytes "
                f"(sha256={actual}) are not what {owners[0].manifest_path} entry "
                f"{owners[0].vintage!r} records for it (sha256="
                f"{expected.sha256}). Identifying this entry would give one "
                "package-local file two identities; reconcile the manifests "
                "or the file before publishing or inventorying the directory."
            )


def _storage_for_fetched_identity(
    recorded_spec: dict[str, Any],
    *,
    identity: RecordedIdentity | None,
    filename: str,
    sha256: str,
    new_r2: dict[str, Any] | None,
    fetched_at: str,
) -> dict[str, Any]:
    """Return one owner's storage after a refetch or explicit revision."""
    recorded_storage = _recorded_storage(recorded_spec)
    holds = identity is not None and identity.holds(
        sha256=sha256,
        filename=filename,
    )
    if holds and identity.r2 is not None:
        # A same-byte copy does not replace the object's recorded history.
        return {**recorded_storage, "r2": _recorded_r2(recorded_spec)}
    if identity is not None and not holds:
        return _superseding_storage(
            recorded_spec,
            recorded_r2=identity.r2,
            new_r2=new_r2,
            superseded_at=fetched_at,
        )
    if new_r2 is not None:
        return {**recorded_storage, "r2": new_r2}
    return dict(recorded_storage)


def _require_identity_segment(value: Any, *, what: str) -> str:
    """Require a registration identity to be one canonical key segment.

    ``_clean_key_part`` normalizes what it is given (strips, folds spaces to
    underscores) because it also renders legacy recorded values; a NEW
    registration identity must already be canonical, or two spellings such as
    ``foo bar`` and ``foo_bar`` would collide in one R2 namespace and a
    separator would shift the key's path shape.
    """
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or value != value.strip()
        or any(character.isspace() for character in value)
        or "/" in value
        or "\\" in value
        or _clean_key_part(value) != value
    ):
        raise IdentitySegmentError(
            f"{what} must be one canonical R2 key segment (no whitespace, "
            f"slashes, or '..'), not {value!r}; R2 key parts cannot be empty "
            "or rewritten."
        )
    return value


class ArtifactFilenameError(
    SourceArtifactManifestError, RegistrationArtifactFilenameError
):
    """An unsafe artifact destination, reported by both command contracts."""


def bare_filename(value: Any, *, what: str = "filename") -> str:
    """Validate an artifact name using the shared registration contract."""
    try:
        return registration_bare_filename(value, what=what)
    except RegistrationArtifactFilenameError as error:
        raise ArtifactFilenameError(str(error)) from error


def _publish_source_artifacts_unlocked(
    root: str | Path,
    *,
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
    source_id: str | None = None,
    package_id: str | None = None,
    r2_bucket: str | None = None,
    r2_prefix: str | None = None,
    wrangler_command: str = DEFAULT_WRANGLER_COMMAND,
    skip_hash_only: bool = False,
    staging_dir: str | Path | None = None,
    _selected_manifest_path: Path | None = None,
    _preflight_only: bool = False,
) -> RawArtifactPublishReport:
    """Validate the full operation in memory before any publication mutation."""
    r2_bucket = r2_bucket or default_r2_raw_bucket()
    root_path = Path(root)
    _manifest_path(Path(), manifest_filename)
    if not root_path.exists():
        return RawArtifactPublishReport(
            root=str(root_path),
            entries=(),
            errors=(f"Root does not exist: {root_path}",),
        )
    entries: list[RawArtifactPublishEntry] = []
    errors: list[str] = []
    prepared: list[tuple[Any, ...]] = []
    manifest_paths = (
        [_selected_manifest_path]
        if _selected_manifest_path is not None
        else _root_manifest_paths(root_path, manifest_filename)
    )
    checked_entries: set[tuple[str, str, int]] = set()
    for manifest_path in manifest_paths:
        try:
            manifest = _read_manifest(manifest_path)
            files = _manifest_files(manifest, manifest_path)
            package_manifests = _package_manifests(
                manifest_path.parent, manifest_path, manifest
            )
        except (OSError, SourceArtifactManifestError) as exc:
            errors.append(f"Could not read {manifest_path}: {exc}")
            continue
        manifest_source_id = (
            source_id if source_id is not None else manifest.get("source_id")
        )
        manifest_package_id = (
            package_id if package_id is not None else manifest.get("package_id")
        )
        kind, _kind_error = safe_manifest_kind(manifest, manifest_path=manifest_path)
        structural_errors: list[str] = []
        entry_errors: list[str] = []
        selected_entry_errors: list[str] = []
        for package_name, package_manifest in package_manifests.items():
            package_path = Path(package_name)
            package_kind, package_kind_error = safe_manifest_kind(
                package_manifest, manifest_path=package_path
            )
            if package_kind_error:
                structural_errors.append(f"{package_kind_error}: {package_path}")
            structural_errors.extend(
                f"{code}: {package_path}"
                for code in validate_manifest_files(package_manifest)
            )
            package_entry_errors = _manifest_entry_validation_errors(
                package_manifest, kind=package_kind, package_dir=package_path.parent
            )
            entry_errors.extend(
                f"{code}: {package_path}" for code in package_entry_errors
            )
            if package_path == manifest_path:
                selected_entry_errors = package_entry_errors
        structural_errors.extend(
            f"{code}: {manifest_path}"
            for code in validate_package_directory(
                package_manifests, entry_digest=_effective_recorded_digest
            )
        )
        try:
            _assert_package_file_owner_identities_agree(package_manifests)
        except SourceArtifactManifestError as exc:
            structural_errors.append(str(exc))
        try:
            _assert_no_package_microdata_identities(package_manifests)
        except SourceArtifactManifestError as exc:
            structural_errors.append(str(exc))
        if structural_errors or entry_errors:
            errors.extend((*structural_errors, *entry_errors))
            if selected_entry_errors and (
                not structural_errors
                or all(
                    code.startswith("non_canonical_filename:")
                    for code in structural_errors
                )
            ):
                for year, spec in files.items():
                    for file_spec in iter_file_specs(spec, kind=kind):
                        name = (
                            file_spec.get("filename")
                            if isinstance(file_spec, dict)
                            else None
                        )
                        alias_error = False
                        try:
                            local_exists = (
                                matching_directory_entry(manifest_path.parent, name)
                                is not None
                            )
                        except ValueError:
                            local_exists = True
                            alias_error = True
                        if not alias_error and not validate_file_entry(
                            file_spec,
                            kind=kind,
                            manifest=manifest,
                            local_file_exists=local_exists,
                        ):
                            continue
                        entry, _ = _publish_raw_manifest_entry(
                            manifest_path,
                            manifest_source_id,
                            manifest_package_id,
                            year,
                            file_spec,
                            manifest=manifest,
                            kind=kind,
                            r2_bucket=r2_bucket,
                            r2_prefix=r2_prefix,
                            wrangler_command=wrangler_command,
                            skip_hash_only=skip_hash_only,
                            staging_dir=staging_dir,
                            package_manifests=package_manifests,
                            preflight_only=True,
                        )
                        entries.append(entry)
            continue
        # Explicit selectors still validate every public owner in the package.
        # Unselected gated siblings require valid metadata, never byte access.
        for package_name, package_manifest in package_manifests.items():
            package_path = Path(package_name)
            package_kind, _ = safe_manifest_kind(
                package_manifest, manifest_path=package_path
            )
            # Preflight every selected manifest with the identity publication
            # will use, even if it is first encountered as another's sibling.
            # Unselected siblings retain their own identifiers.
            package_source_id = (
                source_id
                if source_id is not None and package_path in manifest_paths
                else package_manifest.get("source_id")
            )
            package_id_value = (
                package_id
                if package_id is not None and package_path in manifest_paths
                else package_manifest.get("package_id")
            )
            for year, spec in _manifest_files(package_manifest, package_path).items():
                for index, file_spec in enumerate(
                    iter_file_specs(spec, kind=package_kind)
                ):
                    check_key = (str(package_path), str(year), index)
                    is_selected = package_path in manifest_paths
                    if check_key in checked_entries:
                        continue
                    checked_entries.add(check_key)
                    entry, _ = _publish_raw_manifest_entry(
                        package_path,
                        package_source_id,
                        package_id_value,
                        year,
                        file_spec,
                        manifest=package_manifest,
                        kind=package_kind,
                        r2_bucket=r2_bucket,
                        r2_prefix=r2_prefix,
                        wrangler_command=wrangler_command,
                        skip_hash_only=skip_hash_only or not is_selected,
                        staging_dir=staging_dir,
                        package_manifests=package_manifests,
                        preflight_only=True,
                    )
                    if entry.errors or (is_selected and _preflight_only):
                        entries.append(entry)
        prepared.append(
            (
                manifest_path,
                manifest,
                files,
                kind,
                manifest_source_id,
                manifest_package_id,
                package_manifests,
            )
        )
    if errors or any(entry.errors for entry in entries) or _preflight_only:
        return RawArtifactPublishReport(
            root=str(root_path), entries=tuple(entries), errors=tuple(errors)
        )
    for (
        manifest_path,
        manifest,
        files,
        kind,
        manifest_source_id,
        manifest_package_id,
        package_manifests,
    ) in prepared:
        updated = False
        for year, spec in files.items():
            for file_spec in iter_file_specs(spec, kind=kind):
                entry, updated_spec = _publish_raw_manifest_entry(
                    manifest_path,
                    manifest_source_id,
                    manifest_package_id,
                    year,
                    file_spec,
                    manifest=manifest,
                    kind=kind,
                    r2_bucket=r2_bucket,
                    r2_prefix=r2_prefix,
                    wrangler_command=wrangler_command,
                    skip_hash_only=skip_hash_only,
                    staging_dir=staging_dir,
                    package_manifests=package_manifests,
                )
                entries.append(entry)
                if updated_spec is not None and isinstance(file_spec, dict):
                    file_spec.update(updated_spec)
                    updated = True
        if updated:
            if manifest_source_id:
                manifest.setdefault("source_id", manifest_source_id)
            if manifest_package_id:
                manifest.setdefault("package_id", manifest_package_id)
            manifest_path.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )
    return RawArtifactPublishReport(
        root=str(root_path), entries=tuple(entries), errors=tuple(errors)
    )
