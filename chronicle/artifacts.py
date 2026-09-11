"""Source artifact acquisition and storage helpers for Chronicle."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import posixpath
import re
import shlex
import sqlite3
import stat
import subprocess
from typing import Any, Mapping
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
from chronicle.registration import (
    filename_key,
    is_bare_filename,
    is_manifest_filename,
    load_manifest_document,
    matching_directory_entry,
    package_manifest_paths,
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

# Most packages keep one manifest.yaml. Publisher directories that feed several
# source packages keep one manifest each -- db/data/irs_soi/ira_contributions
# holds manifest_traditional_source_package.yaml beside the Roth one -- so the
# name is an input, not a constant, wherever a caller addresses a package.
DEFAULT_MANIFEST_FILENAME = "manifest.yaml"


def bare_filename(value: Any, *, what: str = "filename") -> str:
    """Return ``value`` as a bare filename, refusing any other spelling.

    ``./table.csv``, ``nested/table.csv`` and an absolute path all resolve
    outside the one-name manifest contract once joined under a package, so
    fetch validates the spelling before reading publisher bytes.
    """
    if not is_bare_filename(value):
        raise ArtifactFilenameError(
            f"{what} must be a bare filename inside the package directory, not "
            f"{value!r}; it may not carry a directory, '.', '..', a trailing "
            "slash, surrounding whitespace, or an absolute path."
        )
    return str(value)


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


def default_r2_raw_bucket() -> str:
    """Resolve the raw bucket: ``$CHRONICLE_R2_RAW_BUCKET`` or the default."""
    return env_value(R2_RAW_BUCKET_ENV, default=DEFAULT_R2_RAW_BUCKET)


def default_r2_derived_bucket() -> str:
    """Resolve the derived bucket: ``$CHRONICLE_R2_DERIVED_BUCKET`` or default."""
    return env_value(R2_DERIVED_BUCKET_ENV, default=DEFAULT_R2_DERIVED_BUCKET)


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


class SourceArtifactManifestError(RuntimeError):
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


class ArtifactFilenameError(SourceArtifactManifestError, ValueError):
    """An artifact filename is not a bare, non-manifest package filename."""


class IdentitySegmentError(SourceArtifactManifestError, ValueError):
    """A registration identity (source_id / package_id) is not one canonical
    R2 key segment."""


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
class _ManifestFileOwner:
    """One manifest entry that names a package-local artifact."""

    manifest_path: Path
    vintage: Any
    spec: dict[str, Any]
    identity: RecordedIdentity | None


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

    @property
    def valid(self) -> bool:
        """Whether this artifact is locally available and checksum-valid."""
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
    def valid(self) -> bool:
        """Whether this raw artifact is published: uploaded now, or already
        held by the recorded object in a preserved bucket (``skipped``)."""
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
    upload_r2: bool = False,
    record_revision: bool = False,
    r2_bucket: str | None = None,
    r2_prefix: str | None = None,
    wrangler_command: str = "npx wrangler",
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
    """
    r2_bucket = r2_bucket or default_r2_raw_bucket()
    output = Path(output_dir)
    manifest_path = _manifest_path(output, manifest_filename)
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
    # These fields become object-key path segments even when this fetch does
    # not upload. Validate them before reading the publisher so a malformed
    # registration identity cannot overwrite package-local bytes and fail only
    # when the prospective R2 key is constructed below.
    _require_identity_segment(source_id, what="source_id")
    _require_identity_segment(package_id, what="package_id")
    _require_identity_segment(str(year), what="year")
    resolved_r2_prefix = resolve_r2_prefix(
        prefix=r2_prefix,
        default_prefix=DEFAULT_R2_PREFIX,
        source_id=source_id,
        package_path=output,
    )
    # Read and validate the entry being written before anything is fetched: a
    # manifest Chronicle cannot read, or a recorded block that names two
    # different objects, is a refusal that need not touch the publisher.
    try:
        _refuse_a_stray_default_manifest(output, manifest_path)
    except SourceArtifactManifestError:
        raise
    except ValueError as error:
        # package_manifest_paths refuses non-regular manifest-named entries
        # with a plain ValueError; surface it as the manifest error the CLI
        # reports rather than a traceback.
        raise MalformedManifestError(str(error)) from error
    existing_manifest = _read_manifest(manifest_path)
    _manifest_files(existing_manifest, manifest_path)
    _assert_manifest_identifies(
        existing_manifest,
        manifest_path,
        source_id=source_id,
        package_id=package_id,
    )
    vintage_key, _existing_value, selected_spec, _index = _select_vintage_entry(
        existing_manifest,
        manifest_path=manifest_path,
        year=year,
    )
    recorded_identity = _recorded_identity(
        selected_spec,
        manifest_path=manifest_path,
        year=vintage_key,
    )
    manifests = _package_manifests(output, manifest_path, existing_manifest)
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
                f"{existing_target} is a symbolic link. Chronicle will not "
                "fetch through a package-local link or overwrite its target."
            )
        if existing_target.name != artifact_filename:
            raise ArtifactFilenameError(
                f"{existing_target} has the same normalized filename as "
                f"{artifact_filename!r}. Chronicle will not create a "
                "physically distinct alias; pass --filename "
                f"{existing_target.name!r}."
            )
        if not existing_target.is_file():
            raise ArtifactFilenameError(
                f"{existing_target} exists but is not a regular file. "
                "Chronicle will not overwrite it with publisher bytes."
            )

    fetched_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    content, _inferred_filename = _read_artifact(source_url)

    sha256 = hashlib.sha256(content).hexdigest()
    size_bytes = len(content)

    # Guard before the cached artifact is touched. A rejected fetch must leave
    # the recorded bytes and their manifest entry exactly as they were.
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
    if not record_revision:
        _assert_siblings_record_these_bytes(
            manifests,
            manifest_path=manifest_path,
            vintage=vintage_key,
            filename=artifact_filename,
            sha256=sha256,
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

    output.mkdir(parents=True, exist_ok=True)
    local_path = output / artifact_filename
    local_path.write_bytes(content)

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
        source_id=source_id,
        package_id=package_id,
        dataset=dataset or f"{source_id}_{package_id}",
        source_page=source_page or source_url,
        table=table or package_id,
        year=year,
        filename=artifact_filename,
        source_url=source_url,
        sha256=sha256,
        size_bytes=size_bytes,
        fetched_at=fetched_at,
        r2_location=(r2_location if upload_r2 and r2_upload and r2_upload.ok else None),
        record_revision=record_revision,
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
        r2_location=r2_location if upload_r2 and r2_upload and r2_upload.ok else None,
        r2_upload=r2_upload,
        errors=tuple(errors),
    )


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
    wrangler_command: str = "npx wrangler",
) -> RawArtifactPublishReport:
    """Upload manifest-declared raw source artifacts and record R2 locations."""
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
    prepared: list[tuple[Path, dict[str, Any], dict[str, Any], str, str]] = []
    preflight_failures: list[RawArtifactPublishEntry] = []
    # ``--source-id`` / ``--package-id`` complete or confirm the identity of
    # every manifest this sweep selects (one with an explicit selector, all of
    # a directory's manifests by default). Decide eligibility from the whole
    # selection up front, so a selected sibling met through another selected
    # manifest's package preflight is not mistaken for an unselected one.
    selected_manifest_paths = list(_root_manifest_paths(root_path, manifest_filename))
    selected_manifests = set(selected_manifest_paths)
    for manifest_path in selected_manifest_paths:
        try:
            manifest = _read_manifest(manifest_path)
            files = _manifest_files(manifest, manifest_path)
            package_manifests = _package_manifests(
                manifest_path.parent, manifest_path, manifest
            )
            _assert_package_file_owner_identities_agree(package_manifests)
        except (OSError, SourceArtifactManifestError) as exc:
            errors.append(f"Could not read {manifest_path}: {exc}")
            continue

        manifest_source_id = (
            source_id if source_id is not None else manifest.get("source_id")
        )
        manifest_package_id = (
            package_id if package_id is not None else manifest.get("package_id")
        )

        for package_manifest_name, package_manifest in package_manifests.items():
            package_manifest_path = Path(package_manifest_name)
            # A selected sibling takes the overrides like the selected manifest
            # itself; an unselected sibling is only preflighted so the package
            # boundary holds and keeps its own identifiers, which may
            # legitimately differ.
            if package_manifest_path == manifest_path:
                package_source_id = manifest_source_id
                package_id_value = manifest_package_id
            elif package_manifest_path in selected_manifests:
                package_source_id = (
                    source_id
                    if source_id is not None
                    else package_manifest.get("source_id")
                )
                package_id_value = (
                    package_id
                    if package_id is not None
                    else package_manifest.get("package_id")
                )
            else:
                package_source_id = package_manifest.get("source_id")
                package_id_value = package_manifest.get("package_id")
            package_files = _manifest_files(package_manifest, package_manifest_path)
            for year, spec in package_files.items():
                entry, _updated_spec = _publish_raw_manifest_entry(
                    package_manifest_path,
                    package_source_id,
                    package_id_value,
                    year,
                    spec,
                    r2_bucket=r2_bucket,
                    r2_prefix=r2_prefix,
                    wrangler_command=wrangler_command,
                    preflight_only=True,
                    manifest_identity=package_manifest,
                )
                if entry.errors:
                    preflight_failures.append(entry)
        prepared.append(
            (
                manifest_path,
                manifest,
                files,
                manifest_source_id,
                manifest_package_id,
            )
        )

    # A root sweep is one requested publish operation. Validate every selected
    # package before the first uploader call or manifest rewrite, otherwise a
    # malformed later package can make the command fail after earlier packages
    # have already changed external and local state.
    if errors or preflight_failures:
        entries.extend(preflight_failures)
        return RawArtifactPublishReport(
            root=str(root_path),
            entries=tuple(entries),
            errors=tuple(errors),
        )

    for (
        manifest_path,
        manifest,
        files,
        manifest_source_id,
        manifest_package_id,
    ) in prepared:
        updated = False
        for year, spec in files.items():
            entry, updated_spec = _publish_raw_manifest_entry(
                manifest_path,
                manifest_source_id,
                manifest_package_id,
                year,
                spec,
                r2_bucket=r2_bucket,
                r2_prefix=r2_prefix,
                wrangler_command=wrangler_command,
                manifest_identity=manifest,
            )
            entries.append(entry)
            if updated_spec is not None and isinstance(spec, dict):
                spec.update(updated_spec)
                updated = True
        if updated:
            if manifest_source_id:
                manifest.setdefault("source_id", manifest_source_id)
            if manifest_package_id:
                manifest.setdefault("package_id", manifest_package_id)
            manifest_path.write_text(
                yaml.safe_dump(manifest, sort_keys=False),
                encoding="utf-8",
            )

    return RawArtifactPublishReport(
        root=str(root_path),
        entries=tuple(entries),
        errors=tuple(errors),
    )


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
) -> ArtifactInventoryReport:
    """Inventory manifest-declared source artifacts under a root directory."""
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
                "r2_link_count": 0,
            },
            entries=(),
            errors=(f"Root does not exist: {root_path}",),
        )

    manifest_paths = _root_manifest_paths(root_path, manifest_filename)
    for manifest_path in manifest_paths:
        try:
            manifest = _read_manifest(manifest_path)
            files = _manifest_files(manifest, manifest_path)
            package_manifests = _package_manifests(
                manifest_path.parent, manifest_path, manifest
            )
            _assert_package_file_owner_identities_agree(package_manifests)
        except (OSError, SourceArtifactManifestError) as exc:
            errors.append(f"Could not read {manifest_path}: {exc}")
            continue
        for year, spec in files.items():
            entries.append(_inventory_entry(manifest_path, year, spec))

    counts = {
        "manifest_count": len(manifest_paths),
        "artifact_count": len(entries),
        "missing_count": sum(1 for entry in entries if not entry.exists),
        "checksum_mismatch_count": sum(
            1 for entry in entries if "checksum_mismatch" in entry.errors
        ),
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


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    """Return a manifest's parsed payload, refusing a document it cannot read.

    An absent or empty manifest reads as an empty mapping: ``fetch-artifact``
    writes the first entry into a package that has none. A document that parses
    as anything else -- a list, a scalar, a truncated or half-merged file -- is
    not an absent manifest, and treating it as one would let the fetch replace
    it with a single entry and drop everything it recorded.
    """
    if manifest_path.is_symlink():
        raise MalformedManifestError(
            f"{manifest_path} is a symlink; manifest reads and writes require "
            "a regular file at its lexical package path."
        )
    if not manifest_path.exists():
        return {}
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


def _select_vintage_entry(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    year: Any,
) -> tuple[Any, Any, dict[str, Any], int | None]:
    """Locate the entry a fetch revises: ``(key, files[key], entry, index)``.

    Integer and quoted-integer keys are two spellings of one vintage. Preserve
    the spelling already present, refuse a manifest that contains both, and
    reject any present non-mapping entry before publisher I/O. The final tuple
    slot matches the stacked #227 selector; table manifests never use a list
    index.
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

    forms: tuple[Any, ...]
    if isinstance(year, bool):
        forms = (year,)
    elif isinstance(year, int):
        forms = (year, str(year))
    else:
        text = str(year)
        if text.isdecimal() and (text == "0" or not text.startswith("0")):
            forms = (year, int(text))
        else:
            forms = (year,)
    present = [form for form in forms if form in files]
    if len(present) > 1:
        raise MalformedManifestError(
            f"{manifest_path}: Vintage {year!r} is recorded under both keys "
            f"{present!r}; one vintage has one key. Merge the entries by hand "
            "first. Chronicle will not choose which entry is the record."
        )
    if not present:
        return year, None, {}, None
    key = present[0]
    existing = files[key]
    if not isinstance(existing, dict):
        raise MalformedManifestError(
            f"{manifest_path} entry {key!r} must be a mapping; it is a "
            f"{type(existing).__name__}."
        )
    return key, existing, existing, None


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


def _split_r2_uri(uri: str) -> tuple[str, str, str] | None:
    """Split ``provider://bucket/key`` into its three parts, or None."""
    provider, separator, remainder = uri.partition("://")
    if not separator or not provider:
        return None
    bucket, separator, key = remainder.partition("/")
    if not separator or not bucket or not key:
        return None
    return (provider, bucket, key)


def _validated_recorded_storage(
    spec: Any,
    *,
    manifest_path: Path,
    year: Any,
) -> dict[str, Any]:
    """Return the entry's ``storage`` mapping, refusing a malformed one."""
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
    return storage


def _validated_recorded_r2(
    spec: Any,
    *,
    manifest_path: Path,
    year: Any,
) -> RecordedR2Object | None:
    """Return the object a recorded ``storage.r2`` block names, or None.

    The block must explicitly record provider ``r2`` and an ``r2://`` URI.
    Every additional locator field it supplies is cross-checked against that
    URI: ``key`` against its path, ``bucket`` against its authority, and the
    resulting key against the canonical content-addressed shape
    :func:`build_r2_key` writes. Reading one field and trusting the rest is
    what lets a block that says two different things survive a preserve or a
    publish.
    """
    storage = _validated_recorded_storage(spec, manifest_path=manifest_path, year=year)
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
) -> RecordedIdentity | None:
    """Return what a manifest entry says its vintage holds, if anything.

    A published entry is identified by its recorded object's content-addressed
    key. An entry that has not been published yet -- registered without an
    upload, or left behind by a failed one -- is identified by its own declared
    ``sha256`` and ``filename``. Both are recorded identities, and a fetch of
    different bytes over either one is a publisher revision.
    """
    recorded_r2 = _validated_recorded_r2(spec, manifest_path=manifest_path, year=year)
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
        for vintage, spec in _manifest_files(payload, manifest_path).items():
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
    not a collision: fall back to the declared field for it.
    """
    try:
        recorded = _validated_recorded_r2(
            entry, manifest_path=Path(manifest_name), year=vintage
        )
    except SourceArtifactManifestError:
        return None
    return None if recorded is None else recorded.sha256


def _assert_package_file_owner_identities_agree(
    manifests: Mapping[str, dict[str, Any]],
) -> None:
    """Refuse contradictory identities for any package-local filename.

    Publish and inventory sweep a manifest at a time, but the physical byte is
    shared by every manifest in its directory. Validate every identified owner
    as one package boundary before a selected manifest can upload anything.
    Entry-shape and local-file errors remain the per-entry preflight's job.
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
        for vintage, spec in _manifest_files(payload, manifest_path).items():
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
    for key, pending in unidentified.items():
        owners = owners_by_filename.get(key)
        if not owners:
            continue
        expected = owners[0].identity
        assert expected is not None
        for manifest_path, vintage, recorded_name in pending:
            try:
                local = matching_directory_entry(manifest_path.parent, recorded_name)
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


#: Entry fields a fetch owns. Every other field already recorded on the entry
#: is carried forward during a refetch or explicit publisher revision.
_FETCH_OWNED_FIELDS: frozenset[str] = frozenset(
    {
        "filename",
        "source_url",
        "sha256",
        "size_bytes",
        "fetched_at",
        "storage",
    }
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


def _upsert_manifest(
    manifest_path: Path,
    *,
    source_id: str,
    package_id: str,
    dataset: str,
    source_page: str,
    table: str,
    year: int,
    filename: str,
    source_url: str,
    sha256: str,
    size_bytes: int,
    fetched_at: str,
    r2_location: ArtifactStorageLocation | None,
    record_revision: bool = False,
) -> None:
    payload = _read_manifest(manifest_path)
    _manifest_files(payload, manifest_path)
    _assert_manifest_identifies(
        payload,
        manifest_path,
        source_id=source_id,
        package_id=package_id,
    )
    manifests = _package_manifests(manifest_path.parent, manifest_path, payload)
    owners = _manifest_file_owners(
        manifests, filename=filename, initializing=(manifest_path, year)
    )
    _assert_shared_owner_identities_agree(owners, filename=filename)
    payload.setdefault("source_id", source_id)
    payload.setdefault("package_id", package_id)
    payload.setdefault("dataset", dataset)
    payload.setdefault("source_page", source_page)
    payload.setdefault("table", table)
    if payload.get("files") is None:
        # setdefault keeps an explicit null (a bare ``files:`` line); the
        # entry below needs a mapping to record into.
        payload["files"] = {}
    key, _existing_value, recorded_spec, _index = _select_vintage_entry(
        payload,
        manifest_path=manifest_path,
        year=year,
    )
    file_entry: dict[str, Any] = {
        "filename": filename,
        "source_url": source_url,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "fetched_at": fetched_at,
    }
    for field, value in recorded_spec.items():
        if field not in _FETCH_OWNED_FIELDS and field not in file_entry:
            file_entry[field] = value
    identity = _recorded_identity(recorded_spec, manifest_path=manifest_path, year=key)
    new_r2 = r2_location.to_dict() if r2_location is not None else None
    r2_bucket = (new_r2 or {}).get("bucket") or default_r2_raw_bucket()
    _assert_recorded_identity_holds_these_bytes(
        identity,
        manifest_path=manifest_path,
        year=key,
        filename=filename,
        sha256=sha256,
        size_bytes=size_bytes,
        r2_bucket=r2_bucket,
        record_revision=record_revision,
    )
    if not record_revision:
        _assert_siblings_record_these_bytes(
            manifests,
            manifest_path=manifest_path,
            vintage=key,
            filename=filename,
            sha256=sha256,
        )
    for owner in owners:
        if owner.manifest_path == manifest_path and owner.spec is recorded_spec:
            continue
        _assert_recorded_identity_holds_these_bytes(
            owner.identity,
            manifest_path=owner.manifest_path,
            year=owner.vintage,
            filename=filename,
            sha256=sha256,
            size_bytes=size_bytes,
            r2_bucket=r2_bucket,
            record_revision=record_revision,
        )

    storage = _storage_for_fetched_identity(
        recorded_spec,
        identity=identity,
        filename=filename,
        sha256=sha256,
        new_r2=new_r2,
        fetched_at=fetched_at,
    )
    # An entry that has no storage to record carries no empty block: a
    # revision over a never-published entry supersedes nothing.
    if storage:
        file_entry["storage"] = storage
    payload["files"][key] = file_entry

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
                {
                    "filename": filename,
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "fetched_at": fetched_at,
                }
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
            manifests[str(owner.manifest_path)]["files"][owner.vintage] = revised_entry
            changed_paths.add(owner.manifest_path)

    rendered = {
        path: yaml.safe_dump(manifests[str(path)], sort_keys=False)
        for path in changed_paths
    }
    for path, text in sorted(rendered.items(), key=lambda item: str(item[0])):
        path.write_text(text, encoding="utf-8")


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


def _publish_raw_manifest_entry(
    manifest_path: Path,
    source_id: str,
    package_id: str,
    year: Any,
    spec: Any,
    *,
    r2_bucket: str,
    r2_prefix: str | None,
    wrangler_command: str,
    preflight_only: bool = False,
    manifest_identity: dict[str, Any] | None = None,
) -> tuple[RawArtifactPublishEntry, dict[str, Any] | None]:
    errors: list[str] = []
    # Validate original values below; reports must still serialize refusals of
    # YAML dates, sets, and other non-string declarations through the CLI.
    reported_source_id = str(source_id) if source_id is not None else ""
    reported_package_id = str(package_id) if package_id is not None else ""
    if not isinstance(spec, dict):
        spec = {}
        errors.append("malformed_file_spec")
    filename = str(spec.get("filename") or "")
    artifact_path = manifest_path.parent
    sha256_actual = None
    size_bytes = None

    def refuse(reason: str | None = None) -> tuple[RawArtifactPublishEntry, None]:
        """Report the entry unpublished, with nothing uploaded or rewritten."""
        if reason is not None:
            errors.append(reason)
        return (
            RawArtifactPublishEntry(
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
            ),
            None,
        )

    if filename and not is_bare_filename(filename):
        return (
            RawArtifactPublishEntry(
                manifest_path=str(manifest_path),
                source_id=reported_source_id,
                package_id=reported_package_id,
                year=str(year),
                filename=filename,
                local_path=str(manifest_path.parent),
                sha256=None,
                size_bytes=None,
                r2_location=None,
                upload=None,
                errors=(f"non_canonical_filename:{filename}",),
            ),
            None,
        )
    if is_manifest_filename(filename):
        return (
            RawArtifactPublishEntry(
                manifest_path=str(manifest_path),
                source_id=reported_source_id,
                package_id=reported_package_id,
                year=str(year),
                filename=filename,
                local_path=str(manifest_path.parent),
                sha256=None,
                size_bytes=None,
                r2_location=None,
                upload=None,
                errors=(f"manifest_named_filename:{filename}",),
            ),
            None,
        )
    try:
        recorded_r2 = _validated_recorded_r2(
            spec, manifest_path=manifest_path, year=year
        )
    except SourceArtifactManifestError as error:
        # A block that does not name one object cannot be treated as history,
        # and publishing under it would ship whichever field was read.
        return refuse(f"recorded_r2_locator_invalid:{error}")
    if recorded_r2 is None:
        try:
            _require_identity_segment(source_id, what="source_id")
            _require_identity_segment(package_id, what="package_id")
            _require_identity_segment(str(year), what="year")
            _assert_manifest_identifies(
                manifest_identity or {},
                manifest_path,
                source_id=source_id,
                package_id=package_id,
            )
        except SourceArtifactManifestError as error:
            return refuse(f"r2_identity_invalid:{error}")

    try:
        artifact_path = (
            matching_directory_entry(manifest_path.parent, filename)
            or manifest_path.parent / filename
        )
        if filename and artifact_path.name != filename:
            errors.append(f"artifact_spelling_mismatch:{filename}:{artifact_path.name}")
    except ValueError:
        errors.append(f"duplicate_artifact_spellings:{filename}")
        artifact_path = manifest_path.parent / filename
    sha256_expected = spec.get("sha256")
    sha256_actual = None
    size_bytes = None
    if not filename:
        errors.append("missing_filename")
    elif errors:
        pass
    elif artifact_path.is_symlink():
        errors.append(f"artifact_path_is_symlink:{filename}")
    elif not artifact_path.is_file():
        errors.append("missing_file")
    else:
        content = artifact_path.read_bytes()
        sha256_actual = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        if sha256_expected and sha256_actual != sha256_expected:
            errors.append("checksum_mismatch")

    if errors:
        return refuse()

    if recorded_r2 is not None and (recorded_r2.sha256, recorded_r2.filename) != (
        sha256_actual or "",
        Path(filename).name,
    ):
        # The recorded object is addressed by different bytes, so it is not
        # this file's history. Uploading anyway would either publish under a
        # key that misdescribes its content or restate a URI that belongs to
        # the superseded bytes. Registering a publisher revision is
        # `fetch-artifact --record-revision`, not a publish-time rewrite.
        return refuse(
            "recorded_r2_identity_mismatch:"
            f"recorded_sha256={recorded_r2.sha256}:"
            f"recorded_filename={recorded_r2.filename}:"
            f"local_sha256={sha256_actual}:"
            f"local_filename={Path(filename).name}"
        )

    if recorded_r2 is not None:
        # A recorded content-addressed object whose tail identifies the bytes
        # in hand is published history. Its source/package/year route may
        # predate today's country prefix or intentionally represent the
        # publisher's explicit route (for example Statbel's 2023 snapshots and
        # USDA's cross-manifest archive). Reconstructing a current route and
        # requiring equality would rewrite that history during a bucket
        # cutover. Only the checksum/filename tail decides byte identity.
        skipped = "recorded_r2_already_published"
        if recorded_r2.bucket != r2_bucket:
            skipped = (
                "recorded_r2_bucket_is_preserved_history:"
                f"recorded={recorded_r2.bucket}:requested={r2_bucket}"
            )
        return (
            RawArtifactPublishEntry(
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
                    bucket=recorded_r2.bucket,
                    key=recorded_r2.key,
                ),
                upload=None,
                errors=(),
                skipped=skipped,
            ),
            None,
        )

    if not source_id:
        return refuse("missing_source_id")
    if not package_id:
        return refuse("missing_package_id")
    try:
        resolved_r2_prefix = resolve_r2_prefix(
            prefix=r2_prefix,
            default_prefix=DEFAULT_R2_PREFIX,
            source_id=source_id,
            package_path=manifest_path,
        )
    except ValueError as error:
        return refuse(f"r2_prefix_invalid:{error}")

    location = ArtifactStorageLocation(
        provider="r2",
        bucket=r2_bucket,
        key=build_r2_key(
            source_id=source_id,
            package_id=package_id,
            year=year,
            sha256=sha256_actual or "",
            filename=filename,
            prefix=resolved_r2_prefix,
            package_path=manifest_path,
        ),
    )
    if preflight_only:
        return (
            RawArtifactPublishEntry(
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
            ),
            None,
        )
    upload = _upload_r2_object(
        location,
        artifact_path,
        wrangler_command=wrangler_command,
    )
    if not upload.ok:
        errors.append("r2_upload_failed")

    updated_spec: dict[str, Any] | None = None
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

    return (
        RawArtifactPublishEntry(
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
        ),
        updated_spec,
    )


def _inventory_entry(
    manifest_path: Path,
    year: Any,
    spec: Any,
) -> ArtifactInventoryEntry:
    errors: list[str] = []
    if not isinstance(spec, dict):
        spec = {}
        errors.append("malformed_file_spec")
    filename = str(spec.get("filename") or "")
    r2 = None
    recorded_r2 = None
    try:
        recorded_r2 = _validated_recorded_r2(
            spec, manifest_path=manifest_path, year=year
        )
    except SourceArtifactManifestError as error:
        errors.append(f"recorded_r2_locator_invalid:{error}")
    if recorded_r2 is not None and (
        recorded_r2.filename != filename
        or (spec.get("sha256") is not None and recorded_r2.sha256 != spec["sha256"])
    ):
        errors.append("recorded_r2_identity_mismatch")
    if errors:
        return ArtifactInventoryEntry(
            manifest_path=str(manifest_path),
            year=str(year),
            filename=filename,
            local_path=str(manifest_path.parent),
            exists=False,
            sha256_expected=spec.get("sha256"),
            sha256_actual=None,
            size_bytes=None,
            source_url=spec.get("source_url"),
            r2=None,
            errors=tuple(errors),
        )
    if filename and not is_bare_filename(filename):
        return ArtifactInventoryEntry(
            manifest_path=str(manifest_path),
            year=str(year),
            filename=filename,
            local_path=str(manifest_path.parent),
            exists=False,
            sha256_expected=spec.get("sha256"),
            sha256_actual=None,
            size_bytes=None,
            source_url=spec.get("source_url"),
            r2=r2,
            errors=(f"non_canonical_filename:{filename}",),
        )
    if is_manifest_filename(filename):
        return ArtifactInventoryEntry(
            manifest_path=str(manifest_path),
            year=str(year),
            filename=filename,
            local_path=str(manifest_path.parent),
            exists=False,
            sha256_expected=spec.get("sha256"),
            sha256_actual=None,
            size_bytes=None,
            source_url=spec.get("source_url"),
            r2=r2,
            errors=(f"manifest_named_filename:{filename}",),
        )
    try:
        artifact_path = (
            matching_directory_entry(manifest_path.parent, filename)
            or manifest_path.parent / filename
        )
        if filename and artifact_path.name != filename:
            errors.append(f"artifact_spelling_mismatch:{filename}:{artifact_path.name}")
    except ValueError:
        errors.append(f"duplicate_artifact_spellings:{filename}")
        artifact_path = manifest_path.parent / filename
    symlink = bool(filename) and artifact_path.is_symlink()
    exists = bool(filename) and not errors and not symlink and artifact_path.is_file()
    sha256_expected = spec.get("sha256")
    sha256_actual = None
    size_bytes = None
    if not filename:
        errors.append("missing_filename")
    elif errors:
        pass
    elif symlink:
        errors.append(f"artifact_path_is_symlink:{filename}")
    elif not exists:
        errors.append("missing_file")
    else:
        content = artifact_path.read_bytes()
        sha256_actual = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        if sha256_expected and sha256_actual != sha256_expected:
            errors.append("checksum_mismatch")
    if recorded_r2 is not None:
        if sha256_actual is not None and sha256_actual != recorded_r2.sha256:
            errors.append("recorded_r2_identity_mismatch")
        if not errors:
            r2 = {
                "provider": recorded_r2.provider,
                "bucket": recorded_r2.bucket,
                "key": recorded_r2.key,
                "uri": recorded_r2.uri,
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
        errors=tuple(errors),
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
    cleaned = value.strip().strip("/")
    if not cleaned:
        raise ValueError("R2 key parts cannot be empty.")
    return cleaned.replace(" ", "_")


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


def _clean_relative_key_parts(value: str) -> tuple[str, ...]:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("R2 artifact paths must be relative.")
    parts = tuple(_clean_key_part(part) for part in path.parts if part != ".")
    if not parts or any(part == ".." for part in parts):
        raise ValueError("R2 artifact paths cannot be empty or contain '..'.")
    return parts
