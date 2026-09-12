"""Register raw microdata releases from a Microcosm source-stages manifest.

Chronicle registers every raw microdata release its consumers build from and
stores the bytes of only those a publisher permits it to redistribute
(``docs/adr-chronicle-raw-microdata-identity.md``). This script drives both
halves of that from the pins Microcosm already reviewed:

``emit``
    Write hash-only ``kind: microdata_release`` manifests for ``licensed`` and
    ``restricted`` releases. Every checksum, size, filename, and vintage is read
    verbatim from the Microcosm source-stages JSON; nothing is recomputed and
    nothing is invented. Each registration is a ``consumer_pin``: it names the
    consumer as the attester and records the repository, path, and commit the
    pin was read from, and carries no verification date of its own. A release
    Microcosm pins without a checksum is reported as a blocker and never
    registered.

``plan``
    Print the exact ``chronicle fetch-artifact ... --upload-r2`` commands to run
    from a networked machine for ``public`` releases, whose bytes Chronicle does
    archive. Every command carries the reviewed identity as arguments: the
    publisher, the vintage, and -- when Microcosm's pin is of the publisher
    bytes -- ``--expected-sha256`` and ``--expected-size-bytes``, so the fetch
    refuses a reissue before archiving it. Publisher URLs are copied verbatim
    from the Microcosm manifest; a release whose manifest carries no URL, no
    publisher-bytes checksum, or no licence-evidence URL prints a ``TODO``
    instead of a guess, and that command cannot run until the TODO is filled.

The catalogue below is the only authored content: it maps a Microcosm artifact
onto Chronicle's ``{source_id, package_id, year, sha256, filename}`` identity
and records the publisher's terms. Run it read-only against a Microcosm
checkout; this script never writes to that repository.

Usage::

    python scripts/register_microdata_releases.py emit \\
        --microcosm-root ~/PolicyEngine/microcosm \\
        --root db/data

    python scripts/register_microdata_releases.py plan \\
        --microcosm-root ~/PolicyEngine/microcosm --root db/data
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit

# Allow `python scripts/register_microdata_releases.py` from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronicle.artifacts import default_r2_raw_bucket  # noqa: E402
from chronicle.registration import (  # noqa: E402
    ACCESS_PUBLIC,
    HASH_SOURCE_CONSUMER_PIN,
    MICRODATA_RELEASE_KIND,
    HashOnlyRegistrationError,
    register_hash_only_artifact,
)

#: The consumer whose reviewed pins every registration here transcribes.
CONSUMER_REPOSITORY = "PolicyEngine/microcosm"

#: Placeholders a planned command prints where Microcosm pins nothing. Each is
#: refused by fetch-artifact as written, so a command carrying one cannot run
#: until a reviewer replaces it.
TODO_PUBLISHER_URL = "TODO_PUBLISHER_URL"
TODO_REVIEWED_SHA256 = "TODO_REVIEWED_SHA256"
TODO_EVIDENCE_URL = "TODO_EVIDENCE_URL"

#: Allowlisted licence identifier for a public-use file of a U.S. federal
#: statistical agency (chronicle/licences.py).
US_GOVERNMENT_WORK = "US-Government-Work"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

#: Microcosm's per-artifact ``kind`` mapped onto Chronicle's access class.
#:
#: ``private_microdata`` maps to ``restricted`` rather than ``licensed``: the
#: bytes are held only in a private mirror, and over-classifying is the safe
#: direction because both classes are registered hash-only and neither ever
#: places bytes in a Chronicle store.
ACCESS_BY_MICROCOSM_KIND: dict[str, str] = {
    "public_microdata": "public",
    "licensed_microdata": "licensed",
    "private_microdata": "restricted",
    "restricted_microdata": "restricted",
}

#: Provenance sentence written onto every registration this script emits.
HASH_PROVENANCE = (
    "SHA-256 transcribed verbatim from the consumer's reviewed pin; Chronicle "
    "holds no bytes for this release and did not recompute the checksum."
)


@dataclass(frozen=True)
class ArtifactSelector:
    """Locate one artifact inside a Microcosm source-stages JSON file.

    ``stage`` names the build stage; ``match`` is a set of artifact fields that
    must equal the given values. ``compare_across_kinds`` lets a release detect
    an access-kind disagreement among otherwise matching cross-stage references
    before enforcing ``kind``. A selector that matches nothing, or matches
    inconsistent bytes across stages, is a hard error rather than a guess.
    """

    stage: str | None = None
    match: Mapping[str, Any] = field(default_factory=dict)
    kind: str | None = None
    compare_across_kinds: bool = False


@dataclass(frozen=True)
class Release:
    """One Chronicle registration drawn from a Microcosm pin."""

    release_id: str
    manifest: str
    selector: ArtifactSelector
    source_id: str
    package_id: str
    package_dir: str
    year: int
    table: str
    publisher: str
    licence: str
    access: str
    #: Artifact field holding the publisher filename, when not ``filename``.
    filename_field: str = "filename"
    #: Artifact field holding the publisher URL, for ``public`` releases.
    url_field: str = "locator"
    study: str | None = None
    doi: str | None = None
    source_page: str | None = None
    access_route: str | None = None
    vintage: str | None = None
    notes: str | None = None
    #: Whether Microcosm's pinned ``sha256`` is of the publisher artifact
    #: itself. It is not when the pin covers a derived file or an extracted
    #: archive member, and such a hash must never be presented as the checksum
    #: a fetch should reproduce.
    pinned_sha_is_publisher_bytes: bool = True
    #: Licence evidence for a public release: who issued the file under the
    #: allowlisted term, the scope statement, and the durable evidence URL.
    #: A missing URL prints a TODO in the plan; it is never guessed.
    licence_evidence_issuer: str | None = None
    licence_evidence_scope: str | None = None
    licence_evidence_url: str | None = None
    #: Set when Microcosm pins the release without a checksum.
    blocker: str | None = None


UK_STAGES = "packages/microcosm-build/src/microcosm/build/uk/source_stages.json"
UK_HMRC_STAGES = (
    "packages/microcosm-build/src/microcosm/build/uk/hmrc_income_source_stages.json"
)
BE_STAGES = "packages/microcosm-build/src/microcosm/build/be/source_stages.json"
US_STAGES = "packages/microcosm-build/src/microcosm/build/us/source_stages.json"
US_ACS_2024 = (
    "packages/microcosm-build/src/microcosm/build/us_runtime/acs_2024_1yr_sources.json"
)

#: FRS 2023-24 tabs, in the order Microcosm's ``frs_spine`` stage lists them.
FRS_TABS: tuple[str, ...] = (
    "accounts",
    "adult",
    "benefits",
    "benunit",
    "child",
    "chldcare",
    "extchild",
    "househol",
    "job",
    "maint",
    "mortgage",
    "oddjob",
    "penprov",
    "pension",
)

FRS_LICENCE = "UK Data Service End User Licence"
FRS_STUDY = "UK Data Service SN 9367"
FRS_DOI = "10.5255/UKDA-SN-9367-2"
FRS_ACCESS_ROUTE = (
    "UK Data Service study SN 9367 under its End User Licence. Bytes stay in "
    "the licensed environment the consumer already operates; no Chronicle "
    "credential grants access to them."
)
FRS_NOTES = (
    "Microcosm's frs_spine stage cites 'UK Data Service SN 9367, DOI "
    "10.5255/UKDA-SN-9367-2'; its frs_employment, frs_council_tax, "
    "frs_education, and frs_legacy_proxies stages cite 'SN 9252' for the same "
    "2023_24 tabs. The tabs are the same bytes across all five stages (identical "
    "SHA-256), so the registration carries the study reference that also carries "
    "a DOI. " + HASH_PROVENANCE
)

CENSUS_SCOPE = (
    "Public-use microdata file published by the U.S. Census Bureau, a federal "
    "agency; a work of the United States Government under 17 U.S.C. §105."
)
FED_SCOPE = (
    "Public data set published by the Board of Governors of the Federal "
    "Reserve System, a federal agency; a work of the United States Government "
    "under 17 U.S.C. §105."
)

CATALOGUE: tuple[Release, ...] = (
    *(
        Release(
            release_id=f"dwp-frs-2023-24:{tab}",
            manifest=UK_STAGES,
            selector=ArtifactSelector(
                kind="licensed_microdata",
                match={"table": tab},
                compare_across_kinds=True,
            ),
            source_id="dwp",
            package_id="dwp-frs-2023-24",
            package_dir="dwp/frs_2023_24",
            year=2023,
            table="Family Resources Survey 2023-24",
            publisher="Department for Work and Pensions",
            licence=FRS_LICENCE,
            access="licensed",
            filename_field="locator",
            study=FRS_STUDY,
            doi=FRS_DOI,
            access_route=FRS_ACCESS_ROUTE,
            notes=FRS_NOTES,
        )
        for tab in FRS_TABS
    ),
    Release(
        release_id="hmrc-spi-public-use-tape-2022-23:put2223uk",
        manifest=UK_HMRC_STAGES,
        selector=ArtifactSelector(
            stage="hmrc_spi_income",
            kind="private_microdata",
            match={"filename": "put2223uk.tab"},
        ),
        source_id="hmrc",
        package_id="hmrc-spi-public-use-tape-2022-23",
        package_dir="hmrc/spi_public_use_tape_2022_23",
        year=2022,
        table="Survey of Personal Incomes Public Use Tape 2022-23",
        publisher="HM Revenue and Customs",
        licence="UK Data Service End User Licence (study SN 9422)",
        access="restricted",
        study="UK Data Service SN 9422",
        doi="10.5255/UKDA-SN-9422-1",
        access_route=(
            "UK Data Service study SN 9422. Microcosm reaches the bytes through "
            "PolicyEngine's licensed copy in the private "
            "policyengine/policyengine-uk-data-private Hugging Face repository "
            "(spi_2022_23.zip); no Chronicle credential grants access to them."
        ),
        notes=(
            "Microcosm classes this artifact kind: private_microdata with "
            "access: private_local_input. Chronicle registers it restricted "
            "because the bytes are held only in a private mirror; if the UKDS "
            "terms for SN 9422 are confirmed as End User Licence it can be "
            "reclassified licensed, which changes nothing about storage — both "
            "classes are hash-only. " + HASH_PROVENANCE
        ),
    ),
    Release(
        release_id="statbel-be-silc-2023",
        manifest=BE_STAGES,
        selector=ArtifactSelector(
            stage="silc_load",
            kind="restricted_microdata",
        ),
        source_id="statbel",
        package_id="statbel-be-silc-2023",
        package_dir="statbel/be_silc_2023",
        year=2023,
        table="BE-SILC 2023 scientific-use files (D, R, H, P)",
        publisher="Statbel",
        licence="Statbel/Eurostat scientific-use",
        access="restricted",
        source_page="https://statbel.fgov.be/en/themes/households/poverty-and-living-conditions",
        access_route=(
            "Statbel BE-SILC scientific-use files: D (household register), "
            "R (personal register), H (household data), P (personal data)."
        ),
        blocker=(
            "Microcosm's be/source_stages.json pins the BE-SILC scientific-use "
            "files with no sha256, no size_bytes, and no per-file filename — it "
            "names only the four file roles. A registration is identified by "
            "{source_id, package_id, year, sha256, filename}, so this release "
            "cannot be registered until the consumer publishes a reviewed "
            "checksum per file. No hash is invented here."
        ),
    ),
    # ---- public releases: bytes are archived, so these are fetch-and-upload ----
    Release(
        release_id="census-cps-asec-2023",
        manifest=US_STAGES,
        selector=ArtifactSelector(
            stage="weeks_unemployed_input",
            kind="public_microdata",
            match={"member": "pppub23.csv"},
        ),
        source_id="census_cps",
        package_id="census-cps-asec-2023",
        package_dir="census/cps_asec_2023",
        year=2023,
        table="CPS Annual Social and Economic Supplement 2023 public-use files",
        publisher="U.S. Census Bureau",
        licence=US_GOVERNMENT_WORK,
        licence_evidence_issuer="U.S. Census Bureau",
        licence_evidence_scope=CENSUS_SCOPE,
        access=ACCESS_PUBLIC,
        source_page="https://www.census.gov/programs-surveys/cps/data/datasets.html",
    ),
    Release(
        release_id="census-cps-basic-monthly-2024",
        manifest=US_STAGES,
        selector=ArtifactSelector(
            stage="org_wages",
            kind="public_microdata",
        ),
        source_id="census_cps",
        package_id="census-cps-basic-monthly-2024",
        package_dir="census/cps_basic_monthly_2024",
        year=2024,
        table="CPS basic monthly public-use files, January-December 2024",
        publisher="U.S. Census Bureau",
        licence=US_GOVERNMENT_WORK,
        licence_evidence_issuer="U.S. Census Bureau",
        licence_evidence_scope=CENSUS_SCOPE,
        access=ACCESS_PUBLIC,
        source_page="https://www2.census.gov/programs-surveys/cps/datasets/2024/basic/",
        notes=(
            "Microcosm pins the twelve monthly files as the locator string "
            "'jan24pub through dec24pub' with no per-file URL, filename "
            "extension, or checksum. The publisher directory is the stage "
            "source; the twelve filenames must be read off that directory "
            "before the fetch commands can be completed."
        ),
    ),
    Release(
        release_id="census-acs-pums-2022-household",
        manifest=US_STAGES,
        selector=ArtifactSelector(
            stage="acs_rent",
            kind="versioned_derived_microdata",
        ),
        source_id="census_acs",
        package_id="census-acs-pums-2022-1yr",
        package_dir="census/acs_pums_2022_1yr",
        year=2022,
        table="ACS 2022 1-Year PUMS household file",
        publisher="U.S. Census Bureau",
        licence=US_GOVERNMENT_WORK,
        licence_evidence_issuer="U.S. Census Bureau",
        licence_evidence_scope=CENSUS_SCOPE,
        access=ACCESS_PUBLIC,
        url_field="official_household_source",
        source_page="https://www.census.gov/programs-surveys/acs",
        pinned_sha_is_publisher_bytes=False,
        notes=(
            "The Microcosm artifact's own sha256 belongs to the derived "
            "acs_2022.h5, not to the publisher zip; only the publisher URL is "
            "reused here. The fetch computes the release checksum."
        ),
    ),
    Release(
        release_id="census-acs-pums-2022-person",
        manifest=US_STAGES,
        selector=ArtifactSelector(
            stage="acs_rent",
            kind="versioned_derived_microdata",
        ),
        source_id="census_acs",
        package_id="census-acs-pums-2022-1yr",
        package_dir="census/acs_pums_2022_1yr",
        year=2022,
        table="ACS 2022 1-Year PUMS person file",
        publisher="U.S. Census Bureau",
        licence=US_GOVERNMENT_WORK,
        licence_evidence_issuer="U.S. Census Bureau",
        licence_evidence_scope=CENSUS_SCOPE,
        access=ACCESS_PUBLIC,
        url_field="official_person_source",
        source_page="https://www.census.gov/programs-surveys/acs",
        pinned_sha_is_publisher_bytes=False,
        notes=(
            "The Microcosm artifact's own sha256 belongs to the derived "
            "acs_2022.h5, not to the publisher zip; only the publisher URL is "
            "reused here. The fetch computes the release checksum."
        ),
    ),
    Release(
        release_id="census-acs-pums-2024-household",
        manifest=US_ACS_2024,
        selector=ArtifactSelector(match={"role": "household"}),
        source_id="census_acs",
        package_id="census-acs-pums-2024-1yr",
        package_dir="census/acs_pums_2024_1yr",
        year=2024,
        table="ACS 2024 1-Year PUMS household file",
        publisher="U.S. Census Bureau",
        licence=US_GOVERNMENT_WORK,
        licence_evidence_issuer="U.S. Census Bureau",
        licence_evidence_scope=CENSUS_SCOPE,
        access=ACCESS_PUBLIC,
        url_field="url",
        vintage="2024",
        source_page="https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/",
    ),
    Release(
        release_id="census-acs-pums-2024-person",
        manifest=US_ACS_2024,
        selector=ArtifactSelector(match={"role": "person"}),
        source_id="census_acs",
        package_id="census-acs-pums-2024-1yr",
        package_dir="census/acs_pums_2024_1yr",
        year=2024,
        table="ACS 2024 1-Year PUMS person file",
        publisher="U.S. Census Bureau",
        licence=US_GOVERNMENT_WORK,
        licence_evidence_issuer="U.S. Census Bureau",
        licence_evidence_scope=CENSUS_SCOPE,
        access=ACCESS_PUBLIC,
        url_field="url",
        vintage="2024",
        source_page="https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/",
    ),
    Release(
        release_id="federal-reserve-scf-2022-summary",
        manifest=US_STAGES,
        selector=ArtifactSelector(
            stage="scf_wealth",
            kind="public_microdata",
            match={"member": "rscfp2022.dta"},
        ),
        source_id="federal_reserve",
        package_id="federal-reserve-scf-2022",
        package_dir="federal_reserve/scf_2022",
        year=2022,
        table="Survey of Consumer Finances 2022 summary extract",
        publisher="Board of Governors of the Federal Reserve System",
        licence=US_GOVERNMENT_WORK,
        licence_evidence_issuer="Board of Governors of the Federal Reserve System",
        licence_evidence_scope=FED_SCOPE,
        access=ACCESS_PUBLIC,
        source_page="https://www.federalreserve.gov/econres/scfindex.htm",
    ),
    Release(
        release_id="federal-reserve-scf-2022-full",
        manifest=US_STAGES,
        selector=ArtifactSelector(
            stage="scf_wealth",
            kind="public_microdata",
            match={"member": "p22i6.dta"},
        ),
        source_id="federal_reserve",
        package_id="federal-reserve-scf-2022",
        package_dir="federal_reserve/scf_2022",
        year=2022,
        table="Survey of Consumer Finances 2022 full public data set",
        publisher="Board of Governors of the Federal Reserve System",
        licence=US_GOVERNMENT_WORK,
        licence_evidence_issuer="Board of Governors of the Federal Reserve System",
        licence_evidence_scope=FED_SCOPE,
        access=ACCESS_PUBLIC,
        source_page="https://www.federalreserve.gov/econres/scfindex.htm",
        notes=(
            "Microcosm records no checksum for this zip: 'Full-file SHA-256 "
            "pending one network-enabled provisioning fetch'. The fetch below "
            "computes and registers it."
        ),
    ),
    Release(
        release_id="census-sipp-2023",
        manifest=US_STAGES,
        selector=ArtifactSelector(
            stage="scf_wealth",
            kind="public_microdata",
            match={"member": "pu2023.csv"},
        ),
        source_id="census_sipp",
        package_id="census-sipp-2023",
        package_dir="census/sipp_2023",
        year=2023,
        table="Survey of Income and Program Participation 2023 public-use file",
        publisher="U.S. Census Bureau",
        licence=US_GOVERNMENT_WORK,
        licence_evidence_issuer="U.S. Census Bureau",
        licence_evidence_scope=CENSUS_SCOPE,
        access=ACCESS_PUBLIC,
        source_page="https://www.census.gov/programs-surveys/sipp.html",
        pinned_sha_is_publisher_bytes=False,
        notes=(
            "Microcosm reaches this file through an immutable Hugging Face "
            "mirror (revision 21280dca5995e978d706740a8a4b9b7860cfd7b6) and "
            "records no Census URL, so the publisher URL must be read off the "
            "SIPP dataset page before the fetch. Microcosm's pinned sha256 "
            "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2 "
            "and size 3726010471 are for the mirrored pu2023.csv member, not "
            "for whatever archive the Census page serves, so they are not the "
            "checksum this fetch should be expected to reproduce."
        ),
    ),
)


class CatalogueError(RuntimeError):
    """Raised when the catalogue cannot be resolved against Microcosm."""


def _load_manifest_snapshot(
    microcosm_root: Path, relative: str
) -> tuple[dict[str, Any], bytes]:
    """Read and parse one immutable-in-memory consumer-manifest snapshot."""
    path = microcosm_root / relative
    if not path.exists():
        raise CatalogueError(f"Microcosm manifest not found: {path}")
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise CatalogueError(f"Microcosm manifest must be an object: {path}")
    return payload, content


def load_manifest(microcosm_root: Path, relative: str) -> dict[str, Any]:
    """Load one Microcosm JSON manifest, read-only."""
    payload, _content = _load_manifest_snapshot(microcosm_root, relative)
    return payload


def iter_manifest_artifacts(
    payload: Mapping[str, Any],
) -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Yield ``(stage, artifact)`` pairs from either Microcosm manifest shape.

    Source-stages manifests nest artifacts under ``stages``; the ACS runtime
    manifest carries a flat top-level ``artifacts`` list with no stages.
    """
    stages = payload.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            for artifact in stage.get("artifacts") or ():
                if isinstance(artifact, Mapping):
                    yield stage, artifact
        return
    for artifact in payload.get("artifacts") or ():
        if isinstance(artifact, Mapping):
            yield payload, artifact


def _artifact_byte_identity(artifact: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return the immutable locator/checksum pair used across stage labels."""
    locator = artifact.get("locator")
    sha256 = artifact.get("sha256")
    if not isinstance(locator, str) or not locator.strip():
        return None
    if not isinstance(sha256, str) or not sha256.strip():
        return None
    return locator, sha256


def select_artifact(
    payload: Mapping[str, Any],
    selector: ArtifactSelector,
    *,
    release_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the single ``(stage, artifact)`` a selector identifies.

    Several Microcosm stages reference the same bytes. Duplicates are accepted
    only when they agree on every field the registration reads; disagreement is
    an error, never a silent first-match.
    """
    candidates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for stage, artifact in iter_manifest_artifacts(payload):
        if selector.stage is not None and stage.get("stage") != selector.stage:
            continue
        candidates.append((stage, artifact))

    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for stage, artifact in candidates:
        if (
            selector.kind is not None
            and artifact.get("kind") != selector.kind
            and not selector.compare_across_kinds
        ):
            continue
        if any(artifact.get(key) != value for key, value in selector.match.items()):
            continue
        matches.append((stage, artifact))
    if selector.compare_across_kinds:
        anchor_ids = {id(artifact) for _stage, artifact in matches}
        byte_identities = {
            identity
            for _stage, artifact in matches
            if (identity := _artifact_byte_identity(artifact)) is not None
        }
        matches = [
            (stage, artifact)
            for stage, artifact in candidates
            if id(artifact) in anchor_ids
            or _artifact_byte_identity(artifact) in byte_identities
        ]
    if not matches:
        raise CatalogueError(
            f"{release_id}: no Microcosm artifact matches {selector}. The "
            "consumer manifest changed; re-derive the catalogue rather than "
            "hand-editing a registration."
        )
    first_stage, first = matches[0]
    for _stage, artifact in matches[1:]:
        if dict(artifact) != dict(first):
            raise CatalogueError(
                f"{release_id}: Microcosm pins conflicting values for this "
                "artifact across stages; refusing to choose between them."
            )
    if selector.kind is not None and first.get("kind") != selector.kind:
        raise CatalogueError(
            f"{release_id}: matching Microcosm artifacts declare "
            f"kind={first.get('kind')!r}, not {selector.kind!r}. The consumer "
            "manifest changed; re-derive the catalogue rather than silently "
            "reclassifying its bytes."
        )
    return first_stage, first


@dataclass(frozen=True)
class ResolvedRelease:
    """A catalogue entry resolved against Microcosm's pinned artifact."""

    release: Release
    stage: Mapping[str, Any]
    artifact: Mapping[str, Any]
    manifest_bytes: bytes = field(repr=False)

    @property
    def filename(self) -> str | None:
        """Publisher filename, from the release's declared artifact field."""
        value = self.artifact.get(self.release.filename_field)
        if not value:
            return None
        # A locator may be a URL or a bare filename; take the last path segment.
        return str(value).rstrip("/").rsplit("/", 1)[-1]

    @property
    def url(self) -> str | None:
        """Publisher URL, when the Microcosm artifact records one."""
        value = self.artifact.get(self.release.url_field)
        text = str(value).strip() if value else ""
        return text if text.startswith(("http://", "https://")) else None

    @property
    def sha256(self) -> str | None:
        """Checksum Microcosm pins for these bytes, if any."""
        value = self.artifact.get("sha256")
        return str(value) if value else None

    @property
    def size_bytes(self) -> int | None:
        """Size Microcosm pins for these bytes, if any."""
        value = self.artifact.get("size_bytes")
        return int(value) if isinstance(value, int) else None

    @property
    def vintage(self) -> str:
        """Publisher vintage label, from the release or the artifact."""
        return str(self.release.vintage or self.artifact.get("vintage") or "")


def resolve(
    microcosm_root: Path,
    releases: Sequence[Release],
) -> list[ResolvedRelease]:
    """Resolve every catalogue entry against the Microcosm checkout."""
    snapshots: dict[str, tuple[dict[str, Any], bytes]] = {}
    resolved: list[ResolvedRelease] = []
    for release in releases:
        if release.manifest not in snapshots:
            snapshots[release.manifest] = _load_manifest_snapshot(
                microcosm_root, release.manifest
            )
        payload, content = snapshots[release.manifest]
        stage, artifact = select_artifact(
            payload,
            release.selector,
            release_id=release.release_id,
        )
        resolved.append(
            ResolvedRelease(
                release=release,
                stage=stage,
                artifact=artifact,
                manifest_bytes=content,
            )
        )
    return resolved


def assert_consumer_repository_root(microcosm_root: Path) -> Path:
    """Require the consumer checkout root to be the Git repository root."""
    expected = microcosm_root.resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(microcosm_root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CatalogueError(
            f"Cannot read the Git repository root for {microcosm_root}: {exc}. "
            "Pass --microcosm-commit with the reviewed commit after using the "
            "actual PolicyEngine/microcosm checkout root."
        ) from exc
    repository_root = Path(completed.stdout.strip()).resolve()
    if repository_root != expected:
        raise CatalogueError(
            f"Git repository root {repository_root} does not equal "
            f"--microcosm-root {expected}. The recorded "
            "PolicyEngine/microcosm path must identify the exact blob checked; "
            "pass the repository root itself."
        )
    return repository_root


def verified_consumer_repository(microcosm_root: Path) -> str:
    """Read the checkout's origin and require the expected GitHub repository.

    Normalize HTTPS, SSH URL, and SSH scp-style remotes locally; no remote is
    contacted. The returned repository identity is the value emission records.
    """
    assert_consumer_repository_root(microcosm_root)
    try:
        origin = subprocess.run(
            ["git", "-C", str(microcosm_root), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CatalogueError(
            "Cannot verify consumer repository identity: the checkout needs an "
            f"origin remote for {CONSUMER_REPOSITORY} on github.com."
        ) from exc

    scp_remote = re.fullmatch(r"git@([^/:]+):(.+)", origin)
    if scp_remote:
        origin = f"ssh://git@{scp_remote[1]}/{scp_remote[2]}"
    try:
        remote = urlsplit(origin)
        repository = remote.path.strip("/").removesuffix(".git")
        valid_route = (
            remote.hostname == "github.com"
            and remote.scheme in ("https", "ssh")
            and remote.port in (None, 443 if remote.scheme == "https" else 22)
            and not remote.query
            and not remote.fragment
        )
    except ValueError as exc:
        raise CatalogueError(
            "Cannot verify consumer repository identity: origin is not a valid "
            f"GitHub remote for {CONSUMER_REPOSITORY}."
        ) from exc
    if not valid_route or repository.casefold() != CONSUMER_REPOSITORY.casefold():
        # Avoid printing credentials embedded in an HTTPS remote URL.
        raise CatalogueError(
            f"Consumer repository identity from origin is "
            f"{remote.hostname or 'unknown host'}/{repository}; expected "
            f"github.com/{CONSUMER_REPOSITORY}. Refusing to emit consumer pins."
        )
    return repository


def pin_commit(microcosm_root: Path, relative: str) -> str:
    """Return the last commit that changed a consumer manifest, read-only.

    The caller separately verifies that this candidate commit's blob is the
    exact byte snapshot it parsed. Keeping discovery and verification separate
    lets explicit commit overrides pass through the same mandatory check.
    """
    assert_consumer_repository_root(microcosm_root)
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(microcosm_root),
                "log",
                "-1",
                "--format=%H",
                "--",
                relative,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CatalogueError(
            f"Cannot read the commit of {relative} in {microcosm_root}: {exc}. "
            "Pass --microcosm-commit with the reviewed commit."
        ) from exc
    commit = completed.stdout.strip()
    if not _COMMIT_RE.match(commit):
        raise CatalogueError(
            f"{microcosm_root} records no commit for {relative}; pass "
            "--microcosm-commit with the reviewed commit."
        )
    return commit


def assert_manifest_matches_commit(
    microcosm_root: Path,
    relative: str,
    commit: str,
    *,
    loaded_bytes: bytes,
) -> None:
    """Require ``loaded_bytes`` to equal ``relative``'s blob at ``commit``."""
    relative_path = Path(relative)
    if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
        raise CatalogueError(
            f"Consumer manifest path must stay inside {microcosm_root}: {relative!r}."
        )
    try:
        object_type = subprocess.run(
            ["git", "-C", str(microcosm_root), "cat-file", "-t", commit],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = exc.stderr.strip()
        suffix = f" ({detail})" if detail else ""
        raise CatalogueError(
            f"Cannot read consumer manifest {relative} at commit {commit} from "
            f"{microcosm_root}{suffix}. The recorded commit must be a readable "
            "Git commit containing the exact reviewed blob."
        ) from exc
    if object_type != "commit":
        raise CatalogueError(
            f"Consumer manifest pin {commit} names a Git {object_type or 'unknown'} "
            "object, not a commit. Refusing to record it as pinned_from.commit."
        )
    assert_consumer_repository_root(microcosm_root)
    object_name = f"{commit}:./{relative_path.as_posix()}"
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(microcosm_root),
                "cat-file",
                "blob",
                object_name,
            ],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = exc.stderr.decode(errors="replace").strip()
        suffix = f" ({detail})" if detail else ""
        raise CatalogueError(
            f"Cannot read consumer manifest {relative} at commit {commit} from "
            f"{microcosm_root}{suffix}. The recorded commit must contain the "
            "exact reviewed blob."
        ) from exc
    if completed.stdout != loaded_bytes:
        raise CatalogueError(
            f"Consumer manifest bytes loaded from {microcosm_root / relative} do "
            f"not match {relative} at commit {commit}. Refusing to record that "
            "commit for dirty, staged, or otherwise different bytes."
        )


def parse_pin_commits(values: Sequence[str]) -> dict[str, str]:
    """Parse ``--microcosm-commit`` values into ``{manifest path or '*': commit}``."""
    commits: dict[str, str] = {}
    for value in values:
        path, separator, commit = value.rpartition("=")
        key = path if separator else "*"
        if not _COMMIT_RE.match(commit):
            raise CatalogueError(
                f"--microcosm-commit must name a 40-hex commit, not {value!r}."
            )
        if key in commits and commits[key] != commit:
            raise CatalogueError(
                f"--microcosm-commit names two commits for {key!r}; pass one."
            )
        commits[key] = commit
    return commits


def pinned_from(release: Release, commit: str, repository: str) -> dict[str, str]:
    """Return the ``pinned_from`` block a consumer_pin registration records."""
    return {
        "repository": repository,
        "path": release.manifest,
        "commit": commit,
    }


def emit(
    resolved: Sequence[ResolvedRelease],
    *,
    root: Path,
    pin_commits: Mapping[str, str],
    consumer_repository: str,
    allow_reissue: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Write hash-only manifests for every registrable non-public release.

    ``pin_commits`` maps each consumer manifest path to the verified commit its
    pins are read from; ``consumer_repository`` is the verified origin identity.
    Returns ``(registrations, blockers)``. A release Microcosm
    pins without a checksum is a blocker, not a registration: no hash is ever
    invented.
    """
    registrations: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    for item in resolved:
        release = item.release
        if release.access == ACCESS_PUBLIC:
            continue
        if release.blocker:
            blockers.append({"release": release.release_id, "reason": release.blocker})
            continue
        filename = item.filename
        checksum = item.sha256
        if not filename or not checksum:
            blockers.append(
                {
                    "release": release.release_id,
                    "reason": (
                        "Microcosm pins this release without a "
                        f"{'filename' if not filename else 'sha256'}; a "
                        "registration needs both. No value is invented."
                    ),
                }
            )
            continue
        report = register_hash_only_artifact(
            source_id=release.source_id,
            package_id=release.package_id,
            year=release.year,
            output_dir=root / release.package_dir,
            filename=filename,
            sha256=checksum,
            licence=release.licence,
            access=release.access,
            vintage=item.vintage,
            hash_source=HASH_SOURCE_CONSUMER_PIN,
            attested_by=consumer_repository,
            pinned_from=pinned_from(
                release, pin_commits[release.manifest], consumer_repository
            ),
            size_bytes=item.size_bytes,
            source_page=release.source_page,
            access_route=release.access_route,
            doi=release.doi,
            study=release.study,
            table=release.table,
            publisher=release.publisher,
            notes=release.notes or HASH_PROVENANCE,
            allow_reissue=allow_reissue,
        )
        registrations.append(report.to_dict())
    return registrations, blockers


def fetch_command(
    item: ResolvedRelease,
    *,
    root: Path,
    r2_bucket: str,
) -> tuple[str, list[str]]:
    """Return ``(command, todos)`` for one public release.

    The command is the exact ``chronicle fetch-artifact`` invocation to run from
    a networked machine. Anything Microcosm does not pin becomes a TODO rather
    than a fabricated value.
    """
    release = item.release
    todos: list[str] = []
    url = item.url
    if url is None:
        url = TODO_PUBLISHER_URL
        todos.append(
            f"{release.release_id}: Microcosm records no publisher URL "
            f"(field {release.url_field!r}); read it off {release.source_page}."
        )
    argv = [
        "uv",
        "run",
        "chronicle",
        "fetch-artifact",
        "--source-id",
        release.source_id,
        "--package-id",
        release.package_id,
        "--year",
        str(release.year),
        "--out-dir",
        str(root / release.package_dir),
        "--url",
        url,
    ]
    if item.filename:
        argv += ["--filename", item.filename]
    if release.source_page:
        argv += ["--source-page", release.source_page]
    vintage = item.vintage
    if not vintage:
        todos.append(
            f"{release.release_id}: Microcosm records no vintage; add "
            "Release.vintage to the catalogue before running this command."
        )
    argv += [
        "--table",
        release.table,
        "--publisher",
        release.publisher,
        "--vintage",
        vintage or "TODO_VINTAGE",
        "--access",
        ACCESS_PUBLIC,
        "--licence",
        release.licence,
        # A public microdata release is archived, but it is still a release:
        # several files share one vintage and no source package parses it.
        "--kind",
        MICRODATA_RELEASE_KIND,
    ]
    # The reviewed identity travels as arguments, never as a comment: the
    # fetch refuses bytes that hash differently before archiving anything.
    if item.sha256 and release.pinned_sha_is_publisher_bytes:
        argv += ["--expected-sha256", item.sha256]
        if item.size_bytes:
            argv += ["--expected-size-bytes", str(item.size_bytes)]
    else:
        argv += ["--expected-sha256", TODO_REVIEWED_SHA256]
        if item.sha256:
            todos.append(
                f"{release.release_id}: Microcosm's pinned sha256 {item.sha256} "
                "is NOT the publisher artifact's checksum. A public release is "
                "archived only against a reviewed checksum its licence evidence "
                "covers; review the publisher bytes and replace "
                f"{TODO_REVIEWED_SHA256} before running this command."
            )
        else:
            todos.append(
                f"{release.release_id}: Microcosm pins no checksum for this "
                "release. A public release is archived only against a reviewed "
                f"checksum; replace {TODO_REVIEWED_SHA256} before running this "
                "command."
            )
    argv += [
        "--licence-evidence-issuer",
        release.licence_evidence_issuer or release.publisher,
        "--licence-evidence-scope",
        release.licence_evidence_scope or "TODO_EVIDENCE_SCOPE",
        "--licence-evidence-url",
        release.licence_evidence_url or TODO_EVIDENCE_URL,
    ]
    if not release.licence_evidence_url:
        todos.append(
            f"{release.release_id}: no durable licence-evidence URL is "
            f"catalogued; replace {TODO_EVIDENCE_URL} with the publisher's "
            "statement that this file is issued under "
            f"{release.licence} before running this command."
        )
    argv += ["--upload-r2", "--r2-bucket", r2_bucket]
    if release.notes:
        todos.append(f"{release.release_id}: {release.notes}")
    return shlex.join(argv), todos


def plan(
    resolved: Sequence[ResolvedRelease],
    *,
    root: Path,
    r2_bucket: str,
) -> tuple[list[str], list[str]]:
    """Return the fetch commands and TODOs for every public release."""
    commands: list[str] = []
    todos: list[str] = []
    for item in resolved:
        if item.release.access != ACCESS_PUBLIC:
            continue
        command, item_todos = fetch_command(item, root=root, r2_bucket=r2_bucket)
        commands.append(command)
        todos.extend(item_todos)
    return commands, todos


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--microcosm-root",
        type=Path,
        default=Path.home() / "PolicyEngine" / "microcosm",
        help="Read-only path to a PolicyEngine/microcosm checkout.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("db/data"),
        help="Chronicle data root that holds the package directories.",
    )
    parser.add_argument(
        "--release",
        action="append",
        default=None,
        help="Limit to these release IDs. Repeatable.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of prose.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit_parser = subparsers.add_parser(
        "emit",
        help="Write hash-only manifests for licensed and restricted releases",
    )
    emit_parser.add_argument(
        "--microcosm-commit",
        action="append",
        default=None,
        metavar="[PATH=]COMMIT",
        help=(
            "Commit the consumer pins are read from, recorded as "
            "pinned_from.commit on every registration. A bare COMMIT applies "
            "to every consumer manifest; PATH=COMMIT (repeatable) names the "
            "commit for one manifest path. Defaults to the last commit that "
            "changed each consumer manifest, read from the checkout's git "
            "history."
        ),
    )
    emit_parser.add_argument(
        "--allow-reissue",
        action="store_true",
        help="Register different bytes alongside an existing pin for a filename.",
    )

    plan_parser = subparsers.add_parser(
        "plan",
        help="Print the fetch commands to run for public releases",
    )
    plan_parser.add_argument(
        "--r2-bucket",
        default=None,
        help=(
            "Raw bucket the fetch should upload to. Defaults to "
            "$CHRONICLE_R2_RAW_BUCKET, else the ledger-era default."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the registration script."""
    args = build_parser().parse_args(argv)
    releases = CATALOGUE
    if args.release:
        wanted = set(args.release)
        releases = tuple(r for r in CATALOGUE if r.release_id in wanted)
        missing = wanted - {r.release_id for r in releases}
        if missing:
            print(f"Unknown release IDs: {sorted(missing)}", file=sys.stderr)
            return 2

    microcosm_root = args.microcosm_root.expanduser()
    try:
        resolved = resolve(microcosm_root, releases)
    except CatalogueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.command == "emit":
        try:
            declared = parse_pin_commits(args.microcosm_commit or ())
        except CatalogueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        # Only a registrable release needs its pin's commit: a public release
        # is fetched, not transcribed, and a blocked one is never registered.
        registrable = sorted(
            {
                item.release.manifest
                for item in resolved
                if item.release.access != ACCESS_PUBLIC and not item.release.blocker
            }
        )
        try:
            snapshots: dict[str, bytes] = {}
            for item in resolved:
                manifest = item.release.manifest
                if manifest not in registrable:
                    continue
                if manifest in snapshots and snapshots[manifest] != item.manifest_bytes:
                    raise CatalogueError(
                        f"Consumer manifest {manifest} changed during resolution."
                    )
                snapshots[manifest] = item.manifest_bytes

            pin_commits: dict[str, str] = {}
            for manifest in registrable:
                commit = (
                    declared.get(manifest)
                    or declared.get("*")
                    or pin_commit(microcosm_root, manifest)
                )
                assert_manifest_matches_commit(
                    microcosm_root,
                    manifest,
                    commit,
                    loaded_bytes=snapshots[manifest],
                )
                pin_commits[manifest] = commit
            consumer_repository = (
                verified_consumer_repository(microcosm_root) if pin_commits else ""
            )
        except CatalogueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        try:
            registrations, blockers = emit(
                resolved,
                root=args.root,
                pin_commits=pin_commits,
                consumer_repository=consumer_repository,
                allow_reissue=args.allow_reissue,
            )
        except HashOnlyRegistrationError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.json:
            print(
                json.dumps(
                    {"registrations": registrations, "blockers": blockers},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            for registration in registrations:
                print(f"registered {registration['registration']}")
            for blocker in blockers:
                print(f"BLOCKED {blocker['release']}: {blocker['reason']}")
            print(
                f"\n{len(registrations)} registration(s), {len(blockers)} blocker(s)."
            )
        return 0

    commands, todos = plan(
        resolved,
        root=args.root,
        r2_bucket=args.r2_bucket or default_r2_raw_bucket(),
    )
    if args.json:
        print(json.dumps({"commands": commands, "todos": todos}, indent=2))
        return 0
    print("# Run from a networked machine with R2 credentials.\n")
    for command in commands:
        print(command + "\n")
    if todos:
        print("# TODO")
        for todo in todos:
            print(f"#   {todo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
