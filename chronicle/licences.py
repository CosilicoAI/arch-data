"""Redistributable licence terms Chronicle may archive microdata bytes under.

Being downloadable is not a licence, and a licence *name* on a manifest entry
is not evidence that a particular file was issued under it. Chronicle archives
a microdata release's bytes only when the entry's ``licence`` is one of the
terms below and the entry carries ``licence_evidence`` binding the artifact to
that term (``docs/adr-chronicle-raw-microdata-identity.md``). This module is
the allowlist, kept in code so every term carries the evidence for why
redistribution is permitted.

Adding a term is a reviewed code change: give it a stable identifier (SPDX
where one exists), the legal basis, and a durable URL to the terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

__all__ = [
    "LICENCE_EVIDENCE_FIELDS",
    "REDISTRIBUTABLE_LICENCES",
    "RedistributableLicence",
    "is_durable_url",
    "is_redistributable_licence",
    "licence_evidence_errors",
]


@dataclass(frozen=True)
class RedistributableLicence:
    """One term under which Chronicle may hold and re-serve publisher bytes."""

    identifier: str
    name: str
    basis: str
    evidence_url: str


REDISTRIBUTABLE_LICENCES: Mapping[str, RedistributableLicence] = MappingProxyType(
    {
        "US-Government-Work": RedistributableLicence(
            identifier="US-Government-Work",
            name="Work of the United States Government",
            basis=(
                "17 U.S.C. §105: copyright protection is not available for any "
                "work of the United States Government, so a federal statistical "
                "agency's public-use file may be copied and redistributed."
            ),
            evidence_url=(
                "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-"
                "title17-section105"
            ),
        ),
        "OGL-UK-3.0": RedistributableLicence(
            identifier="OGL-UK-3.0",
            name="Open Government Licence v3.0",
            basis=(
                "The licence grants a worldwide, royalty-free, perpetual, "
                "non-exclusive licence to copy, publish, distribute and transmit "
                "the information, subject to attribution."
            ),
            evidence_url=(
                "https://www.nationalarchives.gov.uk/doc/open-government-licence/"
                "version/3/"
            ),
        ),
        "CC0-1.0": RedistributableLicence(
            identifier="CC0-1.0",
            name="Creative Commons CC0 1.0 Universal",
            basis=(
                "The affirmer waives all copyright and related rights, so the "
                "work may be copied and redistributed without restriction."
            ),
            evidence_url="https://creativecommons.org/publicdomain/zero/1.0/legalcode",
        ),
        "CC-BY-4.0": RedistributableLicence(
            identifier="CC-BY-4.0",
            name="Creative Commons Attribution 4.0 International",
            basis=(
                "Section 2(a)(1) grants a worldwide, royalty-free, non-exclusive "
                "licence to reproduce and share the licensed material, subject to "
                "attribution."
            ),
            evidence_url="https://creativecommons.org/licenses/by/4.0/legalcode",
        ),
    }
)

#: Fields a ``licence_evidence`` block must carry to bind an artifact to a term.
LICENCE_EVIDENCE_FIELDS: tuple[str, ...] = (
    "issuer",
    "licence",
    "scope",
    "url",
    "sha256",
)


def is_redistributable_licence(licence: Any) -> bool:
    """Whether ``licence`` names a term on the allowlist."""
    return isinstance(licence, str) and licence.strip() in REDISTRIBUTABLE_LICENCES


def licence_evidence_errors(
    evidence: Any,
    *,
    licence: Any,
    sha256: Any,
) -> list[str]:
    """Return error codes for a ``licence_evidence`` block.

    The block binds one artifact to one allowlisted term: its ``licence`` must
    be the entry's own (allowlisted) licence, its ``sha256`` the entry's own
    checksum, and its ``url`` a durable http(s) location of the evidence.
    """
    if evidence is None:
        return ["missing_licence_evidence"]
    if not isinstance(evidence, Mapping):
        return ["malformed_licence_evidence"]
    errors: list[str] = []
    for field in LICENCE_EVIDENCE_FIELDS:
        value = evidence.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"licence_evidence_missing_field:{field}")
    if errors:
        return errors
    if evidence["licence"].strip() != str(licence).strip():
        errors.append("licence_evidence_licence_mismatch")
    if not is_redistributable_licence(evidence["licence"]):
        errors.append(f"licence_not_redistributable:{evidence['licence'].strip()}")
    if evidence["sha256"].strip() != str(sha256 or "").strip():
        errors.append("licence_evidence_sha256_mismatch")
    if not is_durable_url(evidence["url"]):
        errors.append("licence_evidence_url_not_durable")
    return errors


def is_durable_url(value: Any) -> bool:
    """Whether ``value`` is an http(s) URL with a host and no whitespace.

    A bare scheme, a padded string, or a URL with a space is not a location
    a reviewer can follow back to the publisher's statement.
    """
    if not isinstance(value, str) or value != value.strip() or not value:
        return False
    if any(character.isspace() for character in value):
        return False
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)
