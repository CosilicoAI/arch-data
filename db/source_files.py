"""Generic source-file ingestion for Arch source artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import pandas as pd
from sqlalchemy import delete
from sqlmodel import Session, select

from .schema import (
    Jurisdiction,
    SourceArtifact,
    SourceColumn,
    SourceRow,
    SourceTable,
)

SUPPORTED_SUFFIXES = {
    ".csv",
    ".gz",
    ".html",
    ".htm",
    ".json",
    ".ods",
    ".txt",
    ".xlsx",
    ".zip",
}


@dataclass(frozen=True)
class SourceArtifactSpec:
    """A source artifact to parse into the Arch source-file tables."""

    slug: str
    origin_project: str
    pipeline: str
    jurisdiction: Jurisdiction
    source_id: str
    path: Path | None = None
    source_name: str | None = None
    source_url: str | None = None
    filename: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.path is None and self.source_url is None:
            raise ValueError("SourceArtifactSpec requires either path or source_url")


@dataclass(frozen=True)
class ParsedSourceTable:
    """A parsed table from a source artifact."""

    name: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class IngestResult:
    """Counts from one ingested source artifact."""

    slug: str
    table_count: int
    row_count: int


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def _fetch_url(url: str) -> tuple[bytes, str | None, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "cosilico-arch/0.1",
            "Accept": "*/*",
        },
    )
    with urlopen(request, timeout=120) as response:
        content = response.read()
        content_type = response.headers.get_content_type()
        final_url = response.geturl()
    return content, content_type, final_url


def _filename_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    name = unquote(Path(parsed.path).name)
    return name or None


def _suffix_for_content_type(content_type: str | None) -> str:
    if content_type is None:
        return ".txt"
    content_type = content_type.lower().split(";")[0].strip()
    if content_type in {"application/json", "text/json"}:
        return ".json"
    if content_type in {"text/csv", "application/csv"}:
        return ".csv"
    if content_type in {"text/html", "application/xhtml+xml"}:
        return ".html"
    if content_type == "application/zip":
        return ".zip"
    if content_type in {
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/x-vnd.oasis.opendocument.spreadsheet",
    }:
        return ".ods"
    if content_type in {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    }:
        return ".xlsx"
    return ".txt"


def _artifact_name(
    spec: SourceArtifactSpec,
    content_type: str | None,
    final_url: str | None,
) -> str:
    if spec.filename:
        return spec.filename
    if spec.path is not None:
        return spec.path.name
    name = _filename_from_url(final_url or spec.source_url or "")
    if not name:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.slug.split("/")[-1]).strip("_")
    if not Path(name).suffix:
        name = f"{name}{_suffix_for_content_type(content_type)}"
    return name


def _read_spec_content(spec: SourceArtifactSpec) -> tuple[bytes, str, str | None]:
    if spec.path is not None:
        return _read_bytes(spec.path), spec.path.name, mimetypes.guess_type(spec.path.name)[0]
    if spec.source_url is None:
        raise ValueError(f"Source artifact {spec.slug} has no path or URL")
    content, content_type, final_url = _fetch_url(spec.source_url)
    return content, _artifact_name(spec, content_type, final_url), content_type


def _stable_slug(*parts: str) -> str:
    raw = "/".join(part.strip("/").replace(" ", "_") for part in parts if part)
    return (
        raw.lower()
        .replace("__", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def dataframe_to_records(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Convert a DataFrame to source-row JSON payloads."""
    work = df.copy()
    work = work.dropna(axis=0, how="all").dropna(axis=1, how="all")
    work.columns = [str(col) for col in work.columns]
    work = work.astype(object).where(pd.notna(work), None)
    records = work.to_dict(orient="records")
    payloads = [
        json.dumps(record, ensure_ascii=True, default=str, separators=(",", ":"))
        for record in records
    ]
    return list(work.columns), payloads


def _parse_csv(content: bytes, name: str, suffix: str) -> list[ParsedSourceTable]:
    compression = "gzip" if suffix == ".gz" or name.endswith(".csv.gz") else None
    df = pd.read_csv(
        io.BytesIO(content),
        dtype=str,
        comment="#",
        compression=compression,
        low_memory=False,
    )
    return [ParsedSourceTable(name=name, frame=df)]


def _parse_text(content: bytes, name: str) -> list[ParsedSourceTable]:
    try:
        df = pd.read_csv(io.BytesIO(content), dtype=str, low_memory=False)
    except Exception:
        text = content.decode("utf-8", errors="replace")
        df = pd.DataFrame(
            {
                "line_number": range(1, len(text.splitlines()) + 1),
                "line": text.splitlines(),
            }
        )
    return [ParsedSourceTable(name=name, frame=df)]


def _parse_json(content: bytes, name: str) -> list[ParsedSourceTable]:
    data = json.loads(content.decode("utf-8"))
    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], list)
        and all(isinstance(item, str) for item in data[0])
    ):
        df = pd.DataFrame(data[1:], columns=data[0])
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = {"key": key}
                for child_key, child_value in value.items():
                    if isinstance(child_value, (dict, list)):
                        row[child_key] = json.dumps(
                            child_value, ensure_ascii=True, default=str
                        )
                    else:
                        row[child_key] = child_value
                rows.append(row)
            else:
                rows.append(
                    {
                        "key": key,
                        "value": json.dumps(value, ensure_ascii=True, default=str),
                    }
                )
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            [{"value": json.dumps(data, ensure_ascii=True, default=str)}]
        )
    return [ParsedSourceTable(name=name, frame=df)]


def _parse_excel(content: bytes, name: str, suffix: str) -> list[ParsedSourceTable]:
    engine = "odf" if suffix == ".ods" else "openpyxl"
    xls = pd.ExcelFile(io.BytesIO(content), engine=engine)
    tables = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=object)
        tables.append(ParsedSourceTable(name=f"{name}:{sheet_name}", frame=df))
    return tables


def parse_source_artifact(path: Path, content: bytes | None = None) -> list[ParsedSourceTable]:
    """Parse a supported source artifact into one or more source tables."""
    if content is None:
        content = _read_bytes(path)
    name = path.name
    suffix = path.suffix.lower()

    if name.endswith(".csv.gz"):
        return _parse_csv(content, name, ".gz")
    if suffix == ".csv":
        return _parse_csv(content, name, suffix)
    if suffix == ".txt":
        return _parse_text(content, name)
    if suffix in {".html", ".htm"}:
        return _parse_text(content, name)
    if suffix == ".json":
        return _parse_json(content, name)
    if suffix in {".xlsx", ".ods"}:
        return _parse_excel(content, name, suffix)
    if suffix == ".zip":
        tables = []
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for member in archive.namelist():
                member_path = Path(member)
                if member.endswith("/") or member_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                member_content = archive.read(member)
                for table in parse_source_artifact(member_path, member_content):
                    table_name = table.name
                    if table_name.startswith(member_path.name):
                        table_name = table_name.replace(member_path.name, member, 1)
                    tables.append(
                        ParsedSourceTable(
                            name=f"{name}:{table_name}",
                            frame=table.frame,
                        )
                    )
        return tables

    raise ValueError(f"Unsupported source artifact format: {path}")


def _delete_artifact(session: Session, artifact_id: int) -> None:
    table_ids = session.exec(
        select(SourceTable.id).where(SourceTable.artifact_id == artifact_id)
    ).all()
    if table_ids:
        session.exec(delete(SourceRow).where(SourceRow.table_id.in_(table_ids)))
        session.exec(delete(SourceColumn).where(SourceColumn.table_id.in_(table_ids)))
        session.exec(delete(SourceTable).where(SourceTable.id.in_(table_ids)))
    session.exec(delete(SourceArtifact).where(SourceArtifact.id == artifact_id))
    session.flush()


def ingest_source_artifact(session: Session, spec: SourceArtifactSpec) -> IngestResult:
    """Parse and persist one source artifact, replacing any prior load by slug."""
    content, artifact_name, content_type = _read_spec_content(spec)
    checksum = _sha256(content)

    existing = session.exec(
        select(SourceArtifact).where(SourceArtifact.slug == spec.slug)
    ).first()
    if existing is not None and existing.id is not None:
        _delete_artifact(session, existing.id)

    artifact = SourceArtifact(
        slug=spec.slug,
        origin_project=spec.origin_project,
        pipeline=spec.pipeline,
        jurisdiction=spec.jurisdiction,
        source_id=spec.source_id,
        source_name=spec.source_name,
        local_path=str(spec.path) if spec.path is not None else None,
        source_url=spec.source_url,
        content_type=content_type,
        size_bytes=len(content),
        sha256=checksum,
        notes=spec.notes,
    )
    session.add(artifact)
    session.flush()

    table_count = 0
    row_count = 0
    for parsed in parse_source_artifact(Path(artifact_name), content):
        columns, row_payloads = dataframe_to_records(parsed.frame)
        table = SourceTable(
            artifact_id=artifact.id,
            name=parsed.name,
            row_count=len(row_payloads),
            column_count=len(columns),
        )
        session.add(table)
        session.flush()

        session.bulk_insert_mappings(
            SourceColumn,
            [
                {"table_id": table.id, "position": i, "name": column}
                for i, column in enumerate(columns)
            ],
        )
        row_batch_size = 5_000
        for start in range(0, len(row_payloads), row_batch_size):
            session.bulk_insert_mappings(
                SourceRow,
                [
                    {
                        "table_id": table.id,
                        "row_number": i,
                        "values_json": payload,
                    }
                    for i, payload in enumerate(
                        row_payloads[start : start + row_batch_size],
                        start=start,
                    )
                ],
            )
        table_count += 1
        row_count += len(row_payloads)

    return IngestResult(slug=spec.slug, table_count=table_count, row_count=row_count)


def ingest_source_artifacts(
    session: Session, specs: Iterable[SourceArtifactSpec]
) -> list[IngestResult]:
    """Parse and persist multiple source artifacts."""
    results = []
    for spec in specs:
        results.append(ingest_source_artifact(session, spec))
        session.commit()
    return results


def prune_source_artifacts(session: Session, specs: Iterable[SourceArtifactSpec]) -> int:
    """Delete artifacts in the same inventory scope that are no longer expected."""
    specs = list(specs)
    expected_slugs = {spec.slug for spec in specs}
    scope = {(spec.origin_project, spec.pipeline) for spec in specs}
    removed = 0

    for origin_project, pipeline in scope:
        artifacts = session.exec(
            select(SourceArtifact).where(
                SourceArtifact.origin_project == origin_project,
                SourceArtifact.pipeline == pipeline,
            )
        ).all()
        for artifact in artifacts:
            if artifact.slug not in expected_slugs and artifact.id is not None:
                _delete_artifact(session, artifact.id)
                removed += 1
    session.commit()
    return removed


def make_slug(origin_project: str, pipeline: str, root: Path, path: Path) -> str:
    """Build a stable artifact slug from a path relative to its source root."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path.name
    return _stable_slug(origin_project, pipeline, str(rel))


def make_url_slug(origin_project: str, pipeline: str, filename: str) -> str:
    """Build a stable artifact slug for a URL-backed source artifact."""
    return _stable_slug(origin_project, pipeline, filename)
