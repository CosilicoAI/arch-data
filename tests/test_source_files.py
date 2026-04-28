"""Tests for persisted source-file ingestion."""

from __future__ import annotations

import json
import zipfile

from sqlmodel import Session, select

from db.pe_source_inventory import pe_source_specs
from db.pe_source_inventory import UK_TARGET_URL_FILES
from db.schema import (
    Jurisdiction,
    SourceArtifact,
    SourceRow,
    SourceTable,
    get_engine,
    init_db,
)
from db import source_files
from db.source_files import SourceArtifactSpec, ingest_source_artifact


def test_ingest_csv_source_artifact(tmp_path):
    source_path = tmp_path / "sample.csv"
    source_path.write_text("state,value\nCA,1\nNY,2\n", encoding="utf-8")
    db_path = tmp_path / "sources.db"
    init_db(db_path)

    spec = SourceArtifactSpec(
        slug="test/sample",
        path=source_path,
        origin_project="policyengine-us-data",
        pipeline="database",
        jurisdiction=Jurisdiction.US,
        source_id="test-source",
    )

    with Session(get_engine(db_path)) as session:
        result = ingest_source_artifact(session, spec)
        session.commit()
        artifact = session.exec(select(SourceArtifact)).one()
        table = session.exec(select(SourceTable)).one()
        rows = session.exec(select(SourceRow).order_by(SourceRow.row_number)).all()

    assert result.row_count == 2
    assert artifact.sha256
    assert table.column_count == 2
    assert json.loads(rows[0].values_json) == {"state": "CA", "value": "1"}


def test_ingest_zip_source_artifact(tmp_path):
    source_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("nested/data.csv", "name,count\nalpha,3\n")
    db_path = tmp_path / "sources.db"
    init_db(db_path)

    spec = SourceArtifactSpec(
        slug="test/bundle",
        path=source_path,
        origin_project="policyengine-us-data",
        pipeline="database",
        jurisdiction=Jurisdiction.US,
        source_id="test-source",
    )

    with Session(get_engine(db_path)) as session:
        result = ingest_source_artifact(session, spec)
        session.commit()
        table = session.exec(select(SourceTable)).one()
        row = session.exec(select(SourceRow)).one()

    assert result.table_count == 1
    assert "nested/data.csv" in table.name
    assert json.loads(row.values_json) == {"name": "alpha", "count": "3"}


def test_ingest_url_source_artifact(tmp_path, monkeypatch):
    def fake_fetch_url(_url):
        return b"name,count\nalpha,3\n", "text/csv", "https://example.test/source.csv"

    monkeypatch.setattr(source_files, "_fetch_url", fake_fetch_url)
    db_path = tmp_path / "sources.db"
    init_db(db_path)

    spec = SourceArtifactSpec(
        slug="test/url",
        source_url="https://example.test/source.csv",
        filename="source.csv",
        origin_project="policyengine-uk-data",
        pipeline="target-registry-live-sources",
        jurisdiction=Jurisdiction.UK,
        source_id="test-source",
    )

    with Session(get_engine(db_path)) as session:
        result = ingest_source_artifact(session, spec)
        session.commit()
        artifact = session.exec(select(SourceArtifact)).one()
        row = session.exec(select(SourceRow)).one()

    assert result.row_count == 1
    assert artifact.local_path is None
    assert artifact.source_url == "https://example.test/source.csv"
    assert json.loads(row.values_json) == {"name": "alpha", "count": "3"}


def test_ingest_url_source_artifact_records_fetch_error(tmp_path, monkeypatch):
    def fake_fetch_url(_url):
        raise RuntimeError("blocked")

    monkeypatch.setattr(source_files, "_fetch_url", fake_fetch_url)
    db_path = tmp_path / "sources.db"
    init_db(db_path)

    spec = SourceArtifactSpec(
        slug="test/url-error",
        source_url="https://example.test/source.html",
        filename="source.html",
        origin_project="policyengine-uk-data",
        pipeline="target-registry-live-sources",
        jurisdiction=Jurisdiction.UK,
        source_id="test-source",
    )

    with Session(get_engine(db_path)) as session:
        result = ingest_source_artifact(session, spec)
        session.commit()
        artifact = session.exec(select(SourceArtifact)).one()
        rows = session.exec(select(SourceRow).order_by(SourceRow.row_number)).all()

    lines = "\n".join(json.loads(row.values_json)["line"] for row in rows)
    assert result.row_count == 5
    assert artifact.content_type == "text/plain"
    assert artifact.source_url == "https://example.test/source.html"
    assert "RuntimeError" in lines


def test_ingest_pdf_source_artifact(tmp_path):
    source_path = tmp_path / "source.pdf"
    content = b"%PDF-1.4\nsource bytes\n%%EOF"
    source_path.write_bytes(content)
    db_path = tmp_path / "sources.db"
    init_db(db_path)

    spec = SourceArtifactSpec(
        slug="test/pdf",
        path=source_path,
        origin_project="policyengine-uk-data",
        pipeline="target-registry-live-sources",
        jurisdiction=Jurisdiction.UK,
        source_id="test-source",
    )

    with Session(get_engine(db_path)) as session:
        result = ingest_source_artifact(session, spec)
        session.commit()
        row = session.exec(select(SourceRow)).one()

    payload = json.loads(row.values_json)
    assert result.row_count == 1
    assert payload["filename"] == "source.pdf"
    assert payload["size_bytes"] == len(content)
    assert payload["content_base64"]


def test_pe_source_inventory_finds_both_pipeline_roots(tmp_path):
    pe_us = tmp_path / "policyengine-us-data"
    raw_inputs = pe_us / "policyengine_us_data" / "storage" / "calibration" / "raw_inputs"
    target_inputs = (
        pe_us / "policyengine_us_data" / "storage" / "calibration_targets"
    )
    raw_inputs.mkdir(parents=True)
    target_inputs.mkdir(parents=True)
    (raw_inputs / "irs_soi_sample.csv").write_text("x\n1\n", encoding="utf-8")
    (target_inputs / "snap_state.csv").write_text("x\n1\n", encoding="utf-8")

    specs = pe_source_specs(pe_us_root=pe_us, include_uk=False)

    assert [spec.pipeline for spec in specs] == ["database", "legacy-loss-targets"]
    assert {spec.source_id for spec in specs} == {"irs-soi", "usda-snap"}


def test_pe_source_inventory_includes_long_term_ssa_inputs(tmp_path):
    pe_us = tmp_path / "policyengine-us-data"
    storage = pe_us / "policyengine_us_data" / "storage"
    long_term = storage / "long_term_target_sources"
    long_term.mkdir(parents=True)
    (long_term / "sources.json").write_text("{}", encoding="utf-8")
    (long_term / "trustees_2025_current_law.csv").write_text(
        "year,value\n2025,1\n", encoding="utf-8"
    )
    (storage / "SSPopJul_TR2024.csv").write_text(
        "Year,Age,Total\n2025,0,1\n", encoding="utf-8"
    )
    (storage / "social_security_aux.csv").write_text(
        "year,value\n2025,1\n", encoding="utf-8"
    )

    specs = pe_source_specs(pe_us_root=pe_us, include_uk=False)
    slugs = {spec.slug for spec in specs}

    assert any("sspopjul_tr2024.csv" in slug for slug in slugs)
    assert any("social_security_aux.csv" in slug for slug in slugs)
    assert any(spec.pipeline == "long-term-target-references" for spec in specs)


def test_pe_uk_inventory_includes_registry_config_and_reference_pages(tmp_path):
    pe_uk = tmp_path / "policyengine-uk-data"
    targets = pe_uk / "policyengine_uk_data" / "targets"
    targets.mkdir(parents=True)
    (targets / "sources.yaml").write_text("dwp: {}\n", encoding="utf-8")

    specs = pe_source_specs(pe_uk_root=pe_uk, include_us=False)
    pipelines = {spec.pipeline for spec in specs}
    filenames = {spec.filename for spec in specs if spec.filename}

    assert "target-registry-config" in pipelines
    assert "target-registry-live-sources" in pipelines
    assert len(filenames) >= len(UK_TARGET_URL_FILES)
    assert "dwp_stat_xplore.html" in filenames
    assert "uk_government_2025_spp_review.pdf" in filenames
