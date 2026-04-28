"""Tests for persisted source-file ingestion."""

from __future__ import annotations

import json
import zipfile

from sqlmodel import Session, select

from db.pe_source_inventory import pe_source_specs
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
