"""Tests for first-class Arch source and fact models."""

from arch.facts import DerivationStep, SourceFact
from arch.sources import SourceFile, SourceReference
from arch.targets import DataSource, Jurisdiction
from arch.normalization import convert_units


def test_source_reference_and_file_capture_external_provenance():
    source = SourceReference(
        source=DataSource.USDA_SNAP,
        institution="USDA FNS",
        dataset="SNAP national summary",
        jurisdiction=Jurisdiction.US,
        url="https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap",
        update_frequency="annual",
    )
    source_file = SourceFile(
        source=source,
        r2_key="sources/usda/snap/2023/national_summary.xlsx",
        checksum="sha256:test",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert source_file.source is source
    assert source_file.r2_key.startswith("sources/usda/snap")
    assert source_file.source.source == DataSource.USDA_SNAP


def test_source_fact_can_reference_source_file_and_derivation():
    source = SourceReference(
        source="usda-snap",
        institution="USDA FNS",
        dataset="SNAP national summary",
    )
    source_file = SourceFile(source=source, r2_key="sources/usda/snap/2023.csv")
    fact = SourceFact(
        name="snap_households",
        value=22_323,
        period=2023,
        unit="thousands",
        source=DataSource.USDA_SNAP,
        jurisdiction=Jurisdiction.US,
        source_file=source_file,
    )

    converted = convert_units(fact, 1000, "count")

    assert converted.source_file is source_file
    assert converted.value == 22_323_000
    assert converted.derivation == (
        DerivationStep(
            operation="convert_units",
            parameters={
                "factor": 1000,
                "input_unit": "thousands",
                "output_unit": "count",
            },
        ),
    )
