"""Tests for Arch structural normalization helpers."""

import pytest

from arch.targets import DataSource, Jurisdiction, TargetType
from arch.normalization import (
    SourceFact,
    apply_share,
    as_target,
    convert_units,
    target_kwargs,
)


def test_convert_units_records_derivation():
    fact = SourceFact(
        name="snap_households",
        value=22_323,
        period=2023,
        unit="thousands",
        source=DataSource.USDA_SNAP,
        jurisdiction=Jurisdiction.US,
    )

    converted = convert_units(fact, 1000, "count")

    assert converted.value == 22_323_000
    assert converted.unit == "count"
    assert converted.derivation[0].operation == "convert_units"
    assert converted.derivation[0].parameters["factor"] == 1000


def test_apply_share_and_as_target_preserve_provenance():
    total = SourceFact(
        name="aca_marketplace_enrollment",
        value=21_446_150,
        period=2024,
        unit="count",
        source="cms-aca",
        jurisdiction="us",
        source_table="Marketplace OEP Report",
    )

    silver = apply_share(total, 0.54, name="silver_marketplace_enrollment")
    target = as_target(
        silver,
        variable="aca_metal_level_enrollment",
        target_type=TargetType.COUNT,
        stratum_name="US ACA Marketplace Silver Enrollment",
        constraints=(("metal_level", "==", "silver"),),
    )

    assert target.value == pytest.approx(11_580_921)
    assert target.source == DataSource.CMS_ACA
    assert target.jurisdiction == Jurisdiction.US
    assert target.source_table == "Marketplace OEP Report"
    assert target.constraints == (("metal_level", "==", "silver"),)
    assert "apply_share" in target.notes


def test_target_kwargs_maps_blueprint_to_target_fields():
    fact = SourceFact(
        name="snap_households",
        value=22_323,
        period=2023,
        unit="thousands",
        source=DataSource.USDA_SNAP,
        jurisdiction=Jurisdiction.US,
    )
    target = as_target(
        convert_units(fact, 1000, "count"),
        target_type=TargetType.COUNT,
        stratum_name="US SNAP Households",
    )

    kwargs = target_kwargs(target, stratum_id=7)

    assert kwargs["stratum_id"] == 7
    assert kwargs["variable"] == "snap_households"
    assert kwargs["value"] == 22_323_000
    assert kwargs["source"] == DataSource.USDA_SNAP
    assert "convert_units" in kwargs["notes"]
