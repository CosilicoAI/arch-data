"""Tests for the Arch namespace."""

from arch.client import get_supabase_client
from arch.microdata import get_table_name, query_cps_asec
from arch.normalization import convert_units
from arch.targets import (
    Target,
    TargetSpec,
    TargetType,
    build_microplex_constraints,
    get_targets,
    query_targets,
)
from db.schema import Target as DbTarget
from db.supabase_client import (
    ARCH_SCHEMA,
    MICRODATA_SCHEMA,
    TARGETS_SCHEMA,
    query_targets as db_query_targets,
)


def test_arch_targets_reexport_schema_objects():
    assert Target is DbTarget
    assert TargetType.COUNT.value == "count"
    assert TargetSpec.__name__ == "TargetSpec"


def test_arch_targets_reexport_client_helpers():
    assert query_targets is db_query_targets
    assert callable(get_targets)


def test_arch_targets_reexport_microplex_adapters():
    assert callable(build_microplex_constraints)


def test_arch_microdata_reexport_client_helpers():
    assert get_table_name("us", "census", "cps_asec", 2024, "person") == (
        "us_census_cps_asec_2024_person"
    )
    assert callable(query_cps_asec)


def test_arch_client_reexports_supabase_client():
    assert callable(get_supabase_client)


def test_arch_supabase_schema_boundaries_are_defaulted():
    assert ARCH_SCHEMA == "arch"
    assert MICRODATA_SCHEMA == "microdata"
    assert TARGETS_SCHEMA == "targets"


def test_arch_normalization_exports_helpers():
    assert callable(convert_units)
