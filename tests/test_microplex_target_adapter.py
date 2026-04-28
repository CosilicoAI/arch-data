"""Tests for Arch target adapters used by Microplex."""

import numpy as np
import pandas as pd
from sqlmodel import Session

from arch.pipelines.microplex import build_constraints_from_target_specs
from arch.targets import (
    DataSource,
    Jurisdiction,
    Stratum,
    StratumConstraint,
    Target,
    TargetSpec,
    TargetType,
    build_hierarchical_microplex_constraints,
    build_microplex_constraints,
    constraints_to_ipf_dicts,
    init_db,
    load_microplex_targets,
)


def _insert_simple_target(db_path):
    engine = init_db(db_path)
    with Session(engine) as session:
        constraints = [("is_tax_filer", "==", "1")]
        stratum = Stratum(
            name="All tax filers",
            description="All tax filers",
            jurisdiction=Jurisdiction.US,
            definition_hash=Stratum.compute_hash(constraints, Jurisdiction.US),
        )
        session.add(stratum)
        session.flush()

        session.add(
            StratumConstraint(
                stratum_id=stratum.id,
                variable="is_tax_filer",
                operator="==",
                value="1",
            )
        )
        session.add(
            Target(
                stratum_id=stratum.id,
                variable="tax_unit_count",
                period=2024,
                value=1000,
                target_type=TargetType.COUNT,
                source=DataSource.IRS_SOI,
            )
        )
        session.commit()


def test_load_microplex_targets_reads_arch_db(tmp_path):
    db_path = tmp_path / "targets.db"
    _insert_simple_target(db_path)

    targets = load_microplex_targets(db_path=db_path, jurisdiction="us", year=2024)

    assert len(targets) == 1
    assert targets[0] == TargetSpec(
        variable="tax_unit_count",
        value=1000,
        target_type=TargetType.COUNT,
        constraints=[("is_tax_filer", "==", "1")],
        source=DataSource.IRS_SOI,
        period=2024,
        stratum_name="All tax filers",
    )


def test_build_microplex_constraints_from_target_specs():
    microdata = pd.DataFrame(
        {
            "is_tax_filer": [1, 0, 1],
            "adjusted_gross_income": [10_000, 20_000, 30_000],
        }
    )
    targets = [
        TargetSpec(
            variable="tax_unit_count",
            value=2,
            target_type=TargetType.COUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
        ),
        TargetSpec(
            variable="adjusted_gross_income",
            value=40_000,
            target_type=TargetType.AMOUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
        ),
    ]

    constraints = build_microplex_constraints(microdata, targets=targets)

    assert len(constraints) == 2
    np.testing.assert_array_equal(constraints[0].indicator, np.array([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(
        constraints[1].indicator,
        np.array([10_000.0, 0.0, 30_000.0]),
    )


def test_build_hierarchical_microplex_constraints_aggregates_to_households():
    households = pd.DataFrame({"household_id": [1, 2]})
    people = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "age": [10, 40, 70],
        }
    )
    targets = [
        TargetSpec(
            variable="person_count",
            value=2,
            target_type=TargetType.COUNT,
            constraints=[("age", ">=", "18")],
            source=DataSource.CENSUS_ACS,
            period=2024,
        )
    ]

    constraints = build_hierarchical_microplex_constraints(
        households,
        people,
        targets=targets,
    )

    assert len(constraints) == 1
    np.testing.assert_array_equal(constraints[0].indicator, np.array([1, 1]))


def test_constraints_to_ipf_dicts_preserves_values():
    microdata = pd.DataFrame({"is_tax_filer": [1, 0, 1]})
    targets = [
        TargetSpec(
            variable="tax_unit_count",
            value=2,
            target_type=TargetType.COUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
            stratum_name="filers",
        )
    ]
    constraints = build_microplex_constraints(microdata, targets=targets)

    dicts = constraints_to_ipf_dicts(constraints)

    assert dicts[0]["target_value"] == 2
    assert dicts[0]["variable"] == "tax_unit_count"
    assert dicts[0]["target_type"] == "count"
    assert dicts[0]["stratum"] == "filers"
    assert dicts[0]["n_obs"] == 2
    np.testing.assert_array_equal(dicts[0]["indicator"], np.array([1.0, 0.0, 1.0]))


def test_legacy_microplex_pipeline_accepts_target_specs():
    microdata = pd.DataFrame(
        {
            "adjusted_gross_income": [10_000, 20_000, 60_000],
        }
    )
    targets = [
        TargetSpec(
            variable="tax_unit_count",
            value=1,
            target_type=TargetType.COUNT,
            constraints=[
                ("adjusted_gross_income", ">=", "50000"),
                ("adjusted_gross_income", "<", "75000"),
            ],
            source=DataSource.IRS_SOI,
            period=2024,
            stratum_name="AGI 50k to 75k",
        )
    ]

    constraints = build_constraints_from_target_specs(microdata, targets, min_obs=1)

    assert len(constraints) == 1
    assert constraints[0]["target_value"] == 1
    assert constraints[0]["n_obs"] == 1
    np.testing.assert_array_equal(
        constraints[0]["indicator"],
        np.array([0.0, 0.0, 1.0]),
    )
