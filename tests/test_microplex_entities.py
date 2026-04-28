"""Tests for linked Microplex US entity frames."""

import pandas as pd

from micro.us.entities import build_microplex_entities, with_household_weights


def test_build_microplex_entities_uses_household_person_tax_unit_links():
    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "tax_unit_id": [10, 10, 20],
            "spm_unit_id": [100, 100, 200],
            "person_seq": [1, 2, 1],
            "age": [40, 38, 12],
            "state_fips": [6, 6, 48],
            "weight": [100.0, 100.0, 200.0],
            "wage_salary_income": [50_000.0, 20_000.0, 0.0],
            "total_person_income": [50_000.0, 20_000.0, 0.0],
        }
    )

    entities = build_microplex_entities(persons)

    assert len(entities.households) == 2
    assert len(entities.persons) == 3
    assert len(entities.tax_units) == 2
    assert entities.households["person_count"].tolist() == [2, 1]
    assert entities.households["tax_unit_count"].tolist() == [1, 1]
    assert entities.tax_units["person_count"].tolist() == [2, 1]
    assert entities.tax_units["wage_income"].tolist() == [70_000.0, 0.0]
    assert entities.tax_units["is_tax_filer"].tolist() == [1, 0]
    assert entities.persons["tax_unit_entity_id"].nunique() == 2
    assert entities.persons["spm_unit_entity_id"].nunique() == 2


def test_with_household_weights_maps_weights_to_linked_entities():
    entities = build_microplex_entities(
        pd.DataFrame(
            {
                "household_id": [1, 1, 2],
                "tax_unit_id": [10, 10, 20],
                "person_seq": [1, 2, 1],
                "age": [40, 38, 55],
                "weight": [100.0, 100.0, 200.0],
                "wage_salary_income": [50_000.0, 20_000.0, 30_000.0],
            }
        )
    )
    households = entities.households.copy()
    households["original_weight"] = households["weight"]
    households["weight"] = [120.0, 180.0]
    households["calibrated_weight"] = households["weight"]
    households["weight_adjustment"] = [1.2, 0.9]

    weighted = with_household_weights(entities, households)

    assert weighted.persons["weight"].tolist() == [120.0, 120.0, 180.0]
    assert weighted.tax_units["weight"].tolist() == [120.0, 180.0]
    assert weighted.tax_units["weight_adjustment"].tolist() == [1.2, 0.9]
