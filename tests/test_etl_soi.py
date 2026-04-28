from db.etl_soi import (
    available_soi_table_1_4_years,
    available_soi_years,
    load_soi_table_1_1_data,
    load_soi_table_1_4_data,
)


def test_available_soi_years_include_latest_table_1_1_sources():
    assert available_soi_years() == [2020, 2021, 2022, 2023]


def test_load_soi_table_1_1_data_reads_packaged_2023_source():
    data = load_soi_table_1_1_data(2023)

    assert data["source_url"] == "https://www.irs.gov/pub/irs-soi/23in11si.xls"
    assert data["total_returns"] == 160_602_107
    assert data["total_agi"] == 15_286_017_359_000
    assert data["total_income_tax"] == 2_147_909_818_000
    assert data["returns_by_agi_bracket"]["100k_to_200k"] == 27_602_755
    assert data["agi_by_bracket"]["100k_to_200k"] == 3_818_295_141_000
    assert data["income_tax_by_bracket"]["100k_to_200k"] == 409_532_689_000


def test_available_soi_years_include_latest_table_1_4_sources():
    assert available_soi_table_1_4_years() == [2021, 2022, 2023]


def test_load_soi_table_1_4_data_reads_packaged_2023_wage_source():
    data = load_soi_table_1_4_data(2023)

    assert data["source_url"] == "https://www.irs.gov/pub/irs-soi/23in14ar.xls"
    assert data["total_employment_income_returns"] == 128_591_050
    assert data["total_employment_income"] == 10_204_095_705_000
    assert data["employment_income_returns_by_agi_bracket"]["100k_to_200k"] == 23_193_910
    assert data["employment_income_by_agi_bracket"]["100k_to_200k"] == 2_774_550_975_000
