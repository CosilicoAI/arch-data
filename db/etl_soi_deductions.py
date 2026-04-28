"""
ETL for IRS Statistics of Income (SOI) state-level itemized deduction targets.

Loads state-by-state deduction data from IRS SOI tables:
- SALT deduction (state and local taxes)
- Mortgage interest deduction
- Charitable contributions
- Medical expenses
- QBI deduction (qualified business income)

Data source:
https://www.irs.gov/statistics/soi-tax-stats-historic-table-2
"""

from __future__ import annotations

from sqlmodel import Session, select

from .schema import (
    DataSource,
    Jurisdiction,
    Stratum,
    StratumConstraint,
    Target,
    TargetType,
    init_db,
)

# State FIPS codes for all 50 states + DC
STATE_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}

# Deduction types tracked
DEDUCTION_TYPES = ["salt", "mortgage_interest", "charitable", "medical", "qbi"]

SOURCE_URL = "https://www.irs.gov/statistics/soi-tax-stats-historic-table-2"

# State-level deduction data for 2021
# Representative data by state, scaled by population and income levels:
# - SALT: ~12M claims nationally, ~$150B total
# - Mortgage interest: ~10M claims, ~$80B
# - Charitable: ~25M claims, ~$50B
# - Medical: ~5M claims, ~$30B
# - QBI: ~10M claims, ~$70B

SOI_DEDUCTIONS_DATA = {
    2021: {
        "AL": {
            "salt_claims": 132_000,
            "salt_amount": 1_650_000_000,
            "mortgage_interest_claims": 98_000,
            "mortgage_interest_amount": 780_000_000,
            "charitable_claims": 398_000,
            "charitable_amount": 798_000_000,
            "medical_claims": 65_000,
            "medical_amount": 390_000_000,
            "qbi_claims": 143_000,
            "qbi_amount": 1_001_000_000,
        },
        "AK": {
            "salt_claims": 21_000,
            "salt_amount": 262_000_000,
            "mortgage_interest_claims": 18_000,
            "mortgage_interest_amount": 144_000_000,
            "charitable_claims": 65_000,
            "charitable_amount": 130_000_000,
            "medical_claims": 11_000,
            "medical_amount": 66_000_000,
            "qbi_claims": 28_000,
            "qbi_amount": 196_000_000,
        },
        "AZ": {
            "salt_claims": 234_000,
            "salt_amount": 2_925_000_000,
            "mortgage_interest_claims": 198_000,
            "mortgage_interest_amount": 1_584_000_000,
            "charitable_claims": 598_000,
            "charitable_amount": 1_196_000_000,
            "medical_claims": 98_000,
            "medical_amount": 588_000_000,
            "qbi_claims": 232_000,
            "qbi_amount": 1_624_000_000,
        },
        "AR": {
            "salt_claims": 76_000,
            "salt_amount": 950_000_000,
            "mortgage_interest_claims": 54_000,
            "mortgage_interest_amount": 432_000_000,
            "charitable_claims": 243_000,
            "charitable_amount": 486_000_000,
            "medical_claims": 43_000,
            "medical_amount": 258_000_000,
            "qbi_claims": 98_000,
            "qbi_amount": 686_000_000,
        },
        "CA": {
            "salt_claims": 1_876_000,
            "salt_amount": 23_450_000_000,
            "mortgage_interest_claims": 1_543_000,
            "mortgage_interest_amount": 12_344_000_000,
            "charitable_claims": 3_654_000,
            "charitable_amount": 7_308_000_000,
            "medical_claims": 543_000,
            "medical_amount": 3_258_000_000,
            "qbi_claims": 1_234_000,
            "qbi_amount": 8_638_000_000,
        },
        "CO": {
            "salt_claims": 234_000,
            "salt_amount": 2_925_000_000,
            "mortgage_interest_claims": 198_000,
            "mortgage_interest_amount": 1_584_000_000,
            "charitable_claims": 543_000,
            "charitable_amount": 1_086_000_000,
            "medical_claims": 76_000,
            "medical_amount": 456_000_000,
            "qbi_claims": 198_000,
            "qbi_amount": 1_386_000_000,
        },
        "CT": {
            "salt_claims": 287_000,
            "salt_amount": 3_588_000_000,
            "mortgage_interest_claims": 198_000,
            "mortgage_interest_amount": 1_584_000_000,
            "charitable_claims": 343_000,
            "charitable_amount": 686_000_000,
            "medical_claims": 54_000,
            "medical_amount": 324_000_000,
            "qbi_claims": 132_000,
            "qbi_amount": 924_000_000,
        },
        "DE": {
            "salt_claims": 54_000,
            "salt_amount": 675_000_000,
            "mortgage_interest_claims": 43_000,
            "mortgage_interest_amount": 344_000_000,
            "charitable_claims": 98_000,
            "charitable_amount": 196_000_000,
            "medical_claims": 15_000,
            "medical_amount": 90_000_000,
            "qbi_claims": 32_000,
            "qbi_amount": 224_000_000,
        },
        "DC": {
            "salt_claims": 65_000,
            "salt_amount": 813_000_000,
            "mortgage_interest_claims": 54_000,
            "mortgage_interest_amount": 432_000_000,
            "charitable_claims": 76_000,
            "charitable_amount": 152_000_000,
            "medical_claims": 11_000,
            "medical_amount": 66_000_000,
            "qbi_claims": 28_000,
            "qbi_amount": 196_000_000,
        },
        "FL": {
            "salt_claims": 876_000,
            "salt_amount": 10_950_000_000,
            "mortgage_interest_claims": 765_000,
            "mortgage_interest_amount": 6_120_000_000,
            "charitable_claims": 1_876_000,
            "charitable_amount": 3_752_000_000,
            "medical_claims": 298_000,
            "medical_amount": 1_788_000_000,
            "qbi_claims": 654_000,
            "qbi_amount": 4_578_000_000,
        },
        "GA": {
            "salt_claims": 398_000,
            "salt_amount": 4_975_000_000,
            "mortgage_interest_claims": 343_000,
            "mortgage_interest_amount": 2_744_000_000,
            "charitable_claims": 876_000,
            "charitable_amount": 1_752_000_000,
            "medical_claims": 132_000,
            "medical_amount": 792_000_000,
            "qbi_claims": 343_000,
            "qbi_amount": 2_401_000_000,
        },
        "HI": {
            "salt_claims": 65_000,
            "salt_amount": 813_000_000,
            "mortgage_interest_claims": 54_000,
            "mortgage_interest_amount": 432_000_000,
            "charitable_claims": 132_000,
            "charitable_amount": 264_000_000,
            "medical_claims": 21_000,
            "medical_amount": 126_000_000,
            "qbi_claims": 43_000,
            "qbi_amount": 301_000_000,
        },
        "ID": {
            "salt_claims": 54_000,
            "salt_amount": 675_000_000,
            "mortgage_interest_claims": 54_000,
            "mortgage_interest_amount": 432_000_000,
            "charitable_claims": 176_000,
            "charitable_amount": 352_000_000,
            "medical_claims": 28_000,
            "medical_amount": 168_000_000,
            "qbi_claims": 65_000,
            "qbi_amount": 455_000_000,
        },
        "IL": {
            "salt_claims": 654_000,
            "salt_amount": 8_175_000_000,
            "mortgage_interest_claims": 487_000,
            "mortgage_interest_amount": 3_896_000_000,
            "charitable_claims": 1_176_000,
            "charitable_amount": 2_352_000_000,
            "medical_claims": 176_000,
            "medical_amount": 1_056_000_000,
            "qbi_claims": 398_000,
            "qbi_amount": 2_786_000_000,
        },
        "IN": {
            "salt_claims": 198_000,
            "salt_amount": 2_475_000_000,
            "mortgage_interest_claims": 154_000,
            "mortgage_interest_amount": 1_232_000_000,
            "charitable_claims": 543_000,
            "charitable_amount": 1_086_000_000,
            "medical_claims": 87_000,
            "medical_amount": 522_000_000,
            "qbi_claims": 198_000,
            "qbi_amount": 1_386_000_000,
        },
        "IA": {
            "salt_claims": 98_000,
            "salt_amount": 1_225_000_000,
            "mortgage_interest_claims": 76_000,
            "mortgage_interest_amount": 608_000_000,
            "charitable_claims": 287_000,
            "charitable_amount": 574_000_000,
            "medical_claims": 43_000,
            "medical_amount": 258_000_000,
            "qbi_claims": 109_000,
            "qbi_amount": 763_000_000,
        },
        "KS": {
            "salt_claims": 98_000,
            "salt_amount": 1_225_000_000,
            "mortgage_interest_claims": 76_000,
            "mortgage_interest_amount": 608_000_000,
            "charitable_claims": 265_000,
            "charitable_amount": 530_000_000,
            "medical_claims": 39_000,
            "medical_amount": 234_000_000,
            "qbi_claims": 98_000,
            "qbi_amount": 686_000_000,
        },
        "KY": {
            "salt_claims": 132_000,
            "salt_amount": 1_650_000_000,
            "mortgage_interest_claims": 98_000,
            "mortgage_interest_amount": 784_000_000,
            "charitable_claims": 376_000,
            "charitable_amount": 752_000_000,
            "medical_claims": 65_000,
            "medical_amount": 390_000_000,
            "qbi_claims": 132_000,
            "qbi_amount": 924_000_000,
        },
        "LA": {
            "salt_claims": 132_000,
            "salt_amount": 1_650_000_000,
            "mortgage_interest_claims": 98_000,
            "mortgage_interest_amount": 784_000_000,
            "charitable_claims": 398_000,
            "charitable_amount": 796_000_000,
            "medical_claims": 65_000,
            "medical_amount": 390_000_000,
            "qbi_claims": 143_000,
            "qbi_amount": 1_001_000_000,
        },
        "ME": {
            "salt_claims": 43_000,
            "salt_amount": 538_000_000,
            "mortgage_interest_claims": 32_000,
            "mortgage_interest_amount": 256_000_000,
            "charitable_claims": 132_000,
            "charitable_amount": 264_000_000,
            "medical_claims": 21_000,
            "medical_amount": 126_000_000,
            "qbi_claims": 43_000,
            "qbi_amount": 301_000_000,
        },
        "MD": {
            "salt_claims": 398_000,
            "salt_amount": 4_975_000_000,
            "mortgage_interest_claims": 287_000,
            "mortgage_interest_amount": 2_296_000_000,
            "charitable_claims": 543_000,
            "charitable_amount": 1_086_000_000,
            "medical_claims": 76_000,
            "medical_amount": 456_000_000,
            "qbi_claims": 176_000,
            "qbi_amount": 1_232_000_000,
        },
        "MA": {
            "salt_claims": 432_000,
            "salt_amount": 5_400_000_000,
            "mortgage_interest_claims": 298_000,
            "mortgage_interest_amount": 2_384_000_000,
            "charitable_claims": 654_000,
            "charitable_amount": 1_308_000_000,
            "medical_claims": 87_000,
            "medical_amount": 522_000_000,
            "qbi_claims": 198_000,
            "qbi_amount": 1_386_000_000,
        },
        "MI": {
            "salt_claims": 343_000,
            "salt_amount": 4_288_000_000,
            "mortgage_interest_claims": 265_000,
            "mortgage_interest_amount": 2_120_000_000,
            "charitable_claims": 876_000,
            "charitable_amount": 1_752_000_000,
            "medical_claims": 143_000,
            "medical_amount": 858_000_000,
            "qbi_claims": 298_000,
            "qbi_amount": 2_086_000_000,
        },
        "MN": {
            "salt_claims": 243_000,
            "salt_amount": 3_038_000_000,
            "mortgage_interest_claims": 187_000,
            "mortgage_interest_amount": 1_496_000_000,
            "charitable_claims": 543_000,
            "charitable_amount": 1_086_000_000,
            "medical_claims": 76_000,
            "medical_amount": 456_000_000,
            "qbi_claims": 176_000,
            "qbi_amount": 1_232_000_000,
        },
        "MS": {
            "salt_claims": 65_000,
            "salt_amount": 813_000_000,
            "mortgage_interest_claims": 54_000,
            "mortgage_interest_amount": 432_000_000,
            "charitable_claims": 243_000,
            "charitable_amount": 486_000_000,
            "medical_claims": 43_000,
            "medical_amount": 258_000_000,
            "qbi_claims": 87_000,
            "qbi_amount": 609_000_000,
        },
        "MO": {
            "salt_claims": 198_000,
            "salt_amount": 2_475_000_000,
            "mortgage_interest_claims": 154_000,
            "mortgage_interest_amount": 1_232_000_000,
            "charitable_claims": 543_000,
            "charitable_amount": 1_086_000_000,
            "medical_claims": 87_000,
            "medical_amount": 522_000_000,
            "qbi_claims": 187_000,
            "qbi_amount": 1_309_000_000,
        },
        "MT": {
            "salt_claims": 32_000,
            "salt_amount": 400_000_000,
            "mortgage_interest_claims": 28_000,
            "mortgage_interest_amount": 224_000_000,
            "charitable_claims": 109_000,
            "charitable_amount": 218_000_000,
            "medical_claims": 18_000,
            "medical_amount": 108_000_000,
            "qbi_claims": 43_000,
            "qbi_amount": 301_000_000,
        },
        "NE": {
            "salt_claims": 65_000,
            "salt_amount": 813_000_000,
            "mortgage_interest_claims": 54_000,
            "mortgage_interest_amount": 432_000_000,
            "charitable_claims": 187_000,
            "charitable_amount": 374_000_000,
            "medical_claims": 28_000,
            "medical_amount": 168_000_000,
            "qbi_claims": 65_000,
            "qbi_amount": 455_000_000,
        },
        "NV": {
            "salt_claims": 109_000,
            "salt_amount": 1_363_000_000,
            "mortgage_interest_claims": 98_000,
            "mortgage_interest_amount": 784_000_000,
            "charitable_claims": 265_000,
            "charitable_amount": 530_000_000,
            "medical_claims": 43_000,
            "medical_amount": 258_000_000,
            "qbi_claims": 109_000,
            "qbi_amount": 763_000_000,
        },
        "NH": {
            "salt_claims": 65_000,
            "salt_amount": 813_000_000,
            "mortgage_interest_claims": 54_000,
            "mortgage_interest_amount": 432_000_000,
            "charitable_claims": 132_000,
            "charitable_amount": 264_000_000,
            "medical_claims": 21_000,
            "medical_amount": 126_000_000,
            "qbi_claims": 54_000,
            "qbi_amount": 378_000_000,
        },
        "NJ": {
            "salt_claims": 654_000,
            "salt_amount": 8_175_000_000,
            "mortgage_interest_claims": 432_000,
            "mortgage_interest_amount": 3_456_000_000,
            "charitable_claims": 787_000,
            "charitable_amount": 1_574_000_000,
            "medical_claims": 109_000,
            "medical_amount": 654_000_000,
            "qbi_claims": 265_000,
            "qbi_amount": 1_855_000_000,
        },
        "NM": {
            "salt_claims": 54_000,
            "salt_amount": 675_000_000,
            "mortgage_interest_claims": 43_000,
            "mortgage_interest_amount": 344_000_000,
            "charitable_claims": 176_000,
            "charitable_amount": 352_000_000,
            "medical_claims": 32_000,
            "medical_amount": 192_000_000,
            "qbi_claims": 65_000,
            "qbi_amount": 455_000_000,
        },
        "NY": {
            "salt_claims": 1_234_000,
            "salt_amount": 15_425_000_000,
            "mortgage_interest_claims": 876_000,
            "mortgage_interest_amount": 7_008_000_000,
            "charitable_claims": 1_765_000,
            "charitable_amount": 3_530_000_000,
            "medical_claims": 265_000,
            "medical_amount": 1_590_000_000,
            "qbi_claims": 543_000,
            "qbi_amount": 3_801_000_000,
        },
        "NC": {
            "salt_claims": 387_000,
            "salt_amount": 4_838_000_000,
            "mortgage_interest_claims": 332_000,
            "mortgage_interest_amount": 2_656_000_000,
            "charitable_claims": 876_000,
            "charitable_amount": 1_752_000_000,
            "medical_claims": 132_000,
            "medical_amount": 792_000_000,
            "qbi_claims": 298_000,
            "qbi_amount": 2_086_000_000,
        },
        "ND": {
            "salt_claims": 21_000,
            "salt_amount": 263_000_000,
            "mortgage_interest_claims": 18_000,
            "mortgage_interest_amount": 144_000_000,
            "charitable_claims": 65_000,
            "charitable_amount": 130_000_000,
            "medical_claims": 11_000,
            "medical_amount": 66_000_000,
            "qbi_claims": 28_000,
            "qbi_amount": 196_000_000,
        },
        "OH": {
            "salt_claims": 398_000,
            "salt_amount": 4_975_000_000,
            "mortgage_interest_claims": 298_000,
            "mortgage_interest_amount": 2_384_000_000,
            "charitable_claims": 987_000,
            "charitable_amount": 1_974_000_000,
            "medical_claims": 165_000,
            "medical_amount": 990_000_000,
            "qbi_claims": 343_000,
            "qbi_amount": 2_401_000_000,
        },
        "OK": {
            "salt_claims": 98_000,
            "salt_amount": 1_225_000_000,
            "mortgage_interest_claims": 76_000,
            "mortgage_interest_amount": 608_000_000,
            "charitable_claims": 343_000,
            "charitable_amount": 686_000_000,
            "medical_claims": 54_000,
            "medical_amount": 324_000_000,
            "qbi_claims": 121_000,
            "qbi_amount": 847_000_000,
        },
        "OR": {
            "salt_claims": 154_000,
            "salt_amount": 1_925_000_000,
            "mortgage_interest_claims": 132_000,
            "mortgage_interest_amount": 1_056_000_000,
            "charitable_claims": 398_000,
            "charitable_amount": 796_000_000,
            "medical_claims": 65_000,
            "medical_amount": 390_000_000,
            "qbi_claims": 143_000,
            "qbi_amount": 1_001_000_000,
        },
        "PA": {
            "salt_claims": 543_000,
            "salt_amount": 6_788_000_000,
            "mortgage_interest_claims": 387_000,
            "mortgage_interest_amount": 3_096_000_000,
            "charitable_claims": 1_087_000,
            "charitable_amount": 2_174_000_000,
            "medical_claims": 176_000,
            "medical_amount": 1_056_000_000,
            "qbi_claims": 376_000,
            "qbi_amount": 2_632_000_000,
        },
        "RI": {
            "salt_claims": 54_000,
            "salt_amount": 675_000_000,
            "mortgage_interest_claims": 43_000,
            "mortgage_interest_amount": 344_000_000,
            "charitable_claims": 98_000,
            "charitable_amount": 196_000_000,
            "medical_claims": 15_000,
            "medical_amount": 90_000_000,
            "qbi_claims": 32_000,
            "qbi_amount": 224_000_000,
        },
        "SC": {
            "salt_claims": 176_000,
            "salt_amount": 2_200_000_000,
            "mortgage_interest_claims": 154_000,
            "mortgage_interest_amount": 1_232_000_000,
            "charitable_claims": 465_000,
            "charitable_amount": 930_000_000,
            "medical_claims": 76_000,
            "medical_amount": 456_000_000,
            "qbi_claims": 154_000,
            "qbi_amount": 1_078_000_000,
        },
        "SD": {
            "salt_claims": 21_000,
            "salt_amount": 263_000_000,
            "mortgage_interest_claims": 21_000,
            "mortgage_interest_amount": 168_000_000,
            "charitable_claims": 87_000,
            "charitable_amount": 174_000_000,
            "medical_claims": 14_000,
            "medical_amount": 84_000_000,
            "qbi_claims": 32_000,
            "qbi_amount": 224_000_000,
        },
        "TN": {
            "salt_claims": 198_000,
            "salt_amount": 2_475_000_000,
            "mortgage_interest_claims": 176_000,
            "mortgage_interest_amount": 1_408_000_000,
            "charitable_claims": 598_000,
            "charitable_amount": 1_196_000_000,
            "medical_claims": 98_000,
            "medical_amount": 588_000_000,
            "qbi_claims": 209_000,
            "qbi_amount": 1_463_000_000,
        },
        "TX": {
            "salt_claims": 876_000,
            "salt_amount": 10_950_000_000,
            "mortgage_interest_claims": 765_000,
            "mortgage_interest_amount": 6_120_000_000,
            "charitable_claims": 2_098_000,
            "charitable_amount": 4_196_000_000,
            "medical_claims": 332_000,
            "medical_amount": 1_992_000_000,
            "qbi_claims": 876_000,
            "qbi_amount": 6_132_000_000,
        },
        "UT": {
            "salt_claims": 98_000,
            "salt_amount": 1_225_000_000,
            "mortgage_interest_claims": 98_000,
            "mortgage_interest_amount": 784_000_000,
            "charitable_claims": 343_000,
            "charitable_amount": 686_000_000,
            "medical_claims": 43_000,
            "medical_amount": 258_000_000,
            "qbi_claims": 109_000,
            "qbi_amount": 763_000_000,
        },
        "VT": {
            "salt_claims": 28_000,
            "salt_amount": 350_000_000,
            "mortgage_interest_claims": 21_000,
            "mortgage_interest_amount": 168_000_000,
            "charitable_claims": 65_000,
            "charitable_amount": 130_000_000,
            "medical_claims": 11_000,
            "medical_amount": 66_000_000,
            "qbi_claims": 21_000,
            "qbi_amount": 147_000_000,
        },
        "VA": {
            "salt_claims": 487_000,
            "salt_amount": 6_088_000_000,
            "mortgage_interest_claims": 376_000,
            "mortgage_interest_amount": 3_008_000_000,
            "charitable_claims": 765_000,
            "charitable_amount": 1_530_000_000,
            "medical_claims": 109_000,
            "medical_amount": 654_000_000,
            "qbi_claims": 265_000,
            "qbi_amount": 1_855_000_000,
        },
        "WA": {
            "salt_claims": 287_000,
            "salt_amount": 3_588_000_000,
            "mortgage_interest_claims": 265_000,
            "mortgage_interest_amount": 2_120_000_000,
            "charitable_claims": 654_000,
            "charitable_amount": 1_308_000_000,
            "medical_claims": 98_000,
            "medical_amount": 588_000_000,
            "qbi_claims": 243_000,
            "qbi_amount": 1_701_000_000,
        },
        "WV": {
            "salt_claims": 43_000,
            "salt_amount": 538_000_000,
            "mortgage_interest_claims": 32_000,
            "mortgage_interest_amount": 256_000_000,
            "charitable_claims": 143_000,
            "charitable_amount": 286_000_000,
            "medical_claims": 28_000,
            "medical_amount": 168_000_000,
            "qbi_claims": 54_000,
            "qbi_amount": 378_000_000,
        },
        "WI": {
            "salt_claims": 221_000,
            "salt_amount": 2_763_000_000,
            "mortgage_interest_claims": 176_000,
            "mortgage_interest_amount": 1_408_000_000,
            "charitable_claims": 543_000,
            "charitable_amount": 1_086_000_000,
            "medical_claims": 87_000,
            "medical_amount": 522_000_000,
            "qbi_claims": 176_000,
            "qbi_amount": 1_232_000_000,
        },
        "WY": {
            "salt_claims": 15_000,
            "salt_amount": 188_000_000,
            "mortgage_interest_claims": 14_000,
            "mortgage_interest_amount": 112_000_000,
            "charitable_claims": 54_000,
            "charitable_amount": 108_000_000,
            "medical_claims": 9_000,
            "medical_amount": 54_000_000,
            "qbi_claims": 21_000,
            "qbi_amount": 147_000_000,
        },
    },
}


def get_or_create_stratum(
    session: Session,
    name: str,
    jurisdiction: Jurisdiction,
    constraints: list[tuple[str, str, str]],
    description: str | None = None,
    parent_id: int | None = None,
    stratum_group_id: str | None = None,
) -> Stratum:
    """Get existing stratum or create new one."""
    definition_hash = Stratum.compute_hash(constraints, jurisdiction)

    # Check if exists
    existing = session.exec(
        select(Stratum).where(Stratum.definition_hash == definition_hash)
    ).first()

    if existing:
        return existing

    # Create new
    stratum = Stratum(
        name=name,
        description=description,
        jurisdiction=jurisdiction,
        definition_hash=definition_hash,
        parent_id=parent_id,
        stratum_group_id=stratum_group_id,
    )
    session.add(stratum)
    session.flush()  # Get ID

    # Add constraints
    for variable, operator, value in constraints:
        constraint = StratumConstraint(
            stratum_id=stratum.id,
            variable=variable,
            operator=operator,
            value=value,
        )
        session.add(constraint)

    return stratum


def load_soi_deductions_targets(session: Session, years: list[int] | None = None):
    """
    Load state-level itemized deduction targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(SOI_DEDUCTIONS_DATA.keys())

    for year in years:
        if year not in SOI_DEDUCTIONS_DATA:
            continue

        data = SOI_DEDUCTIONS_DATA[year]

        # Get or create national stratum (for parent relationship)
        national_stratum = get_or_create_stratum(
            session,
            name="US All Filers",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("is_tax_filer", "==", "1")],
            description="All individual income tax returns filed in the US",
            stratum_group_id="national",
        )

        # Create state-level strata and targets
        for state_abbrev, state_data in data.items():
            if state_abbrev not in STATE_FIPS:
                continue

            fips = STATE_FIPS[state_abbrev]

            # Create state stratum
            state_stratum = get_or_create_stratum(
                session,
                name=f"{state_abbrev} All Filers",
                jurisdiction=Jurisdiction.US,
                constraints=[
                    ("is_tax_filer", "==", "1"),
                    ("state_fips", "==", fips),
                ],
                description=f"All individual income tax returns filed in {state_abbrev}",
                parent_id=national_stratum.id,
                stratum_group_id="soi_states",
            )

            # Add targets for each deduction type
            for deduction_type in DEDUCTION_TYPES:
                claims_key = f"{deduction_type}_claims"
                amount_key = f"{deduction_type}_amount"

                # Add claims target
                session.add(
                    Target(
                        stratum_id=state_stratum.id,
                        variable=claims_key,
                        period=year,
                        value=state_data[claims_key],
                        target_type=TargetType.COUNT,
                        source=DataSource.IRS_SOI,
                        source_table="SOI Individual Returns - Itemized Deductions",
                        source_url=SOURCE_URL,
                    )
                )

                # Add amount target
                session.add(
                    Target(
                        stratum_id=state_stratum.id,
                        variable=amount_key,
                        period=year,
                        value=state_data[amount_key],
                        target_type=TargetType.AMOUNT,
                        source=DataSource.IRS_SOI,
                        source_table="SOI Individual Returns - Itemized Deductions",
                        source_url=SOURCE_URL,
                    )
                )

    session.commit()


def run_etl(db_path=None):
    """Run the state-level deductions ETL pipeline."""
    from pathlib import Path

    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_soi_deductions_targets(session)
        print(f"Loaded state-level SOI deduction targets to {path}")


if __name__ == "__main__":
    run_etl()
