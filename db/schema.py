"""
Database schema for targets and strata.

Three-table design (following policyengine-us-data):
- strata: Population subgroups defined by constraints
- stratum_constraints: Rules defining each stratum
- targets: Administrative totals linked to strata
"""

import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Relationship, Session, SQLModel, create_engine

# Default storage location (local dev; production uses Supabase)
DEFAULT_DB_PATH = Path(__file__).parent.parent / "macro" / "targets.db"


class Jurisdiction(str, Enum):
    """Jurisdictions we support."""

    US = "us"
    US_FEDERAL = "us-federal"
    UK = "uk"
    # States can be added as needed
    US_CA = "us-ca"
    US_NY = "us-ny"
    US_TX = "us-tx"


class DataSource(str, Enum):
    """Administrative data sources."""

    # US sources
    IRS_SOI = "irs-soi"
    CENSUS_ACS = "census-acs"
    USDA_SNAP = "usda-snap"
    SSA = "ssa"
    BLS = "bls"
    CMS_MEDICAID = "cms-medicaid"
    CMS_ACA = "cms-aca"  # ACA Marketplace enrollment data
    CBO = "cbo"  # Congressional Budget Office projections

    # UK sources
    HMRC = "hmrc"
    DWP = "dwp"
    ONS = "ons"
    OBR = "obr"  # Office for Budget Responsibility projections


class TargetType(str, Enum):
    """Whether target is a count or monetary amount."""

    COUNT = "count"
    AMOUNT = "amount"
    RATE = "rate"


class GeographicLevel(str, Enum):
    """Geographic aggregation levels for targets."""

    NATIONAL = "national"
    STATE = "state"
    CONGRESSIONAL_DISTRICT = "congressional_district"
    COUNTY = "county"
    ZIP_CODE = "zip_code"
    METRO_AREA = "metro_area"


class Stratum(SQLModel, table=True):
    """
    A population subgroup defined by constraints.

    Examples:
    - All US tax filers
    - California residents with AGI $50k-$75k
    - Single filers under age 65
    """

    __tablename__ = "strata"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    jurisdiction: Jurisdiction
    definition_hash: str = Field(unique=True, index=True)

    # Hierarchy: strata can nest (e.g., state within national)
    parent_id: Optional[int] = Field(default=None, foreign_key="strata.id")
    stratum_group_id: Optional[str] = Field(
        default=None, index=True, description="For grouping related strata in calibration"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    constraints: list["StratumConstraint"] = Relationship(back_populates="stratum")
    targets: list["Target"] = Relationship(back_populates="stratum")

    @classmethod
    def compute_hash(
        cls,
        constraints: list[tuple[str, str, str]],
        jurisdiction: "Jurisdiction | None" = None,
    ) -> str:
        """Compute unique hash from constraint definitions and jurisdiction."""
        sorted_constraints = sorted(constraints)
        # Include jurisdiction to avoid collisions between US/UK strata
        hash_input = f"{jurisdiction}:{sorted_constraints}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class StratumConstraint(SQLModel, table=True):
    """
    A single constraint defining a stratum.

    Examples:
    - variable="state_fips", operator="==", value="06" (California)
    - variable="agi", operator=">=", value="50000"
    - variable="filing_status", operator="==", value="single"
    """

    __tablename__ = "stratum_constraints"

    id: Optional[int] = Field(default=None, primary_key=True)
    stratum_id: int = Field(foreign_key="strata.id", index=True)

    variable: str = Field(index=True)
    operator: str  # "==", "!=", ">", ">=", "<", "<=", "in"
    value: str  # Stored as string, parsed based on variable dtype

    # Relationship
    stratum: Optional[Stratum] = Relationship(back_populates="constraints")


class Target(SQLModel, table=True):
    """
    An administrative target value for a stratum.

    Examples:
    - Total returns in California: 18.5M
    - Total AGI for $50k-$75k bracket: $1.2T
    - SNAP households in Texas: 1.4M
    """

    __tablename__ = "targets"

    id: Optional[int] = Field(default=None, primary_key=True)
    stratum_id: int = Field(foreign_key="strata.id", index=True)

    variable: str = Field(index=True, description="Arch target input variable ID")
    period: int = Field(index=True, description="Year")
    value: float
    target_type: TargetType = Field(default=TargetType.COUNT)

    # Geographic level for this target
    geographic_level: Optional[GeographicLevel] = Field(
        default=None, index=True, description="Geographic aggregation level"
    )

    source: DataSource
    source_table: Optional[str] = None  # e.g., "Table 1.1"
    source_url: Optional[str] = None
    notes: Optional[str] = None

    # Confidence/quality indicators
    is_preliminary: bool = Field(default=False)
    margin_of_error: Optional[float] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationship
    stratum: Optional[Stratum] = Relationship(back_populates="targets")


def get_engine(db_path: Path = DEFAULT_DB_PATH):
    """Get SQLAlchemy engine for the targets database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: Path = DEFAULT_DB_PATH):
    """Initialize database tables."""
    engine = get_engine(db_path)
    SQLModel.metadata.create_all(engine)
    return engine


def get_session(db_path: Path = DEFAULT_DB_PATH) -> Session:
    """Get a database session."""
    engine = get_engine(db_path)
    return Session(engine)
