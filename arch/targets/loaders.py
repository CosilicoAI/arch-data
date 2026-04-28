"""ETL loaders that populate Arch target inputs."""

from db.etl_aca_enrollment import load_aca_enrollment_targets
from db.etl_bls import load_bls_targets
from db.etl_cbo import load_cbo_targets
from db.etl_census import load_census_targets
from db.etl_cps import load_cps_targets
from db.etl_hmrc import load_hmrc_targets
from db.etl_medicaid import load_medicaid_targets
from db.etl_obr import load_obr_targets
from db.etl_ons import load_ons_targets
from db.etl_snap import load_snap_targets
from db.etl_soi import load_soi_targets
from db.etl_soi_credits import load_soi_credits_targets
from db.etl_soi_deductions import load_soi_deductions_targets
from db.etl_soi_income_sources import load_soi_income_sources_targets
from db.etl_soi_state import load_soi_state_targets
from db.etl_ssa import load_ssa_targets
from db.etl_ssi import load_ssi_targets

__all__ = [
    "load_aca_enrollment_targets",
    "load_bls_targets",
    "load_cbo_targets",
    "load_census_targets",
    "load_cps_targets",
    "load_hmrc_targets",
    "load_medicaid_targets",
    "load_obr_targets",
    "load_ons_targets",
    "load_snap_targets",
    "load_soi_targets",
    "load_soi_credits_targets",
    "load_soi_deductions_targets",
    "load_soi_income_sources_targets",
    "load_soi_state_targets",
    "load_ssa_targets",
    "load_ssi_targets",
]
