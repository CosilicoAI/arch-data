"""United Kingdom Arch target loaders."""

from arch.targets.loaders import (
    load_hmrc_targets,
    load_obr_targets,
    load_ons_targets,
)

__all__ = ["load_hmrc_targets", "load_obr_targets", "load_ons_targets"]
