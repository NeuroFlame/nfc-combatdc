"""Load and validate one site's inputs for the first computation round."""

import logging
import os
from typing import Any, Dict, List

from .types import (
    DEFAULT_MIN_SITE_SUMMARY_N,
    DEFAULT_SHARE_SITE_SUMMARIES,
    CombatType,
    SiteInputs,
)
from .validation import ALLOWED_COLUMN_TYPES, validate_and_get_inputs

REQUIRED_PARAMETERS = ("covariate_file", "data_file", "combat_algo", "covariates_types")


def load_inputs(
    data_dir: str, parameters: Dict[str, Any], logger: logging.Logger
) -> SiteInputs:
    """Read the site's covariate and data files named in the parameters.

    Args:
        data_dir: Site-local input directory.
        parameters: Computation parameters.
        logger: Site logger.

    Returns:
        The validated, type-converted site inputs.

    Raises:
        ValueError: If a parameter is missing or invalid, or the input files
            fail validation.
    """
    missing = [name for name in REQUIRED_PARAMETERS if name not in parameters]
    if missing:
        raise ValueError(f"Missing required computation parameters: {missing}")

    try:
        combat_algo = CombatType(parameters["combat_algo"])
    except ValueError:
        allowed = [t.value for t in CombatType]
        raise ValueError(
            f"Invalid combat_algo '{parameters['combat_algo']}'. Allowed: {allowed}"
        ) from None

    covariates_types = parameters["covariates_types"]
    if not isinstance(covariates_types, dict) or not covariates_types:
        raise ValueError("covariates_types must be a non-empty object")
    for column_name, column_type in covariates_types.items():
        if str(column_type).strip().lower() not in ALLOWED_COLUMN_TYPES:
            raise ValueError(
                f"Invalid datatype '{column_type}' for covariate '{column_name}'. "
                f"Allowed datatypes are {', '.join(ALLOWED_COLUMN_TYPES)}."
            )

    share_site_summaries = parameters.get(
        "share_site_summaries", DEFAULT_SHARE_SITE_SUMMARIES
    )
    if not isinstance(share_site_summaries, bool):
        raise ValueError("share_site_summaries must be true or false")
    min_site_summary_n = parameters.get(
        "min_site_summary_n", DEFAULT_MIN_SITE_SUMMARY_N
    )
    if (
        isinstance(min_site_summary_n, bool)
        or not isinstance(min_site_summary_n, int)
        or min_site_summary_n < 2
    ):
        raise ValueError("min_site_summary_n must be an integer of at least 2")

    categorical_columns = [
        name
        for name, kind in covariates_types.items()
        if str(kind).strip().lower() == "str"
    ]
    categorical_levels = parameters.get("categorical_levels") or {}
    _validate_declared_levels(categorical_levels, categorical_columns)

    covariates_path = os.path.join(data_dir, parameters["covariate_file"])
    data_path = os.path.join(data_dir, parameters["data_file"])
    logger.info("Reading site files: %s, %s", covariates_path, data_path)

    covariates, data = validate_and_get_inputs(
        covariates_path, data_path, combat_algo, covariates_types, logger
    )
    for name, levels in categorical_levels.items():
        unexpected = sorted(set(covariates[name].astype(str)) - set(levels))
        if unexpected:
            raise ValueError(
                f"Covariate '{name}' has values not listed in categorical_levels: "
                f"{unexpected}"
            )
    return SiteInputs(
        covariates=covariates,
        data=data,
        combat_algo=combat_algo,
        categorical_columns=categorical_columns,
    )


def _validate_declared_levels(
    categorical_levels: Any, categorical_columns: List[str]
) -> None:
    if not isinstance(categorical_levels, dict):
        raise ValueError("categorical_levels must be an object")
    for name, levels in categorical_levels.items():
        if name not in categorical_columns:
            raise ValueError(
                f"categorical_levels lists '{name}', which is not a 'str' covariate "
                "in covariates_types"
            )
        if (
            not isinstance(levels, list)
            or not levels
            or not all(isinstance(level, str) for level in levels)
            or len(set(levels)) != len(levels)
        ):
            raise ValueError(
                f"categorical_levels['{name}'] must be a non-empty list of unique "
                "strings"
            )
