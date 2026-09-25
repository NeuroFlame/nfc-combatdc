"""Load and validate one site's inputs for the first computation round."""

import logging
import os
from typing import Any, Dict

from .types import CombatType, SiteInputs
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

    covariates_path = os.path.join(data_dir, parameters["covariate_file"])
    data_path = os.path.join(data_dir, parameters["data_file"])
    logger.info("Reading site files: %s, %s", covariates_path, data_path)

    covariates, data = validate_and_get_inputs(
        covariates_path, data_path, combat_algo, covariates_types, logger
    )
    return SiteInputs(covariates=covariates, data=data, combat_algo=combat_algo)
