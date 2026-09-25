"""Validate and type-convert one site's covariate and data files."""

import logging
from typing import Dict, List, Tuple

import pandas as pd

from .types import CombatType

ALLOWED_COLUMN_TYPES = ("int", "float", "str", "bool")


def validate_and_get_inputs(
    covariates_path: str,
    data_path: str,
    combat_algo: CombatType,
    covariates_types: Dict[str, str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and type-convert the covariate and data files.

    Args:
        covariates_path: Path to the site's covariate CSV.
        data_path: Path to the site's dependent-variable CSV.
        combat_algo: Algorithm variant; only ``combatMegaDC`` accepts missing
            data values.
        covariates_types: Mapping of covariate column name to type name.
        logger: Site logger.

    Returns:
        The converted covariates (columns ordered as in ``covariates_types``)
        and the converted data.

    Raises:
        ValueError: If a covariate column is absent, a covariate value is
            missing or invalid, or a data value is missing under ``combatDC``.
    """
    covariates = pd.read_csv(covariates_path)
    data = pd.read_csv(data_path)
    data_types = {column: "float" for column in data.columns}

    missing_columns = [c for c in covariates_types if c not in covariates.columns]
    if missing_columns:
        raise ValueError(
            "The following covariates are not present in the covariate file: "
            f"{missing_columns}"
        )

    logger.info("Validating data file: %s", data_path)
    invalid_data_rows = _find_invalid_rows(data, data_types, logger)
    invalid_covariate_rows = _find_invalid_rows(covariates, covariates_types, logger)

    if invalid_covariate_rows:
        raise ValueError(
            "The covariate file has missing or invalid values in rows: "
            f"{invalid_covariate_rows}"
        )
    if invalid_data_rows and combat_algo == CombatType.COMBAT_DC:
        raise ValueError(
            "The data file has missing or invalid values in rows "
            f"{invalid_data_rows}; use combatMegaDC to interpolate missing data"
        )

    covariates = convert_data_to_given_type(covariates, covariates_types)
    data = convert_data_to_given_type(data, data_types)
    return covariates, data


def convert_data_to_given_type(
    data_df: pd.DataFrame, column_types: Dict[str, str]
) -> pd.DataFrame:
    """Cast each listed column to its type and keep only those columns."""
    for column_name, column_type in column_types.items():
        column_type = column_type.strip().lower()
        if column_type == "int":
            data_df[column_name] = pd.to_numeric(
                data_df[column_name], errors="coerce"
            ).astype("int")
        elif column_type == "float":
            data_df[column_name] = pd.to_numeric(
                data_df[column_name], errors="coerce"
            ).astype("float")
        elif column_type == "str":
            data_df[column_name] = data_df[column_name].astype("object")
        elif column_type == "bool":
            data_df[column_name] = pd.to_numeric(
                data_df[column_name], errors="coerce"
            ).astype("bool")
        else:
            raise ValueError(_invalid_type_message(column_name, column_type))

    return data_df[list(column_types)]


def _find_invalid_rows(
    data_df: pd.DataFrame, column_types: Dict[str, str], logger: logging.Logger
) -> List[int]:
    invalid_rows = set()
    for column_name, column_type in column_types.items():
        column_type = column_type.strip().lower()
        if column_type in ("int", "float", "bool"):
            converted = pd.to_numeric(data_df[column_name], errors="coerce")
        elif column_type == "str":
            converted = data_df[column_name].astype("object")
        else:
            raise ValueError(_invalid_type_message(column_name, column_type))

        null_rows = data_df[converted.isnull()].index.tolist()
        empty_rows = []
        if column_type == "str":
            empty_rows = data_df[converted.str.strip() == ""].index.tolist()

        invalid_rows.update(null_rows, empty_rows)
        if null_rows or empty_rows:
            logger.warning(
                "Column %s has missing or invalid values in rows %s",
                column_name,
                sorted(set(null_rows) | set(empty_rows)),
            )

    return sorted(invalid_rows)


def _invalid_type_message(column_name: str, column_type: str) -> str:
    return (
        f"Invalid datatype '{column_type}' for column '{column_name}'. "
        f"Allowed datatypes are {', '.join(ALLOWED_COLUMN_TYPES)}."
    )
