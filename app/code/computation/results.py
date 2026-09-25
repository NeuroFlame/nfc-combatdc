"""Write each site's harmonized data, covariate copy, and results page."""

import logging
import os
import shutil
from typing import Any, Dict

from .diagnostics import harmonized_data_is_reliable
from .report import build_report
from .types import CrossSiteSummaries, SiteState

HARMONIZED_DATA_FILE = "harmonized_data.csv"
RESULTS_PAGE_FILE = "index.html"


def write_outputs(
    summaries: CrossSiteSummaries,
    state: SiteState,
    parameters: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Write this site's harmonized data, covariate copy, and results page.

    The CSV is written without an index column, so it is written directly
    rather than through the framework's standard CSV writer. It is not written
    when the pooled design matrix is rank-deficient, because the harmonized
    values are then unreliable; the report explains why. This site's covariate
    file is copied unchanged next to it for convenience; it stays at the site.

    Args:
        summaries: Cross-site summaries (empty when sharing is disabled).
        state: Site state after harmonization.
        parameters: Computation parameters.
        data_dir: Site input directory.
        output_dir: Site output directory.
        logger: Site logger.
    """
    if harmonized_data_is_reliable(state):
        state.harmonization.harmonized.to_csv(
            os.path.join(output_dir, HARMONIZED_DATA_FILE), index=False
        )
    else:
        logger.error(
            "Not writing %s: the design matrix is rank-deficient; see %s",
            HARMONIZED_DATA_FILE,
            RESULTS_PAGE_FILE,
        )

    covariate_file_name = copy_covariate_file(
        data_dir, parameters["covariate_file"], output_dir
    )

    report = build_report(
        state, summaries, parameters, HARMONIZED_DATA_FILE, covariate_file_name
    )
    with open(
        os.path.join(output_dir, RESULTS_PAGE_FILE), "w", encoding="utf-8"
    ) as results_page:
        results_page.write(report)
    logger.info("Wrote results to %s", output_dir)


def copy_covariate_file(data_dir: str, covariate_file: str, output_dir: str) -> str:
    """Copy the site's covariate file into the output directory unchanged.

    Args:
        data_dir: Site input directory.
        covariate_file: Covariate file path relative to ``data_dir``.
        output_dir: Site output directory.

    Returns:
        The copy's file name, prefixed with ``covariates_`` if it would
        otherwise replace another output file.
    """
    name = os.path.basename(covariate_file)
    if name in (HARMONIZED_DATA_FILE, RESULTS_PAGE_FILE):
        name = f"covariates_{name}"
    shutil.copyfile(
        os.path.join(data_dir, covariate_file), os.path.join(output_dir, name)
    )
    return name
