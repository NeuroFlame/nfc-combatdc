"""Write each site's harmonized data and results page."""

import logging
import os

from .local_math import harmonize_site_data
from .types import PooledVariance, SiteState

HARMONIZED_DATA_FILE = "harmonized_data.csv"
RESULTS_PAGE_FILE = "index.html"


def write_harmonized_data(
    pooled: PooledVariance,
    state: SiteState,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Harmonize this site's data and write it with a results page.

    The CSV is written without an index column, so it is written directly
    rather than through the framework's standard CSV writer.

    Args:
        pooled: Global pooled residual variance.
        state: Site inputs, site-indicator columns, and global regression.
        output_dir: Site output directory.
        logger: Site logger.
    """
    harmonized = harmonize_site_data(pooled, state)
    output_path = os.path.join(output_dir, HARMONIZED_DATA_FILE)
    harmonized.to_csv(output_path, index=False)

    with open(
        os.path.join(output_dir, RESULTS_PAGE_FILE), "w", encoding="utf-8"
    ) as results_page:
        results_page.write(build_results_page(HARMONIZED_DATA_FILE))
    logger.info("Wrote harmonized data to %s", output_path)


def build_results_page(output_file_name: str) -> str:
    """Return an HTML page linking to the harmonized data file."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Combat DC Results</title>
</head>
<body>
    <h1>Results</h1>
    <p><a href="{output_file_name}">Download {output_file_name}</a></p>
</body>
</html>
"""
