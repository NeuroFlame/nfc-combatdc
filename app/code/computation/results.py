"""Write each site's harmonized data."""

import logging
import os

from .local_math import harmonize_site_data
from .types import PooledVariance, SiteState

HARMONIZED_DATA_FILE = "harmonized_data.csv"


def write_harmonized_data(
    pooled: PooledVariance,
    state: SiteState,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Harmonize this site's data and write it without an index column.

    Args:
        pooled: Global pooled residual variance.
        state: Site inputs, site-indicator columns, and global regression.
        output_dir: Site output directory.
        logger: Site logger.
    """
    harmonized = harmonize_site_data(pooled, state)
    output_path = os.path.join(output_dir, HARMONIZED_DATA_FILE)
    harmonized.to_csv(output_path, index=False)
    logger.info("Wrote harmonized data to %s", output_path)
