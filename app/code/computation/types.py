"""Define values exchanged by the Decentralized ComBat computation.

Site identity: the framework does not expose a site's own NVFlare identity to
author code, but each site must know which one-hot site-indicator column in the
shared design matrix is its own. Each site therefore mints a random token in
its first step; the aggregator, which receives results keyed by site display
name, maps every token to that site's column and broadcasts the mapping.

Categorical covariates: every site must encode the same levels in the same
order, dropping one shared reference level, or the pooled design matrix is
misaligned or rank-deficient. Sites report the levels they observe in the
first step, unless the levels are declared in ``categorical_levels``, and the
aggregator broadcasts the combined, sorted list.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_SHARE_SITE_SUMMARIES = False
DEFAULT_MIN_SITE_SUMMARY_N = 10


class CombatType(str, Enum):
    """Supported ComBat algorithm variants."""

    COMBAT_DC = "combatDC"
    COMBAT_MEGA_DC = "combatMegaDC"


@dataclass
class SiteInputs:
    """Validated, type-converted site inputs."""

    covariates: pd.DataFrame
    data: pd.DataFrame
    combat_algo: CombatType
    categorical_columns: List[str]


@dataclass
class SiteRegistration:
    """A site's registration token and observed categorical levels.

    ``category_levels`` omits covariates whose levels are declared in the
    ``categorical_levels`` parameter.
    """

    token: str
    category_levels: Dict[str, List[str]]


@dataclass
class DesignLayout:
    """Column layout shared by every site's design matrix.

    ``category_levels`` lists each categorical covariate's levels, reference
    level first. ``level_site_counts`` gives how many sites observed each level
    of a covariate whose levels were discovered rather than declared.
    """

    site_covar_list: List[str]
    token_columns: Dict[str, str]
    category_levels: Dict[str, List[str]]
    level_site_counts: Dict[str, Dict[str, int]]


@dataclass
class LocalCrossProducts:
    """Local normal-equation terms for the global regression."""

    local_sample_count: int
    design_columns: List[str]
    XtransposeX_local: np.ndarray
    Xtransposey_local: np.ndarray
    lambda_value: float


@dataclass
class GlobalRegression:
    """Global regression coefficients, grand mean, and design diagnostics."""

    B_hat: np.ndarray
    n_sample: int
    n_batch: int
    grand_mean: np.ndarray
    design_rank: int
    design_columns: int
    design_condition: float
    design_column_names: List[str]
    dependent_column_groups: List[List[str]]


@dataclass
class LocalVariance:
    """One site's contribution to the pooled residual variance."""

    local_var_pooled: np.ndarray


@dataclass
class PooledVariance:
    """Global pooled residual variance."""

    global_var_pooled: np.ndarray


@dataclass
class ROISummary:
    """Per-ROI mean and SD of one site's data before and after harmonization."""

    sample_count: int
    mean_before: np.ndarray
    sd_before: np.ndarray
    mean_after: np.ndarray
    sd_after: np.ndarray


@dataclass
class SiteSummaryShare:
    """A site's optional contribution to the cross-site comparison.

    ``summary`` is empty when sharing is disabled or the site withheld it.
    """

    summary: Optional[ROISummary] = None
    withheld: bool = False


@dataclass
class CrossSiteSummaries:
    """Site summaries shared with every site when sharing is enabled."""

    enabled: bool
    sites: Dict[str, ROISummary]
    withheld_sites: List[str]


@dataclass
class HarmonizationResult:
    """A site's harmonized data and estimated site effects."""

    harmonized: pd.DataFrame
    gamma_star: np.ndarray
    delta_star: np.ndarray
    pooled_sd: np.ndarray


@dataclass
class SiteState:
    """Site-local state cached between rounds.

    The framework keeps one local state type per workflow, so later-round
    fields are optional and filled in as the rounds progress.
    """

    token: str
    covariates: pd.DataFrame
    data: pd.DataFrame
    combat_algo: str
    categorical_columns: List[str]
    interpolated_counts: Optional[Dict[str, int]] = None
    observed_levels: Optional[Dict[str, List[str]]] = None
    layout: Optional[DesignLayout] = None
    regression: Optional[GlobalRegression] = None
    harmonization: Optional[HarmonizationResult] = None
