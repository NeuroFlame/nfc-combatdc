"""Define values exchanged by the Decentralized ComBat computation.

Site identity: the framework does not expose a site's own NVFlare identity to
author code, but each site must know which one-hot site-indicator column in the
shared design matrix is its own. Each site therefore mints a random token in
its first step; the aggregator, which receives results keyed by site display
name, maps every token to that site's column and broadcasts the mapping.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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


@dataclass
class SiteRegistration:
    """Opaque token a site uses to recognize its own site-indicator column."""

    token: str


@dataclass
class SiteColumns:
    """Site-indicator columns shared by every site's design matrix."""

    site_covar_list: List[str]
    token_columns: Dict[str, str]


@dataclass
class LocalCrossProducts:
    """Local normal-equation terms for the global regression."""

    local_sample_count: int
    XtransposeX_local: np.ndarray
    Xtransposey_local: np.ndarray
    lambda_value: float


@dataclass
class GlobalRegression:
    """Global regression coefficients and grand mean."""

    B_hat: np.ndarray
    n_sample: int
    n_batch: int
    grand_mean: np.ndarray


@dataclass
class LocalVariance:
    """One site's contribution to the pooled residual variance."""

    local_var_pooled: np.ndarray


@dataclass
class PooledVariance:
    """Global pooled residual variance."""

    global_var_pooled: np.ndarray


@dataclass
class SiteState:
    """Site-local state cached between rounds.

    The framework keeps one local state type per workflow, so later-round
    fields are optional and filled in as the rounds progress.
    """

    token: str
    covariates: pd.DataFrame
    data: pd.DataFrame
    site_columns: Optional[SiteColumns] = None
    regression: Optional[GlobalRegression] = None
