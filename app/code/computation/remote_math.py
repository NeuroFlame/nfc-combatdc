"""Aggregate site summaries for Decentralized ComBat."""

import logging
from typing import Dict, List, Optional

import numpy as np

from .types import (
    DEFAULT_SHARE_SITE_SUMMARIES,
    CrossSiteSummaries,
    DesignLayout,
    GlobalRegression,
    LocalCrossProducts,
    LocalVariance,
    PooledVariance,
    SiteRegistration,
    SiteSummaryShare,
)

# Above this condition number the inverted XᵀX is numerically unreliable.
ILL_CONDITIONED_THRESHOLD = 1e10
# Eigenvalues of the scaled XᵀX below this fraction of the largest are treated
# as zero when naming dependent columns.
NULL_EIGENVALUE_TOLERANCE = 1e-10
# A column belongs to a dependent group when its weight in the null vector is at
# least this fraction of the largest weight.
DEPENDENT_COLUMN_WEIGHT = 0.1


def assign_design_layout(
    site_results: Dict[str, SiteRegistration],
    categorical_levels: Optional[Dict[str, List[str]]] = None,
) -> DesignLayout:
    """Assign site-indicator columns and the shared categorical levels.

    Args:
        site_results: Site registrations keyed by site display name.
        categorical_levels: Declared levels per categorical covariate, reference
            level first; other categorical covariates use the sorted union of
            the levels the sites observed.

    Returns:
        The design layout every site encodes against.
    """
    site_names = sorted(site_results)
    token_columns = {site_results[name].token: f"site_{name}" for name in site_names}
    if len(token_columns) != len(site_names):
        raise ValueError("Sites registered duplicate tokens")

    category_levels = {
        name: list(levels) for name, levels in (categorical_levels or {}).items()
    }
    level_site_counts: Dict[str, Dict[str, int]] = {}
    for name in site_names:
        for covariate, levels in site_results[name].category_levels.items():
            counts = level_site_counts.setdefault(covariate, {})
            for level in levels:
                counts[level] = counts.get(level, 0) + 1
    for covariate, counts in level_site_counts.items():
        category_levels[covariate] = sorted(counts)

    return DesignLayout(
        site_covar_list=[f"site_{name}" for name in site_names],
        token_columns=token_columns,
        category_levels=category_levels,
        level_site_counts=level_site_counts,
    )


def compute_global_regression(
    site_results: Dict[str, LocalCrossProducts], logger: logging.Logger
) -> GlobalRegression:
    """Solve the global normal equations and compute the grand mean.

    Args:
        site_results: Local cross products keyed by site display name.
        logger: Aggregator logger.

    Returns:
        The global coefficients, the sample-weighted grand mean of the
        site-indicator coefficients, and the rank and condition number of the
        inverted XᵀX.
    """
    sites = sorted(site_results)
    column_names = site_results[sites[0]].design_columns
    if any(site_results[site].design_columns != column_names for site in sites):
        raise ValueError("Sites built different design columns")

    all_lambdas = [site_results[site].lambda_value for site in sites]
    if np.unique(all_lambdas).shape[0] != 1:
        raise ValueError("Unequal lambdas at local sites")

    XtransposeX = sum(site_results[site].XtransposeX_local for site in sites)
    XtransposeX = XtransposeX + np.unique(all_lambdas) * np.eye(XtransposeX.shape[0])
    design_rank = int(np.linalg.matrix_rank(XtransposeX))
    design_columns = int(XtransposeX.shape[0])
    design_condition = float(np.linalg.cond(XtransposeX))
    rank_deficient = design_rank < design_columns
    ill_conditioned = design_condition > ILL_CONDITIONED_THRESHOLD
    dependent_groups = (
        dependent_column_groups(XtransposeX, column_names, rank_deficient)
        if rank_deficient or ill_conditioned
        else []
    )
    if rank_deficient:
        logger.warning(
            "XᵀX is rank-deficient (rank %d of %d; dependent columns %s); "
            "harmonized values are unreliable",
            design_rank,
            design_columns,
            dependent_groups,
        )
    elif ill_conditioned:
        logger.warning(
            "XᵀX is ill-conditioned (condition %.3g; nearly dependent columns %s)",
            design_condition,
            dependent_groups,
        )
    # A singular XᵀX has no inverse. The pseudo-inverse lets the run finish so
    # every site's report can explain the problem; its harmonized data is not
    # written (see results.write_outputs).
    XtransposeX_inverse = (
        np.linalg.pinv(XtransposeX) if rank_deficient else np.linalg.inv(XtransposeX)
    )

    B_hat = sum(
        np.matmul(XtransposeX_inverse, site_results[site].Xtransposey_local)
        for site in sites
    )

    n_batch = len(sites)
    sample_per_batch = np.array(
        [site_results[site].local_sample_count for site in sites]
    )
    n_sample = int(sum(sample_per_batch))
    grand_mean = np.dot((sample_per_batch / float(n_sample)).T, B_hat[-n_batch:, :])

    return GlobalRegression(
        B_hat=B_hat,
        n_sample=n_sample,
        n_batch=n_batch,
        grand_mean=grand_mean,
        design_rank=design_rank,
        design_columns=design_columns,
        design_condition=design_condition,
        design_column_names=list(column_names),
        dependent_column_groups=dependent_groups,
    )


def dependent_column_groups(
    xtx: np.ndarray, column_names: List[str], rank_deficient: bool
) -> List[List[str]]:
    """Name the design columns that are (nearly) linear combinations of others.

    Columns are scaled to unit norm first so that their weights in each null
    vector are comparable. For a rank-deficient matrix every null vector gives a
    group; otherwise the direction of the smallest eigenvalue does.

    Args:
        xtx: The pooled XᵀX.
        column_names: Design column names, in XᵀX order.
        rank_deficient: Whether XᵀX is rank-deficient.

    Returns:
        Groups of column names, each in design order.
    """
    diagonal = np.diag(xtx)
    groups = [[column_names[i]] for i in np.where(diagonal <= 0)[0]]
    kept = np.where(diagonal > 0)[0]
    if len(kept) == 0:
        return groups

    scale = 1.0 / np.sqrt(diagonal[kept])
    eigenvalues, eigenvectors = np.linalg.eigh(
        xtx[np.ix_(kept, kept)] * np.outer(scale, scale)
    )
    null = np.where(eigenvalues <= eigenvalues.max() * NULL_EIGENVALUE_TOLERANCE)[0]
    if not rank_deficient or len(null) == 0:
        null = [int(np.argmin(eigenvalues))]

    for index in null:
        weights = np.abs(eigenvectors[:, index])
        members = [
            column_names[kept[i]]
            for i in np.where(weights >= weights.max() * DEPENDENT_COLUMN_WEIGHT)[0]
        ]
        if members not in groups:
            groups.append(members)
    return groups


def compute_pooled_variance(site_results: Dict[str, LocalVariance]) -> PooledVariance:
    """Sum the sites' pooled-variance contributions.

    Args:
        site_results: Local variance contributions keyed by site display name.

    Returns:
        The global pooled residual variance.
    """
    return PooledVariance(
        global_var_pooled=sum(
            site_results[site].local_var_pooled for site in sorted(site_results)
        )
    )


def collect_site_summaries(
    site_results: Dict[str, SiteSummaryShare],
    share_site_summaries: bool = DEFAULT_SHARE_SITE_SUMMARIES,
) -> CrossSiteSummaries:
    """Gather the sites' shared summaries for the cross-site comparison.

    Args:
        site_results: Shared summaries keyed by site display name.
        share_site_summaries: Whether cross-site sharing is enabled.

    Returns:
        The shared summaries and the names of sites that withheld theirs;
        empty when sharing is disabled.
    """
    if share_site_summaries is not True:
        return CrossSiteSummaries(enabled=False, sites={}, withheld_sites=[])

    sites = sorted(site_results)
    return CrossSiteSummaries(
        enabled=True,
        sites={
            site: site_results[site].summary
            for site in sites
            if site_results[site].summary is not None
        },
        withheld_sites=[site for site in sites if site_results[site].withheld],
    )
