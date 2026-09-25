"""Aggregate site summaries for Decentralized ComBat."""

from typing import Dict

import numpy as np

from .types import (
    GlobalRegression,
    LocalCrossProducts,
    LocalVariance,
    PooledVariance,
    SiteColumns,
    SiteRegistration,
)


def assign_site_columns(site_results: Dict[str, SiteRegistration]) -> SiteColumns:
    """Name one site-indicator column per site and map each token to it.

    Args:
        site_results: Registration tokens keyed by site display name.

    Returns:
        The ordered site-indicator columns and the token-to-column mapping.
    """
    site_names = sorted(site_results)
    token_columns = {site_results[name].token: f"site_{name}" for name in site_names}
    if len(token_columns) != len(site_names):
        raise ValueError("Sites registered duplicate tokens")

    return SiteColumns(
        site_covar_list=[f"site_{name}" for name in site_names],
        token_columns=token_columns,
    )


def compute_global_regression(
    site_results: Dict[str, LocalCrossProducts],
) -> GlobalRegression:
    """Solve the global normal equations and compute the grand mean.

    Args:
        site_results: Local cross products keyed by site display name.

    Returns:
        The global coefficients and the sample-weighted grand mean of the
        site-indicator coefficients.
    """
    sites = sorted(site_results)

    all_lambdas = [site_results[site].lambda_value for site in sites]
    if np.unique(all_lambdas).shape[0] != 1:
        raise ValueError("Unequal lambdas at local sites")

    XtransposeX = sum(site_results[site].XtransposeX_local for site in sites)
    XtransposeX = XtransposeX + np.unique(all_lambdas) * np.eye(XtransposeX.shape[0])
    XtransposeX_inverse = np.linalg.inv(XtransposeX)

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
        B_hat=B_hat, n_sample=n_sample, n_batch=n_batch, grand_mean=grand_mean
    )


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
