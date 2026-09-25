"""Compute site-local ComBat summaries and the final site harmonization."""

import math
import secrets
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
import pandas as pd
import statsmodels.api as sm
from framework import with_state
from statsmodels.regression.linear_model import OLS

from .types import (
    DEFAULT_MIN_SITE_SUMMARY_N,
    DEFAULT_SHARE_SITE_SUMMARIES,
    CombatType,
    DesignLayout,
    GlobalRegression,
    HarmonizationResult,
    LocalCrossProducts,
    LocalVariance,
    PooledVariance,
    ROISummary,
    SiteInputs,
    SiteRegistration,
    SiteState,
    SiteSummaryShare,
)

LAMBDA_VALUE = 0.0


def prepare_site(
    inputs: SiteInputs, categorical_levels: Optional[Dict[str, List[str]]] = None
):
    """Register the site and report the categorical levels it observes.

    Args:
        inputs: Validated site inputs.
        categorical_levels: Declared levels per categorical covariate; levels of
            declared covariates are not reported.

    Returns:
        The site's registration, caching the validated inputs as state.
    """
    declared = categorical_levels or {}
    observed_levels = {
        name: sorted(set(inputs.covariates[name].astype(str)))
        for name in inputs.categorical_columns
    }

    missing_counts = inputs.data.isna().sum()
    interpolated_counts = {
        str(column): int(count) for column, count in missing_counts.items() if count
    }

    token = secrets.token_hex(16)
    return with_state(
        SiteRegistration(
            token=token,
            category_levels={
                name: levels
                for name, levels in observed_levels.items()
                if name not in declared
            },
        ),
        SiteState(
            token=token,
            covariates=inputs.covariates,
            data=inputs.data,
            combat_algo=str(inputs.combat_algo.value),
            categorical_columns=list(inputs.categorical_columns),
            interpolated_counts=interpolated_counts,
            observed_levels=observed_levels,
        ),
    )


def compute_local_cross_products(layout: DesignLayout, state: SiteState):
    """Encode covariates, interpolate missing data, and compute XᵀX and Xᵀy.

    Args:
        layout: Design layout assigned by the aggregator.
        state: Validated site inputs.

    Returns:
        The local cross products, caching the encoded covariates, the
        interpolated data, and the layout as state.
    """
    covariates = encode_covariates(state.covariates, layout.category_levels)
    data_values = state.data.to_numpy(dtype=float)
    if state.combat_algo == CombatType.COMBAT_MEGA_DC:
        data_values = interpolate_missing_data(
            data_values.T, covariates.to_numpy(dtype=float)
        ).T
    state = replace(
        state,
        covariates=covariates,
        data=pd.DataFrame(data_values, columns=state.data.columns),
        layout=layout,
    )

    design = build_design(state)
    design_values = design.to_numpy(dtype=float)
    local_cross_products = LocalCrossProducts(
        local_sample_count=len(data_values),
        design_columns=[str(c) for c in design.columns],
        XtransposeX_local=np.matmul(design_values.T, design_values),
        Xtransposey_local=np.matmul(design_values.T, data_values),
        lambda_value=LAMBDA_VALUE,
    )
    return with_state(local_cross_products, state)


def encode_covariates(
    covariates: pd.DataFrame, category_levels: Dict[str, List[str]]
) -> pd.DataFrame:
    """Cast covariates to float, dummy-coding categorical ones.

    Each categorical covariate gets one 0/1 column per level except the first
    (reference) level, using the levels shared by every site.
    """
    columns = {}
    for name in covariates.columns:
        if name in category_levels:
            values = covariates[name].astype(str)
            for level in category_levels[name][1:]:
                columns[f"{name}_{level}"] = (values == level).astype(float)
        else:
            columns[name] = covariates[name].astype(float)

    if len(columns) != sum(
        len(category_levels[name]) - 1 if name in category_levels else 1
        for name in covariates.columns
    ):
        raise ValueError("Encoded covariate column names collide; rename a covariate")
    return pd.DataFrame(columns, index=covariates.index).reset_index(drop=True)


def compute_local_variance(regression: GlobalRegression, state: SiteState):
    """Compute this site's contribution to the pooled residual variance.

    Args:
        regression: Global regression coefficients and grand mean.
        state: Encoded site inputs and design layout.

    Returns:
        The local variance contribution, caching the regression as state.
    """
    design = build_design(state).to_numpy(dtype=float)
    data = state.data.to_numpy(dtype=float).T
    local_n_sample = design.shape[0]

    local_var_pooled = np.dot(
        (data - np.dot(design, regression.B_hat).T) ** 2,
        np.ones((local_n_sample, 1)) / float(regression.n_sample),
    )
    return with_state(
        LocalVariance(local_var_pooled=local_var_pooled),
        replace(state, regression=regression),
    )


def harmonize_site(
    pooled: PooledVariance,
    state: SiteState,
    share_site_summaries: bool = DEFAULT_SHARE_SITE_SUMMARIES,
    min_site_summary_n: int = DEFAULT_MIN_SITE_SUMMARY_N,
):
    """Harmonize this site's data and optionally share its summary statistics.

    Args:
        pooled: Global pooled residual variance.
        state: Site inputs, design layout, and global regression.
        share_site_summaries: Whether to share per-ROI means and SDs.
        min_site_summary_n: Smallest site size whose summaries may be shared.

    Returns:
        The site's shared summary (empty unless sharing is enabled and the site
        is large enough), caching the harmonization result as state.
    """
    harmonization = harmonize_site_data(pooled, state)
    state = replace(state, harmonization=harmonization)

    if not share_site_summaries:
        return with_state(SiteSummaryShare(), state)
    if len(state.data) < min_site_summary_n:
        return with_state(SiteSummaryShare(withheld=True), state)
    return with_state(
        SiteSummaryShare(summary=summarize_site(state.data, harmonization.harmonized)),
        state,
    )


def summarize_site(data: pd.DataFrame, harmonized: pd.DataFrame) -> ROISummary:
    """Return per-ROI means and SDs of the data before and after harmonization."""
    mean_before, sd_before = roi_mean_sd(data)
    mean_after, sd_after = roi_mean_sd(harmonized)
    return ROISummary(
        sample_count=len(data),
        mean_before=mean_before,
        sd_before=sd_before,
        mean_after=mean_after,
        sd_after=sd_after,
    )


def roi_mean_sd(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return the per-column mean and sample SD (NaN for fewer than 2 rows)."""
    values = frame.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    if len(values) < 2:
        return mean, np.full(values.shape[1], np.nan)
    return mean, values.std(axis=0, ddof=1)


def harmonize_site_data(
    pooled: PooledVariance, state: SiteState
) -> HarmonizationResult:
    """Remove site effects from this site's data with empirical Bayes.

    Args:
        pooled: Global pooled residual variance.
        state: Site inputs, design layout, and global regression.

    Returns:
        The harmonized data, with the same columns as the input data file,
        and the estimated additive and multiplicative site effects.
    """
    regression = state.regression
    design = build_design(state).to_numpy(dtype=float)
    data = state.data.to_numpy(dtype=float).T
    var_pooled = pooled.global_var_pooled
    local_n_sample = design.shape[0]

    # Every column of the old per-sample standardized mean equals the grand mean.
    stand_mean = np.dot(
        regression.grand_mean.reshape((-1, 1)), np.ones((1, local_n_sample))
    )
    covariate_design = design.copy()
    covariate_design[:, range(-regression.n_batch, 0)] = 0
    mod_mean = np.transpose(np.dot(covariate_design, regression.B_hat))

    s_data = (data - stand_mean - mod_mean) / np.dot(
        np.sqrt(var_pooled), np.ones((1, local_n_sample))
    )

    batch_design = np.array([[1] * local_n_sample]).T
    gamma_hat = np.dot(
        np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T),
        s_data.T,
    )
    delta_hat = [_convert_zeroes(np.var(s_data, axis=1, ddof=1))]

    with np.errstate(divide="ignore"):
        gamma_star, delta_star = find_non_parametric_adjustments(
            s_data, gamma_hat, delta_hat
        )
    bayesdata = adjust_data_final(
        s_data,
        batch_design,
        gamma_star,
        delta_star,
        stand_mean,
        mod_mean,
        var_pooled,
        local_n_sample,
    )
    return HarmonizationResult(
        harmonized=pd.DataFrame(np.transpose(bayesdata), columns=state.data.columns),
        gamma_star=np.asarray(gamma_star, dtype=float).reshape(-1),
        delta_star=np.asarray(delta_star, dtype=float).reshape(-1),
        pooled_sd=np.sqrt(np.asarray(var_pooled, dtype=float)).reshape(-1),
    )


def build_design(state: SiteState) -> pd.DataFrame:
    """Return the site's covariates with its site-indicator columns appended."""
    return add_site_covariates(
        state.covariates,
        state.layout.site_covar_list,
        state.layout.token_columns[state.token],
    )


def add_site_covariates(
    covariates: pd.DataFrame, site_covar_list: List[str], own_column: str
) -> pd.DataFrame:
    """Append one-hot site-indicator columns with this site's column set to 1."""
    sample_count = len(covariates)
    site_df = pd.DataFrame(
        np.zeros((sample_count, len(site_covar_list)), dtype=int),
        columns=site_covar_list,
    )
    site_df[own_column] = 1

    return pd.concat(
        [covariates.reset_index(drop=True), site_df.reset_index(drop=True)], axis=1
    )


def interpolate_missing_data(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Interpolate missing values in each feature row of ``Y`` in place.

    A feature with some (but not all) values missing is interpolated from an
    OLS fit on the covariates ``X``; otherwise missing values take the feature
    mean.
    """
    if np.isnan(Y).any():
        for j in range(Y.shape[0]):
            Y_j = Y[j, :]
            is_na = np.where(np.isnan(Y_j))[0]

            if len(is_na) > 0 and len(is_na) < np.shape(Y)[1] and np.any(X):
                if len(is_na) == 1:
                    X_is_na = X[is_na, :].reshape(1, -1)
                else:
                    X_is_na = X[is_na, :]

                lm_model = OLS(
                    Y_j.astype(float).T,
                    sm.tools.tools.add_constant(X.astype(float)),
                    missing="drop",
                ).fit()
                beta = lm_model.params
                beta[np.isnan(beta)] = 0

                Y[j, is_na] = np.dot(
                    np.hstack([np.ones((len(is_na), 1)), X_is_na]), beta
                )
            else:
                Y[j, is_na] = np.nanmean(Y_j)

    return Y


def find_non_parametric_adjustments(s_data, gamma_hat, delta_hat):
    """Estimate additive and multiplicative site effects non-parametrically."""
    gamma_star, delta_star = _int_eprior(s_data, gamma_hat, delta_hat)
    return np.array([gamma_star]), np.array([delta_star])


def adjust_data_final(
    s_data,
    batch_design,
    gamma_star,
    delta_star,
    stand_mean,
    mod_mean,
    var_pooled,
    local_n_sample,
):
    """Remove site effects and return data to the original measurement scale."""
    dsq = np.sqrt(delta_star)
    denom = np.dot(dsq.T, np.ones((1, local_n_sample)))
    numer = np.array(s_data - np.dot(batch_design, gamma_star).T)
    bayesdata = numer / denom
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))

    return (
        bayesdata * np.dot(vpsq, np.ones((1, local_n_sample))) + stand_mean + mod_mean
    )


def _convert_zeroes(x):
    x[x == 0] = 1
    return x


def _int_eprior(sdat, g_hat, d_hat):
    r = sdat.shape[0]
    gamma_star, delta_star = [], []
    for i in range(0, r, 1):
        g = np.delete(g_hat, i)
        d = np.delete(d_hat, i)
        x = sdat[i, :]
        n = x.shape[0]
        j = np.repeat(1, n)
        A = np.repeat(x, g.shape[0])
        A = A.reshape(n, g.shape[0])
        A = np.transpose(A)
        B = np.repeat(g, n)
        B = B.reshape(g.shape[0], n)
        resid2 = np.square(A - B)
        sum2 = resid2.dot(j)
        LH = 1 / (2 * math.pi * d) ** (n / 2) * np.exp(-sum2 / (2 * d))
        LH = np.nan_to_num(LH)
        gamma_star.append(sum(g * LH) / sum(LH))
        delta_star.append(sum(d * LH) / sum(LH))
    return gamma_star, delta_star
