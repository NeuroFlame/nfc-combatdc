"""Compute site-local ComBat summaries and the final site harmonization."""

import math
import secrets
from dataclasses import replace
from typing import List

import numpy as np
import numpy.linalg as la
import pandas as pd
import statsmodels.api as sm
from framework import with_state
from sklearn.preprocessing import OneHotEncoder
from statsmodels.regression.linear_model import OLS

from .types import (
    CombatType,
    GlobalRegression,
    LocalCrossProducts,
    LocalVariance,
    PooledVariance,
    SiteColumns,
    SiteInputs,
    SiteRegistration,
    SiteState,
)

LAMBDA_VALUE = 0.0


def prepare_site(inputs: SiteInputs):
    """Encode covariates, interpolate missing data, and register the site.

    Args:
        inputs: Validated site inputs.

    Returns:
        The site's registration token, caching the prepared inputs as state.
    """
    covariates = inputs.covariates
    covariate_categories = identify_categorical_covariates(covariates)
    if str in covariate_categories:
        covariates = encode_covariates(covariates, covariate_categories)

    data_values = inputs.data.values
    if inputs.combat_algo == CombatType.COMBAT_MEGA_DC:
        data_values = interpolate_missing_data(data_values.T, covariates.to_numpy()).T
    data = pd.DataFrame(data_values, columns=inputs.data.columns)

    token = secrets.token_hex(16)
    return with_state(
        SiteRegistration(token=token),
        SiteState(token=token, covariates=covariates, data=data),
    )


def compute_local_cross_products(site_columns: SiteColumns, state: SiteState):
    """Add site-indicator columns and compute the local XᵀX and Xᵀy.

    Args:
        site_columns: Site-indicator columns assigned by the aggregator.
        state: Prepared site inputs.

    Returns:
        The local cross products, caching the site columns as state.
    """
    state = replace(state, site_columns=site_columns)
    design_values = build_design(state).to_numpy(dtype=float)
    data_values = state.data.to_numpy(dtype=float)

    local_cross_products = LocalCrossProducts(
        local_sample_count=len(data_values),
        XtransposeX_local=np.matmul(design_values.T, design_values),
        Xtransposey_local=np.matmul(design_values.T, data_values),
        lambda_value=LAMBDA_VALUE,
    )
    return with_state(local_cross_products, state)


def compute_local_variance(regression: GlobalRegression, state: SiteState):
    """Compute this site's contribution to the pooled residual variance.

    Args:
        regression: Global regression coefficients and grand mean.
        state: Site inputs and site-indicator columns.

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


def harmonize_site_data(pooled: PooledVariance, state: SiteState) -> pd.DataFrame:
    """Remove site effects from this site's data with empirical Bayes.

    Args:
        pooled: Global pooled residual variance.
        state: Site inputs, site-indicator columns, and global regression.

    Returns:
        The harmonized data, with the same columns as the input data file.
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
    return pd.DataFrame(np.transpose(bayesdata), columns=state.data.columns)


def build_design(state: SiteState) -> pd.DataFrame:
    """Return the site's covariates with its site-indicator columns appended."""
    return add_site_covariates(
        state.covariates,
        state.site_columns.site_covar_list,
        state.site_columns.token_columns[state.token],
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


def identify_categorical_covariates(covariates: pd.DataFrame) -> List[type]:
    """Return the Python type of each covariate, from the first row."""
    return [type(value) for value in list(covariates.values[0, :])]


def encode_covariates(
    covariates: pd.DataFrame, covariate_categories: List[type]
) -> pd.DataFrame:
    """One-hot encode string covariates and cast the rest to float."""
    column_names = []
    blocks = []
    covariate_names = np.expand_dims(covariates.columns.values, axis=1)

    for idx, covariate_category in enumerate(covariate_categories):
        column_values = np.expand_dims(covariates.values[:, idx], axis=1)
        if covariate_category is str:
            one_hot_encoder = OneHotEncoder().fit(column_values)
            blocks.append(one_hot_encoder.transform(column_values).toarray())
            column_names.extend(
                one_hot_encoder.get_feature_names_out(covariate_names[idx])
            )
        else:
            blocks.append(np.array(list(column_values), dtype=float))
            column_names.append(covariate_names[idx][0])

    return pd.DataFrame(data=np.hstack(blocks), columns=column_names)


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
