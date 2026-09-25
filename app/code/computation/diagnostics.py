"""Detect problems with a site's harmonization and recommend fixes."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .local_math import roi_mean_sd
from .remote_math import ILL_CONDITIONED_THRESHOLD
from .types import DEFAULT_MIN_SITE_SUMMARY_N, CrossSiteSummaries, SiteState

SITE_COLUMN_PREFIX = "site_"
MAX_LISTED_ITEMS = 10
# Sites smaller than this give noisy site-effect estimates.
SMALL_SITE_N = 20
# ROIs with more than this fraction of values interpolated are flagged.
HIGH_MISSING_FRACTION = 0.2
# |γ*| above this (pooled-SD units) is a large additive site effect.
LARGE_ADDITIVE_EFFECT = 1.0
# δ* outside this range is a large multiplicative site effect.
MULTIPLICATIVE_EFFECT_RANGE = (0.5, 2.0)
# A site mean this many orders of magnitude from the other sites' median
# suggests a unit mismatch.
UNIT_MISMATCH_ORDERS = 2.5
# Increases in between-site spread smaller than this (pooled-SD units) are
# treated as noise; 0.1 is half of a conventional "small" effect size.
SPREAD_INCREASE_TOLERANCE = 0.1

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Finding:
    """One diagnostic finding with an optional recommendation."""

    level: str
    title: str
    detail: str = ""
    recommendation: str = ""


def harmonized_data_is_reliable(state: SiteState) -> bool:
    """Return whether the pooled design matrix has full rank."""
    return state.regression.design_rank == state.regression.design_columns


def site_name(state: SiteState) -> str:
    """Return this site's display name."""
    return state.layout.token_columns[state.token][len(SITE_COLUMN_PREFIX) :]


def collect_findings(
    state: SiteState, summaries: CrossSiteSummaries, parameters: Dict[str, Any]
) -> List[Finding]:
    """Return this site's findings, most severe first.

    Args:
        state: Site state after harmonization.
        summaries: Cross-site summaries (empty when sharing is disabled).
        parameters: Computation parameters.

    Returns:
        The findings; a single "ok" finding leads when there are no errors or
        warnings. Checks of the harmonized results are skipped when the design
        matrix is rank-deficient, because those results are not meaningful.
    """
    reliable = harmonized_data_is_reliable(state)
    findings = (
        _design_findings(state)
        + _categorical_findings(state)
        + _constant_covariate_findings(state)
        + _site_size_findings(state)
        + _missing_data_findings(state)
        + _cross_site_findings(state, summaries, parameters, reliable)
    )
    if reliable:
        findings += _output_findings(state) + _site_effect_findings(state)
    findings.sort(key=lambda f: _SEVERITY_ORDER[f.level])
    if not any(f.level in ("error", "warning") for f in findings):
        findings.insert(0, Finding("ok", "No problems detected."))
    return findings


def between_site_spread(
    summaries: CrossSiteSummaries, pooled_sd
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the SD of site means per ROI before and after, in pooled-SD units."""
    names = sorted(summaries.sites)
    before = np.vstack([summaries.sites[n].mean_before for n in names])
    after = np.vstack([summaries.sites[n].mean_after for n in names])
    scale = np.where(np.asarray(pooled_sd) > 0, pooled_sd, np.nan)
    return before.std(axis=0) / scale, after.std(axis=0) / scale


# ── Design matrix ───────────────────────────────────────────────────────────


def _design_findings(state: SiteState) -> List[Finding]:
    regression = state.regression
    rank_deficient = regression.design_rank < regression.design_columns
    if not rank_deficient and regression.design_condition <= ILL_CONDITIONED_THRESHOLD:
        return []

    site_columns = set(state.layout.site_covar_list)
    covariate_of = _covariate_of_column(state)
    described = []
    recommendations = []
    for group in regression.dependent_column_groups:
        covariates = _unique(
            covariate_of.get(c, c) for c in group if c not in site_columns
        )
        involves_site = any(c in site_columns for c in group)
        described.append(", ".join(group))
        recommendations.append(
            _dependence_recommendation(covariates, involves_site, len(group))
        )

    if rank_deficient:
        title = (
            "The design matrix is rank-deficient "
            f"(rank {regression.design_rank} of {regression.design_columns})."
        )
        detail = (
            "The pooled regression has no unique solution, so no harmonized data "
            "was written and checks of the harmonized values are skipped."
        )
        level = "error"
    else:
        title = (
            "The design matrix is ill-conditioned "
            f"(condition number {regression.design_condition:.3g})."
        )
        detail = "Small numerical errors may be strongly amplified."
        level = "warning"
    if described:
        qualifier = "Linearly" if rank_deficient else "Nearly linearly"
        detail += f" {qualifier} dependent columns: " + "; ".join(described) + "."
    return [
        Finding(
            level,
            title,
            detail,
            " ".join(_unique(recommendations))
            or "Check for covariates that duplicate each other or site membership.",
        )
    ]


def _dependence_recommendation(
    covariates: List[str], involves_site: bool, group_size: int
) -> str:
    if not covariates:
        return (
            "The site indicator columns are dependent on each other; check that "
            "every site contributed subjects."
        )
    names = _quoted(covariates)
    if group_size == 1:
        return f"{names} is zero for every subject; remove it from covariates_types."
    if involves_site:
        return (
            f"{names} cannot be separated from site: its values line up with site "
            "membership, for example it is constant within each site or its levels "
            "coincide with sites. Remove it from covariates_types, or check with the "
            "affected sites that it was recorded consistently."
        )
    return (
        f"{names} are linear combinations of each other, for example one is the "
        "sum of others or two encode the same information. Keep only one of them."
    )


def _covariate_of_column(state: SiteState) -> Dict[str, str]:
    mapping = {str(c): str(c) for c in state.covariates.columns}
    for name, levels in state.layout.category_levels.items():
        for level in levels[1:]:
            mapping[f"{name}_{level}"] = name
    return mapping


# ── Covariates ──────────────────────────────────────────────────────────────


def _categorical_findings(state: SiteState) -> List[Finding]:
    findings = []
    layout = state.layout
    site_count = len(layout.site_covar_list)
    own_levels = state.observed_levels or {}
    for name in state.categorical_columns:
        levels = layout.category_levels.get(name, [])
        if len(levels) < 2:
            findings.append(
                Finding(
                    "info",
                    f"Covariate '{name}' has a single level across all sites "
                    f"({_quoted(levels)}) and was left out of the model.",
                    recommendation=(
                        f"Remove '{name}' from covariates_types, or check whether "
                        "other sites coded it differently."
                    ),
                )
            )
            continue

        variants: Dict[str, List[str]] = {}
        for level in levels:
            variants.setdefault(level.strip().casefold(), []).append(level)
        duplicates = [group for group in variants.values() if len(group) > 1]
        if duplicates:
            findings.append(
                Finding(
                    "warning",
                    f"Covariate '{name}' has levels that differ only in "
                    "capitalization or spacing.",
                    "Levels: "
                    + "; ".join(_quoted(group) for group in duplicates)
                    + ".",
                    "Recode these values the same way at every site. Listing the "
                    "intended levels in categorical_levels makes any site with other "
                    "spellings stop with an error, which identifies it.",
                )
            )

        counts = layout.level_site_counts.get(name, {})
        unique_here = [
            level
            for level in own_levels.get(name, [])
            if counts.get(level) == 1 and site_count > 1
        ]
        if unique_here:
            findings.append(
                Finding(
                    "warning",
                    f"Level(s) {_quoted(unique_here)} of covariate '{name}' occur only "
                    "at this site.",
                    "Their effect is estimated from this site alone, so it cannot be "
                    "fully separated from this site's effect.",
                    "Merge them with a related level, or check that they are coded "
                    "the same way as at the other sites.",
                )
            )
    return findings


def _constant_covariate_findings(state: SiteState) -> List[Finding]:
    # Covariates already named in a design-matrix finding are not repeated here.
    covariate_of = _covariate_of_column(state)
    explained = {
        covariate_of.get(column, column)
        for group in state.regression.dependent_column_groups
        for column in group
    }
    constant = []
    for name, levels in (state.observed_levels or {}).items():
        if (
            len(levels) == 1
            and len(state.layout.category_levels.get(name, [])) > 1
            and name not in explained
        ):
            constant.append(f"{name} = {levels[0]}")
    categorical = set(state.categorical_columns)
    for name in state.covariates.columns:
        if covariate_of.get(str(name)) in categorical or str(name) in explained:
            continue
        values = state.covariates[name].to_numpy(dtype=float)
        if len(values) and np.all(values == values[0]):
            constant.append(f"{name} = {values[0]:g}")
    if not constant:
        return []
    return [
        Finding(
            "info",
            "Some covariates have the same value for every subject at this site: "
            + "; ".join(constant)
            + ".",
            "Their effects are estimated from the other sites only.",
            "This is fine if the covariate genuinely does not vary here. If it "
            "differs systematically between sites, for example all patients at "
            "one site, part of its effect may be absorbed into the site effect.",
        )
    ]


# ── Site data ───────────────────────────────────────────────────────────────


def _site_size_findings(state: SiteState) -> List[Finding]:
    sample_count = len(state.data)
    if sample_count >= SMALL_SITE_N:
        return []
    return [
        Finding(
            "warning",
            f"This site has only {sample_count} subjects.",
            "Site-effect estimates from small sites are noisy.",
            "If this site shares a scanner and protocol with another site, "
            "consider combining their data as one site; otherwise interpret its "
            "harmonized values with care.",
        )
    ]


def _missing_data_findings(state: SiteState) -> List[Finding]:
    interpolated = state.interpolated_counts or {}
    if not interpolated:
        return []
    sample_count = len(state.data)
    findings = [
        Finding(
            "info",
            f"{sum(interpolated.values())} missing value(s) in {len(interpolated)} "
            "ROI(s) were interpolated before harmonization.",
            "See Run settings for the per-ROI counts.",
        )
    ]
    heavy = [
        f"{roi} ({count / sample_count:.0%})"
        for roi, count in interpolated.items()
        if count / sample_count > HIGH_MISSING_FRACTION
    ]
    if heavy:
        findings.append(
            Finding(
                "warning",
                f"{len(heavy)} ROI(s) are missing more than "
                f"{HIGH_MISSING_FRACTION:.0%} of their values at this site.",
                "Affected: " + _truncated(heavy) + ".",
                "Interpolated values are predicted from the covariates, so they "
                "carry no information of their own. Check the data extraction "
                "for these ROIs at this site, or exclude them from the data file.",
            )
        )
    return findings


def _output_findings(state: SiteState) -> List[Finding]:
    input_min = state.data.to_numpy(dtype=float).min(axis=0)
    harmonized_min = state.harmonization.harmonized.to_numpy(dtype=float).min(axis=0)
    negative = [
        str(roi)
        for i, roi in enumerate(state.data.columns)
        if input_min[i] >= 0 and harmonized_min[i] < 0
    ]
    if not negative:
        return []
    return [
        Finding(
            "warning",
            f"{len(negative)} ROI(s) have negative harmonized values although every "
            "input value was non-negative.",
            "Affected: " + _truncated(negative) + ".",
            "Resolve any design-matrix error above first. Otherwise check these "
            "ROIs at this site for outliers, data-entry errors, or unit differences.",
        )
    ]


def _site_effect_findings(state: SiteState) -> List[Finding]:
    harmonization = state.harmonization
    low, high = MULTIPLICATIVE_EFFECT_RANGE
    large = [
        str(roi)
        for i, roi in enumerate(state.data.columns)
        if abs(harmonization.gamma_star[i]) > LARGE_ADDITIVE_EFFECT
        or not low <= harmonization.delta_star[i] <= high
    ]
    if not large:
        return []
    return [
        Finding(
            "warning",
            f"This site has large site effects for {len(large)} ROI(s).",
            f"Affected (|γ*| > {LARGE_ADDITIVE_EFFECT:g} or δ* outside "
            f"{low:g}–{high:g}): " + _truncated(large) + ".",
            "Check whether this site used a different scanner, acquisition "
            "protocol, preprocessing pipeline or software version, or different "
            "measurement units for these ROIs.",
        )
    ]


# ── Cross-site ──────────────────────────────────────────────────────────────


def _cross_site_findings(
    state: SiteState,
    summaries: CrossSiteSummaries,
    parameters: Dict[str, Any],
    check_results: bool,
) -> List[Finding]:
    if not summaries.enabled:
        return []
    findings = []
    own = site_name(state)
    if own in summaries.withheld_sites:
        min_n = parameters.get("min_site_summary_n", DEFAULT_MIN_SITE_SUMMARY_N)
        findings.append(
            Finding(
                "info",
                "This site's summaries were not shared because it has fewer than "
                f"{min_n} subjects.",
            )
        )

    others = [summaries.sites[n] for n in sorted(summaries.sites) if n != own]
    if others:
        own_mean, _ = roi_mean_sd(state.data)
        other_median = np.median(np.vstack([o.mean_before for o in others]), axis=0)
        mismatched = []
        for i, roi in enumerate(state.data.columns):
            if own_mean[i] > 0 and other_median[i] > 0:
                orders = math.log10(own_mean[i] / other_median[i])
                if abs(orders) >= UNIT_MISMATCH_ORDERS:
                    mismatched.append(f"{roi} (×{10 ** round(orders):g})")
        if mismatched:
            findings.append(
                Finding(
                    "warning",
                    f"Values for {len(mismatched)} ROI(s) at this site differ from "
                    "the other sites by orders of magnitude.",
                    "Affected, with this site's mean relative to the other sites' "
                    "median: " + _truncated(mismatched) + ".",
                    "This usually means different units, for example mm³ and cm³. "
                    "Convert this site's values to the units the other sites use "
                    "and rerun.",
                )
            )

    if check_results and len(summaries.sites) >= 2:
        before, after = between_site_spread(summaries, state.harmonization.pooled_sd)
        increased = [
            str(roi)
            for i, roi in enumerate(state.data.columns)
            if after[i] > before[i] + SPREAD_INCREASE_TOLERANCE
        ]
        if increased:
            findings.append(
                Finding(
                    "warning",
                    f"Harmonization increased the between-site spread for "
                    f"{len(increased)} ROI(s).",
                    "Affected: " + _truncated(increased) + ".",
                    "Resolve any errors above first. With a healthy design matrix "
                    "this can happen when sites differ in their covariate "
                    "distributions, such as age, because covariate effects are "
                    "preserved; otherwise check these ROIs for outliers.",
                )
            )
    return findings


# ── Helpers ─────────────────────────────────────────────────────────────────


def _unique(items) -> List[str]:
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _quoted(items: List[str]) -> str:
    return ", ".join(f"'{item}'" for item in items)


def _truncated(items: List[str]) -> str:
    shown = ", ".join(items[:MAX_LISTED_ITEMS])
    extra = len(items) - MAX_LISTED_ITEMS
    return f"{shown}, and {extra} more" if extra > 0 else shown
