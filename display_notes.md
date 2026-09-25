### Computation Description

#### Overview

This computation implements Decentralized ComBat (CombatDC), a privacy-preserving algorithm that harmonizes neuroimaging data stored at multiple federated sites without ever transferring raw data off-site. Each site shares only encrypted summary statistics with a lightweight aggregator; raw measurements never leave the originating lab. The result is a harmonized, site-effect-free dataset that provides the same statistical power as traditional centralized ComBat while respecting legal, storage, and security constraints.

Two algorithm variants are supported:
- **`combatDC`** — standard Decentralized ComBat. Each site's data is assumed to be complete.
- **`combatMegaDC`** — extends `combatDC` with missing-data interpolation at each site before the federated rounds begin, allowing subjects with partial measurements to be included.


#### Example

```json
{
    "covariate_file": "CatCovariate.csv",
    "data_file": "Data.csv",
    "combat_algo": "combatMegaDC",
    "covariates_types": {
        "isControl": "bool",
        "age": "float",
        "sex": "str"
    },
    "share_site_summaries": false,
    "min_site_summary_n": 10
}
```

#### Settings Specification

| Variable Name | Type | Description | Allowed Options | Default | Required |
|---|---|---|---|---|---|
| `covariate_file` | `string` | Filename of the covariate CSV within the site's data directory. | any valid filename | — | ✅ Yes |
| `data_file` | `string` | Filename of the dependent variables (ROI) CSV within the site's data directory. | any valid filename | — | ✅ Yes |
| `combat_algo` | `string` | Algorithm variant to use. `combatMegaDC` additionally interpolates missing values before harmonization. | `"combatDC"`, `"combatMegaDC"` | — | ✅ Yes |
| `covariates_types` | `dict` | Maps each covariate column name to its Python primitive type. Categorical (`str`) columns are dummy-coded with one column per level except a reference level shared by all sites. | keys: column names; values: `"int"`, `"float"`, `"str"`, `"bool"` | — | ✅ Yes |
| `categorical_levels` | `dict` | Declares the levels of `str` covariates, reference level first. Sites do not report the levels of declared covariates, and a site with a value that is not listed stops with an error. Undeclared covariates use the sorted union of the levels the sites observe. | keys: `str` covariate names; values: non-empty lists of unique strings | levels collected from the sites | ❌ No |
| `share_site_summaries` | `boolean` | Share each site's per-ROI means and SDs (before and after harmonization) and its subject count with all participating sites, so each site's report can compare sites. When `false`, no summaries are exchanged. | `true`, `false` | `false` | ❌ No |
| `min_site_summary_n` | `integer` | Sites with fewer subjects than this withhold their summaries even when sharing is enabled. | integer ≥ 2 | `10` | ❌ No |
| `log_level` | `string` | Verbosity of the site and aggregator logs. | `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"` | `"info"` | ❌ No |


#### Input Description

The computation requires two CSV files at each site, with filenames matching the values provided in `parameters.json`.

1. **Covariates File** (e.g. `CatCovariate.csv`)
2. **Dependent Variables File** (e.g. `Data.csv`)

Both files must reside in the site's data directory, and row order must be consistent — row *i* in the covariates file corresponds to the same subject as row *i* in the data file.

##### Covariates File

- **Format**: CSV (Comma-Separated Values)
- **Headers**: One column per covariate. Column names must match the keys in `covariates_types`.
- **Rows**: One row per subject.

**General Structure**:
```csv
<Covariate_1>,<Covariate_2>,...,<Covariate_N>
<value_1>,<value_2>,...,<value_N>
<value_1>,<value_2>,...,<value_N>
...
```

##### Dependent Variables File

- **Format**: CSV (Comma-Separated Values)
- **Headers**: One column per brain ROI or imaging-derived phenotype. All cell values must be numeric (`float`) or empty.
- **Rows**: One row per subject, in the same order as the covariates file.

**General Structure**:
```csv
<ROI_1>,<ROI_2>,...,<ROI_N>
<value_1>,<value_2>,...,<value_N>
<value_1>,<value_2>,...,<value_N>
...
```

---

#### Algorithm Description

The computation runs five federated rounds:

1. **Round 0 — Validation and Preparation (all sites)**:
   - Each site reads and validates its covariate and data files and reports the levels it observes for each categorical covariate not declared in `categorical_levels`.
   - The controller assigns one site-indicator column per participating site and combines the reported levels into one sorted list per categorical covariate; the first level is the reference. No data values are transmitted at this stage.

2. **Round 1 — Local Regression (all sites)**:
   - Each site dummy-codes its categorical covariates against the shared levels, leaving out the reference level, and, if `combatMegaDC` is selected, interpolates missing values in the data matrix using the encoded covariates.
   - Each site augments its design matrix with the site-indicator columns and computes the local cross-product matrices **XᵀX** and **Xᵀy**, sending the design column names so the controller can check that all sites built the same columns.
   - The controller aggregates these matrices across all sites, solves the global normal equations to obtain grand-mean regression coefficients (**B̂**), and broadcasts the global parameters back to every site.
   - The controller also records the rank and condition number of the pooled **XᵀX** and names any (nearly) linearly dependent columns, so the report can explain an unreliable regression. If **XᵀX** is rank-deficient, the run still finishes so every site receives its report, but no harmonized data is written.

3. **Round 2 — Variance Estimation (all sites)**:
   - Each site uses the global **B̂** to compute its local contribution to the pooled variance.
   - The controller aggregates the per-site variance contributions into a global pooled variance and broadcasts it to all sites.

4. **Round 3 — Harmonization (all sites)**:
   - Each site standardizes its data using the global grand mean and pooled variance.
   - Site-specific additive (γ) and multiplicative (δ) batch effects are estimated via non-parametric empirical Bayes.
   - The estimated effects are removed from the data.
   - If `share_site_summaries` is `true` and the site has at least `min_site_summary_n` subjects, the site sends its per-ROI means and SDs before and after harmonization, plus its subject count. Otherwise it sends nothing.

5. **Round 4 — Cross-Site Comparison and Output (all sites)**:
   - If sharing is enabled, the controller sends the collected site summaries to every site.
   - Each site writes its harmonized data and its results report.


#### Assumptions

- The `covariate_file` and `data_file` specified in `parameters.json` are present in each site's data directory.
- Each site's covariate file must contain all columns listed in `covariates_types`; the computation raises a validation error locally if any expected column is missing.
- Row order is consistent between the covariate file and the data file at each site.
- All values in the data file are either numeric (`float`) or empty (NaN). Empty values are only supported with `combatMegaDC`.
- Columns in the covariate file that are not listed in `covariates_types` are ignored.
- Categorical values are spelled the same way at every site; the report flags levels that differ only in capitalization or spacing.
- The computation runs in a federated environment with at least two participating sites.


#### Output Description

- **Output files**: `harmonized_data.csv`, a copy of the site's covariate file, and an `index.html` results report, written to each site's output directory at the end of Round 4. `harmonized_data.csv` is not written when the design matrix is rank-deficient; the report explains the cause. The covariate file is copied unchanged under its original file name, with rows in the same order as `harmonized_data.csv`; it stays at the site and is never shared.

##### Harmonized Data

`harmonized_data.csv` contains the harmonized version of the site's dependent variable measurements. It has the same column structure as the input data file (one column per ROI), with site-batch additive and multiplicative effects removed. Values are on the original measurement scale, ready for pooled downstream analysis.

| Column | Type | Description |
|---|---|---|
| `<ROI_1>` ... `<ROI_N>` | `float` | Harmonized measurement for each brain region, with site effects removed via empirical Bayes adjustment |

##### Results Report

`index.html` is a self-contained page (no external resources, light and dark themes) with these sections:

- **Summary**: number of sites, total and local subjects, ROIs, covariates, and algorithm. Site labels are shown only for sites whose results appear in the report: this site, plus the other sites in the cross-site comparison when it is shown.
- **Diagnostics and recommendations**: problems found at this site, most severe first, each with a concrete recommendation:
  - *Design matrix*: an error when it is rank-deficient, or a warning when it is ill-conditioned, naming the dependent columns and the covariate to remove or fix, for example a covariate that is constant within each site.
  - *Categorical covariates*: a level that occurs only at this site, levels that differ only in capitalization or spacing, and a covariate with a single level across all sites.
  - *This site's data*: covariates that are constant at this site, fewer than 20 subjects, ROIs with more than 20% of values interpolated, negative harmonized values where every input was non-negative, and large site effects (|γ\*| > 1 or δ\* outside 0.5–2).
  - *With sharing enabled*: this site's means differing from the other sites' by orders of magnitude (a likely unit mismatch), and ROIs whose between-site spread increased by at least 0.1 pooled SD.
  - When the design matrix is rank-deficient, checks of the harmonized values are skipped because those values are not meaningful.
- **Site effects at this site**: charts of the additive (γ\*, in pooled-SD units; 0 means no effect) and multiplicative (δ\*, variance ratio; 1 means no effect) site effects per ROI, and a table of this site's means and SDs before and after harmonization alongside the pooled SD.
- **Cross-site comparison**: shown only when `share_site_summaries` is `true`. It charts the between-site spread of site means per ROI before and after harmonization, where lower is better, and lists subjects per site and each site's means. Sites below `min_site_summary_n` are listed as withheld. When sharing is disabled, the section shows a note instead.
- **Run settings**: the parameters used, the encoded design columns, the categorical levels with their reference level and source (declared or collected from sites), and per-ROI interpolation counts.
- **Output**: download links for `harmonized_data.csv` (or an explanation of why it was not written) and the covariate file copy.
