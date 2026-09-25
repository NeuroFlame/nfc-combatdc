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
    }
}
```

#### Settings Specification

| Variable Name | Type | Description | Allowed Options | Default | Required |
|---|---|---|---|---|---|
| `covariate_file` | `string` | Filename of the covariate CSV within the site's data directory. | any valid filename | — | ✅ Yes |
| `data_file` | `string` | Filename of the dependent variables (ROI) CSV within the site's data directory. | any valid filename | — | ✅ Yes |
| `combat_algo` | `string` | Algorithm variant to use. `combatMegaDC` additionally interpolates missing values before harmonization. | `"combatDC"`, `"combatMegaDC"` | — | ✅ Yes |
| `covariates_types` | `dict` | Maps each covariate column name to its Python primitive type. Categorical (`str`) columns are automatically dummy-encoded. | keys: column names; values: `"int"`, `"float"`, `"str"`, `"bool"` | — | ✅ Yes |


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

The computation runs four federated rounds across three harmonization stages:

1. **Round 0 — Validation and Preparation (all sites)**:
   - Each site reads and validates its covariate and data files, dummy-encodes categorical covariates, and, if `combatMegaDC` is selected, interpolates missing values in the data matrix using the covariate matrix.
   - The controller assigns one site-indicator column per participating site. No data values are transmitted at this stage.

2. **Round 1 — Local Regression (all sites)**:
   - Each site augments its design matrix with the site-indicator columns and computes the local cross-product matrices **XᵀX** and **Xᵀy**.
   - The controller aggregates these matrices across all sites, solves the global normal equations to obtain grand-mean regression coefficients (**B̂**), and broadcasts the global parameters back to every site.

3. **Round 2 — Variance Estimation (all sites)**:
   - Each site uses the global **B̂** to compute its local contribution to the pooled variance.
   - The controller aggregates the per-site variance contributions into a global pooled variance and broadcasts it to all sites.

4. **Round 3 — Harmonization and Output (all sites)**:
   - Each site standardizes its data using the global grand mean and pooled variance.
   - Site-specific additive (γ) and multiplicative (δ) batch effects are estimated via non-parametric empirical Bayes.
   - The estimated effects are removed from the data, and the harmonized measurements are written to disk as a CSV file.


#### Assumptions

- The `covariate_file` and `data_file` specified in `parameters.json` are present in each site's data directory.
- Each site's covariate file must contain all columns listed in `covariates_types`; the computation raises a validation error locally if any expected column is missing.
- Row order is consistent between the covariate file and the data file at each site.
- All values in the data file are either numeric (`float`) or empty (NaN). Empty values are only supported with `combatMegaDC`.
- The `covariates_types` dictionary covers every column in the covariate file.
- The computation runs in a federated environment with at least two participating sites.


#### Output Description

- **Output file**: `harmonized_data.csv` — written to each site's output directory at the end of Round 3.

The output file contains the harmonized version of the site's dependent variable measurements. It has the same column structure as the input data file (one column per ROI), with site-batch additive and multiplicative effects removed. Values are on the original measurement scale, ready for pooled downstream analysis.

| Column | Type | Description |
|---|---|---|
| `<ROI_1>` ... `<ROI_N>` | `float` | Harmonized measurement for each brain region, with site effects removed via empirical Bayes adjustment |
