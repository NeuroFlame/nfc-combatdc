# Neuroflare Computation for Decentralized Combat Algorithm 
[![Paper: PMC 8965063](https://img.shields.io/static/v1?label=Paper&message=PMC8965063&color=005279)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8965063/)

## 📑 Table of Contents
- [Requirements](#requirements)
- [Overview](#overview)
  - [Participants](#participants)
  - [Stage 1 - Local Summary Extraction](#stage-1---local-summary-extraction)
  - [Stage 2 - Global Statistics Aggregation](#stage-2---global-statistics-aggregation)
  - [Stage 3 – Site‑wise Harmonization](#stage-3---site-wise-harmonization)
  - [Stage 4 - Optional Cross-Site Comparison](#stage-4---optional-cross-site-comparison)
- [Data Format Specification](#data-format-specification)
  - [Covariates File](#covariates-file)
    - [General Structure](#general-structure)
  - [Dependent Variables File](#dependent-variables-file)
    - [General Structure](#general-structure-1)
- [Configurations](#configurations)
  - [Parameters file](#parameters-file)
    - [Schema](#schema)
    - [Categorical covariates](#categorical-covariates)
  - [Logs](#logs)
- [Output](#output)
  - [Results Report](#results-report)
    - [Diagnostics](#diagnostics)
- [Developer Instructions](#developer-instructions)


## Requirements:

  - Python 3.11
  - NVFlare 2.8.0
  - Built on [computation-nvflare-boilerplate](https://github.com/NeuroFlame/computation-nvflare-boilerplate) `0.1.0` (see [`.neuroflame.json`](.neuroflame.json))

## Overview:

Decentralized ComBat is a privacy‑preserving tool that harmonizes neuroimaging data stored at multiple labs without ever copying raw files to a central server. Each site runs the ComBat math locally, shares only encrypted summary statistics with a lightweight aggregator, and then adjusts its data using the combined grand mean and variance. The result is a dataset that is statistically “site‑neutral,” giving analyses the same power and consistency as traditional, centralized ComBat while sidestepping legal, storage, and security hurdles. Tested on traumatic‑brain‑injury studies and large‑scale simulations, the method matches centralized results, scales cleanly to many sites, and lets researchers blend public and private datasets that previously could not be combined. In short, Decentralized ComBat makes multi‑center neuroimaging studies easier, safer, and more statistically robust.

Below are the key steps in the algorithm:
### Participants:
In our decentralized environment, we have two types of nodes: The first type is the aggregator node, also known as the remote node, which does not hold any data and acts as a storage of intermediate results and performs simple operations such as aggregation. The second node type is the local/regional node where datasets are located.

### Stage 1 - Local Summary Extraction:
1. Sites agree on how to encode categorical covariates: each site reports the levels it observes (unless they are declared in `categorical_levels`), and the aggregator sends back one combined, sorted list. Every site then dummy-codes against that list, leaving out the first (reference) level.
2. Each participating site runs a local regression to obtain initial β‑coefficients by computing local cross-product matrices (XᵀX and Xᵀy).
3. Using those coefficients, the site computes its local mean and local variance.
4. These summary statistics—never raw data—are securely sent to the remote aggregator node.

### Stage 2 - Global Statistics Aggregation:
1. The aggregator combines all incoming summaries to derive the grand mean and grand variance across sites.
2. It broadcasts those global values back to every local node.

### Stage 3 - Site-wise Harmonization:
1. Each node uses the grand statistics to standardize its own dataset.
2. It then estimates site‑specific effects via non-parametric empirical Bayes and adjusts its data accordingly.
3. The result: harmonized, site‑neutral data that remain in place and ready for pooled analysis.

### Stage 4 - Optional Cross-Site Comparison:
This stage always runs but only exchanges data when `share_site_summaries` is `true` (it is `false` by default).
1. Each site with at least `min_site_summary_n` subjects shares its per-ROI mean and SD before and after harmonization, plus its subject count. Smaller sites withhold their summaries.
2. The aggregator shares the collected summaries with every site, and each site's report compares them.

With sharing enabled, every site can see the other sites' summary statistics. With it disabled, no data is exchanged in this stage.


## Data Format Specification:

The computation requires two `csv` files as input:
1. **Covariates File (`CatCovariate.csv`)**
2. **Dependent Variables File (`Data.csv`)**

Both files must follow a consistent format, though the specific covariates and dependents may vary from study to study. The computation expects these files to match the covariate and dependent variable names specified in the [`parameters.json`](test_data/server/parameters.json) file.

### Covariates File:
  The key `covariate_file` in the [parameters.json](test_data/server/parameters.json#L2) should match the file_name in local site.
    
  Example: `test_data/site1/CatCovariate.csv` 

- **Format**: CSV (Comma-Separated Values)
- **Headers**: The file must include a header row where each column name corresponds to a covariate specified in the [`parameters.json`](test_data/server/parameters.json#L5).
- **Rows**: Each row represents a subject, where each column contains the value for a specific covariate.
- **Variable Names**: The names of the covariates in the header must match the entries in the `covariates_types` section of the [`parameters.json`](test_data/server/parameters.json#L6).

#### General Structure:
```csv
<Covariate_1>,<Covariate_2>,...,<Covariate_N>
<value_1>,<value_2>,...,<value_N>
<value_1>,<value_2>,...,<value_N>
...
```


### Dependent Variables File:
  The key `data_file` in the [parameters.json](test_data/server/parameters.json#L3) should match the file_name in local site.
- **Format**: CSV (Comma-Separated Values)
- **Headers**: The file must include a header row where each column name corresponds to a ROI in brain region.
- **Rows**: Each row represents the same subject as in the `covariates.csv`, with values for the dependent variables.

#### General Structure:
```csv
<Dependent_1>,<Dependent_2>,...,<Dependent_N>
<value_1>,<value_2>,...,<value_N>
<value_1>,<value_2>,...,<value_N>
...
```
---

#### Assumptions
- The data provided by each site follows the specified format (standardized covariate and dependent variable headers).
- The computation is run in a federated environment, and each site contributes valid data.


## Configurations:

### Parameters file:
The framework loads this file on the central node and shares it with every site. The site input loader ([`inputs.py`](app/code/computation/inputs.py)) validates it before any data is read.

Example: [test_data/server/parameters.json](test_data/server/parameters.json)


#### Schema

| Key | Type | Required | Description | Example |
|-----|------|----------|-------------|---------|
| `covariate_file` | `string` | ✅ | Covariate file name inside edge node data directory | `"CatCovariate.csv"` |
| `data_file` | `string` | ✅ | Dependent file name inside edge node data directory | `"Data.csv"` |
| `combat_algo` | `string` | ✅ | Which type of algorithm to implement during computation | `combatDC` or `combatMegaDC`|
| `covariates_types` | `object` | ✅ | Maps each covariate column name to its type. Categorical (`str`) columns are dummy-coded with one column per level except a shared reference level (see [Categorical covariates](#categorical-covariates)). | `{"isControl": "bool", "age": "float", "sex": "str"}` |
| `covariates_types.['key_name']` | `string` | ✅ | Primitive type name for each covariate column. | `"int"`, `"float"`, `"str"`, `"bool"` |
| `share_site_summaries` | `boolean` | ❌ | Share each site's per-ROI means, SDs, and subject count with all sites for the report's cross-site comparison. Default `false`. | `true` |
| `min_site_summary_n` | `integer` | ❌ | Sites with fewer subjects than this withhold their summaries even when sharing is enabled. Must be at least 2. Default `10`. | `10` |
| `categorical_levels` | `object` | ❌ | Declares the levels of one or more `str` covariates, reference level first. Declared covariates are not reported by sites, and a site with a value not listed stops with an error. Default: levels are collected from the sites. | `{"sex": ["female", "male"]}` |
| `log_level` | `string` | ❌ | Log verbosity. Default `info`. | `"debug"` |

#### Categorical covariates
Each categorical covariate is coded as one 0/1 column per level except a reference level, and every site uses the same levels in the same order. By default, sites report the levels they observe and the aggregator uses the sorted union, so the alphabetically first level is the reference. The choice of reference level does not change the harmonized data.

Set `categorical_levels` to choose the levels and reference yourself. This avoids sharing which levels each site has and turns inconsistent spellings at a site into an error at that site.


> Note: In the dependent file, each cell value is assumed to be either empty or of type `float`.

### Logs:
The framework writes each site's messages to `<site-id>.log` in that site's output directory, and server-side messages to `aggregator.remote.log`. Set the optional `log_level` computation parameter to `debug`, `info`, `warning`, `error`, or `critical` (default `info`).

## Output:

Once the computation is completed, each site's output directory contains:
- `harmonized_data.csv`: the harmonized data. It has the same column structure as the input data file, with site-batch effects removed and values on the original measurement scale. It is not written when the design matrix is rank-deficient, because the harmonized values would be unreliable; the report explains the cause instead.
- A copy of the site's covariate file (same file name as `covariate_file`), unchanged, for convenience. Its rows are in the same order as `harmonized_data.csv`. It stays at the site and is never shared.
- `index.html`: the results report for that site.

### Results Report:
The report is a single self-contained HTML file with no external resources. It supports light and dark themes and has these sections:

| Section | Contents |
|---------|----------|
| Summary header | Number of sites, total and local subjects, ROIs, covariates, and algorithm. Site labels are shown only for sites whose results appear in the report: this site, plus the other sites in the cross-site comparison when it is shown |
| Diagnostics and recommendations | Problems found at this site, each with a recommendation. See [Diagnostics](#diagnostics) |
| Site effects at this site | Charts of the estimated additive (γ\*) and multiplicative (δ\*) site effects per ROI, and a table of this site's means and SDs before and after harmonization alongside the pooled SD |
| Cross-site comparison | Only when `share_site_summaries` is `true`: the between-site spread of site means per ROI before and after harmonization, subjects per site, and each site's means. Otherwise, a note that the comparison is disabled |
| Run settings | Parameters, the encoded design columns, and per-ROI interpolation counts |
| Output | Download links for `harmonized_data.csv` (or an explanation of why it was not written) and the covariate file copy |

#### Diagnostics
| Check | Level | Recommendation given |
|-------|-------|----------------------|
| Design matrix is rank-deficient; the report names the dependent columns | error | Which covariate duplicates site membership or another covariate, and to remove or fix it. No harmonized data is written and the result checks below are skipped |
| Design matrix is ill-conditioned (condition number > 1e10) | warning | Which columns are nearly dependent |
| A categorical level occurs only at this site | warning | Merge the level or check its coding |
| Levels differ only in capitalization or spacing (e.g. `Male`, `male`) | warning | Recode consistently or declare `categorical_levels` |
| A categorical covariate has one level across all sites | info | Remove it from `covariates_types` |
| A covariate is constant at this site | info | When this is a concern |
| This site has fewer than 20 subjects | warning | Combine with a matching site or interpret with care |
| More than 20% of an ROI's values were interpolated | warning | Check data extraction or exclude the ROI |
| Missing values were interpolated (`combatMegaDC`) | info | — |
| Negative harmonized values where every input was non-negative | warning | Check for outliers, data-entry or unit errors |
| Large site effect (\|γ\*\| > 1 or δ\* outside 0.5–2) | warning | Check scanner, protocol, pipeline, or units |
| This site's means differ from the other sites' by orders of magnitude (sharing on) | warning | Convert units, e.g. mm³ vs cm³ |
| Harmonization increased the between-site spread by ≥ 0.1 pooled SD (sharing on) | warning | Check covariate distributions and outliers |
| This site withheld its summaries (sharing on) | info | — |

## Developer Instructions:
The computation logic lives in [`app/code/computation/`](app/code/computation/). The boilerplate owns `app/code/framework/`, `app/code/runtime/`, `app/config/`, `system/`, and the Dockerfiles; update those only by re-running the boilerplate's `scripts/migrate_computation.py`.

| File | Contents |
|------|----------|
| [`spec.py`](app/code/computation/spec.py) | Workflow declaration (four local/remote rounds plus the output step) |
| [`inputs.py`](app/code/computation/inputs.py), [`validation.py`](app/code/computation/validation.py) | Parameter checks, file loading, and type conversion |
| [`local_math.py`](app/code/computation/local_math.py) | Site-side encoding, interpolation, cross products, variance, harmonization, and summary statistics |
| [`remote_math.py`](app/code/computation/remote_math.py) | Shared design layout (site columns, categorical levels), aggregation of site summaries, and design-matrix diagnostics |
| [`results.py`](app/code/computation/results.py) | Output file writing |
| [`diagnostics.py`](app/code/computation/diagnostics.py) | Per-site findings and recommendations |
| [`report.py`](app/code/computation/report.py) | HTML results report |
| [`types.py`](app/code/computation/types.py) | Values exchanged between rounds and cached site state |

1. Run lint, format checks, and unit tests:
    > make check
2. Run a local NVFlare simulation (builds the dev image first; add `--no-build` for source-only changes):
    > ./run_local_simulation.sh site1,site2
3. Each site's results are written to `test_output/simulate_job/<site>/`.
4. Build the production image without publishing it:
    > ./dockerPush.sh --no-push
