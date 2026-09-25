# Neuroflare Computation for Decentralized Combat Algorithm 
[![Paper: PMC 8965063](https://img.shields.io/static/v1?label=Paper&message=PMC8965063&color=005279)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8965063/)

## 📑 Table of Contents
- [Requirements](#requirements)
- [Overview](#overview)
  - [Participants](#participants)
  - [Stage 1 - Local Summary Extraction](#stage-1---local-summary-extraction)
  - [Stage 2 - Global Statistics Aggregation](#stage-2---global-statistics-aggregation)
  - [Stage 3 – Site‑wise Harmonization](#stage-3---site-wise-harmonization)
- [Data Format Specification](#data-format-specification)
  - [Covariates File](#covariates-file)
    - [General Structure](#general-structure)
  - [Dependent Variables File](#dependent-variables-file)
    - [General Structure](#general-structure-1)
- [Configurations](#configurations)
  - [Parameters file](#parameters-file)
    - [Schema](#schema)
  - [Logs](#logs)
- [Output](#output)
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
1. Each participating site runs a local regression to obtain initial β‑coefficients by computing local cross-product matrices (XᵀX and Xᵀy).
2. Using those coefficients, the site computes its local mean and local variance.
3. These summary statistics—never raw data—are securely sent to the remote aggregator node.

### Stage 2 - Global Statistics Aggregation:
1. The aggregator combines all incoming summaries to derive the grand mean and grand variance across sites.
2. It broadcasts those global values back to every local node.

### Stage 3 - Site-wise Harmonization:
1. Each node uses the grand statistics to standardize its own dataset.
2. It then estimates site‑specific effects via non-parametric empirical Bayes and adjusts its data accordingly.
3. The result: harmonized, site‑neutral data that remain in place and ready for pooled analysis.


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
| `covariates_types` | `object` | ✅ | Maps each covariate column name to its type. Categorical (`str`) columns are dummy-encoded automatically. | `{"isControl": "bool", "age": "float", "sex": "str"}` |
| `covariates_types.['key_name']` | `string` | ✅ | Primitive type name for each covariate column. | `"int"`, `"float"`, `"str"`, `"bool"` |


> Note: In the dependent file, each cell value is assumed to be either empty or of type `float`.

### Logs:
The framework writes each site's messages to `<site-id>.log` in that site's output directory, and server-side messages to `aggregator.remote.log`. Set the optional `log_level` computation parameter to `debug`, `info`, `warning`, `error`, or `critical` (default `info`).

## Output:

Once the computation is completed, each site's output directory contains a harmonized CSV file named `harmonized_data.csv` and an `index.html` results page linking to it. This file has the same column structure as the input data file, with site-batch effects removed and values on the original measurement scale.

## Developer Instructions:
The computation logic lives in [`app/code/computation/`](app/code/computation/). The boilerplate owns `app/code/framework/`, `app/code/runtime/`, `app/config/`, `system/`, and the Dockerfiles; update those only by re-running the boilerplate's `scripts/migrate_computation.py`.

| File | Contents |
|------|----------|
| [`spec.py`](app/code/computation/spec.py) | Workflow declaration (three local/remote rounds plus the output step) |
| [`inputs.py`](app/code/computation/inputs.py), [`validation.py`](app/code/computation/validation.py) | Parameter checks, file loading, and type conversion |
| [`local_math.py`](app/code/computation/local_math.py) | Site-side encoding, interpolation, cross products, variance, and harmonization |
| [`remote_math.py`](app/code/computation/remote_math.py) | Aggregation of site summaries |
| [`results.py`](app/code/computation/results.py) | Output file writing |
| [`types.py`](app/code/computation/types.py) | Values exchanged between rounds and cached site state |

1. Run lint, format checks, and unit tests:
    > make check
2. Run a local NVFlare simulation (builds the dev image first; add `--no-build` for source-only changes):
    > ./run_local_simulation.sh site1,site2
3. Each site's results are written to `test_output/simulate_job/<site>/`.
4. Build the production image without publishing it:
    > ./dockerPush.sh --no-push
