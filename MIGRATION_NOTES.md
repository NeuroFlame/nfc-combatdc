# Migration to computation-nvflare-boilerplate

This computation was migrated from the hand-written NVFlare
`Controller`/`Executor`/`Aggregator` architecture (NVFlare 2.4, Python 3.8) to
[`computation-nvflare-boilerplate`](https://github.com/NeuroFlame/computation-nvflare-boilerplate)
`0.1.0` (NVFlare 2.8, Python 3.11). Framework-owned files were applied with the
boilerplate's `scripts/migrate_computation.py --in-place --force`; the ComBat
math was ported into `app/code/computation/`.

## Workflow

The old four tasks map to a `stepped_workflow` with three local/remote pairs
and a final output step:

| Old task | New local step | New remote step |
|---|---|---|
| `perform_local_client_step1` | `prepare_site` (with `load_inputs`) | `assign_site_columns` |
| `perform_local_client_step2` | `compute_local_cross_products` | `compute_global_regression` |
| `perform_local_client_step3` | `compute_local_variance` | `compute_pooled_variance` |
| `perform_local_client_step4` | `write_harmonized_data` (output step) | — |

Input validation, covariate encoding, and missing-data interpolation now run
in the first round instead of the second, so invalid inputs fail before any
summaries are exchanged.

## Site identity

The old code read each site's own NVFlare client name to (a) pick its one-hot
site-indicator column by substring match and (b) derive a numeric
`site_index` from the digits in that name. The framework does not expose a
site's own identity, and both mechanisms were already fragile: the substring
match confuses `site1` with `site10`, and it silently fails when the
display names from `site_id_name_map` differ from the raw site IDs.

Each site now mints a random token in `prepare_site`; the aggregator maps
tokens to `site_<display name>` columns and broadcasts the mapping. The
`site_index`/`site_array` mechanism was only used to select rows of the
broadcast standardized-mean matrix, all of whose columns equal the grand mean,
so it was replaced by broadcasting the grand mean directly. This also stops
sending a features × total-samples matrix to every site.

## Other intentional differences

- **Output file** is `harmonized_data.csv` (was
  `harmonized_site_{site_index}_data.csv`); each site already has its own
  output directory. The CSV header now contains the plain input column names;
  the old header wrote them as tuple strings such as
  `"('Left-Lateral-Ventricle',)"`.
- **Validation messages** list the missing covariate columns and invalid rows,
  and an unknown `combat_algo` or column type is rejected up front.
- **Logging** uses the framework logger (`log_level` parameter) instead of the
  JSON logger and `LOG_LEVEL` environment variable. Data values are no longer
  logged.
- **Unused code removed**: numba regression helpers and unused
  empirical-Bayes prior computations. None fed into the output.
- **Dependencies**: trimmed to boilerplate pins plus scipy, statsmodels,
  scikit-learn, and their direct dependencies.

## Verification

- **Numeric parity** against the pre-migration implementation (run in its
  original Python 3.8/NVFlare 2.4 image) with numeric covariates
  (`Covariate.csv`, `Data.csv`, `combatDC`): max relative difference ~1.7e-14 in
  the NVFlare 2.8 simulator. This output is kept as a regression fixture in
  `tests/fixtures/legacy_combatdc_numeric/`.
- **`combatMegaDC`** with the default `parameters.json` runs end to end in the
  simulator and produces NaN-free output of the input's shape.
- `make check`, `migrate_computation.py --check` (0 differing paths), and
  `./dockerPush.sh --no-push` pass.

## Known issue carried over (deferred)

**Status:** deferred for a later revisit; out of scope for this migration.


With a categorical covariate, `encode_covariates` keeps every one-hot level,
for example both `sex_female` and `sex_male`. Their sum equals the sum of the site
indicator columns, so the global XᵀX is singular (rank 5 of 6, condition
number ~5e19 on the test data), and `np.linalg.inv` returns values dominated
by round-off. The default `parameters.json` hits this case. Both the old and
new implementations produce implausible values here (for example, negative
ventricle volumes), and the two outputs do not match because round-off noise
is not reproducible. Fixing this, for example by dropping one level per
categorical covariate, changes the algorithm and needs a separate decision.
Until then, only covariate sets without categorical (`str`) columns, such as
`Covariate.csv` in the test data, give meaningful results.
