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

The final step was later split into a fourth round and a new output step; see
[Post-migration changes](#post-migration-changes).

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
  output directory. The `index.html` results page added in #4 links to it.
  The CSV header now contains the plain input column names; the old header
  wrote them as tuple strings such as `"('Left-Lateral-Ventricle',)"`.
- **Validation messages** list the missing covariate columns and invalid rows,
  and an unknown `combat_algo` or column type is rejected up front.
- **Logging** uses the framework logger instead of the JSON logger. The
  `log_level` computation parameter added in #4 is read by the framework.
  Data values are no longer logged.
- **Unused code removed**: numba regression helpers and unused
  empirical-Bayes prior computations. None fed into the output.
- **Dependencies**: trimmed to boilerplate pins plus scipy, statsmodels,
  scikit-learn, and their direct dependencies. (scikit-learn was later
  removed; see [Post-migration changes](#post-migration-changes).)

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

## Known issue carried over (resolved)

**Status:** resolved after the migration by shared categorical levels with a
reference level; see
[Categorical covariates and per-site recommendations](#categorical-covariates-and-per-site-recommendations-2026-09-25).
The original description follows.

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

## Post-migration changes

### Results report and optional cross-site comparison (2026-09-25)

The minimal `index.html` from #4, which was a single download link, was
replaced by a full results report (`app/code/computation/report.py`),
comparable to the other NeuroFLAME computations. The report is self-contained,
with inline SVG charts and no external scripts, so it also works on sites
without network access. It shows run diagnostics, the per-ROI site effects
(γ\*, δ\*), this site's before/after statistics, and the run settings.

Workflow changes:

| Round | Local step | Remote step |
|---|---|---|
| 1–3 | unchanged | unchanged |
| 4 (new) | `harmonize_site` | `collect_site_summaries` |
| output | `write_outputs` (was `write_harmonized_data`) | — |

- **Harmonization moved** from the output step into `harmonize_site`, so sites
  can share post-harmonization statistics before the output step. The
  harmonized values are unchanged; the legacy-parity test still passes.
- **Design diagnostics**: `compute_global_regression` now also returns the rank
  and condition number of the pooled XᵀX, and logs a warning when XᵀX is
  rank-deficient or ill-conditioned (condition number > 1e10).
- **New parameters**:
  - `share_site_summaries` (default `false`): when `true`, each site shares
    its per-ROI means and SDs before and after harmonization, plus its subject
    count, and the aggregator sends them to every site for the report's
    cross-site comparison. With sharing enabled, every site can see the other
    sites' summary statistics; previously only the aggregator saw per-site
    aggregates (through Xᵀy). The framework fixes a workflow's rounds when the
    computation is loaded, so round 4 always runs; when sharing is disabled it
    carries empty payloads.
  - `min_site_summary_n` (default `10`, minimum 2): sites with fewer subjects
    withhold their summaries even when sharing is enabled, because summary
    statistics from very small sites reveal more about individuals. The report
    lists withheld sites.
- Both parameters are validated with the other inputs in the first round.
- Tests cover default-off sharing, unchanged harmonized values with sharing on,
  withholding, rank-deficiency flagging, interpolation reporting, and HTML
  escaping of site and column names.

### Categorical covariates and per-site recommendations (2026-09-25)

**Categorical covariates (resolves the known issue above).** Two problems are
fixed together:

- **Rank deficiency.** Every level of a categorical covariate was encoded, so
  the levels summed to the same column as the site indicators and XᵀX was
  singular. Each categorical covariate now drops a reference level. On the
  test data with `sex`, XᵀX is full rank (5 of 5, condition number ~5e4 instead
  of ~5e19) and the harmonized values are plausible. The harmonized data does
  not depend on which level is the reference; a test checks this.
- **Inconsistent encoding across sites.** Each site built its categories from
  its own data (scikit-learn's `OneHotEncoder`). A site whose subjects all had
  one level encoded fewer columns and the run crashed; with the same number of
  different levels, the columns would silently misalign.
  - In round 1, each site now reports the levels it observes, and the
    aggregator returns the sorted union, reference level first, in the new
    `DesignLayout` (which replaces `SiteColumns`).
  - Encoding and `combatMegaDC` interpolation moved from round 1 to round 2 so
    they use the shared levels.
  - Sites also send their design column names, and the aggregator stops if
    they differ.
  - The aggregator now sees which levels each site has, and every site sees the
    combined list and how many sites have each level. A new optional
    `categorical_levels` parameter declares levels (and the reference) instead;
    sites then do not report them, and a site with an unlisted value stops with
    an error.

scikit-learn, joblib, and threadpoolctl were removed from `requirements.txt`.
Results for numeric-only covariates are unchanged; the legacy-parity test
still passes.

**Rank-deficient designs.** A covariate exactly confounded with site, for
example a scanner ID that is constant within each site, makes XᵀX exactly
singular. Previously `np.linalg.inv` raised and the run failed with only
"Singular matrix". The aggregator now uses the pseudo-inverse in that case so
the run finishes, every site receives a report naming the cause, and
`harmonized_data.csv` is not written because its values would be unreliable.
Full-rank designs still use the ordinary inverse.

**Per-site recommendations.** A new `diagnostics.py` produces the report's
"Diagnostics and recommendations" section. Every error and warning carries a
recommendation, and a test checks this:

- The aggregator names the (nearly) linearly dependent design columns, so the
  report can say which covariate duplicates site membership or another
  covariate.
- New checks cover:
  - a categorical level that occurs only at this site;
  - levels that differ only in capitalization or spacing;
  - a categorical covariate with a single level;
  - covariates that are constant at this site;
  - sites with fewer than 20 subjects;
  - ROIs with more than 20% of values interpolated;
  - large site effects;
  - with sharing enabled, likely unit mismatches and ROIs whose between-site
    spread increased by at least 0.1 pooled SD. A 0.01 threshold flagged
    noise-level changes on healthy test data.
- Checks of the harmonized values are skipped when the design is
  rank-deficient, so symptoms don't crowd out the cause.

### Report site labels and covariate copy (2026-09-25)

- The report header shows a site label only for sites whose results appear in
  the report: the site itself, plus the other sites in the cross-site
  comparison when it is shown (sharing enabled and at least two sites shared).
  The site count remains.
- Each site's output directory also gets an unchanged copy of that site's own
  covariate file, under its original file name, so it can be paired with
  `harmonized_data.csv` row by row. The copy is written even when the
  harmonized data is not. It is a local file and is not sent to the aggregator
  or other sites. If its name would replace another output file, it is
  prefixed with `covariates_`.
