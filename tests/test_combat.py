import json
import logging
import os
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_DIR, "app", "code"))

from computation import local_math, remote_math  # noqa: E402
from computation.diagnostics import collect_findings  # noqa: E402
from computation.inputs import load_inputs  # noqa: E402
from computation.report import build_report  # noqa: E402
from computation.results import (  # noqa: E402
    HARMONIZED_DATA_FILE,
    RESULTS_PAGE_FILE,
    write_outputs,
)
from computation.types import (  # noqa: E402
    CrossSiteSummaries,
    DesignLayout,
    GlobalRegression,
    LocalCrossProducts,
    LocalVariance,
    PooledVariance,
    SiteRegistration,
    SiteState,
    SiteSummaryShare,
)
from framework.serialization import deserialize_value, serialize_value  # noqa: E402
from framework.workflow import get_task_names  # noqa: E402

TEST_DATA_DIR = os.path.join(REPO_DIR, "test_data")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
LOGGER = logging.getLogger("combat-tests")

NUMERIC_PARAMETERS = {
    "covariate_file": "Covariate.csv",
    "data_file": "Data.csv",
    "combat_algo": "combatDC",
    "covariates_types": {"isControl": "bool", "age": "float"},
}
CATEGORICAL_PARAMETERS = {
    "covariate_file": "CatCovariate.csv",
    "data_file": "Data.csv",
    "combat_algo": "combatDC",
    "covariates_types": {"isControl": "bool", "age": "float", "sex": "str"},
}
MEGA_PARAMETERS = {
    "covariate_file": "Covariate.csv",
    "data_file": "MissingData.csv",
    "combat_algo": "combatMegaDC",
    "covariates_types": {"isControl": "bool", "age": "float"},
}


def _round_trip(value, expected_type):
    """Serialize and rebuild a value the way the runtime transports it."""
    encoded = json.loads(json.dumps(serialize_value(value)))
    return deserialize_value(encoded, expected_type)


def run_combat(parameters, site_dirs):
    """Run every workflow step in-process; site_dirs maps display name to dir."""
    payloads, states = {}, {}
    declared = parameters.get("categorical_levels")
    for name, data_dir in site_dirs.items():
        result = local_math.prepare_site(
            load_inputs(data_dir, parameters, LOGGER), declared
        )
        payloads[name] = _round_trip(result.payload, SiteRegistration)
        states[name] = _round_trip(result.state, SiteState)

    layout = _round_trip(
        remote_math.assign_design_layout(payloads, declared), DesignLayout
    )
    for name in site_dirs:
        result = local_math.compute_local_cross_products(layout, states[name])
        payloads[name] = _round_trip(result.payload, LocalCrossProducts)
        states[name] = _round_trip(result.state, SiteState)

    regression = _round_trip(
        remote_math.compute_global_regression(payloads, LOGGER), GlobalRegression
    )
    for name in site_dirs:
        result = local_math.compute_local_variance(regression, states[name])
        payloads[name] = _round_trip(result.payload, LocalVariance)
        states[name] = _round_trip(result.state, SiteState)

    pooled = _round_trip(remote_math.compute_pooled_variance(payloads), PooledVariance)
    sharing = {
        key: parameters[key]
        for key in ("share_site_summaries", "min_site_summary_n")
        if key in parameters
    }
    for name in site_dirs:
        result = local_math.harmonize_site(pooled, states[name], **sharing)
        payloads[name] = _round_trip(result.payload, SiteSummaryShare)
        states[name] = _round_trip(result.state, SiteState)

    summaries = _round_trip(
        remote_math.collect_site_summaries(
            payloads, sharing.get("share_site_summaries", False)
        ),
        CrossSiteSummaries,
    )
    harmonized = {name: states[name].harmonization.harmonized for name in site_dirs}
    return harmonized, states, summaries


def _test_sites():
    return {
        "site1": os.path.join(TEST_DATA_DIR, "site1"),
        "site2": os.path.join(TEST_DATA_DIR, "site2"),
    }


class _ModifiedSites:
    """Copy the test sites to a temporary directory and apply edits.

    ``edits`` maps a site name to a function taking and returning
    ``(covariates, data)`` DataFrames read from ``covariate_file`` and
    ``data_file``.
    """

    def __init__(self, parameters, edits):
        self.parameters = parameters
        self.edits = edits

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        sites = {}
        for name, source in _test_sites().items():
            target = os.path.join(self._tmp.name, name)
            shutil.copytree(source, target)
            if name in self.edits:
                covariate_path = os.path.join(target, self.parameters["covariate_file"])
                data_path = os.path.join(target, self.parameters["data_file"])
                covariates, data = self.edits[name](
                    pd.read_csv(covariate_path), pd.read_csv(data_path)
                )
                covariates.to_csv(covariate_path, index=False)
                data.to_csv(data_path, index=False)
            sites[name] = target
        return sites

    def __exit__(self, *exc):
        self._tmp.cleanup()


def _titles(state, summaries, parameters):
    return [f.title for f in collect_findings(state, summaries, parameters)]


class CombatWorkflowTests(unittest.TestCase):
    def test_numeric_covariates_match_legacy_implementation(self):
        harmonized, _, _ = run_combat(NUMERIC_PARAMETERS, _test_sites())

        for name in ("site1", "site2"):
            expected = pd.read_csv(
                os.path.join(FIXTURE_DIR, "legacy_combatdc_numeric", f"{name}.csv")
            )
            self.assertEqual(list(harmonized[name].columns), list(expected.columns))
            np.testing.assert_allclose(
                harmonized[name].to_numpy(), expected.to_numpy(), rtol=1e-10
            )

    def test_mega_dc_interpolates_missing_values(self):
        harmonized, _, _ = run_combat(MEGA_PARAMETERS, _test_sites())

        for name, data_dir in _test_sites().items():
            source = pd.read_csv(os.path.join(data_dir, "MissingData.csv"))
            self.assertTrue(source.isna().any().any())
            self.assertEqual(harmonized[name].shape, source.shape)
            self.assertFalse(harmonized[name].isna().any().any())

    def test_each_site_sets_only_its_own_indicator_column(self):
        # "site1" is a substring of "site10"; display names also differ from
        # the directory names the sites were loaded from.
        sites = {
            "site10": os.path.join(TEST_DATA_DIR, "site1"),
            "site1": os.path.join(TEST_DATA_DIR, "site2"),
        }
        _, states, _ = run_combat(NUMERIC_PARAMETERS, sites)

        for name, state in states.items():
            indicators = local_math.build_design(state)[["site_site1", "site_site10"]]
            expected_column = f"site_{name}"
            self.assertTrue((indicators[expected_column] == 1).all())
            self.assertTrue((indicators.drop(columns=expected_column) == 0).all().all())

    def test_output_step_writes_csv_without_index(self):
        harmonized, states, summaries = run_combat(NUMERIC_PARAMETERS, _test_sites())

        data_dir = _test_sites()["site1"]
        with tempfile.TemporaryDirectory() as output_dir:
            write_outputs(
                summaries,
                states["site1"],
                NUMERIC_PARAMETERS,
                data_dir,
                output_dir,
                LOGGER,
            )
            written = pd.read_csv(os.path.join(output_dir, HARMONIZED_DATA_FILE))
            with open(os.path.join(output_dir, RESULTS_PAGE_FILE)) as page:
                results_page = page.read()
            with open(os.path.join(output_dir, "Covariate.csv"), "rb") as copy:
                copied = copy.read()
        with open(os.path.join(data_dir, "Covariate.csv"), "rb") as original:
            self.assertEqual(copied, original.read())

        self.assertIn(f'href="{HARMONIZED_DATA_FILE}"', results_page)
        self.assertIn('href="Covariate.csv"', results_page)
        self.assertEqual(list(written.columns), list(harmonized["site1"].columns))
        np.testing.assert_allclose(
            written.to_numpy(), harmonized["site1"].to_numpy(), rtol=1e-12
        )


class InputValidationTests(unittest.TestCase):
    def test_combat_dc_rejects_missing_data(self):
        parameters = dict(MEGA_PARAMETERS, combat_algo="combatDC")
        with self.assertRaisesRegex(ValueError, "combatMegaDC"):
            load_inputs(_test_sites()["site1"], parameters, LOGGER)

    def test_rejects_unknown_algorithm(self):
        parameters = dict(NUMERIC_PARAMETERS, combat_algo="combat")
        with self.assertRaisesRegex(ValueError, "Invalid combat_algo"):
            load_inputs(_test_sites()["site1"], parameters, LOGGER)

    def test_rejects_missing_covariate_column(self):
        parameters = dict(
            NUMERIC_PARAMETERS,
            covariates_types={"isControl": "bool", "age": "float", "sex": "str"},
        )
        with self.assertRaisesRegex(ValueError, "sex"):
            load_inputs(_test_sites()["site1"], parameters, LOGGER)

    def test_rejects_missing_parameter(self):
        parameters = dict(NUMERIC_PARAMETERS)
        del parameters["data_file"]
        with self.assertRaisesRegex(ValueError, "data_file"):
            load_inputs(_test_sites()["site1"], parameters, LOGGER)


class SiteSummarySharingTests(unittest.TestCase):
    def test_sharing_is_disabled_by_default(self):
        _, states, summaries = run_combat(NUMERIC_PARAMETERS, _test_sites())

        self.assertFalse(summaries.enabled)
        self.assertEqual(summaries.sites, {})
        report = build_report(
            states["site1"], summaries, NUMERIC_PARAMETERS, HARMONIZED_DATA_FILE
        )
        self.assertIn("Cross-site comparison is disabled", report)

    def test_sharing_does_not_change_harmonized_values(self):
        parameters = dict(NUMERIC_PARAMETERS, share_site_summaries=True)
        shared, _, _ = run_combat(parameters, _test_sites())
        unshared, _, _ = run_combat(NUMERIC_PARAMETERS, _test_sites())

        for name in shared:
            np.testing.assert_array_equal(
                shared[name].to_numpy(), unshared[name].to_numpy()
            )

    def test_shared_summaries_match_site_data(self):
        parameters = dict(NUMERIC_PARAMETERS, share_site_summaries=True)
        harmonized, states, summaries = run_combat(parameters, _test_sites())

        self.assertTrue(summaries.enabled)
        self.assertEqual(sorted(summaries.sites), ["site1", "site2"])
        for name, summary in summaries.sites.items():
            self.assertEqual(summary.sample_count, len(states[name].data))
            np.testing.assert_allclose(
                summary.mean_before, states[name].data.mean().to_numpy()
            )
            np.testing.assert_allclose(
                summary.sd_after, harmonized[name].std().to_numpy()
            )

        report = build_report(
            states["site1"], summaries, parameters, HARMONIZED_DATA_FILE
        )
        self.assertIn("Between-site spread", report)

    def test_small_sites_withhold_their_summaries(self):
        # site1 has 25 subjects and site2 has 24.
        parameters = dict(
            NUMERIC_PARAMETERS, share_site_summaries=True, min_site_summary_n=25
        )
        _, states, summaries = run_combat(parameters, _test_sites())

        self.assertEqual(list(summaries.sites), ["site1"])
        self.assertEqual(summaries.withheld_sites, ["site2"])
        report = build_report(
            states["site2"], summaries, parameters, HARMONIZED_DATA_FILE
        )
        self.assertIn("summaries were not shared", report)
        self.assertIn("Fewer than two sites shared summaries", report)

    def test_rejects_invalid_sharing_parameters(self):
        for bad in (
            {"share_site_summaries": "true"},
            {"min_site_summary_n": 1},
            {"min_site_summary_n": 5.5},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_inputs(
                    _test_sites()["site1"], dict(NUMERIC_PARAMETERS, **bad), LOGGER
                )


class ReportTests(unittest.TestCase):
    def test_well_conditioned_run_reports_no_problems(self):
        _, states, summaries = run_combat(NUMERIC_PARAMETERS, _test_sites())

        self.assertEqual(states["site1"].regression.design_rank, 4)
        report = build_report(
            states["site1"], summaries, NUMERIC_PARAMETERS, HARMONIZED_DATA_FILE
        )
        self.assertIn("No problems detected", report)
        self.assertIn("<svg", report)

    def test_covariate_confounded_with_site_is_flagged_with_recommendation(self):
        parameters = dict(
            NUMERIC_PARAMETERS,
            covariates_types={"isControl": "bool", "age": "float", "scanner": "int"},
        )

        def scanner(value):
            return lambda cov, data: (cov.assign(scanner=value), data)

        with _ModifiedSites(
            parameters, {"site1": scanner(1), "site2": scanner(2)}
        ) as sites:
            _, states, summaries = run_combat(parameters, sites)

        regression = states["site1"].regression
        self.assertLess(regression.design_rank, regression.design_columns)
        self.assertIn(
            ["scanner", "site_site1", "site_site2"],
            regression.dependent_column_groups,
        )
        findings = collect_findings(states["site1"], summaries, parameters)
        self.assertEqual([f.level for f in findings], ["error"])
        design_error = findings[0]
        self.assertEqual(design_error.level, "error")
        self.assertIn("scanner", design_error.detail)
        self.assertIn(
            "'scanner' cannot be separated from site", design_error.recommendation
        )
        report = build_report(
            states["site1"], summaries, parameters, HARMONIZED_DATA_FILE
        )
        self.assertIn("rank-deficient", report)
        self.assertIn("Recommendation", report)
        self.assertNotIn("No problems detected", report)
        self.assertIn("No harmonized data was written", report)

        with tempfile.TemporaryDirectory() as output_dir:
            write_outputs(
                summaries,
                states["site1"],
                parameters,
                _test_sites()["site1"],
                output_dir,
                LOGGER,
            )
            self.assertEqual(
                sorted(os.listdir(output_dir)), ["Covariate.csv", RESULTS_PAGE_FILE]
            )

    def test_interpolation_is_reported(self):
        _, states, summaries = run_combat(MEGA_PARAMETERS, _test_sites())

        counts = states["site1"].interpolated_counts
        self.assertEqual(
            sum(counts.values()),
            int(
                pd.read_csv(os.path.join(TEST_DATA_DIR, "site1", "MissingData.csv"))
                .isna()
                .sum()
                .sum()
            ),
        )
        report = build_report(
            states["site1"], summaries, MEGA_PARAMETERS, HARMONIZED_DATA_FILE
        )
        self.assertIn("were interpolated", report)

    def test_pills_only_for_sites_whose_results_are_shown(self):
        def pills(report):
            return re.findall(r'<span class="pill[^"]*">([^<]*)</span>', report)

        _, states, summaries = run_combat(NUMERIC_PARAMETERS, _test_sites())
        report = build_report(
            states["site1"], summaries, NUMERIC_PARAMETERS, HARMONIZED_DATA_FILE
        )
        self.assertEqual(pills(report), ["site1"])
        self.assertIn("Sites <b>2</b>", report)

        shared = dict(NUMERIC_PARAMETERS, share_site_summaries=True)
        _, states, summaries = run_combat(shared, _test_sites())
        report = build_report(states["site1"], summaries, shared, HARMONIZED_DATA_FILE)
        self.assertEqual(pills(report), ["site1", "site2"])

        # site2 (24 subjects) withholds its summaries.
        withheld = dict(shared, min_site_summary_n=25)
        _, states, summaries = run_combat(withheld, _test_sites())
        for name, expected in (("site1", ["site1"]), ("site2", ["site2"])):
            report = build_report(
                states[name], summaries, withheld, HARMONIZED_DATA_FILE
            )
            self.assertEqual(pills(report), expected)

    def test_report_escapes_site_and_column_names(self):
        sites = {
            "<b>evil</b>": _test_sites()["site1"],
            "site2": _test_sites()["site2"],
        }
        _, states, summaries = run_combat(NUMERIC_PARAMETERS, sites)
        state = states["<b>evil</b>"]
        state.data = state.data.rename(
            columns={state.data.columns[0]: "<script>x</script>"}
        )

        report = build_report(state, summaries, NUMERIC_PARAMETERS, "a&b.csv")
        self.assertNotIn("<b>evil</b>", report)
        self.assertNotIn("<script>x</script>", report)
        self.assertIn("&lt;b&gt;evil&lt;/b&gt;", report)
        self.assertIn("a&amp;b.csv", report)


class CategoricalCovariateTests(unittest.TestCase):
    def test_categorical_covariate_gives_full_rank_design(self):
        harmonized, states, summaries = run_combat(
            CATEGORICAL_PARAMETERS, _test_sites()
        )

        state = states["site1"]
        self.assertEqual(state.layout.category_levels, {"sex": ["female", "male"]})
        self.assertEqual(
            state.regression.design_column_names,
            ["isControl", "age", "sex_male", "site_site1", "site_site2"],
        )
        self.assertEqual(state.regression.design_rank, 5)
        for name, frame in harmonized.items():
            self.assertTrue((frame.to_numpy() >= 0).all(), name)
        self.assertIn(
            "No problems detected.",
            _titles(state, summaries, CATEGORICAL_PARAMETERS),
        )

    def test_healthy_default_run_reports_no_problems_first(self):
        with open(os.path.join(TEST_DATA_DIR, "server", "parameters.json")) as file:
            parameters = dict(json.load(file), share_site_summaries=True)
        _, states, summaries = run_combat(parameters, _test_sites())

        for state in states.values():
            findings = collect_findings(state, summaries, parameters)
            self.assertEqual(findings[0].title, "No problems detected.")
            self.assertEqual(
                {f.level for f in findings[1:]}, {"info"}, [f.title for f in findings]
            )

    def test_reference_level_does_not_change_harmonized_values(self):
        discovered, _, _ = run_combat(CATEGORICAL_PARAMETERS, _test_sites())
        parameters = dict(
            CATEGORICAL_PARAMETERS, categorical_levels={"sex": ["male", "female"]}
        )
        declared, states, _ = run_combat(parameters, _test_sites())

        self.assertEqual(
            states["site1"].regression.design_column_names[:3],
            ["isControl", "age", "sex_female"],
        )
        for name in discovered:
            np.testing.assert_allclose(
                declared[name].to_numpy(), discovered[name].to_numpy(), rtol=1e-10
            )

    def test_declared_levels_are_not_reported_by_sites(self):
        parameters = dict(
            CATEGORICAL_PARAMETERS, categorical_levels={"sex": ["female", "male"]}
        )
        inputs = load_inputs(_test_sites()["site1"], parameters, LOGGER)
        registration = local_math.prepare_site(inputs, parameters["categorical_levels"])

        self.assertEqual(registration.payload.category_levels, {})

    def test_site_with_a_single_level_uses_the_shared_columns(self):
        def only_male(cov, data):
            keep = cov["sex"] == "male"
            return cov[keep], data[keep]

        with _ModifiedSites(CATEGORICAL_PARAMETERS, {"site2": only_male}) as sites:
            _, states, summaries = run_combat(CATEGORICAL_PARAMETERS, sites)

        self.assertEqual(
            list(states["site2"].covariates.columns), ["isControl", "age", "sex_male"]
        )
        self.assertEqual(states["site2"].regression.design_rank, 5)
        self.assertTrue(
            any(
                "same value for every subject" in title and "sex = male" in title
                for title in _titles(states["site2"], summaries, CATEGORICAL_PARAMETERS)
            )
        )

    def test_level_found_at_only_one_site_is_flagged(self):
        def add_other(cov, data):
            cov = cov.copy()
            cov.loc[:2, "sex"] = "other"
            return cov, data

        with _ModifiedSites(CATEGORICAL_PARAMETERS, {"site1": add_other}) as sites:
            _, states, summaries = run_combat(CATEGORICAL_PARAMETERS, sites)

        self.assertEqual(
            states["site1"].layout.level_site_counts["sex"],
            {"female": 2, "male": 2, "other": 1},
        )
        site1 = _titles(states["site1"], summaries, CATEGORICAL_PARAMETERS)
        site2 = _titles(states["site2"], summaries, CATEGORICAL_PARAMETERS)
        self.assertTrue(any("'other'" in t and "only at this site" in t for t in site1))
        self.assertFalse(any("only at this site" in t for t in site2))

    def test_inconsistent_level_spelling_is_flagged(self):
        def capitalize(cov, data):
            return cov.assign(sex=cov["sex"].str.capitalize()), data

        with _ModifiedSites(CATEGORICAL_PARAMETERS, {"site2": capitalize}) as sites:
            _, states, summaries = run_combat(CATEGORICAL_PARAMETERS, sites)

        findings = collect_findings(states["site1"], summaries, CATEGORICAL_PARAMETERS)
        spelling = [f for f in findings if "capitalization" in f.title]
        self.assertEqual(len(spelling), 1)
        self.assertIn("'Female', 'female'", spelling[0].detail)
        self.assertIn("categorical_levels", spelling[0].recommendation)

    def test_values_outside_declared_levels_are_rejected(self):
        parameters = dict(
            CATEGORICAL_PARAMETERS, categorical_levels={"sex": ["female"]}
        )
        with self.assertRaisesRegex(ValueError, "not listed in categorical_levels"):
            load_inputs(_test_sites()["site1"], parameters, LOGGER)

    def test_rejects_invalid_declared_levels(self):
        for bad in (
            {"age": ["young"]},
            {"sex": []},
            {"sex": ["male", "male"]},
            {"sex": "male"},
        ):
            parameters = dict(CATEGORICAL_PARAMETERS, categorical_levels=bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_inputs(_test_sites()["site1"], parameters, LOGGER)


class SiteRecommendationTests(unittest.TestCase):
    def test_unit_mismatch_is_flagged_for_the_affected_site(self):
        parameters = dict(NUMERIC_PARAMETERS, share_site_summaries=True)

        def to_cubic_mm(cov, data):
            return cov, data * 1000

        with _ModifiedSites(parameters, {"site2": to_cubic_mm}) as sites:
            _, states, summaries = run_combat(parameters, sites)

        site2 = collect_findings(states["site2"], summaries, parameters)
        mismatch = [f for f in site2 if "orders of magnitude" in f.title]
        self.assertEqual(len(mismatch), 1)
        self.assertIn("×1000", mismatch[0].detail)
        self.assertIn("units", mismatch[0].recommendation)

    def test_small_site_and_heavy_missing_data_are_flagged(self):
        def shrink_and_blank(cov, data):
            data = data.head(8).copy()
            data.iloc[:4, 0] = np.nan
            return cov.head(8), data

        with _ModifiedSites(MEGA_PARAMETERS, {"site2": shrink_and_blank}) as sites:
            _, states, summaries = run_combat(MEGA_PARAMETERS, sites)

        site2 = collect_findings(states["site2"], summaries, MEGA_PARAMETERS)
        titles = [f.title for f in site2]
        self.assertIn("This site has only 8 subjects.", titles)
        heavy = [f for f in site2 if "missing more than" in f.title]
        self.assertEqual(len(heavy), 1)
        self.assertIn("Left-Lateral-Ventricle (50%)", heavy[0].detail)
        self.assertNotIn(
            "This site has only 8 subjects.",
            _titles(states["site1"], summaries, MEGA_PARAMETERS),
        )

    def test_every_warning_and_error_has_a_recommendation(self):
        parameters = dict(CATEGORICAL_PARAMETERS, share_site_summaries=True)

        def troublesome(cov, data):
            cov = cov.head(8).copy()
            cov.loc[:1, "sex"] = "Male"
            return cov, data.head(8) * 1000

        with _ModifiedSites(parameters, {"site2": troublesome}) as sites:
            _, states, summaries = run_combat(parameters, sites)

        for name, state in states.items():
            for finding in collect_findings(state, summaries, parameters):
                if finding.level in ("error", "warning"):
                    self.assertTrue(finding.recommendation, (name, finding.title))


class SpecTests(unittest.TestCase):
    def test_runtime_rebuilds_local_state_as_site_state(self):
        from computation.spec import SPEC

        self.assertIs(SPEC.workflow.local_state_type, SiteState)

    def test_task_names(self):
        from computation.spec import SPEC

        self.assertEqual(
            get_task_names(SPEC.workflow),
            [
                "prepare_site",
                "compute_local_cross_products",
                "compute_local_variance",
                "harmonize_site",
                "write_outputs",
            ],
        )


if __name__ == "__main__":
    unittest.main()
