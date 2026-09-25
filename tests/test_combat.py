import json
import logging
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_DIR, "app", "code"))

from computation import local_math, remote_math  # noqa: E402
from computation.inputs import load_inputs  # noqa: E402
from computation.results import (  # noqa: E402
    HARMONIZED_DATA_FILE,
    write_harmonized_data,
)
from computation.types import (  # noqa: E402
    GlobalRegression,
    LocalCrossProducts,
    LocalVariance,
    PooledVariance,
    SiteColumns,
    SiteRegistration,
    SiteState,
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
    for name, data_dir in site_dirs.items():
        result = local_math.prepare_site(load_inputs(data_dir, parameters, LOGGER))
        payloads[name] = _round_trip(result.payload, SiteRegistration)
        states[name] = _round_trip(result.state, SiteState)

    site_columns = _round_trip(remote_math.assign_site_columns(payloads), SiteColumns)
    for name in site_dirs:
        result = local_math.compute_local_cross_products(site_columns, states[name])
        payloads[name] = _round_trip(result.payload, LocalCrossProducts)
        states[name] = _round_trip(result.state, SiteState)

    regression = _round_trip(
        remote_math.compute_global_regression(payloads), GlobalRegression
    )
    for name in site_dirs:
        result = local_math.compute_local_variance(regression, states[name])
        payloads[name] = _round_trip(result.payload, LocalVariance)
        states[name] = _round_trip(result.state, SiteState)

    pooled = _round_trip(remote_math.compute_pooled_variance(payloads), PooledVariance)
    harmonized = {
        name: local_math.harmonize_site_data(pooled, states[name]) for name in site_dirs
    }
    return harmonized, states, pooled


def _test_sites():
    return {
        "site1": os.path.join(TEST_DATA_DIR, "site1"),
        "site2": os.path.join(TEST_DATA_DIR, "site2"),
    }


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
        harmonized, states, pooled = run_combat(NUMERIC_PARAMETERS, _test_sites())

        with tempfile.TemporaryDirectory() as output_dir:
            write_harmonized_data(pooled, states["site1"], output_dir, LOGGER)
            written = pd.read_csv(os.path.join(output_dir, HARMONIZED_DATA_FILE))

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
                "write_harmonized_data",
            ],
        )


if __name__ == "__main__":
    unittest.main()
