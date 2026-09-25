"""Declare the Decentralized ComBat workflow."""

from framework import (
    ComputationSpec,
    local_step,
    remote_step,
    site_output_step,
    stepped_workflow,
)

from .inputs import load_inputs
from .local_math import (
    compute_local_cross_products,
    compute_local_variance,
    prepare_site,
)
from .remote_math import (
    assign_site_columns,
    compute_global_regression,
    compute_pooled_variance,
)
from .results import write_harmonized_data

SPEC = ComputationSpec(
    workflow=stepped_workflow(
        local_step(fn=prepare_site, input_fn=load_inputs),
        remote_step(fn=assign_site_columns),
        local_step(fn=compute_local_cross_products),
        remote_step(fn=compute_global_regression),
        local_step(fn=compute_local_variance),
        remote_step(fn=compute_pooled_variance),
        site_output_step(fn=write_harmonized_data),
    ),
)
