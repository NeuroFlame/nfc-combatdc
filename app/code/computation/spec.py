"""Declare the Decentralized ComBat workflow.

The final local/remote pair always runs. It only carries site summaries when
``share_site_summaries`` is enabled, because the framework fixes the workflow's
rounds when the computation is loaded.
"""

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
    harmonize_site,
    prepare_site,
)
from .remote_math import (
    assign_design_layout,
    collect_site_summaries,
    compute_global_regression,
    compute_pooled_variance,
)
from .results import write_outputs

SPEC = ComputationSpec(
    workflow=stepped_workflow(
        local_step(fn=prepare_site, input_fn=load_inputs),
        remote_step(fn=assign_design_layout),
        local_step(fn=compute_local_cross_products),
        remote_step(fn=compute_global_regression),
        local_step(fn=compute_local_variance),
        remote_step(fn=compute_pooled_variance),
        local_step(fn=harmonize_site),
        remote_step(fn=collect_site_summaries),
        site_output_step(fn=write_outputs),
    ),
)
