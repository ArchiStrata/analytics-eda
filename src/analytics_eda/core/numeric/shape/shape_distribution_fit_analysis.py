# Copyright 2025 ArchiStrata, LLC and Andrew Dabrowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shape distribution-fit analysis for numeric series."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context
from analytics_eda.core.visualization.validation import numeric_validator

from .shape_ecdf_vs_cdf_plot import ShapeECDFvsCDFContext, ShapeECDFvsCDFPlot
from .shape_qq_fit_plot import ShapeQqFitContext, ShapeQqFitPlot


@dataclass
class ShapeDistributionFitAnalysisContext(AnalysisContext):
    """Context for running distribution fit diagnostics (ECDF vs. CDF, Q–Q)."""

    report_name: str = "shape_distribution_fit_analysis"
    report_relative_path: str = "shape"
    report_file_name: str = "shape_distribution_fit_analysis.json"

    distribution_names: Sequence[str] = ("norm", "lognorm", "gamma", "expon")
    ecdf_vs_cdf_context: ShapeECDFvsCDFContext | None = None
    qq_fit_context: ShapeQqFitContext | None = None


class ShapeDistributionFitAnalysis(BaseAnalysis):
    """
    Run ECDF-vs-CDF and Q–Q diagnostics across candidate distributions.

    Big idea:
        Quantify how well the series aligns with common theoretical distributions
        via side-by-side ECDF/CDF comparisons and Q–Q fits for each candidate.
    """

    semantic_version = "1.0.0"
    context: ShapeDistributionFitAnalysisContext

    def __init__(self, context: ShapeDistributionFitAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure a named numeric series; keep nulls for plotting."""
        return numeric_validator(dropna=False, coerce_numeric=False).validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Produce ECDF-vs-CDF and Q–Q plots per distribution."""
        assert isinstance(data_input, pd.Series)
        series_name = data_input.name or "series"
        base_kwargs = {**self.base_kwargs(), "name": series_name}

        # TODO: Rank candidate fits (aggregate KS/AD/CvM/QQ metrics; pick best-fit distribution and confidence note).
        # TODO: Enrich QQ/ECDF plots (shape_qq_fit_plot.py, shape_ecdf_vs_cdf_plot.py) with tail curvature notes, fit score labels, and consolidated normality verdict for norm.
        fits: dict[str, Any] = {}
        for dist_name in self.context.distribution_names:
            ecdf_vs_cdf_ctx = build_plot_context(
                ShapeECDFvsCDFContext,
                base=self.context.ecdf_vs_cdf_context,
                overrides={**base_kwargs, "distribution_name": dist_name},
            )
            qq_ctx = build_plot_context(
                ShapeQqFitContext,
                base=self.context.qq_fit_context,
                overrides={**base_kwargs, "distribution_name": dist_name},
            )

            fits[dist_name] = {
                "ecdf_vs_cdf": ShapeECDFvsCDFPlot(ecdf_vs_cdf_ctx).run(data_input),
                "qq_fit": ShapeQqFitPlot(qq_ctx).run(data_input),
            }

        return {"distribution_fits": fits}
