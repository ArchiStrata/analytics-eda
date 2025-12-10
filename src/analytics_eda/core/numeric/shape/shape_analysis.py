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
"""Shape analysis for numeric series built on BaseAnalysis."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context
from analytics_eda.core.visualization.validation import numeric_validator

from .shape_density_plot import ShapeDensityContext, ShapeDensityPlot
from .shape_distribution_fit_analysis import (
    ShapeDistributionFitAnalysis,
    ShapeDistributionFitAnalysisContext,
)
from .shape_probability_function_plot import (
    ShapeProbabilityFunctionContext,
    ShapeProbabilityFunctionPlot,
)


@dataclass
class ShapeAnalysisContext(AnalysisContext):
    """Context for numeric shape analysis."""

    report_name: str = "shape_analysis"
    report_relative_path: str = "shape"
    report_file_name: str = "shape_analysis.json"

    density_context: ShapeDensityContext | None = None
    probability_function_context: ShapeProbabilityFunctionContext | None = None
    distribution_fit_context: ShapeDistributionFitAnalysisContext | None = None


class ShapeAnalysis(BaseAnalysis):
    """
    Assess distribution shape and fit for a numeric series.

    Big idea:
        Provide a concise view of empirical vs. theoretical shape: density, ECDF gaps,
        ECDF vs. CDF fits, Q–Q diagnostics, and probability function overlays.
    """

    semantic_version = "1.0.0"
    context: ShapeAnalysisContext

    def __init__(self, context: ShapeAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure a named numeric series; preserve nulls for plotting."""
        return numeric_validator(dropna=False, coerce_numeric=False).validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Run shape-focused plots."""
        assert isinstance(data_input, pd.Series)
        series_name = data_input.name or "series"
        base_kwargs = {**self.base_kwargs(), "name": series_name}

        density_ctx = build_plot_context(
            ShapeDensityContext,
            base=self.context.density_context,
            overrides=base_kwargs,
        )
        prob_ctx = build_plot_context(
            ShapeProbabilityFunctionContext,
            base=self.context.probability_function_context,
            overrides=base_kwargs,
        )

        # TODO: DispersionZScoreHistogramPlot
        # TODO: DispersionRobustZScoreHistogramPlot

        # TODO: Shape time series analysis

        return {
            "density": ShapeDensityPlot(density_ctx).run(data_input),
            "probability_function": ShapeProbabilityFunctionPlot(prob_ctx).run(data_input),
            "distribution_fits": self._run_distribution_fits(data_input),
        }

    def _run_distribution_fits(self, data_input: pd.Series) -> dict[str, Any]:
        """Run distribution fit analysis and return its payload."""
        fit_ctx = self.context.distribution_fit_context or ShapeDistributionFitAnalysisContext(
            base_dir=self.report_dir(),
            data_source=self.context.data_source,
            filter_desc=self.context.filter_desc,
            return_full_report=True,
            save_json_report=False,
        )
        fit_analysis = ShapeDistributionFitAnalysis(fit_ctx)
        fit_report = fit_analysis.run(data_input)
        return fit_report
