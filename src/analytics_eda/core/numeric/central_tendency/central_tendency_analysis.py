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
"""Central tendency analysis for numeric series built on BaseAnalysis."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context
from analytics_eda.core.visualization.validation import numeric_validator

from .central_tendency_histogram_plot import (
    CentralTendencyHistogramContext,
    CentralTendencyHistogramPlot,
)
from .central_tendency_mean_point_ci_plot import (
    CentralTendencyMeanPointCIContext,
    CentralTendencyMeanPointCIPlot,
)
from .central_tendency_median_point_ci_plot import (
    CentralTendencyMedianPointCIContext,
    CentralTendencyMedianPointCIPlot,
)


@dataclass
class CentralTendencyAnalysisContext(AnalysisContext):
    """Context for central tendency analysis."""

    report_name: str = "central_tendency_analysis"
    report_relative_path: str = "central_tendency"
    report_file_name: str = "central_tendency_analysis.json"

    histogram_context: CentralTendencyHistogramContext | None = None
    mean_point_ci_context: CentralTendencyMeanPointCIContext | None = None
    median_point_ci_context: CentralTendencyMedianPointCIContext | None = None


class CentralTendencyAnalysis(BaseAnalysis):
    """
    Summarize central tendency for a numeric series with histogram and point+CIs.

    Big idea:
        Provide a concise view of location (mean/median) and distribution shape
        so analysts can gauge typical values and uncertainty at a glance.
    """

    semantic_version = "1.0.0"
    context: CentralTendencyAnalysisContext

    def __init__(self, context: CentralTendencyAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure a named numeric series; keep nulls for plotting."""
        return numeric_validator(dropna=False, coerce_numeric=False).validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Run histogram, mean CI, and median CI plots."""
        assert isinstance(data_input, pd.Series)
        series_name = data_input.name or "series"
        base_kwargs = {**self.base_kwargs(), "name": series_name}

        hist_ctx = build_plot_context(
            CentralTendencyHistogramContext,
            base=self.context.histogram_context,
            overrides=base_kwargs,
        )
        mean_ctx = build_plot_context(
            CentralTendencyMeanPointCIContext,
            base=self.context.mean_point_ci_context,
            overrides=base_kwargs,
        )
        median_ctx = build_plot_context(
            CentralTendencyMedianPointCIContext,
            base=self.context.median_point_ci_context,
            overrides=base_kwargs,
        )

        histogram = CentralTendencyHistogramPlot(hist_ctx).run(data_input)
        mean_point_ci = CentralTendencyMeanPointCIPlot(mean_ctx).run(data_input)
        median_point_ci = CentralTendencyMedianPointCIPlot(median_ctx).run(data_input)

        # TODO: Central Tendency time series analysis
        # * Trend Direction
        # * Peaks & Troughs
        # * Volatility / Stability
        # * Seasonal Patterns / Cycles
        # * Change Magnitude

        return {
            "histogram": histogram,
            "mean_point_ci": mean_point_ci,
            "median_point_ci": median_point_ci,
        }
