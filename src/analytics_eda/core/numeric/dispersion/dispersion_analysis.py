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
"""Dispersion analysis for numeric series built on BaseAnalysis."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context
from analytics_eda.core.visualization.validation import numeric_validator

from .dispersion_box_plot import DispersionBoxPlot, DispersionBoxPlotContext
from .dispersion_decile_plot import DispersionDecilePlot, DispersionDecilePlotContext
from .dispersion_sigma_bands_plot import (
    DispersionSigmaBandsPlot,
    DispersionSigmaBandsPlotContext,
)


@dataclass
class DispersionAnalysisContext(AnalysisContext):
    """Context for dispersion analysis."""

    report_name: str = "dispersion_analysis"
    report_relative_path: str = "dispersion"
    report_file_name: str = "dispersion_analysis.json"

    boxplot_context: DispersionBoxPlotContext | None = None
    sigma_bands_context: DispersionSigmaBandsPlotContext | None = None
    decile_context: DispersionDecilePlotContext | None = None


class DispersionAnalysis(BaseAnalysis):
    """
    Summarize spread and variability for a numeric series.

    Big idea:
        Provide a concise view of spread (boxplot), variability bands, and deciles
        so analysts can gauge dispersion and outliers quickly.
    """

    semantic_version = "1.0.0"
    context: DispersionAnalysisContext

    def __init__(self, context: DispersionAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure a named numeric series; preserve nulls for plotting."""
        return numeric_validator(dropna=False, coerce_numeric=False).validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Run dispersion plots (boxplot, sigma bands, deciles)."""
        assert isinstance(data_input, pd.Series)
        series_name = data_input.name or "series"
        base_kwargs = {**self.base_kwargs(), "name": series_name}

        box_ctx = build_plot_context(
            DispersionBoxPlotContext,
            base=self.context.boxplot_context,
            overrides=base_kwargs,
        )
        sigma_ctx = build_plot_context(
            DispersionSigmaBandsPlotContext,
            base=self.context.sigma_bands_context,
            overrides=base_kwargs,
        )
        decile_ctx = build_plot_context(
            DispersionDecilePlotContext,
            base=self.context.decile_context,
            overrides=base_kwargs,
        )

        # TODO: plot for var and cv descriptive stats

        # TODO: DispersionZScoreHistogramPlot
        # TODO: DispersionRobustZScoreHistogramPlot

        # TODO: population variance
        # TODO: population standard deviation
        # TODO: Coefficient of variation

        # TODO: Dispersion time series analysis

        return {
            "boxplot": DispersionBoxPlot(box_ctx).run(data_input),
            "sigma_bands": DispersionSigmaBandsPlot(sigma_ctx).run(data_input),
            "deciles": DispersionDecilePlot(decile_ctx).run(data_input),
        }
