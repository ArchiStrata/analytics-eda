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
"""Frequency distribution analysis for categorical data."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context
from analytics_eda.core.visualization.validation import categorical_validator

from .frequency_pareto_plot import FrequencyParetoPlot, FrequencyParetoPlotContext


@dataclass
class CategoricalFrequencyDistributionAnalysisContext(AnalysisContext):
    """Context for configuring categorical frequency distribution analysis."""

    report_name: str = "categorical_frequency_distribution_analysis"
    report_relative_path: str = "frequency_distribution"
    report_file_name: str = "categorical_frequency_distribution_analysis.json"
    frequency_pareto_plot_context: FrequencyParetoPlotContext | None = None


class CategoricalFrequencyDistributionAnalysis(BaseAnalysis):
    """
    Profile categorical levels by quantifying how much each category contributes.

    Big idea:
        Provide a structured view of categorical dominance so analysts instantly
        see which categories drive the majority of observations and which ones
        are rare.

    What this analysis does:
        Builds a frequency distribution (counts and proportions) and visualizes
        the cumulative contribution of categories using a Pareto chart.

    Why it matters:
        Frequency concentration highlights imbalance risks (oversized dominant
        categories or long-tailed rare ones) that influence sampling, modeling,
        and feature engineering choices.
    """

    semantic_version = "1.0.0"
    context: CategoricalFrequencyDistributionAnalysisContext

    def __init__(self, context: CategoricalFrequencyDistributionAnalysisContext) -> None:
        """
        Create a frequency-distribution analysis bound to the supplied context.

        Parameters
        ----------
        context : CategoricalFrequencyDistributionAnalysisContext
            Carries IO settings (paths, report behaviour) plus optional
            plot-level configuration overrides for the Pareto visualization.
        """
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """
        Ensure the input is a named categorical series (optionally with nulls).

        Returns
        -------
        pd.Series
            A copy of the validated categorical series ready for plotting.
        """
        return categorical_validator().validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """
        Generate the Pareto frequency plot for the validated categorical data.

        Pareto Chart
            What it is: A bar chart sorted by share-of-total with a cumulative
            contribution line, optionally highlighting the threshold (e.g., 80%).
            Why it matters: Pinpoints the “vital few” categories that dominate
            the distribution and exposes how quickly the tail decays, informing
            prioritization, grouping, or rare-category treatment.
        """
        assert isinstance(data_input, pd.Series)

        fp_ctx = build_plot_context(
            FrequencyParetoPlotContext,
            base=self.context.frequency_pareto_plot_context,
            overrides=self.base_kwargs(),
        )
        fp_plot = FrequencyParetoPlot(fp_ctx)
        return {"pareto": fp_plot.run(data_input)}
