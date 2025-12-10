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
"""Data-quality completeness pillar analysis built on BaseAnalysis."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context
from analytics_eda.core.visualization.validation import named_only_validator

from .completeness_ecdf_gap_plot import CompletenessECDFGapContext, CompletenessECDFGapPlot
from .completeness_issues_bar_plot import (
    CompletenessIssuesBarContext,
    CompletenessIssuesBarPlot,
)


@dataclass
class CompletenessIssuesAnalysisContext(AnalysisContext):
    """
    Context for completeness-focused data-quality analysis.

    Carries IO/report settings plus optional plot-level overrides for the
    completeness issues bar chart.
    """

    report_name: str = "completeness_issues_analysis"
    report_relative_path: str = "completeness"
    report_file_name: str = "completeness_issues_analysis.json"
    completeness_issues_bar_context: CompletenessIssuesBarContext | None = None
    completeness_ecdf_gap_context: CompletenessECDFGapContext | None = None


class CompletenessIssuesAnalysis(BaseAnalysis):
    """
    Quantify and visualize completeness gaps for a single series.

    Big idea:
        Surface how often values are missing, null, blank, or encoded as
        missing so data quality risk is explicit.
    
    Answers: Are values present?

    What this analysis does:
        Runs the completeness issues bar plot to tally gap types, report
        percentages, and save an artifact for reporting.
    """

    semantic_version = "1.0.0"
    context: CompletenessIssuesAnalysisContext

    def __init__(self, context: CompletenessIssuesAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure the incoming data is a named series (string-safe, keeps nulls)."""
        return named_only_validator(dropna=False, cast_str=False).validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Produce the completeness issues bar plot and return it as the sole artifact."""
        assert isinstance(data_input, pd.Series)
        series_name = data_input.name or "series"
        base_kwargs = {**self.base_kwargs(), "name": series_name}

        comp_ctx = build_plot_context(
            CompletenessIssuesBarContext,
            base=self.context.completeness_issues_bar_context,
            overrides=base_kwargs,
        )
        comp_plot = CompletenessIssuesBarPlot(comp_ctx)

        ecdf_gap_ctx = build_plot_context(
            CompletenessECDFGapContext,
            base=self.context.completeness_ecdf_gap_context,
            overrides=base_kwargs,
        )
        ecdf_gap_plot = CompletenessECDFGapPlot(ecdf_gap_ctx)

        return {
            "completeness_issues": comp_plot.run(data_input),
            "ecdf_gap": ecdf_gap_plot.run(data_input),
        }
