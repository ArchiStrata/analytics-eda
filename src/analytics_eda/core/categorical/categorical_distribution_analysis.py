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
"""End-to-end categorical distribution analysis built on BaseAnalysis."""

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.validation import categorical_validator

from .balance import (
    CategoricalBalanceAnalysis,
    CategoricalBalanceAnalysisContext,
)
from .frequency_distribution import (
    CategoricalFrequencyDistributionAnalysis,
    CategoricalFrequencyDistributionAnalysisContext,
)


@dataclass
class CategoricalDistributionAnalysisContext(AnalysisContext):
    """Context bundling nested contexts for frequency and balance sub-analyses."""

    report_name: str = "categorical_distribution_analysis"
    report_relative_path: str = "categorical_distribution_analysis"
    report_file_name: str = "categorical_distribution_analysis_report.json"
    save_json_report: bool = True
    return_full_report: bool = False
    frequency_context: CategoricalFrequencyDistributionAnalysisContext | None = None
    balance_context: CategoricalBalanceAnalysisContext | None = None


class CategoricalDistributionAnalysis(BaseAnalysis):
    """
    Run categorical frequency and balance analyses under one cohesive report.

    Big idea:
        Deliver both dominance and inequality perspectives together so consumers
        can understand categorical behavior without orchestrating multiple runs.

    What this analysis does:
        Executes `CategoricalFrequencyDistributionAnalysis` (Pareto-based counts)
        and `CategoricalBalanceAnalysis` (dispersion, chi-square, Lorenz, rare categories)
        while sharing IO metadata and paths.

    Why it matters:
        Downstream workflows (profiling, fairness checks) often require both views
        simultaneously; this orchestration keeps reporting consistent and avoids
        duplicated plumbing.
    """

    semantic_version = "1.0.0"
    context: CategoricalDistributionAnalysisContext

    def __init__(self, context: CategoricalDistributionAnalysisContext) -> None:
        """Retain the aggregated context for downstream orchestration."""
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Validate the series and stamp the report filename for downstream IO."""
        cleaned = categorical_validator().validate(data_input)
        sanitized_name = cleaned.name.replace(" ", "_")
        self.context.report_file_name = f"{sanitized_name}_categorical_distribution_analysis_report.json"
        return cleaned

    def _resolve_frequency_context(self) -> CategoricalFrequencyDistributionAnalysisContext:
        proto = self.context.frequency_context or CategoricalFrequencyDistributionAnalysisContext()
        return replace(
            proto,
            data_source=proto.data_source or self.context.data_source,
            filter_desc=proto.filter_desc or self.context.filter_desc,
            base_dir=self.report_dir(),
        )

    def _resolve_balance_context(self) -> CategoricalBalanceAnalysisContext:
        proto = self.context.balance_context or CategoricalBalanceAnalysisContext()
        return replace(
            proto,
            data_source=proto.data_source or self.context.data_source,
            filter_desc=proto.filter_desc or self.context.filter_desc,
            base_dir=self.report_dir(),
        )

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """
        Execute the nested analyses and merge their outputs under clear sections.

        Sections
        --------
        frequency_distribution
            What it is: Pareto-style view of category counts/proportions.
            Why it matters: Identifies dominant vs. rare categories for prioritization
            or grouping.

        balance
            What it is: Density, dispersion, rare-category, chi-square, and Lorenz
            diagnostics.
            Why it matters: Reveals inequality and long-tail behavior that impact
            modeling, fairness, and reporting decisions.
        """
        freq_ctx = self._resolve_frequency_context()
        freq_analysis = CategoricalFrequencyDistributionAnalysis(freq_ctx)

        balance_ctx = self._resolve_balance_context()
        balance_analysis = CategoricalBalanceAnalysis(balance_ctx)

        # TODO: Word cloud
        # TODO: Categorical time series analysis - Category Drift: Do category definitions or distributions change over time?

        freq_report = freq_analysis.run(data_input)
        balance_report = balance_analysis.run(data_input)

        return {
            "frequency_distribution": freq_report,
            "balance": balance_report,
        }
