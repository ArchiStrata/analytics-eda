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
"""Data-quality uniqueness pillar analysis built on BaseAnalysis."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context
from analytics_eda.core.visualization.validation import named_only_validator

from .uniqueness_cardinality_bar_plot import (
    UniquenessCardinalityBarContext,
    UniquenessCardinalityBarPlot,
)
from .uniqueness_duplicate_summary_bar_plot import (
    UniquenessDuplicateSummaryBarContext,
    UniquenessDuplicateSummaryBarPlot,
)


@dataclass
class UniquenessAnalysisContext(AnalysisContext):
    """
    Context for uniqueness-focused data-quality analysis.

    Carries IO/report settings plus optional plot-level overrides for the
    cardinality and duplicate-summary bar charts.
    """

    report_name: str = "uniqueness_analysis"
    report_relative_path: str = "uniqueness"
    report_file_name: str = "uniqueness_analysis.json"

    uniqueness_cardinality_bar_context: UniquenessCardinalityBarContext | None = None
    uniqueness_duplicate_summary_bar_context: UniquenessDuplicateSummaryBarContext | None = None


class UniquenessAnalysis(BaseAnalysis):
    """
    Quantify distinctness and duplication risk for a single series.

    Big idea:
        Surface how many unique values exist, how concentrated the values are,
        and what share of the column is duplicated.

    Answers: How many distinct values exist?

    What this analysis does:
        Runs the cardinality bar chart and duplicate summary plot, saving both
        artifacts and returning their payloads for reporting pipelines.
    """

    semantic_version = "1.0.0"
    context: UniquenessAnalysisContext

    def __init__(self, context: UniquenessAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure the incoming data is a named series (nulls preserved)."""
        return named_only_validator(dropna=False, cast_str=False).validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Produce cardinality and duplicate-summary plots."""
        assert isinstance(data_input, pd.Series)
        series_name = data_input.name or "series"
        base_kwargs = {**self.base_kwargs(), "name": series_name}

        dup_ctx = build_plot_context(
            UniquenessDuplicateSummaryBarContext,
            base=self.context.uniqueness_duplicate_summary_bar_context,
            overrides=base_kwargs,
        )
        dup_plot = UniquenessDuplicateSummaryBarPlot(dup_ctx)

        artifacts: dict[str, Any] = {
            "duplicate_summary": dup_plot.run(data_input),
        }

        if is_numeric_dtype(data_input):
            card_ctx = build_plot_context(
                UniquenessCardinalityBarContext,
                base=self.context.uniqueness_cardinality_bar_context,
                overrides=base_kwargs,
            )
            card_plot = UniquenessCardinalityBarPlot(card_ctx)
            artifacts["cardinality"] = card_plot.run(data_input)
        else:
            artifacts["cardinality"] = {
                "descriptive_stats": {
                    "skip_plot": True,
                    "error": "Series is not numeric; cardinality not assessed.",
                    "total": int(data_input.size),
                    "total_nonnull": int((~data_input.isna()).sum()),
                },
                "chart_metadata": {
                    "title": f"Cardinality Check — Discrete vs. Continuous for {series_name} (not run)",
                    "file_name": None,
                    "data_source": self.context.data_source,
                },
                "draft_descriptive_findings": {},
                "inferential_stats": {},
                "draft_inferential_findings": {},
            }

        return artifacts
