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
"""Univariate categorical analysis built on BaseAnalysis."""

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from analytics_eda.core.categorical.categorical_distribution_analysis import (
    CategoricalDistributionAnalysis,
    CategoricalDistributionAnalysisContext,
)
from analytics_eda.core.data_quality import (
    SeriesDataQualityAnalysis,
    SeriesDataQualityAnalysisContext,
)
from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.validation import categorical_validator


@dataclass
class UnivariateCategoricalAnalysisContext(AnalysisContext):
    """Context for univariate categorical analysis."""

    report_name: str = "univariate_categorical_analysis"
    report_relative_path: str = "univariate/categorical"
    report_file_name: str = "univariate_categorical_analysis.json"
    save_json_report: bool = True
    return_full_report: bool = False

    # Nested contexts (optional overrides)
    data_quality_context: SeriesDataQualityAnalysisContext | None = None
    distribution_context: CategoricalDistributionAnalysisContext | None = None


class UnivariateCategoricalAnalysis(BaseAnalysis):
    """
    Run a categorical univariate analysis covering data quality and distribution.

    Big idea:
        Deliver a single report that covers the core data-quality pillars plus
        frequency/balance distribution views for a categorical series.
    """

    semantic_version = "1.0.0"
    context: UnivariateCategoricalAnalysisContext

    def __init__(self, context: UnivariateCategoricalAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Validate the categorical series; preserve nulls and name."""
        cleaned = categorical_validator(dropna=False, cast_str=False).validate(data_input)
        sanitized_name = cleaned.name.replace(" ", "_")
        self.context.report_file_name = f"{sanitized_name}_univariate_categorical_analysis.json"
        return cleaned

    def _resolve_data_quality_context(self) -> SeriesDataQualityAnalysisContext:
        proto = self.context.data_quality_context or SeriesDataQualityAnalysisContext()
        return replace(
            proto,
            base_dir=self.report_dir(),
            data_source=proto.data_source or self.context.data_source,
            filter_desc=proto.filter_desc or self.context.filter_desc,
            return_full_report=True,
            save_json_report=False,
        )

    def _resolve_distribution_context(self) -> CategoricalDistributionAnalysisContext:
        proto = self.context.distribution_context or CategoricalDistributionAnalysisContext()
        return replace(
            proto,
            base_dir=self.report_dir(),
            data_source=proto.data_source or self.context.data_source,
            filter_desc=proto.filter_desc or self.context.filter_desc,
            return_full_report=True,
            save_json_report=False,
        )

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Run data-quality and distribution analyses and assemble the report."""
        dq_ctx = self._resolve_data_quality_context()
        dist_ctx = self._resolve_distribution_context()

        dq_report = SeriesDataQualityAnalysis(dq_ctx).run(data_input)
        dist_report = CategoricalDistributionAnalysis(dist_ctx).run(data_input)

        return {
            "data_quality": dq_report,
            "distribution": dist_report,
        }
