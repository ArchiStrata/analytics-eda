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
"""Univariate numeric analysis built on BaseAnalysis."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from analytics_eda.core.data_quality import SeriesDataQualityAnalysis, SeriesDataQualityAnalysisContext
from analytics_eda.core.numeric import (
    NumericDistributionAnalysis,
    NumericDistributionAnalysisContext,
    TransformEvaluationAnalysis,
    TransformEvaluationAnalysisContext,
)
from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.validation import numeric_validator


@dataclass
class UnivariateNumericAnalysisContext(AnalysisContext):
    """Context for univariate numeric analysis."""

    report_name: str = "univariate_numeric_analysis"
    report_relative_path: str = "univariate/numeric"
    report_file_name: str = "univariate_numeric_analysis.json"
    save_json_report: bool = True
    return_full_report: bool = False

    distribution_names: Sequence[str] = ("norm", "lognorm", "gamma", "expon")

    data_quality_context: SeriesDataQualityAnalysisContext | None = None
    distribution_context: NumericDistributionAnalysisContext | None = None
    transform_evaluation_context: TransformEvaluationAnalysisContext | None = None


class UnivariateNumericAnalysis(BaseAnalysis):
    """Run numeric univariate analysis covering data quality, distribution, and transforms."""

    semantic_version = "1.0.0"
    context: UnivariateNumericAnalysisContext

    def __init__(self, context: UnivariateNumericAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Validate numeric series and set report filename."""
        cleaned = numeric_validator(dropna=False, coerce_numeric=False).validate(data_input)
        sanitized = cleaned.name.replace(" ", "_")
        self.context.report_file_name = f"{sanitized}_univariate_analysis_report.json"
        return cleaned

    def _merge_ctx(self, ctx_obj, default_cls):
        if ctx_obj is None:
            ctx_obj = default_cls()
        return replace(
            ctx_obj,
            base_dir=self.report_dir(),
            data_source=self.context.data_source,
            filter_desc=self.context.filter_desc,
            return_full_report=True,
            save_json_report=False,
        )

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Run pillar analyses and assemble the univariate numeric report."""
        dq_ctx = self._merge_ctx(self.context.data_quality_context, SeriesDataQualityAnalysisContext)
        dist_ctx = self._merge_ctx(self.context.distribution_context, NumericDistributionAnalysisContext)
        dist_ctx = replace(dist_ctx, distribution_names=self.context.distribution_names)

        dq_report = SeriesDataQualityAnalysis(dq_ctx).run(data_input)
        dist_report = NumericDistributionAnalysis(dist_ctx).run(data_input)

        # Prepare transform evaluation context using norm fit stats if available
        transform_ctx = self.context.transform_evaluation_context
        transform_section = {}
        dist_data = dist_report.get("data", dist_report) or {}
        shape_report = dist_data.get("shape", {}) or {}
        shape_data = shape_report.get("data", shape_report) or {}
        dist_fit_report = shape_data.get("distribution_fits", {}) or {}
        dist_fit_data = dist_fit_report.get("data", dist_fit_report) or {}
        norm_stats = (dist_fit_data.get("distribution_fits", {}) or {}).get("norm", {})
        if norm_stats:
            qq_fit = norm_stats.get("qq_fit", {})
            descriptive_stats = qq_fit.get("descriptive_stats", {})
            normality_tests = qq_fit.get("inferential_stats", {})
            if transform_ctx is None:
                transform_ctx = TransformEvaluationAnalysisContext(
                    descriptive_stats=descriptive_stats,
                    normality_tests=normality_tests,
                    distribution_names=self.context.distribution_names,
                    base_dir=self.report_dir(),
                    data_source=self.context.data_source,
                    filter_desc=self.context.filter_desc,
                    return_full_report=True,
                    save_json_report=False,
                )
            else:
                transform_ctx = replace(
                    transform_ctx,
                    descriptive_stats=transform_ctx.descriptive_stats or descriptive_stats,
                    normality_tests=transform_ctx.normality_tests or normality_tests,
                    base_dir=self.report_dir(),
                    data_source=self.context.data_source,
                    filter_desc=self.context.filter_desc,
                    return_full_report=True,
                    save_json_report=False,
                )
            transform_analysis = TransformEvaluationAnalysis(transform_ctx)
            transform_section = transform_analysis.run(data_input)

        return {
            "data_quality": dq_report,
            "distribution": dist_report,
            "transforms": transform_section,
        }
