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
"""Evaluate transforms via NumericDistributionAnalysis."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.validation import numeric_validator

from ..numeric_distribution_analysis import (
    NumericDistributionAnalysis,
    NumericDistributionAnalysisContext,
)
from .select_transforms import select_transforms
from .transform_series import transform_series


@dataclass
class TransformEvaluationAnalysisContext(AnalysisContext):
    """Context for evaluating transforms on a numeric series."""

    report_name: str = "transform_evaluation_analysis"
    report_relative_path: str = "transforms"
    report_file_name: str = "transform_evaluation_analysis.json"

    distribution_names: Sequence[str] = ("norm", "lognorm", "gamma", "expon")
    descriptive_stats: dict[str, Any] | None = None
    normality_tests: dict[str, Any] | None = None
    transform_names: Sequence[str] | None = None

    numeric_distribution_context: NumericDistributionAnalysisContext | None = None


class TransformEvaluationAnalysis(BaseAnalysis):
    """
    Apply candidate transforms and run numeric distribution analysis on each.

    Big idea:
        Use existing distribution diagnostics to decide which transforms to try, then
        profile each transformed series using NumericDistributionAnalysis.
    """

    semantic_version = "1.0.0"
    context: TransformEvaluationAnalysisContext

    def __init__(self, context: TransformEvaluationAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure a named numeric series; preserve nulls."""
        return numeric_validator(dropna=False, coerce_numeric=False).validate(data_input)

    def _resolve_transforms(self) -> Sequence[str]:
        if self.context.transform_names:
            return list(self.context.transform_names)
        return select_transforms(self.context.descriptive_stats or {}, self.context.normality_tests or {})

    def _build_numeric_context(self) -> NumericDistributionAnalysisContext:
        proto = self.context.numeric_distribution_context or NumericDistributionAnalysisContext(
            distribution_names=self.context.distribution_names,
        )
        return NumericDistributionAnalysisContext(
            **{
                **proto.__dict__,
                "base_dir": self.report_dir(),
                "data_source": proto.data_source or self.context.data_source,
                "filter_desc": proto.filter_desc or self.context.filter_desc,
                "return_full_report": True,
                "save_json_report": False,
            }
        )

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Evaluate transforms and run numeric distribution analysis on each."""
        transforms: dict[str, Any] = {}
        candidates = self._resolve_transforms()
        numeric_ctx = self._build_numeric_context()
        for transform_name in candidates:
            transform_dir = self.report_dir() / transform_name
            transform_dir.mkdir(parents=True, exist_ok=True)
            transformed = transform_series(data_input, transform_name)
            ctx = NumericDistributionAnalysisContext(
                **{
                    **numeric_ctx.__dict__,
                    "base_dir": transform_dir,
                    "filter_desc": f"{self.context.filter_desc or ''} ({transform_name})".strip(),
                }
            )
            analysis = NumericDistributionAnalysis(ctx)
            report = analysis.run(transformed)
            transforms[transform_name] = report

        return {"transforms": transforms}
