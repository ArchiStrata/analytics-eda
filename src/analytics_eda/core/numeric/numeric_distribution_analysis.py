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
"""Numeric distribution analysis built on BaseAnalysis (central tendency, dispersion, shape)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.numeric.central_tendency import (
    CentralTendencyAnalysis,
    CentralTendencyAnalysisContext,
)
from analytics_eda.core.numeric.dispersion import (
    DispersionAnalysis,
    DispersionAnalysisContext,
)
from analytics_eda.core.numeric.shape import ShapeAnalysis, ShapeAnalysisContext
from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.validation import numeric_validator


@dataclass
class NumericDistributionAnalysisContext(AnalysisContext):
    """Context bundling central tendency, dispersion, and shape analyses."""

    report_name: str = "numeric_distribution_analysis"
    report_relative_path: str = "numeric_distribution_analysis"
    report_file_name: str = "numeric_distribution_analysis_report.json"
    save_json_report: bool = False
    return_full_report: bool = True

    distribution_names: Sequence[str] = ("norm", "lognorm", "gamma", "expon")
    central_tendency_context: CentralTendencyAnalysisContext | None = None
    dispersion_context: DispersionAnalysisContext | None = None
    shape_context: ShapeAnalysisContext | None = None


class NumericDistributionAnalysis(BaseAnalysis):
    """
    Run numeric distribution analysis covering central tendency, dispersion, and shape.

    Big idea:
        Provide a cohesive report of location, spread, and shape/fit diagnostics for a numeric series.
    """

    semantic_version = "1.0.0"
    context: NumericDistributionAnalysisContext

    def __init__(self, context: NumericDistributionAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure a named numeric series; preserve nulls for plotting."""
        return numeric_validator(dropna=False, coerce_numeric=False).validate(data_input)

    def _merge_ctx(self, ctx_obj, default_cls):
        if ctx_obj is None:
            ctx_obj = default_cls()
        ctx_obj = ctx_obj.__class__(**{**ctx_obj.__dict__})
        ctx_obj.base_dir = self.report_dir()
        ctx_obj.data_source = ctx_obj.data_source or self.context.data_source
        ctx_obj.filter_desc = ctx_obj.filter_desc or self.context.filter_desc
        ctx_obj.return_full_report = True
        ctx_obj.save_json_report = False
        return ctx_obj

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Run central tendency, dispersion, and shape analyses."""
        ct_ctx = self._merge_ctx(self.context.central_tendency_context, CentralTendencyAnalysisContext)
        disp_ctx = self._merge_ctx(self.context.dispersion_context, DispersionAnalysisContext)
        shape_ctx = self._merge_ctx(self.context.shape_context, ShapeAnalysisContext)
        # pass distribution names into shape fit context if provided
        if getattr(shape_ctx, "distribution_fit_context", None):
            shape_ctx.distribution_fit_context.distribution_names = self.context.distribution_names
        else:
            from analytics_eda.core.numeric.shape import ShapeDistributionFitAnalysisContext

            shape_ctx.distribution_fit_context = ShapeDistributionFitAnalysisContext(
                distribution_names=self.context.distribution_names,
                base_dir=self.report_dir(),
                data_source=shape_ctx.data_source,
                filter_desc=shape_ctx.filter_desc,
                return_full_report=True,
                save_json_report=False,
            )

        ct_report = CentralTendencyAnalysis(ct_ctx).run(data_input)
        disp_report = DispersionAnalysis(disp_ctx).run(data_input)
        shape_report = ShapeAnalysis(shape_ctx).run(data_input)

        return {
            "central_tendency": ct_report,
            "dispersion": disp_report,
            "shape": shape_report,
        }
