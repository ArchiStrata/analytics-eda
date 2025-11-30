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
"""Validity pillar analysis: value compliance across categorical/numeric domains."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context
from analytics_eda.core.visualization.validation import categorical_validator, named_only_validator

from .validity_allowed_categories_bar_plot import (
    ValidityAllowedCategoriesBarContext,
    ValidityAllowedCategoriesBarPlot,
)


@dataclass
class ValidityValueComplianceAnalysisContext(AnalysisContext):
    """Context for value-compliance validity analysis."""

    report_name: str = "validity_value_compliance_analysis"
    report_relative_path: str = "validity"
    report_file_name: str = "validity_value_compliance_analysis.json"

    allowed_categories_bar_context: ValidityAllowedCategoriesBarContext | None = None


class ValidityValueComplianceAnalysis(BaseAnalysis):
    """
    Assess whether values adhere to validity rules (allowed categories, with room for numeric rules).

    Big idea:
        Highlight invalid categories and show how distinct counts change after removing them; provide
        a scaffold for future numeric validity checks.

    What this analysis does:
        For categorical/object/string series, runs the allowed-categories bar plot. For other dtypes,
        returns a skip stub (until numeric validity plots are added).
    """

    semantic_version = "1.0.0"
    context: ValidityValueComplianceAnalysisContext

    def __init__(self, context: ValidityValueComplianceAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Validate input; prefer categorical validator when applicable."""
        if is_categorical_dtype(data_input) or is_object_dtype(data_input) or is_string_dtype(data_input):
            return categorical_validator(dropna=False, cast_str=False).validate(data_input)
        return named_only_validator(dropna=False, cast_str=False).validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Run allowed categories plot for categorical inputs; otherwise skip."""
        assert isinstance(data_input, pd.Series)
        series_name = data_input.name or "series"
        base_kwargs = {**self.base_kwargs(), "name": series_name}

        if is_categorical_dtype(data_input) or is_object_dtype(data_input) or is_string_dtype(data_input):
            allowed_ctx = build_plot_context(
                ValidityAllowedCategoriesBarContext,
                base=self.context.allowed_categories_bar_context,
                overrides=base_kwargs,
            )
            plot = ValidityAllowedCategoriesBarPlot(allowed_ctx)
            return {"allowed_categories": plot.run(data_input)}

        # Placeholder for future numeric validity plots
        if is_numeric_dtype(data_input):
            return {
                "allowed_categories": {
                    "descriptive_stats": {
                        "skip_plot": True,
                        "error": "Series is numeric; allowed categories not assessed (numeric validity plots TBD).",
                        "total": int(data_input.size),
                        "total_nonnull": int((~data_input.isna()).sum()),
                    },
                    "chart_metadata": {
                        "title": f"Validity: Allowed Categories for {series_name} (not run)",
                        "file_name": None,
                        "data_source": self.context.data_source,
                    },
                    "draft_descriptive_findings": {},
                    "inferential_stats": {},
                    "draft_inferential_findings": {},
                }
            }

        return {
            "allowed_categories": {
                "descriptive_stats": {
                    "skip_plot": True,
                    "error": "Series is not categorical; allowed categories not assessed.",
                    "total": int(data_input.size),
                    "total_nonnull": int((~data_input.isna()).sum()),
                },
                "chart_metadata": {
                    "title": f"Validity: Allowed Categories for {series_name} (not run)",
                    "file_name": None,
                    "data_source": self.context.data_source,
                },
                "draft_descriptive_findings": {},
                "inferential_stats": {},
                "draft_inferential_findings": {},
            }
        }
