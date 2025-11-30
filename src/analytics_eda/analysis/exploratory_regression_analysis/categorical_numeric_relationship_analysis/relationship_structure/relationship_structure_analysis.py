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
"""Relationship-structure analysis for categorical↔numeric variables."""

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.relationship_structure.relationship_structure_group_size_bar_plot import (  # noqa: E501
    RelationshipStructureGroupSizeBarContext,
    RelationshipStructureGroupSizeBarPlot,
)
from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.relationship_structure.relationship_structure_variance_homogeneity_box_plot import (  # noqa: E501
    RelationshipStructureVarianceHomogeneityBoxPlot,
    RelationshipStructureVarianceHomogeneityContext,
)
from analytics_eda.analysis.univariate.univariate_numeric_analysis import (
    UnivariateNumericAnalysis,
    UnivariateNumericAnalysisContext,
)
from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context import build_plot_context


@dataclass
class RelationshipStructureAnalysisContext(AnalysisContext):
    """Context for relationship-structure analysis."""

    report_name: str = "relationship_structure_analysis"
    report_relative_path: str = "relationship_structure"
    report_file_name: str = "relationship_structure_analysis.json"

    categorical_col: str | None = None
    numeric_col: str | None = None

    group_size_context: RelationshipStructureGroupSizeBarContext | None = None
    variance_homogeneity_context: RelationshipStructureVarianceHomogeneityContext | None = None
    univariate_numeric_context: UnivariateNumericAnalysisContext | None = None


class RelationshipStructureAnalysis(BaseAnalysis):
    """Explore shape and spread across categories (sizes, variance, univariate)."""

    semantic_version = "1.0.0"
    context: RelationshipStructureAnalysisContext

    def __init__(self, context: RelationshipStructureAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Validate dataframe and column types."""
        if not isinstance(data_input, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        cat = self.context.categorical_col or ""
        num = self.context.numeric_col or ""
        if cat not in data_input.columns:
            raise KeyError(f"Categorical column '{cat}' not found.")
        if num not in data_input.columns:
            raise KeyError(f"Numeric column '{num}' not found.")
        if not (isinstance(data_input[cat].dtype, pd.CategoricalDtype) or is_object_dtype(data_input[cat])):
            raise TypeError(f"Column '{cat}' must be categorical or object.")
        if not is_numeric_dtype(data_input[num]):
            raise TypeError(f"Column '{num}' must be numeric.")
        return data_input.copy()

    def _category_filter_desc(self, category_slug: str) -> str | None:
        base_filter = f"filtered by {self.context.categorical_col}={category_slug}"
        if self.context.filter_desc:
            return f"{self.context.filter_desc}; {base_filter}"
        return base_filter

    def build_artifacts(self, data_input: pd.DataFrame) -> dict[str, Any]:
        """Build relationship-structure plots and per-category univariate reports."""
        base_kwargs = self.base_kwargs()
        role_map = {"x": self.context.categorical_col, "y": self.context.numeric_col}
        cols = [self.context.categorical_col, self.context.numeric_col]

        group_size_ctx = build_plot_context(
            RelationshipStructureGroupSizeBarContext,
            base=self.context.group_size_context,
            overrides=base_kwargs,
        )
        group_size_plot = RelationshipStructureGroupSizeBarPlot(group_size_ctx)

        variance_ctx = build_plot_context(
            RelationshipStructureVarianceHomogeneityContext,
            base=self.context.variance_homogeneity_context,
            overrides=base_kwargs,
        )
        variance_plot = RelationshipStructureVarianceHomogeneityBoxPlot(variance_ctx)

        numeric_distribution_by_category: dict[Any, Any] = {}
        cat_col = self.context.categorical_col or ""
        num_col = self.context.numeric_col or ""
        for category, group_df in data_input.groupby(cat_col, observed=True):
            slug = str(category).replace(" ", "_")
            uni_ctx = self.context.univariate_numeric_context or UnivariateNumericAnalysisContext()
            uni_ctx = replace(
                uni_ctx,
                base_dir=self.report_dir() / f"{cat_col}_{slug}",
                data_source=self.context.data_source,
                filter_desc=self._category_filter_desc(slug),
                save_json_report=False,
                return_full_report=True,
            )
            try:
                uni_report = UnivariateNumericAnalysis(uni_ctx).run(group_df[num_col])
                numeric_distribution_by_category[category] = uni_report
            except Exception as exc:
                numeric_distribution_by_category[category] = {"error": str(exc)}

        return {
            "group_size_barchart": group_size_plot.run(data_input, cols=cols, role_map=role_map),
            "variance_homogeneity_boxplot": variance_plot.run(data_input, cols=cols, role_map=role_map),
            "numeric_distribution_by_category": numeric_distribution_by_category,
        }
