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
"""Assess direction of association between two numeric columns via OLS trend."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis.direction_of_association.direction_association_scatter_ols_trend_plot import (
    DirectionAssociationScatterOLSTrendContext,
    DirectionAssociationScatterOLSTrendPlot,
)
from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context


@dataclass
class DirectionOfAssociationAnalysisContext(AnalysisContext):
    """Context for direction-of-association analysis between two numeric columns."""

    report_name: str = "direction_of_association_analysis"
    report_relative_path: str = "direction_of_association"
    report_file_name: str = "direction_of_association_analysis.json"

    x_col: str | None = None
    y_col: str | None = None

    scatter_trend_context: DirectionAssociationScatterOLSTrendContext | None = None


class DirectionOfAssociationAnalysis(BaseAnalysis):
    """Assess direction of association via OLS scatter with trend overlay."""

    semantic_version = "1.0.0"
    context: DirectionOfAssociationAnalysisContext

    def __init__(self, context: DirectionOfAssociationAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Ensure dataframe contains requested numeric columns."""
        if not isinstance(data_input, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        x_col = self.context.x_col or ""
        y_col = self.context.y_col or ""
        if x_col not in data_input.columns:
            raise KeyError(f"X column '{x_col}' not found.")
        if y_col not in data_input.columns:
            raise KeyError(f"Y column '{y_col}' not found.")
        if not is_numeric_dtype(data_input[x_col]):
            raise TypeError(f"Column '{x_col}' must be numeric.")
        if not is_numeric_dtype(data_input[y_col]):
            raise TypeError(f"Column '{y_col}' must be numeric.")
        return data_input.copy()

    def build_artifacts(self, data_input: pd.DataFrame) -> dict[str, Any]:
        """Build direction-of-association plot."""
        role_map = {"x": self.context.x_col, "y": self.context.y_col}
        cols = [self.context.x_col, self.context.y_col]
        base_kwargs = self.base_kwargs()

        scatter_trend_ctx = build_plot_context(
            DirectionAssociationScatterOLSTrendContext,
            base=self.context.scatter_trend_context,
            overrides=base_kwargs,
        )
        scatter_trend_plot = DirectionAssociationScatterOLSTrendPlot(scatter_trend_ctx)

        return {
            "scatter_ols_trend": scatter_trend_plot.run(data_input, cols=cols, role_map=role_map),
        }
