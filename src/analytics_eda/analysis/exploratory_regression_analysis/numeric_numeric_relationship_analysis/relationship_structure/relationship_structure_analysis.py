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
"""Analyze structure of numeric↔numeric relationships via scatter and LOWESS."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis.relationship_structure.relationship_structure_scatter_lowess_plot import (
    RelationshipStructureScatterLowessContext,
    RelationshipStructureScatterLowessPlot,
)
from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis.relationship_structure.relationship_structure_scatter_plot import (
    RelationshipStructureScatterContext,
    RelationshipStructureScatterPlot,
)
from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context


@dataclass
class RelationshipStructureAnalysisContext(AnalysisContext):
    """Context for relationship structure (scatter, LOWESS) between two numeric columns."""

    report_name: str = "relationship_structure_analysis"
    report_relative_path: str = "relationship_structure"
    report_file_name: str = "relationship_structure_analysis.json"

    x_col: str | None = None
    y_col: str | None = None

    scatter_context: RelationshipStructureScatterContext | None = None
    scatter_lowess_context: RelationshipStructureScatterLowessContext | None = None


class RelationshipStructureAnalysis(BaseAnalysis):
    """Analyze structure of numeric↔numeric relationship using scatter and LOWESS plots."""

    semantic_version = "1.0.0"
    context: RelationshipStructureAnalysisContext

    def __init__(self, context: RelationshipStructureAnalysisContext) -> None:
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
        """Build scatter and LOWESS plots for relationship structure."""
        role_map = {"x": self.context.x_col, "y": self.context.y_col}
        cols = [self.context.x_col, self.context.y_col]
        base_kwargs = self.base_kwargs()
        scatter_ctx = build_plot_context(
            RelationshipStructureScatterContext,
            base=self.context.scatter_context,
            overrides=base_kwargs,
        )
        scatter_plot = RelationshipStructureScatterPlot(scatter_ctx)

        scatter_lowess_ctx = build_plot_context(
            RelationshipStructureScatterLowessContext,
            base=self.context.scatter_lowess_context,
            overrides=base_kwargs,
        )
        scatter_lowess_plot = RelationshipStructureScatterLowessPlot(scatter_lowess_ctx)

        return {
            "scatter": scatter_plot.run(data_input, cols=cols, role_map=role_map),
            "scatter_lowess": scatter_lowess_plot.run(data_input, cols=cols, role_map=role_map),
        }
