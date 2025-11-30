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
"""Direction-of-association analysis for categorical↔numeric variables."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.direction_of_association.direction_posthoc_tukey_hsd_plot import (  # noqa: E501
    DirectionPosthocTukeyHsdContext,
    DirectionPosthocTukeyHsdPlot,
)
from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context import build_plot_context


@dataclass
class DirectionOfAssociationAnalysisContext(AnalysisContext):
    """Context for direction-of-association analysis."""

    report_name: str = "direction_of_association_analysis"
    report_relative_path: str = "direction_of_association"
    report_file_name: str = "direction_of_association_analysis.json"

    categorical_col: str | None = None
    numeric_col: str | None = None

    posthoc_tukey_hsd_context: DirectionPosthocTukeyHsdContext | None = None


class DirectionOfAssociationAnalysis(BaseAnalysis):
    """Assess directionality via Tukey HSD post-hoc mean differences."""

    semantic_version = "1.0.0"
    context: DirectionOfAssociationAnalysisContext

    def __init__(self, context: DirectionOfAssociationAnalysisContext) -> None:
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

    def build_artifacts(self, data_input: pd.DataFrame) -> dict[str, Any]:
        """Build post-hoc Tukey HSD analysis."""
        base_kwargs = self.base_kwargs()
        role_map = {"x": self.context.categorical_col, "y": self.context.numeric_col}
        cols = [self.context.categorical_col, self.context.numeric_col]

        tukey_ctx = build_plot_context(
            DirectionPosthocTukeyHsdContext,
            base=self.context.posthoc_tukey_hsd_context,
            overrides=base_kwargs,
        )
        tukey_plot = DirectionPosthocTukeyHsdPlot(tukey_ctx)

        return {
            "posthoc_tukey_hsd": tukey_plot.run(data_input, cols=cols, role_map=role_map),
        }
