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
"""Categorical↔numeric relationship analysis built on BaseAnalysis."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.direction_of_association.direction_of_association_analysis import (  # noqa: E501
    DirectionOfAssociationAnalysis,
    DirectionOfAssociationAnalysisContext,
)
from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.magnitude_of_association.magnitude_of_association_analysis import (  # noqa: E501
    MagnitudeOfAssociationAnalysis,
    MagnitudeOfAssociationAnalysisContext,
)
from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.relationship_structure.relationship_structure_analysis import (  # noqa: E501
    RelationshipStructureAnalysis,
    RelationshipStructureAnalysisContext,
)
from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis


@dataclass
class CategoricalNumericRelationshipAnalysisContext(AnalysisContext):
    """Context for categorical↔numeric relationship analysis."""

    report_name: str = "categorical_numeric_relationship_analysis"
    report_relative_path: str = "categorical_numeric_relationship_analysis"
    report_file_name: str = "categorical_numeric_relationship_analysis_report.json"
    base_dir: Path | None = Path("reports/eda/bivariate/categorical_numeric_relationship_analysis")
    save_json_report: bool = True
    return_full_report: bool = False

    categorical_col: str | None = None
    numeric_col: str | None = None

    relationship_structure_context: RelationshipStructureAnalysisContext | None = None
    magnitude_of_association_context: MagnitudeOfAssociationAnalysisContext | None = None
    direction_of_association_context: DirectionOfAssociationAnalysisContext | None = None


class CategoricalNumericRelationshipAnalysis(BaseAnalysis):
    """Orchestrate pillar analyses for categorical↔numeric relationships."""

    semantic_version = "1.0.0"
    context: CategoricalNumericRelationshipAnalysisContext

    def __init__(self, context: CategoricalNumericRelationshipAnalysisContext) -> None:
        super().__init__(context)

    # --- Validation -------------------------------------------------

    def validate(self, data_input: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Validate dataframe and required columns."""
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

        safe_cat = str(cat).replace(" ", "_")
        safe_num = str(num).replace(" ", "_")
        self.context.report_relative_path = f"categorical_{safe_cat}_numeric_{safe_num}_relationship_analysis"
        self.context.report_file_name = f"{self.context.report_relative_path}_report.json"

        return data_input.copy()

    # --- Helpers ----------------------------------------------------

    def _prepare_pillar_context(self, ctx: AnalysisContext | None, ctx_cls: type[AnalysisContext]) -> AnalysisContext:
        base = ctx or ctx_cls()
        return replace(
            base,
            base_dir=self.report_dir(),
            data_source=self.context.data_source,
            filter_desc=self.context.filter_desc,
            save_json_report=False,
            return_full_report=True,
        )

    @staticmethod
    def _merge_data_with_metadata(report: dict[str, Any]) -> dict[str, Any]:
        data = report.get("data", report)
        metadata = report.get("metadata")
        if isinstance(data, dict) and isinstance(metadata, dict):
            return {**data, "metadata": metadata}
        return data

    # --- Artifacts --------------------------------------------------

    def build_artifacts(self, data_input: pd.DataFrame | pd.Series) -> dict[str, Any]:
        """Run pillar analyses and assemble the composite report."""
        cat = self.context.categorical_col
        num = self.context.numeric_col

        structure_ctx = self._prepare_pillar_context(
            self.context.relationship_structure_context,
            RelationshipStructureAnalysisContext,
        )
        structure_ctx = replace(structure_ctx, categorical_col=cat, numeric_col=num)
        structure_report = RelationshipStructureAnalysis(structure_ctx).run(data_input)
        structure_data = self._merge_data_with_metadata(structure_report)

        magnitude_ctx = self._prepare_pillar_context(
            self.context.magnitude_of_association_context,
            MagnitudeOfAssociationAnalysisContext,
        )
        magnitude_ctx = replace(magnitude_ctx, categorical_col=cat, numeric_col=num)
        magnitude_report = MagnitudeOfAssociationAnalysis(magnitude_ctx).run(data_input)
        magnitude_data = self._merge_data_with_metadata(magnitude_report)

        direction_ctx = self._prepare_pillar_context(
            self.context.direction_of_association_context,
            DirectionOfAssociationAnalysisContext,
        )
        direction_ctx = replace(direction_ctx, categorical_col=cat, numeric_col=num)
        direction_report = DirectionOfAssociationAnalysis(direction_ctx).run(data_input)
        direction_data = self._merge_data_with_metadata(direction_report)

        return {
            "relationship_structure": structure_data,
            "magnitude_of_association": magnitude_data,
            "direction_of_association": direction_data,
        }
