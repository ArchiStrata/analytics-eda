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
"""Bundle numeric–numeric relationship pillars: structure, magnitude, and direction."""

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis

from .direction_of_association.direction_of_association_analysis import (
    DirectionOfAssociationAnalysis,
    DirectionOfAssociationAnalysisContext,
)
from .magnitude_of_association.magnitude_of_association_analysis import (
    MagnitudeOfAssociationAnalysis,
    MagnitudeOfAssociationAnalysisContext,
)
from .relationship_structure.relationship_structure_analysis import (
    RelationshipStructureAnalysis,
    RelationshipStructureAnalysisContext,
)


@dataclass
class NumericNumericRelationshipAnalysisContext(AnalysisContext):
    """Context for numeric↔numeric relationship analysis (structure, magnitude, direction)."""

    report_name: str = "numeric_numeric_relationship_analysis"
    report_relative_path: str = "numeric_numeric_relationship_analysis"
    report_file_name: str = "numeric_numeric_relationship_analysis_report.json"
    save_json_report: bool = True
    return_full_report: bool = False

    x_col: str | None = None
    y_col: str | None = None

    relationship_structure_context: RelationshipStructureAnalysisContext | None = None
    magnitude_of_association_context: MagnitudeOfAssociationAnalysisContext | None = None
    direction_of_association_context: DirectionOfAssociationAnalysisContext | None = None


class NumericNumericRelationshipAnalysis(BaseAnalysis):
    """Bundle structure, magnitude, and direction pillar analyses for two numeric columns."""

    semantic_version = "1.0.0"
    context: NumericNumericRelationshipAnalysisContext

    def __init__(self, context: NumericNumericRelationshipAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Validate DataFrame and numeric columns once for all pillar analyses."""
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

    def _ctx_with_pair(self, ctx_obj: AnalysisContext | None, default_cls):
        if ctx_obj is None:
            ctx_obj = default_cls()
        return replace(
            ctx_obj,
            base_dir=self.report_dir(),
            data_source=self.context.data_source,
            filter_desc=self.context.filter_desc,
            report_relative_path="",
            return_full_report=True,
            save_json_report=False,
            x_col=self.context.x_col,
            y_col=self.context.y_col,
        )

    def build_artifacts(self, data_input: pd.DataFrame) -> dict[str, Any]:
        """Run pillar analyses and merge their payloads."""
        structure_ctx = self._ctx_with_pair(self.context.relationship_structure_context, RelationshipStructureAnalysisContext)
        magnitude_ctx = self._ctx_with_pair(self.context.magnitude_of_association_context, MagnitudeOfAssociationAnalysisContext)
        direction_ctx = self._ctx_with_pair(self.context.direction_of_association_context, DirectionOfAssociationAnalysisContext)

        structure = RelationshipStructureAnalysis(structure_ctx).run(data_input)
        magnitude = MagnitudeOfAssociationAnalysis(magnitude_ctx).run(data_input)
        direction = DirectionOfAssociationAnalysis(direction_ctx).run(data_input)

        return {
            "relationship_structure": structure.get("data", structure),
            "magnitude_of_association": magnitude.get("data", magnitude),
            "direction_of_association": direction.get("data", direction),
        }
