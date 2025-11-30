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
"""Series-level data quality analysis across completeness, validity, consistency, and uniqueness pillars."""

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.validation import named_only_validator

from .completeness import (
    CompletenessIssuesAnalysis,
    CompletenessIssuesAnalysisContext,
)
from .consistency import (
    ConsistencyTypeAnalysis,
    ConsistencyTypeAnalysisContext,
)
from .uniqueness import (
    UniquenessAnalysis,
    UniquenessAnalysisContext,
)
from .validity import (
    ValidityValueComplianceAnalysis,
    ValidityValueComplianceAnalysisContext,
)


@dataclass
class SeriesDataQualityAnalysisContext(AnalysisContext):
    """Context for the multi-pillar series data-quality analysis."""

    report_name: str = "series_data_quality_analysis"
    report_relative_path: str = "data_quality"
    report_file_name: str = "series_data_quality_analysis.json"

    completeness_context: CompletenessIssuesAnalysisContext | None = None
    validity_context: ValidityValueComplianceAnalysisContext | None = None
    consistency_context: ConsistencyTypeAnalysisContext | None = None
    uniqueness_context: UniquenessAnalysisContext | None = None


class SeriesDataQualityAnalysis(BaseAnalysis):
    """
    Assess series-level data quality across completeness, validity, consistency, and uniqueness.

    Big idea:
        Provide a single report that answers the core data-quality pillars for a series:
        how complete it is, whether values are allowed, how consistent types/formats are,
        and how unique/distinct the values are.

    What this analysis does:
        Delegates to the pillar analyses (completeness, validity, consistency, uniqueness),
        runs them on the same series, and packages their artifacts under one report.
    """

    semantic_version = "1.0.0"
    context: SeriesDataQualityAnalysisContext

    def __init__(self, context: SeriesDataQualityAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Validate input as a named series; preserve nulls."""
        return named_only_validator(dropna=False, cast_str=False).validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Run pillar analyses and assemble the series data-quality report."""
        assert isinstance(data_input, pd.Series)
        series = data_input

        # Prepare child contexts with inherited base_dir
        def _merge_ctx(ctx_obj, default_cls):
            if ctx_obj is None:
                ctx_obj = default_cls()
            ctx_obj = replace(
                ctx_obj,
                base_dir=self.report_dir(),
                data_source=self.context.data_source,
                filter_desc=self.context.filter_desc,
                return_full_report=True,
                save_json_report=False,
            )
            return ctx_obj

        completeness_ctx = _merge_ctx(self.context.completeness_context, CompletenessIssuesAnalysisContext)
        validity_ctx = _merge_ctx(self.context.validity_context, ValidityValueComplianceAnalysisContext)
        consistency_ctx = _merge_ctx(self.context.consistency_context, ConsistencyTypeAnalysisContext)
        uniqueness_ctx = _merge_ctx(self.context.uniqueness_context, UniquenessAnalysisContext)

        completeness = CompletenessIssuesAnalysis(completeness_ctx).run(series)
        validity = ValidityValueComplianceAnalysis(validity_ctx).run(series)
        consistency = ConsistencyTypeAnalysis(consistency_ctx).run(series)
        uniqueness = UniquenessAnalysis(uniqueness_ctx).run(series)

        return {
            "completeness": completeness,
            "validity": validity,
            "consistency": consistency,
            "uniqueness": uniqueness,
        }
