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
"""Data-quality consistency pillar analysis (type inference and format consistency)."""

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
from analytics_eda.core.visualization.validation import named_only_validator

from .consistency_casing_normalization_bar_plot import (
    ConsistencyCasingNormalizationBarContext,
    ConsistencyCasingNormalizationBarPlot,
)
from .consistency_character_hygiene_bar_plot import (
    ConsistencyCharacterHygieneBarContext,
    ConsistencyCharacterHygieneBarPlot,
)
from .consistency_decimal_precision_bar_plot import (
    ConsistencyDecimalPrecisionBarContext,
    ConsistencyDecimalPrecisionBarPlot,
)
from .consistency_format_consistency_bar_plot import (
    ConsistencyFormatConsistencyBarContext,
    ConsistencyFormatConsistencyBarPlot,
)
from .consistency_numeric_coercion_bar_plot import (
    ConsistencyNumericCoercionBarContext,
    ConsistencyNumericCoercionBarPlot,
)
from .consistency_type_composition_bar_plot import (
    ConsistencyTypeCompositionBarContext,
    ConsistencyTypeCompositionBarPlot,
)
from .consistency_unit_frequency_bar_plot import (
    ConsistencyUnitFrequencyBarContext,
    ConsistencyUnitFrequencyBarPlot,
)
from .consistency_whitespace_normalization_bar_plot import (
    ConsistencyWhitespaceNormalizationBarContext,
    ConsistencyWhitespaceNormalizationBarPlot,
)


@dataclass
class ConsistencyTypeAnalysisContext(AnalysisContext):
    """
    Context for type inference and consistency analysis.

    Controls IO, report naming, and optional per-plot overrides.
    """

    report_name: str = "consistency_type_analysis"
    report_relative_path: str = "consistency"
    report_file_name: str = "consistency_type_analysis.json"

    type_composition_bar_context: ConsistencyTypeCompositionBarContext | None = None
    format_consistency_bar_context: ConsistencyFormatConsistencyBarContext | None = None
    decimal_precision_bar_context: ConsistencyDecimalPrecisionBarContext | None = None
    unit_frequency_bar_context: ConsistencyUnitFrequencyBarContext | None = None
    casing_normalization_bar_context: ConsistencyCasingNormalizationBarContext | None = None
    whitespace_normalization_bar_context: ConsistencyWhitespaceNormalizationBarContext | None = None
    character_hygiene_bar_context: ConsistencyCharacterHygieneBarContext | None = None
    numeric_coercion_bar_context: ConsistencyNumericCoercionBarContext | None = None


class ConsistencyTypeAnalysis(BaseAnalysis):
    """
    Assess whether a column has a dominant type and consistent formatting.

    Big idea:
        Infer the column's type mix and quantify format fragmentation so inconsistencies are
        surfaced before downstream modeling or validation.

    What this analysis does:
        Runs the type composition bar plot (dominant type vs. mixed) and the format consistency
        plot (pattern clusters), saving artifacts and returning their payloads.
    """

    semantic_version = "1.0.0"
    context: ConsistencyTypeAnalysisContext

    def __init__(self, context: ConsistencyTypeAnalysisContext) -> None:
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure a named series; keep nulls for composition/format auditing."""
        return named_only_validator(dropna=False, cast_str=False).validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """Run type composition and format consistency plots."""
        assert isinstance(data_input, pd.Series)
        series_name = data_input.name or "series"
        base_kwargs = {**self.base_kwargs(), "name": series_name}

        type_comp_ctx = build_plot_context(
            ConsistencyTypeCompositionBarContext,
            base=self.context.type_composition_bar_context,
            overrides=base_kwargs,
        )
        type_comp_plot = ConsistencyTypeCompositionBarPlot(type_comp_ctx)

        format_ctx = build_plot_context(
            ConsistencyFormatConsistencyBarContext,
            base=self.context.format_consistency_bar_context,
            overrides=base_kwargs,
        )
        format_plot = ConsistencyFormatConsistencyBarPlot(format_ctx)

        artifacts: dict[str, Any] = {
            "type_composition": type_comp_plot.run(data_input),
            "format_consistency": format_plot.run(data_input),
        }

        if is_numeric_dtype(data_input):
            precision_ctx = build_plot_context(
                ConsistencyDecimalPrecisionBarContext,
                base=self.context.decimal_precision_bar_context,
                overrides=base_kwargs,
            )
            precision_plot = ConsistencyDecimalPrecisionBarPlot(precision_ctx)
            artifacts["decimal_precision"] = precision_plot.run(data_input)

            unit_ctx = build_plot_context(
                ConsistencyUnitFrequencyBarContext,
                base=self.context.unit_frequency_bar_context,
                overrides=base_kwargs,
            )
            unit_plot = ConsistencyUnitFrequencyBarPlot(unit_ctx)
            artifacts["unit_frequency"] = unit_plot.run(data_input)
        else:
            artifacts["decimal_precision"] = {
                "descriptive_stats": {
                    "skip_plot": True,
                    "error": "Series is not numeric; decimal precision not assessed.",
                    "total": int(data_input.size),
                    "total_nonnull": int((~data_input.isna()).sum()),
                },
                "chart_metadata": {
                    "title": f"Decimal Precision for {series_name} (not run)",
                    "file_name": None,
                    "data_source": self.context.data_source,
                },
                "draft_descriptive_findings": {},
                "inferential_stats": {},
                "draft_inferential_findings": {},
            }
            artifacts["unit_frequency"] = {
                "descriptive_stats": {
                    "skip_plot": True,
                    "error": "Series is not numeric; unit frequency not assessed.",
                    "total": int(data_input.size),
                    "total_nonnull": int((~data_input.isna()).sum()),
                },
                "chart_metadata": {
                    "title": f"Unit Frequency for {series_name} (not run)",
                    "file_name": None,
                    "data_source": self.context.data_source,
                },
                "draft_descriptive_findings": {},
                "inferential_stats": {},
                "draft_inferential_findings": {},
            }

        if is_object_dtype(data_input) or is_categorical_dtype(data_input):
            whitespace_ctx = build_plot_context(
                ConsistencyWhitespaceNormalizationBarContext,
                base=self.context.whitespace_normalization_bar_context,
                overrides=base_kwargs,
            )
            whitespace_plot = ConsistencyWhitespaceNormalizationBarPlot(whitespace_ctx)
            artifacts["whitespace_normalization"] = whitespace_plot.run(data_input)

            casing_ctx = build_plot_context(
                ConsistencyCasingNormalizationBarContext,
                base=self.context.casing_normalization_bar_context,
                overrides=base_kwargs,
            )
            casing_plot = ConsistencyCasingNormalizationBarPlot(casing_ctx)
            artifacts["casing_normalization"] = casing_plot.run(data_input)

            char_hygiene_ctx = build_plot_context(
                ConsistencyCharacterHygieneBarContext,
                base=self.context.character_hygiene_bar_context,
                overrides=base_kwargs,
            )
            char_hygiene_plot = ConsistencyCharacterHygieneBarPlot(char_hygiene_ctx)
            artifacts["character_hygiene"] = char_hygiene_plot.run(data_input)
        else:
            for key, title in [
                ("whitespace_normalization", "Whitespace Normalization"),
                ("casing_normalization", "Casing Normalization"),
                ("character_hygiene", "Character Hygiene"),
            ]:
                artifacts[key] = {
                    "descriptive_stats": {
                        "skip_plot": True,
                        "error": "Series is not categorical; this assessment not run.",
                        "total": int(data_input.size),
                        "total_nonnull": int((~data_input.isna()).sum()),
                    },
                    "chart_metadata": {
                        "title": f"{title} for {series_name} (not run)",
                        "file_name": None,
                        "data_source": self.context.data_source,
                    },
                    "draft_descriptive_findings": {},
                    "inferential_stats": {},
                    "draft_inferential_findings": {},
                }

        if is_object_dtype(data_input) or is_categorical_dtype(data_input) or is_string_dtype(data_input):
            numeric_coercion_ctx = build_plot_context(
                ConsistencyNumericCoercionBarContext,
                base=self.context.numeric_coercion_bar_context,
                overrides=base_kwargs,
            )
            numeric_coercion_plot = ConsistencyNumericCoercionBarPlot(numeric_coercion_ctx)
            artifacts["numeric_coercion"] = numeric_coercion_plot.run(data_input)

        return artifacts
