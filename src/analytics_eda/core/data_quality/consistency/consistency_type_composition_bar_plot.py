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
"""Type Composition Bar Plot — show how values split across inferred types."""

from dataclasses import dataclass
import datetime
from typing import Any

import numpy as np
import pandas as pd

from analytics_eda.core.visualization.base_plot import BasePlot
from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import (
    SeriesBarChartContext,
    SeriesBarChartMixin,
)
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import named_only_validator


@dataclass
class ConsistencyTypeCompositionBarContext(SeriesBarChartContext):
    """Context for the consistency type composition bar plot."""

    title_template: str = "Type Composition for {name}{modifiers}"
    xlabel: str = "Percent of total"
    ylabel: str = "Detected type"
    is_orientation_vertical: bool = False
    bar_sort_descending: bool = True
    show_subtitle: bool = True


class ConsistencyTypeCompositionBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Reveal whether a column is dominated by one type or mixed across several.

    Why this matters:
        Type drift or mixed representations (numeric strings, dates as text) undermine downstream
        validation, encoding, and modeling. A quick type split surfaces columns that need cleanup.

    What this plot does:
        Infers simple type buckets (numeric, text, datetime, boolean, null), computes their share of
        the column, highlights the dominant type, and flags when the series is mixed.
    """

    def __init__(self, ctx: ConsistencyTypeCompositionBarContext):
        parts = PlotParts(series_validator=named_only_validator(dropna=False, cast_str=False))
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version for this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return default placeholders including type dominance fields."""
        base = SeriesBarChartMixin.default_descriptive(self)
        base.update(
            {
                "dominant_type": None,
                "dominant_ratio": 0.0,
                "type_counts": {},
            }
        )
        return base

    def _is_bool(self, v: Any) -> bool:
        return isinstance(v, bool | np.bool_)

    def _is_datetime(self, v: Any) -> bool:
        return isinstance(v, pd.Timestamp | datetime.datetime | datetime.date | np.datetime64)

    def _is_numeric(self, v: Any) -> bool:
        if self._is_bool(v) or self._is_datetime(v):
            return False
        try:
            return np.isfinite(float(v))
        except Exception:
            return False

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute type shares across numeric, text, datetime, boolean, and null buckets."""
        total = int(s.size)
        null_count = int(s.isna().sum())

        bool_count = datetime_count = numeric_count = text_count = 0

        for v in s.dropna():
            if self._is_bool(v):
                bool_count += 1
                continue
            if self._is_datetime(v):
                datetime_count += 1
                continue
            if self._is_numeric(v):
                numeric_count += 1
                continue
            text_count += 1

        counts = {
            "Numeric": numeric_count,
            "Text": text_count,
            "Datetime": datetime_count,
            "Boolean": bool_count,
            "Null / Missing": null_count,
        }

        desc = self.build_series_bar_desc(
            s,
            counts,
            denominator_key="pct_of_total",
            extra_params={
                "total": total,
                "total_nonnull": total - null_count,
            },
        )

        # Dominant type among non-null categories (exclude nulls)
        non_null_counts = {k: v for k, v in counts.items() if k != "Null / Missing"}
        non_null_total = max(1, total - null_count)
        if non_null_counts:
            dominant_type, dom_count = max(non_null_counts.items(), key=lambda kv: kv[1])
            desc.update(
                {
                    "dominant_type": dominant_type if dom_count > 0 else None,
                    "dominant_ratio": float(dom_count / non_null_total),
                    "type_counts": non_null_counts,
                }
            )

        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize whether the column is single-type or mixed."""
        total = int(desc.get("total", 0))
        total_nonnull = int(desc.get("total_nonnull", 0))
        dominant_type = desc.get("dominant_type")
        dom_ratio = float(desc.get("dominant_ratio", 0.0))

        if total == 0:
            return {
                "context": "0 total values",
                "primary_finding": "The series is empty.",
                "secondary_finding": None,
            }

        context = f"{total_nonnull:,} non-null of {total:,} total"
        if total_nonnull == 0:
            return {
                "context": context,
                "primary_finding": "No non-null values; type composition cannot be assessed.",
                "secondary_finding": None,
            }

        findings = {
            "context": context,
            "primary_finding": "",
            "secondary_finding": None,
        }

        if dominant_type and dom_ratio >= 0.9:
            findings["primary_finding"] = f"Column is predominantly {dominant_type.lower()} " f"({self.formatter.format_percent(dom_ratio)} of non-null)."
        else:
            findings["primary_finding"] = "Column is mixed-type; no single type exceeds 90% of non-null."

        return findings

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Show dominant type and mix level in the subtitle."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return ""
        dominant_type = desc.get("dominant_type")
        dom_ratio = float(desc.get("dominant_ratio", 0.0))
        dom_pct = self.formatter.format_percent(dom_ratio) if dominant_type else "0%"
        return f"Non-null: {total_nonnull:,} • Dominant: {dominant_type or 'None'} ({dom_pct})"
