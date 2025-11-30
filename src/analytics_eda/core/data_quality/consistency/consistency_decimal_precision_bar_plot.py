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
"""ConsistencyDecimalPrecisionBarPlot — show precision fragmentation for numeric columns."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd

from analytics_eda.core.visualization.base_plot import BasePlot
from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import (
    SeriesBarChartContext,
    SeriesBarChartMixin,
)
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator


@dataclass
class ConsistencyDecimalPrecisionBarContext(SeriesBarChartContext):
    """Context for decimal precision consistency plot."""

    title_template: str = "Decimal Precision for {name}{modifiers}"
    xlabel: str = "Percent of non-null"
    ylabel: str = "Decimal places"
    is_orientation_vertical: bool = False
    bar_sort_descending: bool = True
    show_subtitle: bool = True


class ConsistencyDecimalPrecisionBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Quantify how many decimal places values use and whether precision is consistent.

    Why this matters:
        Mixed precisions (0dp IDs mixed with 3dp measurements) create rounding drift, dirty joins,
        and inconsistent aggregations. Surfacing the dominant precision makes cleanup actionable.

    What this plot does:
        Buckets numeric values by observed decimal places (including native floats), reports the
        percent of non-null rows per bucket, and highlights whether one precision dominates.
    """

    def __init__(self, ctx: ConsistencyDecimalPrecisionBarContext):
        parts = PlotParts(series_validator=numeric_validator(dropna=False, coerce_numeric=False))
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version for this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return base descriptive stats with dominant precision placeholders."""
        base = SeriesBarChartMixin.default_descriptive(self)
        base.update(
            {
                "dominant_precision_dp": None,
                "dominant_ratio": 0.0,
            }
        )
        return base

    @staticmethod
    def _decimal_places(v: Any) -> int:
        """Count decimal places for a numeric value (ints → 0)."""
        if isinstance(v, bool | np.bool_):
            return 0
        if isinstance(v, int | np.integer):
            return 0
        try:
            d = Decimal(str(v)).normalize()
            exp = d.as_tuple().exponent
            return abs(exp) if exp < 0 else 0
        except (InvalidOperation, ValueError):
            return 0

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Bucket values by decimal places and compute dominance."""
        total = int(s.size)
        total_nonnull = int((~s.isna()).sum())
        if total_nonnull == 0:
            desc = self.default_descriptive()
            desc.update(
                {
                    "total": total,
                    "total_nonnull": 0,
                    "skip_plot": True,
                    "error": "no non-null values to summarize",
                }
            )
            return desc

        counts: dict[str, int] = {}
        for v in s.dropna():
            dp = self._decimal_places(v)
            label = f"{dp} dp"
            counts[label] = counts.get(label, 0) + 1

        desc = self.build_series_bar_desc(
            s,
            counts,
            denominator_key="pct_of_nonnull",
            extra_params={},
        )

        dominant_label, dom_count = max(counts.items(), key=lambda kv: kv[1])
        dominant_dp = int(dominant_label.split()[0])
        dominant_ratio = dom_count / total_nonnull

        desc.update(
            {
                "dominant_precision_dp": dominant_dp,
                "dominant_ratio": float(dominant_ratio),
            }
        )
        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize precision dominance or fragmentation."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        total = int(desc.get("total", total_nonnull))
        dom_dp = desc.get("dominant_precision_dp")
        dom_ratio = float(desc.get("dominant_ratio", 0.0))

        if total_nonnull == 0:
            return {
                "context": f"0 non-null of {total:,} total",
                "primary_finding": "No non-null values; precision cannot be assessed.",
                "secondary_finding": None,
            }

        context = f"{total_nonnull:,} non-null values"
        dominant_pct = self.formatter.format_percent(dom_ratio)

        if dom_ratio >= 0.9:
            primary = f"Precision is consistent at {dom_dp}dp ({dominant_pct} of non-null)."
        else:
            primary = f"Precision is fragmented; top bucket {dom_dp}dp covers {dominant_pct} of non-null."

        return {
            "context": context,
            "primary_finding": primary,
            "secondary_finding": None,
        }

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Subtitle with non-null count and dominant precision coverage."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return ""
        dom_dp = desc.get("dominant_precision_dp")
        dom_ratio = float(desc.get("dominant_ratio", 0.0))
        return f"Non-null: {total_nonnull:,} • Dominant precision: {dom_dp}dp ({self.formatter.format_percent(dom_ratio)})"
