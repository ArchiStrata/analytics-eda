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
"""ConsistencyUnitFrequencyBarPlot — surface unit mixing for numeric columns."""

from dataclasses import dataclass
import re
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
class ConsistencyUnitFrequencyBarContext(SeriesBarChartContext):
    """Context for unit frequency plot."""

    title_template: str = "Unit Frequency for {name}{modifiers}"
    xlabel: str = "Percent of non-null"
    ylabel: str = "Detected unit"
    is_orientation_vertical: bool = False
    bar_sort_descending: bool = True
    show_subtitle: bool = True


class ConsistencyUnitFrequencyBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Show how many values carry each unit to flag mixing that needs standardization.

    Why this matters:
        Mixed units (e.g., kg vs lbs, m vs ft) quietly corrupt aggregations and models. A quick unit
        split makes conversion requirements explicit.

    What this plot does:
        Parses simple unit suffixes on numeric-like values, counts their frequency, reports percent
        of non-null per unit, and highlights whether one unit dominates.
    """

    def __init__(self, ctx: ConsistencyUnitFrequencyBarContext):
        parts = PlotParts(series_validator=named_only_validator(dropna=False, cast_str=False))
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version for this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return base descriptive stats with dominant unit placeholders."""
        base = SeriesBarChartMixin.default_descriptive(self)
        base.update(
            {
                "dominant_unit": None,
                "dominant_ratio": 0.0,
            }
        )
        return base

    @staticmethod
    def _extract_unit(v: Any) -> str:
        """Return unit suffix if present, else 'Unitless' or 'Unrecognized'."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "Null / Missing"
        text = str(v).strip()
        # match numeric with optional unit suffix
        match = re.match(r"^[+-]?[\d.,]+(?:\s*)([A-Za-z]+)?$", text)
        if match:
            unit = match.group(1)
            return unit.upper() if unit else "Unitless"
        return "Unrecognized"

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Count unit frequencies and determine dominance."""
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
            unit = self._extract_unit(v)
            if unit == "Null / Missing":
                continue
            counts[unit] = counts.get(unit, 0) + 1

        desc = self.build_series_bar_desc(
            s,
            counts,
            denominator_key="pct_of_nonnull",
            extra_params={},
        )

        dominant_unit, dom_count = max(counts.items(), key=lambda kv: kv[1])
        dominant_ratio = dom_count / total_nonnull
        desc.update(
            {
                "dominant_unit": dominant_unit,
                "dominant_ratio": float(dominant_ratio),
            }
        )
        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize whether units are consistent or mixed."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        total = int(desc.get("total", total_nonnull))
        dom_unit = desc.get("dominant_unit")
        dom_ratio = float(desc.get("dominant_ratio", 0.0))

        if total_nonnull == 0:
            return {
                "context": f"0 non-null of {total:,} total",
                "primary_finding": "No non-null values; unit consistency cannot be assessed.",
                "secondary_finding": None,
            }

        context = f"{total_nonnull:,} non-null values"
        dominant_pct = self.formatter.format_percent(dom_ratio)

        if dom_ratio >= 0.9:
            primary = f"Units are consistent: {dom_unit} covers {dominant_pct} of non-null."
        else:
            primary = f"Units are mixed; top unit {dom_unit} covers {dominant_pct} of non-null."

        return {
            "context": context,
            "primary_finding": primary,
            "secondary_finding": None,
        }

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Subtitle with non-null count and dominant unit coverage."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return ""
        dom_unit = desc.get("dominant_unit") or "None"
        dom_ratio = float(desc.get("dominant_ratio", 0.0))
        return f"Non-null: {total_nonnull:,} • Dominant unit: {dom_unit} ({self.formatter.format_percent(dom_ratio)})"
