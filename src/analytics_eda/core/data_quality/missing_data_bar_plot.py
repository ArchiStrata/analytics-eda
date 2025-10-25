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
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analytics_eda.core.visualization.context.plot_context import AxisFormat
from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import (
    SeriesBarChartContext,
    SeriesBarChartMixin,
)
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import named_only_validator

from ..visualization.base_plot import BasePlot


@dataclass
class MissingDataBarContext(SeriesBarChartContext):
    title_template: str = "Missing Data for {name}{modifiers}"
    xlabel: str = "Status"
    ylabel: str = "Percentage of Total"
    show_subtitle: bool = True
    bar_sort_descending: bool = True

    x_format: AxisFormat = field(
        default_factory=lambda: AxisFormat(kind="category")
    )
    y_format: AxisFormat = field(
        default_factory=lambda: AxisFormat(kind="percent", decimals=1, percent_scale_0to1=True)
    )

class MissingDataBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Shows the share of missing values to quickly assess data quality risk.

    Why this matters:
    - Missingness inflates bias and reduces statistical power; early visibility guides cleaning/imputation.

    What this plot does:
    - Computes present/missing counts and percentages, and annotates bars with both.
    - Y-axis is percentage for quick scanning.

    Returns (BasePlot.run schema):
      {
        "descriptive_stats": {
          "total": int,                 # total observations (including NA)
          "total_nonnull": int,         # total non-null observations
          "bars": {}
        }
      }
    """

    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=named_only_validator(dropna=False, cast_str=False)
        )
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.
        """
        return "1.0.0"

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        counts = s.isna().map({False: "Present", True: "Missing"}).value_counts().to_dict()
        # Ensure stable order presence
        counts = {"Present": counts.get("Present", 0), "Missing": counts.get("Missing", 0)}

        # Build reporting bars + cache draw arrays
        desc = self.build_series_bar_desc(
            s,
            counts,
            denominator_key="pct_of_total"
        )

        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        findings = {
            # how many total values were included in the plot analysis?
            "context": f"{desc['total']:,} values",
            "primary_finding": "",
            "secondary_finding": None
        }
        if not desc or desc.get("total", 0) == 0:
            findings["primary_finding"] = "The series is empty."
            return findings

        # how many missing values were there?
        pct_missing = desc['bars']['Missing']['pct_of_total']
        total = desc['total']
        findings["primary_finding"] =  f"{self.formatter.format_percent(pct_missing)} of {total:,} values are missing."

        if pct_missing == 0:
            findings["secondary_finding"] = "No missing values detected."
        elif pct_missing == 1.0:
            findings["secondary_finding"] = "All values are missing."
        elif pct_missing < 0.5:
            findings["secondary_finding"] = "Most values are present."
        elif pct_missing == 0.5:
            findings["secondary_finding"] = "Missing and present values are evenly split."
        else:
            findings["secondary_finding"] = "Missing values exceed present values."

        return findings

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        if not desc or desc.get("total", 0) == 0:
            return ""
        pct_missing = desc["bars"]["Missing"]["pct_of_total"]
        total = desc["total"]
        return f"{self.formatter.format_percent(pct_missing)} of {total:,} values missing"
