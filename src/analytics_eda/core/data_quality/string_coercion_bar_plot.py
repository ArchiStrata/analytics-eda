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

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

from analytics_eda.core.utils.plot_mixins.series_bar_chart_mixin import SeriesBarChartContext, SeriesBarChartMixin

from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import named_only_validator
from ..visualization.base_plot import BasePlot


@dataclass
class StringCoercionBarContext(SeriesBarChartContext):
    """
    Identify and visualize non-numeric (string-like) values in a series by
    attempting numeric coercion and collecting the values that fail to parse.
    """
    title_template: str = "Non-Numeric (String) Values in {name}{modifiers}"
    xlabel: str = "Percent of non‑null"
    ylabel: str = "Category"
    is_orientation_vertical: bool = False

    show_subtitle: bool = True

    # plot-specific knobs
    include_na_literal: bool = False      # if True, include literal strings like "NaN", "None" if they fail coercion


class StringCoercionBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Bar chart of string (category) → count for values that failed numeric coercion.

    Why:
        Numeric columns often contain rogue string tokens (e.g., "N/A", "—", "TBD", "three").
        Detecting and quantifying these helps with cleaning, type enforcement, and
        understanding input data quality.

    What:
        Attempts to coerce the series to numeric (using pandas `to_numeric(errors="coerce")`).
        Values that fail to coerce (and are not missing) are treated as non-numeric
        “string categories”, which are tallied and plotted in a horizontal bar chart.

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


    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        # Base tallies
        nonnull_mask = ~s.isna()

        # Attempt coercion
        # Note: everything that becomes NaN while original was non-null is “non-numeric”
        coerced = pd.to_numeric(s, errors="coerce")

        non_numeric_mask = nonnull_mask & coerced.isna()

        # Optional handling of literal NA strings (e.g., "NaN", "None") if you *don't* want them:
        if not self.ctx.include_na_literal:
            # Common “NA-like” tokens to drop from the non-numeric set
            na_like = {"nan", "na", "n/a", "none", "null", "", " "}
            # exclude case-insensitively those exact tokens
            def _is_na_like(val) -> bool:
                try:
                    return str(val).strip().lower() in na_like
                except Exception:
                    return False
            non_numeric_mask = non_numeric_mask & ~s.map(_is_na_like)

        non_numeric_values = s[non_numeric_mask].astype("object")

        counts_dict = non_numeric_values.astype(str).value_counts().to_dict()

        # Build bar payload (percent-of-nonnull), then cache for draw
        desc = self.build_series_bar_desc(s, counts_dict, denominator_key="pct_of_nonnull")

        desc["params"]["include_na_literal"] = bool(self.ctx.include_na_literal)
        
        return desc

    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        total_nonnull = desc["total_nonnull"]

        findings = {
            # how many nonnull values were included in the plot analysis?
            "context": f"N (non‑null) = {total_nonnull:,}",
            "primary_finding": "",
            "secondary_finding": None
        }

        if not desc or desc.get("total", 0) == 0 or desc.get("total_nonnull", 0) == 0:
            findings["primary_finding"] = "The series is empty."
            return findings
        
        total_nonnum = desc.get("subset_count", 0)  # rows that failed numeric coercion (sum of bars)
        pct_nonnum = float(desc.get("pct_subset", 0.0)) * 100.0

        # Nothing to report (all values numeric after coercion)
        if total_nonnum == 0:
            findings["primary_finding"] = "No non-numeric values detected."
            return findings

        # how many distinct failed coercion issues are there and how common are they?
        # Primary: overall rate + count
        findings["primary_finding"] = (
            f"{pct_nonnum:.1f}% of values failed numeric coercion "
            f"({total_nonnum:,} rows)."
        )

        # which issues had the most, how common, and how many?
        bars: Dict[str, Any] = desc.get("bars", {})
        denom_key = desc.get("denominator_key", "pct_of_nonnull")

        top_labels: list[str] = list(desc.get("top_labels", []))

        if not top_labels:
            return findings  # nothing meaningful to add

        # Build tie-aware secondary finding
        parts = []
        for lbl in sorted(top_labels):
            data = bars.get(lbl, {})
            pct = float(data.get(denom_key, 0.0)) * 100.0
            cnt = int(data.get("count", 0))
            parts.append(f"{repr(lbl)} ({pct:.1f}%, {cnt:,} rows)")

        if len(parts) == 1:
            findings["secondary_finding"] = f"Most frequent token: {parts[0]}."
        else:
            findings["secondary_finding"] = "Most frequent tokens (tie): " + ", ".join(parts) + "."

        return findings

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        if not desc or desc.get("total_nonnull", 0) == 0:
            return ""

        total_nonnull = desc["total_nonnull"]
        total_nonnum = desc.get("subset_count", 0)
        pct_nonnum = float(desc.get("pct_subset", 0.0)) * 100.0

        if total_nonnum == 0:
            return "No non-numeric values detected"

        return f"{pct_nonnum:.1f}% of {total_nonnull:,} non-null values are non-numeric ({total_nonnum:,} rows)"
