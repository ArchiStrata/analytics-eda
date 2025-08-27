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
import numpy as np
import pandas as pd

from analytics_eda.core.utils.plot_mixins.series_bar_chart_mixin import SeriesBarChartContext, SeriesBarChartMixin

from ..utils.base_plot import BasePlot
from .validate_categorical_named_series import CategoricalSeriesMixin


@dataclass
class BalanceRareCategoriesContext(SeriesBarChartContext):
    """
    Identify and visualize rare categories (very low frequency).

    extreme_lower_bound:
        - If <1: interpreted as a proportion threshold of total count (e.g., 0.01 = 1%)
        - If >=1: interpreted as an absolute count threshold (e.g., 5 observations)
    """
    title_template: str = "Rare Categories of {name}{modifiers}"
    xlabel: str = "Percent of total"
    ylabel: str = "Category"
    is_orientation_vertical: bool = False
    format_value_axis_as_percent: bool = True
    show_footer_summary: bool = True

    # plot-specific knobs
    extreme_lower_bound: float = 0.01  # default: 1% of total if <1, else absolute count


class BalanceRareCategoriesPlot(CategoricalSeriesMixin, SeriesBarChartMixin, BasePlot):
    """
    Highlights and visualizes low-frequency categories in a categorical distribution.

    Why:
        In many categorical datasets, a small number of categories account for most 
        of the observations, while some categories occur rarely. Identifying these 
        rare categories is useful for data cleaning, grouping, or rebalancing 
        decisions, and can help detect anomalies or data quality issues.

    What:
        Filters categories whose counts fall below a configurable lower-frequency 
        threshold (either as a proportion of total count or as an absolute count) 
        and plots them in a horizontal bar chart for easy inspection.

    Returns (BasePlot.run schema):
      {
        "descriptive_stats": {
            "k",                        # number of unique categories
            "n_rare",                   # number of rare categories found
            "threshold_type",           # "proportion" or "count"
            "threshold_value_count",    # cutoff in counts
            "threshold_value_prop"      # cutoff in proportion
        },
      }
    """
    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.
        """
        return "1.0.0"

    # ---- defaults when empty/degenerate ----
    def default_descriptive(self) -> Dict[str, Any]:
        desc = super().default_descriptive()
        desc["k"] = 0
        desc["n_rare"] = 0

        desc["params"] = {
            "threshold_type": "proportion",
            "threshold_value_count": 0,
            "threshold_value_prop": 0.0,
        }

        return desc

    # ---- computations ----
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        # frequency table (drop NAs for category analysis)
        counts_all = s.dropna().value_counts()
        total = int(counts_all.sum())
        k = int(counts_all.size)

        # Resolve threshold
        bound = float(self.ctx.extreme_lower_bound)
        if bound < 1.0:
            thr_type = "proportion"
            thr_count = int(np.floor(bound * total))
            if 0 < bound < 1.0 and thr_count == 0:
                thr_count = 1
            thr_prop = bound
        else:
            thr_type = "count"
            thr_count = int(bound)
            thr_prop = (thr_count / total) if total > 0 else 0.0

        # Identify rare categories (<= cutoff), sort ascending (smallest first)
        if total > 0:
            rare_counts = counts_all[counts_all <= thr_count].sort_values(ascending=True)
        else:
            rare_counts = counts_all.iloc[0:0]

        # Build the bars payload using the mixin
        # Pass only the *rare* subset as counts; denominator is the series' non-null total.
        bars_desc = self.build_series_bar_desc(
            s,
            counts=rare_counts.to_dict(),
            denominator_key="pct_of_nonnull",
            extra_params={
                "extreme_lower_bound": self.ctx.extreme_lower_bound,
                "threshold_type": thr_type,
                "threshold_value_count": thr_count,
                "threshold_value_prop": float(thr_prop),
            },
        )

        # Add rare-specific summary fields
        bars_desc.update({
            "k": k,  # total unique categories in the series (not just rare)
            "n_rare": int(rare_counts.size),
        })

        # Skip plotting if nothing to show
        if bars_desc["n_rare"] == 0 or bars_desc["total"] == 0:
            bars_desc["skip_plot"] = True
            bars_desc["error"] = "no rare categories under threshold"

        return bars_desc
    
    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        if not desc or desc.get("total", 0) == 0:
            return {}
        # Short & factual: how many rare; threshold; base
        ttype = desc.get("params", {}).get("threshold_type", desc.get("threshold_type"))
        thr_c = desc.get("params", {}).get("threshold_value_count", desc.get("threshold_value_count"))
        thr_p = desc.get("params", {}).get("threshold_value_prop", desc.get("threshold_value_prop"))
        return {
            "context": f"Base = {desc.get('total_nonnull', 0):,} non-null; K = {desc.get('k', 0)} total categories.",
            "primary_finding": f"Rare categories determined by {ttype} threshold: {desc.get('n_rare', 0)} (≤ {thr_p:.1%} or ≤ {thr_c} count(s)).",
            "secondary_finding": None
        }

    # ---- drawing ----
    def footer_summary_text(self, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]) -> str:
        ttype = desc.get("params", {}).get("threshold_type", desc.get("threshold_type"))
        thr_c = desc.get("params", {}).get("threshold_value_count", desc.get("threshold_value_count"))
        thr_p = desc.get("params", {}).get("threshold_value_prop", desc.get("threshold_value_prop"))
        return f"Rare={desc.get('n_rare',0)} • Total non-null={desc.get('total_nonnull',0):,} • Threshold ({ttype}): ≤ {thr_p:.2%} (≤ {thr_c} count)"
