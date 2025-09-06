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

from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import SeriesBarChartContext, SeriesBarChartMixin
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import categorical_validator

from ..visualization.base_plot import BasePlot


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
    show_subtitle: bool = True

    # plot-specific knobs
    extreme_lower_bound: float = 0.01  # default: 1% of total if <1, else absolute count


class BalanceRareCategoriesPlot(SeriesBarChartMixin, BasePlot):
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
            "threshold_type",           # "proportion" or "count"
            "threshold_value_count",    # cutoff in counts
            "threshold_value_prop"      # cutoff in proportion
        },
      }
    """
    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=categorical_validator()
        )
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.
        """
        return "1.0.0"

    # ---- defaults when empty/degenerate ----
    def default_descriptive(self) -> Dict[str, Any]:
        desc = super().default_descriptive()

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

        return bars_desc
    
    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        if not desc or desc.get("total", 0) == 0:
            return {}

        total_nonnull = desc.get("total_nonnull", 0)
        k_total = desc.get("unique_categories_total", 0)
        n_rare = desc.get("input_nonzero_categories", 0)

        params = desc.get("params", {})
        ttype = params.get("threshold_type", desc.get("threshold_type"))
        thr_c = params.get("threshold_value_count", desc.get("threshold_value_count"))
        thr_p = params.get("threshold_value_prop", desc.get("threshold_value_prop"))

        findings = {
            "context": f"Base = {total_nonnull:,} non-null; K = {k_total} total categories",
            "secondary_finding": None
        }

        # Present if no categories meet the threshold
        if n_rare == 0:
            findings["primary_finding"] = "No categories meet the rare threshold."
            return findings

        # How many rare categories are there and how were they determined?
        pct_rare_rows = float(desc.get("pct_subset", 0.0)) * 100.0
        n_rare_rows = int(desc.get("subset_count", 0))

        findings["primary_finding"] = (
                f"Rare categories (threshold={ttype}: ≤ {thr_p:.1%} or ≤ {thr_c} count) "
                f"found: {n_rare}; they account for {pct_rare_rows:.1f}% of rows "
                f"({n_rare_rows:,})."
            )
        return findings

    # ---- drawing ----
    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        # No data → no subtitle
        if not desc or desc.get("total", 0) == 0 or desc.get("total_nonnull", 0) == 0:
            return ""

        n_rare = int(desc.get("input_nonzero_categories", 0))
        total_nonnull = int(desc.get("total_nonnull", 0))

        # Nothing to show
        if n_rare == 0:
            return "No categories meet the rare threshold"

        # Magnitude: percent + count of rows in rare categories
        pct_rare_rows = float(desc.get("pct_subset", 0.0)) * 100.0
        n_rare_rows = int(desc.get("subset_count", 0))

        p = desc.get("params", {})
        thr_type = p.get("threshold_type")
        thr_c = p.get("threshold_value_count")
        thr_p = p.get("threshold_value_prop")

        parts = [
            f"{n_rare} rare categor{'y' if n_rare == 1 else 'ies'}",
            f"{pct_rare_rows:.1f}% of {total_nonnull:,} rows ({n_rare_rows:,})",
        ]
        # Include threshold only if helpful and available
        if thr_type in {"proportion", "count"}:
            thr_bits = []
            if isinstance(thr_p, (int, float)) and thr_p > 0:
                thr_bits.append(f"≤{thr_p:.1%}")
            if isinstance(thr_c, (int, float)) and thr_c > 0:
                thr_bits.append(f"≤{int(thr_c)}")
            if thr_bits:
                parts.append(f"Threshold {' / '.join(thr_bits)}")

        return " • ".join(parts)
