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
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils.base_plot import BasePlot, PlotContext
from .validate_categorical_named_series import CategoricalSeriesMixin


@dataclass
class BalanceRareCategoriesContext(PlotContext):
    """
    Identify and visualize rare categories (very low frequency).

    extreme_lower_bound:
        - If <1: interpreted as a proportion threshold of total count (e.g., 0.01 = 1%)
        - If >=1: interpreted as an absolute count threshold (e.g., 5 observations)
    """
    title_template: str = "Rare Categories of {name}{modifiers}"
    xlabel: str = "Category"
    ylabel: str = "Count"
    figsize: Tuple[int, int] = (10, 6)

    # plot-specific knobs
    extreme_lower_bound: float = 0.01  # default: 1% of total if <1, else absolute count
    max_bars: Optional[int] = None     # optionally cap number of bars (smallest first)
    show_percent_labels: bool = True   # annotate bars with % of total


class BalanceRareCategoriesPlot(CategoricalSeriesMixin, BasePlot):
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

    How:
        - Computes counts for each category (excluding NAs by default).
        - Determines a cutoff from `extreme_lower_bound`:
            * If <1: treated as a proportion (e.g., 0.01 = 1% of total count).
            * If ≥1: treated as an absolute count threshold.
        - Selects categories at or below this cutoff.
        - Optionally caps number of bars and annotates with percentages.

    Returns (BasePlot.run schema):
      {
        "descriptive_stats": {
            "total",                    # total non-null observations
            "k",                        # number of unique categories
            "threshold_type",           # "proportion" or "count"
            "threshold_value_count",    # cutoff in counts
            "threshold_value_prop",     # cutoff in proportion
            "n_rare",                   # number of rare categories found
            "rare_categories",          # list of category names
            "rare_counts"               # list of category counts
        },
        "inferential_stats": {},
        "chart_metadata": {...}
      }
    """

    # ---- defaults when empty/degenerate ----
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "total": 0,
            "k": 0,
            "threshold_type": "proportion",
            "threshold_value_count": 0,
            "threshold_value_prop": 0.0,
            "n_rare": 0,
            "rare_categories": [],
            "rare_counts": [],
        }

    # ---- computations ----
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        # frequency table (drop NAs by default for category count plots)
        counts = s.dropna().value_counts()
        total = int(counts.sum())
        k = int(counts.size)

        # Resolve threshold
        bound = float(self.ctx.extreme_lower_bound)
        if bound < 1.0:
            # proportion → convert to count cutoff (inclusive)
            thr_count = int(np.floor(bound * total))
            # Ensure at least 1 if bound > 0 but floor is 0, so “rare” makes sense
            if 0 < bound < 1.0 and thr_count == 0:
                thr_count = 1
            thr_prop = bound
            thr_type = "proportion"
        else:
            thr_count = int(bound)
            thr_prop = (thr_count / total) if total > 0 else 0.0
            thr_type = "count"

        # Identify rare subset
        rare_mask = counts <= thr_count if total > 0 else pd.Series([], dtype=bool)
        rare_counts = counts[rare_mask].sort_values(ascending=True)

        # Optional cap (smallest first)
        if self.ctx.max_bars is not None and self.ctx.max_bars > 0:
            rare_counts = rare_counts.iloc[: int(self.ctx.max_bars)]
    
        desc = {
            "total": total,
            "k": k,
            "threshold_type": thr_type,
            "threshold_value_count": thr_count,
            "threshold_value_prop": float(thr_prop),
            "n_rare": int(rare_counts.size),
            "rare_categories": rare_counts.index.tolist(),
            "rare_counts": rare_counts.astype(int).tolist(),
        }

        # Decide early whether to skip plotting
        if desc["n_rare"] == 0 or desc["total"] == 0:
            desc["skip_plot"] = True
            desc["error"] = "no rare categories under threshold"

        return desc

    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        # No inferential stats for a simple balance display
        return {}

    # ---- drawing ----
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):

        sns.set_palette("colorblind")
        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        cats: List[str] = desc["rare_categories"]
        vals: List[int] = desc["rare_counts"]
        total = max(1, int(desc["total"]))  # avoid division by zero

        # Horizontal bar chart (rare → small → easier to read with long labels)
        order_idx = np.argsort(vals)  # ensure ascending (just in case)
        vals_sorted = np.array(vals)[order_idx]
        cats_sorted = np.array(cats, dtype=object)[order_idx]

        sns.barplot(x=vals_sorted, y=cats_sorted, ax=ax)

        # Labels & title
        ax.set_title(chart_metadata["title"])
        ax.set_xlabel(chart_metadata["xlabel"] or "Count")
        ax.set_ylabel(chart_metadata["ylabel"] or "Category")

        # Annotate with percentages if requested
        if self.ctx.show_percent_labels:
            for i, v in enumerate(vals_sorted):
                pct = 100.0 * (v / total)
                ax.text(
                    v, i, f" {v} ({pct:.1f}%)",
                    va="center", ha="left", fontsize="small"
                )

        # Threshold footer
        if desc["threshold_type"] == "proportion":
            thr_text = f"Threshold ≤ {desc['threshold_value_prop']:.2%} of total"
        else:
            thr_text = f"Threshold ≤ {desc['threshold_value_count']} count(s)"
        footer = f"Rare categories = {desc['n_rare']} | Total n = {desc['total']} | {thr_text}"
        fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize="small", color="gray")

        return fig, ax
