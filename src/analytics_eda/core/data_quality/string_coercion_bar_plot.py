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
from ..utils.named_series_mixin import NamedSeriesMixin


@dataclass
class StringCoercionBarContext(PlotContext):
    """
    Identify and visualize non-numeric (string-like) values in a series by
    attempting numeric coercion and collecting the values that fail to parse.
    """
    title_template: str = "Non-Numeric (String) Values in {name}{modifiers}"
    xlabel: str = "Count"
    ylabel: str = "Category"
    figsize: Tuple[int, int] = (10, 6)

    # plot-specific knobs
    max_bars: Optional[int] = 30          # cap the number of distinct strings shown (smallest first if many ties)
    show_percent_labels: bool = True      # annotate bars with % of total (non-null base)
    sort_ascending: bool = True           # sort by count
    include_na_literal: bool = False      # if True, include literal strings like "NaN", "None" if they fail coercion


class StringCoercionBarPlot(NamedSeriesMixin, BasePlot):
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

    How:
        - Build mask of non-null values where coercion → NaN.
        - Count unique string representations of those values.
        - Optionally cap the number of bars and annotate with percentages of total
          *non-null* observations.

    Returns (BasePlot.run schema):
      {
        "descriptive_stats": {
          "total": int,                 # total observations (including NA)
          "total_nonnull": int,         # total non-null observations
          "n_non_numeric": int,         # number of observations failing coercion
          "pct_non_numeric": float,     # of total_nonnull
          "k_non_numeric": int,         # number of distinct non-numeric categories
          # payload for draw:
          "labels": List[str],          # category labels (strings)
          "counts": List[int],          # counts per label
        },
        "inferential_stats": {},
        "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
      }
    """

    # Defaults when empty
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "total": 0,
            "total_nonnull": 0,
            "n_non_numeric": 0,
            "pct_non_numeric": 0.0,
            "k_non_numeric": 0,
            "labels": [],
            "counts": [],
        }

    # Compute descriptive stats (+ payload for drawing)
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        # Base tallies
        total = int(s.size)
        nonnull_mask = ~s.isna()
        total_nonnull = int(nonnull_mask.sum())

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

        counts = non_numeric_values.astype(str).value_counts()
        k_non_numeric = int(counts.size)
        n_non_numeric = int(counts.sum())
        pct_non_numeric = float(n_non_numeric / total_nonnull) if total_nonnull else 0.0

        # Sorting + capping
        if self.ctx.sort_ascending:
            counts = counts.sort_values(ascending=True)
        if self.ctx.max_bars is not None and self.ctx.max_bars > 0:
            counts = counts.iloc[: int(self.ctx.max_bars)]

        desc = {
            "total": total,
            "total_nonnull": total_nonnull,
            "n_non_numeric": n_non_numeric,
            "pct_non_numeric": pct_non_numeric,
            "k_non_numeric": k_non_numeric,
            "labels": counts.index.tolist(),
            "counts": counts.astype(int).tolist(),
        }

        # Skip plotting when nothing to show
        if desc["n_non_numeric"] == 0 or desc["total_nonnull"] == 0:
            desc["skip_plot"] = True
            desc["error"] = "no non-numeric values detected"

        return desc

    # No inferential stats
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # Draw chart
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        sns.set_palette("colorblind")
        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        labels: List[str] = desc["labels"]
        counts: List[int] = desc["counts"]
        total_nonnull = max(1, int(desc["total_nonnull"]))  # avoid divide-by-zero

        # Horizontal bar chart (safer for long string labels)
        # Ensure ascending order (already handled in compute, but keep robust)
        order_idx = np.argsort(counts) if self.ctx.sort_ascending else np.argsort(counts)[::-1]
        counts_sorted = np.array(counts)[order_idx]
        labels_sorted = np.array(labels, dtype=object)[order_idx]

        sns.barplot(x=counts_sorted, y=labels_sorted, ax=ax)

        # Labels & title
        ax.set_title(chart_metadata["title"])
        ax.set_xlabel(chart_metadata["xlabel"] or "Count")
        ax.set_ylabel(chart_metadata["ylabel"] or "Category")

        # Annotate with percentages of total non-null if requested
        if self.ctx.show_percent_labels:
            for i, v in enumerate(counts_sorted):
                pct = 100.0 * (v / total_nonnull)
                ax.text(
                    v, i, f" {v} ({pct:.1f}%)",
                    va="center", ha="left", fontsize="small"
                )

        # Footer summary
        footer = (
            f"Non-numeric values = {desc['n_non_numeric']:,} "
            f"({desc['pct_non_numeric']*100:.1f}% of non-null); "
            f"Distinct = {desc['k_non_numeric']:,}; "
            f"N (non-null) = {desc['total_nonnull']:,}; N (total) = {desc['total']:,}"
        )
        fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize="small", color="gray")

        return fig, ax
