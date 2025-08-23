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
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from ..utils.base_plot import BasePlot, PlotContext
from ..utils.named_series_mixin import NamedSeriesMixin


@dataclass
class StringCoercionBarContext(PlotContext):
    """
    Identify and visualize non-numeric (string-like) values in a series by
    attempting numeric coercion and collecting the values that fail to parse.
    """
    title_template: str = "Non-Numeric (String) Values in {name}{modifiers}"
    xlabel: str = "Percent of non‑null"
    ylabel: str = "Category"
    is_orientation_vertical: bool = False
    format_value_axis_as_percent: bool = True

    # plot-specific knobs
    max_bars: Optional[int] = 30          # cap the number of distinct strings shown (smallest first if many ties)
    show_count_in_label: bool = False
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

    Returns (BasePlot.run schema):
      {
        "descriptive_stats": {
          "total": int,                 # total observations (including NA)
          "total_nonnull": int,         # total non-null observations
          "n_non_numeric": int,         # number of observations failing coercion
          "pct_non_numeric": float,     # of total_nonnull
          "k_non_numeric": int         # number of distinct non-numeric categories
        },
        "inferential_stats": {},
        "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
      }
    """
    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.
        """
        return "1.0.0"

    # Defaults when empty
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "total": 0,
            "total_nonnull": 0,
            "n_non_numeric": 0,
            "pct_non_numeric": 0.0,
            "k_non_numeric": 0,
            "category_labels": [],
            "category_counts": [],
            "category_pcts": [],
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

        # per-category percents for bar labels (and plotting, if desired)
        per_category_pcts = (counts.astype(int) / max(1, total_nonnull)).astype(float)

        desc = {
            "params": {
                "include_na_literal": bool(self.ctx.include_na_literal),
            },
            "total": total,
            "total_nonnull": total_nonnull,
            "n_non_numeric": n_non_numeric,
            "pct_non_numeric": pct_non_numeric, # overall summary %
            "k_non_numeric": k_non_numeric,
            "category_labels": counts.index.tolist(),
            "category_counts": counts.astype(int).tolist(),
            "category_pcts": per_category_pcts.tolist(),               # per-category %
        }

        # Skip plotting when nothing to show
        if desc["n_non_numeric"] == 0 or desc["total_nonnull"] == 0:
            desc["skip_plot"] = True
            desc["error"] = "no non-numeric values detected"

        return desc

    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        if not desc or desc.get("total_nonnull", 0) == 0:
            return {}

        overall_pct = desc.get("pct_non_numeric", 0.0) * 100
        labels = desc.get("category_labels", [])
        pcts   = desc.get("category_pcts", [])
        counts = desc.get("category_counts", [])

        # No issues
        if desc.get("n_non_numeric", 0) == 0:
            return {
                "summary": "No non‑numeric tokens detected among non‑null values.",
                "coverage": f"Base = {desc.get('total_nonnull', 0):,} non‑null rows."
            }

        # Build top entries (bars are already sorted by count/your setting, but guard anyway)
        if not labels or not pcts:
            return {
                "summary": f"Non‑numeric share: {overall_pct:.1f}% of non‑null.",
                "coverage": f"Base = {desc.get('total_nonnull', 0):,} non‑null rows."
            }

        # Find top1 and (if present) top2 by percent
        order = np.argsort(pcts)[::-1]
        top1 = order[0]
        msg = {
            "summary": f"Non‑numeric share: {overall_pct:.1f}% of non‑null.",
            "top_issue": f"{labels[top1]} ({pcts[top1]*100:.1f}%; n={counts[top1]:,}).",
            "coverage": f"Base = {desc.get('total_nonnull', 0):,} non‑null rows."
        }
        if len(order) > 1:
            top2 = order[1]
            msg["secondary"] = f"Next: {labels[top2]} ({pcts[top2]*100:.1f}%; n={counts[top2]:,})."
        return msg

    # No inferential stats
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # Draw chart
    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):

        labels: List[str] = desc["category_labels"]
        counts: List[int] = desc["category_counts"]
        pcts:   List[float] = desc.get("category_pcts", [])

        # Respect existing sort choice (desc already sorted). Just build arrays.
        labels_arr = np.array(labels, dtype=object)
        counts_arr = np.array(counts, dtype=int)
        pcts_arr   = np.array(pcts, dtype=float)

        # Horizontal bar chart
        bars = ax.barh(labels_arr, pcts_arr, color=self.neutral_grey())  # start all gray

        # Highlight the top issue (largest pcts):
        # If sorted ascending, the last bar is the top; else the first.
        highlight_idx = -1 if self.ctx.sort_ascending else 0
        if len(bars) > 0:
            bars[highlight_idx].set_color(palette[0])

        # Always show percent; optionally append count
        bar_labels = [
            f"{pct*100:.1f}%{f' (n={cnt:,})' if self.ctx.show_count_in_label else ''}"
            for pct, cnt in zip(pcts_arr, counts_arr)
        ]

        ax.bar_label(
            bars,
            labels=bar_labels,
            label_type="edge",
            padding=3,
            fontsize="small"
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
