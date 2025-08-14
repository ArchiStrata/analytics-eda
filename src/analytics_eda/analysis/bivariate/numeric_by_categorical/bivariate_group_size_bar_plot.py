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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ....core.utils.base_plot import PlotContext, BasePlot

# ---------------- Context ----------------

@dataclass
class BivariateGroupSizeBarContext(PlotContext):
    title_template: str = "Group Sizes for {name}{modifiers}"
    xlabel: str = "Group"
    ylabel: str = "Total"
    figsize: Tuple[int, int] = (10, 6)

    # plot-specific knobs
    top_k: Optional[int] = None          # show top-k groups by count (None = all)
    min_count: Optional[int] = None      # drop groups with count < min_count
    sort_desc: bool = True               # sort by count desc
    annotate: bool = True                # show value labels above bars
    rotate_xticks: int = 45              # rotation for readability; 0 to disable
    max_label_len: Optional[int] = 30    # truncate long labels; None = no truncation

# -------------- Plot ---------------------

class BivariateGroupSizeBarPlot(BasePlot):
    """
    Shows the sum of the numeric values in each category (ΣY by X), highlighting groups that contribute most magnitude.

    Why
    ---
    In numeric-by-categorical analysis, understanding each group's total contribution is
    a foundational sanity check before variance tests, modeling, or resource allocation
    decisions. Sums highlight dominance/imbalance that simple counts can miss.

    What
    ----
    - X = categorical column; Y = numeric column.
    - Aggregation: `total_y = df.groupby(X, observed=True)[Y].sum(min_count=0)`.
      (NaNs in Y contribute 0 to the total.)
    - Optional presentation controls: top-k, minimum count threshold, sorting,
      truncated labels, and value annotations.

    Inputs
    ------
    Call via the DataFrame path:
      - `cols=['<categorical>', '<numeric>']`  OR  `role_map={'x': '<categorical>', 'y': '<numeric>'}`

    Outputs
    -------
    Returns a payload with:
      - descriptive_stats:
          {
            "n_groups": int,
            "group_sizes": list[float],   # totals per group (same as "counts")
            "labels": list[str],          # display labels (possibly truncated)
            "counts": list[float],        # alias of totals for downstream compatibility
            "total": float,               # sum of totals across groups
            "raw_labels": list[str],
            "col": str                    # categorical column name
          }
      - inferential_stats: {}             # (reserved for future extensions)
      - chart_metadata:
          {"title","xlabel","ylabel","data_source","file_name","k_groups": int}

    """

    # ---------- Helpers ----------
    def _resolve_cat_col(
        self, df: pd.DataFrame, cols: Sequence[str], role_map: Optional[Mapping[str, str]]
    ) -> str:
        if role_map and role_map.get("x"):
            col = role_map["x"]
        else:
            if not cols:
                raise ValueError("Provide cols=[<categorical>, <numeric>] or role_map={'x': <categorical>, 'y': <numeric>} .")
            col = cols[0]
        if col not in df.columns:
            raise KeyError(f"Categorical column '{col}' not in DataFrame.")
        return col

    def _resolve_num_col(
        self, df: pd.DataFrame, cols: Sequence[str], role_map: Optional[Mapping[str, str]]
    ) -> str:
        if role_map and role_map.get("y"):
            col = role_map["y"]
        else:
            if not cols or len(cols) < 2:
                raise ValueError("Provide both categorical and numeric columns (e.g., cols=['cat','value']) or set role_map={'x': 'cat', 'y': 'value'}.")
            col = cols[1]
        if col not in df.columns:
            raise KeyError(f"Numeric column '{col}' not in DataFrame.")
        return col

    def _postprocess_counts(self, counts: pd.Series) -> pd.Series:
        ctx = self.ctx  # type: BivariateGroupSizeBarContext
        if ctx.min_count is not None:
            counts = counts[counts >= ctx.min_count]
        if ctx.sort_desc:
            counts = counts.sort_values(ascending=False)
        if ctx.top_k is not None and ctx.top_k > 0:
            counts = counts.iloc[: ctx.top_k]
        return counts

    def _truncate_labels(self, labels: List[str]) -> List[str]:
        m = self.ctx.max_label_len
        if not m:
            return labels
        return [lbl if len(lbl) <= m else (lbl[: max(0, m - 1)] + "…") for lbl in labels]

    # ---------- Frame API ----------
    def validate_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> pd.DataFrame:
        """
        Ensure categorical (x) and numeric (y) columns exist.
        Drop rows where the **categorical** is NA (numeric NA are allowed; they just won't be counted).
        """
        cat_col = self._resolve_cat_col(df, cols, role_map)
        _ = self._resolve_num_col(df, cols, role_map)  # validate presence only
        return df.dropna(subset=[cat_col])

    def compute_descriptive_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Sum numeric values per category:
            totals = df.groupby(cat, observed=True)[num].sum()
        """
        cat_col = self._resolve_cat_col(df, cols, role_map)
        num_col = self._resolve_num_col(df, cols, role_map)

        # Sum numeric values in each category (NaNs contribute 0)
        totals = (
            df.groupby(cat_col, observed=True)[num_col]
            .sum(min_count=0)        # pandas >=1.1 supports min_count
            .fillna(0)
        )

        totals = self._postprocess_counts(totals)

        labels_raw = totals.index.astype(str).tolist()
        labels_disp = self._truncate_labels(labels_raw)
        values = totals.values.tolist()

        # total of totals: keep numeric type (float if sums are float)
        total_sum = float(totals.sum())

        return {
            "n_groups": int(len(totals)),
            "group_sizes": values,   # now "totals per group"
            "labels": labels_disp,
            "counts": values,        # keep key name if downstream expects it
            "total": total_sum,
            "raw_labels": labels_raw,
            "col": cat_col,
        }

    def compute_inferential_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, Any]:
        """No inferential stats for plot."""
        return {}

    def draw_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ):
        """Render the bar chart using the frame-derived payload."""
        fig, ax = plt.subplots(figsize=self.ctx.figsize)
        x = np.arange(len(desc["labels"]))
        ax.bar(x, desc["counts"])

        ax.set_title(chart_metadata["title"], pad=20)
        ax.set_xlabel(self.ctx.xlabel or "Group")
        ax.set_ylabel(self.ctx.ylabel or "Total")
        ax.set_xticks(x)
        ax.set_xticklabels(
            desc["labels"],
            rotation=self.ctx.rotate_xticks or 0,
            ha="right" if (self.ctx.rotate_xticks or 0) else "center",
        )

        if getattr(self.ctx, "annotate", True):
            for xi, yi in zip(x, desc["counts"]):
                ax.text(xi, yi, f"{yi:,}", ha="center", va="bottom", fontsize="small")

        # subtle subtitle: #groups and total
        try:
            k = desc["n_groups"]
            total = desc["total"]
            subtitle = f"{k} group{'s' if k != 1 else ''} • total ={total:,}"
            ax.text(
                0.5, 1.01, subtitle,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize="small", color="dimgray"
            )
        except Exception:
            pass

        return fig, ax
