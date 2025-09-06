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
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np
import pandas as pd

from ..utils.utils import resolve_cat_col, resolve_num_col, dropna_on, truncate_labels, postprocess_series, agg_sum

from ....core.visualization.base_plot import BasePlot
from ....core.visualization.context import PlotContext

# ---------------- Context ----------------

@dataclass
class RelationshipStructureGroupSizeBarContext(PlotContext):
    title_template: str = "Group Sizes for {name}{modifiers}"
    xlabel: str = "Group"
    ylabel: str = "Total"

    # plot-specific knobs
    top_k: Optional[int] = None          # show top-k groups by count (None = all)
    min_count: Optional[int] = None      # drop groups with count < min_count
    sort_desc: bool = True               # sort by count desc
    annotate: bool = True                # show value labels above bars
    rotate_xticks: int = 45              # rotation for readability; 0 to disable
    max_label_len: Optional[int] = 30    # truncate long labels; None = no truncation

# -------------- Plot ---------------------

class RelationshipStructureGroupSizeBarPlot(BasePlot):
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
        cat = resolve_cat_col(df, cols, role_map)
        resolve_num_col(df, cols, role_map)  # validate presence
        return dropna_on(df, cat)

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
        cat_col = resolve_cat_col(df, cols, role_map)
        num_col = resolve_num_col(df, cols, role_map)

        totals = postprocess_series(
            agg_sum(df, cat_col, num_col),
            min_value=self.ctx.min_count,
            sort_desc=self.ctx.sort_desc,
            top_k=self.ctx.top_k,
        )

        labels_raw = totals.index.astype(str).tolist()
        labels_disp = truncate_labels(labels_raw, self.ctx.max_label_len)
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

    def draw_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str,str]] = None,
        fig=None,
        ax=None,
        palette=None,
    ):
        """Render the bar chart using the frame-derived payload."""
        x = np.arange(len(desc["labels"]))
        ax.bar(x, desc["counts"])

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
