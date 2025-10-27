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
"""Post-hoc mean-difference plot using Tukey's HSD.

Compares all pairs of group means after a significant ANOVA to show direction,
magnitude, and significance of differences.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ....core.visualization.base_plot import BasePlot
from ....core.visualization.context import PlotContext
from ..utils.utils import resolve_cat_col, resolve_num_col, truncate_labels

# ---------------- Context ----------------


@dataclass
class DirectionPosthocTukeyHsdContext(PlotContext):
    """Configuration for the post-hoc Tukey HSD mean-difference plot."""

    title_template: str = "Post-hoc Mean Differences (Tukey HSD) for {name}{modifiers}"
    xlabel: str = "Mean difference"
    ylabel: str = "Comparison"

    alpha: float = 0.05
    max_label_len: int | None = 30
    sort_by: str = "magnitude"  # "magnitude" | "diff" | "none"
    capsize: float = 4.0
    line_alpha_nonsig: float = 0.35
    marker_size: float = 6.0


# -------------- Plot ---------------------


class DirectionPosthocTukeyHsdPlot(BasePlot):
    """
    Show **direction and significance** of pairwise mean differences using Tukey's HSD.

    Why
    ---
    After a significant global test (e.g., ANOVA), analysts need to know *which* group
    means differ, *by how much*, and in *which direction*. A mean-difference plot with
    confidence intervals communicates direction (sign), magnitude, and significance at once.

    What
    ----
    • X = categorical, Y = numeric (one-way design).
    • Computes Tukey’s HSD pairwise comparisons at α (default 0.05).
    • Visual: horizontal **difference ± CI** for each pair, with a vertical reference at 0.
      Non-significant intervals are de-emphasized (lower alpha); significant ones stand out.
    • Sorting options: by |difference| (default), by raw difference, or preserve original order.

    Returns
    -------
    {
      "descriptive_stats": {
        "n_groups": int,
        "pairs": [
          {"i": int, "j": int, "g1": str, "g2": str,
           "diff": float, "ci_low": float, "ci_high": float, "reject": bool}
          ...
        ],
        "alpha": float
      },
      "inferential_stats": {},
      "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
    }
    """

    def default_descriptive(self) -> dict[str, Any]:
        """Return an empty/default descriptive-stats structure."""
        return {}

    # ---------- Frame API ----------

    def validate_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Mapping[str, str] | None = None) -> pd.DataFrame:
        """Require categorical (x) and numeric (y); drop rows with NA in either."""
        cat = resolve_cat_col(df, cols, role_map)
        num = resolve_num_col(df, cols, role_map)
        if not pd.api.types.is_numeric_dtype(df[num]):
            df = df.copy()
            df[num] = pd.to_numeric(df[num], errors="coerce")
        return df.dropna(subset=[cat, num])

    def compute_descriptive_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Run Tukey HSD and format pairwise mean-difference results."""
        ctx = self.ctx  # type: DirectionPosthocTukeyHsdContext
        cat = resolve_cat_col(df, cols, role_map)
        num = resolve_num_col(df, cols, role_map)

        # If fewer than 2 groups or trivially small n, return empty
        group_names = [g for g, _ in df.groupby(cat, observed=True)]
        n_groups = len(group_names)
        if n_groups < 2 or len(df) < 2:
            return {
                "n_groups": n_groups,
                "pairs": [],
                "alpha": ctx.alpha,
                # keep raw labels for drawing (even if empty)
                # TODO: BasePlot support caching descriptive stats calculated specifically for drawing
                "_labels_disp": truncate_labels([str(x) for x in group_names], ctx.max_label_len),
            }

        # Tukey HSD via statsmodels
        res = pairwise_tukeyhsd(endog=df[num].to_numpy(), groups=df[cat].astype(str).to_numpy(), alpha=ctx.alpha)

        # statsmodels result provides summary with group1, group2, meandiff, lower, upper, reject
        # Build structured list
        pairs: list[dict[str, Any]] = []
        # Note: res._results_table.data includes header row; use res.summary() or directly res._multicomp.pairindices
        table = res.summary()
        # rows start at index 1; columns: group1, group2, meandiff, p-adj, lower, upper, reject
        for row in table.data[1:]:
            g1, g2 = str(row[0]), str(row[1])
            diff = float(row[2])
            ci_low = float(row[4])
            ci_high = float(row[5])
            reject = bool(row[6])
            pairs.append(
                {
                    "i": None,
                    "j": None,  # indices are not essential for Tukey; leave None
                    "g1": g1,
                    "g2": g2,
                    "diff": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "reject": reject,
                }
            )

        # Sorting policy
        if ctx.sort_by == "magnitude":
            pairs.sort(key=lambda d: abs(d["diff"]), reverse=True)
        elif ctx.sort_by == "diff":
            pairs.sort(key=lambda d: d["diff"])
        # else: "none" — keep as is

        return {
            "n_groups": n_groups,
            "pairs": pairs,
            "alpha": ctx.alpha,
            "_labels_disp": truncate_labels([str(x) for x in group_names], ctx.max_label_len),
        }

    def compute_inferential_frame(self, df: pd.DataFrame, desc: dict[str, Any], *, cols: Sequence[str], role_map: Mapping[str, str] | None = None) -> dict[str, Any]:
        """No additional inferential stats; Tukey HSD results are descriptive outputs here."""
        return {}

    def draw_frame(
        self,
        df: pd.DataFrame,
        desc: dict[str, Any],
        inf: dict[str, Any],
        chart_metadata: dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Mapping[str, str] | None = None,
        fig=None,
        ax=None,
        palette=None,
    ):
        """Horizontal mean-difference ± CI per pair; reference line at 0; de-emphasize non-significant."""
        ctx = self.ctx  # type: DirectionPosthocTukeyHsdContext

        pairs: list[dict[str, Any]] = desc["pairs"]

        if not pairs:
            ax.axvline(0.0, linewidth=1, linestyle="--", alpha=0.5)
            return fig, ax

        # y positions (top to bottom)
        y_pos = np.arange(len(pairs))[::-1]
        y_labels = [f"{p['g1']} – {p['g2']}" for p in pairs]

        # Draw each CI as a horizontal errorbar
        diffs = np.array([p["diff"] for p in pairs], dtype=float)
        lo = np.array([p["ci_low"] for p in pairs], dtype=float)
        hi = np.array([p["ci_high"] for p in pairs], dtype=float)
        reject = np.array([bool(p["reject"]) for p in pairs], dtype=bool)

        # Split by significance for styling
        for idx in range(len(pairs)):
            color_alpha = 1.0 if reject[idx] else ctx.line_alpha_nonsig
            ax.hlines(y=y_pos[idx], xmin=lo[idx], xmax=hi[idx], alpha=color_alpha)
            ax.plot(diffs[idx], y_pos[idx], "o", ms=ctx.marker_size, alpha=color_alpha)

        # Reference line at 0
        ax.axvline(0.0, linewidth=1, linestyle="--", alpha=0.6)

        # Axes cosmetics
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)

        return fig, ax
