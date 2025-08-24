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
from typing import Any, Dict, List, Mapping, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import bartlett, levene

from ..utils.utils import (
    resolve_cat_col, resolve_num_col, grouped_arrays, truncate_labels
)
from ....core.utils.base_plot import PlotContext, BasePlot


# ---------------- Context ----------------

@dataclass
class RelationshipStructureVarianceHomogeneityContext(PlotContext):
    title_template: str = "Variance Homogeneity for {name}{modifiers}"
    xlabel: str = "Group"
    ylabel: str = "Value"

    # drawing knobs
    rotate_xticks: int = 45
    max_label_len: Optional[int] = 30

    # violin / box / errorbar styling
    violin_alpha: float = 0.35
    box_width: float = 0.6
    showfliers: bool = True
    errorbar_capsize: float = 4.0

    # behavior
    overlay_errorbars_threshold: int = 4   # ≤ this → add mean±SD error bars


# -------------- Plot ---------------------

class RelationshipStructureVarianceHomogeneityBoxPlot(BasePlot):
    """
    Test and visualize whether group variances are comparable (homoscedasticity) in a
    numeric-by-categorical setting.

    Why
    ---
    Variance homogeneity is a key assumption behind common parametric tests and
    affects how we compare groups and choose models. Pairing visual cues with formal
    tests reduces misinterpretation.

    What
    ----
    • X = categorical, Y = numeric.
    • Visuals: per-group violins (shape) + boxplots (spread); for ≤4 groups, overlay
      mean ± SD error bars.
    • Stats: Bartlett’s test (most powerful under normality) and Levene’s test
      (robust to non-normality; uses median center).

    Returns
    -------
    {
      "descriptive_stats": {
        "n_groups": int,
        "group_labels": list[str],
        "group_ns": list[int],
        "means": list[float],
        "stds": list[float]
      },
      "inferential_stats": {
        "bartlett": {"statistic": float, "p_value": float, "reject": bool, "alpha": float},
        "levene":   {"statistic": float, "p_value": float, "reject": bool, "alpha": float}
      },
      "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
    }
    """

    # ---------- Frame API ----------

    def validate_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """Require categorical (x) and numeric (y); drop rows with NA in either."""
        cat = resolve_cat_col(df, cols, role_map)
        num = resolve_num_col(df, cols, role_map)
        if not pd.api.types.is_numeric_dtype(df[num]):
            df = df.copy()
            df[num] = pd.to_numeric(df[num], errors="coerce")
        return df.dropna(subset=[cat, num])

    def compute_descriptive_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        """Compute per‑group arrays and basic stats used for plotting."""
        ctx = self.ctx  # type: RelationshipStructureVarianceHomogeneityContext
        cat = resolve_cat_col(df, cols, role_map)
        num = resolve_num_col(df, cols, role_map)

        # arrays & labels
        arrays = grouped_arrays(df, cat, num)  # list[np.ndarray] (NaNs dropped)
        labels_raw = [g for g, _ in df.groupby(cat, observed=True)]
        labels = truncate_labels([str(x) for x in labels_raw], ctx.max_label_len)

        n_groups = len(arrays)
        ns = [int(a.size) for a in arrays]
        means = [float(np.mean(a)) if a.size else float("nan") for a in arrays]
        stds = [float(np.std(a, ddof=1)) if a.size > 1 else float("nan") for a in arrays]

        return {
            "n_groups": n_groups,
            "group_labels": labels,
            "group_ns": ns,
            "means": means,
            "stds": stds,
            # stash raw labels & arrays for inferential/draw
            "_labels_raw": [str(x) for x in labels_raw],
            "_arrays": arrays,
        }

    def compute_inferential_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        """Bartlett & Levene across groups (skip gracefully if < 2 groups or tiny n)."""
        arrays: List[np.ndarray] = desc.get("_arrays", [])
        alpha = getattr(self.ctx, "alpha", 0.05)  # reuse context alpha if present; else 0.05

        if len(arrays) < 2 or any(len(a) == 0 for a in arrays):
            return {
                "bartlett": {"statistic": float("nan"), "p_value": float("nan"), "reject": False, "alpha": alpha},
                "levene":   {"statistic": float("nan"), "p_value": float("nan"), "reject": False, "alpha": alpha},
            }

        # Bartlett (sensitive to non-normality)
        b_stat, b_p = bartlett(*arrays)
        # Levene (center=median is more robust, but scipy default is mean; use median via center='median')
        l_stat, l_p = levene(*arrays, center="median")

        return {
            "bartlett": {"statistic": float(b_stat), "p_value": float(b_p), "reject": bool(b_p < alpha), "alpha": alpha},
            "levene":   {"statistic": float(l_stat), "p_value": float(l_p), "reject": bool(l_p < alpha), "alpha": alpha},
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
        """Violin + box for all; add mean±SD error bars if groups ≤ threshold."""
        ctx = self.ctx  # type: RelationshipStructureVarianceHomogeneityContext

        labels = desc["group_labels"]
        arrays: List[np.ndarray] = desc["_arrays"]
        means, stds = desc["means"], desc["stds"]
        n_groups = desc["n_groups"]

        if n_groups == 0:
            return fig, ax

        x = np.arange(n_groups)

        # Violin (shape)
        parts = ax.violinplot(
            arrays,
            positions=x,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.9,
        )
        for b in parts["bodies"]:
            b.set_alpha(ctx.violin_alpha)

        # Box (spread)
        ax.boxplot(
            arrays,
            positions=x,
            widths=ctx.box_width,
            showfliers=ctx.showfliers,
            manage_ticks=False,
        )

        # Error bars (mean ± SD) if small number of groups
        if n_groups <= int(getattr(ctx, "overlay_errorbars_threshold", 4)):
            ax.errorbar(
                x,
                means,
                yerr=stds,
                fmt="o",
                capsize=ctx.errorbar_capsize,
                label="Mean ± SD",
            )

        # Axes cosmetics
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=getattr(ctx, "rotate_xticks", 0) or 0,
            ha="right" if (getattr(ctx, "rotate_xticks", 0) or 0) else "center",
        )

        # Optional legend if error bars are shown
        if n_groups <= int(getattr(ctx, "overlay_errorbars_threshold", 4)):
            ax.legend(frameon=False, loc="best")

        return fig, ax
