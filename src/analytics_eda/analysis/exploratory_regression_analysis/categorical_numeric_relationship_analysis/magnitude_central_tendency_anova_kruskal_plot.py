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
import math
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal

from ..utils.utils import (
    resolve_cat_col,
    resolve_num_col,
    grouped_arrays,
    truncate_labels,
)
from ....core.utils.base_plot import PlotContext, BasePlot


# ---------------- Context ----------------

@dataclass
class MagnitudeCentralTendencyAnovaKruskalContext(PlotContext):
    title_template: str = "Magnitude of Differences for {name}{modifiers}"
    xlabel: str = "Group"
    ylabel: str = "Value"

    # visual emphasis
    bg_alpha: float = 0.18          # background violins/boxes alpha (de-emphasize)
    box_width: float = 0.6
    showfliers: bool = False
    rotate_xticks: int = 45
    max_label_len: Optional[int] = 30

    # which center & CI to emphasize on the chart
    #   "anova"   -> draw means with 95% CI (mean ± 1.96 * s/sqrt(n))
    #   "kruskal" -> draw medians with bootstrap 95% CI
    mode: str = "anova"             # "anova" or "kruskal"

    # bootstrap for median CIs
    bootstrap_iters: int = 2000
    bootstrap_ci: float = 0.95
    random_state: Optional[int] = None

    # annotate global test in subtitle if significant
    alpha: float = 0.05


# -------------- Plot ---------------------

class MagnitudeCentralTendencyAnovaKruskalPlot(BasePlot):
    """
    Quantify and visualize *magnitude of group differences* in a numeric-by-categorical analysis.

    Why
    ---
    When comparing multiple categories, you need to know whether their central tendencies
    differ beyond chance (global hypothesis) **and** how large the differences look.
    Pairing ANOVA/Kruskal tests with clear center markers + confidence intervals makes
    the story both statistically sound and visually compelling.

    What
    ----
    • X = categorical, Y = numeric.
    • Global tests:
        – ANOVA (parametric): `scipy.stats.f_oneway`
        – Kruskal–Wallis (nonparametric): `scipy.stats.kruskal`
    • Visual emphasis (pick via `mode`):
        – `"anova"`: plot **means** with 95% CI (mean ± 1.96·SE)
        – `"kruskal"`: plot **medians** with **bootstrap** 95% CI (percentile)
      Background (de-emphasized): violins + boxplots for context (shape & spread).
    • Subtitle shows global test p-values; highlights significance at α.

    Returns
    -------
    {
      "descriptive_stats": {
        "n_groups": int,
        "group_labels": list[str],
        "group_ns": list[int],
        "means": list[float],
        "mean_ci_lo": list[float],
        "mean_ci_hi": list[float],
        "medians": list[float],
        "median_ci_lo": list[float],
        "median_ci_hi": list[float]
      },
      "inferential_stats": {
        "anova":   {"statistic": float, "p_value": float, "reject": bool, "alpha": float},
        "kruskal": {"statistic": float, "p_value": float, "reject": bool, "alpha": float}
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
        """Compute per-group arrays and center/CI summaries."""
        ctx = self.ctx  # type: MagnitudeCentralTendencyAnovaKruskalContext
        cat = resolve_cat_col(df, cols, role_map)
        num = resolve_num_col(df, cols, role_map)

        arrays = grouped_arrays(df, cat, num)  # list[np.ndarray], NaNs removed
        labels_raw = [g for g, _ in df.groupby(cat, observed=True)]
        labels = truncate_labels([str(x) for x in labels_raw], ctx.max_label_len)

        n_groups = len(arrays)
        ns = [int(a.size) for a in arrays]

        # Means & 95% CI (normal approx)
        means, mean_lo, mean_hi = [], [], []
        for a in arrays:
            if a.size == 0:
                means.append(np.nan); mean_lo.append(np.nan); mean_hi.append(np.nan)
                continue
            m = float(np.mean(a))
            s = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
            se = s / math.sqrt(a.size) if a.size > 0 else np.nan
            ci = 1.96 * se if np.isfinite(se) else np.nan  # z ~ 1.96
            means.append(m)
            mean_lo.append(m - ci if np.isfinite(ci) else np.nan)
            mean_hi.append(m + ci if np.isfinite(ci) else np.nan)

        # Medians & bootstrap CI
        rng = np.random.default_rng(ctx.random_state)
        q_lo = (1 - ctx.bootstrap_ci) / 2
        q_hi = 1 - q_lo
        medians, med_lo, med_hi = [], [], []
        for a in arrays:
            if a.size == 0:
                medians.append(np.nan); med_lo.append(np.nan); med_hi.append(np.nan)
                continue
            med = float(np.median(a))
            if a.size == 1:
                # CI undefined with 1 sample → keep median, CI=NaN
                medians.append(med); med_lo.append(np.nan); med_hi.append(np.nan)
                continue
            # bootstrap
            bs = rng.choice(a, size=(ctx.bootstrap_iters, a.size), replace=True)
            bs_meds = np.median(bs, axis=1)
            lo = float(np.quantile(bs_meds, q_lo))
            hi = float(np.quantile(bs_meds, q_hi))
            medians.append(med); med_lo.append(lo); med_hi.append(hi)

        return {
            "n_groups": n_groups,
            "group_labels": labels,
            "group_ns": ns,
            "means": means,
            "mean_ci_lo": mean_lo,
            "mean_ci_hi": mean_hi,
            "medians": medians,
            "median_ci_lo": med_lo,
            "median_ci_hi": med_hi,
            # stash arrays for tests/draw
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
        """Global ANOVA & Kruskal–Wallis (gracefully handle <2 groups)."""
        arrays: List[np.ndarray] = desc.get("_arrays", [])
        alpha = getattr(self.ctx, "alpha", 0.05)

        if len(arrays) < 2 or any(a.size == 0 for a in arrays):
            return {
                "anova":   {"statistic": float("nan"), "p_value": float("nan"), "reject": False, "alpha": alpha},
                "kruskal": {"statistic": float("nan"), "p_value": float("nan"), "reject": False, "alpha": alpha},
            }

        a_stat, a_p = f_oneway(*arrays)
        k_stat, k_p = kruskal(*arrays)

        return {
            "anova":   {"statistic": float(a_stat), "p_value": float(a_p), "reject": bool(a_p < alpha), "alpha": alpha},
            "kruskal": {"statistic": float(k_stat), "p_value": float(k_p), "reject": bool(k_p < alpha), "alpha": alpha},
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
        """Background violins/boxes; emphasize selected center + CI; subtitle with global tests."""
        ctx = self.ctx  # type: MagnitudeCentralTendencyAnovaKruskalContext

        labels = desc["group_labels"]
        arrays: List[np.ndarray] = desc["_arrays"]
        n_groups = desc["n_groups"]

        if n_groups == 0:
            return fig, ax

        x = np.arange(n_groups)

        # --- De-emphasized background: violins + boxplots ---
        parts = ax.violinplot(
            arrays,
            positions=x,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.9,
        )
        for b in parts["bodies"]:
            b.set_alpha(ctx.bg_alpha)

        box = ax.boxplot(
            arrays,
            positions=x,
            widths=ctx.box_width,
            showfliers=ctx.showfliers,
            manage_ticks=False,
            patch_artist=True,
        )
        for patch in box["boxes"]:
            patch.set_alpha(ctx.bg_alpha)

        # --- Foreground: center markers + CIs ---
        mode = (ctx.mode or "anova").lower()
        if mode == "kruskal":
            centers = desc["medians"]
            lo = desc["median_ci_lo"]
            hi = desc["median_ci_hi"]
            label = "Median (bootstrap 95% CI)"
            marker_kwargs = dict(fmt="o", capsize=4)
        else:
            centers = desc["means"]
            lo = desc["mean_ci_lo"]
            hi = desc["mean_ci_hi"]
            label = "Mean (95% CI)"
            marker_kwargs = dict(fmt="o", capsize=4)

        yerr = np.array([np.array(centers) - np.array(lo), np.array(hi) - np.array(centers)])
        ax.errorbar(x, centers, yerr=yerr, **marker_kwargs)

        # --- Titles / labels ---
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=ctx.rotate_xticks or 0,
            ha="right" if (ctx.rotate_xticks or 0) else "center",
        )
        ax.legend([label], frameon=False, loc="best")

        # --- Subtitle with global tests ---
        try:
            a = inf.get("anova", {})
            k = inf.get("kruskal", {})
            star_a = "★" if a.get("reject") else ""
            star_k = "★" if k.get("reject") else ""
            subtitle = f"ANOVA p={a.get('p_value', float('nan')):.3g}{star_a} • Kruskal p={k.get('p_value', float('nan')):.3g}{star_k}"
            ax.text(
                0.5, 1.01, subtitle,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize="small", color="dimgray"
            )
        except Exception:
            pass

        return fig, ax
