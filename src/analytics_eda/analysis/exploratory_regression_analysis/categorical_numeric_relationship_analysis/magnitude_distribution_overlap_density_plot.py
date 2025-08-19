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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..utils.utils import resolve_cat_col, resolve_num_col, grouped_arrays, truncate_labels, agg_mean

from ....core.utils.base_plot import PlotContext, BasePlot

# ---------------- Context ----------------

@dataclass
class MagnitudeDistributionOverlapDensityContext(PlotContext):
    title_template: str = "Distribution Shape & Overlap for {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = "Density"
    figsize: Tuple[int, int] = (10, 6)

    # plot-specific knobs
    bw: Optional[float | str] = "scott"   # "scott", "silverman", or a float bandwidth
    grid_size: int = 256                  # number of x grid points
    padding: float = 0.05                 # extra range padding as fraction of data range
    facet_cols: int = 3                   # columns when faceting
    alpha: float = 0.7                    # line alpha for overlays
    linewidth: float = 2.0                # line width
    sort_groups_by: Optional[str] = "median"  # None|"mean"|"median" for facet ordering

# -------------- Plot ---------------------

class MagnitudeDistributionOverlapDensityPlot(BasePlot):
    """
    Compare numeric distributions across categories and quantify their overlap.

    Why
    ---
    Before variance tests or post-hoc comparisons, understand how group
    distributions differ. Visual density comparison paired with overlap metrics
    (Overlap Coefficient & Bhattacharyya Distance) shows separation vs. mixing.

    What
    ----
    - X = categorical, Y = numeric.
    - Shared-grid Gaussian KDE for each group (common bandwidth for fair comparison).
    - Pairwise metrics per (group_i, group_j):
        • overlap_coeff = ∫ min(f_i, f_j) dx  ∈ [0,1]
        • bhattacharyya = -ln ∫ sqrt(f_i f_j) dx  (smaller = more overlap)
    - Drawing:
        • ≤ 4 groups → overlay densities on one axis.
        • ≥ 5 groups → faceted small multiples (shared limits, ordered by mean/median if set).

    Inputs
    ------
    Call via DataFrame path:
      - cols=['<categorical>', '<numeric>'] OR role_map={'x': '<categorical>', 'y': '<numeric>'}

    Outputs
    -------
    {
      "descriptive_stats": {
        "n_groups": int,
        "group_labels": list[str],
        "group_ns": list[int],
        "grid": np.ndarray,             # x grid used for KDE
        "kde": Dict[str, np.ndarray],   # density values per group on 'grid'
        "overlap": List[Dict]]          # [{i,j,labels,(overlap_coeff),(bhattacharyya)}...]
      },
      "inferential_stats": {},
      "chart_metadata": {...}
    }
    """

    # ---------- Frame API ----------

    def validate_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> pd.DataFrame:
        """Require categorical (x) and numeric (y); drop rows with NA in either."""
        cat = resolve_cat_col(df, cols, role_map)
        num = resolve_num_col(df, cols, role_map)
        # ensure numeric dtype (coerce if needed, then drop NA)
        if not pd.api.types.is_numeric_dtype(df[num]):
            df = df.copy()
            df[num] = pd.to_numeric(df[num], errors="coerce")
        return df.dropna(subset=[cat, num])

    def compute_descriptive_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, Any]:
        """Build shared KDEs and pairwise overlap metrics."""
        ctx = self.ctx
        cat = resolve_cat_col(df, cols, role_map)
        num = resolve_num_col(df, cols, role_map)

        # --- group data & labels ---
        groups = grouped_arrays(df, cat, num)
        labels_raw = [g for g, _ in df.groupby(cat, observed=True)]
        labels_raw_str = [str(x) for x in labels_raw]               # stable keys for dicts/indexing
        labels_disp = truncate_labels([str(x) for x in labels_raw],
                                    getattr(ctx, "max_label_len", None))

        n_groups = len(groups)
        if n_groups == 0:
            return {
                "n_groups": 0,
                "group_labels": [],
                "group_ns": [],
                "grid": np.array([]),
                "kde": {},
                "overlap": [],
            }

        # --- shared grid ---
        pooled = np.concatenate(groups)
        xmin, xmax = np.min(pooled), np.max(pooled)
        if xmin == xmax:  # degenerate: widen slightly
            w = 1.0 if xmin == 0 else abs(xmin) * 0.05
            xmin -= w
            xmax += w
        pad = float(ctx.padding) * (xmax - xmin)
        grid = np.linspace(xmin - pad, xmax + pad, int(ctx.grid_size))

        # --- shared bandwidth (Scott / Silverman or numeric) on pooled data ---
        h = self._bandwidth(pooled, method=ctx.bw)

        # --- KDE per group on shared grid ---
        kde_map: Dict[str, np.ndarray] = {}
        ns: List[int] = []
        for lab, arr in zip(labels_raw_str, groups):
            ns.append(arr.size)
            kde_map[lab] = self._kde_gaussian(arr, grid, h)

        # --- pairwise overlap metrics ---
        overlaps: List[Dict[str, Any]] = []
        for i in range(n_groups):
            fi = kde_map[labels_raw_str[i]]
            for j in range(i + 1, n_groups):
                fj = kde_map[labels_raw_str[j]]
                # Overlap Coefficient: ∫ min(fi, fj) dx
                ovl = np.trapezoid(np.minimum(fi, fj), grid)
                # Bhattacharyya distance: -ln ∫ sqrt(fi * fj) dx
                bc = np.trapezoid(np.sqrt(fi * fj), grid)
                # guard against log(0)
                bhatta = float("-inf") if bc <= 0 else float(-math.log(bc))
                overlaps.append({
                    "i": i, "j": j,
                    "labels": (labels_raw_str[i], labels_raw_str[j]),
                    "overlap_coeff": float(ovl),
                    "bhattacharyya": bhatta,
                })

        # ----- facet ordering (store once here) -----
        facet_order = [str(x) for x in labels_raw]  # default: original order
        if ctx.sort_groups_by == "mean":
            means = agg_mean(df, cat, num)                      # Series indexed by categories
            facet_order = [str(x) for x in means.sort_values().index]
        elif ctx.sort_groups_by == "median":
            med = df.groupby(cat, observed=True)[num].median()  # Series
            facet_order = [str(x) for x in med.sort_values().index]

        return {
            "n_groups": n_groups,
            "group_labels": labels_disp,   # display labels (truncated)
            "group_labels_raw": labels_raw_str,  # raw labels for lookups
            "group_ns": ns,
            "grid": grid,
            "kde": kde_map,
            "overlap": overlaps,
            "facet_order": facet_order,
        }

    def compute_inferential_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, Any]:
        """No formal hypothesis tests here; metrics are reported in descriptive_stats."""
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
        """Overlay ≤4 groups; facet 5+ groups. All share common x/y limits."""
        ctx = self.ctx  # type: MagnitudeDistributionOverlapDensityContext
        labels: List[str] = desc["group_labels"]
        labels_raw = desc.get("group_labels_raw", labels)
        grid: np.ndarray = desc["grid"]
        kde: Dict[str, np.ndarray] = desc["kde"]
        n_groups: int = desc["n_groups"]

        if n_groups == 0:
            # empty figure
            fig, ax = plt.subplots(figsize=ctx.figsize)
            ax.set_title(chart_metadata["title"])
            ax.set_xlabel(ctx.xlabel)
            ax.set_ylabel(ctx.ylabel)
            return fig, ax

        # Common limits
        ymin = 0.0
        ymax = max((np.max(kde[l]) for l in labels), default=1.0) * 1.05

        if n_groups <= 4:
            # Overlay: use raw for lookup, display for legend
            fig, ax = plt.subplots(figsize=ctx.figsize)
            for lab_raw, lab_disp in zip(labels_raw, labels):
                ax.plot(grid, kde[str(lab_raw)], label=lab_disp,
                        alpha=ctx.alpha, linewidth=ctx.linewidth)
            ax.set_title(chart_metadata["title"])
            ax.set_xlabel(ctx.xlabel)
            ax.set_ylabel(ctx.ylabel)
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(ymin, ymax)
            ax.legend(title="Group", loc="best", frameon=False)
            return fig, ax

        # ----- faceted small multiples -----
        order = desc.get("facet_order", labels_raw)  # order in raw space
        # Map raw order -> display (truncated) labels for panel titles
        disp_map = dict(zip(labels_raw, labels))
        ordered_disp = [disp_map.get(lab, str(lab)) for lab in order]

        n = len(order)
        ncols = max(1, int(ctx.facet_cols))
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(ctx.figsize[0], max(ctx.figsize[1], 2 + 2*nrows)),
                                sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)

        for idx, (ax, lab_raw, lab_disp) in enumerate(zip(axes, order, ordered_disp)):
            ax.plot(grid, kde[str(lab_raw)], alpha=ctx.alpha, linewidth=ctx.linewidth)
            ax.set_title(lab_disp, fontsize="medium")
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(0.0, ymax)

            # manual row/col checks (portable)
            row = idx // ncols
            col = idx % ncols
            if row == nrows - 1:
                ax.set_xlabel(ctx.xlabel)
            if col == 0:
                ax.set_ylabel(ctx.ylabel)

        for ax in axes[len(order):]:
            ax.axis("off")

        fig.suptitle(chart_metadata["title"], y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig, axes[0]

    # ---------- Helpers ----------

    @staticmethod
    def _bandwidth(x: np.ndarray, method: Optional[float | str]) -> float:
        """Scott/Silverman or explicit numeric bandwidth; fall back safely if variance=0."""
        x = np.asarray(x, dtype=float)
        n = max(1, x.size)
        sd = np.nanstd(x, ddof=1) if n > 1 else 0.0
        if isinstance(method, (int, float)) and method > 0:
            h = float(method)
        else:
            if method == "silverman":
                # 0.9 * min(sd, IQR/1.34) * n**(-1/5)
                iqr = np.subtract(*np.nanpercentile(x, [75, 25])) if n > 1 else 0.0
                sigma = min(sd, iqr / 1.34) if (sd > 0 and iqr > 0) else max(sd, iqr / 1.34)
                h = 0.9 * (sigma if sigma > 0 else 1.0) * (n ** (-1/5))
            else:  # default "scott"
                h = (sd if sd > 0 else 1.0) * (n ** (-1/5))
        # avoid zero/NaN
        return max(h, np.finfo(float).eps)

    @staticmethod
    def _kde_gaussian(samples: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
        """Univariate Gaussian KDE evaluated on grid; integrates to ~1."""
        x = np.asarray(samples, dtype=float).reshape(1, -1)  # (1, n)
        g = np.asarray(grid, dtype=float).reshape(-1, 1)     # (m, 1)
        h = float(bandwidth)
        # gaussian kernel
        z = (g - x) / h  # (m, n)
        # exp(-0.5 z^2) / sqrt(2π)
        K = np.exp(-0.5 * (z * z)) / math.sqrt(2.0 * math.pi)
        # average over samples and normalize by bandwidth
        f = K.mean(axis=1) / h  # (m,)
        # ensure non-negative & tiny numeric cleanup
        f[f < 0] = 0.0
        # normalize to integrate to 1 over the grid with trapezoid rule
        area = np.trapezoid(f, grid)
        if area > 0:
            f = f / area
        return f
