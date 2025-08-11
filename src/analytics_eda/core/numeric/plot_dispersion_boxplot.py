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
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils.base_plot import BasePlot, PlotContext
from .validate_numeric_named_series import NumericSeriesMixin

@dataclass
class DispersionBoxplotContext(PlotContext):
    title_template: str = "Dispersion of {name}{modifiers} (IQR & Outliers)"
    ylabel: str = "Value"
    figsize: Tuple[int, int] = (8, 6)

    # plot-specific knobs
    std_outlier_multiplier: float = 4.0

class DispersionBoxplotNumericPlot(NumericSeriesMixin, BasePlot):
    """
    Generate a boxplot (with violin silhouette) that effectively communicates
    the dispersion of a numeric variable, flagging extreme values and returning
    key statistics.

    Why:
        Understanding the spread of a dataset is essential for identifying variability, outliers, and patterns 
        that aren't evident from central tendency alone. This function helps analysts and data storytellers 
        visually and numerically communicate how values are distributed and dispersed in a dataset.

    What:
        - Accepts a pandas Series of numeric values.
        - Plots a boxplot with visual annotations that highlight data dispersion.
        - Computes and returns key dispersion metrics: standard deviation, variance, range, MAD (mean absolute deviation), coefficient of variation, and select percentiles.
        - Optionally includes data source annotation, saves the plot, and returns metadata for reproducibility.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "params": {"std_outlier_multiplier": float},
          "n","mean","std","var","min","max","range","mad","cv",
          "pct_10","pct_25","pct_75","pct_90","iqr",
          "extreme_lower_count","extreme_upper_count"
        },
        "inferential_stats": {},
        "chart_metadata": {"title","ylabel","data_source","file_name"}
      }
    """

    # Match original chart_metadata keys (exclude xlabel)
    def build_chart_metadata(self, series: pd.Series) -> Dict[str, Any]:
        from ..utils.build_chart_title import build_chart_title  # local import to mirror your utils structure
        title = build_chart_title(
            name=self.ctx.name,
            series=series,
            filter_desc=self.ctx.filter_desc,
            transform_desc=self.ctx.transform_desc,
            title_template=self.ctx.title_template,
        )
        return {
            "title": title,
            "ylabel": self.ctx.ylabel,
            "data_source": self.ctx.data_source,
            "file_name": self.ctx.file_name,
        }

    # (2) default when empty
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "params": {"std_outlier_multiplier": float(self.ctx.std_outlier_multiplier)},
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "var": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "range": float("nan"),
            "mad": float("nan"),
            "cv": float("nan"),
            "pct_10": float("nan"),
            "pct_25": float("nan"),
            "pct_75": float("nan"),
            "pct_90": float("nan"),
            "iqr": float("nan"),
            "extreme_lower_count": 0,
            "extreme_upper_count": 0,
        }

    # (3) descriptive stats
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        n = int(s.size)
        mean = float(s.mean())
        std = float(s.std())
        var = float(s.var())
        min_val = float(s.min())
        max_val = float(s.max())
        range_val = float(max_val - min_val)
        mad = float((s - s.mean()).abs().mean())
        cv = float(std / mean) if mean != 0 else float("nan")
        pct_10 = float(s.quantile(0.10))
        pct_25 = float(s.quantile(0.25))
        pct_75 = float(s.quantile(0.75))
        pct_90 = float(s.quantile(0.90))
        iqr = float(pct_75 - pct_25)

        m = float(self.ctx.std_outlier_multiplier)
        lower_bound = mean - m * std
        upper_bound = mean + m * std
        n_lower = int((s < lower_bound).sum())
        n_upper = int((s > upper_bound).sum())

        return {
            "params": {"std_outlier_multiplier": m},
            "n": n,
            "mean": mean,
            "std": std,
            "var": var,
            "min": min_val,
            "max": max_val,
            "range": range_val,
            "mad": mad,
            "cv": cv,
            "pct_10": pct_10,
            "pct_25": pct_25,
            "pct_75": pct_75,
            "pct_90": pct_90,
            "iqr": iqr,
            "extreme_lower_count": n_lower,
            "extreme_upper_count": n_upper,
        }

    # (4) no inferential stats
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # (5) draw
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        sns.set_palette("colorblind")
        palette = sns.color_palette("colorblind")

        title = chart_metadata["title"]
        ylabel = chart_metadata["ylabel"] or "Value"

        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        # 1) Thin violin silhouette behind box
        parts = ax.violinplot(
            s.to_numpy(),
            vert=True, positions=[0], widths=0.8,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(palette[0])
            pc.set_edgecolor(palette[0])
            pc.set_alpha(0.15)
            pc.set_linewidth(0.8)
            pc.set_zorder(1)

        # 2) Boxplot
        ax.boxplot(
            s.to_numpy(),
            positions=[0], widths=0.4, notch=False, patch_artist=True, showcaps=True,
            boxprops=dict(facecolor="white", linewidth=1.2),
            whiskerprops=dict(linewidth=1),
            medianprops=dict(linewidth=1.5, color=palette[1]),
            flierprops=dict(marker="o", markersize=0),  # hide default fliers
            zorder=2,
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)

        # Mean dot
        mean = desc["mean"]
        ax.scatter([0], [mean], color=palette[1], marker="o", s=60, zorder=4, label=f"Mean = {mean:.2f}")

        # Extremes (±kσ) lines and counts
        m = desc["params"]["std_outlier_multiplier"]
        lower_bound = desc["mean"] - m * desc["std"]
        upper_bound = desc["mean"] + m * desc["std"]
        ax.axhline(lower_bound, color=palette[2], linestyle="--",
                   label=f"Lower {m}σ = {lower_bound:.2f} ({desc['extreme_lower_count']})")
        ax.axhline(upper_bound, color=palette[3], linestyle="--",
                   label=f"Upper {m}σ = {upper_bound:.2f} ({desc['extreme_upper_count']})")

        # Highlight extreme points
        lower_mask = s < lower_bound
        upper_mask = s > upper_bound
        if lower_mask.any():
            ax.scatter([0] * int(lower_mask.sum()), s[lower_mask], color=palette[2], zorder=3)
        if upper_mask.any():
            ax.scatter([0] * int(upper_mask.sum()), s[upper_mask], color=palette[3], zorder=3)

        # 10th/90th percentile lines
        ax.axhline(desc["pct_10"], color="purple", linestyle=":", label=f"10th pct = {desc['pct_10']:.2f}")
        ax.axhline(desc["pct_90"], color="purple", linestyle=":", label=f"90th pct = {desc['pct_90']:.2f}")

        ax.legend(loc="upper left", fontsize="small", frameon=False)

        # Dispersion stats textbox
        text = (
            f"Std Dev = {desc['std']:.2f}\n"
            f"Variance = {desc['var']:.2f}\n"
            f"Min = {desc['min']:.2f}, Max = {desc['max']:.2f}\n"
            f"Range = {desc['range']:.2f}\n"
            f"MAD = {desc['mad']:.2f}\n"
            f"CV = {desc['cv']:.2f}\n"
            f"IQR = {desc['iqr']:.2f}"
        )
        ax.text(
            0.95, 0.95, text, transform=ax.transAxes,
            va="top", ha="right", fontsize="small",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # Sample size footer (BasePlot.run will add data_source footer if present)
        fig.text(0.99, 0.01, f"n = {desc['n']}", ha="right", va="bottom",
                 fontsize="small", color="gray")

        return fig, ax


def plot_dispersion_boxplot(
    series: pd.Series,
    /,
    *,
    ctx: Optional[DispersionBoxplotContext] = None,
    **kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Back-compat wrapper that delegates to the class-based implementation.
    - If `ctx` is provided, it's used (optionally overridden by kwargs).
    - Otherwise we construct DispersionBoxplotContext(**kwargs).
    """
    if ctx is None:
        ctx = DispersionBoxplotContext(**kwargs)
    else:
        for k, v in kwargs.items():
            setattr(ctx, k, v)

    plot = DispersionBoxplotNumericPlot(ctx)
    return plot.run(series)
