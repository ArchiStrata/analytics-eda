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
from typing import Dict, Any
import pandas as pd

from ..visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

@dataclass
class DispersionBoxplotContext(PlotContext):
    title_template: str = "Dispersion of {name}{modifiers} (IQR & Outliers)"
    ylabel: str = "Value"

    # plot-specific knobs
    std_outlier_multiplier: float = 4.0

class DispersionBoxPlot(BasePlot):
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
    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=numeric_validator()
        )
        super().__init__(ctx, parts)

    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "params": {"std_outlier_multiplier": float(self.ctx.std_outlier_multiplier)},
            "n": 0,
            "mean": None,
            "std": None,
            "var": None,
            "min": None,
            "max": None,
            "range": None,
            "mad": None,
            "cv": None,
            "pct_10": None,
            "pct_25": None,
            "pct_75": None,
            "pct_90": None,
            "iqr": None,
            "extreme_lower_count": 0,
            "extreme_upper_count": 0,
            "extreme_lower_bound": 0,
            "extreme_upper_bound": 0,
        }

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        n = int(s.size)
        mean = float(s.mean())
        std = float(s.std())
        var = float(s.var())
        min_val = float(s.min())
        max_val = float(s.max())
        range_val = float(max_val - min_val)
        mad = float((s - s.mean()).abs().mean())
        cv = float(std / mean) if mean != 0 else None
        pct_10 = float(s.quantile(0.10))
        pct_25 = float(s.quantile(0.25))
        pct_75 = float(s.quantile(0.75))
        pct_90 = float(s.quantile(0.90))
        iqr = float(pct_75 - pct_25)

        m = float(self.ctx.std_outlier_multiplier)
        extreme_lower_bound = mean - m * std
        extreme_upper_bound = mean + m * std
        extreme_lower_count = int((s < extreme_lower_bound).sum())
        extreme_upper_count = int((s > extreme_upper_bound).sum())

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
            "extreme_lower_count": extreme_lower_count,
            "extreme_upper_count": extreme_upper_count,
            "extreme_lower_bound": extreme_lower_bound,
            "extreme_upper_bound": extreme_upper_bound,
        }

    def draw(
        self,
        s: pd.Series,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
        *,
        fig,
        ax,
        palette,
    ):

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
