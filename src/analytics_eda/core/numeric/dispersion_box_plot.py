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
"""Box/violin dispersion plot with outlier bands and summary statistics."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

from ..visualization.base_plot import BasePlot, PlotContext


@dataclass
class DispersionBoxplotContext(PlotContext):
    """Context options for the dispersion (box/violin) plot."""

    title_template: str = "Dispersion of {name}{modifiers} (IQR & Outliers)"
    ylabel: str = "Value"
    show_footer_summary: bool = True

    # plot-specific knobs
    std_outlier_multiplier: float = 4.0


class DispersionBoxPlot(BasePlot):
    """
    Generate a boxplot (with violin silhouette) that communicates dispersion and flags extreme values.

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
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    def default_descriptive(self) -> dict[str, Any]:
        """Return default descriptive payload and placeholders for drawing."""
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

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute dispersion metrics, outlier bands, and values used for drawing."""
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

        desc = {
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

        if n == 0:
            desc["skip_plot"] = True
            desc["error"] = "no data to display"

        return desc

    def footer_summary_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Return a compact footer summary string (e.g., sample size)."""
        return f"n = {desc['n']}"

    def draw(
        self,
        s: pd.Series,
        desc: dict[str, Any],
        inf: dict[str, Any],
        chart_metadata: dict[str, Any],
        *,
        fig,
        ax,
        palette,
    ):
        """Render violin silhouette, boxplot, outlier bands/points, lines, legend, and stats box."""
        # 1) Thin violin silhouette behind box
        parts = ax.violinplot(
            s.to_numpy(),
            vert=True,
            positions=[0],
            widths=0.8,
            showmeans=False,
            showmedians=False,
            showextrema=False,
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
            positions=[0],
            widths=0.4,
            notch=False,
            patch_artist=True,
            showcaps=True,
            boxprops=dict(facecolor="white", linewidth=1.2),
            whiskerprops=dict(linewidth=1),
            medianprops=dict(linewidth=1.5, color=palette[1]),
            flierprops=dict(marker="o", markersize=0),  # hide default fliers
            zorder=2,
        )

        # --- Safe pulls
        mean = desc.get("mean")
        std = desc.get("std")
        var_ = desc.get("var")
        min_ = desc.get("min")
        max_ = desc.get("max")
        rng = desc.get("range")
        mad = desc.get("mad")
        cv = desc.get("cv")
        iqr = desc.get("iqr")
        pct10 = desc.get("pct_10")
        pct90 = desc.get("pct_90")

        # 3) Mean dot (only if finite)
        if self.is_finite(mean):
            ax.scatter([0], [mean], color=palette[1], marker="o", s=60, zorder=4, label=f"Mean = {self.formatter.format_numeric_value(mean, decimals=2)}")

        # 4) σ-bands and extreme counts (only if mean & std are finite)
        m = desc["params"]["std_outlier_multiplier"]
        if self.is_finite(mean) and self.is_finite(std):
            lower_bound = mean - m * std
            upper_bound = mean + m * std

            ax.axhline(lower_bound, color=palette[2], linestyle="--", label=f"Lower {m}σ = {self.formatter.format_numeric_value(lower_bound, decimals=2)} ({int(desc.get('extreme_lower_count', 0))})")
            ax.axhline(upper_bound, color=palette[3], linestyle="--", label=f"Upper {m}σ = {self.formatter.format_numeric_value(upper_bound, decimals=2)} ({int(desc.get('extreme_upper_count', 0))})")

            # Highlight extreme points
            lower_mask = s < lower_bound
            upper_mask = s > upper_bound
            if lower_mask.any():
                ax.scatter([0] * int(lower_mask.sum()), s[lower_mask], color=palette[2], zorder=3)
            if upper_mask.any():
                ax.scatter([0] * int(upper_mask.sum()), s[upper_mask], color=palette[3], zorder=3)

        # 5) 10th/90th percentile lines (only if finite)
        if self.is_finite(pct10):
            ax.axhline(pct10, color="purple", linestyle=":", label=f"10th pct = {self.formatter.format_numeric_value(pct10, decimals=2)}")
        if self.is_finite(pct90):
            ax.axhline(pct90, color="purple", linestyle=":", label=f"90th pct = {self.formatter.format_numeric_value(pct90, decimals=2)}")

        ax.legend(loc="upper left", fontsize="small", frameon=False)

        # 7) Dispersion stats textbox — use safe formatter (returns "NA" for None/NaN)
        text = (
            f"Std Dev = {self.formatter.format_numeric_value(std,  decimals=2)}\n"
            f"Variance = {self.formatter.format_numeric_value(var_, decimals=2)}\n"
            f"Min = {self.formatter.format_numeric_value(min_, decimals=2)}, "
            f"Max = {self.formatter.format_numeric_value(max_, decimals=2)}\n"
            f"Range = {self.formatter.format_numeric_value(rng,  decimals=2)}\n"
            f"MAD = {self.formatter.format_numeric_value(mad,  decimals=2)}\n"
            f"CV = {self.formatter.format_numeric_value(cv,   decimals=2)}\n"
            f"IQR = {self.formatter.format_numeric_value(iqr, decimals=2)}"
        )
        ax.text(
            0.95,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize="small",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        return fig, ax
