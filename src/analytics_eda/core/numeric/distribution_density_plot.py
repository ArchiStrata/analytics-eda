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
from typing import Dict, Any, Optional, Literal, Sequence, Callable
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from ..utils.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator
from .binning_rules import doane_bins, freedman_diaconis_bins, scott_bins, sturges_bins

BinMethod = Literal['sturges', 'scott', 'freedman_diaconis', 'doane']

@dataclass
class DistributionDensityContext(PlotContext):
    title_template: str = "Distribution Density of {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = "Density"

    # plot-specific knobs
    bin_method: Optional[BinMethod] = None
    bins: Optional[int | Sequence[float]] = None
    hist_alpha: float = 0.4
    bw_adjust: float = 1.0


class DistributionDensityPlot(BasePlot):
    """
    Generate a histogram overlaid with a KDE to communicate the shape of a numeric distribution.

    Why:
        Understanding a distribution’s shape—its skewness, tail‐weight, and number of peaks—reveals 
        subpopulations, asymmetries, and heavy tails that a simple histogram or boxplot may obscure.

    What:
        - Accepts a pandas Series of numeric values.
        - Plots a smooth Kernel Density Estimate with:
          • vertical lines at Q1, median (Q2), Q3
          • shaded tail regions (below 10th, above 90th percentiles)
          • markers for each local mode (peak) in the KDE
        - Annotates skewness, kurtosis, quartile skewness, and mode count.
        - Optionally saves the figure to disk.
        - Returns computed shape metrics and chart parameters.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "params": {"bins": int|list, "bin_method": str|None},
          "n","entropy_bits","skewness","kurtosis","modes_count","quartile_skew",
          "pct_10","pct_25","pct_50","pct_75","pct_90"
        },
        "inferential_stats": {},
        "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
      }
    """
    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=numeric_validator()
        )
        super().__init__(ctx, parts)


    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "params": {"bins": self.ctx.bins, "bin_method": self.ctx.bin_method},
            "n": 0,
            "entropy_bits": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "modes_count": 0,
            "quartile_skew": float("nan"),
            "pct_10": float("nan"),
            "pct_25": float("nan"),
            "pct_50": float("nan"),
            "pct_75": float("nan"),
            "pct_90": float("nan")
        }

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        n = int(s.size)

        # Resolve bins: bin_method > ctx.bins > default 30
        chosen_bins: int | Sequence[float] | None = self.ctx.bins
        if self.ctx.bin_method:
            methods: dict[BinMethod, Callable[[pd.Series], int]] = {
                "sturges": sturges_bins,
                "scott": scott_bins,
                "freedman_diaconis": freedman_diaconis_bins,
                "doane": doane_bins,
            }
            if self.ctx.bin_method not in methods:
                raise ValueError(f"Unknown bin_method: {self.ctx.bin_method!r}. Choose from {list(methods)}.")
            chosen_bins = methods[self.ctx.bin_method](s)
        elif chosen_bins is None:
            chosen_bins = 30

        # Percentiles & quartile skew
        q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
        pct_10, pct_90 = s.quantile([0.10, 0.90])
        iqr = q3 - q1
        quartile_skew = float(((q3 + q1 - 2 * q2) / iqr) if iqr != 0 else np.nan)

        # Shape
        skewness = float(s.skew())
        kurtosis = float(s.kurtosis())

        # KDE grid
        kde = stats.gaussian_kde(s.to_numpy())
        kde.set_bandwidth(bw_method=kde.factor * self.ctx.bw_adjust)
        grid = np.linspace(float(s.min()), float(s.max()), 512)
        density = kde(grid)

        # Modes via peaks in KDE
        peaks, _ = find_peaks(density)
        modes_count = int(len(peaks))
        mode_x = grid[peaks]
        mode_y = density[peaks]

        # Entropy over histogram (normalize to probabilities)
        hist_counts, _ = np.histogram(s.to_numpy(), bins=chosen_bins)
        probs = hist_counts / hist_counts.sum() if hist_counts.sum() > 0 else np.array([])
        entropy_bits = float(-np.sum(probs * np.log2(probs + 1e-12))) if probs.size else float("nan")

        return {
            "params": {"bins": chosen_bins, "bin_method": self.ctx.bin_method},
            "n": n,
            "entropy_bits": entropy_bits,
            "skewness": skewness,
            "kurtosis": float(kurtosis),
            "modes_count": modes_count,
            "quartile_skew": quartile_skew,
            "pct_10": float(pct_10),
            "pct_25": float(q1),
            "pct_50": float(q2),
            "pct_75": float(q3),
            "pct_90": float(pct_90),
            # payload for draw:
            "bins_resolved": chosen_bins,
            "grid": grid,
            "density": density,
            "mode_x": mode_x.tolist(),
            "mode_y": mode_y.tolist(),
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

        # Histogram (density)
        ax.hist(
            s.to_numpy(),
            bins=desc["bins_resolved"],
            density=True,
            alpha=self.ctx.hist_alpha,
            label="Histogram",
        )

        # KDE
        ax.plot(desc["grid"], desc["density"], lw=2, label="KDE")

        # Tails
        ax.fill_between(
            desc["grid"], desc["density"],
            where=(desc["grid"] < desc["pct_10"]),
            alpha=0.3, label="Bottom 10%",
        )
        ax.fill_between(
            desc["grid"], desc["density"],
            where=(desc["grid"] > desc["pct_90"]),
            alpha=0.3, label="Top 10%",
        )

        # Quartiles & median
        ax.axvline(desc["pct_25"], linestyle="--", label=f"Q1 = {desc['pct_25']:.2f}")
        ax.axvline(desc["pct_50"], linestyle="-", label=f"Median = {desc['pct_50']:.2f}")
        ax.axvline(desc["pct_75"], linestyle="--", label=f"Q3 = {desc['pct_75']:.2f}")

        # Modes
        if desc["modes_count"] > 0:
            ax.scatter(desc["mode_x"], desc["mode_y"], color="green", marker="o",
                       label=f"{desc['modes_count']} mode(s)")
            for x_loc, y_loc in zip(desc["mode_x"], desc["mode_y"]):
                ax.text(x_loc, y_loc, f"{x_loc:.2f}",
                        ha="left", va="bottom", fontsize="x-small", color="green")

        # Stats textbox
        stats_text = (
            f"n = {desc['n']}\n"
            f"Entropy = {desc['entropy_bits']:.2f} bits\n"
            f"Skewness = {desc['skewness']:.2f}\n"
            f"Kurtosis = {desc['kurtosis']:.2f}\n"
            f"Quartile skew = {desc['quartile_skew']:.2f}"
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                ha="right", va="top", fontsize="small",
                bbox=dict(facecolor="white", alpha=0.5))

        # Legend ordering similar to legacy presentation
        handles, labels = ax.get_legend_handles_labels()
        order = [
            "Histogram", "KDE",
            f"Q1 = {desc['pct_25']:.2f}",
            f"Median = {desc['pct_50']:.2f}",
            f"Q3 = {desc['pct_75']:.2f}",
            "Bottom 10%", "Top 10%",
            f"{desc['modes_count']} mode(s)",
        ]
        ordered = [(h, l) for key in order for h, l in zip(handles, labels) if l == key]
        if ordered:
            h_ord, l_ord = zip(*ordered)
            ax.legend(h_ord, l_ord)
        else:
            ax.legend(handles, labels)

        return fig, ax
