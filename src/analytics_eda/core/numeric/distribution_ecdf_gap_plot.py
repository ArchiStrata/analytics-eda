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
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from ..visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

@dataclass
class DistributionECDFGapContext(PlotContext):
    title_template: str = "ECDF Gap Analysis of {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = "ECDF"
    enable_legend: bool = True

    # plot-specific knobs
    threshold: Optional[float] = None

class DistributionECDFGapPlot(BasePlot):
    """
    Generate an Empirical Cumulative Distribution Function (ECDF) plot that highlights and quantifies gaps in a numeric distribution.

    Why:
        Gaps—intervals with no observations—reveal holes in your data range.  
        Understanding their size, frequency, and location is critical for sampling
        strategies, imputation decisions, and recognizing subpopulation boundaries.

    What:
        - Computes sorted-value gaps between each pair of unique consecutive values.
        - Summarizes:
          • max_gap, median_gap, gap percentiles (P10, P50, P90)
          • count of gaps above a given threshold
          • total_gap_prop (fraction of range with no data)
          • max_gap_loc (midpoint of the largest gap)
        - Plots the ECDF (step function) of all observations.
        - Annotates the largest gap with a double-headed arrow and label.
        - Optionally annotates data source, and saves the figure.
        - Returns both the gap metrics and chart metadata.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "params": {"threshold": float|None},
          "n","n_unique","gaps","max_gap","median_gap",
          "pct10_gap","pct50_gap","pct90_gap",
          "n_gaps_above_thr","total_gap_prop","max_gap_loc"
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
            "params": {"threshold": self.ctx.threshold},
            "n": 0,
            "n_unique": 0,
            "gaps": [],
            "max_gap": None,
            "median_gap": None,
            "pct10_gap": None,
            "pct50_gap": None,
            "pct90_gap": None,
            "n_gaps_above_thr": (0 if self.ctx.threshold is not None else None),
            "total_gap_prop": None,
            "max_gap_loc": None
        }

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        clean = s.sort_values()
        n = int(clean.size)
        unique_vals = clean.unique()
        n_unique = int(unique_vals.size)

        if n_unique >= 2:
            gaps = np.diff(unique_vals)
            gaps_list = gaps.tolist()
            max_gap = float(gaps.max())
            median_gap = float(np.median(gaps))
            pct10_gap = float(np.percentile(gaps, 10))
            pct50_gap = float(np.percentile(gaps, 50))
            pct90_gap = float(np.percentile(gaps, 90))

            denom = float(unique_vals[-1] - unique_vals[0])
            total_gap_prop = float(gaps.sum() / denom) if denom != 0 else None

            # location of max gap midpoint
            max_idx = int(np.argmax(gaps))
            max_gap_loc = float((unique_vals[max_idx] + unique_vals[max_idx + 1]) / 2)

            thr = self.ctx.threshold
            n_gaps_above = int((gaps > thr).sum()) if thr is not None else None
        else:
            # not enough distinct values
            gaps_list = []
            max_gap = median_gap = pct10_gap = pct50_gap = pct90_gap = total_gap_prop = max_gap_loc = None
            max_idx = None
            n_gaps_above = 0 if self.ctx.threshold is not None else None

        return {
            "params": {"threshold": self.ctx.threshold},
            "n": n,
            "n_unique": n_unique,
            "gaps": gaps_list,
            "max_gap": max_gap,
            "median_gap": median_gap,
            "pct10_gap": pct10_gap,
            "pct50_gap": pct50_gap,
            "pct90_gap": pct90_gap,
            "n_gaps_above_thr": n_gaps_above,
            "total_gap_prop": total_gap_prop,
            "max_gap_loc": max_gap_loc,
            # payload for draw:
            "unique_vals": unique_vals,
            "max_gap_idx": max_idx,
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

        # ECDF from full cleaned series (not just uniques)
        clean = s.dropna().sort_values()
        n = desc["n"]
        ecdf_x = clean.values
        ecdf_y = (np.arange(1, n + 1) / n) if n > 0 else np.array([])

        if n > 0:
            ax.step(ecdf_x, ecdf_y, where="post", label="ECDF")

        # Annotate largest gap
        unique_vals = desc["unique_vals"]
        idx = desc["max_gap_idx"]
        if idx is not None:
            # ECDF level just before the gap
            count_le = int(np.searchsorted(ecdf_x, unique_vals[idx], side="right"))
            y_level = count_le / n if n > 0 else 0.0

            ax.annotate(
                "",
                xy=(unique_vals[idx], y_level),
                xytext=(unique_vals[idx + 1], y_level),
                arrowprops=dict(arrowstyle="<->", color="red"),
            )
            ax.text(
                float((unique_vals[idx] + unique_vals[idx + 1]) / 2),
                y_level + 0.02,
                f"Max gap = {desc['max_gap']:.2f}",
                ha="center",
                va="bottom",
                color="red",
                fontsize="small",
            )

        # Stats textbox
        stats_text = (
            f"n = {desc['n']}\n"
            f"n_unique = {desc['n_unique']}\n"
            f"max_gap = {desc['max_gap']:.2f}\n"
            f"median_gap = {desc['median_gap']:.2f}\n"
            f"P10 = {desc['pct10_gap']:.2f}, P50 = {desc['pct50_gap']:.2f}, P90 = {desc['pct90_gap']:.2f}\n"
            f"total_gap_prop = {desc['total_gap_prop']:.2f}"
            + (f"\n(gaps > {self.ctx.threshold}) = {desc['n_gaps_above_thr']}" if self.ctx.threshold is not None else "")
        )
        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize="small",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        return fig, ax
