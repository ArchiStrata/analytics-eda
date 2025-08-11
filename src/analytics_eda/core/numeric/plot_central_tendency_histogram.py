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
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils.base_plot import BasePlot, PlotContext
from .validate_numeric_named_series import NumericSeriesMixin

@dataclass
class CentralTendencyHistContext(PlotContext):
    title_template: str = "Distribution of {name}{modifiers}: Central Tendency"
    xlabel: str = "Value"
    ylabel: str = "Count"
    figsize: Tuple[int, int] = (10, 6)

    # plot-specific knob
    bins: Optional[int] = None  # if None, use sqrt rule (ceil(sqrt(n)))

class CentralTendencyHistogramPlot(NumericSeriesMixin, BasePlot):
    """
    Generate a histogram that effectively communicates the central tendency of a numeric variable.

    Why:
        This function enables analysts and data storytellers to visually communicate the distribution and central tendency of a numeric variable. By directly annotating key statistics—mean, median, mode, and a 95% confidence interval—on the histogram, the chart becomes clearer, more informative, and easier to interpret.

    What:
        - Accepts a pandas Series of numeric values.
        - Plots a histogram with annotated vertical lines for mean, median, mode(s), and 95% CI.
        - Optionally saves the figure to disk.
        - Returns descriptive statistics and chart metadata for reporting or reproducibility.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "params": {"bins": int, "mode_method": str},
          "n": int, "mean": float, "median": float, "modes": List[float]
        },
        "inferential_stats": {},
        "chart_metadata": {..., "bins": int}
      }
    """

    # (2) default when empty
    def default_descriptive(self) -> Dict[str, Any]:
        # decide bins using sqrt rule with n=0 -> 0 bins (or keep None)
        chosen_bins = self.ctx.bins if self.ctx.bins is not None else 0
        return {
            "params": {"bins": int(chosen_bins), "mode_method": None},
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "modes": [],
        }

    # (3) descriptive stats (+ payload for drawing)
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        n = int(s.size)
        # Determine bins (ctx override > sqrt rule)
        chosen_bins = self.ctx.bins if self.ctx.bins is not None else (int(math.ceil(math.sqrt(n))) if n > 0 else 0)

        mean = float(s.mean()) if n else float("nan")
        median = float(s.median()) if n else float("nan")

        mode_method: Optional[str]
        modes: List[float]

        if n:
            raw_modes = s.mode().tolist()
            if len(raw_modes) == 1:
                # A clear single mode in the data → use it
                modes = [float(raw_modes[0])]
                mode_method = "series.mode"
            else:
                # Ambiguous or multimodal → use histogram‐based bin centers
                # Calculate mode based on most frequent bins
            # use at least 1 bin when computing histogram; pass through edges if provided
                hist_bins = chosen_bins if (isinstance(chosen_bins, (list, tuple, np.ndarray)) and len(chosen_bins) > 0) \
                            else (max(int(chosen_bins), 1) if isinstance(chosen_bins, int) else 1)
                counts, edges = np.histogram(s.to_numpy(), bins=hist_bins, density=False)
                top = int(np.argmax(counts))
                max_count = counts[top]
                top_bins = np.where(counts == max_count)[0]
                modes = [float(0.5 * (edges[i] + edges[i + 1])) for i in top_bins]
                mode_method = "histogram_bin_centers"
        else:
            modes = []
            mode_method = None

        return {
            "params": {"bins": chosen_bins, "mode_method": mode_method},
            "n": n,
            "mean": mean,
            "median": median,
            "modes": modes,
        }

    # (4) inferential: none
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # (5) draw (also set chart_metadata['bins'] so the wrapper output matches legacy)
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        sns.set_palette("colorblind")

        title = chart_metadata["title"]
        xlabel = chart_metadata["xlabel"] or "Value"
        ylabel = chart_metadata["ylabel"] or "Count"

        chosen_bins = desc["params"]["bins"]
        chart_metadata["bins"] = chosen_bins  # ensure returned metadata includes final bins

        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        if isinstance(chosen_bins, (list, tuple, np.ndarray)):
            bins_arg = chosen_bins if len(chosen_bins) > 0 else 1
        elif isinstance(chosen_bins, int):
            bins_arg = max(chosen_bins, 1)
        else:
            bins_arg = 1
        sns.histplot(s, bins=bins_arg, ax=ax)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Lines: mean & median
        if desc["n"] > 0:
            ax.axvline(desc["mean"], color="black", linestyle="--", label=f"Mean = {desc['mean']:.2f}")
            ax.axvline(desc["median"], color="firebrick", linestyle="-.", label=f"Median = {desc['median']:.2f}")

            # Modes
            for i, center in enumerate(desc["modes"], start=1):
                label = "Mode" if len(desc["modes"]) == 1 else f"Mode {i}"
                ax.axvline(center, color="green", linestyle=":", linewidth=1, label=f"{label} ≈ {center:.2f}")

        # Sample size footer (BasePlot will also add data_source if present)
        fig.text(0.99, 0.01, f"n = {desc['n']}", ha="right", va="bottom", fontsize="small", color="gray")

        ax.legend()
        return fig, ax


def plot_central_tendency_histogram(
    series: pd.Series,
    /,
    *,
    ctx: Optional[CentralTendencyHistContext] = None,
    **kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Back-compat wrapper that delegates to the class-based implementation.
    - If `ctx` is provided, it's used (optionally overridden by kwargs).
    - Otherwise we construct CentralTendencyHistContext(**kwargs).
    """
    if ctx is None:
        ctx = CentralTendencyHistContext(**kwargs)
    else:
        for k, v in kwargs.items():
            setattr(ctx, k, v)

    plot = CentralTendencyHistogramPlot(ctx)
    return plot.run(series)
