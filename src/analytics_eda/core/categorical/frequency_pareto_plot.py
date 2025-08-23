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

from ..utils.base_plot import BasePlot, PlotContext
from .validate_categorical_named_series import CategoricalSeriesMixin

@dataclass
class FrequencyParetoContext(PlotContext):
    title_template: str = "Pareto Chart of {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = "Count"
    min_value: Optional[int] = None          # threshold to collapse small categories into "Others"
    horizontal: bool = False                 # draw horizontal bars if True

class FrequencyParetoPlot(CategoricalSeriesMixin, BasePlot):
    """
    Pareto chart for categorical frequency distribution.
    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
            "mode", "total_count", "n_categories", "cumulative_count_at_80pct"
        },
        "inferential_stats": {},
        "chart_metadata": {...}
      }
    """

    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "mode": None,
            "total_count": 0,
            "n_categories": 0,
            "cumulative_count_at_80pct": 0,
            # payload to keep draw() simple (all empty)
            "counts_index": [],
            "counts_values": np.array([], dtype=float),
            "rel_freq": np.array([], dtype=float),
            "cumperc": np.array([], dtype=float),
            "threshold_idx": -1,
            "threshold_count": 0,
        }

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        counts = s.value_counts()

        # Group small categories into "Others", if requested
        min_value = getattr(self.ctx, "min_value", None)
        if min_value is not None:
            small = counts[counts < int(min_value)]
            if not small.empty:
                counts = counts[counts >= int(min_value)]
                counts["Others"] = int(small.sum())

        # Relative freq (%) and cumulative %
        rel_freq = counts / counts.sum() * 100.0
        cumperc = rel_freq.cumsum()

        # First index where cumulative >= 80%
        # (Pareto principle—there will always be one since cumperc[-1] == 100)
        threshold_idx = int(np.argmax(cumperc.values >= 80.0))
        threshold_count = int(counts.values[: threshold_idx + 1].sum())

        desc = {
            "mode": counts.index[0] if len(counts) > 0 else None,
            "total_count": int(counts.sum()),
            "n_categories": int(len(counts)),
            "cumulative_count_at_80pct": threshold_count,
            # payload for draw()
            "counts_index": counts.index.tolist(),
            "counts_values": counts.values.astype(float),
            "rel_freq": rel_freq.values.astype(float),
            "cumperc": cumperc.values.astype(float),
            "threshold_idx": threshold_idx,
            "threshold_count": threshold_count,
        }
        return desc

    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):

        # Colors
        muted = "#999999"
        accent = "#0072B2"

        idx = desc["counts_index"]
        vals = desc["counts_values"]
        rel = desc["rel_freq"]
        cum = desc["cumperc"]
        thr_i = desc["threshold_idx"]
        thr_count = desc["threshold_count"]

        n = len(idx)
        ticks = np.arange(n)

        # Choose bar orientation
        if getattr(self.ctx, "horizontal", False):
            # Colors: highlight bars up to threshold_idx inclusive
            bar_colors = [accent if i <= thr_i else muted for i in range(n)]
            bars = ax.barh(idx, vals, color=bar_colors, edgecolor="black")

            # Count + % annotations
            for bar, count, pct in zip(bars, vals, rel):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2,
                        f"{int(count)} ({pct:.1f}%)", ha="left", va="center")

            # Cumulative % on the top axis
            ax2 = ax.twiny()
            ax2.plot(cum, ticks, marker="o", linestyle="-", color="black")
            ax2.set_xlabel("Cumulative %")
            ax2.set_xlim(0, 110)
            ax2.axvline(80, color=accent, linestyle="--")
            if n > 0:
                ax2.text(80, ticks[-1], "80% threshold", ha="left", va="top", color=accent)

        else:
            bar_colors = [accent if i <= thr_i else muted for i in range(n)]
            bars = ax.bar(idx, vals, color=bar_colors, edgecolor="black")

            # Count + % annotations
            for bar, count, pct in zip(bars, vals, rel):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f"{int(count)}\n({pct:.1f}%)", ha="center", va="bottom")

            # Axes & labels
            ax.set_xticks(ticks)
            ax.set_xticklabels(idx, rotation=45, ha="right")

            # Cumulative % on right axis
            ax2 = ax.twinx()
            ax2.plot(ticks, cum, marker="o", linestyle="-", color="black")
            ax2.set_ylabel("Cumulative %")
            ax2.set_ylim(0, 110)
            if n > 0:
                ax2.axhline(80, color=accent, linestyle="--")
                ax2.text(ticks[-1], 80, "80% threshold", ha="right", va="bottom", color=accent)


        # Footnote
        fig.text(0.99, 0.01, f"Cumulative count at 80%: {thr_count}",
                 ha="right", va="bottom", fontsize=8, color="gray")

        return fig, ax
