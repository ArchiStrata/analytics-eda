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
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils.base_plot import BasePlot, PlotContext
from .validate_categorical_named_series import CategoricalSeriesMixin

@dataclass
class LorenzPlotContext(PlotContext):
    # Override defaults to match the original function behavior
    title_template: str = "Lorenz Curve of {name}{modifiers}"
    xlabel: str = "Cumulative share of categories"
    ylabel: str = "Cumulative share of counts"

class LorenzCurveCategoricalPlot(CategoricalSeriesMixin, BasePlot):
    """
    Visualizes category imbalance with a Lorenz curve and reports the Gini index.

    Returns (in BasePlot.run schema):
        {
          "descriptive_stats": {"total", "k", "gini_index"},
          "inferential_stats": {},
          "chart_metadata": {...}
        }
    """

    # ---- internal helpers (moved here) ----
    @staticmethod
    def _lorenz_curve_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Lorenz curve for nonnegative weights (e.g., category frequencies).
        Returns (x, y): cumulative share of categories (x) vs cumulative share of counts (y).
        """
        if counts.size == 0 or np.sum(counts) == 0:
            # Degenerate: return the diagonal (no inequality information)
            x = np.array([0.0, 1.0])
            y = np.array([0.0, 1.0])
            return x, y

        sorted_vals = np.sort(counts.astype(float))
        cum_vals = np.cumsum(sorted_vals)
        total = cum_vals[-1]

        # Prepend the origin (0,0)
        y = np.insert(cum_vals / total, 0, 0.0)
        x = np.linspace(0.0, 1.0, len(y))
        return x, y

    @staticmethod
    def _gini_from_lorenz(x: np.ndarray, y: np.ndarray) -> float:
        """
        Gini = 1 - 2 * area under Lorenz curve.
        Assumes x spans [0,1] and y starts at 0 and ends at 1.
        """
        area = np.trapezoid(y, x)
        return float(1.0 - 2.0 * area)

    # ---- BasePlot hooks ----

    def default_descriptive(self) -> Dict[str, Any]:
        return {"total": 0, "k": 0, "gini_index": float("nan")}

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        counts = s.value_counts()
        freq_values = counts.values.astype(float)
        total = int(freq_values.sum())
        k = int(freq_values.size)

        x_lorenz, y_lorenz = self._lorenz_curve_from_counts(freq_values)
        gini = self._gini_from_lorenz(x_lorenz, y_lorenz)

        return {
            "total": total,
            "k": k,
            "gini_index": float(gini),
            # payload for drawing:
            "x_lorenz": x_lorenz,
            "y_lorenz": y_lorenz,
        }

    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        sns.set_palette("colorblind")

        title = chart_metadata["title"]
        xlabel = chart_metadata["xlabel"] or "Cumulative share of categories"
        ylabel = chart_metadata["ylabel"] or "Cumulative share of counts"

        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        # Lorenz curve
        x_lorenz = desc["x_lorenz"]
        y_lorenz = desc["y_lorenz"]
        sns.lineplot(x=x_lorenz, y=y_lorenz, ax=ax, label="Lorenz curve")

        # Equality line
        sns.lineplot(x=[0.0, 1.0], y=[0.0, 1.0], ax=ax, linestyle="--", label="Equality line")

        # Shade gap between equality and Lorenz
        ax.fill_between(x_lorenz, y_lorenz, x_lorenz, alpha=0.25)

        # Labels & title
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        # Annotation
        ann = f"Gini index = {desc['gini_index']:.3f}\nCategories = {desc['k']}\nTotal = {desc['total']}"
        ax.text(
            0.98, 0.02, ann, transform=ax.transAxes,
            ha="right", va="bottom", fontsize="small",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        return fig, ax


def plot_balance_lorenz_curve(
    series: pd.Series,
    /,                         # only 'series' may be positional
    *,                         # everything else must be keyword args
    ctx: Optional[LorenzPlotContext] = None,
    **kwargs: Dict[str, Any],
):
    """
    Back-compat wrapper that delegates to the class-based implementation.
    - If `ctx` is provided, it's used (optionally overridden by kwargs).
    - Otherwise we construct LorenzPlotContext(**kwargs).
    """
    if ctx is None:
        ctx = LorenzPlotContext(**kwargs)          # uses the original defaults baked into the subclass
    else:
        # allow selective overrides via kwargs
        for k, v in kwargs.items():
            setattr(ctx, k, v)

    plot = LorenzCurveCategoricalPlot(ctx)
    return plot.run(series)
