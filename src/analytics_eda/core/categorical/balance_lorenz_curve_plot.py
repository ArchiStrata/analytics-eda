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
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.base_plot import BasePlot, PlotContext
from .validate_categorical_named_series import CategoricalSeriesMixin

@dataclass
class BalanceLorenzCurveContext(PlotContext):
    title_template: str = "Lorenz Curve of {name}{modifiers}"
    xlabel: str = "Cumulative share of categories"
    ylabel: str = "Cumulative share of counts"
    show_footer_summary: bool = True

class BalanceLorenzCurvePlot(CategoricalSeriesMixin, BasePlot):
    """
    Visualizes category imbalance using a Lorenz curve and summarizes it with the Gini index.

    Why this matters:
    - Large imbalance can signal sampling bias, operational drift, or fairness risks.

    What this plot does:
    - Computes the Lorenz curve over category counts and reports the Gini index (0=perfect balance, 1=max imbalance).
    Returns (in BasePlot.run schema):
        {
          "descriptive_stats": {"total", "k", "gini_index"},
          "inferential_stats": {},
          "chart_metadata": {...}
        }
    """
    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.
        """
        return "1.0.0"

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

        self._draw_set("lorenz_curve", "x_lorenz", x_lorenz)
        self._draw_set("lorenz_curve", "y_lorenz", y_lorenz)

        return {
            "total": total,
            "k": k,
            "gini_index": float(gini)
        }

    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        if not desc or desc.get("total", 0) == 0 or np.isnan(desc.get("gini_index", float("nan"))):
            return {"summary": "No imbalance signal (no data)."}
        g = float(desc["gini_index"])
        k = int(desc["k"])
        return {
            "summary": f"Gini = {g:.3f} (0=balanced, 1=imbalanced).",
            "coverage": f"Categories = {k}, Total = {desc['total']:,}.",
        }

    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):

        # Lorenz curve
        x_lorenz = self._draw_get("lorenz_curve", "x_lorenz")
        y_lorenz = self._draw_get("lorenz_curve", "y_lorenz")

        # Equality line
        sns.lineplot(x=[0.0, 1.0], y=[0.0, 1.0], ax=ax, linestyle="--", label="Equality line", color=self.neutral_grey())

        # Lorenz curve highlighted with palette[0]
        sns.lineplot(x=x_lorenz, y=y_lorenz, ax=ax, label="Lorenz curve", color=palette[0])

        # Shade gap between equality and Lorenz
        ax.fill_between(x_lorenz, y_lorenz, x_lorenz, alpha=0.25, color=palette[0])

        # Axes: both as % with fixed domain and helpful ticks
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        
        ax.legend()

        return fig, ax

    def footer_summary_text(
        self,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
    ) -> str:
        """
        Optional override: return a short footer summary derived from descriptive/inferential stats.
        Return ''/None to suppress.
        """
        return f"Gini={desc['gini_index']:.3f} • Categories={desc['k']} • Total={desc['total']:,}"
