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
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils.base_plot import BasePlot, PlotContext
from .validate_numeric_named_series import NumericSeriesMixin

@dataclass
class CardinalityBarContext(PlotContext):
    title_template: str = "Cardinality — Top {top_k} Value Counts for {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = "Count"
    figsize: Tuple[int, int] = (8, 6)

    # plot-specific
    top_k: int = 10
    max_unique_fraction: float = 0.05
    max_unique_values: int = 20
    integer_tolerance: float = 1e-8

class CardinalityBarPlot(NumericSeriesMixin, BasePlot):
    """
    Generate a bar chart that tells the cardinality story of a numeric variable.

    Why:
        Cardinality measures the number of distinct values.  
        • High cardinality → treat as continuous (histograms, density plots).  
        • Low cardinality → may be discrete/categorical; consider bar plots or bucketing.

    What:
        - Computes number of unique values (nunique).  
        - Ranks values by frequency and displays the top k in a bar chart.  
        - Optionally annotates data source and saves the figure.  
        - Returns cardinality metric and chart metadata for reporting.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "params": {...},
          "total", "nunique", "uniqueness_ratio", "is_discrete"
        },
        "inferential_stats": {},
        "chart_metadata": {..., "top_k": int}
      }
    """

    # (1) Title must include {top_k}; override metadata builder to format it.
    def title_kwargs(self, *, series=None, cols=None, role_map=None) -> Dict[str, Any]:
        # Make {top_k} available to the title_template AND optionally add a modifier
        return {
            "top_k": int(self.ctx.top_k),
            # If you also want "(Top 10)" in the (...) modifiers, add an extra_desc:
            # "extra_desc": f"Top {int(self.ctx.top_k)}",
        }
    
    def metadata_overrides(self, *, series=None, cols=None, role_map=None) -> Dict[str, Any]:
        # Put top_k into chart metadata payload for consumers/tests
        return {
            "top_k": int(self.ctx.top_k),
        }

    # (2) default when empty
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "params": {
                "max_unique_fraction": float(self.ctx.max_unique_fraction),
                "max_unique_values": int(self.ctx.max_unique_values),
                "integer_tolerance": float(self.ctx.integer_tolerance),
            },
            "total": 0,
            "nunique": 0,
            "uniqueness_ratio": 0.0,
            "coverage_top_k": 0.0,
            "is_discrete": None,
            # payload for draw():
            "labels": [],
            "values": np.array([], dtype=float),
        }

    # (3) descriptive stats (+ payload for drawing)
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        total = int(s.size)
        nunique = int(s.nunique())
        uniqueness_ratio = (nunique / total) if total else 0.0

        is_discrete = self._is_discrete_numeric(
            s,
            max_unique_fraction=self.ctx.max_unique_fraction,
            max_unique_values=self.ctx.max_unique_values,
            integer_tolerance=self.ctx.integer_tolerance,
        )

        counts = s.value_counts().head(self.ctx.top_k)
        labels = counts.index.astype(str).tolist()
        values = counts.values.astype(float)

        # compute coverage
        coverage = float(values.sum()) / total if total else 0.0

        return {
            "params": {
                "max_unique_fraction": float(self.ctx.max_unique_fraction),
                "max_unique_values": int(self.ctx.max_unique_values),
                "integer_tolerance": float(self.ctx.integer_tolerance),
            },
            "total": total,
            "nunique": nunique,
            "uniqueness_ratio": float(uniqueness_ratio),
            "coverage_top_k": coverage,
            "is_discrete": bool(is_discrete),
            # payload:
            "labels": labels,
            "values": values,
        }

    # (4) no inferential stats
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # (5) draw
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        title = chart_metadata["title"]
        xlabel = chart_metadata["xlabel"] or "Value"
        ylabel = chart_metadata["ylabel"] or "Count"

        sns.set_palette("colorblind")
        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        # horizontal bars
        ax.barh(desc["labels"], desc["values"])
        ax.invert_yaxis()  # highest count at top

        # build subtitle from descriptive stats
        coverage = desc.get("coverage_top_k", 0.0)
        subtitle = (
            f"{'Discrete' if desc['is_discrete'] else 'Continuous'} "
            f"| Unique: {desc['nunique']:,} ({desc['uniqueness_ratio']:.1%})"
        )
        if coverage > 0:
            subtitle += f" | Top-{self.ctx.top_k} coverage: {coverage:.1%}"

        ax.set_title(f"{title}\n{subtitle}")

        # flip axis labels to match barh orientation
        ax.set_xlabel(ylabel)   # Count
        ax.set_ylabel(xlabel)   # Value

        return fig, ax

    # ---- helper: discreteness ----
    @staticmethod
    def _is_discrete_numeric(
        s: pd.Series,
        max_unique_fraction: float = 0.05,
        max_unique_values: int = 20,
        integer_tolerance: float = 1e-8,
    ) -> bool:
        """
        Determine whether a numeric pandas Series should be treated as discrete.

        A series is considered discrete if:
        - It has an integer dtype and either:
            * The ratio of unique values to non-null entries is below `max_unique_fraction`, or
            * The total number of unique values is below `max_unique_values`.
        - It has a float dtype and either:
            * All values are within `integer_tolerance` of a whole number, or
            * Its unique-value ratio or count falls below the specified thresholds.

        Parameters
        ----------
        s : pd.Series
            Numeric data to evaluate. NaNs are ignored in all calculations.
        max_unique_fraction : float, default=0.05
            Maximum fraction of unique values (unique / total non-null) to still call discrete.
        max_unique_values : int, default=20
            Maximum absolute count of unique values to still call discrete.
        integer_tolerance : float, default=1e-8
            Tolerance for treating float values as effectively integers (e.g. 3.0000000001).

        Returns
        -------
        bool
            True if the series meets the criteria for discreteness; False otherwise.
        """
        # 1) Integer dtype
        if pd.api.types.is_integer_dtype(s.dtype):
            return (s.nunique() / len(s)) <= max_unique_fraction or s.nunique() < max_unique_values
        # 2) Float dtype
        if pd.api.types.is_float_dtype(s.dtype):
            # 2a. effectively all whole numbers?
            if np.isclose(s % 1, 0, atol=integer_tolerance).all():
                return True
            frac = s.nunique() / len(s)
            # 2b. low cardinality
            if frac < max_unique_fraction or s.nunique() < max_unique_values:
                return True
        return False
