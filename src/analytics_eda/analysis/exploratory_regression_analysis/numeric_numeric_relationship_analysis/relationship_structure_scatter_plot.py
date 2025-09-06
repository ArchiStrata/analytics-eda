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
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np
import pandas as pd

from ..utils.utils import resolve_num_col, dropna_on
from ....core.visualization.base_plot import BasePlot
from ....core.visualization.context import PlotContext

# ---------------- Context ----------------

@dataclass
class RelationshipStructureScatterContext(PlotContext):
    title_template: str = "Scatter Plot of {xlabel} vs {ylabel}{modifiers}"
    xlabel: str = "X"
    ylabel: str = "Y"

# -------------- Plot ---------------------

class RelationshipStructureScatterPlot(BasePlot):
    """
    Shows the raw scatter plot of two numeric variables, highlighting the *structure*
    of their relationship (form, spread, clustering, outliers).

    Why
    ---
    In numeric–numeric EDA, the scatter plot is the foundational diagnostic for 
    understanding relationship *structure*. Unlike correlation or regression, 
    which quantify magnitude and direction, the scatter plot shows:
      - whether a relationship exists,
      - what form it takes (linear, curvilinear, clusters),
      - where outliers or anomalies may distort later analysis.

    What
    ----
    - X = numeric column; Y = numeric column.
    - No smoothing, no regression line: this is the raw view.
    - Descriptive stats summarize each variable and their joint variability.

    Inputs
    ------
    Call via the DataFrame path:
      - `cols=['<numeric_x>', '<numeric_y>']`  
      OR  
      - `role_map={'x': '<numeric_x>', 'y': '<numeric_y>'}`

    Outputs
    -------
    Returns a payload with:
      - descriptive_stats:
          {
            "n_obs": int,               # number of valid pairs
            "x_mean": float,
            "x_std": float,
            "y_mean": float,
            "y_std": float,
            "covariance": float,
            "x_min": float, "x_max": float,
            "y_min": float, "y_max": float
          }
      - inferential_stats: {}           # none at structure stage
      - chart_metadata:
          {"title","xlabel","ylabel","data_source","file_name"}
    """

    # ---------- Frame API ----------
    def validate_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> pd.DataFrame:
        """Ensure both X and Y numeric columns exist; drop NA pairs."""
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        df = dropna_on(df, x_col)
        df = dropna_on(df, y_col)
        return df

    def compute_descriptive_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, Any]:
        """Compute basic descriptive statistics directly tied to structure."""
        x_col = role_map["x"]
        y_col = role_map["y"]

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        return {
            "n_obs": len(x),
            "x_mean": float(np.mean(x)),
            "x_std": float(np.std(x, ddof=1)),
            "y_mean": float(np.mean(y)),
            "y_std": float(np.std(y, ddof=1)),
            "covariance": float(np.cov(x, y, ddof=1)[0, 1]),
            "x_min": float(np.min(x)), "x_max": float(np.max(x)),
            "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        }

    def draw_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str,str]] = None,
        fig=None,
        ax=None,
        palette=None,
    ):
        """Render the plain scatter plot."""
        x_col = role_map["x"]
        y_col = role_map["y"]

        ax.scatter(df[x_col], df[y_col], alpha=0.6)

        return fig, ax
