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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from ....core.visualization.base_plot import BasePlot
from ....core.visualization.context import PlotContext
from ..utils.utils import dropna_on, resolve_num_col

# ---------------- Context ----------------

@dataclass
class RelationshipStructureScatterLowessContext(PlotContext):
    title_template: str = "LOWESS smooth Scatter Plot of {xlabel} vs {ylabel}{modifiers}"
    xlabel: str = "X"
    ylabel: str = "Y"

    # LOWESS controls (mirrors statsmodels.lowess)
    frac: float = 0.3
    iters: int = 1


# -------------- Plot ---------------------

class RelationshipStructureScatterLowessPlot(BasePlot):
    """
    Use a robust, non-parametric LOWESS curve to visualize structure (linear,
    curved, plateau) without committing to a global regression model.

    Why
    ---
    Clarifies relationship *structure* before introducing magnitude (correlation/R²)
    or direction (slope) analyses.

    What
    ----
    - Plots: raw scatter + LOWESS curve (from statsmodels).
    - Descriptive stats: univariate center/spread, covariance, ranges.
    - No inferential stats at this stage.

    Parameters
    ----------
    frac : float
        Fraction of data used to compute each local fit (bandwidth). Typical 0.2–0.4.
    iters : int
        Robustness iterations (reweighting). 0–1 are common for EDA.
    """

    # ---------- Frame API ----------
    def validate_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Mapping[str, str] | None = None
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
        role_map: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        """Compute descriptive statistics tied to structure (no inference)."""
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        return {
            "n_obs": int(len(x)),
            "x_mean": float(np.mean(x)) if len(x) else np.nan,
            "x_std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "y_mean": float(np.mean(y)) if len(y) else np.nan,
            "y_std": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
            "covariance": float(np.cov(x, y, ddof=1)[0, 1]) if len(x) > 1 else np.nan,
            "x_min": float(np.min(x)) if len(x) else np.nan,
            "x_max": float(np.max(x)) if len(x) else np.nan,
            "y_min": float(np.min(y)) if len(y) else np.nan,
            "y_max": float(np.max(y)) if len(y) else np.nan,
        }

    def draw_frame(
        self,
        df: pd.DataFrame,
        desc: dict[str, Any],
        inf: dict[str, Any],
        chart_metadata: dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Mapping[str, str] | None = None,
        fig=None,
        ax=None,
        palette=None,
    ):
        """Render scatter + LOWESS smooth via statsmodels."""
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        # scatter
        ax.scatter(x, y, alpha=0.6)

        # LOWESS curve (requires at least 2 points)
        if len(x) >= 2:
            smoothed = lowess(endog=y, exog=x, frac=self.ctx.frac, it=self.ctx.iters, return_sorted=True)
            if smoothed.size:
                ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2)

        return fig, ax
