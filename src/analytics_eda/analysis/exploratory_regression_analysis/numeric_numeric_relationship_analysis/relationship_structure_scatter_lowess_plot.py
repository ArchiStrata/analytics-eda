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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

from ..utils.utils import resolve_num_col, dropna_on
from ....core.utils.base_plot import PlotContext, BasePlot


# ---------------- Context ----------------

@dataclass
class RelationshipStructureScatterLowessContext(PlotContext):
    title_template: str = "Scatter Plot of {x} vs {y}{modifiers}"
    xlabel: str = "X"
    ylabel: str = "Y"
    figsize: Tuple[int, int] = (8, 6)

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

    # Provide {x} and {y} for the title template and add a subtle descriptor
    def title_kwargs(
        self,
        *,
        series=None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
    ) -> Dict[str, Any]:
        x_col = (role_map or {}).get("x") or (cols[0] if cols else "")
        y_col = (role_map or {}).get("y") or (cols[1] if cols and len(cols) >= 2 else "")
        return {"x": x_col, "y": y_col, "extra_desc": "LOWESS smooth"}

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

    def compute_inferential_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, Any]:
        """No inferential stats for structure-stage LOWESS overlay."""
        return {}

    def draw_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ):
        """Render scatter + LOWESS smooth via statsmodels."""
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        # scatter
        ax.scatter(x, y, alpha=0.6)

        # LOWESS curve (requires at least 2 points)
        if len(x) >= 2:
            smoothed = lowess(endog=y, exog=x, frac=self.ctx.frac, it=self.ctx.iters, return_sorted=True)
            if smoothed.size:
                ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2)

        # labels & title
        ax.set_title(chart_metadata["title"], pad=20)
        ax.set_xlabel(self.ctx.xlabel or x_col)
        ax.set_ylabel(self.ctx.ylabel or y_col)

        return fig, ax
