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
from typing import Dict, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as sps

from ..utils.utils import resolve_num_col, dropna_on
from ....core.utils.base_plot import PlotContext, BasePlot


# ---------------- Context ----------------

@dataclass
class DirectionAssociationScatterOLSTrendContext(PlotContext):
    title_template: str = "Trend Line (OLS): {y} vs {x}{modifiers}"
    xlabel: str = "X"
    ylabel: str = "Y"
    figsize: Tuple[int, int] = (8, 6)
    alpha: float = 0.05  # for slope CI


# -------------- Plot ---------------------

class DirectionAssociationScatterOLSTrendPlot(BasePlot):
    """
    Direction is about the sign and steepness of the relationship. An OLS slope (β₁)
    gives a clear, unit‑based indication of whether Y increases, decreases, or is flat
    as X changes.

    Why
    ---
    Stakeholders need to know *which way* the relationship goes and whether that slope
    is statistically different from zero. This plot isolates that question without
    duplicating magnitude metrics (e.g., r, R²).

    What
    ----
    - Inputs: numeric X and Y (rows with NA in either are dropped).
    - Descriptive stats: slope β₁, intercept β₀, sign of slope (+/–/~0), n.
    - Inferential stats: t‑test for H₀: β₁=0 (p‑value), and CI for β₁.
    - Output: payload with descriptive_stats, inferential_stats, and chart_metadata.
    """

    # provide {x} and {y} to the title template
    def title_kwargs(
        self,
        *,
        series=None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
    ) -> Dict[str, str]:
        x_col = (role_map or {}).get("x") or (cols[0] if cols else "")
        y_col = (role_map or {}).get("y") or (cols[1] if cols and len(cols) >= 2 else "")
        return {"x": x_col, "y": y_col, "extra_desc": "OLS trend"}

    # ---------- Frame API ----------
    def validate_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> pd.DataFrame:
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
    ) -> Dict[str, float]:
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        n = len(x)

        if n < 2 or np.all(x == x[0]):  # not enough info or zero variance in X
            return {
                "n_obs": float(n),
                "slope": np.nan,
                "intercept": np.nan,
                "slope_sign": np.nan,  # +1 / -1 / 0; nan when undefined
            }

        # OLS slope/intercept
        slope, intercept = np.polyfit(x, y, 1)
        slope_sign = float(np.sign(slope)) if np.isfinite(slope) and slope != 0 else 0.0

        return {
            "n_obs": float(n),
            "slope": float(slope),
            "intercept": float(intercept),
            "slope_sign": slope_sign,
        }

    def compute_inferential_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, float],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, float]:
        n = int(desc.get("n_obs", 0))
        slope = desc.get("slope", np.nan)
        alpha = getattr(self.ctx, "alpha", 0.05)

        # guard: need at least 3 points for slope test in simple regression
        if not (n >= 3) or not np.isfinite(slope):
            return {"slope_p_value": np.nan, "slope_ci_low": np.nan, "slope_ci_high": np.nan}

        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean) ** 2)
        if sxx <= 0:
            return {"slope_p_value": np.nan, "slope_ci_low": np.nan, "slope_ci_high": np.nan}

        # residuals & standard error
        y_hat = slope * x + desc.get("intercept", 0.0)
        resid = y - y_hat
        s2 = np.sum(resid ** 2) / max(1, (n - 2))         # residual variance
        se_slope = np.sqrt(s2 / sxx) if s2 >= 0 else np.nan

        if not np.isfinite(se_slope) or se_slope == 0:
            # perfect fit: p=0, CI collapses at slope
            return {
                "slope_p_value": 0.0,
                "slope_ci_low": float(slope),
                "slope_ci_high": float(slope),
            }

        # t-test for slope and (1-alpha) CI
        t_stat = slope / se_slope
        dfree = max(1, n - 2)
        p_val = 2 * sps.t.sf(abs(t_stat), df=dfree)
        tcrit = sps.t.ppf(1 - alpha / 2.0, df=dfree)
        ci_low = slope - tcrit * se_slope
        ci_high = slope + tcrit * se_slope

        return {
            "slope_p_value": float(p_val),
            "slope_ci_low": float(ci_low),
            "slope_ci_high": float(ci_high),
        }

    def draw_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, float],
        inf: Dict[str, float],
        chart_metadata: Dict[str, str],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ):
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        fig, ax = plt.subplots(figsize=self.ctx.figsize)
        ax.scatter(x, y, alpha=0.6)

        if len(x) >= 2 and np.isfinite(desc.get("slope", np.nan)):
            m, b = desc["slope"], desc["intercept"]
            xs = np.array([np.min(x), np.max(x)])
            ys = m * xs + b
            ax.plot(xs, ys, linewidth=2)

        ax.set_title(chart_metadata["title"], pad=20)
        ax.set_xlabel(self.ctx.xlabel or x_col)
        ax.set_ylabel(self.ctx.ylabel or y_col)
        return fig, ax
