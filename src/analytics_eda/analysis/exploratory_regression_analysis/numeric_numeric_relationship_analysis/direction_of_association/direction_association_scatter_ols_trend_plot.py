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
"""Direction-of-association plot: OLS trend line and slope inference for Y vs X."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from analytics_eda.core.visualization.base_plot import BasePlot
from analytics_eda.core.visualization.context import PlotContext

from ...utils.utils import dropna_on, resolve_num_col

# ---------------- Context ----------------


@dataclass
class DirectionAssociationScatterOLSTrendContext(PlotContext):
    """Configuration for the OLS trend (direction-of-association) plot."""

    title_template: str = "Direction of Association (OLS Trend): {ylabel} vs {xlabel}{modifiers}"
    xlabel: str = "X"
    ylabel: str = "Y"
    alpha: float = 0.05  # for slope CI


# -------------- Plot ---------------------


class DirectionAssociationScatterOLSTrendPlot(BasePlot):
    """Reveal direction of association via OLS slope (β₁) for two numeric variables.

    Use the sign and size of the slope to communicate whether Y tends to increase,
    decrease, or remain flat as X changes.

    Why this matters (purpose)
    ---
    Business decisions often hinge on *which way* an effect goes. By focusing narrowly
    on β₁ (not on overall fit magnitude like R²), this plot cleanly communicates
    whether higher X tends to push Y up or down—and whether that trend is
    statistically distinguishable from zero.

    What this plot does (high level)
    --------------------------------
    - Plots a scatter of Y vs X with the **OLS trend line**.
    - Reports **descriptive** direction metrics: slope β₁, intercept β₀, slope sign, n.
    - Runs an **inferential** t-test for H₀: β₁ = 0, with p-value, α, decision (reject),
      and a confidence interval for β₁.
    - Returns a payload with `descriptive_stats`, `inferential_stats` (grouped by test name),
      and `chart_metadata`.
    """

    # ---- Defaults for empty/degenerate inputs ----
    def default_descriptive(self) -> dict[str, Any]:
        """Return default/empty descriptive stats structure for the plot."""
        return {
            "params": {
                # Add descriptive parameters here if the context ever includes any
            },
            "n_obs": 0.0,
            "slope": None,
            "intercept": None,
            "slope_sign": None,
        }

    def default_inferential(self) -> dict[str, Any]:
        """Return default/empty inferential stats structure, including α."""
        alpha = float(getattr(self.ctx, "alpha", 0.05))
        return {
            "params": {"alpha": alpha},
            "slope_t_test": {
                "statistic": None,
                "df": None,
                "p_value": None,
                "alpha": alpha,
                "reject": False,
                "ci": (None, None),
            },
        }

    # ---------- Frame API ----------
    def validate_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Mapping[str, str] | None = None) -> pd.DataFrame:
        """Ensure numeric X and Y exist and drop rows with NA in either column."""
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        df = dropna_on(df, x_col)
        df = dropna_on(df, y_col)
        return df

    def compute_descriptive_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Compute OLS slope/intercept, slope sign, and sample size n."""
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        n = int(len(x))

        out: dict[str, Any] = {
            "params": {
                # echo descriptive params from context here if/when added
            },
            "n_obs": float(n),
            "slope": None,
            "intercept": None,
            "slope_sign": None,
        }

        # Not enough info or zero variance in X
        if n < 2 or np.all(x == x[0]):
            return out

        # OLS slope/intercept
        slope, intercept = np.polyfit(x, y, 1)
        slope_sign = float(np.sign(slope)) if np.isfinite(slope) and slope != 0 else 0.0

        out.update(
            {
                "slope": float(slope),
                "intercept": float(intercept),
                "slope_sign": slope_sign,
            }
        )
        return out

    def compute_inferential_frame(self, df: pd.DataFrame, desc: dict[str, Any], *, cols: Sequence[str], role_map: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Run t-test for slope (β₁), returning statistic, df, p-value, decision, and CI."""
        alpha = float(getattr(self.ctx, "alpha", 0.05))
        n = int(desc.get("n_obs", 0))
        slope = desc.get("slope", np.nan)
        intercept = desc.get("intercept", 0.0)

        out: dict[str, Any] = {"params": {"alpha": alpha}}

        # guard: need at least 3 points and finite slope
        if not (n >= 3) or not np.isfinite(slope):
            out["slope_t_test"] = {
                "statistic": np.nan,
                "df": np.nan,
                "p_value": np.nan,
                "alpha": alpha,
                "reject": False,
                "ci": (np.nan, np.nan),
            }
            return out

        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean) ** 2)
        if sxx <= 0:
            out["slope_t_test"] = {
                "statistic": np.nan,
                "df": np.nan,
                "p_value": np.nan,
                "alpha": alpha,
                "reject": False,
                "ci": (np.nan, np.nan),
            }
            return out

        # residuals & standard error of slope
        y_hat = slope * x + intercept
        resid = y - y_hat
        s2 = np.sum(resid**2) / max(1, (n - 2))  # residual variance
        se_slope = np.sqrt(s2 / sxx) if s2 >= 0 else np.nan

        if not np.isfinite(se_slope) or se_slope == 0:
            # perfect fit: p=0, CI collapses at slope
            out["slope_t_test"] = {
                "statistic": float("inf"),
                "df": float(n - 2),
                "p_value": 0.0,
                "alpha": alpha,
                "reject": True,
                "ci": (float(slope), float(slope)),
            }
            return out

        # t-test for slope and (1-alpha) CI
        t_stat = slope / se_slope
        dfree = max(1, n - 2)
        p_val = 2 * sps.t.sf(abs(t_stat), df=dfree)
        tcrit = sps.t.ppf(1 - alpha / 2.0, df=dfree)
        ci_low = slope - tcrit * se_slope
        ci_high = slope + tcrit * se_slope

        out["slope_t_test"] = {
            "statistic": float(t_stat),
            "df": float(dfree),
            "p_value": float(p_val),
            "alpha": alpha,
            "reject": bool(p_val < alpha),
            "ci": (float(ci_low), float(ci_high)),
        }
        return out

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
        """Render scatter of Y vs X and overlay the OLS trend line."""
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        ax.scatter(x, y, alpha=0.6)

        if len(x) >= 2 and np.isfinite(desc.get("slope", np.nan)):
            m, b = desc["slope"], desc["intercept"]
            xs = np.array([np.min(x), np.max(x)])
            ys = m * xs + b
            ax.plot(xs, ys, linewidth=2)

        return fig, ax
