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
from scipy import stats as sps

from ..utils.utils import resolve_num_col, dropna_on
from ....core.utils.base_plot import PlotContext, BasePlot


# ---------------- Context ----------------

@dataclass
class DirectionAssociationScatterOLSTrendContext(PlotContext):
    title_template: str = "Direction of Association (OLS Trend): {ylabel} vs {xlabel}{modifiers}"
    xlabel: str = "X"
    ylabel: str = "Y"
    alpha: float = 0.05  # for slope CI


# -------------- Plot ---------------------

class DirectionAssociationScatterOLSTrendPlot(BasePlot):
    """
    Reveal the **direction** of the relationship between two numeric variables using
    the OLS slope (β₁). The slope’s sign and size answer whether Y tends to increase,
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
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "params": {
                # Add descriptive parameters here if the context ever includes any
            },
            "n_obs": 0.0,
            "slope": float("nan"),
            "intercept": float("nan"),
            "slope_sign": float("nan"),
        }

    def default_inferential(self) -> Dict[str, Any]:
        alpha = float(getattr(self.ctx, "alpha", 0.05))
        return {
            "params": {"alpha": alpha},
            "slope_t_test": {
                "statistic": float("nan"),
                "df": float("nan"),
                "p_value": float("nan"),
                "alpha": alpha,
                "reject": False,
                "ci": (float("nan"), float("nan")),
            }
        }
    
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
    ) -> Dict[str, Any]:
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        n = int(len(x))

        out: Dict[str, Any] = {
            "params": {
                # echo descriptive params from context here if/when added
            },
            "n_obs": float(n),
            "slope": float("nan"),
            "intercept": float("nan"),
            "slope_sign": float("nan"),
        }

        # Not enough info or zero variance in X
        if n < 2 or np.all(x == x[0]):
            return out

        # OLS slope/intercept
        slope, intercept = np.polyfit(x, y, 1)
        slope_sign = float(np.sign(slope)) if np.isfinite(slope) and slope != 0 else 0.0

        out.update({
            "slope": float(slope),
            "intercept": float(intercept),
            "slope_sign": slope_sign,
        })
        return out

    def compute_inferential_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, Any]:
        alpha = float(getattr(self.ctx, "alpha", 0.05))
        n = int(desc.get("n_obs", 0))
        slope = desc.get("slope", np.nan)
        intercept = desc.get("intercept", 0.0)

        out: Dict[str, Any] = {
            "params": {"alpha": alpha}
        }

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
        s2 = np.sum(resid ** 2) / max(1, (n - 2))  # residual variance
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
