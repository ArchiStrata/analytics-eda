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
from scipy import stats as sps

from ..utils.utils import resolve_num_col, dropna_on
from ....core.utils.base_plot import PlotContext, BasePlot


# ---------------- Context ----------------

@dataclass
class MagnitudeAssociationResidualContext(PlotContext):
    title_template: str = "Residuals vs Fitted: {y} on {x}{modifiers}"
    xlabel: str = "Fitted values"
    ylabel: str = "Residuals"
    figsize: Tuple[int, int] = (8, 6)
    alpha: float = 0.05  # for tests using significance thresholds


# -------------- Plot ---------------------

class MagnitudeAssociationResidualPlot(BasePlot):
    """
    Residual plot for assessing model adequacy before full regression analysis.
    Use residuals vs fitted values to visually and numerically check if a linear
    model is a reasonable summary (random scatter around zero, constant spread).

    Why
    ---
    Before relying on correlation/R² or doing full inference, confirm that basic
    linear-model assumptions are not obviously violated (e.g., curvature or
    heteroscedasticity). This improves the credibility of downstream regression.

    What
    ----
    - Inputs: numeric X and Y; rows with NA in either are dropped.
    - Descriptive stats: mean residual (~0), residual std (σ̂), residual min/max and range.
      (Pattern checks—curvature, heteroscedasticity, clustering—are visual, not numeric.)
    - Inferential stats (optional EDA diagnostics):
        • Shapiro–Wilk normality test of residuals.
        • Breusch–Pagan and White tests for homoscedasticity.
    - Output: payload with descriptive_stats, inferential_stats, and chart_metadata.
    """

    # Provide {x} and {y} to the title template
    def title_kwargs(
        self,
        *,
        series=None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
    ) -> Dict[str, str]:
        x_col = (role_map or {}).get("x") or (cols[0] if cols else "")
        y_col = (role_map or {}).get("y") or (cols[1] if cols and len(cols) >= 2 else "")
        return {"x": x_col, "y": y_col, "extra_desc": "EDA diagnostics"}

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

    def _ols_fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Return (slope, intercept) from simple OLS via polyfit."""
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)

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

        # include any descriptive parameters here (none specific now; keep scaffold)
        out: Dict[str, Any] = {
            "params": {
                # add future descriptive controls here if introduced
            },
            "n_obs": float(n),
            "resid_mean": np.nan,
            "resid_std": np.nan,
            "resid_min": np.nan,
            "resid_max": np.nan,
            "resid_range": np.nan,
            # also expose slope/intercept used to compute fitted/residuals
            "slope": np.nan,
            "intercept": np.nan,
        }

        if n < 2 or np.all(x == x[0]):
            # Not enough information to form residuals
            self._cache = {}
            return out

        slope, intercept = self._ols_fit(x, y)
        fitted = slope * x + intercept
        resid = y - fitted

        resid_mean = float(np.mean(resid))
        resid_std = float(np.std(resid, ddof=1)) if n > 1 else np.nan
        resid_min = float(np.min(resid))
        resid_max = float(np.max(resid))
        resid_range = float(resid_max - resid_min)

        # cache for inference/draw
        self._cache = {"fitted": fitted, "residuals": resid}

        out.update({
            "resid_mean": resid_mean,
            "resid_std": resid_std,
            "resid_min": resid_min,
            "resid_max": resid_max,
            "resid_range": resid_range,
            "slope": float(slope),
            "intercept": float(intercept),
        })
        return out

    def compute_inferential_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, float],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, float]:
        alpha = float(getattr(self.ctx, "alpha", 0.05))
        n = int(desc.get("n_obs", 0))
        resid = (getattr(self, "_cache", {}) or {}).get("residuals", None)

        out: Dict[str, Any] = {
            "params": {"alpha": alpha}
        }

        # Not enough data to run tests
        if resid is None or n < 3:
            out["normality_shapiro"] = {
                "statistic": np.nan, "p_value": np.nan, "alpha": alpha, "reject": False
            }
            out["homoscedasticity_breusch_pagan"] = {
                "statistic": np.nan, "df": np.nan, "p_value": np.nan,
                "alpha": alpha, "reject": False
            }
            out["homoscedasticity_white"] = {
                "statistic": np.nan, "df": np.nan, "p_value": np.nan,
                "alpha": alpha, "reject": False
            }
            return out

        # Shapiro–Wilk (3 <= n <= 5000 per SciPy)
        try:
            if 3 <= n <= 5000:
                W, pW = sps.shapiro(resid.astype(float))
                out["normality_shapiro"] = {
                    "statistic": float(W),
                    "p_value": float(pW),
                    "alpha": alpha,
                    "reject": bool(pW < alpha),
                }
            else:
                out["normality_shapiro"] = {
                    "statistic": np.nan, "p_value": np.nan, "alpha": alpha, "reject": False
                }
        except Exception:
            out["normality_shapiro"] = {
                "statistic": np.nan, "p_value": np.nan, "alpha": alpha, "reject": False
            }

        # Breusch–Pagan: e^2 ~ 1 + x
        try:
            x_col = resolve_num_col(df, cols, role_map, role="x")
            x = df[x_col].to_numpy()
            e2 = resid ** 2
            X = np.column_stack([np.ones(n), x])
            beta = np.linalg.pinv(X) @ e2
            yhat = X @ beta
            ss_tot = np.sum((e2 - e2.mean()) ** 2)
            ss_res = np.sum((e2 - yhat) ** 2)
            r2_aux = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            bp_stat = float(n * r2_aux)
            bp_df = 1
            bp_p = float(sps.chi2.sf(bp_stat, df=bp_df))
            out["homoscedasticity_breusch_pagan"] = {
                "statistic": bp_stat,
                "df": float(bp_df),
                "p_value": bp_p,
                "alpha": alpha,
                "reject": bool(bp_p < alpha),
            }
        except Exception:
            out["homoscedasticity_breusch_pagan"] = {
                "statistic": np.nan, "df": np.nan, "p_value": np.nan,
                "alpha": alpha, "reject": False
            }

        # White test: e^2 ~ 1 + x + x^2
        try:
            x_col = resolve_num_col(df, cols, role_map, role="x")
            x = df[x_col].to_numpy()
            e2 = resid ** 2
            Xw = np.column_stack([np.ones(n), x, x ** 2])
            beta_w = np.linalg.pinv(Xw) @ e2
            yhat_w = Xw @ beta_w
            ss_tot_w = np.sum((e2 - e2.mean()) ** 2)
            ss_res_w = np.sum((e2 - yhat_w) ** 2)
            r2_aux_w = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else 0.0
            white_stat = float(n * r2_aux_w)
            white_df = 2
            white_p = float(sps.chi2.sf(white_stat, df=white_df))
            out["homoscedasticity_white"] = {
                "statistic": white_stat,
                "df": float(white_df),
                "p_value": white_p,
                "alpha": alpha,
                "reject": bool(white_p < alpha),
            }
        except Exception:
            out["homoscedasticity_white"] = {
                "statistic": np.nan, "df": np.nan, "p_value": np.nan,
                "alpha": alpha, "reject": False
            }

        return out

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
        # Pull fitted/residuals from cache (computed in descriptive step)
        fitted = (getattr(self, "_cache", {}) or {}).get("fitted", None)
        resid = (getattr(self, "_cache", {}) or {}).get("residuals", None)

        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        if fitted is None or resid is None:
            # Fallback: draw empty axes with zero line
            ax.axhline(0.0, linestyle="--", linewidth=1)
        else:
            ax.scatter(fitted, resid, alpha=0.6)
            ax.axhline(0.0, linestyle="--", linewidth=1)

        ax.set_title(chart_metadata["title"], pad=20)
        ax.set_xlabel(self.ctx.xlabel or "Fitted values")
        ax.set_ylabel(self.ctx.ylabel or "Residuals")
        return fig, ax
