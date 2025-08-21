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
class MagnitudeAssociationScatterOLSContext(PlotContext):
    title_template: str = "Scatter + OLS: {y} vs {x}{modifiers}"
    xlabel: str = "X"
    ylabel: str = "Y"
    figsize: Tuple[int, int] = (8, 6)
    alpha: float = 0.05                # for confidence intervals
    include_spearman: bool = True      # optional monotonic effect size


# -------------- Plot ---------------------

class MagnitudeAssociationScatterOLSPlot(BasePlot):
    """
    Quantifies the *strength* of a numeric–numeric relationship with a scatter plot
    and an OLS regression line by pairing the visual fit (OLS line) with effect-size descriptors (r, R²) to convey
    how strongly X relates to Y—separate from structure (LOESS) or direction-only narratives.

    Why
    ---
    Stakeholders need a concise measure of association magnitude. Pearson’s r and R²
    summarize strength; optional Spearman’s ρ covers monotonic patterns. Inference
    (p-value for r, CI for R² via Fisher z) adds reliability.

    What
    ----
    - Inputs: X and Y numeric columns; rows with NA in either are dropped.
    - Descriptive: Pearson’s r; optional Spearman’s ρ; R²; adjusted R²; OLS slope & intercept.
    - Inferential: two-sided p-value for H₀: ρ=0; CI for R² (derived from Fisher z CI on r).
    - Output: payload with descriptive_stats, inferential_stats, and chart_metadata.
    """

    # supply placeholders for {x} and {y} in title_template
    def title_kwargs(
        self,
        *,
        series=None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
    ) -> Dict[str, str]:
        x_col = (role_map or {}).get("x") or (cols[0] if cols else "")
        y_col = (role_map or {}).get("y") or (cols[1] if cols and len(cols) >= 2 else "")
        return {"x": x_col, "y": y_col}

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

        out: Dict[str, Any] = {
            "params": {
                "include_spearman": bool(getattr(self.ctx, "include_spearman", True)),
            },
            "n_obs": float(n),
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }

        if n < 2:
            return out

        # OLS (degree=1)
        slope, intercept = np.polyfit(x, y, 1)
        out["slope"] = float(slope)
        out["intercept"] = float(intercept)

        # Pearson r and R²
        r = float(np.corrcoef(x, y)[0, 1])
        r2 = r * r
        out["pearson_r"] = r
        out["r2"] = r2

        # Adjusted R² (p=1 predictor)
        p = 1
        out["adj_r2"] = 1.0 - (1.0 - r2) * (n - 1) / max(1, (n - p - 1))

        # Optional Spearman's rho (monotonic strength)
        if out["params"]["include_spearman"]:
            rho = float(sps.spearmanr(x, y).correlation)
            out["spearman_rho"] = rho

        return out

    def compute_inferential_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, float],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None
    ) -> Dict[str, float]:
        n = int(desc.get("n_obs", 0))
        r = desc.get("pearson_r", np.nan)
        alpha = float(getattr(self.ctx, "alpha", 0.05))

        # Always include params
        out: Dict[str, Any] = {
            "params": {
                "alpha": alpha,
            }
        }

        # Not enough info or missing r → return params only
        if not (n >= 3) or not np.isfinite(r):
            out["pearson_correlation"] = {
                "statistic": float(r) if np.isfinite(r) else np.nan,
                "p_value": np.nan,
                "ci_r": (np.nan, np.nan),
                "ci_r2": (np.nan, np.nan),
            }
            return out

        # Use resolved columns for p-value
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        _, p_val = sps.pearsonr(df[x_col], df[y_col])

        # Handle perfect correlation: Fisher z undefined at |r|=1
        EPS = 1e-12
        if abs(r) >= 1.0 - EPS:
            r = float(np.sign(r))  # snap to exactly ±1
            out["pearson_correlation"] = {
                "statistic": r,
                "p_value": float(p_val),    # SciPy returns 0.0 here
                "reject": bool(p_val < alpha),
                "ci_r": (r, r),
                "ci_r2": (1.0, 1.0),
            }
            return out

        # Fisher z CI for r, then map to R² CI via squaring endpoints
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1.0 / np.sqrt(n - 3)
        zcrit = sps.norm.ppf(1 - alpha / 2.0)
        z_lo, z_hi = z - zcrit * se, z + zcrit * se
        r_lo, r_hi = np.tanh(z_lo), np.tanh(z_hi)

        r2_lo = max(0.0, r_lo * r_lo)
        r2_hi = max(0.0, r_hi * r_hi)

        out["pearson_correlation"] = {
            "statistic": float(r),
            "p_value": float(p_val),
            "reject": bool(p_val < alpha),
            "ci_r": (float(r_lo), float(r_hi)),
            "ci_r2": (float(r2_lo), float(r2_hi)),
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
