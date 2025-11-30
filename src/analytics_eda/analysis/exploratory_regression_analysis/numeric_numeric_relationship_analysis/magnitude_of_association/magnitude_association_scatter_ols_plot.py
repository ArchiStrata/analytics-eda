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
# --- add at the very top of the file (fixes D100) ---
"""Scatter plot with OLS line to quantify association magnitude between two numeric variables.

Computes Pearson r, optional Spearman ρ, R²/adjusted R², draws the OLS fit, and
returns descriptive and inferential statistics alongside chart metadata.
"""

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
class MagnitudeAssociationScatterOLSContext(PlotContext):
    """Configuration for the magnitude-of-association (scatter + OLS) plot."""

    title_template: str = "Scatter + OLS: {ylabel} vs {xlabel}{modifiers}"
    xlabel: str = "X"
    ylabel: str = "Y"

    alpha: float = 0.05  # for confidence intervals
    include_spearman: bool = True  # optional monotonic effect size


# -------------- Plot ---------------------


class MagnitudeAssociationScatterOLSPlot(BasePlot):
    """Quantify the strength of a numeric–numeric relationship with scatter + OLS.

    Pairs the visual fit (OLS line) with effect-size descriptors (r, R²) to convey how
    strongly X relates to Y—separate from structure (LOWESS) or direction-only narratives.

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

    def default_descriptive(self) -> dict[str, Any]:
        """Return an empty/default descriptive-stats structure."""
        return {}

    # ---------- Frame API ----------
    def validate_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Mapping[str, str] | None = None) -> pd.DataFrame:
        """Validate that X and Y exist and drop rows with NA in either column."""
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        df = dropna_on(df, x_col)
        df = dropna_on(df, y_col)
        return df

    def compute_descriptive_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Mapping[str, str] | None = None) -> dict[str, float]:
        """Compute OLS slope/intercept, Pearson r, R²/adj-R², and optional Spearman ρ."""
        x_col = resolve_num_col(df, cols, role_map, role="x")
        y_col = resolve_num_col(df, cols, role_map, role="y")
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        n = len(x)

        out: dict[str, Any] = {
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

    def compute_inferential_frame(self, df: pd.DataFrame, desc: dict[str, float], *, cols: Sequence[str], role_map: Mapping[str, str] | None = None) -> dict[str, float]:
        """Compute inference for correlation (two-sided p, CI for r and R² via Fisher z)."""
        n = int(desc.get("n_obs", 0))
        r = desc.get("pearson_r", np.nan)
        alpha = float(getattr(self.ctx, "alpha", 0.05))

        # Always include params
        out: dict[str, Any] = {
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
                "p_value": float(p_val),  # SciPy returns 0.0 here
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
        """Render scatter points and the fitted OLS line (if available)."""
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
