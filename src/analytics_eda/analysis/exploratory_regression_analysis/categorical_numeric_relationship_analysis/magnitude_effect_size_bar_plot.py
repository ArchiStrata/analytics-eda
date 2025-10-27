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
"""Effect-size bar plot for categorical–numeric EDA (η², ω², ε²)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from ....core.visualization.base_plot import BasePlot
from ....core.visualization.context import PlotContext
from ..utils.utils import grouped_arrays, resolve_cat_col, resolve_num_col, truncate_labels

# ---------------- Context ----------------


@dataclass
class MagnitudeEffectSizeBarContext(PlotContext):
    """Context for effect-size bar chart."""

    title_template: str = "Effect Sizes for {name}{modifiers}"
    xlabel: str = "Effect Size"
    ylabel: str = "Magnitude (0–1)"

    # display
    annotate: bool = True
    bar_alpha: float = 0.9

    # Cohen-style reference thresholds (for η²/ω² guidance)
    thresh_small: float = 0.01
    thresh_medium: float = 0.06
    thresh_large: float = 0.14

    # label handling
    max_label_len: int | None = 30


# -------------- Plot ---------------------


class MagnitudeEffectSizeBarPlot(BasePlot):
    """
    Summarize the magnitude of association between a categorical factor and a numeric response using effect sizes.

    Why
    ---
    Hypothesis tests tell you *if* groups differ; effect sizes tell you **how much**.
    A compact bar chart with reference lines conveys magnitude clearly at a glance.

    What
    ----
    • Inputs: X = categorical, Y = numeric.
    • Metrics (0–1 scale):
      – Eta-squared (η²) — proportion of total variance explained by groups.
      – Omega-squared (ω²) — bias-corrected η².
      – Epsilon-squared (ε²) — Kruskal–Wallis–based nonparametric analogue.
    • Visual: three bars (η², ω², ε²) + dashed “small/medium/large” guides.
    """

    def default_descriptive(self) -> dict[str, Any]:
        """Return an empty/default descriptive-stats structure."""
        return {}

    # ---------- Frame API ----------

    def validate_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        """Require categorical (x) and numeric (y); drop rows with NA in either."""
        cat = resolve_cat_col(df, cols, role_map)
        num = resolve_num_col(df, cols, role_map)
        if not pd.api.types.is_numeric_dtype(df[num]):
            df = df.copy()
            df[num] = pd.to_numeric(df[num], errors="coerce")
        return df.dropna(subset=[cat, num])

    def compute_descriptive_frame(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compute η², ω², ε² from grouped data (graceful NaNs on degenerate cases)."""
        ctx = self.ctx  # type: MagnitudeEffectSizeBarContext
        cat = resolve_cat_col(df, cols, role_map)
        num = resolve_num_col(df, cols, role_map)

        arrays = grouped_arrays(df, cat, num)  # list[np.ndarray], NaNs removed
        labels_raw = [g for g, _ in df.groupby(cat, observed=True)]
        labels_disp = truncate_labels([str(x) for x in labels_raw], ctx.max_label_len)

        # Flattened values and counts
        all_values = df[num].dropna().to_numpy()
        N = int(all_values.size)
        k = int(len(arrays))

        # Default NaNs (for degenerate cases)
        eta2 = np.nan
        omega2 = np.nan
        eps2 = np.nan

        if k >= 2 and N > k and N > 1:
            grand_mean = float(np.mean(all_values))
            # SS_total
            ss_total = float(((all_values - grand_mean) ** 2).sum())
            # SS_between
            ss_between = 0.0
            for a in arrays:
                if a.size:
                    m = float(np.mean(a))
                    ss_between += a.size * (m - grand_mean) ** 2
            # SS_within
            ss_within = ss_total - ss_between

            # Eta-squared
            if ss_total > 0:
                eta2 = ss_between / ss_total

            # Omega-squared
            df_within = N - k
            if df_within > 0 and ss_total > 0:
                ms_within = ss_within / df_within
                denom = ss_total + ms_within
                omega2 = ((ss_between - (k - 1) * ms_within) / denom) if denom > 0 else np.nan

            # Epsilon-squared via Kruskal–Wallis
            # (Note: nonparametric effect-size analogue)
            if all(a.size > 0 for a in arrays):
                kw_stat, _ = kruskal(*arrays)
                eps2 = ((kw_stat - (k - 1)) / (N - k)) if (N - k) > 0 else np.nan

        return {
            "n_groups": k,
            "group_labels": labels_disp,
            "effect_sizes": {
                "eta_squared": float(eta2) if np.isfinite(eta2) else np.nan,
                "omega_squared": float(omega2) if np.isfinite(omega2) else np.nan,
                "epsilon_squared": float(eps2) if np.isfinite(eps2) else np.nan,
            },
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
        """Render bar chart with η², ω², ε² and dashed reference lines."""
        ctx = self.ctx  # type: MagnitudeEffectSizeBarContext
        es = desc["effect_sizes"]
        labels = ["η²", "ω²", "ε²"]
        values = [es["eta_squared"], es["omega_squared"], es["epsilon_squared"]]

        x = np.arange(len(labels))

        ax.bar(x, values, alpha=ctx.bar_alpha)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)

        # Reference thresholds (Cohen-style for η²/ω² guidance)
        for y, name in [
            (ctx.thresh_small, "small"),
            (ctx.thresh_medium, "medium"),
            (ctx.thresh_large, "large"),
        ]:
            ax.axhline(y, linestyle="--", linewidth=1, color="gray")
            ax.text(1.02, y, name, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize="small", color="dimgray")

        # annotations
        if ctx.annotate:
            for xi, yi in zip(x, values, strict=True):
                if np.isfinite(yi):
                    ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize="small")

        return fig, ax
