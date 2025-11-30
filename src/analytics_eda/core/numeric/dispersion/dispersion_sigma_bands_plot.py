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
"""Sigma bands plot that emphasizes distance from the mean in standard deviation units."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from analytics_eda.core.visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator


@dataclass
class DispersionSigmaBandsPlotContext(PlotContext):
    """Context for configuring sigma band plots."""

    title_template: str = "Sigma Bands for {name}{modifiers}"
    xlabel: str = ""
    ylabel: str = "Value"
    show_subtitle: bool = True
    show_footer_summary: bool = True
    std_outlier_multiplier: float = 3.0


class DispersionSigmaBandsPlot(BasePlot):
    """Shows how data deviate from the mean using σ bands and outlier markers.

    Why this matters:
    Quality control, risk, and operational monitoring often depends on
    understanding how far observations stray from expected ranges when measured
    in standard deviation units.

    What this plot does:
    Computes the mean, σ, ±1σ/±2σ/±3σ bands, counts observations in each band,
    marks outliers beyond a configurable σ threshold, and annotates the chart
    with a concise breakdown of where the sample falls relative to the mean.
    """

    def __init__(self, ctx: DispersionSigmaBandsPlotContext):
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return default descriptive payload for sigma band plots."""
        return {
            "params": {"std_outlier_multiplier": float(self.ctx.std_outlier_multiplier)},
            "n": 0,
            "mean": None,
            "std": None,
            "sigma_1_lower": None,
            "sigma_1_upper": None,
            "sigma_2_lower": None,
            "sigma_2_upper": None,
            "sigma_3_lower": None,
            "sigma_3_upper": None,
            "extreme_lower_bound": None,
            "extreme_upper_bound": None,
            "count_within_1_sigma": 0,
            "count_between_1_2_sigma": 0,
            "count_between_2_3_sigma": 0,
            "count_beyond_3_sigma": 0,
            "count_beyond_outlier_threshold": 0,
            "extreme_lower_count": 0,
            "extreme_upper_count": 0,
        }

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute sigma boundaries, band counts, and extreme markers."""
        n = int(s.size)
        if n == 0:
            return {**self.default_descriptive(), "n": 0, "skip_plot": True, "error": "no data to display"}

        m = float(self.ctx.std_outlier_multiplier)
        mean = float(s.mean())
        std = float(s.std())
        std_finite = float(std) if np.isfinite(std) else None

        desc = self.default_descriptive()
        desc.update({"n": n, "mean": mean, "std": std_finite, "params": {"std_outlier_multiplier": m}})

        if not (std_finite and std_finite > 0):
            # Degenerate dispersion; everything collapses to the mean
            desc["sigma_1_lower"] = desc["sigma_1_upper"] = mean
            desc["sigma_2_lower"] = desc["sigma_2_upper"] = mean
            desc["sigma_3_lower"] = desc["sigma_3_upper"] = mean
            desc["extreme_lower_bound"] = desc["extreme_upper_bound"] = mean
            desc["count_within_1_sigma"] = n
            return desc

        for k in (1, 2, 3):
            desc[f"sigma_{k}_lower"] = mean - k * std_finite
            desc[f"sigma_{k}_upper"] = mean + k * std_finite

        extreme_lower = mean - m * std_finite
        extreme_upper = mean + m * std_finite
        desc["extreme_lower_bound"] = extreme_lower
        desc["extreme_upper_bound"] = extreme_upper

        z = ((s - mean) / std_finite).to_numpy()
        abs_z = np.abs(z)

        within_1 = int((abs_z <= 1).sum())
        between_1_2 = int(((abs_z > 1) & (abs_z <= 2)).sum())
        between_2_3 = int(((abs_z > 2) & (abs_z <= 3)).sum())
        beyond_3 = int((abs_z > 3).sum())
        beyond_outlier = int((abs_z > m).sum())

        lower_extreme = int((z < -m).sum())
        upper_extreme = int((z > m).sum())

        desc["count_within_1_sigma"] = within_1
        desc["count_between_1_2_sigma"] = between_1_2
        desc["count_between_2_3_sigma"] = between_2_3
        desc["count_beyond_3_sigma"] = beyond_3
        desc["count_beyond_outlier_threshold"] = beyond_outlier
        desc["extreme_lower_count"] = lower_extreme
        desc["extreme_upper_count"] = upper_extreme

        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize how observations distribute across σ bands and extremes."""
        if not desc:
            return {}

        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return {
                "context": "No non-null observations.",
                "primary_finding": None,
                "secondary_finding": None,
            }

        mean = desc.get("mean")
        std = desc.get("std")
        params = desc.get("params") or {}
        m = params.get("std_outlier_multiplier")

        context_parts = [f"n = {n:,}"]
        if self.is_finite(mean):
            context_parts.append(f"mean {self.formatter.format_numeric_value(mean, decimals=2)}")
        if self.is_finite(std):
            context_parts.append(f"σ {self.formatter.format_numeric_value(std, decimals=2)}")
        if m:
            context_parts.append(f"Outliers beyond ±{m}σ")
        context = " • ".join(context_parts)

        within_1 = int(desc.get("count_within_1_sigma", 0) or 0)
        between_1_2 = int(desc.get("count_between_1_2_sigma", 0) or 0)
        share_1 = within_1 / n
        share_2 = (within_1 + between_1_2) / n
        primary = f"{self.formatter.format_percent(share_1)} of observations fall within ±1σ; " f"{self.formatter.format_percent(share_2)} stay within ±2σ."

        outliers = int(desc.get("count_beyond_outlier_threshold", 0) or 0)
        secondary: str | None = None
        if outliers > 0 and m:
            pct_outliers = self.formatter.format_percent(outliers / n)
            lower_extreme = int(desc.get("extreme_lower_count", 0) or 0)
            upper_extreme = int(desc.get("extreme_upper_count", 0) or 0)
            secondary = f"{pct_outliers} exceed ±{m}σ " f"({lower_extreme:,} low / {upper_extreme:,} high)."

        return {
            "context": context,
            "primary_finding": primary,
            "secondary_finding": secondary,
        }

    def subtitle_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Return a compact subtitle highlighting sample size, σ, and outlier threshold."""
        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return "No non-null observations."

        parts = [f"n = {n:,}"]
        std_value = desc.get("std")
        if self.is_finite(std_value):
            parts.append(f"σ = {self.formatter.format_numeric_value(std_value, decimals=2)}")

        m = desc.get("params", {}).get("std_outlier_multiplier")
        if m:
            parts.append(f"Outliers beyond ±{m}σ")
        return " • ".join(parts)

    def footer_summary_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Return footer summary with sample size."""
        return f"n = {desc['n']}"

    def draw(
        self,
        s: pd.Series,
        desc: dict[str, Any],
        inf: dict[str, Any],
        chart_metadata: dict[str, Any],
        *,
        fig,
        ax,
        palette,
    ):
        """Render sigma bands, scatter points, extreme markers, and band counts."""
        series = s.dropna()
        if series.empty:
            return fig, ax

        mean = desc.get("mean")
        std = desc.get("std")
        n = int(desc.get("n", series.size))

        def palette_color(idx: int) -> Any:
            return palette[min(idx, len(palette) - 1)]

        # Shade ±1σ/±2σ/±3σ regions if available
        if self.is_finite(mean) and self.is_finite(std) and std and std > 0:
            band_alphas = {1: 0.18, 2: 0.12, 3: 0.08}
            for k, alpha in band_alphas.items():
                lower = desc.get(f"sigma_{k}_lower")
                upper = desc.get(f"sigma_{k}_upper")
                if self.is_finite(lower) and self.is_finite(upper):
                    ax.axhspan(lower, upper, color=palette_color(k - 1), alpha=alpha, zorder=0)

            mean_label = self.formatter.format_numeric_value(mean, decimals=2)
            ax.axhline(mean, color=palette_color(3), linewidth=1.5, label=f"Mean = {mean_label}", zorder=2)

            for k in (1, 2, 3):
                lower = desc.get(f"sigma_{k}_lower")
                upper = desc.get(f"sigma_{k}_upper")
                if self.is_finite(lower) and self.is_finite(upper):
                    color = palette_color(k)
                    ax.axhline(upper, color=color, linestyle="--", linewidth=1, label=f"+{k}σ", zorder=1.5)
                    ax.axhline(lower, color=color, linestyle="--", linewidth=1, label=f"-{k}σ", zorder=1.5)

        # Scatter plot of individual observations (sorted for readability)
        sorted_values = series.sort_values().to_numpy()
        x_positions = np.linspace(-0.18, 0.18, len(sorted_values)) if len(sorted_values) > 1 else np.array([0.0])
        extreme_lower = desc.get("extreme_lower_bound")
        extreme_upper = desc.get("extreme_upper_bound")

        if self.is_finite(extreme_lower) and self.is_finite(extreme_upper):
            extreme_mask = (sorted_values < extreme_lower) | (sorted_values > extreme_upper)
        else:
            extreme_mask = np.zeros_like(sorted_values, dtype=bool)

        inlier_mask = ~extreme_mask
        if inlier_mask.any():
            ax.scatter(
                x_positions[inlier_mask],
                sorted_values[inlier_mask],
                color=palette_color(0),
                edgecolor="white",
                linewidth=0.3,
                s=36,
                alpha=0.9,
                zorder=3,
                label="Within bands",
            )
        if extreme_mask.any():
            ax.scatter(
                x_positions[extreme_mask],
                sorted_values[extreme_mask],
                color=palette_color(4),
                edgecolor="black",
                linewidth=0.6,
                s=48,
                marker="X",
                zorder=4,
                label=f"Beyond ±{desc['params']['std_outlier_multiplier']}σ",
            )

        # Highlight mean point
        if self.is_finite(mean):
            ax.scatter(
                [0],
                [mean],
                color=palette_color(5),
                edgecolor="black",
                linewidth=0.6,
                s=70,
                marker="D",
                zorder=5,
            )

        # Annotate counts per band
        def pct_of_total(count: int) -> float:
            return (count / n) * 100 if n else 0.0

        band_lines = [
            f"Within ±1σ: {desc['count_within_1_sigma']} ({pct_of_total(desc['count_within_1_sigma']):.1f}%)",
            f"1–2σ: {desc['count_between_1_2_sigma']} ({pct_of_total(desc['count_between_1_2_sigma']):.1f}%)",
            f"2–3σ: {desc['count_between_2_3_sigma']} ({pct_of_total(desc['count_between_2_3_sigma']):.1f}%)",
            f">3σ: {desc['count_beyond_3_sigma']} ({pct_of_total(desc['count_beyond_3_sigma']):.1f}%)",
        ]
        ax.text(
            0.02,
            0.98,
            "\n".join(band_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=6,
        )

        ax.set_xticks([])
        ax.set_xlim(-0.35, 0.45)

        ax.legend(loc="upper right", fontsize="small", frameon=False)
        return fig, ax
