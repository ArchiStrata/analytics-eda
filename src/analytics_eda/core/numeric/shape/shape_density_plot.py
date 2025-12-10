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
"""Histogram + KDE density plot with shape metrics (modes, skew, tails)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from analytics_eda.core.visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

from ..binning_rules import doane_bins, freedman_diaconis_bins, scott_bins, sturges_bins

BinMethod = Literal["sturges", "scott", "freedman_diaconis", "doane"]


@dataclass
class ShapeDensityContext(PlotContext):
    """Context for the density plot (labels, binning, alpha, bandwidth)."""

    title_template: str = "Distribution Density of {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = "Density"

    # plot-specific knobs
    bin_method: BinMethod | None = None
    bins: int | Sequence[float] | None = None
    hist_alpha: float = 0.4
    bw_adjust: float = 1.0


class ShapeDensityPlot(BasePlot):
    """
    Show how a numeric distribution's shape behaves via histogram plus KDE.

    Why this matters:
    Skew, tail weight, and the number of peaks reveal sub-populations and
    asymmetries that a table of summary stats or a bare histogram can miss.

    What this plot does:
    Accepts a numeric Series, overlays a KDE on top of a histogram, shades the
    tails (10th/90th), marks local modes, and reports shape metrics including
    skewness, kurtosis, entropy, quartile skew, and key percentiles.
    """

    def __init__(self, ctx):
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return default descriptive payload and placeholders for drawing."""
        return {
            "params": {"bins": self.ctx.bins, "bin_method": self.ctx.bin_method},
            "n": 0,
            "entropy_bits": None,
            "skewness": None,
            "kurtosis": None,
            "modes_count": 0,
            "quartile_skew": None,
            "pct_10": None,
            "pct_25": None,
            "pct_50": None,
            "pct_75": None,
            "pct_90": None,
        }

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute shape stats, resolve bins, estimate KDE, and find modes."""
        n = int(s.size)
        desc = self.default_descriptive()
        desc["n"] = n

        # Resolve bins: bin_method > ctx.bins > default 30
        chosen_bins: int | Sequence[float] | None = self.ctx.bins
        if self.ctx.bin_method:
            methods: dict[BinMethod, Callable[[pd.Series], int]] = {
                "sturges": sturges_bins,
                "scott": scott_bins,
                "freedman_diaconis": freedman_diaconis_bins,
                "doane": doane_bins,
            }
            if self.ctx.bin_method not in methods:
                raise ValueError(f"Unknown bin_method: {self.ctx.bin_method!r}. Choose from {list(methods)}.")
            chosen_bins = methods[self.ctx.bin_method](s)
        elif chosen_bins is None:
            chosen_bins = 30

        # Percentiles & quartile skew
        q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
        pct_10, pct_90 = s.quantile([0.10, 0.90])
        iqr = q3 - q1
        quartile_skew = float(((q3 + q1 - 2 * q2) / iqr) if iqr != 0 else np.nan)

        # Shape
        skewness = float(s.skew())
        kurtosis_val = float(s.kurtosis())

        # KDE grid
        kde = stats.gaussian_kde(s.to_numpy())
        kde.set_bandwidth(bw_method=kde.factor * self.ctx.bw_adjust)
        grid = np.linspace(float(s.min()), float(s.max()), 512)
        density = kde(grid)

        # Modes via peaks in KDE
        peaks, _ = find_peaks(density)
        modes_count = int(len(peaks))
        mode_x = grid[peaks]
        mode_y = density[peaks]

        # Entropy over histogram (normalize to probabilities)
        hist_counts, _ = np.histogram(s.to_numpy(), bins=chosen_bins)
        probs = hist_counts / hist_counts.sum() if hist_counts.sum() > 0 else np.array([])
        entropy_bits = float(-np.sum(probs * np.log2(probs + 1e-12))) if probs.size else None

        # Cache render-only payload for draw()
        self.draw_cache_set("density", "bins_resolved", chosen_bins)
        self.draw_cache_set("density", "grid", grid)
        self.draw_cache_set("density", "density", density)
        self.draw_cache_set("density", "mode_x", mode_x)
        self.draw_cache_set("density", "mode_y", mode_y)
        self.draw_cache_set("density", "hist_counts", hist_counts)

        desc.update(
            {
                "params": {"bins": chosen_bins, "bin_method": self.ctx.bin_method},
                "entropy_bits": entropy_bits,
                "skewness": skewness,
                "kurtosis": kurtosis_val,
                "modes_count": modes_count,
                "quartile_skew": quartile_skew,
                "pct_10": float(pct_10),
                "pct_25": float(q1),
                "pct_50": float(q2),
                "pct_75": float(q3),
                "pct_90": float(pct_90),
            }
        )

        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize skew/shape and central coverage for reporting."""
        if not desc:
            return {}

        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return {}

        fmt = self.formatter.format_numeric_value
        skewness = desc.get("skewness")
        quartile_skew = desc.get("quartile_skew")
        kurtosis_val = desc.get("kurtosis")
        modes_count = int(desc.get("modes_count", 0) or 0)
        pct_10 = desc.get("pct_10")
        pct_90 = desc.get("pct_90")

        skew_metric = quartile_skew if self.is_finite(quartile_skew) else skewness
        if self.is_finite(skew_metric):
            if skew_metric > 0.1:
                skew_phrase = "right-skewed (heavier upper tail)"
            elif skew_metric < -0.1:
                skew_phrase = "left-skewed (heavier lower tail)"
            else:
                skew_phrase = "roughly symmetric tails"
        else:
            skew_phrase = "shape symmetry unclear"

        if modes_count >= 3:
            mode_phrase = f"{modes_count} modes"
        elif modes_count == 2:
            mode_phrase = "bimodal"
        else:
            mode_phrase = "unimodal"

        primary = f"{mode_phrase.capitalize()} and {skew_phrase}."

        secondary_parts: list[str] = []
        if self.is_finite(pct_10) and self.is_finite(pct_90):
            secondary_parts.append(f"Middle 80% spans {fmt(pct_10, decimals=2)} to {fmt(pct_90, decimals=2)}.")
        if self.is_finite(kurtosis_val):
            tail_note = "heavier tails" if kurtosis_val > 0 else "lighter tails" if kurtosis_val < 0 else "normal-like tails"
            secondary_parts.append(f"Kurtosis {fmt(kurtosis_val, decimals=2)} ({tail_note}).")
        secondary = " ".join(secondary_parts) if secondary_parts else None

        context_parts = [f"n = {n:,}"]
        if self.is_finite(skewness):
            context_parts.append(f"skewness {fmt(skewness, decimals=2)}")
        if self.is_finite(kurtosis_val):
            context_parts.append(f"kurtosis {fmt(kurtosis_val, decimals=2)}")

        return {
            "context": " | ".join(context_parts),
            "primary_finding": primary,
            "secondary_finding": secondary,
        }

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
        """Render histogram, KDE, tail shading, quartile lines, modes, and legend."""
        bins_resolved = self.draw_cache_get("density", "bins_resolved")
        grid = self.draw_cache_get("density", "grid")
        density = self.draw_cache_get("density", "density")
        mode_x = self.draw_cache_get("density", "mode_x", np.array([]))
        mode_y = self.draw_cache_get("density", "mode_y", np.array([]))

        if bins_resolved is None or grid is None or density is None:
            desc["skip_plot"] = True
            return fig, ax

        # Histogram (density)
        ax.hist(
            s.to_numpy(),
            bins=bins_resolved,
            density=True,
            alpha=self.ctx.hist_alpha,
            label="Histogram",
        )

        # KDE
        ax.plot(grid, density, lw=2, label="KDE")

        # Tails
        ax.fill_between(
            grid,
            density,
            where=(grid < desc["pct_10"]),
            alpha=0.3,
            label="Bottom 10%",
        )
        ax.fill_between(
            grid,
            density,
            where=(grid > desc["pct_90"]),
            alpha=0.3,
            label="Top 10%",
        )

        # Quartiles & median
        ax.axvline(desc["pct_25"], linestyle="--", label=f"Q1 = {desc['pct_25']:.2f}")
        ax.axvline(desc["pct_50"], linestyle="-", label=f"Median = {desc['pct_50']:.2f}")
        ax.axvline(desc["pct_75"], linestyle="--", label=f"Q3 = {desc['pct_75']:.2f}")

        # Modes
        if desc["modes_count"] > 0:
            ax.scatter(mode_x, mode_y, color="green", marker="o", label=f"{desc['modes_count']} mode(s)")
            for x_loc, y_loc in zip(mode_x, mode_y, strict=True):
                ax.text(x_loc, y_loc, f"{x_loc:.2f}", ha="left", va="bottom", fontsize="x-small", color="green")

        # Stats textbox
        stats_text = (
            f"n = {desc['n']}\n"
            f"Entropy = {desc['entropy_bits']:.2f} bits\n"
            f"Skewness = {desc['skewness']:.2f}\n"
            f"Kurtosis = {desc['kurtosis']:.2f}\n"
            f"Quartile skew = {desc['quartile_skew']:.2f}"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize="small",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Legend ordering similar to legacy presentation
        handles, labels = ax.get_legend_handles_labels()
        order = [
            "Histogram",
            "KDE",
            f"Q1 = {desc['pct_25']:.2f}",
            f"Median = {desc['pct_50']:.2f}",
            f"Q3 = {desc['pct_75']:.2f}",
            "Bottom 10%",
            "Top 10%",
            f"{desc['modes_count']} mode(s)",
        ]
        ordered = [(h, lbl) for key in order for h, lbl in zip(handles, labels, strict=True) if lbl == key]
        if ordered:
            h_ord, l_ord = zip(*ordered, strict=True)
            ax.legend(h_ord, l_ord)
        else:
            ax.legend(handles, labels)

        return fig, ax
