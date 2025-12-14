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
    tails (10th/90th), marks local modes with strength callouts, annotates tail
    balance, and reports shape metrics including skewness, kurtosis, entropy,
    quartile skew, key percentiles, a mean line with reliability guidance, and a
    summary badge (pattern, skew level, mean reliability).
    """

    def __init__(self, ctx):
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.1.0"

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
            "mean": None,
            "mean_median_gap": None,
            "mean_median_gap_ratio": None,
            "mean_reliability": None,
            "mean_reliability_reason": None,
            "skew_level": None,
            "skew_metric_used": None,
            "skew_alignment": None,
            "tail_balance_ratio": None,
            "tail_balance_direction": None,
            "modal_strengths": [],
            "summary_badge": None,
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
        mean_val = float(s.mean())

        skew_metric = quartile_skew if self.is_finite(quartile_skew) else skewness
        skew_metric_used = "quartile_skew" if self.is_finite(quartile_skew) else "moment_skew"
        abs_skew = abs(skew_metric) if self.is_finite(skew_metric) else np.nan
        abs_skew_for_level = abs_skew

        skew_alignment = None
        if self.is_finite(quartile_skew) and self.is_finite(skewness):
            skew_alignment = "aligned" if np.sign(quartile_skew) == np.sign(skewness) or (abs(quartile_skew) < 1e-8 and abs(skewness) < 1e-8) else "divergent"
            abs_skew_for_level = max(abs_skew, abs(skewness) * 0.6)

        skew_level: str | None = None
        if self.is_finite(abs_skew_for_level):
            if abs_skew_for_level < 0.1:
                skew_level = "none"
            elif abs_skew_for_level < 0.4:
                skew_level = "minimal"
            elif abs_skew_for_level < 1.0:
                skew_level = "moderate"
            else:
                skew_level = "strong"

            if n < 15 and skew_level in ("moderate", "strong"):
                skew_level = "moderate" if skew_level == "strong" else "minimal"
        if n < 3:
            skew_level = "unknown"

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
        modal_strengths: list[dict[str, float | str]] = []
        if modes_count > 0 and mode_y.size > 0:
            peak_max = float(np.max(mode_y))
            for x_loc, y_loc in zip(mode_x, mode_y, strict=True):
                rel_height = float(y_loc / peak_max) if peak_max > 0 else 0.0
                if rel_height >= 0.85:
                    strength = "primary"
                elif rel_height >= 0.55:
                    strength = "strong"
                elif rel_height >= 0.35:
                    strength = "supporting"
                else:
                    strength = "weak"
                modal_strengths.append({"x": float(x_loc), "y": float(y_loc), "relative_height": rel_height, "strength": strength})

        # Entropy over histogram (normalize to probabilities)
        hist_counts, _ = np.histogram(s.to_numpy(), bins=chosen_bins)
        probs = hist_counts / hist_counts.sum() if hist_counts.sum() > 0 else np.array([])
        entropy_bits = float(-np.sum(probs * np.log2(probs + 1e-12))) if probs.size else None

        lower_tail_span = float(q1 - pct_10)
        upper_tail_span = float(pct_90 - q3)
        denom = lower_tail_span if lower_tail_span != 0 else (abs(q2) if q2 != 0 else 1.0)
        tail_balance_ratio = float(upper_tail_span / denom) if denom != 0 else np.nan
        tail_balance_direction = None
        if self.is_finite(tail_balance_ratio):
            if tail_balance_ratio > 1.15:
                tail_balance_direction = "upper-heavy"
            elif tail_balance_ratio < 0.87:
                tail_balance_direction = "lower-heavy"
            else:
                tail_balance_direction = "balanced"

        mean_median_gap = float(mean_val - q2)
        denom_gap = iqr if iqr > 0 else (abs(q2) if q2 != 0 else 1.0)
        mean_median_gap_ratio = float(abs(mean_median_gap) / denom_gap)
        mean_reliability = "reliable"
        mean_reliability_reason = "Mean close to median"
        if n < 3:
            mean_reliability = "unknown"
            mean_reliability_reason = "Sample too small to assess mean reliability"
        elif not self.is_finite(skew_metric):
            mean_reliability = "unknown"
            mean_reliability_reason = "Skew unclear"
        elif skew_level in ("moderate", "strong") or mean_median_gap_ratio >= 0.35:
            mean_reliability = "prefer_median"
            mean_reliability_reason = "Skew pulls mean away from median"
        elif mean_median_gap_ratio >= 0.18 or skew_level == "minimal":
            mean_reliability = "caution"
            mean_reliability_reason = "Mild skew nudges mean"
        if n < 10 and mean_reliability == "prefer_median":
            mean_reliability = "caution"
            mean_reliability_reason = "Small sample; be cautious with mean"

        # Cache render-only payload for draw()
        self.draw_cache_set("density", "bins_resolved", chosen_bins)
        self.draw_cache_set("density", "grid", grid)
        self.draw_cache_set("density", "density", density)
        self.draw_cache_set("density", "mode_x", mode_x)
        self.draw_cache_set("density", "mode_y", mode_y)
        self.draw_cache_set("density", "hist_counts", hist_counts)
        self.draw_cache_set("density", "modal_strengths", modal_strengths)

        skew_direction = "unknown"
        if n < 3:
            skew_direction = "unknown"
        elif self.is_finite(skew_metric):
            if skew_metric > 0:
                skew_direction = "right"
            elif skew_metric < 0:
                skew_direction = "left"
            else:
                skew_direction = "flat"

        skew_descriptor = skew_level or "unknown"
        if skew_direction == "unknown":
            skew_badge_text = skew_descriptor if skew_descriptor != "unknown" else "unknown"
        else:
            skew_badge_text = f"{skew_descriptor} {skew_direction}".strip()

        desc.update(
            {
                "params": {"bins": chosen_bins, "bin_method": self.ctx.bin_method},
                "entropy_bits": entropy_bits,
                "skewness": skewness,
                "kurtosis": kurtosis_val,
                "skew_level": skew_level,
                "skew_metric_used": skew_metric_used,
                "skew_alignment": skew_alignment,
                "modes_count": modes_count,
                "modal_strengths": modal_strengths,
                "quartile_skew": quartile_skew,
                "pct_10": float(pct_10),
                "pct_25": float(q1),
                "pct_50": float(q2),
                "pct_75": float(q3),
                "pct_90": float(pct_90),
                "mean": mean_val,
                "mean_median_gap": mean_median_gap,
                "mean_median_gap_ratio": mean_median_gap_ratio,
                "mean_reliability": mean_reliability,
                "mean_reliability_reason": mean_reliability_reason,
                "tail_balance_ratio": tail_balance_ratio,
                "tail_balance_direction": tail_balance_direction,
                "summary_badge": {
                    "pattern": "unimodal" if modes_count <= 1 else "bimodal" if modes_count == 2 else "multimodal",
                    "skew": skew_badge_text,
                    "mean": mean_reliability or "unknown",
                },
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
        skew_level = desc.get("skew_level")
        skew_dir = "right" if self.is_finite(skew_metric) and skew_metric > 0.05 else "left" if self.is_finite(skew_metric) and skew_metric < -0.05 else None
        if skew_level:
            skew_phrase = f"{skew_level} {skew_dir + ' ' if skew_dir else ''}skew".strip()
        else:
            skew_phrase = "shape symmetry unclear"

        tail_dir = desc.get("tail_balance_direction")
        if tail_dir == "upper-heavy":
            tail_phrase = "upper tail stretches further"
        elif tail_dir == "lower-heavy":
            tail_phrase = "lower tail stretches further"
        elif tail_dir == "balanced":
            tail_phrase = "tails are balanced"
        else:
            tail_phrase = None

        if modes_count >= 3:
            mode_phrase = f"{modes_count} modes"
        elif modes_count == 2:
            mode_phrase = "bimodal"
        else:
            mode_phrase = "unimodal"

        primary_parts = [f"{mode_phrase.capitalize()} with {skew_phrase}"]
        if tail_phrase:
            primary_parts.append(tail_phrase)
        primary = "; ".join(primary_parts) + "."

        secondary_parts: list[str] = []
        if self.is_finite(pct_10) and self.is_finite(pct_90):
            secondary_parts.append(f"Middle 80% spans {fmt(pct_10, decimals=2)} to {fmt(pct_90, decimals=2)}.")
        mean_reliability = desc.get("mean_reliability")
        if mean_reliability in ("caution", "prefer_median"):
            secondary_parts.append(desc.get("mean_reliability_reason") or "Mean may be pulled by skew; median is steadier.")
        elif mean_reliability == "unknown":
            secondary_parts.append("Mean reliability unknown; median is steadier.")
        if self.is_finite(kurtosis_val):
            tail_note = "heavier tails" if kurtosis_val > 0 else "lighter tails" if kurtosis_val < 0 else "normal-like tails"
            secondary_parts.append(f"Kurtosis {fmt(kurtosis_val, decimals=2)} ({tail_note}).")
        secondary = " ".join(secondary_parts) if secondary_parts else None

        context_parts = [f"n = {n:,}"]
        if self.is_finite(skewness):
            context_parts.append(f"skewness {fmt(skewness, decimals=2)}")
        if skew_level:
            context_parts.append(f"skew level: {skew_level}")
        mean_reliability = desc.get("mean_reliability")
        if mean_reliability:
            context_parts.append(f"mean reliability: {mean_reliability}")
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
        modal_strengths = self.draw_cache_get("density", "modal_strengths", [])

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

        tail_dir = desc.get("tail_balance_direction")
        tail_ratio = desc.get("tail_balance_ratio")
        if tail_dir:
            tail_ratio_txt = f"{tail_ratio:.2f}x" if self.is_finite(tail_ratio) else ""
            tail_text = f"Tails: {tail_dir.replace('-', ' ')} {tail_ratio_txt}".strip()
            tail_note = ax.text(
                0.02,
                0.02,
                tail_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize="x-small",
                color=self.neutral_grey("dark"),
                bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.3"),
            )
            self.register_annotations(ax, tail_note)

        # Quartiles & median
        ax.axvline(desc["pct_25"], linestyle="--", label=f"Q1 = {desc['pct_25']:.2f}")
        ax.axvline(desc["pct_50"], linestyle="-", label=f"Median = {desc['pct_50']:.2f}")
        ax.axvline(desc["pct_75"], linestyle="--", label=f"Q3 = {desc['pct_75']:.2f}")
        mean_val = desc.get("mean")
        mean_label = None
        if self.is_finite(mean_val):
            mean_label = f"Mean = {mean_val:.2f}"
            ax.axvline(mean_val, linestyle=":", color=palette[1], label=mean_label)

        # Modes
        if desc["modes_count"] > 0:
            ax.scatter(mode_x, mode_y, color="green", marker="o", label=f"{desc['modes_count']} mode(s)")
            strength_map = {}
            for entry in modal_strengths:
                strength_map[entry.get("x")] = entry.get("strength")
            for x_loc, y_loc in zip(mode_x, mode_y, strict=True):
                strength = strength_map.get(float(x_loc)) or "mode"
                ax.text(
                    x_loc,
                    y_loc,
                    f"{x_loc:.2f}\n{strength}",
                    ha="left",
                    va="bottom",
                    fontsize="x-small",
                    color="green",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="green", boxstyle="round,pad=0.2"),
                )

        # Stats textbox
        stats_lines = [f"n = {desc['n']}"]
        if self.is_finite(desc.get("entropy_bits")):
            stats_lines.append(f"Entropy = {desc['entropy_bits']:.2f} bits")
        if self.is_finite(desc.get("skewness")):
            stats_lines.append(f"Skewness = {desc['skewness']:.2f}")
        if desc.get("skew_level"):
            stats_lines.append(f"Skew level: {desc['skew_level']}")
        if self.is_finite(desc.get("kurtosis")):
            stats_lines.append(f"Kurtosis = {desc['kurtosis']:.2f}")
        if self.is_finite(desc.get("quartile_skew")):
            stats_lines.append(f"Quartile skew = {desc['quartile_skew']:.2f}")
        stats_lines.append(f"Mean reliability: {desc.get('mean_reliability') or 'unknown'}")
        stats_text = "\n".join(stats_lines)
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

        # Summary badge (pattern, skew level, mean reliability)
        badge = desc.get("summary_badge") or {}
        badge_lines = []
        if badge.get("pattern"):
            badge_lines.append(f"Pattern: {badge['pattern']}")
        if badge.get("skew"):
            badge_lines.append(f"Skew: {badge['skew']}")
        if badge.get("mean"):
            badge_lines.append(f"Mean: {badge['mean']}")
        if badge_lines:
            badge_text = "\n".join(badge_lines)
            badge_artist = ax.text(
                0.02,
                0.98,
                badge_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize="small",
                color=self.neutral_grey("dark"),
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.4"),
            )
            self.register_annotations(ax, badge_artist)

        # Legend ordering similar to legacy presentation
        handles, labels = ax.get_legend_handles_labels()
        order = [
            "Histogram",
            "KDE",
            f"Q1 = {desc['pct_25']:.2f}",
            f"Median = {desc['pct_50']:.2f}",
        ]
        if mean_label:
            order.append(mean_label)
        order.extend(
            [
                f"Q3 = {desc['pct_75']:.2f}",
                "Bottom 10%",
                "Top 10%",
                f"{desc['modes_count']} mode(s)",
            ]
        )
        ordered = [(h, lbl) for key in order for h, lbl in zip(handles, labels, strict=True) if lbl == key]
        if ordered:
            h_ord, l_ord = zip(*ordered, strict=True)
            ax.legend(h_ord, l_ord)
        else:
            ax.legend(handles, labels)

        return fig, ax
