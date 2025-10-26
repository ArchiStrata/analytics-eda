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
"""Pareto (80/20) chart for categorical frequencies in a Series."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from analytics_eda.core.visualization.context.plot_context import AxisFormat
from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import (
    SeriesBarChartContext,
    SeriesBarChartMixin,
)
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import categorical_validator

from ..visualization.base_plot import BasePlot


@dataclass
class FrequencyParetoContext(SeriesBarChartContext):
    """Context options controlling the frequency Pareto chart."""

    title_template: str = "Pareto Chart of {name}{modifiers}"
    xlabel: str = "Share of total"
    ylabel: str = "Category"
    is_orientation_vertical: bool = False
    show_subtitle: bool = True

    x_format: AxisFormat = field(
        default_factory=lambda: AxisFormat(kind="percent", decimals=1, percent_scale_0to1=True)
    )
    y_format: AxisFormat = field(
        default_factory=lambda: AxisFormat(kind="category")
    )

    bar_sort_descending: bool = True

    pareto_mode: Literal["dual", "shared", "none"] = "shared"
    pareto_threshold_pct: float = 0.8
    pareto_line_color: str = "black"
    pareto_line_marker: str = "o"
    pareto_line_style: str = "-"
    show_threshold_label: bool = True


class FrequencyParetoPlot(SeriesBarChartMixin, BasePlot):
    """
    Shows how a small set of categories accounts for most occurrences (Pareto concentration).

    Why this matters:
    - Identifies the vital few categories that dominate volume, guiding prioritization and focus.

    What this plot does:
    - Ranks categories by share of total (optionally collapsing small ones into “Other”).
    - Draws bars for category shares and a cumulative line.
    - Highlights how many categories are needed to reach a chosen threshold (e.g., 80%).
    """

    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=categorical_validator()
        )
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute counts, shares, cumulative shares, and threshold summary.

        Builds the bar payload (percent of total), caches arrays for the Pareto
        line, and returns a `desc` dict with Pareto-specific fields.
        """
        counts = s.value_counts()

        # Build standard series-bar desc (percent base is pct_of_total for Pareto)
        extra_params = {
            "pareto_mode": getattr(self.ctx, "pareto_mode", "shared"),
            "pareto_threshold_pct": float(getattr(self.ctx, "pareto_threshold_pct", 0.8)),
            "show_threshold_label": bool(getattr(self.ctx, "show_threshold_label", True)),
            "pareto_line_color": getattr(self.ctx, "pareto_line_color", "black"),
            "pareto_line_marker": getattr(self.ctx, "pareto_line_marker", "o"),
            "pareto_line_style": getattr(self.ctx, "pareto_line_style", "-"),
        }

        desc = self.build_series_bar_desc(
            s,
            counts.to_dict(),
            denominator_key="pct_of_total",
            extra_params=extra_params
        )

        if getattr(self.ctx, "pareto_mode", "shared") == "shared" and self.ctx.bar_height_source != "values":
            desc["error"] = "shared mode requires bar_height_source='values'"
            desc["skip_plot"] = True

        # Derive Pareto arrays in the plotted order
        labels  = self.draw_cache_get("series", "labels") or []
        bars    = desc.get("bars", {})
        denom_k = desc.get("denominator_key", "pct_of_total")

        rel = np.array([float(bars[lbl].get(denom_k, 0.0)) for lbl in labels], dtype=float)
        cum = np.cumsum(rel)

        thr_pct = float(getattr(self.ctx, "pareto_threshold_pct", 0.8))
        thr_idx = int(np.argmax(cum >= thr_pct)) if len(cum) else -1
        thr_count = int(sum(int(bars[lbl]["count"]) for lbl in labels[: max(thr_idx, -1) + 1])) if thr_idx >= 0 else 0

        # Cache for draw
        self.draw_cache_set("pareto", "cumperc", cum)
        self.draw_cache_set("pareto", "relpct", rel)
        self.draw_cache_set("pareto", "threshold_idx", thr_idx)
        self.draw_cache_set("pareto", "threshold_pct", thr_pct)
        self.draw_cache_set("pareto", "threshold_count", thr_count)

        # Extend desc with Pareto-specific summaries
        desc.update({
            "cumulative_count_at_threshold": thr_count,
            "threshold_pct": thr_pct,
            "threshold_idx": thr_idx,
        })
        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Return a concise narrative highlighting the Pareto concentration.

        Summarizes how many categories reach the threshold (e.g., 80%) and
        calls out the top category/ies (tie-aware).
        """
        total = int(desc.get("total", 0))
        k = int(desc.get("input_categories", 0))
        if total == 0 or k == 0:
            return {}

        findings = {
            "context": f"N = {total:,} values across {k} categories",
            "secondary_finding": None,
        }

        # Pareto threshold summary
        thr_pct = float(desc.get("threshold_pct", 0.8))
        thr_idx = int(desc.get("threshold_idx", -1))
        if thr_idx < 0:
            findings["primary_finding"] = "Distribution is too sparse to summarize with a Pareto threshold."
            return findings

        labels = self.draw_cache_get("series", "labels") or list((desc.get("bars") or {}).keys())
        bars   = desc.get("bars", {})
        denom_k = desc.get("denominator_key", "pct_of_total")

        # Determine whether the displayed 'Other' bar is inside the threshold cut
        params = desc.get("params") or {}
        other_label = params.get("other_display")
        threshold_slice = labels[: (thr_idx + 1)]
        other_in_cut = bool(other_label and other_label in threshold_slice)

        # Primary: acknowledge Other when it’s part of the cut
        thr_pct_formatted = self.formatter.format_percent(thr_pct, decimals=0)
        if other_in_cut:
            non_other = [lbl for lbl in threshold_slice if lbl != other_label]
            if len(non_other) == 0:
                primary = f"≈{thr_pct_formatted} of occurrences are covered by {other_label}."
            elif len(non_other) == 1:
                primary = f"≈{thr_pct_formatted} of occurrences are covered by {repr(non_other[0])} and {other_label}."
            elif len(non_other) == 2:
                primary = f"≈{thr_pct_formatted} of occurrences are covered by {repr(non_other[0])}, {repr(non_other[1])}, and {other_label}."
            else:
                primary = f"≈{thr_pct_formatted} of occurrences are covered by {len(non_other)} named categories plus {other_label}."
            findings["primary_finding"] = primary
        else:
            findings["primary_finding"] = (
                f"≈{thr_pct_formatted} of occurrences are concentrated in the top {thr_idx + 1} categories."
            )

        # Secondary: top category(ies) + (if applicable) Other’s share inside the cut
        shares = [(lbl, float(bars.get(lbl, {}).get(denom_k, 0.0))) for lbl in labels if lbl in bars]
        if not shares:
            return findings

        max_share = max(v for _, v in shares)
        eps = max(1e-12, 1e-6 * max_share)
        tied = [lbl for lbl, v in shares if abs(v - max_share) <= eps]
        top_pct = self.formatter.format_percent(max_share)

        parts = []
        if len(tied) == 1:
            parts.append(f"Top category: {repr(tied[0])} at {top_pct}.")
        else:
            preview = ", ".join(repr(x) for x in tied[:3])
            more = f" +{len(tied) - 3} more" if len(tied) > 3 else ""
            parts.append(f"Top categories (tie at {top_pct}): {preview}{more}.")

        if other_in_cut and other_label in bars:
            other_pct = self.formatter.format_percent(float(bars[other_label].get(denom_k, 0.0)))
            k_agg = bars[other_label].get("k_agg")
            if isinstance(k_agg, int) and k_agg > 0:
                parts.append(f"{other_label} contributes {other_pct} within the threshold.")
            else:
                parts.append(f"{other_label} contributes {other_pct} within the threshold.")

        findings["secondary_finding"] = " ".join(parts) if parts else None
        return findings

    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):
        """Render bars (via mixin) and the Pareto cumulative line."""
        # 1) Bars via the common mixin
        fig, ax = SeriesBarChartMixin.draw(self, s, desc, inf, chart_metadata, fig=fig, ax=ax, palette=palette)

        # 2) Pareto cumulative line
        mode   = getattr(self.ctx, "pareto_mode", "dual")
        cum    = self.draw_cache_get("pareto", "cumperc", np.array([]))
        thr_pct = float(self.draw_cache_get("pareto", "threshold_pct", 0.8))
        show_thr_label = bool(getattr(self.ctx, "show_threshold_label", True))

        labels = self.draw_cache_get("series", "labels") or []
        ticks = np.arange(len(labels))

        if mode == "none" or len(labels) == 0:
            return fig, ax

        lc   = getattr(self.ctx, "pareto_line_color", "black")
        mk   = getattr(self.ctx, "pareto_line_marker", "o")
        ls   = getattr(self.ctx, "pareto_line_style", "-")

        horiz = not bool(getattr(self.ctx, "is_orientation_vertical", True))
        if horiz:
            # Horizontal bars: value axis is X → put cumulative on top axis for "dual"
            if mode == "shared":
                cum01 = cum
                thr01 = thr_pct
                ax.plot(cum01, ticks, marker=mk, linestyle=ls, color=lc)
                ax.set_xlim(0, max(1.0, float(np.nanmax(cum01)) * 1.05))
                ax.axvline(thr01, color=palette[0], linestyle="--")
                if show_thr_label and len(ticks) > 0:
                    ax.text(thr01, ticks[-1], f"{thr_pct:.0%} threshold", ha="left", va="top", color=palette[0])
            else:
                ax2 = ax.twiny()
                ax2.plot(cum, ticks, marker=mk, linestyle=ls, color=lc)
                ax2.set_xlabel("Cumulative %")
                ax2.set_xlim(0, max(100.0, np.nanmax(cum) * 1.05))
                ax2.axvline(thr_pct, color=palette[0], linestyle="--")
                if show_thr_label and len(ticks) > 0:
                    ax2.text(thr_pct, ticks[-1], f"{thr_pct:.0%}", ha="left", va="top", color=palette[0])
        else:
            # Vertical bars: value axis is Y → put cumulative on right axis for "dual"
            if mode == "shared":
                cum01 = cum
                thr01 = thr_pct
                ax.plot(ticks, cum01, marker=mk, linestyle=ls, color=lc)
                ax.set_ylim(0, max(1.0, float(np.nanmax(cum01)) * 1.05))
                ax.axhline(thr01, color=palette[0], linestyle="--")
                if show_thr_label and len(ticks) > 0:
                    ax.text(ticks[-1], thr01, f"{thr_pct:.0%}", ha="right", va="bottom", color=palette[0])
            else:
                ax2 = ax.twinx()
                ax2.plot(ticks, cum, marker=mk, linestyle=ls, color=lc)
                ax2.set_ylabel("Cumulative %")
                ax2.set_ylim(0, max(100.0, np.nanmax(cum) * 1.05))
                ax2.axhline(thr_pct, color=palette[0], linestyle="--")
                if show_thr_label and len(ticks) > 0:
                    ax2.text(ticks[-1], thr_pct, f"{thr_pct:.0%}", ha="right", va="bottom", color=palette[0])

        return fig, ax

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Return a concise subtitle based on the Pareto summary."""
        if not desc or desc.get("total", 0) == 0 or desc.get("total_nonnull", 0) == 0:
            return ""

        findings = self.draft_descriptive_findings(desc) or {}
        primary = findings.get("primary_finding")
        context = findings.get("context")

        if not primary:
            return ""

        # Subtitle balances context + main message in one clear line
        if context:
            return f"{context} — {primary}"
        else:
            return primary
