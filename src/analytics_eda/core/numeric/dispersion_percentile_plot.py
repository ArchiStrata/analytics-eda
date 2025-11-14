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
"""Percentile ladder plot highlighting distribution shape across ranks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

from ..visualization.base_plot import BasePlot, PlotContext


def _safe_step(value: int | float | None, default: int = 10) -> int:
    try:
        step = int(round(float(value)))
    except (TypeError, ValueError):
        step = default
    return min(50, max(1, step))


@dataclass
class DispersionPercentilePlotContext(PlotContext):
    """Context for percentile-based dispersion plots."""

    title_template: str = "Percentile Dispersion of {name}{modifiers}"
    xlabel: str = "Percentile"
    ylabel: str = "Value"
    show_subtitle: bool = True
    show_footer_summary: bool = True
    percentile_step: int = 10
    highlight_iqr_band: bool = True


class DispersionPercentilePlot(BasePlot):
    """Show percentile ladders so viewers can see spread, skew, and compression.

    Why this matters:
        Percentiles surface asymmetry, plateaus, and compressed regions that variance-
        based metrics miss, which is essential for skewed, truncated, or heavy-tailed
        series.

    What this plot does:
        Computes configurable percentile cut points, IQR, and 90–10 spread, then plots
        a ladder chart with optional IQR shading, median line, and spread annotation to
        highlight where the distribution concentrates or stretches.
    """

    def __init__(self, ctx: DispersionPercentilePlotContext):
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return semantic version for the percentile dispersion plot."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return empty descriptive payload including config params and percentiles."""
        return {
            "params": {"percentile_step": _safe_step(self.ctx.percentile_step)},
            "n": 0,
            "percentile_ranks": [],
            "percentile_values": {},
            "median": None,
            "pct_25": None,
            "pct_75": None,
            "iqr": None,
            "ninety_ten_spread": None,
            "compression_ratio": None,
            "skew_indicator": None,
            "min": None,
            "max": None,
            "range": None,
            "p10": None,
            "p90": None,
        }

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute percentile ladder metrics, IQR, spread, and skew indicators."""
        desc = self.default_descriptive()
        clean = s.dropna()
        n = int(clean.size)
        desc["n"] = n
        if n == 0:
            desc.update({"skip_plot": True, "error": "no data to display"})
            return desc

        step = desc["params"]["percentile_step"]
        percentile_ranks = list(range(step, 100, step))
        quantile_targets = [p / 100 for p in percentile_ranks]
        quantiles = clean.quantile(quantile_targets) if percentile_ranks else pd.Series([], dtype=float)
        percentile_values: dict[str, float] = {}
        for rank, value in zip(percentile_ranks, quantiles.to_list(), strict=False):
            percentile_values[f"p{rank}"] = float(value)

        extra_quantiles = clean.quantile([0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9])
        q0, q25, q50, q75, q100, q10, q90 = (float(extra_quantiles.get(q, np.nan)) for q in (0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9))

        iqr = q75 - q25 if np.isfinite(q75) and np.isfinite(q25) else np.nan
        spread_90_10 = q90 - q10 if np.isfinite(q90) and np.isfinite(q10) else np.nan
        data_range = q100 - q0 if np.isfinite(q100) and np.isfinite(q0) else np.nan

        compression_ratio = (iqr / data_range) if (np.isfinite(iqr) and np.isfinite(data_range) and data_range > 0) else None
        skew_indicator = None
        if np.isfinite(q75) and np.isfinite(q50) and np.isfinite(q25):
            upper = q75 - q50
            lower = q50 - q25
            skew_indicator = upper - lower

        desc.update(
            {
                "percentile_ranks": percentile_ranks,
                "percentile_values": percentile_values,
                "median": q50,
                "pct_25": q25,
                "pct_75": q75,
                "iqr": None if not np.isfinite(iqr) else float(iqr),
                "ninety_ten_spread": None if not np.isfinite(spread_90_10) else float(spread_90_10),
                "compression_ratio": compression_ratio,
                "skew_indicator": skew_indicator,
                "min": q0 if np.isfinite(q0) else None,
                "max": q100 if np.isfinite(q100) else None,
                "range": data_range if np.isfinite(data_range) else None,
                "p10": q10 if np.isfinite(q10) else None,
                "p90": q90 if np.isfinite(q90) else None,
            }
        )

        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize percentile climb and 90–10 spread per plot style rules."""
        if not desc:
            return {}
        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return {"context": "No non-null observations.", "primary_finding": None, "secondary_finding": None}

        step = int((desc.get("params") or {}).get("percentile_step", 10))
        formatter = self.formatter.format_numeric_value
        p10 = desc.get("p10")
        p90 = desc.get("p90")
        spread_90_10 = desc.get("ninety_ten_spread")
        context = f"n = {n:,} • step = {step}%"

        if self.is_finite(p10) and self.is_finite(p90):
            primary = f"Values climb from P10 {formatter(p10, decimals=2)} to P90 {formatter(p90, decimals=2)}, " f"with P25–P75 spanning {formatter(desc.get('iqr'), decimals=2)}."
        else:
            primary = "Percentile ladder computed, but key deciles are unavailable."

        secondary = None
        if self.is_finite(spread_90_10):
            secondary = f"P90–P10 spread is {formatter(spread_90_10, decimals=2)}, indicating {self._spread_descriptor(spread_90_10, desc.get('range'))}."
        return {"context": context, "primary_finding": primary, "secondary_finding": secondary}

    @staticmethod
    def _spread_descriptor(spread: float, total_range: float | None) -> str:
        """Classify spread magnitude relative to the overall range."""
        if not total_range or total_range <= 0:
            return "overall variability comparable to the data range"
        ratio = spread / total_range
        if ratio >= 0.8:
            return "broad dispersion across the sample"
        if ratio >= 0.5:
            return "moderate dispersion"
        return "tight mid-range clustering"

    def subtitle_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Return compact subtitle with sample size, step, and P90–P10 spread."""
        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return "No non-null observations."
        step = int((desc.get("params") or {}).get("percentile_step", 10))
        spread = desc.get("ninety_ten_spread")
        parts = [f"n = {n:,}", f"step = {step}%"]
        if self.is_finite(spread):
            parts.append(f"P90–P10 = {self.formatter.format_numeric_value(spread, decimals=2)}")
        return " • ".join(parts)

    def footer_summary_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Provide footer highlighting IQR and compression ratio."""
        formatter = self.formatter.format_numeric_value
        iqr = formatter(desc.get("iqr"), decimals=2)
        comp = desc.get("compression_ratio")
        comp_txt = f"Compression {comp:.2f}" if comp not in (None, np.nan) else "Compression NA"
        return f"IQR {iqr} • {comp_txt}"

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
        """Render percentile ladder, optional IQR shading, median line, and spread text."""
        ranks = desc.get("percentile_ranks") or []
        values_map = desc.get("percentile_values") or {}
        if not ranks:
            return fig, ax

        y_values = [values_map.get(f"p{int(r)}") for r in ranks]
        if not any(self.is_finite(v) for v in y_values):
            return fig, ax

        xs = np.array(ranks, dtype=float)
        ys = np.array([np.nan if v is None else float(v) for v in y_values], dtype=float)

        ax.plot(xs, ys, color=palette[0], linewidth=2, label="Percentile value")
        ax.scatter(xs, ys, color=palette[1], s=30, zorder=3)

        q1 = desc.get("pct_25")
        q3 = desc.get("pct_75")
        if self.ctx.highlight_iqr_band and self.is_finite(q1) and self.is_finite(q3):
            ax.axhspan(q1, q3, color=palette[0], alpha=0.12, label="IQR", zorder=1)

        median = desc.get("median")
        if self.is_finite(median):
            ax.axhline(median, color=palette[2 if len(palette) > 2 else 0], linestyle="--", linewidth=1.2, label="Median", zorder=2)

        spread = desc.get("ninety_ten_spread")
        if self.is_finite(spread):
            text = f"P90–P10 = {self.formatter.format_numeric_value(spread, decimals=2)}"
            ax.text(0.02, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize="small", bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

        ax.set_xlim(0, 100)
        ax.set_xticks(ranks)
        ax.set_xlabel(self.ctx.xlabel)
        ax.set_ylabel(self.ctx.ylabel)

        return fig, ax
