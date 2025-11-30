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
"""Decile ladder plot highlighting distribution shape across ranks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from analytics_eda.core.visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

DECILE_LABELS = tuple(f"D{i}" for i in range(1, 10))
DECILE_PERCENTILES = tuple(i * 10 for i in range(1, 10))
DECILE_TARGETS = tuple(p / 100 for p in DECILE_PERCENTILES)
DECILE_PARAMS = tuple({"label": label, "percentile": pct} for label, pct in zip(DECILE_LABELS, DECILE_PERCENTILES, strict=True))


@dataclass
class DispersionDecilePlotContext(PlotContext):
    """Context for decile-based dispersion plots."""

    title_template: str = "Decile Dispersion of {name}{modifiers}"
    xlabel: str = "Decile"
    ylabel: str = "Value"
    show_subtitle: bool = True
    show_footer_summary: bool = True
    highlight_iqr_band: bool = True


class DispersionDecilePlot(BasePlot):
    """Show D1–D9 deciles so viewers see how values climb across the distribution.

    Why this matters:
        Deciles are intuitive business language for dispersion: D1, D5, and D9 anchor
        early, middle, and high performers, making skew and compression clear without
        technical detours.

    What this plot does:
        Computes the D1–D9 ladder, quartiles, IQR, and the D9–D1 spread, then draws a
        simple line-and-dot profile with optional IQR shading, a median (D5) guide, and
        annotations that articulate where the distribution concentrates or stretches.
    """

    def __init__(self, ctx: DispersionDecilePlotContext):
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return semantic version for the decile dispersion plot."""
        return "1.1.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return empty descriptive payload including config params and deciles."""
        return {
            "params": {"deciles": DECILE_PARAMS},
            "n": 0,
            "decile_values": {},
            "percentile_ranks": [],
            "median": None,
            "pct_25": None,
            "pct_75": None,
            "iqr": None,
            "decile_spread": None,
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
        """Compute decile ladder metrics, IQR, spread, and skew indicators."""
        desc = self.default_descriptive()
        clean = s.dropna()
        n = int(clean.size)
        desc["n"] = n
        if n == 0:
            desc.update({"skip_plot": True, "error": "no data to display"})
            return desc

        percentile_ranks = list(DECILE_PERCENTILES)
        quantiles = clean.quantile(DECILE_TARGETS)
        decile_values: dict[str, float | None] = {}
        for label, value in zip(DECILE_LABELS, quantiles.to_list(), strict=False):
            val = float(value) if np.isfinite(value) else None
            decile_values[label] = val

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
                "decile_values": decile_values,
                "median": q50,
                "pct_25": q25,
                "pct_75": q75,
                "iqr": None if not np.isfinite(iqr) else float(iqr),
                "decile_spread": None if not np.isfinite(spread_90_10) else float(spread_90_10),
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
        """Summarize the D1–D9 climb and spread per plot style rules."""
        if not desc:
            return {}
        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return {"context": "No non-null observations.", "primary_finding": None, "secondary_finding": None}

        formatter = self.formatter.format_numeric_value
        deciles = desc.get("decile_values") or {}
        d1 = deciles.get("D1")
        d5 = deciles.get("D5", desc.get("median"))
        d9 = deciles.get("D9")
        spread_90_10 = desc.get("decile_spread")
        context = f"n = {n:,} • Deciles D1–D9"

        if self.is_finite(d1) and self.is_finite(d9):
            parts = [f"D1 {formatter(d1, decimals=2)} ➜ D9 {formatter(d9, decimals=2)}"]
            if self.is_finite(d5):
                parts.append(f"D5 midpoint {formatter(d5, decimals=2)}")
            if self.is_finite(desc.get("iqr")):
                parts.append(f"IQR {formatter(desc.get('iqr'), decimals=2)}")
            primary = "; ".join(parts) + "."
        else:
            primary = "Decile ladder computed, but key values are unavailable."

        secondary = None
        if self.is_finite(spread_90_10):
            secondary = f"D9–D1 span is {formatter(spread_90_10, decimals=2)}, indicating {self._spread_descriptor(spread_90_10, desc.get('range'))}."
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
        """Return compact subtitle with sample size, D1/D5/D9 anchors, and D9–D1 span."""
        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return "No non-null observations."
        spread = desc.get("decile_spread")
        deciles = desc.get("decile_values") or {}
        fmt = self.formatter.format_numeric_value
        parts = [f"n = {n:,}"]
        d1 = deciles.get("D1")
        d5 = deciles.get("D5", desc.get("median"))
        d9 = deciles.get("D9")
        if self.is_finite(d1) and self.is_finite(d9):
            parts.append(f"D1 {fmt(d1, decimals=2)} – D9 {fmt(d9, decimals=2)}")
        if self.is_finite(d5):
            parts.append(f"D5 {fmt(d5, decimals=2)}")
        if self.is_finite(spread):
            parts.append(f"D9–D1 = {fmt(spread, decimals=2)}")
        return " • ".join(parts)

    def footer_summary_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Provide footer highlighting decile span, IQR, and compression ratio."""
        formatter = self.formatter.format_numeric_value
        iqr = formatter(desc.get("iqr"), decimals=2)
        spread = formatter(desc.get("decile_spread"), decimals=2)
        comp = desc.get("compression_ratio")
        comp_txt = f"Compression {comp:.2f}" if comp not in (None, np.nan) else "Compression NA"
        return f"D9–D1 {spread} • IQR {iqr} • {comp_txt}"

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
        """Render decile ladder, optional IQR shading, median line, and spread text."""
        labels = list(DECILE_LABELS)
        values_map = desc.get("decile_values") or {}
        y_values = [values_map.get(label) for label in labels]
        if not any(self.is_finite(v) for v in y_values):
            return fig, ax

        xs = np.arange(1, len(labels) + 1, dtype=float)
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

        spread = desc.get("decile_spread")
        if self.is_finite(spread):
            text = f"D9–D1 = {self.formatter.format_numeric_value(spread, decimals=2)}"
            ax.text(0.02, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize="small", bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

        ax.set_xlim(0.5, len(labels) + 0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)

        return fig, ax
