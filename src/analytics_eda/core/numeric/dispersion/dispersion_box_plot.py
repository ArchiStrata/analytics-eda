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
"""Box/violin dispersion plot with quartiles and IQR-based outliers."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator


@dataclass
class DispersionBoxPlotContext(PlotContext):
    """Context options for the quartile-focused dispersion plot."""

    title_template: str = "Dispersion of {name}{modifiers} (IQR & Outliers)"
    xlabel: str = ""
    ylabel: str = "Value"
    show_subtitle: bool = True
    show_footer_summary: bool = True
    show_violin_silhouette: bool = True


class DispersionBoxPlot(BasePlot):
    """Shows how values distribute via quartiles, whiskers, and IQR outliers.

    Why this matters:
    Quartiles and the interquartile range offer an intuitive, defensible view of
    variability and asymmetry without assuming normality, making outlier calls
    easy to explain to stakeholders.

    What this plot does:
    Accepts a numeric Series, optionally overlays a subtle violin silhouette,
    renders a box plot (median, quartiles, whiskers), flags observations beyond
    classical IQR fences, and returns descriptive stats centered on the quartile
    story (n, mean, quartiles, IQR, fences, min, max, range (R)).
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
            "n": 0,
            "mean": None,
            "min": None,
            "max": None,
            "range": None,
            "pct_25": None,
            "median": None,
            "pct_75": None,
            "iqr": None,
            "iqr_lower_bound": None,
            "iqr_upper_bound": None,
            "outlier_lower_count": 0,
            "outlier_upper_count": 0,
        }

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute quartile stats, IQR fences, and counts used for drawing."""
        n = int(s.size)
        if n == 0:
            return {**self.default_descriptive(), "n": 0, "skip_plot": True, "error": "no data to display"}

        mean = float(s.mean())
        min_val = float(s.min())
        max_val = float(s.max())
        range_val = float(max_val - min_val)

        q1 = float(s.quantile(0.25))
        median = float(s.quantile(0.5))
        q3 = float(s.quantile(0.75))
        iqr = float(q3 - q1)

        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        lower_count = int((s < iqr_lower).sum())
        upper_count = int((s > iqr_upper).sum())

        desc = self.default_descriptive()
        desc.update(
            {
                "n": n,
                "mean": mean,
                "min": min_val,
                "max": max_val,
                "range": range_val,
                "pct_25": q1,
                "median": median,
                "pct_75": q3,
                "iqr": iqr,
                "iqr_lower_bound": iqr_lower,
                "iqr_upper_bound": iqr_upper,
                "outlier_lower_count": lower_count,
                "outlier_upper_count": upper_count,
            }
        )

        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize quartile spread, median, and any IQR-based outliers."""
        if not desc:
            return {}

        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return {
                "context": "No non-null observations.",
                "primary_finding": None,
                "secondary_finding": None,
            }

        fmt = self.formatter.format_numeric_value
        q1 = desc.get("pct_25")
        median = desc.get("median")
        q3 = desc.get("pct_75")
        iqr = desc.get("iqr")
        min_val = desc.get("min")
        max_val = desc.get("max")

        context_parts = [f"n = {n:,}"]
        if self.is_finite(q1) and self.is_finite(q3):
            context_parts.append(f"Q1 {fmt(q1, decimals=2)} • Q3 {fmt(q3, decimals=2)}")
        context = " • ".join(context_parts)

        if self.is_finite(median) and self.is_finite(q1) and self.is_finite(q3) and self.is_finite(iqr):
            primary = f"Median {fmt(median, decimals=2)} with middle 50% spanning " f"{fmt(q1, decimals=2)}–{fmt(q3, decimals=2)} (IQR {fmt(iqr, decimals=2)})."
        else:
            primary = "Quartile spread cannot be summarized with the available stats."

        lower_out = int(desc.get("outlier_lower_count", 0) or 0)
        upper_out = int(desc.get("outlier_upper_count", 0) or 0)
        total_out = lower_out + upper_out
        secondary: str | None = None
        if total_out > 0:
            secondary = f"{total_out:,} observations sit outside the IQR fences ({lower_out:,} low / {upper_out:,} high)."
        elif self.is_finite(min_val) and self.is_finite(max_val):
            secondary = f"Overall range extends from {fmt(min_val, decimals=2)} to {fmt(max_val, decimals=2)}."

        return {
            "context": context,
            "primary_finding": primary,
            "secondary_finding": secondary,
        }

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Return a compact subtitle with n, IQR span, and median when available."""
        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return "No non-null observations."
        d = int(desc.get("round_decimals", getattr(self.formatter, "report_default_decimals", 2)))
        parts = [f"n = {n:,}"]
        if self.is_finite(desc.get("iqr")) and self.is_finite(desc.get("pct_25")) and self.is_finite(desc.get("pct_75")):
            q1 = self.formatter.format_numeric_value(desc["pct_25"], decimals=d)
            q3 = self.formatter.format_numeric_value(desc["pct_75"], decimals=d)
            parts.append(f"IQR span: {q1}–{q3}")
        median = desc.get("median")
        if self.is_finite(median):
            parts.append(f"Median {self.formatter.format_numeric_value(median, decimals=d)}")
        return " • ".join(parts)

    def footer_summary_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Return a compact footer summary string (min/IQR/max)."""
        fmt = self.formatter.format_numeric_value
        min_txt = fmt(desc.get("min"), decimals=2)
        max_txt = fmt(desc.get("max"), decimals=2)
        iqr_txt = fmt(desc.get("iqr"), decimals=2)
        return f"Min {min_txt} • IQR {iqr_txt} • Max {max_txt}"

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
        """Render violin silhouette (optional), boxplot, and IQR-based outliers."""
        series = s.dropna()
        if series.empty:
            return fig, ax

        data = series.to_numpy()

        if self.ctx.show_violin_silhouette and len(data) > 1:
            parts = ax.violinplot(
                data,
                vert=True,
                positions=[0],
                widths=0.8,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(palette[0])
                pc.set_edgecolor(palette[0])
                pc.set_alpha(0.12)
                pc.set_linewidth(0.6)
                pc.set_zorder(1)

        ax.boxplot(
            data,
            positions=[0],
            widths=0.4,
            notch=False,
            patch_artist=True,
            showcaps=True,
            boxprops=dict(facecolor="white", linewidth=1.3),
            whiskerprops=dict(linewidth=1.1),
            medianprops=dict(linewidth=1.6, color=palette[1]),
            flierprops=dict(marker="o", markerfacecolor=palette[2], markeredgecolor="white", markersize=5),
            zorder=2,
        )

        iqr_lower = desc.get("iqr_lower_bound")
        iqr_upper = desc.get("iqr_upper_bound")
        lower_mask = self.is_finite(iqr_lower) and (series < iqr_lower)
        upper_mask = self.is_finite(iqr_upper) and (series > iqr_upper)

        def scatter_outliers(values, color, label):
            if values.shape[0]:
                ax.scatter(
                    [0] * values.shape[0],
                    values,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    marker="X",
                    s=55,
                    zorder=3,
                    label=label,
                )

        if isinstance(lower_mask, pd.Series) and lower_mask.any():
            scatter_outliers(series[lower_mask], palette[3], "Lower IQR outlier")
        if isinstance(upper_mask, pd.Series) and upper_mask.any():
            scatter_outliers(series[upper_mask], palette[4] if len(palette) > 4 else palette[3], "Upper IQR outlier")

        q1 = desc.get("pct_25")
        median_val = desc.get("median")
        q3 = desc.get("pct_75")
        text = (
            f"Q1 = {self.formatter.format_numeric_value(q1, decimals=2)}\n"
            f"Median = {self.formatter.format_numeric_value(median_val, decimals=2)}\n"
            f"Q3 = {self.formatter.format_numeric_value(q3, decimals=2)}\n"
            f"IQR = {self.formatter.format_numeric_value(desc.get('iqr'), decimals=2)}\n"
            f"Fences = [{self.formatter.format_numeric_value(iqr_lower, decimals=2)}, "
            f"{self.formatter.format_numeric_value(iqr_upper, decimals=2)}]"
        )
        ax.text(
            0.95,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize="small",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.55),
        )

        ax.set_xticks([])
        ax.set_ylabel(self.ctx.ylabel or "Value")

        return fig, ax
