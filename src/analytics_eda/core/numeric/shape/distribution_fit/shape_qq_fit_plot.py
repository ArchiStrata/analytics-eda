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
"""QQ fit plot with diagnostics for a chosen theoretical distribution.

Why this matters:
    Shows how closely a sample follows a chosen distribution and surfaces
    departures with both visual alignment and goodness-of-fit diagnostics.

What this plot does:
    Fits one of norm/lognorm/gamma/expon, draws sample vs. theoretical
    quantiles, overlays the fitted line and a 45-degree reference, summarizes
    residuals/shape, and runs normality tests when applicable.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from analytics_eda.core.visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

DistributionName = Literal["norm", "lognorm", "gamma", "expon"]


@dataclass
class ShapeQqFitContext(PlotContext):
    """Context options for the QQ fit plot (labels, legend, and distribution)."""

    title_template: str = "QQ Plot Fit Assessment of {name}{modifiers}"
    xlabel: str = "Theoretical Quantiles"
    ylabel: str = "Sample Quantiles"
    enable_legend: bool = True

    # plot-specific
    distribution_name: DistributionName = "norm"
    alpha: float = 0.05


class ShapeQqFitPlot(BasePlot):
    """Generate a QQ plot assessing fit to a specified distribution.

    Why this matters:
        Quantifies how well a numeric variable matches a theoretical
        distribution and highlights departures from that fit.

    What this plot does:
        - Plots sample quantiles vs. theoretical quantiles for the chosen distribution.
        - Shows the fitted line, 45-degree reference, and residual diagnostics.
        - Summarizes shape (skewness, excess kurtosis) and normality tests when applicable.
    """

    def __init__(self, ctx):
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    def title_kwargs(self, *, series=None, cols=None, role_map=None) -> dict[str, Any]:
        """Provide template kwargs for the title (e.g., distribution name and extras)."""
        dist = self.ctx.distribution_name
        return {
            "fit_desc": f"fitted to {dist}",
            "dist": dist,
        }

    def metadata_overrides(self, *, series=None, cols=None, role_map=None) -> dict[str, Any]:
        """Inject extra fields into the returned chart metadata (e.g., alpha, dist name)."""
        return {
            "distribution_name": self.ctx.distribution_name,
            "alpha": float(self.ctx.alpha),
        }

    def default_descriptive(self) -> dict[str, Any]:
        """Return the empty/none defaults for descriptive statistics."""
        return {
            "n": 0,
            "params": {"distribution_name": self.ctx.distribution_name, "distribution_fit": None},
            "intercept": None,
            "slope": None,
            "r_squared": None,
            "median_residual": None,
            "iqr_residual": None,
            "max_abs_residual": None,
            "skewness": None,
            "kurtosis": None,
            "min": None,
            "reference_line": True,
        }

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute QQ line fit, residual summaries, shape stats, and cache draw payload."""
        data = s.dropna().astype(float)
        n = int(data.size)

        desc = self.default_descriptive()
        desc["n"] = n
        if n == 0:
            return desc

        dist_name = self.ctx.distribution_name
        allowed = ("norm", "lognorm", "gamma", "expon")
        if dist_name not in allowed:
            raise ValueError(f"distribution_name must be one of {allowed}")

        # Support checks for domain
        mn = float(data.min())
        desc["min"] = mn
        if dist_name in ("lognorm", "gamma") and mn <= 0:
            desc["error"] = "requires positive data"
            desc["skip_plot"] = True
            return desc
        if dist_name == "expon" and mn < 0:
            desc["error"] = "requires non-negative data"
            desc["skip_plot"] = True
            return desc

        # Fit distribution
        dist = getattr(stats, dist_name)
        params = dist.fit(data)
        *shape_args, loc, scale = params

        # Theoretical quantiles (plotting positions)
        probs = (np.arange(1, n + 1) - 0.5) / n
        osm = dist.ppf(probs, *shape_args, loc=loc, scale=scale)
        osr = np.sort(data.to_numpy())

        # Linear fit osr ~ a + b * osm
        slope, intercept = np.polyfit(osm, osr, 1)
        fitted = intercept + slope * osm

        # R^2 via correlation
        corr = np.corrcoef(osr, fitted)[0, 1]
        r_squared = float(corr**2)

        residuals = osr - fitted
        median_residual = float(np.median(residuals))
        iqr_residual = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
        max_abs_residual = float(np.max(np.abs(residuals)))

        skewness = float(stats.skew(data, bias=False))
        kurtosis = float(stats.kurtosis(data, fisher=True, bias=False))

        # Cache render-only payload for draw()
        self.draw_cache_set("qq", "osm", osm)
        self.draw_cache_set("qq", "osr", osr)
        self.draw_cache_set("qq", "fitted", fitted)

        desc.update(
            {
                "params": {"distribution_name": dist_name, "distribution_fit": tuple(float(np.round(p, 3)) for p in params)},
                "intercept": float(intercept),
                "slope": float(slope),
                "r_squared": r_squared,
                "median_residual": median_residual,
                "iqr_residual": iqr_residual,
                "max_abs_residual": max_abs_residual,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "reference_line": True,
            }
        )

        return desc

    def default_inferential(self) -> dict[str, Any]:
        """Return default inferential payload with parameters (alpha, distribution)."""
        return {"params": {"alpha": self.ctx.alpha, "distribution_name": self.ctx.distribution_name}}

    def compute_inferential(self, s: pd.Series, desc: dict[str, Any]) -> dict[str, Any]:
        """Run normality tests when `distribution_name == 'norm'` and build results."""
        data = s.dropna().astype(float)
        n = int(data.size)
        res: dict[str, Any] = {"params": {"alpha": float(self.ctx.alpha), "distribution_name": self.ctx.distribution_name}}
        if n == 0:
            return res

        if "error" in desc:
            # domain errors - no tests possible
            return res

        if self.ctx.distribution_name == "norm":
            alpha = float(self.ctx.alpha)

            # Shapiro-Wilk for n < 50
            if n < 50:
                stat_sw, p_sw = stats.shapiro(data)
                res["shapiro"] = {"statistic": float(stat_sw), "p_value": float(p_sw), "reject": bool(p_sw < alpha)}

            # D'Agostino-Pearson omnibus for n >= 20
            if n >= 20:
                stat_dp, p_dp = stats.normaltest(data)
                res["dagostino_pearson"] = {"statistic": float(stat_dp), "p_value": float(p_dp), "reject": bool(p_dp < alpha)}

            # Jarque-Bera for n > 2000
            if n > 2000:
                stat_jb, p_jb = stats.jarque_bera(data)
                res["jarque_bera"] = {"statistic": float(stat_jb), "p_value": float(p_jb), "reject": bool(p_jb < alpha)}

            # Overall reject flag
            res["reject_normality"] = any(v.get("reject", False) for k, v in res.items() if isinstance(v, dict))

        return res

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize sample size and fit strength from descriptive stats."""
        if not desc:
            return {}

        n = int(desc.get("n", 0) or 0)
        dist_name = desc.get("params", {}).get("distribution_name", self.ctx.distribution_name)
        if n == 0:
            return {"context": f"n = 0 | distribution = {dist_name}", "primary_finding": None, "secondary_finding": None}
        if desc.get("error"):
            return {"context": f"n = {n} | distribution = {dist_name}", "primary_finding": None, "secondary_finding": None}

        fmt = self.formatter.format_numeric_value
        r2 = desc.get("r_squared")
        slope = desc.get("slope")
        med_resid = desc.get("median_residual")
        primary = f"Sample quantiles align to {dist_name} with R^2 = {fmt(r2, decimals=3)} and slope = {fmt(slope, decimals=3)} over n = {n}."

        iqr_resid = desc.get("iqr_residual")
        max_abs = desc.get("max_abs_residual")
        secondary_parts = []
        if self.is_finite(iqr_resid):
            secondary_parts.append(f"IQR residual = {fmt(iqr_resid, decimals=3)}")
        if self.is_finite(max_abs):
            secondary_parts.append(f"max |residual| = {fmt(max_abs, decimals=3)}")
        if self.is_finite(med_resid):
            secondary_parts.append(f"median residual = {fmt(med_resid, decimals=3)}")
        secondary = "; ".join(secondary_parts) if secondary_parts else None

        return {
            "context": f"n = {n} | distribution = {dist_name}",
            "primary_finding": primary,
            "secondary_finding": secondary,
        }

    def draft_inferential_findings(self, inf: dict[str, Any], desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize normality test outcomes."""
        if not inf:
            return {}

        n = int(desc.get("n", 0) or 0)
        dist_name = desc.get("params", {}).get("distribution_name", self.ctx.distribution_name)
        alpha = float(inf.get("params", {}).get("alpha", self.ctx.alpha))
        if n == 0 or desc.get("error") or dist_name != "norm":
            return {"context": f"n = {n} | distribution = {dist_name} | alpha = {alpha}", "primary_finding": None, "secondary_finding": None}

        fmt = self.formatter.format_numeric_value

        def _decision(test: dict[str, Any] | None) -> str:
            if not test:
                return ""
            return "rejects" if test.get("reject") else "fails to reject"

        primary_parts = []
        if "shapiro" in inf and n < 50:
            sw = inf["shapiro"]
            primary_parts.append(f"Shapiro-Wilk {_decision(sw)} normality (stat = {fmt(sw.get('statistic'), decimals=3)}, p = {fmt(sw.get('p_value'), decimals=3)}).")
        if "dagostino_pearson" in inf and n >= 20:
            dp = inf["dagostino_pearson"]
            primary_parts.append(
                f"D'Agostino-Pearson {_decision(dp)} (stat = {fmt(dp.get('statistic'), decimals=3)}, p = {fmt(dp.get('p_value'), decimals=3)})."
            )
        if "jarque_bera" in inf and n > 2000:
            jb = inf["jarque_bera"]
            primary_parts.append(f"Jarque-Bera {_decision(jb)} (stat = {fmt(jb.get('statistic'), decimals=3)}, p = {fmt(jb.get('p_value'), decimals=3)}).")

        primary = " ".join(primary_parts) or None
        secondary = None
        if "reject_normality" in inf:
            secondary = f"Overall reject flag: {inf['reject_normality']}."

        return {
            "context": f"n = {n} | distribution = {dist_name} | alpha = {fmt(alpha, decimals=3)}",
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
        """Render the QQ scatter, fit line, reference line, and a compact stats textbox."""
        # If domain error (e.g., lognorm with nonpositive), just render title/labels and note error
        if "error" in desc:
            err_txt = ax.text(
                0.5,
                0.5,
                f"Data domain error: {desc['error']}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
                color=self.neutral_grey("dark"),
            )
            self.register_annotations(ax, err_txt)
            return fig, ax

        osm = self.draw_cache_get("qq", "osm")
        osr = self.draw_cache_get("qq", "osr")
        fitted = self.draw_cache_get("qq", "fitted")

        if osm is None or osr is None or fitted is None:
            desc["skip_plot"] = True
            return fig, ax

        # reference line for perfect fit
        ref_color = self.neutral_grey("light")
        ax.plot(osm, osm, color=ref_color, lw=1, linestyle="--", label="45-degree reference")

        # points + fit line
        point_color = palette[0] if len(palette) > 0 else None
        fit_color = palette[1] if len(palette) > 1 else "red"
        sns.scatterplot(x=osm, y=osr, ax=ax, s=20, edgecolor="k", alpha=0.6, label="Quantiles", color=point_color)
        ax.plot(osm, fitted, color=fit_color, lw=1.5, label="Fitted line")

        # Stats textbox (left-top)
        lines = [
            f"n = {desc.get('n', 0)}",
            f"intercept: {desc['intercept']:.2f}",
            f"slope: {desc['slope']:.2f}",
            f"R^2: {desc['r_squared']:.3f}",
            f"median resid: {desc['median_residual']:.2f}",
            f"IQR resid: {desc['iqr_residual']:.2f}",
            f"max |resid|: {desc['max_abs_residual']:.2f}",
            f"skewness: {desc['skewness']:.2f}",
            f"excess kurtosis: {desc['kurtosis']:.2f}",
        ]
        if self.ctx.distribution_name == "norm" and inf:
            lines.append("")  # spacer
            for k, v in inf.items():
                if k == "params":
                    continue
                if k == "reject_normality":
                    lines.append(f"Overall reject: {v}")
                else:
                    p = v.get("p_value", np.nan)
                    lines.append(f"{k}: stat={v['statistic']:.3f}, p={p:.3f}, reject={v['reject']}")

        stats_txt = ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        self.register_annotations(ax, stats_txt)

        # Keep axes aligned to emphasize deviations
        ax.set_aspect("equal", adjustable="box")

        return fig, ax
