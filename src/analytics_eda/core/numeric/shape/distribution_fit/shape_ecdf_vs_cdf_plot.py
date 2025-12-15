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
"""ECDF vs. fitted CDF plot with goodness-of-fit tests."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from analytics_eda.core.visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

DistName = Literal["norm", "lognorm", "gamma", "expon"]


@dataclass
class ShapeECDFvsCDFContext(PlotContext):
    """Context for ECDF vs. theoretical CDF plots.

    Includes labels, legend flag, and test/fit parameters such as the target
    distribution name and alpha level.
    """

    title_template: str = "ECDF vs. Theoretical CDF of {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = "CDF"
    enable_legend: bool = True

    # plot-specific
    distribution_name: DistName = "norm"
    alpha: float = 0.05


class ShapeECDFvsCDFPlot(BasePlot):
    """
    Compare an empirical CDF to a fitted theoretical CDF and quantify the fit.

    Why this matters:
    Shows how closely a sample follows a chosen distribution and surfaces
    departures with formal goodness-of-fit tests (KS, AD, CvM).

    What this plot does:
    Fits one of norm/lognorm/gamma/expon, overlays ECDF vs. fitted CDF, marks
    the Kolmogorov-Smirnov max gap, summarizes fit parameters, and reports test
    decisions at the chosen alpha.
    """

    def __init__(self, ctx):
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    ALLOWED: tuple[DistName, ...] = ("norm", "lognorm", "gamma", "expon")

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    def title_kwargs(self, *, series=None, cols=None, role_map=None) -> dict[str, Any]:
        """Return placeholders used by the title template (e.g., fit description)."""
        dist = self.ctx.distribution_name
        return {
            "fit_desc": f"fitted to {dist}",
            "dist": dist,
        }

    def metadata_overrides(self, *, series=None, cols=None, role_map=None) -> dict[str, Any]:
        """Return metadata overrides derived from context (name, alpha)."""
        return {
            "distribution_name": self.ctx.distribution_name,
            "alpha": float(self.ctx.alpha),
        }

    def default_descriptive(self) -> dict[str, Any]:
        """Return an empty descriptive payload with default params."""
        return {
            "n": 0,
            "params": {
                "distribution_fit": None,
                "distribution_name": self.ctx.distribution_name,
            },
            "ks_D": np.nan,
        }

    def default_inferential(self) -> dict[str, Any]:
        """Return default inferential payload containing alpha."""
        return {"params": {"alpha": float(self.ctx.alpha)}}

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute ECDF, fitted CDF, params, and KS gap; handle domain errors."""
        if self.ctx.distribution_name not in self.ALLOWED:
            raise ValueError(f"distribution_name must be one of {self.ALLOWED}")

        data = s.dropna().astype(float)
        n = int(data.size)

        # support checks (mirror legacy behavior)
        mn = float(data.min()) if n else float("inf")
        err: str | None = None
        if self.ctx.distribution_name in ("lognorm", "gamma") and n > 0 and mn <= 0:
            err = "requires positive data"
        if self.ctx.distribution_name == "expon" and n > 0 and mn < 0:
            err = "requires non-negative data"

        desc: dict[str, Any] = {
            "n": n,
            "params": {"distribution_name": self.ctx.distribution_name},
        }

        if n == 0:
            desc["params"]["distribution_fit"] = None
            desc["ks_D"] = np.nan
            return desc

        if err is not None:
            desc["params"]["distribution_fit"] = None
            desc["error"] = err
            desc["skip_plot"] = True
            desc["ks_D"] = np.nan
            return desc

        # Fit distribution
        dist = getattr(stats, self.ctx.distribution_name)
        fit_params = dist.fit(data)
        fit_params_float = tuple(float(np.round(p, 3)) for p in fit_params)
        desc["params"]["distribution_fit"] = fit_params_float

        # ECDF
        x = np.sort(data.to_numpy())
        ecdf = np.arange(1, n + 1) / n

        # Theoretical CDF
        cdf_theo = dist.cdf(x, *fit_params)

        ks_gap = np.abs(ecdf - cdf_theo)
        ks_idx = int(np.argmax(ks_gap))
        ks_D = float(np.max(ks_gap))
        desc["ks_D"] = ks_D

        # Cache render-only payload for draw()
        self.draw_cache_set("ecdf_cdf", "x", x)
        self.draw_cache_set("ecdf_cdf", "ecdf", ecdf)
        self.draw_cache_set("ecdf_cdf", "cdf_theo", cdf_theo)
        self.draw_cache_set("ecdf_cdf", "ks_idx", ks_idx)

        return desc

    def compute_inferential(self, s: pd.Series, desc: dict[str, Any]) -> dict[str, Any]:
        """Run KS, AD (when applicable), and CvM tests using fitted parameters."""
        out: dict[str, Any] = {"params": {"alpha": float(self.ctx.alpha)}}

        # empty or error -> only params
        if desc.get("n", 0) == 0 or "error" in desc:
            return out

        name = self.ctx.distribution_name
        data = s.dropna().astype(float)
        dist = getattr(stats, name)

        # Refit for test args (or reconstruct from desc if preferred)
        fit_params_float = desc["params"].get("distribution_fit", None)
        if fit_params_float is None:
            fit_params = dist.fit(data)
        else:
            # we can safely reuse rounded params for display; tests generally tolerant
            fit_params = tuple(fit_params_float)

        alpha = self.ctx.alpha

        # 1) KS
        D, p_ks = stats.kstest(data, name, args=fit_params)
        out["ks"] = {"statistic": float(D), "p_value": float(p_ks), "reject": bool(p_ks < alpha)}

        # 2) Anderson-Darling (norm, expon)
        if name in ("norm", "expon"):
            ad = stats.anderson(data, dist=name)
            levels = np.array(ad.significance_level) / 100.0
            idx = int(np.argmin(np.abs(levels - alpha)))
            crit = float(ad.critical_values[idx])
            out["anderson"] = {
                "statistic": float(ad.statistic),
                "critical_value": crit,
                "critical_values": list(map(float, ad.critical_values)),
                "significance_levels": list(map(float, ad.significance_level)),
                "reject": bool(ad.statistic > crit),
            }

        # 3) Cramer-von Mises
        cvm_res = stats.cramervonmises(data, name, args=fit_params)
        out["cvm"] = {
            "statistic": float(cvm_res.statistic),
            "p_value": float(cvm_res.pvalue),
            "reject": bool(cvm_res.pvalue < alpha),
        }

        return out

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize sample size, gap, and fitted parameters."""
        if not desc:
            return {}

        n = int(desc.get("n", 0) or 0)
        dist_name = desc.get("params", {}).get("distribution_name", self.ctx.distribution_name)

        if n == 0:
            return {"context": f"n = 0 | distribution = {dist_name}", "primary_finding": None, "secondary_finding": None}

        if desc.get("error"):
            return {
                "context": f"n = {n} | distribution = {dist_name}",
                "primary_finding": None,
                "secondary_finding": None,
            }

        fmt = self.formatter.format_numeric_value
        ks_D = desc.get("ks_D")
        fit = desc.get("params", {}).get("distribution_fit")
        ks_text = f"KS max gap {fmt(ks_D, decimals=3)}" if self.is_finite(ks_D) else "KS gap unavailable"
        primary = f"Empirical CDF vs fitted {dist_name} shows {ks_text} across n = {n}."

        secondary = None
        if fit is not None:
            fit_vals = ", ".join(fmt(p, decimals=3) for p in fit)
            secondary = f"Fitted parameters: ({fit_vals})."

        return {
            "context": f"n = {n} | distribution = {dist_name}",
            "primary_finding": primary,
            "secondary_finding": secondary,
        }

    def draft_inferential_findings(self, inf: dict[str, Any], desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize KS/AD/CvM decisions at alpha."""
        if not inf:
            return {}

        n = int(desc.get("n", 0) or 0)
        dist_name = desc.get("params", {}).get("distribution_name", self.ctx.distribution_name)
        alpha = float(inf.get("params", {}).get("alpha", self.ctx.alpha))

        if n == 0 or desc.get("error"):
            return {"context": f"n = {n} | distribution = {dist_name} | alpha = {alpha}", "primary_finding": None, "secondary_finding": None}

        fmt = self.formatter.format_numeric_value

        def _decision(test: dict[str, Any] | None) -> str:
            if not test:
                return ""
            return "rejects" if test.get("reject") else "fails to reject"

        ks = inf.get("ks") or {}
        primary = f"KS test {_decision(ks)} the {dist_name} fit (D = {fmt(ks.get('statistic'), decimals=3)}, p = {fmt(ks.get('p_value'), decimals=3)}, alpha = {fmt(alpha, decimals=3)})."

        secondary_parts: list[str] = []
        ad = inf.get("anderson")
        if ad:
            secondary_parts.append(
                f"AD {_decision(ad)} (stat = {fmt(ad.get('statistic'), decimals=3)}, crit = {fmt(ad.get('critical_value'), decimals=3)})."
            )
        cvm = inf.get("cvm")
        if cvm:
            secondary_parts.append(
                f"CvM {_decision(cvm)} (stat = {fmt(cvm.get('statistic'), decimals=3)}, p = {fmt(cvm.get('p_value'), decimals=3)})."
            )
        secondary = " ".join(secondary_parts) if secondary_parts else None

        return {
            "context": f"n = {n} | distribution = {dist_name} | alpha = {fmt(alpha, decimals=3)}",
            "primary_finding": primary,
            "secondary_finding": secondary,
        }

    def subtitle_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Concise summary of test outcomes for the subtitle."""
        if not inf or desc.get("n", 0) == 0 or desc.get("error"):
            return ""

        fmt = self.formatter.format_numeric_value
        alpha = fmt(inf.get("params", {}).get("alpha"), decimals=3)

        def _fmt_test(label: str, test: dict[str, Any] | None, fields: list[str]) -> str | None:
            if not test:
                return None
            status = "reject" if test.get("reject") else "fail"
            parts = [f"{label}: {status}"]
            vals = []
            for f in fields:
                if f in test:
                    vals.append(f"{f.split('_')[0]}={fmt(test[f], decimals=3)}")
            if vals:
                parts.append("(" + ", ".join(vals) + ")")
            return " ".join(parts)

        pieces = [_fmt_test("KS", inf.get("ks"), ["p_value", "statistic"])]
        pieces.append(_fmt_test("AD", inf.get("anderson"), ["statistic"]))
        pieces.append(_fmt_test("CvM", inf.get("cvm"), ["p_value", "statistic"]))
        pieces = [p for p in pieces if p]
        if not pieces:
            return ""

        return f"Goodness-of-fit at alpha={alpha}: " + "; ".join(pieces)

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
        """Render ECDF, theoretical CDF, KS gap marker, and a stats textbox."""
        # In error case: draw a minimal frame with error note (no lines)
        if "error" in desc:
            err_text = ax.text(
                0.5,
                0.5,
                f"Input error: {desc['error']}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=self.neutral_grey("dark"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )
            self.register_annotations(ax, err_text)
            return fig, ax

        x = self.draw_cache_get("ecdf_cdf", "x")
        ecdf = self.draw_cache_get("ecdf_cdf", "ecdf")
        cdf_theo = self.draw_cache_get("ecdf_cdf", "cdf_theo")
        ks_idx = self.draw_cache_get("ecdf_cdf", "ks_idx")
        if x is None or ecdf is None or cdf_theo is None or ks_idx is None:
            desc["skip_plot"] = True
            return fig, ax

        if desc["n"] > 0:
            ecdf_color = palette[0] if len(palette) > 0 else None
            cdf_color = palette[1] if len(palette) > 1 else None
            gap_color = palette[2] if len(palette) > 2 else "red"

            ax.step(x, ecdf, where="post", label="Empirical CDF", color=ecdf_color)
            ax.plot(x, cdf_theo, "--", label=f"{self.ctx.distribution_name} CDF", color=cdf_color, linewidth=2)

            # KS max gap line
            ax.vlines(
                x[ks_idx],
                cdf_theo[ks_idx],
                ecdf[ks_idx],
                color=gap_color,
                linewidth=1.5,
                label=f"KS D = {desc.get('ks_D', np.nan):.3f}",
            )

        # stats textbox (summary)
        lines = [f"n = {desc['n']}"]
        fit = desc["params"].get("distribution_fit")
        if fit is not None:
            lines.append(f"params = {tuple(np.round(fit, 3))}")
        if "ks" in inf:
            lines.append(f"KS stat = {inf['ks']['statistic']:.3f}, p = {inf['ks']['p_value']:.3f}, reject = {inf['ks']['reject']}")
        if "anderson" in inf:
            lines.append(f"AD stat = {inf['anderson']['statistic']:.3f}, crit = {inf['anderson']['critical_value']:.3f}, reject = {inf['anderson']['reject']}")
        if "cvm" in inf:
            lines.append(f"CvM stat = {inf['cvm']['statistic']:.3f}, p = {inf['cvm']['p_value']:.3f}, reject = {inf['cvm']['reject']}")

        stats_box = ax.text(
            0.98,
            0.02,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize="small",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        self.register_annotations(ax, stats_box)

        return fig, ax
