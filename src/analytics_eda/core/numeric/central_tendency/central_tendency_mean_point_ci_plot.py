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
"""Central tendency (mean) point ± CI plot.

Big idea:
Show the sample mean as a single point with a confidence interval for fast,
uncluttered communication of the estimate and its precision.

Why this matters:
Decision makers need to read the average and its uncertainty quickly without
distributional noise or chartjunk.

What this module does:
Provides a PlotContext and Plot class that compute the mean and its CI
(Student’s t or bootstrap), renders a horizontal point-with-error-bars figure,
and optionally reports one-sample inference against a known population mean
(Cohen’s d, t-test, and z-test when σ² is supplied).
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from analytics_eda.core.visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

MeanCIMethod = Literal["t", "bootstrap"]


@dataclass
class CentralTendencyMeanPointCIContext(PlotContext):
    """Context for mean-only point + CI plot (no density)."""

    title_template: str = "Mean ± CI for {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = ""
    mean_ci_method: MeanCIMethod = "t"
    alpha: float = 0.05
    bootstrap_samples: int = 1_000

    # optional population params for tests
    popmean: float | None = None
    popvariance: float | None = None  # if provided → Z-test in addition to t-test

    # styling
    marker: str = "o"
    capsize: float = 5.0
    line_width: float = 2.0
    show_footer_summary: bool = True
    show_subtitle: bool = True


class CentralTendencyMeanPointCIPlot(BasePlot):
    """
    Shows the sample mean with its confidence interval as a single, uncluttered estimate.

    Why this matters
    ----------------
    Decision makers need to read an estimated average and its uncertainty quickly.
    A point ± CI communicates the estimate and precision directly without distributional noise.

    What this plot does
    -------------------
    Computes the mean and a configurable CI (Student's t or bootstrap), draws a single point with
    horizontal error bars, and—if population parameters are provided—reports one-sample tests
    (Cohen’s d, t-test, optional z-test) in a compact annotation.
    """

    def __init__(self, ctx: CentralTendencyMeanPointCIContext):
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return default descriptive payload for an empty or invalid series."""
        return {
            "params": {"mean_ci_method": self.ctx.mean_ci_method, "ci_level": 1.0 - float(self.ctx.alpha), "alpha": self.ctx.alpha, "bootstrap_samples": self.ctx.bootstrap_samples},
            "n": 0,
            "mean": None,
            "mean_formatted": "NA",
            "mean_ci": (None, None),
            "mean_round_decimals": getattr(self.formatter, "report_default_decimals", 2),
        }

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute mean, CI, and formatted outputs for a numeric series.

        Args:
            s: Numeric pandas Series.

        Returns
        -------
            Dict with params (CI method/level), n, mean (raw & formatted),
            CI tuple, and rounding metadata.
        """
        n = int(s.size)
        mean, mean_decimals, mean_fmt = self.formatter.format_mean(s)

        if n:
            if self.ctx.mean_ci_method == "t":
                sem = stats.sem(s, ddof=1)
                ci_low, ci_high = stats.t.interval(1 - self.ctx.alpha, df=n - 1, loc=mean, scale=sem)
                mean_ci = (float(ci_low), float(ci_high))
            elif self.ctx.mean_ci_method == "bootstrap":
                rng = np.random.default_rng()
                boot = rng.choice(s.to_numpy(), size=(self.ctx.bootstrap_samples, n), replace=True)
                boot_means = boot.mean(axis=1)
                lo, hi = np.percentile(boot_means, [100 * self.ctx.alpha / 2, 100 * (1 - self.ctx.alpha / 2)])
                mean_ci = (float(lo), float(hi))
            else:
                raise ValueError("mean_ci_method must be 't' or 'bootstrap'")
        else:
            mean_ci = (None, None)

        return {
            "params": {"mean_ci_method": self.ctx.mean_ci_method, "ci_level": 1.0 - float(self.ctx.alpha), "alpha": self.ctx.alpha, "bootstrap_samples": self.ctx.bootstrap_samples},
            "n": n,
            "mean": mean,
            "mean_formatted": mean_fmt,
            "mean_ci": mean_ci,
            "mean_round_decimals": int(mean_decimals),
        }

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Emit concise, human-readable findings derived strictly from `desc`.

        Returns an empty dict if there is no meaningful content to report.
        """
        if not desc or desc.get("n", 0) == 0:
            return {
                "context": "n = 0",
                "primary_finding": "The series is empty.",
                "secondary_finding": None,
            }
        ci = desc.get("mean_ci", (None, None))
        ci_txt = "CI not available" if None in ci else f"{self.formatter.format_numeric_value(ci[0], decimals=desc['mean_round_decimals'])} to {self.formatter.format_numeric_value(ci[1], decimals=desc['mean_round_decimals'])}"
        return {
            "context": f"n = {desc['n']}",
            "primary_finding": f"Mean is {desc['mean_formatted']} with {int(desc['params']['ci_level'] * 100)}% CI ({ci_txt}).",
            "secondary_finding": None,
        }

    def default_inferential(self) -> dict[str, Any]:
        """Return default inferential parameters (alpha, population values)."""
        return {"params": {"alpha": self.ctx.alpha, "popmean": self.ctx.popmean, "popvariance": self.ctx.popvariance}}

    def compute_inferential(self, s: pd.Series, desc: dict[str, Any]) -> dict[str, Any]:
        """Run optional one-sample tests against a population mean.

        Computes Cohen’s d, a one-sample t-test, and (if σ² is provided)
        a one-sample z-test.

        Args:
            s: Numeric pandas Series.
            desc: Output of `compute_descriptive` (used for n, mean).

        Returns
        -------
            Dict under the `popmean` key with test statistics, p-values,
            alpha, and reject flags (where applicable).
        """
        out: dict[str, Any] = {"params": {"alpha": self.ctx.alpha, "popmean": self.ctx.popmean, "popvariance": self.ctx.popvariance}}
        if self.ctx.popmean is None:
            return out

        n = desc["n"]
        mean = desc["mean"]
        out["popmean"] = {}

        if n >= 2 and mean is not None and np.isfinite(mean):
            sd = float(s.std(ddof=1))
            cohens_d = (mean - self.ctx.popmean) / sd if (sd > 0 and np.isfinite(sd)) else None
            out["popmean"]["cohens_d"] = None if cohens_d is None else float(cohens_d)

            t_stat, t_p = stats.ttest_1samp(s, self.ctx.popmean)
            out["popmean"]["t_test"] = {
                "statistic": float(t_stat),
                "p_value": float(t_p),
                "alpha": float(self.ctx.alpha),
                "reject": bool(t_p < self.ctx.alpha),
            }

            if self.ctx.popvariance is not None and self.ctx.popvariance >= 0:
                sigma = float(np.sqrt(self.ctx.popvariance))
                if sigma > 0:
                    z = (mean - self.ctx.popmean) / (sigma / np.sqrt(n))
                    z_p = 2 * (1 - stats.norm.cdf(abs(z)))
                    out["popmean"]["z_test"] = {
                        "statistic": float(z),
                        "p_value": float(z_p),
                        "alpha": float(self.ctx.alpha),
                        "reject": bool(z_p < self.ctx.alpha),
                    }
        return out

    def draft_inferential_findings(self, inf: dict[str, Any], desc: dict[str, Any]) -> dict[str, Any]:
        """Emit concise, evidence-based statements derived strictly from `inf`.

        Returns
        -------
            Dict with context, primary_finding, and optional secondary_finding,
            or an empty dict if no inference was run.
        """
        if not inf or "popmean" not in inf:
            return {}

        pm = inf["popmean"]
        alpha = float(inf["params"]["alpha"])
        ctx_str = f"One-sample t-test, α = {self.formatter.format_alpha(alpha)}"

        t = pm.get("t_test", {})
        d = pm.get("cohens_d")
        z = pm.get("z_test")

        if not t:
            return {}

        rej = t["reject"]
        ptxt = self.formatter.format_p_value(t["p_value"])
        ttxt = self.formatter.format_test_statistic(t["statistic"])

        # Combine t and d in primary finding
        primary = f"The sample mean {'differs significantly' if rej else 'does not differ significantly'} from the population mean ({ttxt}, {ptxt})."

        if d is not None:
            primary += f" The effect size is {self.formatter.format_cohens_d(d)}."

        # Optional secondary finding for z-test
        secondary = None
        if z:
            ztxt = self.formatter.format_test_statistic(z["statistic"])
            zp = self.formatter.format_p_value(z["p_value"])
            secondary = f"A confirmatory z-test yielded {ztxt} ({zp})."

        return {
            "context": ctx_str,
            "primary_finding": primary,
            "secondary_finding": secondary,
        }

    def subtitle_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """
        Build a short subtitle for the Mean ± CI plot.

        Focus: sample size, CI level/method, optional CI range, and (if applicable)
        a compact one-sample test decision against H₀: μ = popmean.

        Avoids duplicating the title ("Mean ± CI for {name}{modifiers}").
        """
        # Allow caller/context to suppress subtitles entirely
        if not getattr(self.ctx, "show_subtitle", True):
            return ""

        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return "No non-null observations."

        fmt = self.formatter
        params = desc.get("params", {})
        ci_level = params.get("ci_level", 1.0 - float(getattr(self.ctx, "alpha", 0.05)))
        ci_pct = f"{int(round(ci_level * 100))}%"
        method = params.get("mean_ci_method", getattr(self.ctx, "mean_ci_method", "t"))
        method_label = "t" if method == "t" else "bootstrap"

        parts: list[str] = [f"n = {n:,}", f"{ci_pct} CI ({method_label})"]

        # If we have a finite CI, include the range succinctly
        lo, hi = desc.get("mean_ci", (None, None))
        d = int(desc.get("mean_round_decimals", getattr(fmt, "report_default_decimals", 2)))
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            lo_txt = fmt.format_numeric_value(lo, decimals=d)
            hi_txt = fmt.format_numeric_value(hi, decimals=d)
            parts.append(f"{lo_txt}–{hi_txt}")

        # If a population mean is provided and a t-test ran, summarize the decision (no raw stats)
        pm = (inf or {}).get("popmean") if isinstance(inf, dict) else None
        if pm and "t_test" in pm:
            alpha = float((inf.get("params") or {}).get("alpha", getattr(self.ctx, "alpha", 0.05)))
            decision = "significant" if pm["t_test"].get("reject") else "not significant"
            parts.append(f"one-sample t-test: {decision} at α={fmt.format_alpha(alpha)}")

        return " • ".join(parts)

    def footer_summary_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Return a compact footer summary string (e.g., sample size)."""
        return f"n = {desc['n']}"

    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):
        """Render the point ± CI, optional H₀ reference, and test annotation.

        Args:
            s: Numeric pandas Series.
            desc: Descriptive stats dict.
            inf: Inferential stats dict.
            chart_metadata: Title/xlabel/ylabel/data_source/file_name, etc.
            fig: Matplotlib Figure.
            ax: Matplotlib Axes.
            palette: Sequence of colors supplied by the renderer.

        Returns
        -------
            (fig, ax): The same Matplotlib figure and axes, post-render.
        """
        fmt = self.formatter
        color = palette[0]

        ax.axhline(0, color="none")

        mean = desc["mean"]
        lo, hi = desc["mean_ci"]
        if desc["n"] > 0 and mean is not None and np.isfinite(mean):
            if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(
                    x=mean,
                    y=0,
                    xerr=[[mean - lo], [hi - mean]],
                    fmt=getattr(self.ctx, "marker", "o"),
                    color=color,
                    capsize=getattr(self.ctx, "capsize", 5.0),
                    elinewidth=getattr(self.ctx, "line_width", 2.0),
                )
            else:
                ax.scatter([mean], [0], color=color, marker=getattr(self.ctx, "marker", "o"))

        if getattr(self.ctx, "popmean", None) is not None:
            ax.axvline(self.ctx.popmean, color="gray", linestyle="--", linewidth=1)
            ax.text(self.ctx.popmean, 0.1, "H₀", ha="center", va="bottom", fontsize="small", color="gray")

        ax.set_yticks([])

        pm = inf.get("popmean")
        if pm:
            parts = []
            if "cohens_d" in pm and pm["cohens_d"] is not None:
                parts.append(f"d={fmt.format_cohens_d(pm['cohens_d'])}")
            if "t_test" in pm:
                tstat = fmt.format_test_statistic(pm["t_test"]["statistic"], decimals=2)
                ptxt = fmt.format_p_value(pm["t_test"]["p_value"])
                rej = "(reject)" if pm["t_test"]["reject"] else "(ns)"
                parts.append(f"t={tstat}, {ptxt} {rej}")
            if "z_test" in pm:
                zstat = fmt.format_test_statistic(pm["z_test"]["statistic"], decimals=2)
                ptxt = fmt.format_p_value(pm["z_test"]["p_value"])
                rej = "(reject)" if pm["z_test"]["reject"] else "(ns)"
                parts.append(f"z={zstat}, {ptxt} {rej}")
            if parts:
                ax.text(0.01, 0.95, "; ".join(parts), transform=ax.transAxes, va="top", ha="left", fontsize="small", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

        return fig, ax
