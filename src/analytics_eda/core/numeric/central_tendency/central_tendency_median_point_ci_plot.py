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
"""Median point ± CI plot for clear central-tendency reporting.

Big idea:
Show the sample median as a single point with its confidence interval to
communicate the estimate and its precision without distributional clutter.

Why this matters:
Decision makers often need a robust location estimate. A median ± CI focuses
attention on the central tendency and uncertainty, resistant to outliers.

What this module does:
Defines a PlotContext and Plot that compute the median and a configurable CI
(bootstrap or none), render a horizontal point-with-error-bars figure, and,
optionally, run one-sample tests versus a provided population median (Wilcoxon
signed-rank and a binomial sign test).
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from analytics_eda.core.visualization.base_plot import BasePlot, PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

MedianCIMethod = Literal["bootstrap", None]


@dataclass
class CentralTendencyMedianPointCIContext(PlotContext):
    """Context for median-only point + CI plot (no density)."""

    title_template: str = "Median ± CI for {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = ""
    median_ci_method: MedianCIMethod = "bootstrap"
    alpha: float = 0.05
    bootstrap_samples: int = 1_000

    # optional population param for tests
    popmedian: float | None = None

    # styling
    marker: str = "s"
    capsize: float = 5.0
    line_width: float = 2.0
    show_footer_summary: bool = True
    show_subtitle: bool = True


class CentralTendencyMedianPointCIPlot(BasePlot):
    """Median point ± CI for robust central-tendency reporting.

    Why this matters:
    Decision makers often need a location estimate that is resilient to outliers and skew.
    A median with its confidence interval communicates the typical value and its uncertainty
    without distributional clutter.

    What this plot does:
    Computes the sample median and an optional confidence interval (bootstrap or none),
    renders a single horizontal point with error bars, and—if a population median is
    provided—runs one-sample tests (Wilcoxon signed-rank and binomial sign test) and
    summarizes their results for concise, evidence-based interpretation.
    """

    def __init__(self, ctx: CentralTendencyMedianPointCIContext):
        parts = PlotParts(series_validator=numeric_validator())
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return default descriptive payload for an empty or invalid series."""
        return {
            "params": {"median_ci_method": self.ctx.median_ci_method, "ci_level": 1.0 - float(self.ctx.alpha), "alpha": self.ctx.alpha, "bootstrap_samples": self.ctx.bootstrap_samples},
            "n": 0,
            "median": None,
            "median_formatted": "NA",
            "median_ci": (None, None),
            "median_round_decimals": getattr(self.formatter, "report_default_decimals", 2),
        }

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute median, CI, and formatted outputs for a numeric series.

        Args:
            s: Numeric pandas Series.

        Returns
        -------
            Dict with params (CI method/alpha), n, median (raw & formatted),
            CI tuple, and rounding metadata.
        """
        n = int(s.size)
        median, median_decimals, median_fmt = self.formatter.format_median(s)

        if n and self.ctx.median_ci_method == "bootstrap":
            rng = np.random.default_rng()
            boot = rng.choice(s.to_numpy(), size=(self.ctx.bootstrap_samples, n), replace=True)
            boot_meds = np.median(boot, axis=1)
            lo, hi = np.percentile(boot_meds, [100 * self.ctx.alpha / 2, 100 * (1 - self.ctx.alpha / 2)])
            median_ci = (float(lo), float(hi))
        elif self.ctx.median_ci_method is None:
            median_ci = (None, None)
        else:
            raise ValueError("median_ci_method must be 'bootstrap' or None")

        return {
            "params": {"median_ci_method": self.ctx.median_ci_method, "alpha": self.ctx.alpha, "ci_level": 1.0 - float(self.ctx.alpha), "bootstrap_samples": self.ctx.bootstrap_samples},
            "n": n,
            "median": median,
            "median_formatted": median_fmt,
            "median_ci": median_ci,
            "median_round_decimals": int(median_decimals),
        }

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Emit concise, plot-scoped descriptive finding for the median."""
        if not desc or desc.get("n", 0) == 0:
            return {"context": "n = 0", "primary_finding": "The series is empty.", "secondary_finding": None}

        lo, hi = desc.get("median_ci", (None, None))
        if lo is None or hi is None:
            ci_txt = "CI not available"
        else:
            d = desc["median_round_decimals"]
            lo_txt = self.formatter.format_numeric_value(lo, decimals=d)
            hi_txt = self.formatter.format_numeric_value(hi, decimals=d)
            ci_txt = f"{lo_txt} to {hi_txt}"

        level = int(desc["params"]["ci_level"] * 100)
        return {
            "context": f"n = {desc['n']}",
            "primary_finding": f"Median is {desc['median_formatted']} with {level}% CI ({ci_txt}).",
            "secondary_finding": None,
        }

    def default_inferential(self) -> dict[str, Any]:
        """Return default inferential parameters (alpha, population median)."""
        return {"params": {"alpha": self.ctx.alpha, "popmedian": self.ctx.popmedian}}

    def compute_inferential(self, s: pd.Series, desc: dict[str, Any]) -> dict[str, Any]:
        """Run optional one-sample tests against a population median.

        Computes the Wilcoxon signed-rank test and a binomial sign test
        (when applicable).

        Args:
            s: Numeric pandas Series.
            desc: Output of `compute_descriptive` (not strictly required here).

        Returns
        -------
            Dict under `popmedian` with test statistics, p-values, alpha,
            and reject flags (where applicable).
        """
        out: dict[str, Any] = {"params": {"alpha": self.ctx.alpha, "popmedian": self.ctx.popmedian}}
        if self.ctx.popmedian is None:
            return out

        out["popmedian"] = {}
        diff = s - self.ctx.popmedian

        # Wilcoxon: guard all-zeros / insufficient data
        try:
            stat_wr, p_wr = stats.wilcoxon(diff)
            out["popmedian"]["wilcoxon"] = {
                "statistic": float(stat_wr),
                "p_value": float(p_wr),
                "alpha": float(self.ctx.alpha),
                "reject": bool(p_wr < self.ctx.alpha),
            }
            # rank-biserial r = 1 - 2*W / (n*(n+1)/2), using n_eff = len(nonzero diffs)
            nonzero = (s - self.ctx.popmedian).to_numpy()
            nonzero = nonzero[nonzero != 0]
            n_eff = nonzero.size
            if n_eff >= 1 and np.isfinite(stat_wr):
                T = n_eff * (n_eff + 1) / 2.0
                r_rb = 1.0 - 2.0 * stat_wr / T
                out["popmedian"]["wilcoxon"]["rank_biserial"] = float(r_rb)
        except ValueError:
            out["popmedian"]["wilcoxon"] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "alpha": float(self.ctx.alpha),
                "reject": False,
            }

        nonzero = diff[diff != 0]
        n_sign = int(nonzero.size)
        if n_sign > 0:
            pos = int((nonzero > 0).sum())
            bt = stats.binomtest(pos, n_sign, p=0.5)
            out["popmedian"]["sign_test"] = {
                "num_positive": pos,
                "num_negative": n_sign - pos,
                "n": n_sign,
                "statistic": float(bt.statistic),
                "p_value": float(bt.pvalue),
                "alpha": float(self.ctx.alpha),
                "reject": bool(bt.pvalue < self.ctx.alpha),
            }

        return out

    def draft_inferential_findings(self, inf: dict[str, Any], desc: dict[str, Any]) -> dict[str, Any]:
        """Emit concise, evidence-based statements derived strictly from `inf`.

        Returns
        -------
            Dict with context, primary_finding, and optional secondary_finding,
            or an empty dict if no inference was run.
        """
        if not inf or "popmedian" not in inf:
            return {}

        pm = inf["popmedian"]
        alpha = float(inf["params"]["alpha"])
        fmt = self.formatter

        w = pm.get("wilcoxon")
        s = pm.get("sign_test")

        # Helper to check usable p-values
        def usable(test):
            return bool(test) and np.isfinite(test.get("p_value", np.nan))

        # Pull optional extras for context (safe formatting)
        def wilcoxon_context_bits(wdict: dict | None) -> str:
            if not usable(wdict):
                return ""
            bits = []
            # W statistic (integer-like)
            if np.isfinite(wdict.get("statistic", np.nan)):
                wtxt = fmt.format_test_statistic(wdict["statistic"])
                bits.append(f"W={wtxt}")
            # rank-biserial effect size r (if computed)
            r = wdict.get("rank_biserial", None)
            if r is not None and np.isfinite(r):
                # 2 decimals is typical for r
                rtxt = fmt.format_numeric_value(r, decimals=2)
                bits.append(f"r={rtxt}")
            return (" [" + ", ".join(bits) + "]") if bits else ""

        def sign_context_bits(sdict: dict | None) -> str:
            if not usable(sdict):
                return ""
            n = sdict.get("n", None)
            pos = sdict.get("num_positive", None)
            neg = sdict.get("num_negative", None)
            # Only add when we have counts
            if all(v is not None for v in (n, pos, neg)):
                return f" [n={int(n)}, +={int(pos)}, -={int(neg)}]"
            return ""

        # Heuristics to choose primary
        n_sign = s.get("n") if isinstance(s, dict) else None
        many_zeros_or_ties = n_sign is not None and n_sign < 10
        use_sign_as_primary = (not usable(w)) or many_zeros_or_ties

        if use_sign_as_primary and usable(s):
            # Primary (unchanged sentence)
            ptxt = fmt.format_p_value(s["p_value"])
            rej = s["reject"]
            context = f"Binomial sign test, α = {fmt.format_alpha(alpha)}"
            # Add sign-test counts into context
            context += sign_context_bits(s)
            out = {
                "context": context,
                "primary_finding": f"Difference from population median {'is' if rej else 'is not'} statistically significant ({ptxt}).",
                "secondary_finding": None,
            }
            # Secondary (unchanged sentence), keep Wilcoxon as confirmation if available
            if usable(w):
                wp = fmt.format_p_value(w["p_value"])
                wrej = "(reject)" if w["reject"] else "(ns)"
                out["secondary_finding"] = f"Wilcoxon signed-rank: {wp} {wrej}."
                # Enrich context further with Wilcoxon W and r (still not changing the sentences)
                out["context"] += " • Wilcoxon" + wilcoxon_context_bits(w)
            return out

        # Default: Wilcoxon primary
        if usable(w):
            wp = fmt.format_p_value(w["p_value"])
            rej = w["reject"]
            context = f"Wilcoxon signed-rank, α = {fmt.format_alpha(alpha)}"
            # Add Wilcoxon W and r into context
            context += wilcoxon_context_bits(w)
            out = {
                "context": context,
                "primary_finding": f"Difference from population median {'is' if rej else 'is not'} statistically significant ({wp}).",
                "secondary_finding": None,
            }
            if usable(s):
                sp = fmt.format_p_value(s["p_value"])
                srej = "(reject)" if s["reject"] else "(ns)"
                # Secondary (unchanged sentence)
                out["secondary_finding"] = f"Sign test (robust check): {sp} {srej}."
                # Enrich context with sign-test counts
                out["context"] += " • Sign test" + sign_context_bits(s)
            return out

        # If neither is usable, return empty
        return {}

    def subtitle_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """
        Build a short subtitle for the Median ± CI plot.

        Focus: sample size, CI level/method, optional CI range, and (if applicable)
        a compact one-sample test decision against H₀: median = popmedian.

        Avoids duplicating the title ("Median ± CI for {name}{modifiers}").
        """
        if not getattr(self.ctx, "show_subtitle", True):
            return ""

        n = int(desc.get("n", 0) or 0)
        if n == 0:
            return "No non-null observations."

        fmt = self.formatter
        params = desc.get("params", {})
        ci_level = params.get("ci_level", 1.0 - float(getattr(self.ctx, "alpha", 0.05)))
        ci_pct = f"{int(round(ci_level * 100))}%"
        method = params.get("median_ci_method", getattr(self.ctx, "median_ci_method", "bootstrap"))
        method_label = "bootstrap" if method == "bootstrap" else "none"

        parts: list[str] = [f"n = {n:,}", f"{ci_pct} CI ({method_label})"]

        # Optional CI range
        lo, hi = desc.get("median_ci", (None, None))
        d = int(desc.get("median_round_decimals", getattr(fmt, "report_default_decimals", 2)))
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            lo_txt = fmt.format_numeric_value(lo, decimals=d)
            hi_txt = fmt.format_numeric_value(hi, decimals=d)
            parts.append(f"{lo_txt}–{hi_txt}")

        # Optional one-sample test decision (prefer Wilcoxon; fallback to sign test)
        pm = (inf or {}).get("popmedian") if isinstance(inf, dict) else None
        if pm:
            alpha = float((inf.get("params") or {}).get("alpha", getattr(self.ctx, "alpha", 0.05)))

            def usable(test: dict | None) -> bool:
                return bool(test) and np.isfinite(test.get("p_value", np.nan))

            if usable(pm.get("wilcoxon")):
                decision = "significant" if pm["wilcoxon"].get("reject") else "not significant"
                parts.append(f"Wilcoxon: {decision} at α={fmt.format_alpha(alpha)}")
            elif usable(pm.get("sign_test")):
                decision = "significant" if pm["sign_test"].get("reject") else "not significant"
                parts.append(f"Sign test: {decision} at α={fmt.format_alpha(alpha)}")

        return " • ".join(parts)

    def footer_summary_text(self, desc: dict[str, Any], inf: dict[str, Any], chart_metadata: dict[str, Any]) -> str:
        """Return a compact footer summary string (e.g., sample size)."""
        return f"n = {desc['n']}"

    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):
        """Render the median point ± CI, optional H₀ reference, and test annotation.

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

        median = desc["median"]
        lo, hi = desc["median_ci"]
        if desc["n"] > 0 and median is not None and np.isfinite(median):
            if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(
                    x=median,
                    y=0,
                    xerr=[[median - lo], [hi - median]],
                    fmt=getattr(self.ctx, "marker", "s"),
                    color=color,
                    capsize=getattr(self.ctx, "capsize", 5.0),
                    elinewidth=getattr(self.ctx, "line_width", 2.0),
                )
            else:
                ax.scatter([median], [0], color=color, marker=getattr(self.ctx, "marker", "s"))

        if getattr(self.ctx, "popmedian", None) is not None:
            ax.axvline(self.ctx.popmedian, color="gray", linestyle="--", linewidth=1)
            ax.text(self.ctx.popmedian, 0.1, "H₀", ha="center", va="bottom", fontsize="small", color="gray")

        ax.set_yticks([])

        pm = inf.get("popmedian")
        if pm:
            parts = []
            if "wilcoxon" in pm:
                wstat = fmt.format_test_statistic(pm["wilcoxon"]["statistic"], decimals=0)
                ptxt = fmt.format_p_value(pm["wilcoxon"]["p_value"])
                rej = "(reject)" if pm["wilcoxon"]["reject"] else "(ns)"
                parts.append(f"W={wstat}, {ptxt} {rej}")
            if "sign_test" in pm:
                st = pm["sign_test"]
                ptxt = fmt.format_p_value(st["p_value"])
                parts.append(f"Sign: +={st['num_positive']}, -={st['num_negative']}, {ptxt} " f"{'(reject)' if st['reject'] else '(ns)'}")
            if parts:
                ax.text(0.01, 0.95, "; ".join(parts), transform=ax.transAxes, va="top", ha="left", fontsize="small", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

        return fig, ax
