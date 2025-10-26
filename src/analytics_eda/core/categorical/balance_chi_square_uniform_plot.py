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
"""Chi-square goodness-of-fit plot against a uniform distribution."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import categorical_validator

from ..visualization.base_plot import BasePlot, PlotContext


@dataclass
class BalanceChiSquareUniformContext(PlotContext):
    """Context options for the chi-square uniform balance plot."""

    title_template: str = "Chi-Square Goodness-of-Fit: {name}{modifiers}"
    xlabel: str = "Observed − Expected (count)"
    ylabel: str = "Category"
    alpha: float = 0.05
    show_subtitle: bool = True
    is_orientation_vertical: bool = False
    headroom_preserve_symmetry: bool = True

    bar_highlight_top: bool = True
    bar_top_n: int = 1
    bar_top_include_ties: bool = True
    sort_mode: Literal["abs_delta", "signed_delta", "category"] = "abs_delta"
    delta_metric: Literal["count", "std_resid"] = "count"
    show_value_in_bar_label: bool = True
    show_count_in_bar_label: bool = False
    label_value_format: Literal["±count", "ratio", "±count_and_ratio", "std_resid"] = "±count_and_ratio"
    color_by_sign: bool = False


class BalanceChiSquareUniformPlot(BasePlot):
    """
    Tests whether categorical frequencies deviate from a uniform distribution.

    Why this matters:
    - Uniform balance is often an assumption or target (e.g., stratified samples, equitable allocations).
      Large deviations can signal sampling bias, pipeline errors, or drift.

    What this plot does:
    - Computes observed counts by category and the uniform expected counts.
    - Runs a chi-square goodness-of-fit test (H₀: observed ~ Uniform).
    - Visualizes observed vs. expected counts side-by-side and summarizes the test result.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {"total", "k"},
        "inferential_stats": {
            "chi2_gof_null_uniform": {
                "statistic", "p_value", "alpha", "reject", "warning?"}
        },
        "chart_metadata": {...}
      }
    """

    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=categorical_validator()
        )
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return an empty descriptive payload with total and category count."""
        return {"total": 0, "k": 0}

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute observed/expected counts, deltas, sorting, cache arrays, and top labels."""
        freq = s.value_counts()
        categories = sorted(freq.index.tolist())
        observed = np.asarray([int(freq[c]) for c in categories], dtype=float)
        total = int(observed.sum())
        k = int(len(categories))
        expected = np.asarray([total / k] * k, dtype=float) if k > 0 else np.array([])

        if k == 0 or total == 0:
            # clear cache; return minimal desc
            self.draw_cache_set("chi2_uniform", "labels", [])
            self.draw_cache_set("chi2_uniform", "observed", np.array([]))
            self.draw_cache_set("chi2_uniform", "expected", np.array([]))
            self.draw_cache_set("chi2_uniform", "delta", np.array([]))
            self.draw_cache_set("chi2_uniform", "std_resid", np.array([]))
            return {"total": int(total), "k": int(k)}

        delta = observed - expected

        # Guard sqrt(E): std resid = (O-E)/sqrt(E); nan where E==0
        std_resid = np.divide(delta, np.sqrt(expected, where=expected>0), out=np.full_like(delta, np.nan), where=expected>0)
        ratio = np.divide(observed, expected, out=np.full_like(delta, np.nan), where=expected>0)  # O/E
        delta_pct_of_expected = np.where(expected > 0, delta / expected, np.nan)  # signed %

        # Sorting
        mode = getattr(self.ctx, "sort_mode", "abs_delta")
        if mode == "signed_delta":
            order = np.argsort(delta)  # neg .. pos
        elif mode == "category":
            order = np.arange(k)
        else:  # abs_delta (default)
            order = np.argsort(-np.abs(delta))

        labels = np.array(categories)[order].tolist()
        observed_o = observed[order]
        expected_o = expected[order]
        delta_o = delta[order]
        std_resid_o = std_resid[order]
        ratio_o = ratio[order]
        dpexp_o = delta_pct_of_expected[order]

        # Cache for draw
        self.draw_cache_set("chi2_uniform", "labels", labels)
        self.draw_cache_set("chi2_uniform", "observed", observed_o)
        self.draw_cache_set("chi2_uniform", "expected", expected_o)
        self.draw_cache_set("chi2_uniform", "delta", delta_o)
        self.draw_cache_set("chi2_uniform", "std_resid", std_resid_o)
        self.draw_cache_set("chi2_uniform", "ratio", ratio_o)
        self.draw_cache_set("chi2_uniform", "delta_pct_of_expected", dpexp_o)

        # Winners (based on chosen delta_metric)
        metric = getattr(self.ctx, "delta_metric", "count")
        scores = np.abs(std_resid_o) if metric == "std_resid" else np.abs(delta_o)
        top_n = max(1, int(getattr(self.ctx, "bar_top_n", 1)))
        include_ties = bool(getattr(self.ctx, "bar_top_include_ties", True))

        pairs = list(zip(labels, scores, strict=True))
        pairs.sort(key=lambda kv: (-kv[1], kv[0]))  # stable
        if pairs:
            cutoff = pairs[min(top_n, len(pairs)) - 1][1]
            top_labels = [lbl for lbl, sc in pairs if (sc >= cutoff if include_ties else sc > 0 and sc >= cutoff) and sc > 0]
        else:
            top_labels = []

        desc = {
            "params": {
                "sort_mode": mode,
                "delta_metric": metric,
                "bar_top_n": top_n,
                "bar_top_include_ties": include_ties,
                "label_value_format": getattr(self.ctx, "label_value_format", "±count"),
            },
            "total": int(total),
            "k": int(k),
            "n_over": int(np.sum(delta > 0)),
            "n_under": int(np.sum(delta < 0)),
            "max_abs_delta": float(np.nanmax(np.abs(delta))) if k else 0.0,
            "bars": {
                lbl: {
                    "observed": int(o),
                    "expected": float(e),
                    "delta": int(d),
                    "delta_pct_of_expected": float(dp),
                    "std_resid": float(sr) if np.isfinite(sr) else np.nan,
                    "ratio_oe": float(r) if np.isfinite(r) else np.nan,
                }
                for lbl, o, e, d, dp, sr, r in zip(labels, observed_o, expected_o, delta_o, dpexp_o, std_resid_o, ratio_o, strict=True)
            },
            "top_labels": top_labels,
        }
        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize the largest deviations from uniform (human-readable)."""
        total = int(desc.get("total", 0))
        k = int(desc.get("k", 0))
        if total == 0 or k == 0:
            return {}

        delta  = self.draw_cache_get("chi2_uniform", "delta", np.array([]))
        if delta is None or len(delta) == 0:
            return {}

        # All zero deltas -> perfect uniform
        if float(np.nanmax(np.abs(delta))) == 0.0:
            return {
                "context": f"N = {total:,} values across {k} categories",
                "primary_finding": "Observed frequencies match a uniform expectation.",
                "secondary_finding": None,
            }

        winners = list(desc.get("top_labels", []))
        bars = desc.get("bars", {})
        fmt = getattr(self.ctx, "label_value_format", "±count")

        def _part(lbl: str) -> str:
            b = bars.get(lbl, {})
            d = int(b.get("delta", 0))
            if fmt == "std_resid":
                sr = b.get("std_resid", np.nan)
                return f"{lbl} (±{d:+d}, SR={self.formatter.format_numeric_value(sr)})"
            elif fmt == "ratio":
                r = b.get("ratio_oe", np.nan)
                return f"{lbl} (O/E={self.formatter.format_numeric_value(r)})"
            elif fmt == "±count_and_ratio":
                r = b.get("ratio_oe", np.nan)
                return f"{lbl} ({d:+d}, O/E={self.formatter.format_numeric_value(r)})"
            else:  # "±count"
                return f"{lbl} ({d:+d})"

        parts = [_part(lbl) for lbl in sorted(winners)] if winners else []

        primary = "Category frequencies deviate from a uniform expectation."
        secondary = (
            f"Largest absolute deviation: {parts[0]}." if len(parts) == 1
            else ("Largest deviations (tie): " + ", ".join(parts) + ".") if len(parts) > 1
            else None
        )

        return {
            "context": f"N = {total:,} values across {k} categories",
            "primary_finding": primary,
            "secondary_finding": secondary,
        }


    def compute_inferential(self, s: pd.Series, desc: dict[str, Any]) -> dict[str, Any]:
        """Run chi-square GOF vs. uniform, return statistic, df, p-value, alpha, and decision."""
        k = desc.get("k", 0)
        total = desc.get("total", 0)
        if k == 0 or total == 0:
            return {}

        warning = None
        expected = self.draw_cache_get("chi2_uniform", "expected")
        observed = self.draw_cache_get("chi2_uniform", "observed")

        # Assumption checks
        if any(e <= 0 for e in expected):
            return {
                "chi2_gof_null_uniform": {
                    "warning": "Test not computed: at least one expected cell count is 0 (violates chi-square requirements)."
                }
            }

        n_lt5 = int(np.sum(np.asarray(expected) < 5))
        min_exp = float(np.min(expected)) if len(expected) else None
        warning = None
        if n_lt5 > 0:
            warning = (
                f"Assumption caution: {n_lt5} of {k} categories have expected counts < 5 "
                f"(minimum expected = {self.formatter.format_numeric_value(min_exp)}); chi-square results may be unreliable."
            )

        # Test
        chi2_stat, p_val = chisquare(f_obs=observed, f_exp=expected)
        df = max(desc.get("k", 0) - 1, 0)
        alpha = float(getattr(self.ctx, "alpha", 0.05))

        res = {
            "chi2_gof_null_uniform": {
                "df": df,
                "statistic": float(chi2_stat),
                "p_value": float(p_val),
                "alpha": alpha,
                "reject": bool(p_val < alpha),
            }
        }
        if warning:
            res["chi2_gof_null_uniform"]["warning"] = warning
        return res

    def draft_inferential_findings(self, inf: dict[str, Any], desc: dict[str, Any]) -> dict[str, Any]:
        """Render a readable decision string from the chi-square test result."""
        res = (inf or {}).get("chi2_gof_null_uniform")
        if not res:
            return {}

        k = int(desc.get("k", 0))
        total = int(desc.get("total", 0))
        p = float(res.get("p_value", None))
        alpha = float(res.get("alpha", 0.05))
        df = int(res.get("df", max(k - 1, 0)))
        reject = bool(res.get("reject", False))
        stat = float(res.get("statistic", None))
        warning = res.get("warning")

        decision = (
            "Frequencies differ from a uniform distribution"
            if reject else
            "No statistically significant deviation from uniform"
        )

        findings = {
            "context": f"Chi-square GOF on {k} categories (N = {total:,})",
            "primary_finding": f"{decision} ({self.formatter.format_p_value(p)} vs α = {self.formatter.format_alpha(alpha)}).",
            "secondary_finding": f"χ²({self.formatter.format_df(df)}) = {self.formatter.format_test_statistic(stat)}."
        }

        if warning:
            findings["secondary_finding"] = (
                f"{findings['secondary_finding']} Assumption warning: {warning}"
                if findings["secondary_finding"] else
                f"Assumption warning: {warning}"
            )

        return findings

    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):
        """Render signed deltas as a horizontal bar chart with optional coloring and labels."""
        labels = self.draw_cache_get("chi2_uniform", "labels", [])
        delta  = self.draw_cache_get("chi2_uniform", "delta", np.array([]))
        std_r  = self.draw_cache_get("chi2_uniform", "std_resid", np.array([]))
        ratio  = self.draw_cache_get("chi2_uniform", "ratio", np.array([]))

        if delta is None or len(labels) == 0:
            return fig, ax

        y = np.arange(len(labels))

        # Base bars
        bars = ax.barh(y, delta, color=self.neutral_grey())

        # Optional sign color
        if getattr(self.ctx, "color_by_sign", False):
            for i, b in enumerate(bars):
                if delta[i] >= 0:
                    b.set_color(palette[0])  # “over” color
                else:
                    b.set_color(palette[1])  # “under” color

        # Highlight winners (on top of base/sign colors)
        if getattr(self.ctx, "bar_highlight_top", True):
            winners = set(desc.get("top_labels", []))
            for i, lbl in enumerate(labels):
                if lbl in winners:
                    bars[i].set_color(palette[0])

        # Zero reference and symmetric limits
        ax.axvline(0, color=self.neutral_grey("dark"), linewidth=1)
        if getattr(self.ctx, "symmetric_xlim", True):
            m = float(np.nanmax(np.abs(delta))) if len(delta) else 1.0
            ax.set_xlim(-1.05 * m, 1.05 * m)

        # Y ticks
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        # Edge labels
        fmt = getattr(self.ctx, "label_value_format", "±count")
        def edge_text(i: int) -> str:
            if fmt == "std_resid" and np.isfinite(std_r[i]):
                return f"{std_r[i]:+.2f}"
            elif fmt == "ratio" and np.isfinite(ratio[i]):
                return f"O/E={self.formatter.format_numeric_value(ratio[i])}"
            elif fmt == "±count_and_ratio" and np.isfinite(ratio[i]):
                return f"{int(delta[i]):+d} (O/E={self.formatter.format_numeric_value(ratio[i])})"
            else:
                return f"{int(delta[i]):+d}"

        labels_txt = [edge_text(i) for i in range(len(labels))]
        text_objs = ax.bar_label(bars, labels=labels_txt, label_type="edge", padding=3, fontsize="small")
        self.register_annotations(ax, text_objs)

        return fig, ax

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Build a concise subtitle summarizing the GOF decision and p-value."""
        res = inf.get("chi2_gof_null_uniform")
        if res:
            return f"Uniform GOF: {'Reject' if res['reject'] else 'Fail to reject'} at α={self.formatter.format_alpha(res['alpha'])} ({self.formatter.format_p_value(res['p_value'])})"
        return ""
