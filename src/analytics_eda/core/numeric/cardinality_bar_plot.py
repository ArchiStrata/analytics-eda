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
"""Bar chart showing numeric cardinality and discreteness, with top-N coverage."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import (
    SeriesBarChartContext,
    SeriesBarChartMixin,
)

from ..visualization.base_plot import BasePlot
from ..visualization.plot_parts import PlotParts
from ..visualization.validation import numeric_validator


@dataclass
class CardinalityBarContext(SeriesBarChartContext):
    """Context/config for the cardinality bar plot (axes, sorting, thresholds)."""

    title_template: str = "Cardinality Check — Discrete vs. Continuous for {name}{modifiers}"
    xlabel: str = "Number of Records"
    ylabel: str = "Values (Top N)"
    show_subtitle: bool = True
    is_orientation_vertical: bool = False

    bar_height_source: Literal["values","counts"] = "counts"
    bar_sort_descending: bool = True

    # plot-specific
    max_unique_fraction: float = 0.05
    max_unique_values: int = 20
    integer_tolerance: float = 1e-8

class CardinalityBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Show how a numeric field’s mass is concentrated among its most frequent values.

    Why this matters:
        Cardinality (how many unique values) and frequency concentration inform whether a field
        should be treated as discrete or continuous and whether to bucket or keep as-is.

    What this plot does:
        Computes uniqueness and a discreteness heuristic, ranks value counts, displays the top N
        values (aggregating the tail into “Other”), and reports coverage of the named values.
    """

    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=numeric_validator()
        )
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return default placeholders for counts, uniqueness, and bars payload."""
        return {
            "params": {
                "max_unique_fraction": float(self.ctx.max_unique_fraction),
                "max_unique_values": int(self.ctx.max_unique_values),
                "integer_tolerance": float(self.ctx.integer_tolerance),
            },
            "total": 0,
            "nunique_native": 0,
            "uniqueness_ratio": 0.0,
            "coverage_named": 0.0,
            "is_discrete": None,
            "bars": {},
        }

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute value counts, uniqueness ratio, discreteness flag, and top-N coverage."""
        # 1) Build bars once (mixin handles totals, Other, capping, ratios)
        full_counts = s.value_counts().to_dict()
        desc = self.build_series_bar_desc(
            s,
            full_counts,
            denominator_key="pct_of_total",
            extra_params={
                "max_unique_fraction": float(self.ctx.max_unique_fraction),
                "max_unique_values": int(self.ctx.max_unique_values),
                "integer_tolerance": float(self.ctx.integer_tolerance),
            },
            skip_plot_if_zero=True,
        )

        # 2) Cardinality metrics (plot-specific)
        total_nonnull = int(desc.get("total_nonnull", 0))
        nunique_native = int(s.dropna().nunique())
        uniqueness_ratio = (nunique_native / total_nonnull) if total_nonnull else 0.0

        is_discrete = self._is_discrete_numeric(
            s,
            max_unique_fraction=self.ctx.max_unique_fraction,
            max_unique_values=self.ctx.max_unique_values,
            integer_tolerance=self.ctx.integer_tolerance,
        )

        # 3) Coverage of the *named* top bars (exclude aggregated “Other”)
        bars = desc.get("bars", {})
        other_display = (desc.get("params") or {}).get("other_display")
        named_counts = [
            int(bars[k]["count"]) for k in bars.keys()
            if (k != other_display)
        ]
        coverage_named = (sum(named_counts) / total_nonnull) if total_nonnull else 0.0

        # 4) Merge in cardinality stats
        desc.update({
            "nunique_native": nunique_native,
            "uniqueness_ratio": float(uniqueness_ratio),
            "coverage_named": float(coverage_named),
            "is_discrete": bool(is_discrete),
        })

        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Draft readable findings about discreteness, uniqueness, leaders, and coverage."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        nunique = int(desc.get("nunique_native", 0))
        if total_nonnull == 0 or nunique == 0:
            return {}

        ur = self.formatter.format_percent(float(desc.get("uniqueness_ratio", 0.0)))
        is_discrete = bool(desc.get("is_discrete"))
        params = desc.get("params") or {}
        max_k = int(params.get("max_display_bars") or 0)
        other_display = params.get("other_display")
        coverage = float(desc.get("coverage_named", 0.0))

        # ---- Context (add coverage only when a named cut exists and it's informative) ----
        ctx_parts = [f"N = {total_nonnull:,} non-null", f"{nunique:,} unique ({ur})"]
        has_named_cut = bool(other_display) or (max_k and nunique > max_k)
        if has_named_cut and coverage > 0:
            ctx_parts.append(f"Top-{max_k} named coverage: {self.formatter.format_percent(coverage)}")
        context = " | ".join(ctx_parts)

        findings = {
            "context": context,
            "primary_finding": "Variable behaves discrete." if is_discrete else "Variable behaves continuous.",
            "secondary_finding": None,
        }

        # ---- Secondary (only if discrete: who leads; stay concise) ----
        if is_discrete:
            bars = desc.get("bars", {})
            if bars:
                denom_k = desc.get("denominator_key", "pct_of_total")
                # exclude "Other"
                named = [(k, v) for k, v in bars.items() if k != other_display]
                if named:
                    max_share = max(float(v.get(denom_k, 0.0)) for _, v in named)
                    eps = max(1e-12, 1e-6 * max_share)
                    leaders = [k for k, v in named if abs(float(v.get(denom_k, 0.0)) - max_share) <= eps]
                    pct = self.formatter.format_percent(max_share)
                    if len(leaders) == 1:
                        findings["secondary_finding"] = f"Most frequent value {repr(leaders[0])} at {pct}."
                    else:
                        preview = ", ".join(repr(x) for x in leaders[:3])
                        more = f" +{len(leaders) - 3} more" if len(leaders) > 3 else ""
                        findings["secondary_finding"] = f"Top values (tie at {pct}): {preview}{more}."

        return findings

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Build a short subtitle summarizing discreteness, N, unique ratio, and coverage."""
        # Empty / invalid
        if not desc or desc.get("total_nonnull", 0) == 0:
            return ""

        is_discrete = bool(desc.get("is_discrete"))
        total_nonnull = int(desc.get("total_nonnull", 0))
        nunique = int(desc.get("nunique_native", 0))
        ur = self.formatter.format_percent(float(desc.get("uniqueness_ratio", 0.0)))
        coverage = float(desc.get("coverage_named", 0.0))
        max_k = int((desc.get("params") or {}).get("max_display_bars") or 0)
        other_display = (desc.get("params") or {}).get("other_display")

        # Lead with the classification (the plot’s big idea)
        lead = "Discrete" if is_discrete else "Continuous"
        subtitle = f"{lead} • Non-null: {total_nonnull:,} • Unique: {nunique:,} ({ur})"

        # Only show coverage when there is a named cut (top-N or “Other”)
        if (other_display or (max_k and nunique > max_k)) and coverage > 0:
            subtitle += f" • Top-{max_k} named coverage: {self.formatter.format_percent(coverage)}"

        return subtitle

    # ---- helper: discreteness ----
    @staticmethod
    def _is_discrete_numeric(
        s: pd.Series,
        max_unique_fraction: float = 0.05,
        max_unique_values: int = 20,
        integer_tolerance: float = 1e-8,
    ) -> bool:
        """
        Determine whether a numeric pandas Series should be treated as discrete.

        A series is considered discrete if:
        - It has an integer dtype and either:
            * The ratio of unique values to non-null entries is below `max_unique_fraction`, or
            * The total number of unique values is below `max_unique_values`.
        - It has a float dtype and either:
            * All values are within `integer_tolerance` of a whole number, or
            * Its unique-value ratio or count falls below the specified thresholds.

        Parameters
        ----------
        s : pd.Series
            Numeric data to evaluate. NaNs are ignored in all calculations.
        max_unique_fraction : float, default=0.05
            Maximum fraction of unique values (unique / total non-null) to still call discrete.
        max_unique_values : int, default=20
            Maximum absolute count of unique values to still call discrete.
        integer_tolerance : float, default=1e-8
            Tolerance for treating float values as effectively integers (e.g. 3.0000000001).

        Returns
        -------
        bool
            True if the series meets the criteria for discreteness; False otherwise.
        """
        # 1) Integer dtype
        if pd.api.types.is_integer_dtype(s.dtype):
            return (s.nunique() / len(s)) <= max_unique_fraction or s.nunique() < max_unique_values
        # 2) Float dtype
        if pd.api.types.is_float_dtype(s.dtype):
            # 2a. effectively all whole numbers?
            if np.isclose(s % 1, 0, atol=integer_tolerance).all():
                return True
            frac = s.nunique() / len(s)
            # 2b. low cardinality
            if frac < max_unique_fraction or s.nunique() < max_unique_values:
                return True
        return False
