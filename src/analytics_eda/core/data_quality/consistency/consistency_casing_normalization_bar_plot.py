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
"""Consistency pillar: casing normalization impact."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analytics_eda.core.visualization.base_plot import BasePlot
from analytics_eda.core.visualization.context.plot_context import AxisFormat
from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import (
    SeriesBarChartContext,
    SeriesBarChartMixin,
)
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import named_only_validator


@dataclass
class ConsistencyCasingNormalizationBarContext(SeriesBarChartContext):
    """Casing normalization impact — before vs. after canonicalizing case."""

    title_template: str = "Casing Normalization Impact for {name}{modifiers}"
    xlabel: str = "Count"
    ylabel: str = "Measure"
    is_orientation_vertical: bool = False
    bar_sort_descending: bool = False
    show_subtitle: bool = True

    x_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="number", decimals=0))
    y_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="category"))


class ConsistencyCasingNormalizationBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Show how case canonicalization changes category cleanliness and cardinality.

    Why this matters:
        Case variants bloat categories and hide true counts. Measuring the impact of
        lowercasing/trim-based canonicalization clarifies normalization benefits.

    What this plot does:
        Counts distinct categories before normalization, categories that collide on
        lowercase/trimmed forms, and distinct categories after normalization, using
        a horizontal bar chart.
    """

    def __init__(self, ctx: ConsistencyCasingNormalizationBarContext):
        parts = PlotParts(series_validator=named_only_validator(dropna=False, cast_str=False))
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version for this plot implementation."""
        return "1.0.0"

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute counts for casing normalization impact."""
        total = int(s.size)
        nonnull_mask = ~s.isna()
        total_nonnull = int(nonnull_mask.sum())

        desc = self.default_descriptive()
        desc.update({"total": total, "total_nonnull": total_nonnull})

        if total_nonnull == 0:
            desc.update({"skip_plot": True, "error": "no non-null values"})
            return desc

        s_nonnull = s[nonnull_mask].astype("string")
        stripped = s_nonnull.str.strip()
        lowered = stripped.str.lower()

        distinct_raw = int(pd.unique(stripped).size)
        distinct_after = int(pd.unique(lowered).size)

        # Identify canonical forms and variants per canonical
        df = pd.DataFrame({"lower": lowered, "orig": stripped})
        freq = df.groupby(["lower", "orig"], dropna=False).size().rename("n").reset_index()
        variant_counts = freq.groupby("lower")["orig"].nunique()
        collision_canonicals = set(variant_counts[variant_counts > 1].index)

        impacted_mask = lowered.map(lambda v: v in collision_canonicals)
        categories_with_collisions = int(pd.unique(stripped[impacted_mask]).size)
        values_with_collisions = int(impacted_mask.sum())

        counts = {
            "Distinct categories (raw)": distinct_raw,
            "Categories with casing collisions": categories_with_collisions,
            "Distinct after case normalization": distinct_after,
        }

        extra_params = {
            "values_with_collisions": values_with_collisions,
            "distinct_raw": distinct_raw,
            "distinct_after": distinct_after,
        }

        desc = self.build_series_bar_desc(
            s,
            counts,
            denominator_key="pct_of_nonnull",
            extra_params=extra_params,
            skip_plot_if_zero=False,
        )

        desc["distinct_collapse"] = int(distinct_raw - distinct_after)
        desc["distinct_raw"] = distinct_raw
        desc["distinct_after"] = distinct_after
        desc["values_with_collisions"] = values_with_collisions
        if total_nonnull:
            desc["subset_count"] = values_with_collisions
            desc["pct_subset"] = float(values_with_collisions / total_nonnull)
        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize casing normalization impact."""
        if not desc:
            return {}

        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return {
                "context": "0 non-null values",
                "primary_finding": "The series is empty.",
                "secondary_finding": None,
            }

        values_with_collisions = int(desc.get("values_with_collisions", 0))
        distinct_raw = int(desc.get("distinct_raw", desc.get("bars", {}).get("Distinct categories (raw)", {}).get("count", 0)))
        distinct_after = int(desc.get("distinct_after", desc.get("bars", {}).get("Distinct after case normalization", {}).get("count", 0)))
        collapse = int(desc.get("distinct_collapse", distinct_raw - distinct_after))

        context = f"N (non-null) = {total_nonnull:,}"

        if values_with_collisions == 0:
            return {
                "context": context,
                "primary_finding": "Casing is already consistent; no collisions detected.",
                "secondary_finding": None,
            }

        primary = f"{values_with_collisions:,} values participate in casing collisions; distinct categories drop from {distinct_raw:,} to {distinct_after:,} (Δ = {collapse:,})."

        return {
            "context": context,
            "primary_finding": primary,
            "secondary_finding": None,
        }

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Subtitle highlighting collapse after case normalization."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return ""
        distinct_raw = int(desc.get("distinct_raw", desc.get("bars", {}).get("Distinct categories (raw)", {}).get("count", 0)))
        distinct_after = int(desc.get("distinct_after", desc.get("bars", {}).get("Distinct after case normalization", {}).get("count", 0)))
        return f"Non-null: {total_nonnull:,} • Distinct categories: {distinct_raw:,} → {distinct_after:,}"
