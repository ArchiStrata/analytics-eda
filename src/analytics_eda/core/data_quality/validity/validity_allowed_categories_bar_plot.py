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
"""Validity pillar: allowed categories compliance and post-cleaning distinct counts."""

from collections.abc import Iterable
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
class ValidityAllowedCategoriesBarContext(SeriesBarChartContext):
    """Allowed-category validity check — how many categories fall outside the allowlist."""

    title_template: str = "Validity: Allowed Categories for {name}{modifiers}"
    xlabel: str = "Count"
    ylabel: str = "Measure"
    is_orientation_vertical: bool = False
    bar_sort_descending: bool = False
    show_subtitle: bool = True

    x_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="number", decimals=0))
    y_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="category"))

    allowed_categories: Iterable[str] | None = None
    case_sensitive_allowed: bool = False
    treat_empty_as_invalid: bool = True


class ValidityAllowedCategoriesBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Show how many categories violate an allowed list and how distinct counts change after filtering.

    Why this matters:
        Values outside business-approved categories break validation, joins, and downstream quality.
        Quantifying invalid categories and the post-cleanup distinct count clarifies data readiness.

    What this plot does:
        Counts distinct categories present, distinct invalid categories, and distinct categories after
        removing invalid entries, using a horizontal bar chart.
    """

    def __init__(self, ctx: ValidityAllowedCategoriesBarContext):
        parts = PlotParts(series_validator=named_only_validator(dropna=False, cast_str=False))
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version for this plot implementation."""
        return "1.0.0"

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute allowlist compliance counts."""
        total = int(s.size)
        nonnull_mask = ~s.isna()
        total_nonnull = int(nonnull_mask.sum())

        desc = self.default_descriptive()
        desc.update({"total": total, "total_nonnull": total_nonnull})

        if self.ctx.allowed_categories is None:
            desc.update({"skip_plot": True, "error": "allowed_categories not provided"})
            return desc

        if total_nonnull == 0:
            desc.update({"skip_plot": True, "error": "no non-null values"})
            return desc

        s_nonnull = s[nonnull_mask].astype("string")
        stripped = s_nonnull.str.strip()
        normalized = stripped if self.ctx.case_sensitive_allowed else stripped.str.lower()

        allowed = set(self.ctx.allowed_categories)
        if not self.ctx.case_sensitive_allowed:
            allowed = {str(a).strip().lower() for a in allowed}

        def _is_allowed(val: str | None) -> bool:
            if val is None:
                return False
            txt = str(val).strip()
            if self.ctx.treat_empty_as_invalid and txt == "":
                return False
            key = txt if self.ctx.case_sensitive_allowed else txt.lower()
            return key in allowed

        is_allowed_mask = stripped.map(_is_allowed)
        invalid_mask = ~is_allowed_mask

        values_with_invalid = int(invalid_mask.sum())
        invalid_categories = int(pd.unique(normalized[invalid_mask]).size)

        distinct_raw = int(pd.unique(normalized).size)
        distinct_after = int(pd.unique(normalized[is_allowed_mask]).size)

        counts = {
            "Distinct categories (raw)": distinct_raw,
            "Invalid categories": invalid_categories,
            "Distinct after removing invalid": distinct_after,
        }

        extra_params = {
            "allowed_categories_count": len(self.ctx.allowed_categories),
            "case_sensitive_allowed": bool(self.ctx.case_sensitive_allowed),
            "treat_empty_as_invalid": bool(self.ctx.treat_empty_as_invalid),
            "values_with_invalid": values_with_invalid,
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
        desc["values_with_invalid"] = values_with_invalid
        desc["subset_count"] = values_with_invalid
        desc["pct_subset"] = float(values_with_invalid / total_nonnull) if total_nonnull else 0.0
        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize invalid categories and distinct collapse."""
        if not desc:
            return {}

        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return {
                "context": "0 non-null values",
                "primary_finding": "The series is empty.",
                "secondary_finding": None,
            }

        values_with_invalid = int(desc.get("values_with_invalid", 0))
        distinct_raw = int(desc.get("distinct_raw", desc.get("bars", {}).get("Distinct categories (raw)", {}).get("count", 0)))
        distinct_after = int(desc.get("distinct_after", desc.get("bars", {}).get("Distinct after removing invalid", {}).get("count", 0)))
        collapse = int(desc.get("distinct_collapse", distinct_raw - distinct_after))

        context = f"N (non-null) = {total_nonnull:,}"

        if values_with_invalid == 0:
            return {
                "context": context,
                "primary_finding": "All values conform to the allowed categories; no invalid categories detected.",
                "secondary_finding": None,
            }

        primary = f"{values_with_invalid:,} values fall outside the allowed categories; distinct categories drop from {distinct_raw:,} to {distinct_after:,} (Δ = {collapse:,}) after filtering."

        return {
            "context": context,
            "primary_finding": primary,
            "secondary_finding": None,
        }

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Subtitle highlighting collapse after removing invalid categories."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return ""
        distinct_raw = int(desc.get("distinct_raw", desc.get("bars", {}).get("Distinct categories (raw)", {}).get("count", 0)))
        distinct_after = int(desc.get("distinct_after", desc.get("bars", {}).get("Distinct after removing invalid", {}).get("count", 0)))
        return f"Non-null: {total_nonnull:,} • Distinct categories: {distinct_raw:,} → {distinct_after:,}"
