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
"""Format Consistency Bar Plot — show how values cluster into format patterns."""

from dataclasses import dataclass
import datetime
import re
from typing import Any

import numpy as np
import pandas as pd

from analytics_eda.core.visualization.base_plot import BasePlot
from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import (
    SeriesBarChartContext,
    SeriesBarChartMixin,
)
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import named_only_validator


@dataclass
class ConsistencyFormatConsistencyBarContext(SeriesBarChartContext):
    """Context for consistency format consistency bar plot."""

    title_template: str = "Format Consistency for {name}{modifiers}"
    xlabel: str = "Percent of total"
    ylabel: str = "Format group"
    is_orientation_vertical: bool = False
    bar_sort_descending: bool = True
    show_subtitle: bool = True


class ConsistencyFormatConsistencyBarPlot(SeriesBarChartMixin, BasePlot):
    r"""
    Reveal whether values share one format or are fragmented across patterns.

    Why this matters:
        Mixed formats (e.g., 1, 1.0, \"01\", \"1,000\") drive parsing errors and inconsistent
        encodings. Quantifying format clusters makes cleanup requirements explicit.

    What this plot does:
        Buckets values into simple format groups (numeric variants, dates, alphabetic, whitespace,
        nulls), measures their share of the column, and highlights the dominant pattern.
    """

    def __init__(self, ctx: ConsistencyFormatConsistencyBarContext):
        parts = PlotParts(series_validator=named_only_validator(dropna=False, cast_str=False))
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version for this plot implementation."""
        return "1.0.0"

    def default_descriptive(self) -> dict[str, Any]:
        """Return default descriptive stats with dominant format placeholders."""
        base = SeriesBarChartMixin.default_descriptive(self)
        base.update(
            {
                "dominant_format": None,
                "dominant_ratio": 0.0,
                "format_counts": {},
            }
        )
        return base

    @staticmethod
    def _is_datetime_value(v: Any) -> bool:
        return isinstance(v, pd.Timestamp | datetime.datetime | datetime.date | np.datetime64)

    @staticmethod
    def _decimal_places(text: str) -> int:
        match = re.match(r"-?\d+[.,](\d+)$", text)
        return len(match.group(1)) if match else 0

    def _format_label(self, v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "Null / Missing"

        if isinstance(v, bool | np.bool_):
            return "Boolean literal"

        if self._is_datetime_value(v):
            return "Datetime value"

        # Native numerics
        if isinstance(v, int | float | np.integer | np.floating) and not isinstance(v, bool):
            return "Numeric (native float)" if isinstance(v, float | np.floating) else "Numeric (native int)"

        text = str(v)
        stripped = text.strip()
        if stripped == "":
            return "Empty / blank string"
        if stripped != text:
            return "Whitespace padded"

        # String-based patterns
        if re.fullmatch(r"-?\d+", stripped):
            return "Numeric string (int)"
        if re.fullmatch(r"-?\d+[.,]\d+", stripped):
            dp = self._decimal_places(stripped)
            return f"Numeric string (decimal, {dp}dp)" if dp else "Numeric string (decimal)"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
            return "Date string (YYYY-MM-DD)"
        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", stripped):
            return "Date string (MM/DD/YYYY)"
        if re.fullmatch(r"[A-Za-z]+", stripped):
            return "Alphabetic string"
        if re.fullmatch(r"[A-Za-z0-9]+", stripped):
            return "Alphanumeric string"

        return "Other string"

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute format cluster counts and dominance."""
        counts: dict[str, int] = {}
        for v in s:
            label = self._format_label(v)
            counts[label] = counts.get(label, 0) + 1

        desc = self.build_series_bar_desc(
            s,
            counts,
            denominator_key="pct_of_total",
            extra_params={},
        )

        # Dominant format among non-null/blank entries
        total_nonnull = int((~s.isna()).sum())
        non_null_counts = {k: v for k, v in counts.items() if k != "Null / Missing"}
        if non_null_counts and total_nonnull:
            dominant_format, dom_count = max(non_null_counts.items(), key=lambda kv: kv[1])
            desc.update(
                {
                    "dominant_format": dominant_format if dom_count > 0 else None,
                    "dominant_ratio": float(dom_count / total_nonnull),
                    "format_counts": non_null_counts,
                    "total_nonnull": total_nonnull,
                }
            )
        else:
            desc.update({"total_nonnull": total_nonnull})

        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize whether one format dominates or fragmentation exists."""
        total = int(desc.get("total", 0))
        total_nonnull = int(desc.get("total_nonnull", 0))
        dominant_format = desc.get("dominant_format")
        dom_ratio = float(desc.get("dominant_ratio", 0.0))

        if total == 0:
            return {
                "context": "0 total values",
                "primary_finding": "The series is empty.",
                "secondary_finding": None,
            }

        context = f"{total_nonnull:,} non-null of {total:,} total"
        if total_nonnull == 0:
            return {
                "context": context,
                "primary_finding": "No non-null values; format consistency cannot be assessed.",
                "secondary_finding": None,
            }

        findings = {
            "context": context,
            "primary_finding": "",
            "secondary_finding": None,
        }

        if dominant_format and dom_ratio >= 0.9:
            findings["primary_finding"] = f"Values follow a single dominant format: {dominant_format} ({self.formatter.format_percent(dom_ratio)} of non-null)."
        else:
            findings["primary_finding"] = "Formats are fragmented; no single pattern exceeds 90% of non-null."

        return findings

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Show dominant format coverage in subtitle."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return ""
        dominant_format = desc.get("dominant_format") or "None"
        dom_pct = self.formatter.format_percent(float(desc.get("dominant_ratio", 0.0)))
        return f"Non-null: {total_nonnull:,} • Dominant format: {dominant_format} ({dom_pct})"
