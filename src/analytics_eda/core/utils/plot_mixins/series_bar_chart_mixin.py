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

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
import numpy as np
import pandas as pd

from analytics_eda.core.utils.base_plot import PlotContext

@dataclass
class SeriesBarChartContext(PlotContext):
    format_value_axis_as_percent: bool = True

    show_count_in_bar_label: bool = False
    show_value_in_bar_label: bool = True
    bar_height_source: Literal["values", "counts"] = "values"
    bar_sort_descending: bool = False
    bar_highlight_top: bool = True
    max_display_bars: Optional[int] = 15       # cap number of bars to display
    other_label: str = "Other"                 # label used when aggregating capped bars
    other_label_format: str = "{label} (k={k_agg})"


class SeriesBarChartMixin:
    """
    Shared helpers for Series-based bar charts.

    Responsibilities:
      1) Build reporting-friendly 'bars' dict from counts + an explicit denominator key.
      2) Cache arrays (labels, values, counts) for draw (no leakage into descriptive_stats).
      3) Provide a simple, consistent draft_descriptive_findings for 0/1/2+ bars.
      4) Provide a generic draw that respects orientation and optional top highlight.
    """

    # Defaults when empty
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "total": 0,
            "total_nonnull": 0,
            "total_count": 0,
            "bars": {},
        }

    # ---- bars helpers --------------------------------------------------------
    def build_series_bar_desc(
        self,
        s: pd.Series,
        counts: Dict[str, int],
        *,
        denominator_key: Literal["pct_of_nonnull", "pct_of_total", "pct_of_sum"] = "pct_of_nonnull",
        extra_params: Optional[Dict[str, Any]] = None,
        skip_plot_if_zero: bool = True,
    ) -> Dict[str, Any]:
        """
        Build reporting-friendly bar stats AND cache draw arrays.

        Returns a dict ready to merge/return as descriptive_stats:
        {
            "params": {...common bar params..., **extra_params},
            "total": int,
            "total_nonnull": int,
            "total_count": int,
            "denominator_key": str,
            "bars": {label: {"count": int, <denominator_key>: float, ["_is_other": bool]}},
            "k": int,         # non-zero categories (incl. Other if present and non-zero)
            "n_bars": int,    # bars displayed (post-capping)
            # Optional:
            # "skip_plot": True, "error": <str>
        }
        """
        total = int(s.size)
        total_nonnull = int((~s.isna()).sum())
        total_count = int(sum(int(v) for v in counts.values()))

        assert denominator_key in ("pct_of_nonnull", "pct_of_total", "pct_of_sum")
        if denominator_key == "pct_of_nonnull":
            denom = total_nonnull
        elif denominator_key == "pct_of_total":
            denom = total
        elif denominator_key == "pct_of_sum":
            denom = total_count
        else:
            raise ValueError(f"Unknown denominator_key: {denominator_key}")

        # Create (label, count, ratio) tuples
        items = [(k, int(v), float(v)/denom) for k, v in counts.items()]

        # cap & aggregate into "Other"
        max_display_bars = getattr(self.ctx, "max_display_bars", None)
        other_label_base = getattr(self.ctx, "other_label", "Other")
        top_items = items
        clipped: list[tuple[str, int, float]] = []
        if isinstance(max_display_bars, int) and max_display_bars > 0 and len(items) > max_display_bars:
            keep = max_display_bars - 1  # reserve a slot for Other
            top_items = items[:max(0, keep)]
            clipped = items[max(0, keep):]

        bars: Dict[str, Dict[str, float | int]] = {}
        for k, n, r in top_items:
            bars[k] = {"count": n, denominator_key: float(r)}

        other_display = None
        if clipped:
            other_count = int(sum(n for _, n, _ in clipped))
            other_ratio = float(other_count) / denom
            k_agg = len(clipped)

            # extended label with k
            other_display = f"{other_label_base} (k={k_agg})"
            bars[other_display] = {
                "count": other_count,
                denominator_key: other_ratio,
                "_is_other": True,
            }
            self._draw_set("series", "other_label", other_display)
        else:
            self._draw_set("series", "other_label", None)

        # cache arrays for draw
        labels = list(bars.keys())
        values = np.array([bars[k][denominator_key] for k in labels], dtype=float)
        counts_arr = np.array([bars[k]["count"] for k in labels], dtype=int)

        self._draw_set("series", "labels", labels)
        self._draw_set("series", "values", values)
        self._draw_set("series", "counts", counts_arr)
        self._draw_set("series", "value_key", denominator_key)

        desc = {
            "params": {
                "show_count_in_bar_label": bool(getattr(self.ctx, "show_count_in_bar_label", False)),
                "show_value_in_bar_label": bool(getattr(self.ctx, "show_value_in_bar_label", True)),
                "bar_height_source": getattr(self.ctx, "bar_height_source", "values"),
                "bar_sort_descending": bool(getattr(self.ctx, "bar_sort_descending", False)),
                "max_display_bars": max_display_bars,
                "other_label": other_label_base,
                **(extra_params or {}),
            },
            "total": total, # total series
            "total_nonnull": total_nonnull, # total nonnull in series
            "total_count": total_count, # total count displayed
            "pct_total_count": total_count / denom, # total count % of denominator
            "denominator_key": denominator_key,
            "bars": bars,
            "k": int(sum(1 for v in bars.values() if int(v["count"]) > 0)),
            "n_bars": int(len(bars)),
        }

        # skip plot conditions
        if skip_plot_if_zero and (total == 0 or total_count == 0):
            desc["skip_plot"] = True
            desc["error"] = "no categories to display"

        return desc

    # ---- generic draw --------------------------------------------------------
    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):
        """
        Draw cached bars from draw cache under key 'series' with fields:
        - labels: List[str]
        - values: List[float]  # typically 0..1 when plotting percents
        - counts: List[int]

        Behavior:
        - bar_height_source="values" -> plot percentages (0..1)
        - bar_height_source="counts" -> plot raw counts
        - show_value_in_bar_label shows the *primary* metric (the one plotted)
        - show_count_in_bar_label appends raw counts (if not already primary)
        """
        labels = self._draw_get("series", "labels") or []
        values = self._draw_get("series", "values")
        counts = self._draw_get("series", "counts")
        other_label = self._draw_get("series", "other_label")

        if values is None or counts is None:
            # nothing cached; nothing to draw
            return ax

        # Choose bar heights + primary formatter
        if self.ctx.bar_height_source == "counts":
            if counts is None:
                return ax
            heights = np.asarray(counts)
            def fmt_primary(v):  # count
                return f"{int(v):,}"
        else:  # "values" (default)
            if values is None:
                return ax
            heights = np.asarray(values, dtype=float)
            def fmt_primary(v):  # percent
                return f"{v*100:.1f}%"

        # Neutral base
        if self.ctx.is_orientation_vertical:
            bars = ax.bar(labels, heights, color=self.neutral_grey())
        else:
            bars = ax.barh(labels, heights, color=self.neutral_grey())

        # Highlight the top non-"Other" bar (by the primary metric)
        if self.ctx.bar_highlight_top and len(bars) > 0:
            other_label = self._draw_get("series", "other_label")
            # sort indices by height desc
            order = np.argsort(values)[::-1]
            for idx in order:
                if labels[idx] != other_label and values[idx] > 0:
                    bars[idx].set_color(palette[0])
                    break  # first valid winner only

        # Build edge labels
        def _label(h, c):
            parts = []
            if self.ctx.show_value_in_bar_label:
                parts.append(fmt_primary(h))
            # Only add count suffix if it's not already the primary metric
            if self.ctx.show_count_in_bar_label and self.ctx.bar_height_source != "counts" and c is not None:
                parts.append(f"(n={int(c):,})" if self.ctx.show_value_in_bar_label else f"n={int(c):,}")
            return " ".join(parts)

        # zip safely: if one list is shorter, zip truncates—so align by labels length
        if counts is None:
            counts_iter = [None] * len(heights)
        else:
            counts_iter = counts

        edge_labels = [_label(h, c) for h, c in zip(heights, counts_iter)]

        # Draw labels only if something to show
        if any(edge_labels):
            ax.bar_label(bars, labels=edge_labels, label_type="edge", padding=3, fontsize="small")

        return fig, ax
