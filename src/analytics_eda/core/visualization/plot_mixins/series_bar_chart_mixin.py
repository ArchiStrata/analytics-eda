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

from analytics_eda.core.visualization.base_plot import PlotContext

@dataclass
class SeriesBarChartContext(PlotContext):
    show_count_in_bar_label: bool = False
    show_value_in_bar_label: bool = True
    bar_height_source: Literal["values", "counts"] = "values"
    bar_sort_descending: bool = False
    bar_highlight_top: bool = True
    bar_top_n: int = 1 # how many top bars to highlight
    bar_top_include_ties: bool = True # include ties

    max_display_bars: Optional[int] = 15       # cap number of bars to display
    other_label: str = "Other"                 # label used when aggregating capped bars
    other_label_format: str = "{label} (k={k_agg})"

    # optional threshold-based aggregation (pre-cap)
    other_min_count: Optional[int] = None  # collapse categories with count < other_min_count into "Other"
    other_respect_existing: bool = True    # if 'Other' already in counts, merge into it instead of creating a second one


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
            "subset_count": 0,
            "bars": {},
        }

    # ---- bars helpers --------------------------------------------------------
    def build_series_bar_desc(
        self,
        s: pd.Series,
        counts: Dict[str, int],
        *,
        denominator_key: Literal["pct_of_nonnull", "pct_of_total"] = "pct_of_nonnull",
        extra_params: Optional[Dict[str, Any]] = None,
        skip_plot_if_zero: bool = True,
    ) -> Dict[str, Any]:
        """
        Build reporting-friendly bar statistics and cache draw arrays.

        Parameters
        ----------
        s : pd.Series
            Source series from which bars are derived (used to compute denominators, totals, etc.).
        counts : dict[str, int]
            Mapping from bar label -> count. This is typically a filtered/derived subset
            (e.g., rare categories, invalid tokens). Zero or negative values are coerced to int.
        denominator_key : {"pct_of_nonnull", "pct_of_total"}, default "pct_of_nonnull"
            Which base to use for per-bar ratios and `pct_subset`:
            - "pct_of_nonnull": denominator = number of non-null rows in `s`.
            - "pct_of_total"  : denominator = total length of `s` (including nulls).
        extra_params : dict | None
            Extra plot/context parameters to merge into `desc["params"]`.
        skip_plot_if_zero : bool, default True
            If True, set `skip_plot`/`error` when there is no data to display
            (`total == 0` or `subset_count == 0`).

        Returns
        -------
        desc : dict
            A dictionary suitable to merge/return as `descriptive_stats`:

            Core fields
            -----------
            params : dict
                Plot/mixin parameters merged with `extra_params`. Common keys include:
                - "show_count_in_bar_label" : bool
                - "show_value_in_bar_label" : bool
                - "bar_height_source"       : {"values", ...}
                - "bar_sort_descending"     : bool
                - "max_display_bars"        : int | None
                - "other_label"             : str
                (Domain-specific keys from callers may be present as well.)
            total : int
                Total number of rows in `s` (includes nulls).
            total_nonnull : int
                Number of non-null rows in `s`.
            subset_count : int
                Sum of all counts in the provided `counts` mapping (i.e., total rows represented
                by the displayed subset). Example: sum of rare-category counts.
            pct_subset : float
                Fraction of the chosen denominator represented by `subset_count`.
                `pct_subset = subset_count / denominator`, where the denominator is determined
                by `denominator_key`.
            denominator_key : str
                Echo of the denominator selection used for ratios ("pct_of_nonnull" | "pct_of_total").

            Bars payload
            ------------
            bars : dict[str, dict]
                Per-bar stats keyed by label. Each entry has:
                - "count" : int
                    The bar's absolute count.
                - <denominator_key> : float
                    The bar's ratio against the selected denominator.
                - "_is_other" : bool (optional)
                    Present only when an aggregated "Other" bar is created due to capping.
            unique_categories_total : int
                Number of distinct non-null category values in s (full series; pre-filter).
            input_categories : int
                Number of category labels provided in the raw `counts` mapping (pre-capping).
            input_nonzero_categories : int
                Number of category labels in `counts` with positive counts (pre-capping).
            nonzero_categories : int
                Number of bars with a positive count (includes "Other" if present and non-zero).
            n_bars_rendered : int
                Number of bars actually rendered (after capping/aggregation into "Other").

            Optional fields
            ---------------
            skip_plot : bool
                Present when `skip_plot_if_zero` is True and either `total == 0` or `subset_count == 0`.
            error : str
                Diagnostic message accompanying `skip_plot`.

        Notes
        -----
        - Capping & "Other": When `max_display_bars` is set and the number of input labels exceeds the cap,
        surplus labels are aggregated into a single "Other" bar. The label defaults to `other_label`
        (e.g., "Other" or "Other (k=...)") and the bar includes `"_is_other": True`.
        - Draw cache: The function caches arrays used for rendering (labels, values, counts, value_key).
        """
        total = int(s.size)
        total_nonnull = int((~s.isna()).sum())

        # normalize counts: string labels + int counts
        counts = {str(k): int(v) for k, v in counts.items()}

        input_categories = int(len(counts))  # raw labels provided
        input_nonzero_categories = int(sum(int(v) > 0 for v in counts.values()))
        subset_count = int(sum(int(v) for v in counts.values()))

        assert denominator_key in ("pct_of_nonnull", "pct_of_total")
        if denominator_key == "pct_of_nonnull":
            denom = total_nonnull
        elif denominator_key == "pct_of_total":
            denom = total
        else:
            raise ValueError(f"Unknown denominator_key: {denominator_key}")

        # ---- pre-aggregate small categories into 'Other' (threshold policy) ----
        other_label_base = getattr(self.ctx, "other_label", "Other")
        other_min = getattr(self.ctx, "other_min_count", None)
        respect_existing = bool(getattr(self.ctx, "other_respect_existing", True))
        has_existing_other = other_label_base in counts

        # track how many labels folded via the threshold
        threshold_fold_k = 0

        if isinstance(other_min, int) and other_min > 0:
            # do not consider the raw 'Other' key itself for thresholding
            small_keys = [k for k, v in counts.items()
                        if k != other_label_base and int(v) < other_min]
            threshold_fold_k = len(small_keys)
            small_sum = int(sum(int(counts[k]) for k in small_keys))
            for k in small_keys:
                counts.pop(k, None)
            if small_sum > 0:
                if respect_existing and has_existing_other:
                    counts[other_label_base] = int(counts.get(other_label_base, 0)) + small_sum
                else:
                    counts[other_label_base] = small_sum
                    has_existing_other = True

        # Create (label, count, ratio) tuples AFTER thresholding
        items = [(k, int(v), float(v)/denom) for k, v in counts.items()]

        # ---- Cap policy: aggregate overflow into a SINGLE displayed "Other" bar ----
        max_display_bars = getattr(self.ctx, "max_display_bars", None)
        other_label_fmt = getattr(self.ctx, "other_label_format", "{label} (k={k_agg})")

        # Exclude the raw Other key from ranking/clipping so it never gets dropped;
        # we’ll merge it into the displayed Other bar later.
        existing_other_count = int(counts.get(other_label_base, 0)) if has_existing_other else 0
        items_wo_other = [(k, n, r) for (k, n, r) in items if k != other_label_base]

        # Determine sort intent
        sort_desc = bool(getattr(self.ctx, "bar_sort_descending", False))
        # Choose metric to sort by: counts if primary is counts, else ratios
        primary_by_counts = getattr(self.ctx, "bar_height_source", "values") == "counts"
        if sort_desc:
            if primary_by_counts:
                # sort by count desc, then label for stability
                items_wo_other.sort(key=lambda kv: (-kv[1], kv[0]))
            else:
                # sort by ratio desc, then label for stability
                items_wo_other.sort(key=lambda kv: (-kv[2], kv[0]))
        else:
            if primary_by_counts:
                # sort by count asc, then label for stability
                items_wo_other.sort(key=lambda kv: (kv[1], kv[0]))
            else:
                # sort by ratio asc, then label for stability
                items_wo_other.sort(key=lambda kv: (kv[2], kv[0]))

        # Determine top items
        top_items = items_wo_other
        clipped: list[tuple[str, int, float]] = []
        if isinstance(max_display_bars, int) and max_display_bars > 0 and len(items) > max_display_bars:
            keep = max_display_bars - 1  # reserve 1 slot for displayed Other
            top_items = items_wo_other[:max(0, keep)]
            clipped = items_wo_other[max(0, keep):]

        # Assemble bars dict
        bars: Dict[str, Dict[str, float | int]] = {}
        for k, n, r in top_items:
            bars[k] = {"count": n, denominator_key: float(r)}

        other_display = None
        if clipped or existing_other_count > 0:
            # Merge any clipped tail + any pre-existing/raw Other into the displayed Other
            other_count = int(sum(n for _, n, _ in clipped)) + int(existing_other_count)
            other_ratio = float(other_count) / denom if denom > 0 else 0.0
            # k_agg counts how many labels were folded in: clipped tail + raw Other (if present)
            k_agg = threshold_fold_k + len(clipped)
            
            # Format the display label (e.g., "Other (k=3)") if we actually folded something in
            other_display = (other_label_fmt.format(label=other_label_base, k_agg=k_agg)
                            if k_agg > 0 else other_label_base)

            bars[other_display] = {
                "count": other_count,
                denominator_key: other_ratio,
                "_is_other": True,
                "k_agg": k_agg,
            }

        # cache arrays for draw
        labels = list(bars.keys())
        values = np.array([bars[k][denominator_key] for k in labels], dtype=float)
        counts_arr = np.array([bars[k]["count"] for k in labels], dtype=int)

        self.draw_cache_set("series", "labels", labels)
        self.draw_cache_set("series", "values", values)
        self.draw_cache_set("series", "counts", counts_arr)

        unique_categories_total = int(s.dropna().astype("object").nunique())

        # Determine the primary metric used for bar height
        bar_height_source = getattr(self.ctx, "bar_height_source", "values")
        if bar_height_source == "counts":
            primary = counts_arr.astype(float)
        else:
            primary = values.astype(float)

        # Compute top labels according to context
        top_n = max(1, int(getattr(self.ctx, "bar_top_n", 1)))
        include_ties = bool(getattr(self.ctx, "bar_top_include_ties", True))

        label_to_metric = {lbl: float(m) for lbl, m in zip(labels, primary)}

        # Exclude "Other" from consideration
        candidates = [(lbl, label_to_metric[lbl]) for lbl in labels if lbl != other_display]
        # Sort descending by metric, stable second key = label for determinism
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))

        if not candidates:
            top_labels = []
        else:
            if include_ties:
                # metric cutoff at rank top_n (1-indexed)
                cutoff_idx = min(top_n, len(candidates)) - 1
                cutoff_val = candidates[cutoff_idx][1]
                top_labels = [lbl for lbl, m in candidates if m >= cutoff_val and m > 0]
            else:
                top_labels = [lbl for lbl, m in candidates[:top_n] if m > 0]

        desc = {
            "params": {
                "show_count_in_bar_label": bool(getattr(self.ctx, "show_count_in_bar_label", False)),
                "show_value_in_bar_label": bool(getattr(self.ctx, "show_value_in_bar_label", True)),
                "bar_height_source": getattr(self.ctx, "bar_height_source", "values"),
                "bar_sort_descending": bool(getattr(self.ctx, "bar_sort_descending", False)),
                "max_display_bars": max_display_bars,
                "other_label": other_label_base,
                "other_label_format": other_label_fmt,
                "other_display": other_display,
                "other_min_count": other_min,
                "other_respect_existing": respect_existing,
                "bar_top_n": top_n,
                "bar_top_include_ties": include_ties,
                **(extra_params or {}),
            },
            "total": total, # total series
            "total_nonnull": total_nonnull, # total nonnull in series
            "subset_count": subset_count, # subset count
            "pct_subset": (subset_count / denom) if denom > 0 else 0.0, # subset count % of denominator
            "denominator_key": denominator_key,
            "bars": bars,
            "unique_categories_total": unique_categories_total,
            "input_categories": input_categories,                   # size of counts (pre-capping)
            "input_nonzero_categories": input_nonzero_categories,   # positive-count labels (pre-capping)
            "nonzero_categories": int(sum(1 for v in bars.values() if int(v["count"]) > 0)), # post-capping
            "n_bars_rendered": int(len(bars)), # post-capping
            "top_labels": top_labels,
        }

        # skip plot conditions
        if skip_plot_if_zero and (total == 0 or subset_count == 0):
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
        labels = self.draw_cache_get("series", "labels") or []
        values = self.draw_cache_get("series", "values")
        counts = self.draw_cache_get("series", "counts")

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
                return self.formatter.format_percent(float(v))

        # Neutral base
        if self.ctx.is_orientation_vertical:
            bars = ax.bar(labels, heights, color=self.neutral_grey())
        else:
            bars = ax.barh(labels, heights, color=self.neutral_grey())
            ax.invert_yaxis()  # ensure first label (often highest) is displayed at the top

        # Highlight all top labels (already excludes "Other")
        if self.ctx.bar_highlight_top and len(bars) > 0:
            winners = set(desc.get("top_labels", []))
            for i, lbl in enumerate(labels):
                if lbl in winners:
                    bars[i].set_color(palette[0])

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
            self.bar_label(ax, bars, labels=edge_labels, label_type="edge", padding=3, fontsize="small")

        return fig, ax

    def bar_label(self, ax, container, *args, **kwargs):
        """
        Call ax.bar_label and automatically register the Text objects so BasePlot
        can compute headroom. Use this in bar plots instead of ax.bar_label.
        """
        texts = ax.bar_label(container, *args, **kwargs)
        # BasePlot provides register_annotations
        if hasattr(self, "register_annotations"):
            self.register_annotations(ax, texts)
        return texts
