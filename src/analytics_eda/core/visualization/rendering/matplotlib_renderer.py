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
"""Default Matplotlib renderer for Analytics-EDA visualizations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from matplotlib.dates import AutoDateFormatter, AutoDateLocator, DateFormatter
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FormatStrFormatter,
    Formatter,
    FuncFormatter,
    PercentFormatter,
)
import seaborn as sns

from analytics_eda.core.visualization.context.plot_context import AxisFormat
from analytics_eda.core.visualization.rendering.renderer_protocol import (
    DrawFnSeries,
    RendererProtocol,
)


@dataclass
class DefaultMatplotlibRenderer(RendererProtocol):
    """Matplotlib/seaborn renderer.

    Mirrors existing BasePlot draw-time behavior.
    """

    # ---------- figure & palette ----------
    def _new_figure_and_palette(self, ctx):
        fig, ax = plt.subplots(figsize=getattr(ctx, "figsize", (6, 4)), dpi=getattr(ctx, "dpi", 100))
        palette = sns.color_palette("colorblind")
        ax.set_prop_cycle(color=palette)
        return fig, ax, palette

    # ---------- metadata ----------
    def _apply_metadata_to_axes(self, ax, chart_metadata: dict[str, Any]):
        if chart_metadata.get("title"):
            ax.title_ref = ax.set_title(chart_metadata["title"], pad=14, fontsize=12, fontweight="bold")
        if chart_metadata.get("xlabel"):
            ax.set_xlabel(chart_metadata["xlabel"])
        if chart_metadata.get("ylabel"):
            ax.set_ylabel(chart_metadata["ylabel"])

    # ---------- subtitle queue ----------
    def __post_init__(self):
        """Initialize internal queues after dataclass construction."""
        self._subtitle_queue: list[tuple] = []
        self._headroom_texts: dict = {}

    def _apply_subtitle_below_title(self, show_subtitle: bool, ax, subtitle: str, *, fontsize: int = 10, color: str = "gray", gap_from_axes_pts: float = 1.5):
        if not subtitle:
            return

        if show_subtitle and subtitle.strip():
            self._subtitle_queue.append((ax, subtitle, fontsize, color, gap_from_axes_pts))

    # ---------- annotation/headroom ----------
    def register_annotations(self, ax, texts):
        """Register text artists for headroom measurement.

        Safe to call with a single Text, a list/tuple of Texts, or nested lists.
        """
        if texts is None:
            return
        if isinstance(texts, list | tuple):
            flat = []
            stack = list(texts)
            while stack:
                t = stack.pop()
                if t is None:
                    continue
                if isinstance(t, list | tuple):
                    stack.extend(t)
                else:
                    flat.append(t)
        else:
            flat = [texts]
        if flat:
            self._headroom_texts.setdefault(ax, []).extend(flat)

    def _gather_headroom_texts(self, ax):
        # Prefer explicitly registered texts; fall back to visible axis texts.
        texts = self._headroom_texts.get(ax, [])
        return texts if texts else [t for t in ax.texts if t.get_visible()]

    def _measure_overhang_in_data(self, ax):
        """Compute text overhang in data units.

        Returns (left_oh, right_oh, bottom_oh, top_oh) from registered text.
        """
        fig = ax.figure
        # ensure layout is finalized for accurate text extents
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()

        lo_x, hi_x = ax.get_xlim()
        lo_y, hi_y = ax.get_ylim()

        left_oh = right_oh = bottom_oh = top_oh = 0.0
        for t in self._gather_headroom_texts(ax):
            bb = t.get_window_extent(renderer=renderer)
            (x0, y0) = inv.transform((bb.x0, bb.y0))
            (x1, y1) = inv.transform((bb.x1, bb.y1))
            left, right = min(x0, x1), max(x0, x1)
            bottom, top = min(y0, y1), max(y0, y1)

            if left < lo_x:
                left_oh = max(left_oh, lo_x - left)
            if right > hi_x:
                right_oh = max(right_oh, right - hi_x)
            if bottom < lo_y:
                bottom_oh = max(bottom_oh, lo_y - bottom)
            if top > hi_y:
                top_oh = max(top_oh, top - hi_y)

        return left_oh, right_oh, bottom_oh, top_oh

    def _ensure_headroom(self, ctx, ax, *, label_offset=0.02, extra_pad=0.04, max_extra=0.20):
        """Apply text-aware headroom to the value axis.

        Expands limits using measured text overhang; preserves symmetry if requested.
        """
        if not getattr(ctx, "auto_headroom", True):
            return

        val_is_y = bool(getattr(ctx, "is_orientation_vertical", True))
        axis_pref = getattr(ctx, "headroom_axis", "auto") if hasattr(ctx, "headroom_axis") else "auto"
        if axis_pref == "x":
            val_is_y = False
        elif axis_pref == "y":
            val_is_y = True

        preserve_sym = getattr(ctx, "headroom_preserve_symmetry", False)
        use_text = bool(getattr(ctx, "headroom_use_text_extents", True)) if hasattr(ctx, "headroom_use_text_extents") else True

        lo_x, hi_x = ax.get_xlim()
        lo_y, hi_y = ax.get_ylim()

        left_oh = right_oh = bottom_oh = top_oh = 0.0
        if use_text:
            left_oh, right_oh, bottom_oh, top_oh = self._measure_overhang_in_data(ax)

        if val_is_y:
            new_lo = lo_y - bottom_oh
            new_hi = hi_y + top_oh + extra_pad
            if preserve_sym:
                half = max(abs(new_lo), abs(new_hi))
                half = min(half, (1.0 + max_extra) * max(abs(lo_y), abs(hi_y)))
                ax.set_ylim(-half, half)
            else:
                ax.set_ylim(min(lo_y, new_lo), max(hi_y, new_hi))
        else:
            new_lo = lo_x - left_oh
            new_hi = hi_x + right_oh + extra_pad + label_offset
            if preserve_sym:
                half = max(abs(new_lo), abs(new_hi))
                half = min(half, (1.0 + max_extra) * max(abs(lo_x), abs(hi_x)))
                ax.set_xlim(-half, half)
            else:
                ax.set_xlim(min(lo_x, new_lo), max(hi_x, new_hi))

    # ---------- legend ----------
    def _apply_legend(self, ax, enable: bool):
        """Show or hide the legend based on ctx.enable_legend.

        If False, remove an existing legend (if any).
        """
        if enable:
            ax.legend()
        else:
            lg = ax.get_legend()
            if lg is not None:
                lg.remove()

    # ---------- footer summary ----------

    def _apply_footer_summary(self, show_footer_summary: bool, fig, footer_text):
        if show_footer_summary and footer_text.strip():
            fig.text(0.99, 0.01, footer_text, ha="right", va="bottom", fontsize="small", color="gray")

    # ---------- colors ----------
    def neutral_grey(self, variant: str = "medium", alpha: float | None = None):
        """
        Return a colorblind-friendly neutral grey for de-emphasis ("move to background").

        Variants:
          - "light"  (~82% gray):  good for gridlines / subtle guides
          - "medium" (~69% gray):  good for secondary bars/lines/labels (default)
          - "dark"   (~43% gray):  good for text on light backgrounds

        Returns a Matplotlib-compatible color:
          - hex string if alpha is None
          - RGBA tuple if alpha is provided (0..1)
        """
        presets = {
            "light": "#D0D0D0",  # ~82% gray
            "medium": "#B0B0B0",  # ~69% gray (default)
            "dark": "#6E6E6E",  # ~43% gray
        }
        hex_color = presets.get(variant, presets["medium"])
        if alpha is None:
            return hex_color
        # Convert hex to normalized RGBA with requested alpha
        h = hex_color.lstrip("#")
        r, g, b = tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b, float(alpha))

    # ---------- axis format -----------
    def _apply_axis_format(self, axis, fmt: AxisFormat):
        # Custom formatter wins
        if fmt.formatter is not None:
            if isinstance(fmt.formatter, Formatter):
                axis.set_major_formatter(fmt.formatter)
            else:
                axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.formatter(v)))
            return

        # Auto: do nothing (let Matplotlib pick)
        if fmt.kind == "auto":
            return

        if fmt.kind == "percent":
            # If data are 0..1 proportions
            if fmt.percent_scale_0to1:
                axis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=fmt.decimals))
            else:  # data already 0..100
                axis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=fmt.decimals))
            return

        if fmt.kind == "number" or fmt.kind == "currency":
            # Build a numeric format string
            # e.g., "{:,.2f}" or "{:.0f}"
            sep = "," if fmt.thousands_sep else ""
            decimals = fmt.decimals if fmt.decimals is not None else 0
            base_spec = f"{{:{sep}.{decimals}f}}"

            if fmt.kind == "currency":
                code = (fmt.currency_code or "").strip()
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{code} {base_spec.format(v)}".strip()))
            else:
                if fmt.unit_suffix:
                    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{base_spec.format(v)} {fmt.unit_suffix}"))
                else:
                    axis.set_major_formatter(FormatStrFormatter(base_spec))
            return

        if fmt.kind == "datetime":
            # Let Matplotlib handle with AutoDateFormatter if no explicit format given
            if fmt.datetime_format:
                axis.set_major_formatter(DateFormatter(fmt.datetime_format))
            else:
                axis.set_major_formatter(AutoDateFormatter(AutoDateLocator()))
            return

        if fmt.kind == "category":
            # Usually categorical ticks set by plotting function; no formatter required.
            return

    # ---------- finalization ----------
    def _finalize(self, ctx, fig, ax, chart_md: dict[str, Any]) -> str | None:
        """Add source, layout, save/show; return saved file name (or None)."""
        # Ensure all positions are computed
        fig.canvas.draw()

        # headroom
        self._ensure_headroom(
            ctx,
            ax,
            label_offset=getattr(ctx, "headroom_label_offset", 0.02),
            extra_pad=getattr(ctx, "headroom_extra_pad", 0.04),
            max_extra=getattr(ctx, "headroom_max_extra", 0.20),
        )

        # Draw tight layout
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig.canvas.draw()

        # place queued subtitles
        if getattr(self, "_subtitle_queue", None):
            renderer = fig.canvas.get_renderer()
            fig_w_px, fig_h_px = fig.bbox.width, fig.bbox.height

            for ax_, subtitle, fontsize, color, gap_pts in self._subtitle_queue:
                t = getattr(ax_, "title_ref", None)
                if t is not None:
                    tb = t.get_window_extent(renderer=renderer)
                    # convert a gap in points to pixels
                    gap_px = gap_pts * fig.dpi / 72.0
                    # position centered on the title horizontally, just below title
                    x_px = (tb.x0 + tb.x1) / 2.0
                    y_px = tb.y0 + gap_px
                    # convert to figure coordinates (0..1)
                    xf, yf = x_px / fig_w_px, y_px / fig_h_px
                    fig.text(xf, yf, subtitle, ha="center", va="top", fontsize=fontsize, color=color)
            self._subtitle_queue.clear()

        if getattr(ctx, "data_source", None):
            fig.text(0.01, 0.01, f"Source: {ctx.data_source}", ha="left", va="bottom", fontsize="small", color="gray")

        saved_name = None
        if getattr(ctx, "base_dir", None):
            saved_name = getattr(ctx, "file_name", None) or chart_md.get("file_name")
            if not saved_name and getattr(ctx, "auto_file_name", False):
                saved_name = f'{chart_md.get("title","figure")}.png'
            if saved_name:
                os.makedirs(ctx.base_dir, exist_ok=True)
                fig.savefig(os.path.join(ctx.base_dir, saved_name), bbox_inches="tight", dpi=getattr(ctx, "dpi", None))
        if getattr(ctx, "show", False):
            plt.show()
        return saved_name

    # ---------- single entry point used by BasePlot ----------
    def render(
        self,
        *,
        ctx,
        chart_md: dict[str, Any],
        desc: dict[str, Any],
        inf: dict[str, Any],
        draw_fn: DrawFnSeries,
        subtitle_text: str | None,
        footer_text: str | None,
    ) -> str | None:
        """Render the chart and optionally save it.

        Returns the saved filename if persisted, otherwise `None`.
        """
        fig = None
        try:
            fig, ax, palette = self._new_figure_and_palette(ctx)
            self._apply_metadata_to_axes(ax, chart_md)

            self._apply_subtitle_below_title(getattr(ctx, "show_subtitle", False), ax, subtitle_text)

            # call the plot-specific drawing callback
            fig, ax = draw_fn(desc, inf, chart_md, fig, ax, palette)

            self._apply_axis_format(ax.xaxis, ctx.x_format)
            self._apply_axis_format(ax.yaxis, ctx.y_format)

            self._apply_footer_summary(getattr(ctx, "show_footer_summary", False), fig, footer_text)

            self._apply_legend(ax, bool(getattr(ctx, "enable_legend", False)))

            saved = self._finalize(ctx, fig, ax, chart_md)
            return saved
        finally:
            if fig:
                plt.close(fig)
