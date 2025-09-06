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

from abc import ABC
from collections import defaultdict
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import pandas as pd

from analytics_eda.core.visualization.context.plot_context import PlotContext
from analytics_eda.core.visualization.plot_parts import PlotParts

Desc = Union[str, Sequence[str], None]

class BasePlot(ABC):
    def __init__(self, ctx: PlotContext, parts: Optional[PlotParts] = None):
        self.ctx = ctx
        self.parts = parts or PlotParts()
        self._subtitle_queue: list[tuple] = []
        self._draw_cache: dict[str, dict[str, Any]] = {}   # per-run cache (cleared each run)

    def default_descriptive(self) -> Dict[str, Any]:
        return {}
    
    def default_inferential(self) -> Dict[str, Any]:
        return {}
    
    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return draft, human-readable statements derived from descriptive stats.
        Child plots may override. Default: {}.
        """
        return {}

    def draft_inferential_findings(self, inf: Dict[str, Any], desc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return draft, human-readable statements derived from inferential stats
        (and optionally descriptive context). Child plots may override. Default: {}.
        """
        return {}

    # ======== OPTIONAL SERIES API (only implement in univariate plots) ========

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        raise NotImplementedError("Series-based compute descriptive not implemented for this plot.")

    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def draw(
        self,
        s: pd.Series,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
        *,
        fig,
        ax,
        palette,
    ):
        raise NotImplementedError("Series-based draw not implemented for this plot.")

    # ======== OPTIONAL FRAME API (only implement in new bi/multivariate plots) ========

    def compute_descriptive_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Optional[Mapping[str,str]] = None) -> Dict[str, Any]:
        raise NotImplementedError("Frame-based descriptive not implemented for this plot.")

    def compute_inferential_frame(self, df: pd.DataFrame, desc: Dict[str, Any], *, cols: Sequence[str], role_map: Optional[Mapping[str,str]] = None) -> Dict[str, Any]:
        return {}

    def draw_frame(
        self,
        df: pd.DataFrame,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str,str]] = None,
        fig=None,
        ax=None,
        palette=None,
    ):
        raise NotImplementedError("Frame-based draw not implemented for this plot.")
    # ======== Metadata & Title (works for both) ========
    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.

        Defaults to "0.1.0". Child plots may override or extend this to
        indicate changes in functionality or output schema.
        """
        return "0.1.0"
    
    def _build_chart_title(
        self,
        *,
        series: pd.Series | None = None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
    ) -> str:
        """
        Build the full chart title using self.ctx and optional plot-provided overrides.

        Template placeholders available:
          {name}, {modifiers}, {filter}, {transform}, {fit},
          {xlabel}, {ylabel}, plus any extra keys returned by title_kwargs()
          and/or ctx.title_fmt (e.g., {top_k}).
        """
        # --- label precedence: ctx.name > role_map label > joined cols > series.name > "Value"
        def _label_from_roles(role_map_local: Optional[Mapping[str, str]]) -> Optional[str]:
            if not role_map_local:
                return None
            y = role_map_local.get("y"); x = role_map_local.get("x"); hue = role_map_local.get("hue")
            parts = []
            if y and x: parts.append(f"{y} by {x}")
            elif y: parts.append(y)
            elif x: parts.append(x)
            if hue: parts.append(hue)
            return " • ".join(parts) if parts else None

        role_label = _label_from_roles(role_map)
        joined_cols = " • ".join(cols) if cols else None
        base_label = (
            self.ctx.name
            or role_label
            or joined_cols
            or (getattr(series, "name", None) if series is not None else None)
            or "Value"
        )

        # --- subclass-provided overrides / extras (desc strings + extra fmt placeholders)
        tk = dict(self.title_kwargs(series=series, cols=cols, role_map=role_map) or {})
        filter_desc    = tk.pop("filter_desc",    getattr(self.ctx, "filter_desc", None))
        transform_desc = tk.pop("transform_desc", getattr(self.ctx, "transform_desc", None))
        fit_desc       = tk.pop("fit_desc",       getattr(self.ctx, "fit_desc", None))
        extra_desc     = tk.pop("extra_desc",     getattr(self.ctx, "extra_desc", None))

        def _to_list(x: Desc) -> list[str]:
            if x is None: return []
            if isinstance(x, str): return [x] if x else []
            return [str(v) for v in x if v]

        # modifiers in canonical order
        parts = _to_list(filter_desc) + _to_list(transform_desc) + _to_list(fit_desc) + _to_list(extra_desc)
        modifiers = f" ({', '.join(parts)})" if parts else ""

        # fmt dictionary: ctx fmt first, then subclass placeholders override
        ctx_fmt = getattr(self.ctx, "title_fmt", None) or {}
        values = defaultdict(str, {**ctx_fmt, **tk})

        # always expose xlabel / ylabel for templates
        values["xlabel"] = self.ctx.xlabel or ""
        values["ylabel"] = self.ctx.ylabel or ""

        # also expose raw component strings if template wants them
        values["filter"]    = ", ".join(_to_list(filter_desc))
        values["transform"] = ", ".join(_to_list(transform_desc))
        values["fit"]       = ", ".join(_to_list(fit_desc))

        # required keys
        values["name"]      = base_label
        values["modifiers"] = modifiers

        template = self.ctx.title_template or "{name}{modifiers}"
        return template.format_map(values).strip()

    def title_kwargs(
        self,
        *,
        series: pd.Series | None = None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
    ) -> Dict[str, Any]:
        """
        Extra kwargs for build_chart_title:
          - supports keys like extra_desc, fit_desc, filter_desc (override),
            or additional placeholders used by title_template (e.g., top_k=10)
        Default: {}
        """
        return {}
    
    def subtitle_text(
        self,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
    ) -> str:
        """
        Optional override: return a short subtitle derived from descriptive/inferential
        stats. Return ''/None to suppress.
        """
        return ""
    
    def footer_summary_text(
        self,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
    ) -> str:
        """
        Optional override: return a short footer summary derived from descriptive/inferential stats.
        Return ''/None to suppress.
        """
        return ""

    def metadata_overrides(
        self,
        *,
        series: pd.Series | None = None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
    ) -> Dict[str, Any]:
        """
        Extra/override keys to merge into the returned chart_metadata dict.
        Default: {}
        """
        return {}

    def _chart_metadata(
        self,
        *,
        series: pd.Series | None = None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
    ) -> Dict[str, Any]:
        title = self._build_chart_title(series=series, cols=cols, role_map=role_map)

        md = {
            "title": title,
            "xlabel": self.ctx.xlabel,
            "ylabel": self.ctx.ylabel,
            "data_source": self.ctx.data_source,
            "file_name": self.ctx.file_name,
            "version": self.plot_semantic_version(),
        }
        md.update(self.metadata_overrides(series=series, cols=cols, role_map=role_map) or {})
        return md

    # ======== Public API with dispatch (Series OR DataFrame) ========
    def run(
        self,
        data: Union[pd.Series, pd.DataFrame],
        *,
        cols: Optional[Sequence[str]] = None,
        role_map: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:

        # ---- SERIES PATH (unchanged API) ----
        if isinstance(data, pd.Series):
            s_in = data.copy(deep=True)
            s = self._validate_series(s_in)

            return self._pipeline_execute(
                is_empty=s.empty,
                build_md=lambda: self._chart_metadata(series=s),
                desc_fn=lambda: self.compute_descriptive(s),
                inf_fn=lambda desc: self.compute_inferential(s, desc),
                draw_fn=lambda desc, inf, md, fig, ax, palette: self.draw(
                    s, desc, inf, md, fig=fig, ax=ax, palette=palette
                ),
            )

        # ---- FRAME PATH ----
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas Series or DataFrame")

        if not cols:
            raise ValueError("For DataFrame input, provide cols=[...] with one or more column names.")
        
        # Defensive copy of the full input frame
        df_in = data.copy(deep=True)
        missing = [c for c in cols if c not in df_in.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        df = df_in.loc[:, list(cols)]

        # True bi/multivariate path via frame hooks
        df = self._validate_frame(df, cols=cols, role_map=role_map)

        return self._pipeline_execute(
            is_empty=df.empty,
            build_md=lambda: self._chart_metadata(cols=cols, role_map=role_map),
            desc_fn=lambda: self.compute_descriptive_frame(df, cols=cols, role_map=role_map),
            inf_fn=lambda desc: self.compute_inferential_frame(df, desc, cols=cols, role_map=role_map),
            draw_fn=lambda desc, inf, md, fig, ax, palette: self.draw_frame(
                df, desc, inf, md, cols=cols, role_map=role_map, fig=fig, ax=ax, palette=palette
            ),
        )

    # central place to create a publication-quality fig + colorblind palette
    def _new_figure_and_palette(self):
        fig, ax = plt.subplots(figsize=self.ctx.figsize, dpi=self.ctx.dpi)
        palette = sns.color_palette("colorblind")
        # apply palette to this axis' property cycle so lines/bars use it by default
        ax.set_prop_cycle(color=palette)
        return fig, ax, palette
    
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
            "light":  "#D0D0D0",  # ~82% gray
            "medium": "#B0B0B0",  # ~69% gray (default)
            "dark":   "#6E6E6E",  # ~43% gray
        }
        hex_color = presets.get(variant, presets["medium"])
        if alpha is None:
            return hex_color
        # Convert hex to normalized RGBA with requested alpha
        h = hex_color.lstrip("#")
        r, g, b = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        return (r, g, b, float(alpha))
    
    def queue_subtitle_below_title(
        self,
        ax,
        subtitle: str,
        *,
        fontsize: int = 10,
        color: str = "gray",
        gap_from_axes_pts: float = 1.5,
    ):
        """
        Queue a subtitle to be drawn after layout, directly above the axes (outside plot area)
        and below the main title. Call this in draw(); it will be positioned in _finalize_figure().
        """
        if not subtitle:
            return
        self._subtitle_queue.append((ax, subtitle, fontsize, color, gap_from_axes_pts))


    def register_annotations(self, ax, texts):
        """
        Register one or many matplotlib.text.Text objects for headroom measurement.
        Safe to call with a single Text, a list/tuple of Texts, or nested lists.
        """
        if not hasattr(self, "_headroom_texts"):
            self._headroom_texts = {}
        if texts is None:
            return
        flat = []
        stack = list(texts if isinstance(texts, (list, tuple)) else [texts])
        while stack:
            t = stack.pop()
            if t is None:
                continue
            if isinstance(t, (list, tuple)):
                stack.extend(t)
            else:
                flat.append(t)
        if flat:
            self._headroom_texts.setdefault(ax, []).extend(flat)

    def _gather_headroom_texts(self, ax):
        # Prefer explicitly registered texts; fall back to visible axis texts.
        texts = getattr(self, "_headroom_texts", {}).get(ax, [])
        return texts if texts else [t for t in ax.texts if t.get_visible()]

    def _measure_overhang_in_data(self, ax):
        """
        Return (left_oh, right_oh, bottom_oh, top_oh) overhang in DATA units
        caused by registered text extending beyond current x/y limits.
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
            left, right   = min(x0, x1), max(x0, x1)
            bottom, top   = min(y0, y1), max(y0, y1)

            if left   < lo_x: left_oh   = max(left_oh,   lo_x - left)
            if right  > hi_x: right_oh  = max(right_oh,  right - hi_x)
            if bottom < lo_y: bottom_oh = max(bottom_oh, lo_y - bottom)
            if top    > hi_y: top_oh    = max(top_oh,    top - hi_y)

        return left_oh, right_oh, bottom_oh, top_oh

    def ensure_headroom_for_annotations(self, ax, *, label_offset=0.02, extra_pad=0.04, max_extra=0.20):
        """
        Text-aware headroom that expands along the 'value' axis by default.
        Preserves symmetry if requested (either globally or plot-level).
        """
        if not getattr(self.ctx, "auto_headroom", True):
            return

        # orientation -> which axis holds the values
        val_is_y = bool(getattr(self.ctx, "is_orientation_vertical", True))
        axis_pref = getattr(self.ctx, "headroom_axis", "auto") if hasattr(self.ctx, "headroom_axis") else "auto"
        if axis_pref == "x":
            val_is_y = False
        elif axis_pref == "y":
            val_is_y = True

        preserve_sym = getattr(self.ctx, "headroom_preserve_symmetry", False)
        use_text = bool(getattr(self.ctx, "headroom_use_text_extents", True)) if hasattr(self.ctx, "headroom_use_text_extents") else True

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

    def _apply_metadata_to_axes(self, ax, chart_metadata: Dict[str, Any]):
        if chart_metadata.get("title"):
            ax.title_ref = ax.set_title(chart_metadata["title"], pad=14, fontsize=12, fontweight="bold")
        if chart_metadata.get("xlabel"):
            ax.set_xlabel(chart_metadata["xlabel"])
        if chart_metadata.get("ylabel"):
            ax.set_ylabel(chart_metadata["ylabel"])

    def _finalize_figure(self, fig, ax, chart_md: Dict[str, Any]) -> Optional[str]:
        """Add source, layout, save/show; return saved file name (or None)."""
        # 0) Ensure all artists exist & have positions computed
        fig.canvas.draw()

        # 0.5) Centralized, text-aware headroom for ALL plots (once)
        if getattr(self.ctx, "auto_headroom", True):
            self.ensure_headroom_for_annotations(
                ax,
                label_offset=getattr(self.ctx, "headroom_label_offset", 0.02),
                extra_pad=getattr(self.ctx, "headroom_extra_pad", 0.04),
                max_extra=getattr(self.ctx, "headroom_max_extra", 0.20),
            )

        # 1) Run layout first so axes land where they'll stay.
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        # ensure renderer is ready so we can measure text extents accurately
        fig.canvas.draw()

        # 2) Now place queued subtitles relative to the actual title bounding box
        if getattr(self, "_subtitle_queue", None):
            renderer = fig.canvas.get_renderer()
            fig_w_px, fig_h_px = fig.bbox.width, fig.bbox.height

            for ax, subtitle, fontsize, color, gap_pts in self._subtitle_queue:
                t = getattr(ax, "title_ref", None)
                if t is not None:
                    tb = t.get_window_extent(renderer=renderer)  # in display (px)
                    # convert a gap in points to pixels
                    gap_px = gap_pts * fig.dpi / 72.0
                    # position centered on the title horizontally, just below title
                    x_px = (tb.x0 + tb.x1) / 2.0
                    y_px = tb.y0 + gap_px

                    # convert to figure coordinates (0..1)
                    xf, yf = x_px / fig_w_px, y_px / fig_h_px
                    fig.text(xf, yf, subtitle, ha="center", va="top", fontsize=fontsize, color=color)

            self._subtitle_queue.clear()

        # Optional: add data source footer
        if self.ctx.data_source:
            fig.text(
                0.01, 0.01, f"Source: {self.ctx.data_source}",
                ha="left", va="bottom", fontsize="small", color="gray"
            )

        # 3) Save/show
        saved_name = None
        if self.ctx.save_path:
            saved_name = self.ctx.file_name or f'{chart_md["title"]}.png'
            os.makedirs(self.ctx.save_path, exist_ok=True)
            fig.savefig(os.path.join(self.ctx.save_path, saved_name), bbox_inches="tight", dpi=getattr(self.ctx, "dpi", None))
        if self.ctx.show:
            plt.show()
        return saved_name

    def _pipeline_execute(
        self,
        *,
        is_empty: bool,
        build_md: Callable[[], Dict[str, Any]],
        desc_fn: Callable[[], Dict[str, Any]],
        inf_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        draw_fn: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any], Any, Any, Any], Tuple[Any, Any]],
    ) -> Dict[str, Any]:
        """Shared execution flow for both Series and Frame paths."""
        fig = None
        try:
            chart_md = build_md()

            if is_empty:
                desc = self.default_descriptive()
                inf = self.default_inferential()
                return {
                    "descriptive_stats": desc,
                    "inferential_stats": inf,
                    "draft_descriptive_findings": self.draft_descriptive_findings(desc) or {},
                    "draft_inferential_findings": self.draft_inferential_findings(inf, desc) or {},
                    "chart_metadata": chart_md,
                }

            desc = desc_fn()
            inf = inf_fn(desc) or {}

            # If plot is skipped or errored, still return draft findings
            if desc.get("error") or desc.get("skip_plot"):
                chart_md["file_name"] = None
                return {
                    "descriptive_stats": desc,
                    "inferential_stats": inf,
                    "draft_descriptive_findings": self.draft_descriptive_findings(desc) or {},
                    "draft_inferential_findings": self.draft_inferential_findings(inf, desc) or {},
                    "chart_metadata": chart_md,
                }
            
            fig, ax, palette = self._new_figure_and_palette()
            self._apply_metadata_to_axes(ax, chart_md)

            if getattr(self.ctx, "show_subtitle", True):
                sub = self.subtitle_text(desc, inf, chart_md) or ""
                if sub.strip():
                    self.queue_subtitle_below_title(ax, sub)

            fig, ax = draw_fn(desc, inf, chart_md, fig, ax, palette)

            if getattr(self.ctx, "format_orientation_axis_as_percent", False):
                if self.ctx.is_orientation_vertical:
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                else:
                    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))

            if getattr(self.ctx, "show_footer_summary", False):
                ft = self.footer_summary_text(desc, inf, chart_md) or ""
                if ft.strip():
                    fig.text(
                        0.99,
                        0.01,
                        ft,
                        ha="right",
                        va="bottom",
                        fontsize="small",
                        color="gray"
                    )
            
            # Centralized legend toggle
            self._apply_legend(ax)

            chart_md["file_name"] = self._finalize_figure(fig, ax, chart_md)

            return {
                "descriptive_stats": desc,
                "inferential_stats": inf,
                "draft_descriptive_findings": self.draft_descriptive_findings(desc) or {},
                "draft_inferential_findings": self.draft_inferential_findings(inf, desc) or {},
                "chart_metadata": chart_md,
            }
        finally:
            # reset per-run cache
            self._draw_clear()
            if fig:
                plt.close(fig)

    def _apply_legend(self, ax):
        """
        Show or hide the legend based on ctx.enable_legend.
        If False, remove an existing legend (if any).
        """
        if getattr(self.ctx, "enable_legend", False):
            ax.legend()
        else:
            lg = ax.get_legend()
            if lg is not None:
                lg.remove()

    # --- draw cache API ---
    def _draw_set(self, namespace: str, key: str, value: Any) -> None:
        self._draw_cache.setdefault(namespace, {})[key] = value

    def _draw_get(self, namespace: str, key: str, default: Any = None) -> Any:
        return self._draw_cache.get(namespace, {}).get(key, default)

    def _draw_clear(self, namespace: Optional[str] = None) -> None:
        if namespace is None:
            self._draw_cache.clear()
        else:
            self._draw_cache.pop(namespace, None)

    # ----------------- unified validation dispatchers -----------------
    def _validate_series(self, s_in: pd.Series) -> pd.Series:
        """
        Prefer injected SeriesValidator; fall back to existing validate().
        This is the only call site the rest of BasePlot uses.
        """
        if self.parts.series_validator is not None:
            return self.parts.series_validator.validate(s_in)

        raise NotImplementedError("Series-based validate not implemented for this plot.")

    def _validate_frame(
        self,
        df_in: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Prefer injected FrameValidator; fall back to existing validate_frame().
        This is the only call site the rest of BasePlot uses.
        """
        if self.parts.frame_validator is not None:
            return self.parts.frame_validator.validate(df_in, cols=cols, role_map=role_map)
        # Back-compat path:
        return df_in
