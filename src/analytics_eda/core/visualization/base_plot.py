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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from analytics_eda.core.visualization.context.plot_context import PlotContext
from analytics_eda.core.visualization.formatting.report_formatter import ReportFormatter
from analytics_eda.core.visualization.plot_parts import PlotParts

Desc = Union[str, Sequence[str], None]

class BasePlot(ABC):
    def __init__(self, ctx: PlotContext, parts: Optional[PlotParts] = None):
        self.ctx = ctx
        self.parts = parts or PlotParts()
        self._subtitle_queue: list[tuple] = []
        self._draw_cache: dict[str, dict[str, Any]] = {}   # per-run cache (cleared each run)
        self.formatter = ReportFormatter(ctx=self)

    # ======== Metadata & Title ========
    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.

        Defaults to "0.1.0". Child plots may override or extend this to
        indicate changes in functionality or output schema.
        """
        return "0.1.0"

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
        """
        Delegates to the metadata builder. Subclasses can still override
        metadata_overrides() to tweak values.
        """
        return self.parts.chart_metadata_builder.build_metadata(
            ctx=self.ctx,
            version_cb=self.plot_semantic_version,     # preserve your version hook
            series=series,
            cols=cols,
            role_map=role_map,
            title_kwargs_cb=self.title_kwargs,         # subclass hook
            metadata_overrides_cb=self.metadata_overrides,  # subclass hook
        )

    # ======== Descriptive Statistics ========

    def default_descriptive(self) -> Dict[str, Any]:
        return {}
    
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        raise NotImplementedError("Series-based compute descriptive not implemented for this plot.")

    def compute_descriptive_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Optional[Mapping[str,str]] = None) -> Dict[str, Any]:
        raise NotImplementedError("Frame-based descriptive not implemented for this plot.")

    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return draft, human-readable statements derived from descriptive stats.
        Child plots may override. Default: {}.
        """
        return {}

    # ======== Inferential Statistics ========

    def default_inferential(self) -> Dict[str, Any]:
        return {}
    
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}
    
    def compute_inferential_frame(self, df: pd.DataFrame, desc: Dict[str, Any], *, cols: Sequence[str], role_map: Optional[Mapping[str,str]] = None) -> Dict[str, Any]:
        return {}

    def draft_inferential_findings(self, inf: Dict[str, Any], desc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return draft, human-readable statements derived from inferential stats
        (and optionally descriptive context). Child plots may override. Default: {}.
        """
        return {}

    # ======== Draw ========
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
    
    def neutral_grey(self, variant: str = "medium", alpha: float | None = None):
        return self.parts.renderer.neutral_grey(variant, alpha)
    
    def register_annotations(self, ax, texts):
        return self.parts.renderer.register_annotations(ax, texts)

    # --- draw cache API ---
    def draw_cache_set(self, namespace: str, key: str, value: Any) -> None:
        self._draw_cache.setdefault(namespace, {})[key] = value

    def draw_cache_get(self, namespace: str, key: str, default: Any = None) -> Any:
        return self._draw_cache.get(namespace, {}).get(key, default)

    def _draw_clear(self, namespace: Optional[str] = None) -> None:
        if namespace is None:
            self._draw_cache.clear()
        else:
            self._draw_cache.pop(namespace, None)

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
            
            chart_md["file_name"] = self.parts.renderer.render(
                ctx=self.ctx,
                chart_md=chart_md,
                desc=desc,
                inf=inf,
                draw_fn=draw_fn,
                subtitle_text=(self.subtitle_text(desc, inf, chart_md)),
                footer_text=(self.footer_summary_text(desc, inf, chart_md))
            )

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
