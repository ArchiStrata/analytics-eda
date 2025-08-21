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
from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from matplotlib import pyplot as plt
import pandas as pd

from analytics_eda.core.utils import build_chart_title

Desc = Union[str, Sequence[str], None]

@dataclass
class PlotContext:
    name: Optional[str] = None
    filter_desc: Desc = None
    transform_desc: Desc = None
    fit_desc: Desc = None
    extra_desc: Desc = None
    title_fmt: Optional[Dict[str, Any]] = None

    title_template: str = "{name}{modifiers}"
    xlabel: str = ""
    ylabel: str = ""
    data_source: Optional[str] = None
    figsize: Tuple[int, int] = (10, 6)
    save_path: Optional[str] = None
    file_name: Optional[str] = None
    show: bool = False

class BasePlot(ABC):
    def __init__(self, ctx: PlotContext):
        self.ctx = ctx

    def default_descriptive(self) -> Dict[str, Any]:
        return {}
    
    # TODO: "draft_findings" - rule, statement
    # Draft descriptive findings per plot 
    # Draft inferential findings per plot 
    
    def default_inferential(self) -> Dict[str, Any]:
        return {}

    # ======== OPTIONAL SERIES API (only implement in univariate plots) ========
    def validate(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError("Series-based validate not implemented for this plot.")

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        raise NotImplementedError("Series-based compute descriptive not implemented for this plot.")

    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        raise NotImplementedError("Series-based draw not implemented for this plot.")
    
    # ======== OPTIONAL FRAME API (only implement in new bi/multivariate plots) ========
    def validate_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Optional[Mapping[str,str]] = None) -> pd.DataFrame:
        """Override in frame-aware plots; default is identity."""
        return df

    def compute_descriptive_frame(self, df: pd.DataFrame, *, cols: Sequence[str], role_map: Optional[Mapping[str,str]] = None) -> Dict[str, Any]:
        raise NotImplementedError("Frame-based descriptive not implemented for this plot.")

    def compute_inferential_frame(self, df: pd.DataFrame, desc: Dict[str, Any], *, cols: Sequence[str], role_map: Optional[Mapping[str,str]] = None) -> Dict[str, Any]:
        return {}

    def draw_frame(self, df: pd.DataFrame, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any], *, cols: Sequence[str], role_map: Optional[Mapping[str,str]] = None):
        raise NotImplementedError("Frame-based draw not implemented for this plot.")

    # ======== Metadata & Title (works for both) ========
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
        Build metadata for either a Series or a Frame (via cols/role_map).
        Title precedence:
          1) ctx.name (explicit)
          2) label from role_map (e.g., "y by x • hue")
          3) joined cols (e.g., "y • x")
          4) series.name (Series case)
          5) "Value"
        """
        # Decide label + what to pass to build_chart_title as 'series'
        if series is not None and self.ctx.name is None:
            series_for_title = series
        else:
            role_label = _label_from_roles(role_map) if role_map else None
            joined_cols = " • ".join(cols) if cols else None
            label = (
                self.ctx.name
                or role_label
                or joined_cols
                or (getattr(series, "name", None) if series is not None else None)
                or "Value"
            )
            series_for_title = _NameProxy(label)

        # --- gather subclass-provided overrides / extras ---
        tk = dict(self.title_kwargs(series=series, cols=cols, role_map=role_map) or {})

        # Allow subclass to override descs via title_kwargs; otherwise use context
        filter_desc   = tk.pop("filter_desc",   getattr(self.ctx, "filter_desc", None))
        transform_desc= tk.pop("transform_desc",getattr(self.ctx, "transform_desc", None))
        fit_desc      = tk.pop("fit_desc",      getattr(self.ctx, "fit_desc", None))
        extra_desc    = tk.pop("extra_desc",    getattr(self.ctx, "extra_desc", None))

        # Merge fmt: context-level first, then subclass overrides win
        ctx_fmt = getattr(self.ctx, "title_fmt", None) or {}
        fmt_all = {**ctx_fmt, **tk}  # tk now contains only non-desc placeholders

        title = build_chart_title(
            name=self.ctx.name,
            series=series_for_title,
            filter_desc=filter_desc,
            transform_desc=transform_desc,
            fit_desc=fit_desc,
            extra_desc=extra_desc,
            title_template=self.ctx.title_template,
            **fmt_all,
        ).strip()

        # TODO: support plots having semantic version. Each plot will need to override plot_semantic_version
        md = {
            "title": title,
            "xlabel": self.ctx.xlabel,
            "ylabel": self.ctx.ylabel,
            "data_source": self.ctx.data_source,
            "file_name": self.ctx.file_name,
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
            s = self.validate(s_in)

            return self._pipeline_execute(
                is_empty=s.empty,
                build_md=lambda: self._chart_metadata(series=s),
                desc_fn=lambda: self.compute_descriptive(s),
                inf_fn=lambda desc: self.compute_inferential(s, desc),
                draw_fn=lambda desc, inf, md: self.draw(s, desc, inf, md),
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
        df = self.validate_frame(df, cols=cols, role_map=role_map)

        return self._pipeline_execute(
            is_empty=df.empty,
            build_md=lambda: self._chart_metadata(cols=cols, role_map=role_map),
            desc_fn=lambda: self.compute_descriptive_frame(df, cols=cols, role_map=role_map),
            inf_fn=lambda desc: self.compute_inferential_frame(df, desc, cols=cols, role_map=role_map),
            draw_fn=lambda desc, inf, md: self.draw_frame(df, desc, inf, md, cols=cols, role_map=role_map),
        )


    def _finalize_figure(self, fig, chart_md: Dict[str, Any]) -> Optional[str]:
        """Add source, layout, save/show; return saved file name (or None)."""
        if self.ctx.data_source:
            fig.text(
                0.01, 0.01, f"Source: {self.ctx.data_source}",
                ha="left", va="bottom", fontsize="small", color="gray"
            )
        fig.tight_layout()

        saved_name = None
        if self.ctx.save_path:
            saved_name = self.ctx.file_name or f'{chart_md["title"]}.png'
            os.makedirs(self.ctx.save_path, exist_ok=True)
            fig.savefig(os.path.join(self.ctx.save_path, saved_name), bbox_inches="tight")
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
        draw_fn: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Tuple[Any, Any]],
    ) -> Dict[str, Any]:
        """Shared execution flow for both Series and Frame paths."""
        chart_md = build_md()

        if is_empty:
            return {
                "descriptive_stats": self.default_descriptive(),
                "inferential_stats": self.default_inferential(),
                "chart_metadata": chart_md,
            }

        desc = desc_fn()
        inf = inf_fn(desc) or {}

        if desc.get("error") or desc.get("skip_plot"):
            chart_md["file_name"] = None
            return {"descriptive_stats": desc, "inferential_stats": inf, "chart_metadata": chart_md}

        fig, _ = draw_fn(desc, inf, chart_md)
        chart_md["file_name"] = self._finalize_figure(fig, chart_md)

        return {"descriptive_stats": desc, "inferential_stats": inf, "chart_metadata": chart_md}

# --- tiny helpers to avoid changing existing plots ---

class _NameProxy:
    """Object with .name so build_chart_title behaves consistently for frame inputs."""
    def __init__(self, name: str): self.name = name

def _label_from_roles(role_map: Optional[Mapping[str, str]]) -> Optional[str]:
    if not role_map: return None
    y = role_map.get("y"); x = role_map.get("x"); hue = role_map.get("hue")
    parts = []
    if y and x: parts.append(f"{y} by {x}")
    elif y: parts.append(y)
    elif x: parts.append(x)
    if hue: parts.append(hue)
    return " • ".join(parts) if parts else None
