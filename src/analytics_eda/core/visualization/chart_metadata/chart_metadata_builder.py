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
"""Chart title and metadata builders for Analytics-EDA."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

Desc = str | Sequence[str] | None


class ChartMetadataBuilderProtocol:
    """Builds chart title and chart metadata dict.

    BasePlot will pass callbacks to allow subclass overrides to participate
    (title_kwargs / metadata_overrides / version).
    """

    def build_title(
        self,
        *,
        ctx,
        series: pd.Series | None = None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
        title_kwargs_cb: Callable[..., dict[str, Any]] | None = None,
    ) -> str:
        """Return a chart title string built from context/roles/overrides."""
        raise NotImplementedError

    def build_metadata(
        self,
        *,
        ctx,
        version_cb: Callable[[], str],
        series: pd.Series | None = None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
        title_kwargs_cb: Callable[..., dict[str, Any]] | None = None,
        metadata_overrides_cb: Callable[..., dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a chart metadata dict (title, labels, source, file, version, …)."""
        raise NotImplementedError


@dataclass(frozen=True)
class DefaultChartMetadataBuilder(ChartMetadataBuilderProtocol):
    """Default builder mirroring the current BasePlot logic."""

    def _to_list(self, x: Desc) -> list[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [x] if x else []
        return [str(v) for v in x if v]

    def _label_from_roles(self, role_map: Mapping[str, str] | None) -> str | None:
        if not role_map:
            return None

        y = role_map.get("y")
        x = role_map.get("x")
        hue = role_map.get("hue")

        parts = []
        if y and x:
            parts.append(f"{y} by {x}")
        elif y:
            parts.append(y)
        elif x:
            parts.append(x)
        if hue:
            parts.append(hue)
        return " • ".join(parts) if parts else None

    def build_title(
        self,
        *,
        ctx,
        series: pd.Series | None = None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
        title_kwargs_cb: Callable[..., dict[str, Any]] | None = None,
    ) -> str:
        """Compose the final chart title.

        Chooses a base label from ctx/roles/cols/series, applies modifiers
        from filter/transform/fit/extra descriptions, and renders with
        `ctx.title_template` plus optional `title_kwargs_cb`.
        """
        joined_cols = " • ".join(cols) if cols else None
        role_label = self._label_from_roles(role_map)

        base_label = getattr(ctx, "name", None) or role_label or joined_cols or (getattr(series, "name", None) if series is not None else None) or "Value"

        # subclass-provided overrides / extras
        tk = dict((title_kwargs_cb or (lambda **_: {}))(series=series, cols=cols, role_map=role_map) or {})
        filter_desc = tk.pop("filter_desc", getattr(ctx, "filter_desc", None))
        transform_desc = tk.pop("transform_desc", getattr(ctx, "transform_desc", None))
        fit_desc = tk.pop("fit_desc", getattr(ctx, "fit_desc", None))
        extra_desc = tk.pop("extra_desc", getattr(ctx, "extra_desc", None))

        parts = self._to_list(filter_desc) + self._to_list(transform_desc) + self._to_list(fit_desc) + self._to_list(extra_desc)
        modifiers = f" ({', '.join(parts)})" if parts else ""

        # ctx fmt first, then subclass placeholders override
        ctx_fmt = getattr(ctx, "title_fmt", None) or {}
        values = defaultdict(str, {**ctx_fmt, **tk})

        # always expose xlabel / ylabel for templates
        values["xlabel"] = getattr(ctx, "xlabel", "") or ""
        values["ylabel"] = getattr(ctx, "ylabel", "") or ""

        # expose raw component strings if template wants them
        values["filter"] = ", ".join(self._to_list(filter_desc))
        values["transform"] = ", ".join(self._to_list(transform_desc))
        values["fit"] = ", ".join(self._to_list(fit_desc))

        # required keys
        values["name"] = base_label
        values["modifiers"] = modifiers

        template = getattr(ctx, "title_template", None) or "{name}{modifiers}"
        return template.format_map(values).strip()

    def build_metadata(
        self,
        *,
        ctx,
        version_cb: Callable[[], str],
        series: pd.Series | None = None,
        cols: Sequence[str] | None = None,
        role_map: Mapping[str, str] | None = None,
        title_kwargs_cb: Callable[..., dict[str, Any]] | None = None,
        metadata_overrides_cb: Callable[..., dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Assemble chart metadata.

        Returns a dict with title/xlabel/ylabel/data_source/file_name/version,
        merged with any overrides from `metadata_overrides_cb`.
        """
        title = self.build_title(
            ctx=ctx,
            series=series,
            cols=cols,
            role_map=role_map,
            title_kwargs_cb=title_kwargs_cb,
        )
        md = {
            "title": title,
            "xlabel": getattr(ctx, "xlabel", None),
            "ylabel": getattr(ctx, "ylabel", None),
            "data_source": getattr(ctx, "data_source", None),
            "file_name": getattr(ctx, "file_name", None),
            "version": version_cb(),
        }
        overrides = (metadata_overrides_cb or (lambda **_: {}))(series=series, cols=cols, role_map=role_map) or {}
        md.update(overrides)
        return md
