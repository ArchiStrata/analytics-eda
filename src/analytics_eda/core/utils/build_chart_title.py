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
from typing import Any, Sequence, Union
from collections import defaultdict

Desc = Union[str, Sequence[str], None]

def build_chart_title(
    *,
    name: str = None,
    series=None,
    filter_desc: Desc = None,
    transform_desc: Desc = None,
    fit_desc: Desc = None,
    extra_desc: Desc = None,
    title_template: str = "{name}{modifiers}",
    **fmt: Any,  # arbitrary placeholders for the format string, e.g. top_k=10
) -> str:
    """
    Build a chart title from a template, series name (or fallback), and optional modifiers.

    Parameters
    ----------
    name : str, optional
        Explicit label; falls back to series.name or "Value".
    series : object with .name attribute, optional
        Used only as a fallback if name is None.
    filter_desc : str or list[str], optional
        e.g., "filtered by New York" or ["filtered by New York", "city=Andover"].
    transform_desc : str or list[str], optional
        e.g., "log-transformed".
    fit_desc : str or list[str], optional
        e.g., "fitted to Normal".
    extra_desc : str or list[str], optional
        Any additional modifier(s), e.g., "Top 10".
    title_template : str
        Python format string; recognized keys:
          - {name}: base label
          - {modifiers}: space-prefixed "(…)" text or empty
          - {filter}, {transform}, {fit}: raw strings for each (optional use)
          - Any additional keys provided via **fmt (e.g., {top_k})
    **fmt : Any
        Extra placeholders to render into title_template (e.g., top_k=10).

    Returns
    -------
    str
        The rendered title.
    """

    def _to_list(x: Desc) -> list[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [x] if x else []
        # assume Sequence[str]-like
        return [str(v) for v in x if v]

    # base label
    label = name or getattr(series, "name", None) or "Value"

    # collect modifier parts (order: filter → transform → fit → extra)
    parts = _to_list(filter_desc) + _to_list(transform_desc) + _to_list(fit_desc) + _to_list(extra_desc)
    modifiers = f" ({', '.join(parts)})" if parts else ""

    # Build dict for format_map(); missing keys default to ""
    values = defaultdict(
        str,
        {
            "name": label,
            "modifiers": modifiers,
            "filter": ", ".join(_to_list(filter_desc)),
            "transform": ", ".join(_to_list(transform_desc)),
            "fit": ", ".join(_to_list(fit_desc)),
            **fmt,  # e.g., {"top_k": 10}
        },
    )

    # Use format_map to avoid KeyError for unknown placeholders
    return title_template.format_map(values)
