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
"""Plot context and axis-format primitives for Analytics-EDA."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Desc = str | Sequence[str] | None


@dataclass
class AxisFormat:
    """Axis tick/label formatting options.

    Set `kind` to control the formatter, with additional fields for
    decimals, thousands separator, units/currency, datetime format,
    or a custom callable formatter.
    """

    # High-level style
    kind: Literal["auto", "number", "percent", "currency", "category", "datetime"] = "auto"

    # Number/percent/currency options
    decimals: int | None = None  # e.g., 0, 1, 2; None = don't force
    thousands_sep: bool = True  # 12,345 vs 12345
    unit_suffix: str | None = None  # e.g., "ms", "kg" (appended after value)
    currency_code: str | None = None  # e.g., "USD" (used if kind == "currency")

    # Percent options
    percent_scale_0to1: bool = True  # True if data are proportions (0..1)

    # Datetime options
    datetime_format: str | None = None  # e.g., "%Y-%m-%d"

    # Escape hatch: custom formatter
    formatter: Callable[[float], str] | None = None


@dataclass
class PlotContext:
    """Per-plot configuration used by renderers and mixins.

    Includes high-level labels/titles, orientation, saving/showing options,
    legend toggles, axis formatters, automatic headroom for labels, and
    default report-formatting preferences.
    """

    name: str | None = None
    filter_desc: Desc = None
    transform_desc: Desc = None
    fit_desc: Desc = None
    extra_desc: Desc = None
    title_fmt: dict[str, Any] | None = None
    title_template: str = "{name}{modifiers}"
    show_subtitle: bool = False  # auto-draw a subtitle if provided by the plot

    is_orientation_vertical: bool = True  # True = vertical bars/values on Y; False = horizontal
    xlabel: str = ""
    ylabel: str = ""
    data_source: str | None = None

    show_footer_summary: bool = False

    figsize: tuple[int, int] = (14, 9)
    dpi: int = 200
    base_dir: str | None = None
    file_name: str | None = None
    auto_file_name: bool = False
    show: bool = False

    enable_legend: bool = False  # draw a legend when True

    x_format: AxisFormat = field(default_factory=AxisFormat)
    y_format: AxisFormat = field(default_factory=AxisFormat)

    # Auto headroom for bar labels (on by default)
    auto_headroom: bool = True
    headroom_label_offset: float = 0.02
    headroom_extra_pad: float = 0.04
    headroom_max_extra: float = 0.20
    headroom_use_text_extents: bool = True  # measure label text bboxes to set limits
    headroom_preserve_symmetry: bool = False  # keep +/- limits symmetric when expanding
    headroom_axis: Literal["auto", "x", "y"] = "auto"  # which axis to expand (auto = by orientation)

    # Report Formatting
    report_default_decimals: int = 2
    report_max_decimals: int = 6
    report_default_unit: str | None = None
