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
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union


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
    show_subtitle: bool = False   # auto-draw a subtitle if provided by the plot

    is_orientation_vertical: bool = True   # True = vertical bars/values on Y; False = horizontal
    xlabel: str = ""
    ylabel: str = ""
    data_source: Optional[str] = None

    show_footer_summary: bool = False

    figsize: Tuple[int, int] = (14, 9)
    dpi: int = 200
    save_path: Optional[str] = None
    file_name: Optional[str] = None
    show: bool = False

    enable_legend: bool = False # draw a legend when True

    # TODO: Should this include formatting options like numeric units/value_decimals?
    format_orientation_axis_as_percent: bool = False

    # Auto headroom for bar labels (on by default)
    auto_headroom: bool = True
    headroom_label_offset: float = 0.02
    headroom_extra_pad: float = 0.04
    headroom_max_extra: float = 0.20
    headroom_use_text_extents: bool = True        # measure label text bboxes to set limits
    headroom_preserve_symmetry: bool = False      # keep +/- limits symmetric when expanding
    headroom_axis: Literal["auto","x","y"] = "auto"  # which axis to expand (auto = by orientation)
