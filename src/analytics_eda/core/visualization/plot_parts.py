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

from dataclasses import dataclass, field

from analytics_eda.core.visualization.chart_metadata import (
    ChartMetadataBuilderProtocol,
    DefaultChartMetadataBuilder,
)
from analytics_eda.core.visualization.rendering import DefaultMatplotlibRenderer, RendererProtocol
from analytics_eda.core.visualization.validation import FrameValidator, SeriesValidator


@dataclass
class PlotParts:
    series_validator: SeriesValidator | None = None
    frame_validator: FrameValidator | None = None
    chart_metadata_builder: ChartMetadataBuilderProtocol = field(default_factory=DefaultChartMetadataBuilder)
    renderer: RendererProtocol = field(default_factory=DefaultMatplotlibRenderer)
