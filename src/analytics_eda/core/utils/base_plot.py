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

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Tuple

from matplotlib import pyplot as plt
import pandas as pd

from analytics_eda.core.utils import build_chart_title


@dataclass
class PlotContext:
    name: Optional[str] = None
    filter_desc: Optional[str] = None
    transform_desc: Optional[str] = None
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

    @abstractmethod
    def validate(self, series: pd.Series) -> pd.Series:
        ...

    @abstractmethod
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        ...
    
    def default_descriptive(self) -> Dict[str, Any]:
        return {}

    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}
    
    def default_inferential(self) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        ...

    def build_chart_metadata(self, series: pd.Series):
        title = build_chart_title(
                        name=self.ctx.name, series=series,
                        filter_desc=self.ctx.filter_desc,
                        transform_desc=self.ctx.transform_desc,
                        title_template=self.ctx.title_template
                    )
        return {
                    "title": title,
                    "xlabel": self.ctx.xlabel,
                    "ylabel": self.ctx.ylabel,
                    "data_source": self.ctx.data_source,
                    "file_name": self.ctx.file_name,
                }

    def run(self, series: pd.Series) -> Dict[str, Any]:
        s = self.validate(series)
        chart_metadata = self.build_chart_metadata(s)

        if s.empty:
            return {
                "descriptive_stats": self.default_descriptive(),
                "inferential_stats": self.default_inferential(),
                "chart_metadata": chart_metadata,
            }

        desc = self.compute_descriptive(s)
        inf = self.compute_inferential(s, desc) or {}

        if desc.get("error") or desc.get("skip_plot"):
            chart_metadata["file_name"] = None
            return {
                "descriptive_stats": desc,
                "inferential_stats": inf,
                "chart_metadata": chart_metadata,
            }

        fig, _ = self.draw(s, desc, inf, chart_metadata)

        if self.ctx.data_source:
            fig.text(0.01, 0.01, f"Source: {self.ctx.data_source}",
                     ha="left", va="bottom", fontsize="small", color="gray")
        fig.tight_layout()

        saved_name = None
        if self.ctx.save_path:
            saved_name = self.ctx.file_name or f'{chart_metadata["title"]}.png'
            os.makedirs(self.ctx.save_path, exist_ok=True)
            fig.savefig(os.path.join(self.ctx.save_path, saved_name), bbox_inches="tight")
        if self.ctx.show:
            plt.show()

        chart_metadata['file_name'] = saved_name

        return {
            "descriptive_stats": desc,
            "inferential_stats": inf,
            "chart_metadata": chart_metadata,
        }
