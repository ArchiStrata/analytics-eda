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
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from ..utils.base_plot import BasePlot, PlotContext
from ..utils.named_series_mixin import NamedSeriesMixin


@dataclass
class MissingDataBarContext(PlotContext):
    title_template: str = "Missing Data for {name}"
    xlabel: str = ""
    ylabel: str = "Percentage of Total"
    figsize: Tuple[int, int] = (8, 6)


class MissingDataBarPlot(NamedSeriesMixin, BasePlot):
    """
    Bar chart of Present vs Missing with percentage y-axis and count/% annotations.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "total": int,
          "missing": int,
          "pct_missing": float,
          # payload for draw:
          "labels": ["Present","Missing"],
          "counts": np.ndarray[int],
          "pcts": np.ndarray[float]
        },
        "inferential_stats": {},
        "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
      }
    """

    # Build chart metadata from context (same pattern as cardinality)
    def build_chart_metadata(self, series: pd.Series) -> Dict[str, Any]:
        label = self.ctx.name or (series.name if series.name else "Value")
        title = self.ctx.title_template.format(name=label)
        return {
            "title": title,
            "xlabel": self.ctx.xlabel,
            "ylabel": self.ctx.ylabel,
            "data_source": self.ctx.data_source,
            "file_name": self.ctx.file_name,
        }

    # Defaults when empty
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "total": 0,
            "missing": 0,
            "pct_missing": 0.0,
            "labels": ["Present", "Missing"],
            "counts": np.array([0, 0], dtype=int),
            "pcts": np.array([0.0, 0.0], dtype=float),
        }

    # Compute descriptive stats (+ payload for drawing)
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        # counts Present/Missing
        status = s.isna().map({False: "Present", True: "Missing"})
        counts = (
            status.value_counts()
            .reindex(["Present", "Missing"])
            .fillna(0)
            .astype(int)
        )

        total = int(counts.sum())
        missing = int(counts.loc["Missing"])
        pct_missing = float(missing / total) if total else 0.0

        if total:
            pcts = (counts / total * 100.0).to_numpy(dtype=float)
        else:
            pcts = np.array([0.0, 0.0], dtype=float)

        return {
            "total": total,
            "missing": missing,
            "pct_missing": pct_missing,
            "labels": counts.index.tolist(),
            "counts": counts.to_numpy(dtype=int),
            "pcts": pcts,
        }

    # No inferential stats
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # Draw chart
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        sns.set_palette("colorblind")
        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        labels = desc["labels"]
        pcts = desc["pcts"]
        counts = desc["counts"]

        bars = ax.bar(labels, pcts)

        # Annotate counts and %
        for i, (bar, pct, cnt) in enumerate(zip(bars, pcts, counts)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{cnt:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Title & axis labels
        title = chart_metadata["title"]
        # Keep the concise title in chart title; show counts/% as subtitle-like info
        subtitle = f"{desc['missing']:,} of {desc['total']:,} values ({desc['pct_missing']*100:.1f}%) missing"
        ax.set_title(f"{title}: {subtitle}", pad=12)

        ax.set_xlabel(chart_metadata["xlabel"])
        ax.set_ylabel(chart_metadata["ylabel"])
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        sns.despine(left=True)
        fig.tight_layout()

        return fig, ax
