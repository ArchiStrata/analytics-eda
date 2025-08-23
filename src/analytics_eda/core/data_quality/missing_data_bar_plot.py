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
from typing import Dict, Any
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from ..utils.base_plot import BasePlot, PlotContext
from ..utils.named_series_mixin import NamedSeriesMixin


@dataclass
class MissingDataBarContext(PlotContext):
    title_template: str = "Missing Data for {name}{modifiers}"
    xlabel: str = "Status"
    ylabel: str = "Percentage of Total"


class MissingDataBarPlot(NamedSeriesMixin, BasePlot):
    """
    Shows the share of missing values to quickly assess data quality risk.

    Why this matters:
    - Missingness inflates bias and reduces statistical power; early visibility guides cleaning/imputation.

    What this plot does:
    - Computes present/missing counts and percentages, and annotates bars with both.
    - Y-axis is percentage for quick scanning.

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
        "chart_metadata": {"title","xlabel","ylabel","data_source","file_name","version"}
      }
    """
    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.
        """
        return "1.0.0"

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
            .reindex(["Missing", "Present"]) 
            .fillna(0)
            .astype(int)
        )

        total = int(counts.sum())
        missing = int(counts.loc["Missing"])
        pct_missing = float(missing / total) if total else 0.0

        pcts = (counts / total).to_numpy(dtype=float) if total else np.array([0.0, 0.0], dtype=float)

        return {
            "total": total,
            "missing": missing,
            "pct_missing": pct_missing,
            "labels": counts.index.tolist(),
            "counts": counts.to_numpy(dtype=int),
            "pcts": pcts,
        }
    
    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        if not desc or desc.get("total", 0) == 0:
            return {}
        pct = desc.get("pct_missing", 0.0) * 100
        return {
            "summary": f"{pct:.1f}% missing ({desc['missing']:,} of {desc['total']:,}); assess impact and plan handling."
        }

    # No inferential stats
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # Draw chart
    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):

        labels = desc["labels"]
        pcts = desc["pcts"]
        counts = desc["counts"]

        # Color scheme: Missing = palette[0], Present = grey
        colors = [palette[0] if lbl == "Missing" else "#B0B0B0" for lbl in labels]

        bars = ax.bar(labels, pcts, color=colors)

        # Annotate bars: % on first line, count in parentheses
        for bar, pct, cnt in zip(bars, pcts, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{pct*100:.1f}%\n({cnt:,})",
                ha="center", va="bottom", fontsize=10,
            )

        # Title & axis labels
        title = chart_metadata["title"]
        subtitle = f"{desc['pct_missing']*100:.1f}% of {desc['total']:,} values missing"

        ax.set_title(title, pad=6, fontsize=12, fontweight="bold")
        ax.text(
            0.5, 1.02, subtitle,
            ha="center", va="bottom",
            transform=ax.transAxes,
            fontsize=10, color="gray"
        )

        ax.set_xlabel(chart_metadata["xlabel"])     # "Status"
        ax.set_ylabel(chart_metadata["ylabel"])     # "Percentage of Total"
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        return fig, ax
