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
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from ..utils.base_plot import BasePlot, PlotContext
from ..utils.build_chart_title import build_chart_title
from .validate_numeric_named_series import NumericSeriesMixin

@dataclass
class DistributionProbabilityFunctionContext(PlotContext):
    # Two templates so we can choose based on is_discrete at runtime
    title_template_pmf: str = "PMF of {name}{modifiers}"
    title_template_pdf: str = "PDF estimate of {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: Optional[str] = None  # dynamic default if None
    figsize: Tuple[int, int] = (10, 6)

    # plot-specific
    is_discrete: bool = True
    bw_method: Union[str, float] = "scott"  # used for KDE when continuous


class DistributionProbabilityFunctionPlot(NumericSeriesMixin, BasePlot):
    """
    Plots an explicit Probability Mass Function (PMF) for discrete data or an explicit Probability Density Function (PDF) estimate for continuous data.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "n", "mean", "median", "mode", "variance", "std",
          "iqr", "skewness", "kurtosis", "min", "max",
          # payload for draw:
          "x_pmf","y_pmf"   (if discrete) OR
          "x_pdf","y_pdf"   (if continuous)
        },
        "inferential_stats": {},
        "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
      }
    """

    # (1) Build title + ylabel dynamically from context
    def build_chart_metadata(self, series: pd.Series) -> Dict[str, Any]:
        label = self.ctx.name or getattr(series, "name", None) or "Value"

        # Choose template based on discrete/continuous
        template = (
            self.ctx.title_template_pmf
            if self.ctx.is_discrete
            else self.ctx.title_template_pdf
        )
        title = build_chart_title(
            name=label,
            series=series,
            filter_desc=self.ctx.filter_desc,
            transform_desc=self.ctx.transform_desc,
            title_template=template,
        )

        # Dynamic ylabel default if not provided
        ylabel = (
            self.ctx.ylabel
            if self.ctx.ylabel is not None
            else ("Probability P(X = x)" if self.ctx.is_discrete else "Density f(x)")
        )

        return {
            "title": title,
            "xlabel": self.ctx.xlabel,
            "ylabel": ylabel,
            "data_source": self.ctx.data_source,
            "file_name": self.ctx.file_name,
        }

    # (2) default when empty
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "mode": float("nan"),
            "variance": float("nan"),
            "std": float("nan"),
            "iqr": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            # payload:
            "x_pmf": [],
            "y_pmf": [],
            "x_pdf": np.array([], dtype=float),
            "y_pdf": np.array([], dtype=float),
        }

    # (3) descriptive stats (+ payload for drawing)
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        # s is validated & NA-dropped by NumericSeriesMixin.validate
        n = int(s.size)

        # Basic stats
        mean = float(s.mean()) if n else float("nan")
        median = float(s.median()) if n else float("nan")
        mode_val = float(s.mode().iloc[0]) if n and not s.mode().empty else float("nan")
        variance = float(s.var()) if n else float("nan")
        std = float(s.std()) if n else float("nan")
        iqr = float(s.quantile(0.75) - s.quantile(0.25)) if n else float("nan")
        skewness = float(s.skew()) if n else float("nan")
        kurtosis = float(s.kurtosis()) if n else float("nan")
        min_val = float(s.min()) if n else float("nan")
        max_val = float(s.max()) if n else float("nan")

        desc: Dict[str, Any] = {
            "n": n,
            "mean": mean,
            "median": median,
            "mode": mode_val,
            "variance": variance,
            "std": std,
            "iqr": iqr,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "min": min_val,
            "max": max_val,
        }

        # Payload for draw
        if n == 0:
            # defaults already cover payload
            desc.update({
                "x_pmf": [], "y_pmf": [],
                "x_pdf": np.array([], dtype=float),
                "y_pdf": np.array([], dtype=float),
            })
            return desc

        if self.ctx.is_discrete:
            counts = s.value_counts().sort_index()
            pmf = counts / counts.sum()
            desc["x_pmf"] = pmf.index.tolist()
            desc["y_pmf"] = pmf.values.astype(float)
            # also provide empty continuous payload
            desc["x_pdf"] = np.array([], dtype=float)
            desc["y_pdf"] = np.array([], dtype=float)
        else:
            kde = gaussian_kde(s.to_numpy(), bw_method=self.ctx.bw_method)
            x_grid = np.linspace(min_val, max_val, 200) if np.isfinite(min_val) and np.isfinite(max_val) else np.linspace(-1, 1, 200)
            y_pdf = kde(x_grid)
            desc["x_pdf"] = x_grid
            desc["y_pdf"] = y_pdf
            # empty discrete payload
            desc["x_pmf"] = []
            desc["y_pmf"] = []

        return desc

    # (4) draw
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        sns.set_palette("colorblind")
        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        title = chart_metadata["title"]
        xlabel = chart_metadata["xlabel"] or "Value"
        ylabel = chart_metadata["ylabel"] or ("Probability P(X = x)" if self.ctx.is_discrete else "Density f(x)")

        if self.ctx.is_discrete:
            ax.bar(desc["x_pmf"], desc["y_pmf"], edgecolor="black")
        else:
            ax.plot(desc["x_pdf"], desc["y_pdf"], linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax
