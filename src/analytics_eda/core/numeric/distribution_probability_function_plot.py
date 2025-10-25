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
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator

from ..visualization.base_plot import BasePlot, PlotContext


@dataclass
class DistributionProbabilityFunctionContext(PlotContext):
    title_template: str = "{pf_kind} of {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str | None = None  # dynamic default if None

    # plot-specific
    is_discrete: bool = True
    bw_method: str | float = "scott"  # used for KDE when continuous


class DistributionProbabilityFunctionPlot(BasePlot):
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

    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=numeric_validator()
        )
        super().__init__(ctx, parts)

    def title_kwargs(self, *, series=None, cols=None, role_map=None) -> dict[str, Any]:
        is_disc = bool(self.ctx.is_discrete)
        pf_kind = "PMF" if is_disc else "PDF estimate"

        # Optional: show bandwidth inside modifiers for continuous KDE
        extras = {}
        if not is_disc and self.ctx.bw_method is not None:
            extras["extra_desc"] = f"bw={self.ctx.bw_method}"

        return {
            "pf_kind": pf_kind,   # used by {pf_kind} in title_template
            **extras,             # may include extra_desc for modifiers
        }

    def metadata_overrides(self, *, series=None, cols=None, role_map=None) -> dict[str, Any]:
        is_disc = bool(self.ctx.is_discrete)
        # Dynamic default if user didn't set ctx.ylabel
        ylabel = (
            self.ctx.ylabel
            if self.ctx.ylabel is not None
            else ("Probability P(X = x)" if is_disc else "Density f(x)")
        )

        meta = {
            "ylabel": ylabel,
            "is_discrete": is_disc,
        }
        if not is_disc:
            meta["bw_method"] = self.ctx.bw_method
        return meta

    def default_descriptive(self) -> dict[str, Any]:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "mode": None,
            "variance": None,
            "std": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
            "min": None,
            "max": None,
            # payload:
            "x_pmf": [],
            "y_pmf": [],
            "x_pdf": np.array([], dtype=float),
            "y_pdf": np.array([], dtype=float),
        }

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        # s is validated & NA-dropped by NumericSeriesMixin.validate
        n = int(s.size)

        # Basic stats
        mean = float(s.mean()) if n else None
        median = float(s.median()) if n else None
        mode_val = float(s.mode().iloc[0]) if n and not s.mode().empty else None
        variance = float(s.var()) if n else None
        std = float(s.std()) if n else None
        iqr = float(s.quantile(0.75) - s.quantile(0.25)) if n else None
        skewness = float(s.skew()) if n else None
        kurtosis = float(s.kurtosis()) if n else None
        min_val = float(s.min()) if n else None
        max_val = float(s.max()) if n else None

        desc: dict[str, Any] = {
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

    def draw(
        self,
        s: pd.Series,
        desc: dict[str, Any],
        inf: dict[str, Any],
        chart_metadata: dict[str, Any],
        *,
        fig,
        ax,
        palette,
    ):
        if self.ctx.is_discrete:
            ax.bar(desc["x_pmf"], desc["y_pmf"], edgecolor="black")
        else:
            ax.plot(desc["x_pdf"], desc["y_pdf"], linewidth=1.5)

        return fig, ax
