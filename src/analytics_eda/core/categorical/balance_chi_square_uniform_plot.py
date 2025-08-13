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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chisquare

from ..utils.base_plot import BasePlot, PlotContext
from .validate_categorical_named_series import CategoricalSeriesMixin

@dataclass
class BalanceChiSquareUniformContext(PlotContext):
    title_template: str = "Chi-Square Goodness-of-Fit: {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = "Frequency"
    alpha: float = 0.05

class BalanceChiSquareUniformPlot(CategoricalSeriesMixin, BasePlot):
    """
    Chi-square goodness-of-fit vs. uniform for a categorical series.
    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {"total", "k", "categories", "observed", "expected"},
        "inferential_stats": {
            "chi2_gof_null_uniform": {
                "statistic", "p_value", "alpha", "reject", "warning?"}
        },
        "chart_metadata": {...}
      }
    """

    # (2) default response for empty data
    def default_descriptive(self) -> Dict[str, Any]:
        return {"total": 0, "k": 0, "categories": [], "observed": [], "expected": []}

    # (3) descriptive stats
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        freq = s.value_counts()
        categories = sorted(freq.index.tolist())
        observed = [int(freq[c]) for c in categories]
        total = int(sum(observed))
        k = int(len(categories))
        expected = [total / k] * k if k > 0 else []

        return {
            "total": total,
            "k": k,
            "categories": categories,
            "observed": observed,
            "expected": expected,
        }

    # (4) inferential stats
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        inf: Dict[str, Any] = {}
        k = desc["k"]
        if k == 0:
            return inf

        warning = None
        expected = desc["expected"]
        if any(e < 5 for e in expected):
            warning = "Some expected counts are below 5; chi-square test results may not be reliable."

        chi2_stat, p_val = chisquare(f_obs=desc["observed"], f_exp=expected)

        res = {
            "chi2_gof_null_uniform": {
                "statistic": float(chi2_stat),
                "p_value": float(p_val),
                "alpha": float(getattr(self.ctx, "alpha", 0.05)),
                "reject": bool(p_val < getattr(self.ctx, "alpha", 0.05)),
            }
        }
        if warning:
            res["chi2_gof_null_uniform"]["warning"] = warning
        return res

    # (5) draw
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        sns.set_palette("colorblind")

        title = chart_metadata["title"]
        xlabel = chart_metadata["xlabel"] or "Value"
        ylabel = chart_metadata["ylabel"] or "Frequency"

        fig, ax = plt.subplots(figsize=self.ctx.figsize)
        cats = desc["categories"]
        x = range(len(cats))
        width = 0.35

        ax.bar([i - width / 2 for i in x], desc["observed"], width, label="Observed")
        ax.bar([i + width / 2 for i in x], desc["expected"], width, label="Expected")

        ax.set_xticks(list(x))
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

        # annotation
        res = inf.get("chi2_gof_null_uniform")
        if res:
            ann = "\n".join((
                rf"$\chi^2$ = {res['statistic']:.2f}",
                rf"$p$ = {res['p_value']:.3f}",
                rf"$\alpha$ = {res['alpha']:.2f}",
                "Decision: " + ("Reject H₀" if res["reject"] else "Fail to Reject H₀"),
            ))
            ax.text(
                0.95, 0.95, ann, transform=ax.transAxes,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        return fig, ax
