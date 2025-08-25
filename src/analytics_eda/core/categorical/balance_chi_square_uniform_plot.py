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
from scipy.stats import chisquare

from ..utils.base_plot import BasePlot, PlotContext
from .validate_categorical_named_series import CategoricalSeriesMixin

@dataclass
class BalanceChiSquareUniformContext(PlotContext):
    title_template: str = "Chi-Square Goodness-of-Fit: {name}{modifiers}"
    xlabel: str = "Category"
    ylabel: str = "Count (Observed vs Expected)"
    alpha: float = 0.05

class BalanceChiSquareUniformPlot(CategoricalSeriesMixin, BasePlot):
    """
    Tests whether categorical frequencies deviate from a uniform distribution.

    Why this matters:
    - Uniform balance is often an assumption or target (e.g., stratified samples, equitable allocations).
      Large deviations can signal sampling bias, pipeline errors, or drift.

    What this plot does:
    - Computes observed counts by category and the uniform expected counts.
    - Runs a chi-square goodness-of-fit test (H₀: observed ~ Uniform).
    - Visualizes observed vs. expected counts side-by-side and summarizes the test result.

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
    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.
        """
        return "1.0.0"

    def default_descriptive(self) -> Dict[str, Any]:
        return {"total": 0, "k": 0, "categories": [], "observed": [], "expected": []}

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
    
    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        if not desc or desc.get("k", 0) == 0 or desc.get("total", 0) == 0:
            return {}
        cats = desc["categories"]
        obs = desc["observed"]
        exp = desc["expected"]
        diffs = [o - e for o, e in zip(obs, exp)]
        top_idx = int(max(range(len(diffs)), key=lambda i: abs(diffs[i])))
        return {
            "summary": f"Largest deviation: {cats[top_idx]} (obs={obs[top_idx]:,}, exp={exp[top_idx]:.1f}).",
            "coverage": f"Categories={desc['k']}, Total={desc['total']:,}."
        }

    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        k = desc.get("k", 0)
        total = desc.get("total", 0)
        if k == 0 or total == 0:
            return {}

        warning = None
        expected = desc["expected"]
        if any(e <= 0 for e in expected):
            return {"chi2_gof_null_uniform": {"warning": "Expected counts are zero; test not computed."}}

        if any(e < 5 for e in expected):
            warning = "Some expected counts are below 5; chi-square test results may not be reliable."

        chi2_stat, p_val = chisquare(f_obs=desc["observed"], f_exp=expected)

        df = max(desc.get("k", 0) - 1, 0)

        res = {
            "chi2_gof_null_uniform": {
                "df": df,
                "statistic": float(chi2_stat),
                "p_value": float(p_val),
                "alpha": float(getattr(self.ctx, "alpha", 0.05)),
                "reject": bool(p_val < getattr(self.ctx, "alpha", 0.05)),
            }
        }
        if warning:
            res["chi2_gof_null_uniform"]["warning"] = warning
        return res
    
    def draft_inferential_findings(self, inf: Dict[str, Any], desc: Dict[str, Any]) -> Dict[str, Any]:
        res = (inf or {}).get("chi2_gof_null_uniform")
        if not res:
            return {}
        decision = "Reject H₀" if res["reject"] else "Fail to reject H₀"
        return {
            "hypothesis_tests": f"Uniform GOF: {decision} at α={res['alpha']:.2f} (p={res['p_value']:.3f}, χ²={res['statistic']:.2f}, df={res['df']})."
        }

    def draw(self, s, desc, inf, chart_metadata, *, fig, ax, palette):
        cats = desc["categories"]
        y = range(len(cats))
        height = 0.38

        ax.barh([i + height/2 for i in y], desc["observed"], height, label="Observed", color=palette[0])
        ax.barh([i - height/2 for i in y], desc["expected"], height, label="Expected", color=self.neutral_grey())

        ax.set_yticks(list(y))
        ax.set_yticklabels(cats)
        ax.invert_yaxis()  # top-most first
        ax.legend()

        # annotation
        res = inf.get("chi2_gof_null_uniform")
        if res:
            self.queue_subtitle_below_title(
                ax,
                f"Uniform GOF: {'Reject' if res['reject'] else 'Fail to reject'} at α={res['alpha']:.2f} (p={res['p_value']:.3f})"
            )

        return fig, ax
