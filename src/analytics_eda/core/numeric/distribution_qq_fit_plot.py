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
from typing import Dict, Any, Literal
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..utils.base_plot import BasePlot, PlotContext
from .validate_numeric_named_series import NumericSeriesMixin

DistributionName = Literal['norm', 'lognorm', 'gamma', 'expon']

@dataclass
class DistributionQqFitContext(PlotContext):
    title_template: str = "Q–Q Plot Fit Assessment of {name}{modifiers}"
    xlabel: str = "Theoretical Quantiles"
    ylabel: str = "Sample Quantiles"

    # plot-specific
    distribution_name: DistributionName = 'norm'
    alpha: float = 0.05

class DistributionQqFitPlot(NumericSeriesMixin, BasePlot):
    """
    Generate a Q–Q plot that effectively communicates how closely a numeric variable
    follows a distribution type, with quantitative diagnostics.

    Why:
        Assess how well a numeric variable matches a theoretical distribution
        (‘norm’, ‘lognorm’, ‘gamma’ or ‘expon’). Beyond visual alignment, you
        get quantitative measures of fit (linearity, residuals, shape) and,
        when testing normality, formal tests for departures.

    What:
        - Points: sample quantiles vs. theoretical quantiles of the specified distribution.
        - Fit line (intercept α, slope β) and coefficient of determination (R²).
        - Residual diagnostics: median residual, IQR of residuals, maximum absolute residual.
        - Shape metrics: sample skewness and excess kurtosis.
        - If `distribution_name == 'norm'`, conducts:
            • Shapiro–Wilk (n < 50)  
            • D’Agostino–Pearson omnibus (n ≥ 20)  
            • Jarque–Bera (n > 2000)  
            • Overall reject flag if any test rejects H0.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "intercept","slope","r_squared","median_residual","iqr_residual",
          "max_abs_residual","skewness","kurtosis","min"
        },
        "inferential_stats": { ... tests & params ... },
        "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
      }
    """

    def title_kwargs(self, *, series=None, cols=None, role_map=None) -> Dict[str, Any]:
        dist = self.ctx.distribution_name
        return {
            "fit_desc": f"fitted to {dist}",
            # If you also want alpha shown: "extra_desc": f"alpha={self.ctx.alpha:g}",
            # If your template ever needs a placeholder, expose it too (e.g., {dist}):
            "dist": dist,
        }

    def metadata_overrides(self, *, series=None, cols=None, role_map=None) -> Dict[str, Any]:
        return {
            "distribution_name": self.ctx.distribution_name,
            "alpha": float(self.ctx.alpha),
        }

    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "r_squared": float("nan"),
            "median_residual": float("nan"),
            "iqr_residual": float("nan"),
            "max_abs_residual": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "min": float("nan")
        }

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        data = s.dropna().astype(float)
        n = int(data.size)
        if n == 0:
            return self.default_descriptive()

        dist_name = self.ctx.distribution_name
        ALLOWED = ('norm', 'lognorm', 'gamma', 'expon')
        if dist_name not in ALLOWED:
            raise ValueError(f"distribution_name must be one of {ALLOWED}")

        # Support checks for domain
        mn = float(data.min())
        if dist_name in ('lognorm', 'gamma') and mn <= 0:
            # Return empty-style stats, but include a sentinel slope/intercept NaN etc.
            desc = self.default_descriptive()
            # Keep min so callers can reason about transforms later
            desc["min"] = mn
            desc["error"] = "requires positive data"
            desc["skip_plot"] = True
            return desc
        if dist_name == 'expon' and mn < 0:
            desc = self.default_descriptive()
            desc["min"] = mn
            desc["error"] = "requires non-negative data"
            desc["skip_plot"] = True
            return desc

        # Fit distribution
        dist = getattr(stats, dist_name)
        params = dist.fit(data)
        *shape_args, loc, scale = params

        # Theoretical quantiles (plotting positions)
        probs = (np.arange(1, n + 1) - 0.5) / n
        osm = dist.ppf(probs, *shape_args, loc=loc, scale=scale)
        osr = np.sort(data.to_numpy())

        # Linear fit osr ~ a + b * osm
        slope, intercept = np.polyfit(osm, osr, 1)
        fitted = intercept + slope * osm

        # R^2 via correlation
        corr = np.corrcoef(osr, fitted)[0, 1]
        r_squared = float(corr ** 2)

        residuals = osr - fitted
        median_residual = float(np.median(residuals))
        iqr_residual = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
        max_abs_residual = float(np.max(np.abs(residuals)))

        skewness = float(stats.skew(data, bias=False))
        kurtosis = float(stats.kurtosis(data, fisher=True, bias=False))

        return {
            "intercept": float(intercept),
            "slope": float(slope),
            "r_squared": r_squared,
            "median_residual": median_residual,
            "iqr_residual": iqr_residual,
            "max_abs_residual": max_abs_residual,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "min": mn,
            # payload
            "osm": osm,
            "osr": osr,
            "fitted": fitted,
        }
    
    def default_inferential(self) -> Dict[str, Any]:
        return {
            "params": {
                "alpha": self.ctx.alpha,
                'distribution_name': self.ctx.distribution_name
            }
        }

    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        data = s.dropna().astype(float)
        n = int(data.size)
        res: Dict[str, Any] = {"params": {"alpha": float(self.ctx.alpha), "distribution_name": self.ctx.distribution_name}}
        if n == 0:
            return res

        if "error" in desc:
            # domain errors — no tests possible
            return res

        if self.ctx.distribution_name == 'norm':
            alpha = float(self.ctx.alpha)

            # Shapiro–Wilk for n < 50
            if n < 50:
                stat_sw, p_sw = stats.shapiro(data)
                res["shapiro"] = {"statistic": float(stat_sw), "p_value": float(p_sw), "reject": bool(p_sw < alpha)}

            # D’Agostino–Pearson omnibus for n ≥ 20
            if n >= 20:
                stat_dp, p_dp = stats.normaltest(data)
                res["dagostino_pearson"] = {"statistic": float(stat_dp), "p_value": float(p_dp), "reject": bool(p_dp < alpha)}

            # Jarque–Bera for n > 2000
            if n > 2000:
                stat_jb, p_jb = stats.jarque_bera(data)
                res["jarque_bera"] = {"statistic": float(stat_jb), "p_value": float(p_jb), "reject": bool(p_jb < alpha)}

            # Overall reject flag
            res["reject_normality"] = any(v.get("reject", False) for k, v in res.items() if isinstance(v, dict))

        return res

    def draw(
        self,
        s: pd.Series,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
        *,
        fig,
        ax,
        palette,
    ):

        # If domain error (e.g., lognorm with nonpositive), just render title/labels and note error
        if "error" in desc:
            ax.set_title(chart_metadata["title"])
            ax.set_xlabel(chart_metadata["xlabel"])
            ax.set_ylabel(chart_metadata["ylabel"])
            ax.text(
                0.5, 0.5, f"Data domain error: {desc['error']}",
                transform=ax.transAxes, ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6)
            )
            return fig, ax

        osm = desc["osm"]; osr = desc["osr"]; fitted = desc["fitted"]

        # points + fit line
        sns.scatterplot(x=osm, y=osr, ax=ax, s=20, edgecolor="k", alpha=0.6, label="Quantiles")
        ax.plot(osm, fitted, color="red", lw=1, label="Fit line")

        # Stats textbox (left-top)
        lines = [
            f"α (intercept): {desc['intercept']:.2f}",
            f"β (slope): {desc['slope']:.2f}",
            f"R²: {desc['r_squared']:.3f}",
            f"Median resid: {desc['median_residual']:.2f}",
            f"IQR resid: {desc['iqr_residual']:.2f}",
            f"Max abs resid: {desc['max_abs_residual']:.2f}",
            f"Skewness: {desc['skewness']:.2f}",
            f"Excess kurtosis: {desc['kurtosis']:.2f}",
        ]
        if self.ctx.distribution_name == 'norm' and inf:
            lines.append("")  # spacer
            for k, v in inf.items():
                if k == "params":
                    continue
                if k == "reject_normality":
                    lines.append(f"Overall reject: {v}")
                else:
                    p = v.get("p_value", np.nan)
                    lines.append(f"{k}: stat={v['statistic']:.3f}, p={p:.3f}, reject={v['reject']}")

        ax.text(
            0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize="small", bbox=dict(facecolor="white", alpha=0.5)
        )

        ax.legend()
        return fig, ax
