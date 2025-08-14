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
from typing import Dict, Any, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from ..utils.base_plot import BasePlot, PlotContext
from .validate_numeric_named_series import NumericSeriesMixin

DistName = Literal['norm', 'lognorm', 'gamma', 'expon']

@dataclass
class DistributionECDFvsCDFContext(PlotContext):
    title_template: str = "ECDF vs. Theoretical CDF of {name}{modifiers}"
    xlabel: str = "Value"
    ylabel: str = "CDF"
    figsize: Tuple[int, int] = (10, 6)

    # plot-specific
    distribution_name: DistName = 'norm'
    alpha: float = 0.05

class DistributionECDFvsCDFPlot(NumericSeriesMixin, BasePlot):
    """
    Generate an ECDF vs. theoretical CDF plot with goodness-of-fit tests (KS, AD, CvM).

    Why:
        Visualize the fit of data to a theoretical distribution by showing
        the empirical CDF against the fitted CDF, and quantify with formal tests.

    What:
        - Fits parameters for 'norm', 'lognorm', 'gamma', or 'expon'.
        - Computes ECDF and theoretical CDF.
        - Runs:
            • Kolmogorov–Smirnov for all distributions.
            • Anderson–Darling for 'norm' and 'expon'.
            • Cramér–von Mises for all distributions.
        - Annotates ECDF, CDF, max gap (KS D) and includes a stats textbox.
        - Returns descriptive stats, test results, and chart metadata.

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "n": int,
          "params": {"distribution_fit": tuple|None, "distribution_name": str},
          "error": str (optional)
        },
        "inferential_stats": {
          "params": {"alpha": float},
          "ks": {...}, "anderson": {...}, "cvm": {...}  # when applicable
        },
        "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
      }
    """

    ALLOWED: Tuple[DistName, ...] = ('norm', 'lognorm', 'gamma', 'expon')

    def title_kwargs(self, *, series=None, cols=None, role_map=None) -> Dict[str, Any]:
        dist = self.ctx.distribution_name
        return {
            # shows up inside "(...)" via build_chart_title's modifiers
            "fit_desc": f"fitted to {dist}",
            # optional extras (uncomment if you want them in modifiers too)
            # "extra_desc": f"alpha={self.ctx.alpha:g}",
            # also make {dist} available in case your template uses it
            "dist": dist,
        }

    def metadata_overrides(self, *, series=None, cols=None, role_map=None) -> Dict[str, Any]:
        return {
            "distribution_name": self.ctx.distribution_name,
            "alpha": float(self.ctx.alpha),
        }

    # (1) default when empty
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "n": 0,
            "params": {
                "distribution_fit": None,
                "distribution_name": self.ctx.distribution_name,
            },
        }

    # (2) default when empty
    def default_inferential(self) -> Dict[str, Any]:
        return {"params": {"alpha": float(self.ctx.alpha)}}

    # (3) descriptive stats (+ payload for drawing)
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        if self.ctx.distribution_name not in self.ALLOWED:
            raise ValueError(f"distribution_name must be one of {self.ALLOWED}")

        data = s.dropna().astype(float)
        n = int(data.size)

        # support checks (mirror legacy behavior)
        mn = float(data.min()) if n else float('inf')
        err: Optional[str] = None
        if self.ctx.distribution_name in ('lognorm', 'gamma') and n > 0 and mn <= 0:
            err = 'requires positive data'
        if self.ctx.distribution_name == 'expon' and n > 0 and mn < 0:
            err = 'requires non-negative data'

        desc: Dict[str, Any] = {
            "n": n,
            "params": {"distribution_name": self.ctx.distribution_name},
        }

        if n == 0:
            desc["params"]["distribution_fit"] = None
            # payload for draw (unused)
            desc.update({"x": np.array([]), "ecdf": np.array([]), "cdf_theo": np.array([]), "ks_D": np.nan})
            return desc

        if err is not None:
            desc["params"]["distribution_fit"] = None
            desc["error"] = err
            desc["skip_plot"] = True
            # payload for draw (unused)
            desc.update({"x": np.array([]), "ecdf": np.array([]), "cdf_theo": np.array([]), "ks_D": np.nan})
            return desc

        # Fit distribution
        dist = getattr(stats, self.ctx.distribution_name)
        fit_params = dist.fit(data)
        fit_params_float = tuple(float(np.round(p, 3)) for p in fit_params)
        desc["params"]["distribution_fit"] = fit_params_float

        # ECDF
        x = np.sort(data.to_numpy())
        ecdf = np.arange(1, n + 1) / n

        # Theoretical CDF
        cdf_theo = dist.cdf(x, *fit_params)

        # payload for drawing
        ks_D = float(np.max(np.abs(ecdf - cdf_theo)))
        desc.update({"x": x, "ecdf": ecdf, "cdf_theo": cdf_theo, "ks_D": ks_D})
        return desc

    # (4) inferential stats
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"params": {"alpha": float(self.ctx.alpha)}}

        # empty or error → only params
        if desc.get("n", 0) == 0 or "error" in desc:
            return out

        name = self.ctx.distribution_name
        data = s.dropna().astype(float)
        dist = getattr(stats, name)

        # Refit for test args (or reconstruct from desc if preferred)
        fit_params_float = desc["params"].get("distribution_fit", None)
        if fit_params_float is None:
            fit_params = dist.fit(data)
        else:
            # we can safely reuse rounded params for display; tests generally tolerant
            fit_params = tuple(fit_params_float)

        alpha = self.ctx.alpha

        # 1) KS
        D, p_ks = stats.kstest(data, name, args=fit_params)
        out["ks"] = {"statistic": float(D), "p_value": float(p_ks), "reject": bool(p_ks < alpha)}

        # 2) Anderson–Darling (norm, expon)
        if name in ('norm', 'expon'):
            ad = stats.anderson(data, dist=name)
            levels = np.array(ad.significance_level) / 100.0
            idx = int(np.argmin(np.abs(levels - alpha)))
            crit = float(ad.critical_values[idx])
            out["anderson"] = {
                "statistic": float(ad.statistic),
                "critical_value": crit,
                "critical_values": list(map(float, ad.critical_values)),
                "significance_levels": list(map(float, ad.significance_level)),
                "reject": bool(ad.statistic > crit),
            }

        # 3) Cramér–von Mises
        cvm_res = stats.cramervonmises(data, name, args=fit_params)
        out["cvm"] = {
            "statistic": float(cvm_res.statistic),
            "p_value": float(cvm_res.pvalue),
            "reject": bool(cvm_res.pvalue < alpha),
        }

        return out

    # (5) draw
    def draw(
        self,
        s: pd.Series,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
    ):
        sns.set_style("whitegrid")

        title = chart_metadata["title"]
        xlabel = chart_metadata["xlabel"] or "Value"
        ylabel = chart_metadata["ylabel"] or "CDF"

        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        # In error case: draw a minimal frame with error note (no lines)
        if "error" in desc:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.text(
                0.5, 0.5, f"Input error: {desc['error']}",
                ha="center", va="center", transform=ax.transAxes, color="red"
            )
            return fig, ax

        x = desc["x"]
        ecdf = desc["ecdf"]
        cdf_theo = desc["cdf_theo"]

        if desc["n"] > 0:
            ax.step(x, ecdf, where='post', label='Empirical CDF')
            ax.plot(x, cdf_theo, '--', label=f"{self.ctx.distribution_name} CDF")

            # KS max gap line
            idx_gap = int(np.argmax(np.abs(ecdf - cdf_theo)))
            ax.vlines(
                x[idx_gap], cdf_theo[idx_gap], ecdf[idx_gap],
                color='red', linewidth=1.5, label=f"KS D = {desc['ks_D']:.3f}"
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        # stats textbox (summary)
        lines = [f"n = {desc['n']}"]
        fit = desc["params"].get("distribution_fit")
        if fit is not None:
            lines.append(f"params = {tuple(np.round(fit, 3))}")
        if "ks" in inf:
            lines.append(f"KS stat = {inf['ks']['statistic']:.3f}, p = {inf['ks']['p_value']:.3f}, reject = {inf['ks']['reject']}")
        if "anderson" in inf:
            lines.append(f"AD stat = {inf['anderson']['statistic']:.3f}, crit = {inf['anderson']['critical_value']:.3f}, reject = {inf['anderson']['reject']}")
        if "cvm" in inf:
            lines.append(f"CvM stat = {inf['cvm']['statistic']:.3f}, p = {inf['cvm']['p_value']:.3f}, reject = {inf['cvm']['reject']}")

        ax.text(
            0.98, 0.02, "\n".join(lines),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize="small", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        )

        return fig, ax
