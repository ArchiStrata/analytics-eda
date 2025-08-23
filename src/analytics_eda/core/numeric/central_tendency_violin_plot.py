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
from typing import Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..utils.base_plot import BasePlot, PlotContext
from .validate_numeric_named_series import NumericSeriesMixin

MeanCIMethod = Literal['t', 'bootstrap']
MedianCIMethod = Literal['bootstrap', None]

@dataclass
class CentralTendencyViolinContext(PlotContext):
    title_template: str = "Distribution of {name}{modifiers}: Central Tendency (Violin)"
    xlabel: str = "Value"
    ylabel: str = "Density"

    # plot-specific
    mean_ci_method: MeanCIMethod = "t"
    median_ci_method: MedianCIMethod = "bootstrap"
    alpha: float = 0.05
    bootstrap_samples: int = 1_000

    # optional population params for tests
    popmean: Optional[float] = None
    popmedian: Optional[float] = None
    popvariance: Optional[float] = None

class CentralTendencyViolinPlot(NumericSeriesMixin, BasePlot):
    """
    Violin distribution with mean/median and CIs, plus optional inferential tests.

    Tests shown when popmean and/or popmedian and/or popvariance are not None:
      - Cohen’s d, one-sample t-test, and (if popvariance) one-sample Z-test for popmean
      - Wilcoxon signed-rank and sign test for popmedian

    Returns BasePlot.run() schema:
      {
        "descriptive_stats": {
          "params": {"mean_ci_method", "median_ci_method"},
          "n", "mean", "median", "mean_ci", "median_ci"
        },
        "inferential_stats": { ... tests ... },
        "chart_metadata": {...}
      }
    """

    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "params": {
                "mean_ci_method": self.ctx.mean_ci_method,
                "median_ci_method": self.ctx.median_ci_method,
            },
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "mean_ci": (float("nan"), float("nan")),
            "median_ci": (float("nan"), float("nan")),
        }

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        n = int(s.size)
        mean = float(s.mean()) if n else float("nan")
        median = float(s.median()) if n else float("nan")

        # Mean CI
        if n:
            if self.ctx.mean_ci_method == "t":
                sem = stats.sem(s, ddof=1)
                ci_low, ci_high = stats.t.interval(
                    1 - self.ctx.alpha, df=n - 1, loc=mean, scale=sem
                )
                mean_ci = (float(ci_low), float(ci_high))
            elif self.ctx.mean_ci_method == "bootstrap":
                rng = np.random.default_rng()
                boot_means = rng.choice(s.to_numpy(), size=(self.ctx.bootstrap_samples, n), replace=True).mean(axis=1)
                lo, hi = np.percentile(boot_means, [100 * self.ctx.alpha / 2, 100 * (1 - self.ctx.alpha / 2)])
                mean_ci = (float(lo), float(hi))
            else:
                raise ValueError("mean_ci_method must be 't' or 'bootstrap'")
        else:
            mean_ci = (float("nan"), float("nan"))

        # Median CI
        if n and self.ctx.median_ci_method == "bootstrap":
            rng = np.random.default_rng()
            boot = rng.choice(s.to_numpy(), size=(self.ctx.bootstrap_samples, n), replace=True)
            boot_meds = np.median(boot, axis=1)
            lo, hi = np.percentile(boot_meds, [100 * self.ctx.alpha / 2, 100 * (1 - self.ctx.alpha / 2)])
            median_ci = (float(lo), float(hi))
        elif self.ctx.median_ci_method is None:
            median_ci = (float("nan"), float("nan"))
        else:
            # Guard against unexpected value
            raise ValueError("median_ci_method must be 'bootstrap' or None")

        return {
            "params": {
                "mean_ci_method": self.ctx.mean_ci_method,
                "median_ci_method": self.ctx.median_ci_method,
            },
            "n": n,
            "mean": mean,
            "median": median,
            "mean_ci": mean_ci,
            "median_ci": median_ci,
        }
    
    def default_inferential(self) -> Dict[str, Any]:
        return {
            "params": {
                "alpha": self.ctx.alpha,
                "bootstrap_samples": self.ctx.bootstrap_samples,
                "popmean": self.ctx.popmean,
                "popmedian": self.ctx.popmedian,
                "popvariance": self.ctx.popvariance,
            }
        }

    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "params": {
                "alpha": self.ctx.alpha,
                "bootstrap_samples": self.ctx.bootstrap_samples,
                "popmean": self.ctx.popmean,
                "popmedian": self.ctx.popmedian,
                "popvariance": self.ctx.popvariance,
            }
        }

        # Tests for mean
        if self.ctx.popmean is not None:
            out["popmean"] = {}

            # One-Sample Cohen's d
            sd = float(s.std(ddof=1))
            cohens_d = (desc["mean"] - self.ctx.popmean) / sd if sd != 0 else None
            
            # One-Sample t-Test
            t_stat, t_p = stats.ttest_1samp(s, self.ctx.popmean)
            out["popmean"]["cohens_d"] = None if cohens_d is None else float(cohens_d)
            out["popmean"]["t_test"] = {
                "statistic": float(t_stat),
                "p_value": float(t_p),
                "reject": bool(t_p < self.ctx.alpha),
            }

            # One-sample Z-test (requires known σ²)
            if self.ctx.popvariance is not None:
                sigma = float(np.sqrt(self.ctx.popvariance))
                z_stat = (desc["mean"] - self.ctx.popmean) / (sigma / np.sqrt(desc['n']))
                z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                out["popmean"]["z_test"] = {
                    "statistic": float(z_stat),
                    "p_value": float(z_p),
                    "reject": bool(z_p < self.ctx.alpha),
                }

        # Tests for median
        if self.ctx.popmedian is not None:
            out["popmedian"] = {}

            # Wilcoxon Signed-Rank Test
            diff = s - self.ctx.popmedian
            stat_wr, p_wr = stats.wilcoxon(diff)
            out["popmedian"]["wilcoxon"] = {
                "statistic": float(stat_wr),
                "p_value": float(p_wr),
                "reject": bool(p_wr < self.ctx.alpha),
            }

            # Sign test (binomial on signs)
            nonzero = diff[diff != 0]
            n_sign = int(nonzero.size)
            if n_sign > 0:
                pos = int((nonzero > 0).sum())
                sign_res = stats.binomtest(pos, n_sign, p=0.5)
                out["popmedian"]["sign_test"] = {
                    "num_positive": pos,
                    "num_negative": n_sign - pos,
                    "n": n_sign,
                    "p_value": float(sign_res.pvalue),
                    "reject": bool(sign_res.pvalue < self.ctx.alpha),
                }

        return out

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
        mean_col, med_col = palette[0], palette[1]

        # Horizontal violin
        sns.violinplot(x=s, orient="h", inner=None, color="lightgray", ax=ax)

        # Overlay mean and its CI
        mean = desc["mean"]; mean_lo, mean_hi = desc["mean_ci"]
        if desc["n"] > 0 and np.isfinite(mean):
            if np.isfinite(mean_lo) and np.isfinite(mean_hi):
                ax.errorbar(
                    x=mean, y=0,
                    xerr=[[mean - mean_lo], [mean_hi - mean]],
                    fmt="o", capsize=5, color=mean_col,
                    label=f"Mean CI ({self.ctx.mean_ci_method}, {int((1 - self.ctx.alpha) * 100)}%)",
                )
            ax.axvline(mean, color=mean_col, linestyle="--", label=f"Mean = {mean:.2f}")

        # Overlay median and its CI (if requested)
        if desc["n"] > 0 and self.ctx.median_ci_method == "bootstrap":
            med = desc["median"]; med_lo, med_hi = desc["median_ci"]
            if np.isfinite(med):
                if np.isfinite(med_lo) and np.isfinite(med_hi):
                    ax.errorbar(
                        x=med, y=0,
                        xerr=[[med - med_lo], [med_hi - med]],
                        fmt="s", capsize=5, color=med_col,
                        label=f"Median CI (bootstrap, {int((1 - self.ctx.alpha) * 100)}%)",
                    )
                ax.axvline(med, color=med_col, linestyle="-.", label=f"Median = {med:.2f}")

        # Labels/title
        ax.set_yticks([])

        # Compact stats summary textbox (when tests were run)
        stats_lines = []
        pm = inf.get("popmean"); pmed = inf.get("popmedian")
        if pm is not None:
            parts = []
            if "cohens_d" in pm and pm["cohens_d"] is not None:
                parts.append(f"d={pm['cohens_d']:.2f}")
            if "t_test" in pm:
                parts.append(f"t={pm['t_test']['statistic']:.2f}, p={pm['t_test']['p_value']:.3f}"
                             f" {'(reject)' if pm['t_test']['reject'] else '(ns)'}")
            if "z_test" in pm:
                parts.append(f"z={pm['z_test']['statistic']:.2f}, p={pm['z_test']['p_value']:.3f}"
                             f" {'(reject)' if pm['z_test']['reject'] else '(ns)'}")
            if parts: stats_lines.append("Mean vs pop: " + "; ".join(parts))
        if pmed is not None:
            parts = []
            if "wilcoxon" in pmed:
                parts.append(f"W={pmed['wilcoxon']['statistic']:.2f}, p={pmed['wilcoxon']['p_value']:.3f}"
                             f" {'(reject)' if pmed['wilcoxon']['reject'] else '(ns)'}")
            if "sign_test" in pmed:
                st = pmed["sign_test"]
                parts.append(f"Sign: +={st['num_positive']}, -={st['num_negative']}, p={st['p_value']:.3f}"
                             f" {'(reject)' if st['reject'] else '(ns)'}")
            if parts: stats_lines.append("Median vs pop: " + "; ".join(parts))
        if stats_lines:
            ax.text(0.01, 0.95, "\n".join(stats_lines), transform=ax.transAxes,
                    fontsize="small", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

        # Sample size footer (BasePlot will add data_source if present)
        fig.text(0.99, 0.01, f"n = {desc['n']}", ha="right", va="bottom",
                 fontsize="small", color="gray")

        ax.legend()
        return fig, ax
