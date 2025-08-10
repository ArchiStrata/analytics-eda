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
import os
from typing import Literal
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from .validate_numeric_named_series import validate_numeric_named_series
from ..utils.build_chart_title import build_chart_title

def plot_distribution_qq_fit(
    series: pd.Series,
    distribution_name: Literal['norm', 'lognorm', 'gamma', 'expon'],
    title_template: str = "Q–Q Plot Fit Assessment of {name}{modifiers}",
    name: str = None,
    filter_desc: str = None,
    transform_desc: str = None,
    xlabel: str = "Theoretical Quantiles",
    ylabel: str = "Sample Quantiles",
    data_source: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    file_name: str = None,
    alpha: float = 0.05
) -> dict:
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

    Parameters
    ----------
    series : pd.Series
        Numeric data to assess; NaNs will be dropped.
    distribution_name : Literal['norm', 'lognorm', 'gamma', 'expon']
        Name of SciPy distribution (e.g. 'norm', 'lognorm', 'gamma', 'expon').
    title_template: A Python format-string with placeholders:
      - {name}:        series name or label
      - {modifiers}:   combined filter/transform text, empty if none
    name:                Optional override for series.name
    filter_desc:         e.g. "filtered by New York"
    transform_desc:      e.g. "log-transformed"
    xlabel : str, default="Theoretical Quantiles"
        Label for the x-axis.
    ylabel : str, default="Sample Quantiles"
        Label for the y-axis.
    data_source : str, optional
        Annotation for the data source.
    figsize : tuple, default=(10, 6)
        Figure size in inches.
    save_path : str, optional
        Directory to save the plot; created if needed.
    file_name : str, optional
        Filename (with extension) for saving; requires `save_path`.
    alpha : float
        Significance level for all formal tests.

    Returns
    -------
    dict
        {
            'descriptive_stats': {
                'intercept': float,
                'slope': float,
                'r_squared': float,
                'median_residual': float,
                'iqr_residual': float,
                'max_abs_residual': float,
                'skewness': float,
                'kurtosis': float,
                'min': float
            },
            'inferential_stats': {
                'params': {
                    'alpha': float
                    'distribution_name': str
                }
                # only present if distribution_name == 'norm'
                    'shapiro': {...},               # present if n < 50
                    'dagostino_pearson': {...},     # present if n ≥ 20
                    'jarque_bera': {...},           # present if n > 2000
                    'reject_normality': bool
            }
            'chart_metadata': {
                'title': str,
                'xlabel': str,
                'ylabel': str,
                'data_source': str or None,
                'file_name': str or None
            }
        }
    """
    ALLOWED = ('norm','lognorm','gamma','expon')
    if distribution_name not in ALLOWED:
        raise ValueError(f"distribution_name must be one of {ALLOWED}")

    validate_numeric_named_series(series)
    data = series.copy().dropna().astype(float)
    n = data.size

    title = build_chart_title(
                    name=name, series=series,
                    filter_desc=filter_desc,
                    transform_desc=transform_desc,
                    title_template=title_template
                )
    full_title = f"{title} ({distribution_name})"

    # early return for empty series
    if n == 0:
        empty_stats = {
            'intercept': np.nan,
            'slope': np.nan,
            'r_squared': np.nan,
            'median_residual': np.nan,
            'iqr_residual': np.nan,
            'max_abs_residual': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'min': np.nan
        }
        return {
            'descriptive_stats': empty_stats,
            'inferential_stats': {
                'params': {
                    'alpha': alpha,
                    'distribution_name': distribution_name
                }
            },
            'chart_metadata': {
                'title': full_title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'data_source': data_source,
                'file_name': None
            }
        }

    # fit the distribution to the data
    dist = getattr(stats, distribution_name)   # e.g. stats.lognorm, stats.gamma, etc.
    params = dist.fit(data)                    # for lognorm: (shape, loc, scale); for norm: (loc, scale); etc.

    # build the “theoretical” quantiles
    #    use plotting positions (i-0.5)/n – a common unbiased choice
    probs = (np.arange(1, n+1) - 0.5) / n

    #    unpack shape-args vs loc/scale
    *shape_args, loc, scale = params
    osm = dist.ppf(probs, *shape_args, loc=loc, scale=scale)

    # sample quantiles
    osr = np.sort(data)

    # linear fit of sample vs theoretical
    slope, intercept = np.polyfit(osm, osr, 1)
    fitted = intercept + slope * osm

    # R²:
    corr = np.corrcoef(osr, fitted)[0,1]
    r_squared = corr**2

    # residual diagnostics
    residuals = osr - fitted
    median_residual = float(np.median(residuals))
    iqr_residual = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
    max_abs_residual = float(np.max(np.abs(residuals)))

    # shape metrics
    skewness = float(stats.skew(data, bias=False))
    kurtosis = float(stats.kurtosis(data, fisher=True, bias=False))

    tests = {}
    tests['params'] = {
        'alpha': alpha,
        'distribution_name': distribution_name
    }

    # normality tests per size rules
    if distribution_name == 'norm':
        # 1. Shapiro–Wilk for n < 50
        if n < 50:
            stat_sw, p_sw = stats.shapiro(data)
            tests['shapiro'] = {
                'statistic': float(stat_sw),
                'p_value': float(p_sw),
                'reject': bool(p_sw < alpha)
            }

        # 2. D’Agostino–Pearson omnibus for n ≥ 20
        if n >= 20:
            stat_dp, p_dp = stats.normaltest(data)
            tests['dagostino_pearson'] = {
                'statistic': float(stat_dp),
                'p_value': float(p_dp),
                'reject': bool(p_dp < alpha)
            }

        # 3. Jarque–Bera for n > 2000
        if n > 2000:
            stat_jb, p_jb = stats.jarque_bera(data)
            tests['jarque_bera'] = {
                'statistic': float(stat_jb),
                'p_value': float(p_jb),
                'reject': bool(p_jb < alpha)
            }

        # top‐level reject flag
        tests['reject_normality'] = any(v.get('reject', False) for v in tests.values())

    # build plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(x=osm, y=osr, ax=ax, s=20, edgecolor="k", alpha=0.6)
    ax.plot(osm, intercept + slope * osm, color="red", lw=1, label="Fit line")

    ax.set_title(full_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if data_source:
        fig.text(0.01, 0.01, f"Source: {data_source}",
                 ha="left", va="bottom", fontsize="small", color="gray")
    ax.legend()

    # annotate diagnostics and tests
    lines = [
        f"α (intercept): {intercept:.2f}",
        f"β (slope): {slope:.2f}",
        f"R²: {r_squared:.3f}",
        f"Median resid: {median_residual:.2f}",
        f"IQR resid: {iqr_residual:.2f}",
        f"Max abs resid: {max_abs_residual:.2f}",
        f"Skewness: {skewness:.2f}",
        f"Excess kurtosis: {kurtosis:.2f}"
    ]

    # only if we're doing a normal Q–Q do we add those tests  
    if distribution_name == 'norm' and tests:
        lines.append("")  # blank line before tests
        for name, info in tests.items():
            if name == 'params':
                continue
            if name == 'reject_normality':
                # final summary flag
                lines.append(f"Overall reject: {info}")
            else:
                p   = info.get('p_value', np.nan)
                lines.append(f"{name}: stat={info['statistic']:.3f}, p={p:.3f}, reject={info['reject']}")

    stats_text = "\n".join(lines)


    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, ha="left", va="top",
            fontsize="small", bbox=dict(facecolor="white", alpha=0.5))

    # optional save
    if save_path:
        if file_name is None:
            file_name = f"{title}.png"
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches="tight")

    return {
        'descriptive_stats': {
            'intercept': intercept,
            'slope': slope,
            'r_squared': r_squared,
            'median_residual': median_residual,
            'iqr_residual': iqr_residual,
            'max_abs_residual': max_abs_residual,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min': data.min() # return min to support selecting transforms
        },
        'inferential_stats': tests,
        'chart_metadata': {
            'title': full_title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'file_name': file_name
        }
    }
