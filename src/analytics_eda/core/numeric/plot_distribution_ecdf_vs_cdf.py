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

def plot_distribution_ecdf_vs_cdf(
    series: pd.Series,
    distribution_name: Literal['norm', 'lognorm', 'gamma', 'expon'],
    title_template: str = "ECDF vs. Theoretical CDF of {name}{modifiers}",
    name: str = None,
    filter_desc: str = None,
    transform_desc: str = None,
    xlabel: str = "Value",
    ylabel: str = "CDF",
    data_source: str = None,
    alpha: float = 0.05,
    figsize: tuple = (10, 6),
    save_path: str = None,
    file_name: str = None
) -> dict:
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

    Parameters
    ----------
    series : pd.Series
        Numeric data to analyze; NaNs dropped.
    distribution_name : Literal['norm', 'lognorm', 'gamma', 'expon']
        Name of SciPy distribution (e.g. 'norm', 'lognorm', 'gamma', 'expon').
    title_template: A Python format-string with placeholders:
      - {name}:        series name or label
      - {modifiers}:   combined filter/transform text, empty if none
    name:                Optional override for series.name
    filter_desc:         e.g. "filtered by New York"
    transform_desc:      e.g. "log-transformed"
    xlabel : str
    ylabel : str
    data_source : str, optional
    alpha : float
        Significance level for KS test.
    figsize : tuple
    save_path : str, optional
    file_name : str, optional

    Returns
    -------
    dict
        {
            'descriptive_stats': {
                'n': int,
                'distribution': str,
                'params': tuple
            },
            'inferential_stats': {
               'params': {
                    'alpha': float
               },
               'ks': {
                    'statistic': float,
                    'p_value': float,
                    'reject': bool
               },
               'anderson': {
                    'statistic': float,
                    'critical_value': float,
                    'reject': bool
               },
               'cvm': {
                    'statistic': float,
                    'p_value': float,
                    'reject': bool
               }
            }
            'chart_metadata': {
                'title': str,
                'xlabel': str,
                'ylabel': str,
                'data_source': str or None,
                'distribution': str,
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

    default_metadata = {
            'descriptive_stats': {'n': n},
            'inferential_stats': {
                'params': {
                    'alpha': alpha
                }
            },
            'chart_metadata': {
                'title': full_title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'distribution': distribution_name,
                'data_source': data_source,
                'file_name': None
            }
        }

    # Empty series
    if n == 0:
        return default_metadata

    # support checks
    mn = data.min()
    if distribution_name in ('lognorm', 'gamma') and mn <= 0:
        default_metadata['descriptive_stats']['error'] = 'requires positive data'
        return default_metadata
    if distribution_name == 'expon' and mn < 0:
        default_metadata['descriptive_stats']['error'] = 'requires non-negative data'
        return default_metadata

    # fit distribution
    dist = getattr(stats, distribution_name)
    params = dist.fit(data)
    params_float = tuple(float(np.round(p, 3)) for p in params)

    # compute ECDF
    x = np.sort(data)
    ecdf = np.arange(1, n + 1) / n

    # theoretical CDF
    cdf_theo = dist.cdf(x, *params)

    # Tests
    tests = {}
    tests['params'] = {
        'alpha': alpha
    }

    # 1. KS
    D, p_ks = stats.kstest(data, distribution_name, args=params)
    tests['ks'] = {'statistic': float(D), 'p_value': float(p_ks), 'reject': bool(p_ks < alpha)}

    # 2. Anderson–Darling (only norm & expon)
    if distribution_name in ('norm','expon'):
        ad = stats.anderson(data, dist=distribution_name)
        # find critical for alpha
        levels = np.array(ad.significance_level)/100.0
        idx = np.argmin(np.abs(levels - alpha))
        crit = ad.critical_values[idx]
        tests['anderson'] = {
            'statistic': float(ad.statistic),
            'critical_value': float(crit),                     # the one matched to α
            'critical_values': list(map(float, ad.critical_values)),  # full array
            'significance_levels': list(map(float, ad.significance_level)),  # in percent
            'reject': bool(ad.statistic > crit)
        }

    # 3. Cramér–von Mises
    cvm_res = stats.cramervonmises(data, distribution_name, args=params)
    tests['cvm'] = {
        'statistic': float(cvm_res.statistic),
        'p_value': float(cvm_res.pvalue),
        'reject': bool(cvm_res.pvalue < alpha)
    }

    # Plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(x, ecdf, where='post', label='Empirical CDF')
    ax.plot(x, cdf_theo, '--', label=f"{distribution_name} CDF")

    # Max gap line
    idx_gap = np.argmax(np.abs(ecdf - cdf_theo))
    ax.vlines(x[idx_gap], cdf_theo[idx_gap], ecdf[idx_gap], color='red', linewidth=1.5,
              label=f"KS D = {D:.3f}")

    # Title
    ax.set_title(full_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    # data source
    if data_source:
        fig.text(0.01, 0.01, f"Source: {data_source}",
                 ha='left', va='bottom',
                 fontsize='small', color='gray')

    # Stats textbox
    lines = [
        f"n = {n}",
        f"params = {params_float}",
    ]
    # Append each test summary
    for name, info in tests.items():
        if name == 'params':
            continue
        if name == 'anderson':
            lines.append(f"AD stat = {info['statistic']:.3f}, crit = {info['critical_value']:.3f}, reject = {info['reject']}")
        else:
            lines.append(f"{name.upper()} stat = {info['statistic']:.3f}, p = {info['p_value']:.3f}, reject = {info['reject']}")
    stats_text = "\n".join(lines)
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize="small", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # optional save
    if save_path:
        if file_name is None:
            file_name = f"{title}.png"
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')

    return {
        'descriptive_stats': {
            'n': n,
            'distribution': distribution_name,
            'params': params_float
        },
        'inferential_stats': tests,
        'chart_metadata': {
            'title': full_title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'distribution': distribution_name,
            'file_name': file_name
        }
    }
