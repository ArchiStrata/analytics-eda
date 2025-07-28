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

def plot_distribution_ecdf_vs_cdf(
    series: pd.Series,
    distribution_name: Literal['norm', 'lognorm', 'gamma', 'expon'],
    title: str = "ECDF vs. Theoretical CDF",
    xlabel: str = "Value",
    ylabel: str = "CDF",
    data_source: str = None,
    alpha: float = 0.05,
    figsize: tuple = (10, 6),
    save_path: str = None,
    file_name: str = None
) -> dict:
    """
    Generate an ECDF vs. theoretical CDF plot and KS-test for any SciPy continuous distribution.

    Why:
        Visualize the fit of data to a theoretical distribution by showing
        the empirical CDF against the fitted CDF, and quantify with a KS statistic.

    What:
        - Validates data and distribution support.
        - Fits distribution parameters (shape, loc, scale).
        - Computes ECDF and theoretical CDF.
        - Runs a one-sample KS test with fitted parameters.
        - Plots ECDF (step) and theoretical CDF (dashed).
        - Annotates maximum vertical gap (D statistic) on the plot.
        - Returns descriptive stats and chart metadata.

    Parameters
    ----------
    series : pd.Series
        Numeric data to analyze; NaNs dropped.
    distribution_name : Literal['norm', 'lognorm', 'gamma', 'expon']
        Name of SciPy distribution (e.g. 'norm', 'lognorm', 'gamma', 'expon').
    title : str
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
                'params': tuple,
                'ks_statistic': float,
                'ks_p_value': float,
                'ks_reject': bool
            },
            'chart_metadata': {
                'title': str,
                'xlabel': str,
                'ylabel': str,
                'data_source': str or None,
                'distribution': str,
                'alpha': float,
                'relative_path': str or None
            }
        }
    """
    ALLOWED = ('norm','lognorm','gamma','expon')
    if distribution_name not in ALLOWED:
        raise ValueError(f"distribution_name must be one of {ALLOWED}")
    
    validate_numeric_named_series(series)
    data = series.copy().dropna().astype(float)
    n = data.size

    # empty check
    if n == 0:
        return {'descriptive_stats': {'error': 'empty series'}}

    # support checks
    mn = data.min()
    if distribution_name in ('lognorm', 'gamma') and mn <= 0:
        return {'descriptive_stats': {'error': 'requires positive data'}}
    if distribution_name == 'expon' and mn < 0:
        return {'descriptive_stats': {'error': 'requires non-negative data'}}

    # fit distribution
    dist = getattr(stats, distribution_name)
    params = dist.fit(data)

    # compute ECDF
    x = np.sort(data)
    ecdf = np.arange(1, n + 1) / n

    # theoretical CDF
    cdf_theo = dist.cdf(x, *params)

    # KS test with fitted parameters
    D, p = stats.kstest(data, distribution_name, args=params)
    reject = bool(p < alpha)

    # plotting
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(x, ecdf, where='post', label='Empirical CDF')
    ax.plot(x, cdf_theo, linestyle='--', label=f"{distribution_name} CDF")

    # annotate max vertical gap
    idx = np.argmax(np.abs(ecdf - cdf_theo))
    ax.vlines(
        x[idx],
        cdf_theo[idx],
        ecdf[idx],
        color='red',
        lw=1.5,
        label=f"D = {D:.3f}"
    )

    # labels & title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # data source
    if data_source:
        fig.text(0.01, 0.01, f"Source: {data_source}",
                 ha='left', va='bottom',
                 fontsize='small', color='gray')

    # stats textbox
    stats_text = (
        f"n = {n}\n"
        f"dist = {distribution_name}\n"
        f"params = {tuple(np.round(params, 3))}\n"
        f"KS stat = {D:.3f}\n"
        f"p-value = {p:.3f}\n"
        f"reject = {reject}"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize='small',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    ax.legend()

    # optional save
    rel_path = None
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')
        rel_path = os.path.relpath(abs_path)

    return {
        'descriptive_stats': {
            'n': n,
            'distribution': distribution_name,
            'params': params,
            'ks_statistic': D,
            'ks_p_value': p,
            'ks_reject': reject
        },
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'distribution': distribution_name,
            'alpha': alpha,
            'relative_path': rel_path
        }
    }
