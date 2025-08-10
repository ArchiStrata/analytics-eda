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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde

from .validate_numeric_named_series import validate_numeric_named_series
from ..utils.build_chart_title import build_chart_title


def plot_distribution_probability_function(
    series: pd.Series,
    is_discrete: bool,
    bw_method='scott',
    title_template: str = "PMF of {name}{modifiers}" if True else "PDF estimate of {name}{modifiers}",
    name: str = None,
    filter_desc: str = None,
    transform_desc: str = None,
    xlabel: str = "Value",
    ylabel: str = None,  # Set based on discrete/continuous
    data_source: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    file_name: str = None
):
    """
    Plots an explicit Probability Mass Function (PMF) for discrete data or an explicit Probability Density Function (PDF) estimate for continuous data.

    Parameters
    ----------
    series : pandas.Series
        The data series to visualize.
    is_discrete : bool
        If True, plot the PMF; if False, plot the PDF estimate.
    bw_method : str or float
        Bandwidth method for gaussian_kde (e.g., 'scott', 'silverman', or a scalar).
    name : str, optional
        Label to use for the x-axis/title (defaults to series.name).
    data_source : str, optional
        Annotation for data source to display on figure.
    figsize : tuple, optional
        Figure size passed to plt.subplots().
    save_path : str, optional
        Directory to save the figure.
    file_name : str, optional
        Filename for saving the figure.
    """
    validate_numeric_named_series(series)
    vals = series.copy().dropna()

    # Decide ylabel based on discrete or continuous
    if ylabel is None:
        ylabel = "Probability P(X = x)" if is_discrete else "Density f(x)"

    # Build chart title
    chart_title = build_chart_title(
        name=name or series.name or "Value",
        series=series,
        filter_desc=filter_desc,
        transform_desc=transform_desc,
        title_template=title_template
    )

    # If no valid values, return defaults
    if vals.empty:
        return {
            'descriptive_stats': {
                'n': 0,
                'mean': np.nan,
                'median': np.nan,
                'mode': np.nan,
                'variance': np.nan,
                'std': np.nan,
                'iqr': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'min': np.nan,
                'max': np.nan
            },
            'chart_metadata': {
                'title': chart_title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'data_source': data_source,
                'file_name': None
            }
        }

    descriptive_stats = {
        'n':          int(vals.size),
        'mean':       float(vals.mean()),
        'median':     float(vals.median()),
        'mode':       float(vals.mode().iloc[0]),
        'variance':   float(vals.var()),
        'std':        float(vals.std()),
        'iqr':        float(vals.quantile(0.75) - vals.quantile(0.25)),
        'skewness':   float(vals.skew()),
        'kurtosis':   float(vals.kurtosis()),
        'min':        float(vals.min()),
        'max':        float(vals.max()),
    }

    # Prepare plot
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)

    if is_discrete:
        counts = vals.value_counts().sort_index()
        pmf = counts / counts.sum()
        ax.bar(pmf.index, pmf.values, edgecolor='black')
    else:
        kde = gaussian_kde(vals, bw_method=bw_method)
        x_grid = np.linspace(vals.min(), vals.max(), 200)
        pdf_vals = kde(x_grid)
        ax.plot(x_grid, pdf_vals, linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(chart_title)

    # Optional data source annotation
    if data_source:
        fig.text(
            0.01, 0.01, f"Source: {data_source}",
            ha='left', va='bottom',
            fontsize='small', color='gray'
        )

    plt.tight_layout()

    # Optional save
    if save_path:
        if file_name is None:
            file_name = f"{chart_title}.png"
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')

    # Return metadata
    return {
        'descriptive_stats': descriptive_stats,
        'chart_metadata': {
            'title': chart_title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'file_name': file_name
        }
    }
