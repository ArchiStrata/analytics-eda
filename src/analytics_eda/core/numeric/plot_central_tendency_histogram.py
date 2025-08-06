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
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils.build_chart_title import build_chart_title
from .validate_numeric_named_series import validate_numeric_named_series

def plot_central_tendency_histogram(
    series: pd.Series,
    bins: int = None,
    title_template: str = "Distribution of {name}{modifiers}: Central Tendency",
    name: str = None,
    filter_desc: str = None,
    transform_desc: str = None,
    xlabel: str = "Value",
    ylabel: str = "Count",
    data_source: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    file_name: str = None
):
    """
    Generate a histogram that effectively communicates the central tendency of a numeric variable.

    Why:
        This function enables analysts and data storytellers to visually communicate the distribution and central tendency of a numeric variable. By directly annotating key statistics—mean, median, mode, and a 95% confidence interval—on the histogram, the chart becomes clearer, more informative, and easier to interpret.

    What:
        - Accepts a pandas Series of numeric values.
        - Plots a histogram with annotated vertical lines for mean, median, mode(s), and 95% CI.
        - Optionally saves the figure to disk.
        - Returns descriptive statistics and chart metadata for reporting or reproducibility.

    How:
        - Operates on a cleaned copy of the data (missing values dropped).
        - Uses the Square-Root Choice rule to determine bin count when unspecified. ceil(sqrt(n)).
        - Computes central tendency statistics and overlays them on the histogram.
        - Allows customization of chart titles, labels, data source, and export options.

    Parameters
    ----------
    series : pd.Series
        Numeric dataset to plot. Missing values will be dropped.
    bins : int or sequence, optional
        Number of histogram bins or explicit bin edges. Defaults to Square-Root Choice ceil(sqrt(n)).
    title_template: A Python format-string with placeholders:
      - {name}:        series name or label
      - {modifiers}:   combined filter/transform text, empty if none
    name:                Optional override for series.name
    filter_desc:         e.g. "filtered by New York"
    transform_desc:      e.g. "log-transformed"
    xlabel : str, default="Value"
        Label for the x-axis.
    ylabel : str, default="Count"
        Label for the y-axis.
    data_source : str, optional
        Text annotation to show the source of the data in the chart.
    figsize : tuple, default=(10, 6)
        Width and height of the figure in inches. Useful for layout control.
    save_path : str or Path, optional
        Directory where the plot image will be saved. Created if it doesn't exist.
    file_name : str, optional
        Name of the image file (e.g., "histogram.png"). Must be used with `save_path`.

    Returns
    -------
    metadata : dict
        {
            'descriptive_stats': {
                'n': int,                        # see table below
                'mean': float,
                'median': float,
                'modes': list of float
            },
            'chart_metadata': {
                'title': str,
                'xlabel': str,
                'ylabel': str,
                'data_source': str or None,
                'bins': int or sequence,
                'file_name': str or None
            }
        }

    Key Descriptive Statistics
    --------------------------
    | Statistic | What it tells you                                            |
    |-----------|--------------------------------------------------------------|
    | `n`       | Sample size – number of observations                         |
    | `mean`    | Arithmetic average – balance point of the distribution       |
    | `median`  | 50th percentile – midpoint, robust to outliers               |
    | `modes`   | Most frequent value(s)                                       |
    """
    validate_numeric_named_series(series)
    series_clean = series.copy().dropna()
    n = series_clean.size

    # Build chart title
    title = build_chart_title(
        name=name,
        series=series,
        filter_desc=filter_desc,
        transform_desc=transform_desc,
        title_template=title_template
    )

    # Determine bins via Square-Root choice if not specified
    if bins is None:
        bins = math.ceil(math.sqrt(n))

    # If no data after cleaning, return defaults
    if n == 0:
        return {
            'descriptive_stats': {
                'n': 0,
                'mean': np.nan,
                'median': np.nan,
                'modes': []
            },
            'chart_metadata': {
                'title': title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'data_source': data_source,
                'bins': bins,
                'file_name': file_name
            }
        }

    # Compute descriptive statistics
    mean = series_clean.mean()
    median = series_clean.median()

    raw_modes = series_clean.mode().tolist()
    if len(raw_modes) == 1:
        # A clear single mode in the data → use it
        mode_vals = raw_modes
    else:
        # Ambiguous or multimodal → use histogram‐based bin centers
        # Calculate mode based on most frequent bins
        counts, edges = np.histogram(series_clean, bins=bins, density=False)
        first_top = np.argmax(counts)
        max_count = counts[first_top]
        top_bins = np.where(counts == max_count)[0]
        mode_vals = [
            0.5 * (edges[i] + edges[i+1])
            for i in top_bins
        ]

    # Prepare plot
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(series_clean, bins=bins, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Plot mean and median lines
    ax.axvline(mean, color='black', linestyle='--', label=f"Mean = {mean:.2f}")
    ax.axvline(median, color='firebrick', linestyle='-.', label=f"Median = {median:.2f}")

    # Plot mode lines
    for i, center in enumerate(mode_vals, start=1):
        label = "Mode" if len(mode_vals) == 1 else f"Mode {i}"
        ax.axvline(
            x=center,
            color='green',
            linestyle=':',
            linewidth=1,
            label=f"{label} ≈ {center:.2f}"
        )

    # Optional data source annotation
    if data_source:
        fig.text(
            0.01, 0.01, f"Source: {data_source}",
            ha='left', va='bottom',
            fontsize='small', color='gray'
        )

    # Sample size annotation in bottom-right
    fig.text(
        0.99, 0.01, f"n = {n}",
        ha='right', va='bottom',
        fontsize='small', color='gray'
    )

    ax.legend()

    # Optional save
    if save_path:
        if file_name is None:
            file_name = f"{title}.png"
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')

    # Return metadata
    return {
        'descriptive_stats': {
            'n': n,
            'mean': mean,
            'median': median,
            'modes': mode_vals
        },
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'bins': bins,
            'file_name': file_name
        }
    }
