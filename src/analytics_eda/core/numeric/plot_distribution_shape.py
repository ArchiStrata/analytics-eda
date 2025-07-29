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
from typing import Callable, Sequence, Literal, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks

from .binning_rules import doane_bins, freedman_diaconis_bins, scott_bins, sturges_bins
from .validate_numeric_named_series import validate_numeric_named_series

BinMethod = Literal['sturges', 'scott', 'freedman_diaconis', 'doane']

def plot_distribution_shape(
    series: pd.Series,
    title: str = "Distribution Shape: Histogram with KDE",
    xlabel: str = "Value",
    ylabel: str = "Density",
    data_source: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    file_name: str = None,
    bin_method: Optional[BinMethod] = None,
    bins: int | Sequence[float] | None = None,
    hist_alpha: float = 0.4,
    bw_adjust: float = 1.0
):
    """
    Generate a histogram overlaid with a KDE to communicate the shape of a numeric distribution.

    Why:
        Understanding a distribution’s shape—its skewness, tail‐weight, and number of peaks—reveals 
        subpopulations, asymmetries, and heavy tails that a simple histogram or boxplot may obscure.

    What:
        - Accepts a pandas Series of numeric values.
        - Plots a smooth Kernel Density Estimate with:
          • vertical lines at Q1, median (Q2), Q3
          • shaded tail regions (below 10th, above 90th percentiles)
          • markers for each local mode (peak) in the KDE
        - Annotates skewness, kurtosis, quartile skewness, and mode count.
        - Optionally saves the figure to disk.
        - Returns computed shape metrics and chart parameters.

    How:
        - Works on a cleaned copy of the data (missing values dropped).
        - Uses SciPy’s Gaussian KDE over a fine grid to estimate density.
        - Detects peaks via `scipy.signal.find_peaks`.
        - Calculates percentiles, skewness, kurtosis, and Bowley skew.
        - Renders the plot with colorblind-friendly styling and annotations.

    Parameters
    ----------
    series : pd.Series
        Numeric dataset to plot. Missing values will be dropped.
    title : str, default="KDE Plot of Distribution Shape"
        Chart title.
    xlabel : str, default="Value"
        Label for the x-axis.
    ylabel : str, default="Density"
        Label for the y-axis.
    data_source : str, optional
        Text annotation for the data source (bottom-left of figure).
    figsize : tuple, default=(10, 6)
        Figure size in inches (width, height).
    save_path : str or Path, optional
        Directory to save the image. Created if necessary.
    file_name : str, optional
        Filename (with extension) for saving. Requires `save_path`.
    bin_method: str | None = None
        The statistical binning rule used for dynamically setting bins based on the data.
    bins : int or sequence of floats
        Number of histogram bins, or the bin edges.
    hist_alpha : float
        Transparency level for the histogram bars.
    bw_adjust : float
        Bandwidth adjustment factor for KDE (relative to default).

    Returns
    -------
    metadata : dict
        {
            'descriptive_stats': {
                'n': int,                  
                'skewness': float,         
                'kurtosis': float,         
                'modes_count': int,        
                'quartile_skew': float,    
                'pct_10': float,           
                'pct_25': float,           
                'pct_50': float,           
                'pct_75': float,           
                'pct_90': float            
            },
            'chart_metadata': {
                'title': str,
                'xlabel': str,
                'ylabel': str,
                'data_source': str or None,
                'relative_path': str or None
            }
        }

    Key Shape Statistics
    --------------------
    | Statistic          | What it tells you                                                                 |
    |--------------------|-----------------------------------------------------------------------------------|
    | `skewness`         | Asymmetry of the curve: > 0 right‐skewed, < 0 left‐skewed                          |
    | `kurtosis`         | Peakedness/tail‐weight: > 0 heavy tails & sharp peak, < 0 light tails & flat peak |
    | `modes_count`      | Number of humps in the KDE—identifies subpopulations                              |
    | Quartiles (Q1,Q2,Q3) | Central bulk shape: Q2–Q1 vs. Q3–Q2 shows asymmetry; IQR (Q3–Q1) measures spread  |
    | Tail percentiles   | P10/P90 show extreme behavior: wider P90–P10 vs. IQR indicates heavy tails        |
    | `quartile_skew`    | Robust skew: (Q3+Q1−2Q2)/(Q3−Q1) focusing on the IQR                               |
    """
    validate_numeric_named_series(series)
    data = series.copy().dropna()
    n = data.size

    # Early exit for empty data
    if n == 0:
        empty_stats = dict(
            n=0, skewness=np.nan, kurtosis=np.nan, modes_count=0,
            quartile_skew=np.nan, pct_10=np.nan, pct_25=np.nan,
            pct_50=np.nan, pct_75=np.nan, pct_90=np.nan
        )
        return {
            'descriptive_stats': empty_stats,
            'chart_metadata': dict(
                title=title, xlabel=xlabel, ylabel=ylabel,
                data_source=data_source, relative_path=None
            )
        }

    # dynamically pick bins if requested
    if bin_method:
        methods: dict[BinMethod, Callable[[pd.Series], int]] = {
            'sturges': sturges_bins,
            'scott': scott_bins,
            'freedman_diaconis': freedman_diaconis_bins,
            'doane': doane_bins
        }
        try:
            bins = methods[bin_method](data)
        except KeyError as exc:
            raise ValueError(f"Unknown bin_method: {bin_method!r}. "
                             f"Choose from {list(methods)}.") from exc
    elif bins is None:
        # default fallback  
        bins = 30

    # Compute percentiles and robust quartile skew
    q1, q2, q3 = data.quantile([0.25, 0.50, 0.75])
    pct_10, pct_90 = data.quantile([0.10, 0.90])
    iqr = q3 - q1
    quartile_skew = ((q3 + q1 - 2 * q2) / iqr) if iqr != 0 else np.nan

    # Compute skewness & kurtosis
    skewness = data.skew()
    kurtosis = data.kurtosis()

    # KDE estimate with adjustable bandwidth
    kde = stats.gaussian_kde(data)
    kde.set_bandwidth(bw_method=kde.factor * bw_adjust)
    grid = np.linspace(data.min(), data.max(), 512)
    density = kde(grid)

    # Detect peaks (modes)
    peaks, _ = find_peaks(density)
    modes_count = len(peaks)
    mode_locations = grid[peaks].tolist()

    # Plot setup
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram (normalized to density)
    ax.hist(data, bins=bins, density=True,
            alpha=hist_alpha, label="Histogram")

    # KDE line
    ax.plot(grid, density, lw=2, label="KDE")

    # Shade tails
    ax.fill_between(grid, density,
                    where=(grid < pct_10), alpha=0.3,
                    label="Bottom 10%")
    ax.fill_between(grid, density,
                    where=(grid > pct_90), alpha=0.3,
                    label="Top 10%")

    # Quartile & median lines
    ax.axvline(q1, linestyle='--', label=f"Q1 = {q1:.2f}")
    ax.axvline(q2, linestyle='-', label=f"Median = {q2:.2f}")
    ax.axvline(q3, linestyle='--', label=f"Q3 = {q3:.2f}")

    # Mode markers
    if modes_count:
        heights = density[peaks]
        ax.scatter(mode_locations, heights,
                   color='green', marker='o',
                   label=f"{modes_count} mode(s)")
        for x_loc, y_loc in zip(mode_locations, heights):
            ax.text(x_loc, y_loc, f"{x_loc:.2f}",
                    ha='left', va='bottom',
                    fontsize='x-small', color='green')
    # Finalize axes
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Annotate skew/kurt/quartile_skew
    stats_text = (
        f"n = {n}\n"
        f"Skewness = {skewness:.2f}\n"
        f"Kurtosis = {kurtosis:.2f}\n"
        f"Quartile skew = {quartile_skew:.2f}"
    )
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes, ha='right', va='top',
            fontsize='small', bbox=dict(facecolor='white', alpha=0.5))

    # Optional data source
    if data_source:
        fig.text(0.01, 0.01, f"Source: {data_source}",
                 ha='left', va='bottom', fontsize='small', color='gray')

    # Legend ordering
    handles, labels = ax.get_legend_handles_labels()
    order = ["Histogram", "KDE",
             f"Q1 = {q1:.2f}", f"Median = {q2:.2f}", f"Q3 = {q3:.2f}",
             "Bottom 10%", "Top 10%",
             f"{modes_count} mode(s)"]
    ordered = [(h, l) for key in order
               for h, l in zip(handles, labels) if l == key]
    if ordered:
        h_ord, l_ord = zip(*ordered)
        ax.legend(h_ord, l_ord)
    else:
        ax.legend(handles, labels)

    # Optional save
    rel_path = None
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')
        rel_path = os.path.relpath(abs_path)

    return {
        'descriptive_stats': {
            'n': n,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'modes_count': modes_count,
            'quartile_skew': quartile_skew,
            'pct_10': pct_10,
            'pct_25': q1,
            'pct_50': q2,
            'pct_75': q3,
            'pct_90': pct_90
        },
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'relative_path': rel_path
        }
    }
