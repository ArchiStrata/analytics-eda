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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .validate_numeric_named_series import validate_numeric_named_series
from ..utils.build_chart_title import build_chart_title

def plot_dispersion_boxplot(
    series: pd.Series,
    title_template: str = "Dispersion of {name}{modifiers} (IQR & Outliers)",
    name: str = None,
    filter_desc: str = None,
    transform_desc: str = None,
    ylabel: str = "Value",
    data_source: str = None,
    figsize: tuple = (8, 6),
    std_outlier_multiplier: float = 4.0,
    save_path: str = None,
    file_name: str = None
):
    """
    Generate a notched boxplot (with violin silhouette) that effectively communicates
    the dispersion of a numeric variable, flagging extreme values and returning
    key statistics including a 95% CI around the median.

    Why:
        Understanding the spread of a dataset is essential for identifying variability, outliers, and patterns 
        that aren't evident from central tendency alone. This function helps analysts and data storytellers 
        visually and numerically communicate how values are distributed and dispersed in a dataset.

    What:
        - Accepts a pandas Series of numeric values.
        - Plots a boxplot with visual annotations that highlight data dispersion.
        - Computes and returns key dispersion metrics: standard deviation, variance, range, MAD (mean absolute deviation), coefficient of variation, and select percentiles.
        - Optionally includes data source annotation, saves the plot, and returns metadata for reproducibility.

    How:
        - Cleans the data by dropping missing values.
        - Uses seaborn to generate a vertical boxplot with colorblind-friendly styling.
        - Adds annotated statistics to the chart to support effective data storytelling.
        - Allows layout customization via figsize and image export through save_path and file_name.

    Parameters
    ----------
    series : pd.Series
        Numeric dataset to plot. Missing values will be dropped.
    title_template: A Python format-string with placeholders:
      - {name}:        series name or label
      - {modifiers}:   combined filter/transform text, empty if none
    name:                Optional override for series.name
    filter_desc:         e.g. "filtered by New York"
    transform_desc:      e.g. "log-transformed"
    ylabel : str, default="Value"
        Label for the y-axis.
    data_source : str, optional
        Text annotation to show the source of the data in the chart.
    figsize : tuple, default=(8, 6)
        Width and height of the figure in inches.
    std_outlier_multiplier : float, default=4.0
        How many σ away from the mean to flag extremes.
    save_path : str or Path, optional
        Directory where the plot image will be saved. Created if it doesn't exist.
    file_name : str, optional
        Name of the image file (e.g., "boxplot.png"). Must be used with `save_path`.

    Returns
    -------
    metadata : dict
        {
            'descriptive_stats': {
                'n': int,           # see table below
                'mean': float,
                'std': float,
                'var': float,
                'min': float,
                'max': float,
                'range': float,
                'mad': float,
                'cv': float,
                'pct_10': float,
                'pct_25': float,
                'pct_75': float,
                'pct_90': float,
                'iqr': float,
                'median_ci': tuple(float, float),
                'extreme_lower_count': int,
                'extreme_upper_count': int
            },
            'chart_metadata': {
                'title': str,
                'ylabel': str,
                'data_source': str or None,
                'file_name': str or None,
                'std_outlier_multiplier': float
            }
        }

    Key Descriptive Statistics
    --------------------------
    | Statistic | What it tells you                                    |
    |-----------|------------------------------------------------------|
    | `n`       | Sample size (number of observations)                 |
    | `std`     | Standard deviation: typical distance from the mean   |
    | `var`     | Variance: squared average deviation                  |
    | `min`/`max` | Extremes of the data range                         |
    | `range`   | Span of values (max − min)                           |
    | `mad`     | Mean absolute deviation from the mean                |
    | `cv`      | Coefficient of variation (std / mean)                |
    | `pct_10`  | 10th percentile: lower‐tail threshold                |
    | `pct_25`  | 25th percentile (Q1): first quartile                 |
    | `pct_75`  | 75th percentile (Q3): third quartile                 |
    | `pct_90`  | 90th percentile: upper‐tail threshold                |
    | `iqr`     | Interquartile range (IQR)  measures how “wide” the central half of your data is, ignoring the lowest 25 % and highest 25 %. |
    """
    validate_numeric_named_series(series)
    series_clean = series.copy().dropna()
    n = series_clean.size

    title = build_chart_title(
                    name=name, series=series,
                    filter_desc=filter_desc,
                    transform_desc=transform_desc,
                    title_template=title_template
                )

    # Early return on empty series
    if n == 0:
        empty_stats = {
            'n': 0,
            'mean': np.nan,
            'std': np.nan,
            'var': np.nan,
            'min': np.nan,
            'max': np.nan,
            'range': np.nan,
            'mad': np.nan,
            'cv': np.nan,
            'pct_10': np.nan,
            'pct_25': np.nan,
            'pct_75': np.nan,
            'pct_90': np.nan,
            'iqr': np.nan,
            'extreme_lower_count': 0,
            'extreme_upper_count': 0,
            'median_ci': (np.nan, np.nan)
        }
        return {
            'descriptive_stats': empty_stats,
            'chart_metadata': {
                'title': title,
                'ylabel': ylabel,
                'data_source': data_source,
                'file_name': None,
                'std_outlier_multiplier': std_outlier_multiplier
            }
        }

    # Compute dispersion statistics
    std = series_clean.std()
    var = series_clean.var()
    min_val = series_clean.min()
    max_val = series_clean.max()
    range_val = max_val - min_val
    mad = (series_clean - series_clean.mean()).abs().mean()
    mean = series_clean.mean()
    cv = std / mean if mean != 0 else float('nan')
    pct_10 = series_clean.quantile(0.10)
    pct_25 = series_clean.quantile(0.25)
    pct_75 = series_clean.quantile(0.75)
    pct_90 = series_clean.quantile(0.90)
    iqr = pct_75 - pct_25

    # median & its 95% CI (approx via binomial quantile)
    # For median p=0.5, z=1.96, se=0.5/sqrt(n)
    se_med = 0.5 / np.sqrt(n)
    delta = 1.96 * se_med
    lower_q = max(0, (0.5 - delta) * 100)
    upper_q = min(100, (0.5 + delta) * 100)
    med = float(series_clean.median())
    med_ci = (
        float(series_clean.quantile(lower_q/100.0)),
        float(series_clean.quantile(upper_q/100.0))
    )

    # Extreme bounds
    lower_bound = mean - std_outlier_multiplier * std
    upper_bound = mean + std_outlier_multiplier * std
    lower_outliers = series_clean[series_clean < lower_bound]
    upper_outliers = series_clean[series_clean > upper_bound]
    n_lower = lower_outliers.size
    n_upper = upper_outliers.size

    # Plot setup
    sns.set_palette("colorblind")
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)

    # 1) Thin violin silhouette (behind box)
    parts = ax.violinplot(
        series_clean,
        vert=True,
        positions=[0],
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for pc in parts['bodies']:
        pc.set_facecolor(palette[0])
        pc.set_edgecolor(palette[0])
        pc.set_alpha(0.15)
        pc.set_linewidth(0.8)
        pc.set_zorder(1)

    # 2) notched boxplot behind
    ax.boxplot(
        series_clean,
        positions=[0],
        widths=0.4,
        notch=True,
        patch_artist=True,
        showcaps=True,
        boxprops=dict(facecolor='white', linewidth=1.2),
        whiskerprops=dict(linewidth=1),
        medianprops=dict(linewidth=1.5, color=palette[1]),
        flierprops=dict(marker='o', markersize=0),  # hide default fliers
        zorder=2
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    # Mean dot (uses palette[1], free of other annotations)
    ax.scatter(
        [0], [mean],
        color=palette[1],
        marker='o',
        s=60,
        zorder=4,
        label=f"Mean = {mean:.2f}"
    )

    # Annotate extreme‐bound lines
    ax.axhline(lower_bound, color=palette[2], linestyle='--',
               label=f"Lower {std_outlier_multiplier}σ = {lower_bound:.2f} ({n_lower})")
    ax.axhline(upper_bound, color=palette[3], linestyle='--',
               label=f"Upper {std_outlier_multiplier}σ = {upper_bound:.2f} ({n_upper})")

    # Highlight extreme points
    if n_lower:
        ax.scatter([0]*n_lower, lower_outliers, color=palette[2], zorder=3)
    if n_upper:
        ax.scatter([0]*n_upper, upper_outliers, color=palette[3], zorder=3)

    # Annotate 10th/90th percentile lines
    ax.axhline(pct_10, color='purple', linestyle=':', label=f"10th pct = {pct_10:.2f}")
    ax.axhline(pct_90, color='purple', linestyle=':', label=f"90th pct = {pct_90:.2f}")

    ax.legend(loc="upper left", fontsize="small", frameon=False)

    # Dispersion stats textbox
    text = (
        f"Std Dev = {std:.2f}\n"
        f"Variance = {var:.2f}\n"
        f"Min = {min_val:.2f}, Max = {max_val:.2f}\n"
        f"Range = {range_val:.2f}\n"
        f"MAD = {mad:.2f}\n"
        f"CV = {cv:.2f}\n"
        f"IQR = {iqr:.2f}\n"
        f"Median = {med:.2f}\n"
        f"Median 95% CI = ({med_ci[0]:.2f}, {med_ci[1]:.2f})"
    )

    ax.text(
        0.95, 0.95, text,
        transform=ax.transAxes,
        va='top', ha='right',
        fontsize='small',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
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

    # Optional save
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')

    return {
        'descriptive_stats': {
            'n': n,
            'mean': mean,
            'std': std,
            'var': var,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'mad': mad,
            'cv': cv,
            'pct_10': pct_10,
            'pct_25': pct_25,
            'pct_75': pct_75,
            'pct_90': pct_90,
            'iqr': iqr,
            'median_ci': med_ci,
            'extreme_lower_count': n_lower,
            'extreme_upper_count': n_upper
        },
        'chart_metadata': {
            'title': title,
            'ylabel': ylabel,
            'data_source': data_source,
            'file_name': file_name,
            'std_outlier_multiplier': std_outlier_multiplier,
        }
    }
