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

def plot_cardinality_barchart(
    series: pd.Series,
    top_k: int = 10,
    title_template: str = "Value Counts (Top {top_k}) of {name} for Cardinality",
    name: str = None,
    xlabel: str = "Value",
    ylabel: str = "Count",
    data_source: str = None,
    figsize: tuple = (8, 6),
    save_path: str = None,
    file_name: str = None,
    max_unique_fraction: float = 0.05,
    max_unique_values: int = 20,
    integer_tolerance: float = 1e-8
):
    """
    Generate a bar chart that tells the cardinality story of a numeric variable.

    Why:
        Cardinality measures the number of distinct values.  
        • High cardinality → treat as continuous (histograms, density plots).  
        • Low cardinality → may be discrete/categorical; consider bar plots or bucketing.

    What:
        - Computes number of unique values (nunique).  
        - Ranks values by frequency and displays the top k in a bar chart.  
        - Optionally annotates data source and saves the figure.  
        - Returns cardinality metric and chart metadata for reporting.

    How:
        - Cleans the data by dropping missing values.  
        - Uses pandas `value_counts` to get top k frequencies.  
        - Plots a colorblind-friendly bar chart of value vs. count.  
        - Supports layout control via `figsize` and export via `save_path`/`file_name`.

    Parameters
    ----------
    series : pd.Series
        Numeric dataset to analyze. Missing values will be dropped.
    top_k : int, default=10
        Number of most frequent values to display.
    title_template: A Python format-string with placeholders:
      - {name}:        series name or label
      - {modifiers}:   combined filter/transform text, empty if none
    name:                Optional override for series.name
    xlabel : str, default="Value"
        Label for the x-axis.
    ylabel : str, default="Count"
        Label for the y-axis.
    data_source : str, optional
        Text annotation for the data source (bottom-left of figure).
    figsize : tuple, default=(8, 6)
        Figure size in inches (width, height).
    save_path : str or Path, optional
        Directory where the plot image will be saved. Created if needed.
    file_name : str, optional
        Filename (with extension) for saving. Requires `save_path`.
    max_unique_fraction : float
        Relative threshold of unique values / total rows below which we
        treat floats as discrete.
    max_unique_values   : int
        Absolute cap on number of unique values to still call discrete.
    integer_tolerance   : float
        Tolerance for considering float values "whole" (e.g. 1.00000002)

    Returns
    -------
    metadata : dict
        {
            'descriptive_stats': {
                'nunique': int   # number of distinct values
                'is_discrete': bool
            },
            'chart_metadata': {
                'title': str,
                'xlabel': str,
                'ylabel': str,
                'data_source': str or None,
                'top_k': int,
                'file_name': str or None
            }
        }
    """
    validate_numeric_named_series(series)
    clean = series.copy().dropna()
    nunique = int(clean.nunique())
    is_discrete = is_discrete_numeric(
        clean,
        max_unique_fraction=max_unique_fraction,
        max_unique_values=max_unique_values,
        integer_tolerance=integer_tolerance
    )

    label = name or getattr(series, "name", None) or "Value"
    title = title_template.format(name=label, top_k=top_k)

    # Compute top-k frequencies
    counts = clean.value_counts().head(top_k)
    labels = counts.index.astype(str)
    values = counts.values

    # Plot
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=labels, y=values, ax=ax)

    subtitle = "Discrete" if is_discrete else "Continuous"
    ax.set_title(f"{title}  ({subtitle})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Optional data source
    if data_source:
        fig.text(
            0.01, 0.01, f"Source: {data_source}",
            ha='left', va='bottom',
            fontsize='small', color='gray'
        )

    # Optional save
    if save_path:
        if file_name is None:
            file_name = f"{title}.png"
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')

    return {
        'descriptive_stats': {
            'nunique': nunique,
            'is_discrete': is_discrete,
        },
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'top_k': top_k,
            'max_unique_fraction': max_unique_fraction,
            'max_unique_values': max_unique_values,
            'integer_tolerance': integer_tolerance,
            'file_name': file_name
        }
    }

def is_discrete_numeric(s,
        max_unique_fraction: float = 0.05,
        max_unique_values: int = 20,
        integer_tolerance: float = 1e-8) -> bool:
    """
    Determine whether a numeric pandas Series should be treated as discrete.

    A series is considered discrete if:
      - It has an integer dtype and either:
        * The ratio of unique values to non-null entries is below `max_unique_fraction`, or
        * The total number of unique values is below `max_unique_values`.
      - It has a float dtype and either:
        * All values are within `integer_tolerance` of a whole number, or
        * Its unique-value ratio or count falls below the specified thresholds.

    Parameters
    ----------
    s : pd.Series
        Numeric data to evaluate. NaNs are ignored in all calculations.
    max_unique_fraction : float, default=0.05
        Maximum fraction of unique values (unique / total non-null) to still call discrete.
    max_unique_values : int, default=20
        Maximum absolute count of unique values to still call discrete.
    integer_tolerance : float, default=1e-8
        Tolerance for treating float values as effectively integers (e.g. 3.0000000001).

    Returns
    -------
    bool
        True if the series meets the criteria for discreteness; False otherwise.
    """
    # 1. Integer dtype
    if pd.api.types.is_integer_dtype(s.dtype):
        return (s.nunique() / len(s)) <= max_unique_fraction \
            or s.nunique() < max_unique_values
    # 2. Float dtype
    if pd.api.types.is_float_dtype(s.dtype):
        # 2a. effectively all whole numbers?
        if np.isclose(s % 1, 0, atol=integer_tolerance).all():
            return True
        # 2b. low cardinality
        frac = s.nunique() / len(s)
        if frac < max_unique_fraction or s.nunique() < max_unique_values:
            return True
    return False
