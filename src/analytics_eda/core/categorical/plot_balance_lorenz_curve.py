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

from analytics_eda.core.categorical import validate_categorical_named_series
from analytics_eda.core.utils import build_chart_title


def _lorenz_curve_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Lorenz curve for nonnegative weights (e.g., category frequencies).
    Returns (x, y): cumulative share of categories (x) vs cumulative share of counts (y).
    """
    if counts.size == 0 or np.sum(counts) == 0:
        # Degenerate: return the diagonal (no inequality information)
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        return x, y

    # Sort ascending by count (standard for Lorenz)
    sorted_vals = np.sort(counts.astype(float))
    cum_vals = np.cumsum(sorted_vals)
    total = cum_vals[-1]

    # Prepend the origin (0,0)
    y = np.insert(cum_vals / total, 0, 0.0)
    x = np.linspace(0.0, 1.0, len(y))
    return x, y


def _gini_from_lorenz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Gini = 1 - 2 * area under Lorenz curve.
    Assumes x spans [0,1] and y starts at 0 and ends at 1.
    """
    area = np.trapezoid(y, x)
    return float(1.0 - 2.0 * area)


def plot_balance_lorenz_curve(
    series: pd.Series,
    title_template: str = "Lorenz Curve of {name}{modifiers}",
    name: str = None,
    filter_desc: str = None,
    transform_desc: str = None,
    xlabel: str = "Cumulative share of categories",
    ylabel: str = "Cumulative share of counts",
    data_source: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    file_name: str = None
) -> dict:
    """
    Visualizes category imbalance with a Lorenz curve and reports the Gini index.

    Why this matters
    ----------------
    Balance is about how evenly observations are distributed across categories.
    The Lorenz curve shows cumulative concentration; the Gini index summarizes it
    (0 = perfectly even; 1 = extreme concentration). This is an intuitive way to
    spot dominance or long tails in categorical distributions.

    What this does
    --------------
    - Converts a categorical series into frequency counts per category.
    - Computes the Lorenz curve (cumulative share of categories vs. counts).
    - Computes the Gini index from the Lorenz curve.
    - Plots the Lorenz curve, the line of equality, and shades the gap.
    - Optionally annotates the data source and saves the figure.
    - Returns descriptive metrics and chart metadata.

    Parameters
    ----------
    series : pd.Series
        Categorical data (dtype 'category' or 'object'). Missing values are dropped.
    title_template : str
        Template for the chart title; supports {name} and {modifiers}.
    name : str, optional
        Override for series name in the title.
    filter_desc : str, optional
        e.g., "filtered by New York" (appears in title).
    transform_desc : str, optional
        e.g., "standardized" (appears in title).
    xlabel, ylabel : str
        Axis labels.
    data_source : str, optional
        Text to show as a small footer annotation.
    figsize : tuple
        Figure size in inches.
    save_path : str or Path, optional
        Directory to save the figure (created if needed).
    file_name : str, optional
        Filename for saving; if not provided but save_path is set, defaults to "{title}.png".

    Returns
    -------
    dict
        {
            "descriptive_stats": {
                "total": int,        # total observations (non-null)
                "k": int,            # number of categories (non-empty)
                "gini_index": float  # 0..1
            },
            "chart_metadata": {
                "title": str,
                "xlabel": str,
                "ylabel": str,
                "data_source": str | None,
                "file_name": str | None
            }
        }
    """
    # Prepare data
    validate_categorical_named_series(series)
    data = series.copy().dropna().astype(str)
    counts = data.value_counts()

    # Build chart title
    title = build_chart_title(
        name=name,
        series=series,
        filter_desc=filter_desc,
        transform_desc=transform_desc,
        title_template=title_template
    )

    # Early return if empty
    if data.empty:
        return {
            "descriptive_stats": {"total": 0, "k": 0, "gini_index": float("nan")},
            "chart_metadata": {
                "title": title,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "data_source": data_source,
                "file_name": file_name
            }
        }

    # Compute stats
    freq_values = counts.values.astype(float)  # nonnegative counts
    total = int(freq_values.sum())
    k = int(freq_values.size)

    x_lorenz, y_lorenz = _lorenz_curve_from_counts(freq_values)
    gini = _gini_from_lorenz(x_lorenz, y_lorenz)

    # Plotting
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)

    # Lorenz curve
    sns.lineplot(x=x_lorenz, y=y_lorenz, ax=ax, label="Lorenz curve")

    # Line of equality
    sns.lineplot(x=[0.0, 1.0], y=[0.0, 1.0], ax=ax,
                 linestyle="--", label="Equality line")

    # Shade the area between equality and Lorenz curve
    ax.fill_between(x_lorenz, y_lorenz, x_lorenz, alpha=0.25)

    # Labels & title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    # Gini annotation
    ann = f"Gini index = {gini:.3f}\nCategories = {k}\nTotal = {total}"
    ax.text(
        0.98, 0.02, ann, transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize="small",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )

    # Optional data source annotation
    if data_source:
        fig.text(
            0.01, 0.01, f"Source: {data_source}",
            ha="left", va="bottom",
            fontsize="small", color="gray"
        )

    fig.tight_layout()

    # Optional save
    if save_path:
        if file_name is None:
            file_name = f"{title}.png"
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches="tight")

    # Return metadata and results
    return {
        "descriptive_stats": {
            "total": total,
            "k": k,
            "gini_index": float(gini),
        },
        "chart_metadata": {
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "data_source": data_source,
            "file_name": file_name,
        },
    }
