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
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .validate_categorical_named_series import validate_categorical_named_series

from ..utils.build_chart_title import build_chart_title

def plot_frequency_pareto(
    series: pd.Series,
    min_value: int = None,
    title_template: str = "Pareto Chart of {name}{modifiers}",
    name: str = None,
    filter_desc: str = None,
    transform_desc: str = None,
    xlabel: str = "Value",
    ylabel: str ='Count',
    data_source: str = None,
    figsize: tuple = (10, 6),
    horizontal: bool = False,
    save_path: str = None,
    file_name: str = None
):
    """
    Plot a Pareto chart to analyze the categorical distribution of a variable and identify the most impactful categories.

    Why this is important:
        The Pareto chart highlights the most frequent categories in descending order and shows their cumulative contribution. 
        This helps identify the “vital few” that account for the majority of occurrences — a core principle of the 80/20 rule 
        (Pareto Principle). It supports data-driven prioritization and effective decision-making.

    What it does:
        - Plots a bar chart of sorted category counts with percentage annotations
        - Overlays a cumulative percentage line and highlights the 80% threshold
        - Groups infrequent categories into "Others" based on a count threshold (optional)
        - Optionally displays the data source and saves the chart

    Parameters
    ----------
    series : pd.Series
        Categorical data to plot.
    min_value : int, optional
        Minimum count to show as its own bar; smaller categories are grouped into 'Others'.
    title_template : str, default "Pareto Chart of {name}{modifiers}"
        Template for the chart title; supports placeholders for series name and optional descriptors.
    name : str, optional
        Human-readable variable name to display in the title.
    filter_desc : str, optional
        Text describing any filters applied to the data.
    transform_desc : str, optional
        Text describing any transformations applied to the data.
    xlabel, ylabel : str
        Axis labels.
    data_source : str, optional
        Text displayed in the chart footer to identify the data source.
    figsize : tuple, default (10, 6)
        Figure size in inches.
    horizontal : bool, default False
        If True, plots horizontal bars; otherwise, vertical bars.
    save_path : str, optional
        Directory path where the figure should be saved.
    file_name : str, optional
        File name to use when saving the chart.

    Returns
    -------
    dict
        {
        'descriptive_stats': {
            'mode': str or None,  # Most frequent category
            'total_count': int,   # Total number of values
            'n_categories': int,  # Number of unique categories shown
            'cumulative_count_at_80pct': int  # Count at which cumulative frequency reaches 80%
        },
        'chart_metadata': {
            'title': str,
            'xlabel': str,
            'ylabel': str,
            'data_source': str or None,
            'file_name': str or None
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

    if data.empty:
        # Return metadata
        return {
            'descriptive_stats': {
                'mode': None,
                'total_count': 0,
                'n_categories': 0,
                'cumulative_count_at_80pct': 0
            },
            'chart_metadata': {
                'title': title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'data_source': data_source,
                'file_name': file_name
            }
        }

    # Group small categories into 'Others' if min_value is set
    if min_value is not None:
        small_categories = counts[counts < min_value]
        if not small_categories.empty:
            counts = counts[counts >= min_value]
            counts['Others'] = small_categories.sum()

    # Calculate relative frequency and cumulative %
    rel_freq = counts / counts.sum() * 100
    cumperc = rel_freq.cumsum()

    # Determine first bar to reach or exceed 80%
    threshold_idx = int(np.argmax(cumperc.values >= 80))
    threshold_count = counts.values[:threshold_idx + 1].sum()

    # Compute descriptive stats
    descriptive_stats = {
        'mode': counts.index[0] if len(counts) > 0 else None,
        'total_count': int(counts.sum()),
        'n_categories': int(len(counts)),
        'cumulative_count_at_80pct': int(threshold_count)
    }

    # Colors: muted grey and colorblind-friendly accent
    muted = '#999999'
    accent = '#0072B2'  # colorblind-friendly blue

    # Highlight bars up to threshold_idx inclusive
    bar_colors = [accent if i <= threshold_idx else muted for i in range(len(counts))]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        bars = ax.barh(counts.index, counts.values, color=bar_colors, edgecolor='black')
        # Annotations
        for bar, count, pct in zip(bars, counts.values, rel_freq.values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'{int(count)} ({pct:.1f}%)',
                    ha='left', va='center')
        # Axes
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        ticks = np.arange(counts.size)
        ax.set_yticks(ticks)
        ax.set_yticklabels(counts.index)
        # Cumulative % on top axis
        ax2 = ax.twiny()
        ax2.plot(cumperc.values, ticks, marker='o', linestyle='-', color='black')
        ax2.set_xlabel('Cumulative %')
        ax2.set_xlim(0, 110)
        ax2.axvline(80, color=accent, linestyle='--')
        ax2.text(80, ticks[-1], '80% threshold', ha='left', va='top', color=accent)
    else:
        bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor='black')
        # Annotations
        for bar, count, pct in zip(bars, counts.values, rel_freq.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(count)}\n({pct:.1f}%)',
                    ha='center', va='bottom')
        # Axes
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ticks = np.arange(counts.size)
        ax.set_xticks(ticks)
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        # Cumulative % on right axis
        ax2 = ax.twinx()
        ax2.plot(ticks, cumperc.values, marker='o', linestyle='-', color='black')
        ax2.set_ylabel('Cumulative %')
        ax2.set_ylim(0, 110)
        ax2.axhline(80, color=accent, linestyle='--')
        ax2.text(ticks[-1], 80, '80% threshold', ha='right', va='bottom', color=accent)

    # Optional data source annotation
    if data_source:
        fig.text(
            0.01, 0.01, f"Source: {data_source}",
            ha='left', va='bottom',
            fontsize='small', color='gray'
        )

    # Footnote with cumulative count at 80%
    fig.text(0.99, 0.01, f"Cumulative count at 80%: {threshold_count}",
             ha='right', va='bottom', fontsize=8, color='gray')

    fig.tight_layout()

    # Optional save
    if save_path:
        if file_name is None:
            file_name = f"{title}.png"
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')

    # Return metadata
    return {
        'descriptive_stats': descriptive_stats,
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'file_name': file_name
        }
    }
