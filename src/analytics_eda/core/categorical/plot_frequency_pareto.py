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

from matplotlib import pyplot as plt
import numpy as np


def plot_frequency_pareto(series, title=None, xlabel=None, ylabel='Count',
                figsize=(12, 8), data_source='Unknown', min_value=None):
    """
    Plot a Pareto chart for a categorical Pandas Series with:
      - Bar chart of category counts (sorted descending)
      - Count annotations and relative frequency (%) above each bar
      - Cumulative percentage line
      - 80% threshold marker
      - Highlight all bars up to and including the first bar that reaches 80%
      - Optional: Aggregate small categories into 'Others'
      - Data source annotation

    Parameters
    ----------
    series : pd.Series
        Categorical data to plot.
    title : str, optional
        Chart title. Defaults to 'Pareto Chart: <series.name>'.
    xlabel : str, optional
        X-axis label. Defaults to series.name.
    ylabel : str, default 'Count'
        Y-axis label for counts.
    figsize : tuple, default (12, 8)
        Figure size in inches.
    data_source : str, default 'Unknown'
        Data source annotation.
    min_value : int, optional
        Minimum count required to show a category. Categories below this are grouped into 'Others'.

    Returns
    -------
    None
    """
    # Prepare data
    data = series.dropna().astype(str)
    counts = data.value_counts()

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

    # Colors: muted grey and colorblind-friendly accent
    muted = '#999999'
    accent = '#0072B2'  # colorblind-friendly blue

    # Highlight bars up to threshold_idx inclusive
    bar_colors = [accent if i <= threshold_idx else muted for i in range(len(counts))]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor='black')

    # Annotate counts and relative frequencies
    for bar, count, pct in zip(bars, counts.values, rel_freq.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{int(count)}\n({pct:.1f}%)',
                ha='center', va='bottom')

    # Set ticks and labels
    ticks = np.arange(len(counts))
    ax.set_xticks(ticks)
    ax.set_xticklabels(counts.index, rotation=45, ha='right')
    ax.set_xlabel(xlabel or series.name or '')
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Pareto Chart: {series.name}')

    # Cumulative percentage line
    ax2 = ax.twinx()
    ax2.plot(ticks, cumperc.values, marker='o', linestyle='-', color='black')
    ax2.set_ylabel('Cumulative %')
    ax2.set_ylim(0, 110)

    # 80% threshold line
    ax2.axhline(80, color=accent, linestyle='--')
    ax2.text(len(counts) - 1, 80, '80% threshold', va='bottom', ha='right', color=accent)

    # Data source annotation
    fig.text(0.99, 0.01, f"Data Source: {data_source}",
             ha='right', va='bottom', fontsize=8, color='gray')

    # Footnote with cumulative count at 80%
    fig.text(0.01, 0.01, f"Cumulative count at 80%: {threshold_count}",
             ha='left', va='bottom', fontsize=8, color='gray')

    fig.tight_layout()
    plt.show()
