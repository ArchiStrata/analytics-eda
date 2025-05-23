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

import logging
import uuid
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .validate_categorical_named_series import validate_categorical_named_series

logger = logging.getLogger(__name__)

def categorical_distribution_analysis(
    series: pd.Series,
    save_dir: Path,
    top_n: int = 10,
    report_log_id = str(uuid.uuid4())
) -> dict:
    """
    Analyze a categorical pandas Series and produce a structured report with summary
    statistics, a frequency table, and a top-N bar chart (aggregating all others).

    Parameters
    ----------
    series : pd.Series
        Categorical data to analyze (dtype 'category' or 'object').
    save_dir : pathlib.Path
        Directory where the top-N bar chart will be saved. Created if it does not exist.
    top_n : int, optional
        Number of highest-frequency categories to plot. If the series has fewer than
        top_n unique values, top_n is reset to max(1, unique_categories // 2). Default is 10.
    report_log_id (str): report log id.

    Returns
    -------
    dict
        A dict with a single key `'report'`, whose value is another dict containing:

        - **statistics** : dict  
            - **category_length_stats** : dict with  
                - `'max_length'` (int) – length of the longest category label  
                - `'min_length'` (int) – length of the shortest category label  
            - **cardinality** (int) – number of unique non‐null categories  
            - **imbalance_ratio** (float or None) – ratio of most to least frequent category  

        - **frequency_report** : dict  
            - **frequency_table** : dict  
                Mapping each category (str) to a dict with keys  
                - `'count'` (int) – raw frequency  
                - `'proportion'` (float) – frequency divided by total observations  
            - **visualizations** : dict  
                - `'top_n_plot'` (str) – filesystem path to the saved bar chart  
    """
    # validate input
    validate_categorical_named_series(series)

    logger.info(
        "Starting categorical_distribution_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    total = len(series)
    # frequency and proportions
    freq = series.value_counts(dropna=False).rename_axis(series.name)
    props = freq / total
    frequency = {
        str(cat): {
            'count': int(freq_cat),
            'proportion': float(props_cat)
        }
        for cat, freq_cat, props_cat in zip(freq.index, freq.values, props.values)
    }

    # category length stats
    lengths = [len(str(cat)) for cat in freq.index]
    category_length_stats = {'max_length': max(lengths), 'min_length': min(lengths)}

    # cardinality and imbalance
    cardinality = int(series.nunique(dropna=True))
    imbalance_ratio = float(freq.max() / freq.min()) if freq.min() > 0 else None

    # Bar plot of top N categories
    unique_categories = len(freq)

    # If fewer than requested top_n, adjust top_n (e.g., reduce to 5 if <10 categories)
    if unique_categories <= top_n:
        # Set top_n to half of available categories (or at least 1)
        top_n = max(1, unique_categories // 2)

    top = freq.head(top_n)
    others_count = freq.iloc[top_n:].sum()
    top = pd.concat([top, pd.Series({'Others': others_count})])
    top = top.sort_values(ascending=False)

    # Define accessible colors
    color_map = {
        0: "#DAA520",   # Gold
        1: "#C0C0C0",   # Silver
        2: "#CD7F32",   # Bronze
    }
    default_color = "steelblue"
    others_color = "#A9A9A9"

    # Assign colors
    colors = []
    for i, cat in enumerate(top.index):
        if cat == "Others":
            colors.append(others_color)
        elif i in color_map:
            colors.append(color_map[i])
        else:
            colors.append(default_color)

    # Plot
    plt.figure(figsize=(10, 8))
    ax2 = sns.barplot(x=top.values, y=top.index, palette=colors, hue=top.index, legend=False)

    # Add value labels
    buffer = max(top.values) * 0.1
    ax2.set_xlim(0, max(top.values) + buffer)
    
    for i, v in enumerate(top.values):
        ax2.text(v + max(top.values) * 0.01, i, f"{v:,}", va="center", fontsize=10)

    # Set labels and title
    ax2.set_title(f"Top {top_n} Values in {series.name.replace('_', ' ').title()} (+Others Aggregated)",
                fontsize=14, weight="bold")
    ax2.set_xlabel("Count", fontsize=12)
    ax2.set_ylabel(series.name.replace('_', ' ').title(), fontsize=12)

    plt.tight_layout()
    fig2 = ax2.get_figure()

    plot_path = save_dir / f"{series.name.replace(' ', '_')}_top_{top_n}.png"
    fig2.savefig(plot_path)
    plt.close(fig2)

    # compile report
    report = {
        'statistics': {
            'category_length_stats': category_length_stats,
            'cardinality': int(cardinality),
            'imbalance_ratio': imbalance_ratio,
        },
        'frequency_report': {
            'frequency_table': frequency,
            'visualizations': {
                'top_n_plot': str(plot_path)
            }
        }
    }

    logger.info(
        "Completed categorical_distribution_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    return {'report': report}
