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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chisquare

from analytics_eda.core.categorical import validate_categorical_named_series
from analytics_eda.core.utils import build_chart_title


def plot_chi2_gof_uniform(
    series: pd.Series,
    alpha: float = 0.05,
    title_template: str = "Chi-Square Goodness-of-Fit: {name}{modifiers}",
    name: str = None,
    filter_desc: str = None,
    transform_desc: str = None,
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    data_source: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    file_name: str = None
) -> dict:
    """
    Performs a Chi-Square Goodness-of-Fit test against a uniform distribution and 
    visualizes observed vs. expected categorical frequencies using a side-by-side bar chart.

    Why this is important:
        The Chi-Square Goodness-of-Fit test assesses whether the observed distribution 
        of a single categorical variable significantly deviates from a uniform (equal probability) distribution.
        This helps identify category imbalance or concentration.

    What it does:
        - Computes expected frequencies assuming uniform distribution.
        - Runs a chi-square test of goodness-of-fit.
        - Warns if expected counts violate assumptions (e.g., < 5).
        - Plots observed vs. expected counts for intuitive comparison.
        - Annotates the statistical result on the chart.
        - Optionally saves the plot and includes metadata.

    Parameters:
        series (pd.Series): Categorical data series.
        alpha (float): Significance level for hypothesis testing.
        title_template (str): Template for dynamic chart title generation.
        name (str): Optional variable name override.
        filter_desc (str): Optional filter descriptor for title context.
        transform_desc (str): Optional transformation descriptor for title context.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        data_source (str): Optional data source for footer annotation.
        figsize (tuple): Size of the chart figure.
        save_path (str): Optional directory to save the chart.
        file_name (str): Optional filename for saved chart.

    Returns:
        dict: Metadata and test results including:
            - 'descriptive_stats': {'total', 'k'}
            - 'inferential_stats': {'chi2_gof_null_uniform': {statistic, p_value, alpha, reject, warning (if any)}}
            - 'chart_metadata': {title, xlabel, ylabel, data_source, file_name}
    """
    # Prepare data
    validate_categorical_named_series(series)
    data = series.copy().dropna().astype(str)
    freq_table = data.value_counts()

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
            'descriptive_stats': {
                'total': 0,
                'k': 0
            },
            'inferential_stats': {},
            'chart_metadata': {
                'title': title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'data_source': data_source,
                'file_name': file_name
            }
        }

    # Compute stats
    categories = sorted(freq_table.keys())
    observed = [freq_table[cat] for cat in categories]
    total = sum(observed)
    k = len(categories)
    expected = [total / k] * k

    warning = None
    if any(e < 5 for e in expected):
        warning = "Some expected counts are below 5; chi-square test results may not be reliable."

    chi2_stat, p_val = chisquare(f_obs=observed, f_exp=expected)

    # Plotting
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(categories))
    width = 0.35

    ax.bar([i - width / 2 for i in x], observed, width, label='Observed')
    ax.bar([i + width / 2 for i in x], expected, width, label='Expected')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    # Annotate test results
    annotation = '\n'.join((
        rf"$\chi^2$ = {chi2_stat:.2f}",
        rf"$p$ = {p_val:.3f}",
        rf"$\alpha$ = {alpha:.2f}",
        f"Decision: {'Reject H₀' if p_val < alpha else 'Fail to Reject H₀'}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.95, annotation, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Optional data source annotation
    if data_source:
        fig.text(
            0.01, 0.01, f"Source: {data_source}",
            ha='left', va='bottom',
            fontsize='small', color='gray'
        )

    fig.tight_layout()

    # Optional save
    if save_path:
        if file_name is None:
            file_name = f"{title}.png"
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')

    # Return metadata and results
    return {
        'descriptive_stats': {
            'total': total,
            'k': k
        },
        'inferential_stats': {
            'chi2_gof_null_uniform': {
                'statistic': float(chi2_stat),
                'p_value': float(p_val),
                'alpha': float(alpha),
                'reject': bool(p_val < alpha),
                **({'warning': warning} if warning else {})
            }
        },
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'file_name': file_name
        }
    }
