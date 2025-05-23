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
from pathlib import Path
import logging
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import probplot

from .validate_numeric_named_series import validate_numeric_named_series

logger = logging.getLogger(__name__)

def _save_and_close(fig, path):
    """
    Save a Matplotlib figure to `path` and ensure it gets closed.
    """
    try:
        fig.savefig(path)
    finally:
        plt.close(fig)

def numeric_distribution_visualizations(
    s: pd.Series,
    report_dir: Path,
    transform: str = "raw",
    report_log_id: str = str(uuid.uuid4())
) -> dict:
    """
    Display and save distribution plots for a numeric Series, annotating
    whether the data is raw or has been transformed.

    Args:
        s (pd.Series): Series containing the data.
        report_dir (Path): Directory for saving plot files.
        transform (str): Label for the data transformation applied
                         (e.g. "raw", "box-cox", "yeo-johnson").
        report_log_id (str): report log id.

    Returns:
        dict: Mapping from plot type to saved filepath.
    """
    # 1. Input validation
    validate_numeric_named_series(s)

    logger.info(
        "Starting numeric_distribution_visualizations",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    # 2. Drop nulls and short-circuit if empty
    s_clean = s.dropna()
    if s_clean.empty:
        return {}

    # Sanitize transform label for filenames
    label = transform.strip().replace(" ", "_")

    viz_paths: dict[str,str] = {}

    # 3. Raw‐Counts Histogram
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        s_clean,
        stat='count',     # absolute counts
        element='bars',
        fill=True,
        alpha=0.6,
        ax=ax
    )

    ax.set_title(f"Histogram of {s_clean.name} (raw counts)")
    ax.set_xlabel(s_clean.name)
    ax.set_ylabel("Count")
    plt.tight_layout()

    counts_file = report_dir / f"{s_clean.name}_{label}_hist_counts.png"
    _save_and_close(fig, counts_file)
    viz_paths["hist_counts"] = str(counts_file)

    # 4. Histogram + Kernel Density Estimate (KDE)
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot normalized histogram with KDE overlay
    sns.histplot(
        s_clean,
        stat='density',       # show density instead of raw counts
        kde=True,
        element='step',       # cleaner histogram edge
        fill=True,
        alpha=0.4,
        ax=ax
    )

    # Compute and plot mean & median
    mean_val = s_clean.mean()
    median_val = s_clean.median()
    ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.5,
            label=f"Mean = {mean_val:.2f}")
    ax.axvline(median_val, color='red', linestyle='-.', linewidth=1.5,
            label=f"Median = {median_val:.2f}")

    # Title and axes
    ax.set_title(f"Distribution of {s_clean.name} ({transform.capitalize()})", fontsize=14)
    ax.set_xlabel(s_clean.name, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    # Legend and layout
    ax.legend(title="Summary Stats", fontsize=10, title_fontsize=11)
    plt.tight_layout()

    hist_file = report_dir / f"{s_clean.name}_{label}_hist_kde.png"
    _save_and_close(fig, hist_file)
    viz_paths["hist_kde"] = str(hist_file)

    # 5. Boxplot
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(
        x=s_clean,
        ax=ax,
        orient='h',
        notch=True,
        width=0.6,
        boxprops={'facecolor':'lightgray','edgecolor':'black'},
        medianprops={'color':'red','linewidth':2},
        flierprops={'marker':'o','markerfacecolor':'blue','markersize':5,'alpha':0.6},
        whiskerprops={'color':'black'}
    )

    # Compute IQR and outlier count
    q1, q3 = s_clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_mask = (s_clean < q1 - 1.5 * iqr) | (s_clean > q3 + 1.5 * iqr)
    outlier_count = int(outlier_mask.sum())
    median_val = s_clean.median()

    # Annotate median value above the box
    ax.text(
        median_val,
        0.7,
        f"Median = {median_val:.2f}",
        ha='center',
        va='bottom',
        color='red',
        fontsize=10
    )

    # Annotate outlier count in the corner
    ax.text(
        0.95,
        0.95,
        f"Outliers: {outlier_count}",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        color='gray'
    )

    # Titles & labels
    ax.set_title(f"Boxplot of {s_clean.name} ({transform.capitalize()})", fontsize=14)
    ax.set_xlabel(s_clean.name, fontsize=12)
    ax.set_yticks([])  # hide the trivial y-axis

    plt.tight_layout()

    box_file = report_dir / f"{s_clean.name}_{label}_boxplot.png"
    _save_and_close(fig, box_file)
    viz_paths["boxplot"] = str(box_file)

    # 6. Empirical Cumulative Distribution Function (ECDF)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.ecdfplot(s_clean, ax=ax)

    # Annotate key percentiles
    percentiles = [0.25, 0.5, 0.75]
    colors = ['orange', 'red', 'purple']
    for p, c in zip(percentiles, colors):
        x_p = s_clean.quantile(p)
        ax.axvline(x_p, linestyle='--', color=c, linewidth=1)
        ax.text(
            x_p, p,
            f"{int(p*100)}th pct: {x_p:.2f}",
            color=c,
            ha='right', va='bottom',
            fontsize=10
        )

    # Add sample size and grid
    n = len(s_clean)
    ax.set_title(f"Cumulative Distribution of {s_clean.name} ({transform.capitalize()}; n={n})", fontsize=14)
    ax.set_xlabel(s_clean.name, fontsize=12)
    ax.set_ylabel("Proportion ≤ x", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ecdf_file = report_dir / f"{s_clean.name}_{label}_ecdf.png"
    _save_and_close(fig, ecdf_file)
    viz_paths["ecdf"] = str(ecdf_file)

    # 7. Q–Q plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Compute theoretical vs. sample quantiles
    (osm, osr), (slope, intercept, r) = probplot(s_clean, dist="norm", plot=None)

    # Scatter the quantiles
    ax.scatter(osm, osr, s=20, alpha=0.6, edgecolor='k', label='Data Quantiles')

    # Plot 45° reference line
    min_q, max_q = np.min([osm, osr]), np.max([osm, osr])
    ax.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=1, label='45° Line')

    # Plot fitted regression line
    fit_line = slope * osm + intercept
    ax.plot(osm, fit_line, 'b-', linewidth=1.5, label=f'Fit: R\u00b2={r**2:.2f}')

    # Titles & labels
    ax.set_title(f"Q–Q Plot of {s_clean.name} ({transform.capitalize()}; n={len(s_clean)})", fontsize=14)
    ax.set_xlabel("Theoretical Normal Quantiles", fontsize=12)
    ax.set_ylabel("Sample Quantiles", fontsize=12)

    # Legend & grid
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    qq_file = report_dir / f"{s_clean.name}_{label}_qq_plot.png"
    _save_and_close(fig, qq_file)
    viz_paths["qq_plot"] = str(qq_file)

    logger.info(
        "Completed numeric_distribution_visualizations",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    return viz_paths
