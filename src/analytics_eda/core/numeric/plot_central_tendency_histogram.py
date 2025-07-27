import os
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from .validate_numeric_named_series import validate_numeric_named_series

def plot_central_tendency_histogram(
    data: pd.Series,
    bins: int = None,
    title: str = "Histogram with Central Tendency",
    xlabel: str = "Value",
    ylabel: str = "Count",
    data_source: str = None,
    save_path: str = None,
    file_name: str = None
):
    """
    Generate a histogram that effectively communicates the central tendency of a numeric variable.

    Why:
        This function helps analysts and data storytellers visualize the distribution of a numeric variable 
        with key statistical indicators of central tendency: mean, median, mode, and a 95% confidence interval 
        for the mean. Annotating these statistics directly on the histogram enhances clarity and insight.

    What:
        - Accepts a pandas Series of numeric values.
        - Plots a histogram with annotated vertical lines for mean, median, mode(s), and 95% CI.
        - Optionally saves the figure to disk.
        - Returns descriptive statistics and metadata useful for reporting or reproducibility.

    How:
        - Missing values are dropped.
        - If not specified, the number of bins is determined using the Square-Root Choice rule (ceil(sqrt(n))).
        - Modes are estimated based on the bin(s) with the highest frequency count.
        - Mean, median, and 95% CI are calculated and displayed on the plot.
        - Optional metadata such as chart title, axis labels, and data source are customizable.

    Parameters
    ----------
    data : pd.Series
        Numeric dataset to plot. Missing values will be dropped.
    bins : int or sequence, optional
        Number of histogram bins or explicit bin edges. Defaults to Square-Root Choice ceil(sqrt(n)).
    title : str, default="Histogram with Central Tendency"
        Title displayed at the top of the chart.
    xlabel : str, default="Value"
        Label for the x-axis.
    ylabel : str, default="Count"
        Label for the y-axis.
    data_source : str, optional
        Text annotation to show the source of the data in the chart.
    save_path : str or Path, optional
        Directory where the plot image will be saved. Created if it doesn't exist.
    file_name : str, optional
        Name of the image file (e.g., "histogram.png"). Must be used with `save_path`.

    Returns
    -------
    metadata : dict
        {
            'descriptive_stats': {
                'n': int,                        # Sample size
                'mean': float,                   # Arithmetic mean
                'median': float,                 # 50th percentile
                'mode': list of float,           # Most frequent value(s)
                'ci95': (float, float)           # 95% confidence interval for the mean
            },
            'chart_metadata': {
                'title': str,
                'xlabel': str,
                'ylabel': str,
                'data_source': str or None,
                'bins': int or sequence,
                'relative_path': str or None     # Relative path to saved image (if any)
            }
        }
    """
    validate_numeric_named_series(data)
    series = data.dropna()
    n = series.size

    # Determine bins via Square-Root choice if not specified
    if bins is None:
        bins = math.ceil(math.sqrt(n))

    # Compute descriptive statistics
    mean = series.mean()
    median = series.median()
    mode_vals = series.mode().tolist()
    sem = stats.sem(series)
    ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)

    # Prepare plot
    fig, ax = plt.subplots()
    sns.histplot(series, bins=bins, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Plot mean and median lines
    ax.axvline(mean, linestyle='--', label=f"Mean = {mean:.2f}")
    ax.axvline(median, linestyle='-.', label=f"Median = {median:.2f}")

    # Plot mode lines for each raw mode value
    for i, mv in enumerate(mode_vals):
        label = "Mode" if len(mode_vals) == 1 else f"Mode {i+1}"
        ax.axvline(mv, linestyle=':', label=f"{label} = {mv:.2f}")

    # Plot confidence interval
    ax.axvline(ci_low, linestyle=':', alpha=0.7)
    ax.axvline(ci_high, linestyle=':', alpha=0.7)

    # Stats textbox
    mode_text = ", ".join(f"{mv:.2f}" for mv in mode_vals)
    text = (
        f"n = {n}\n"
        f"Mean = {mean:.2f}\n"
        f"Median = {median:.2f}\n"
        f"Mode(s) = {mode_text}\n"
        f"95% CI = [{ci_low:.2f}, {ci_high:.2f}]"
    )
    ax.text(
        0.95, 0.95, text,
        transform=ax.transAxes,
        va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    # Optional data source annotation
    if data_source:
        ax.text(
            0.05, 0.05, f"Source: {data_source}",
            transform=ax.transAxes,
            va='bottom', ha='left',
            fontsize='small', color='gray'
        )

    ax.legend()

    # Optional save
    relative_path = None
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')
        relative_path = os.path.relpath(abs_path)

    # Return split metadata
    return {
        'descriptive_stats': {
            'n': n,
            'mean': mean,
            'median': median,
            'mode': mode_vals,
            'ci95': (ci_low, ci_high)
        },
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'bins': bins,
            'relative_path': relative_path
        }
    }
