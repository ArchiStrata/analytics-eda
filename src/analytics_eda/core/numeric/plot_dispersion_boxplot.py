import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .validate_numeric_named_series import validate_numeric_named_series

def plot_dispersion_boxplot(
    series: pd.Series,
    title: str = "Boxplot with Dispersion Statistics",
    ylabel: str = "Value",
    data_source: str = None,
    figsize: tuple = (8, 6),
    save_path: str = None,
    file_name: str = None
):
    """
    Generate a boxplot that effectively communicates the dispersion of a numeric variable.

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
    title : str, default="Boxplot with Dispersion Statistics"
        Title displayed at the top of the chart.
    ylabel : str, default="Value"
        Label for the y-axis.
    data_source : str, optional
        Text annotation to show the source of the data in the chart.
    figsize : tuple, default=(8, 6)
        Width and height of the figure in inches.
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
                'pct_90': float
            },
            'chart_metadata': {
                'title': str,
                'ylabel': str,
                'data_source': str or None,
                'relative_path': str or None
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
    """
    validate_numeric_named_series(series)
    series_clean = series.copy().dropna()
    n = series_clean.size

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

    # Prepare plot
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(y=series_clean, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    # Stats textbox
    text = (
        f"Std Dev = {std:.2f}\n"
        f"Variance = {var:.2f}\n"
        f"Min = {min_val:.2f}, Max = {max_val:.2f}\n"
        f"Range = {range_val:.2f}\n"
        f"MAD = {mad:.2f}\n"
        f"CV = {cv:.2f}\n"
        f"10th = {pct_10:.2f}, 25th = {pct_25:.2f}\n"
        f"75th = {pct_75:.2f}, 90th = {pct_90:.2f}"
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
    relative_path = None
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')
        relative_path = os.path.relpath(abs_path)

    return {
        'descriptive_stats': {
            'n': n,
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
            'pct_90': pct_90
        },
        'chart_metadata': {
            'title': title,
            'ylabel': ylabel,
            'data_source': data_source,
            'relative_path': relative_path
        }
    }
