import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .validate_numeric_named_series import validate_numeric_named_series

def plot_cardinality_barchart(
    series: pd.Series,
    top_k: int = 10,
    title: str = "Value Counts (Top k) for Cardinality",
    xlabel: str = "Value",
    ylabel: str = "Count",
    data_source: str = None,
    figsize: tuple = (8, 6),
    save_path: str = None,
    file_name: str = None
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
    title : str, default="Value Counts (Top k) for Cardinality"
        Chart title.
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

    Returns
    -------
    metadata : dict
        {
            'descriptive_stats': {
                'nunique': int   # number of distinct values
            },
            'chart_metadata': {
                'title': str,
                'xlabel': str,
                'ylabel': str,
                'data_source': str or None,
                'top_k': int,
                'relative_path': str or None
            }
        }
    """
    validate_numeric_named_series(series)
    clean = series.copy().dropna()
    nunique = int(clean.nunique())

    # Compute top-k frequencies
    counts = clean.value_counts().head(top_k)
    labels = counts.index.astype(str)
    values = counts.values

    # Plot
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=labels, y=values, ax=ax)
    ax.set_title(title)
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
    relative_path = None
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')
        relative_path = os.path.relpath(abs_path)

    return {
        'descriptive_stats': {
            'nunique': nunique
        },
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'top_k': top_k,
            'relative_path': relative_path
        }
    }
