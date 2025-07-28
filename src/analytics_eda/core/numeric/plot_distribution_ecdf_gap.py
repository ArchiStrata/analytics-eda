import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .validate_numeric_named_series import validate_numeric_named_series

def plot_distribution_ecdf_gap(
    series: pd.Series,
    title: str = "ECDF with Gap Analysis",
    xlabel: str = "Value",
    ylabel: str = "ECDF",
    data_source: str = None,
    threshold: float = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    file_name: str = None
):
    """
    Generate an Empirical Cumulative Distribution Function (ECDF) plot that highlights and quantifies gaps in a numeric distribution.

    Why:
        Gaps—intervals with no observations—reveal holes in your data range.  
        Understanding their size, frequency, and location is critical for sampling
        strategies, imputation decisions, and recognizing subpopulation boundaries.

    What:
        - Computes sorted-value gaps between each pair of unique consecutive values.
        - Summarizes:
          • max_gap, median_gap, gap percentiles (P10, P50, P90)
          • count of gaps above a given threshold
          • total_gap_prop (fraction of range with no data)
          • max_gap_loc (midpoint of the largest gap)
        - Plots the ECDF (step function) of all observations.
        - Annotates the largest gap with a double-headed arrow and label.
        - Optionally annotates data source, and saves the figure.
        - Returns both the gap metrics and chart metadata.

    How:
        - Cleans the data by dropping missing values.
        - Uses NumPy to sort unique values and compute diffs.
        - Builds the ECDF via a step plot over all observations.
        - Draws an arrow between the two values that form the largest gap.
        - Places a stats textbox in the plot corner.
        - Supports layout via figsize, and image export via save_path/file_name.

    Parameters
    ----------
    series : pd.Series
        Numeric dataset to analyze. Missing values will be dropped.
    title : str, default="ECDF with Gap Analysis"
        Plot title.
    xlabel : str, default="Value"
        Label for the x-axis.
    ylabel : str, default="ECDF"
        Label for the y-axis.
    data_source : str, optional
        Text annotation for the data source (bottom-left).
    threshold : float, optional
        Gap size threshold for counting large gaps.
    figsize : tuple, default=(10, 6)
        Figure size in inches.
    save_path : str or Path, optional
        Directory to save the plot image (created if needed).
    file_name : str, optional
        Filename (with extension) for saving. Requires `save_path`.

    Returns
    -------
    metadata : dict
        {
            'descriptive_stats': {
                'n': int,                    # number of observations
                'n_unique': int,             # number of distinct values
                'gaps': list of float,       # all raw gap sizes g_i
                'max_gap': float,            # largest gap size
                'median_gap': float,         # median gap size
                'pct10_gap': float,          # 10th percentile of gaps
                'pct50_gap': float,          # 50th percentile (median) of gaps
                'pct90_gap': float,          # 90th percentile of gaps
                'n_gaps_above_thr': int or None,  # count of gaps > threshold
                'total_gap_prop': float,     # sum(gaps)/(max-min)
                'max_gap_loc': float         # midpoint of largest gap
            },
            'chart_metadata': {
                'title': str,
                'xlabel': str,
                'ylabel': str,
                'data_source': str or None,
                'threshold': float or None,
                'relative_path': str or None
            }
        }

    Key Statistics
    --------------
    | Statistic            | What it tells you                                                         |
    |----------------------|---------------------------------------------------------------------------|
    | `gaps`               | All raw spacings between consecutive unique values                        |
    | `max_gap`            | Largest single hole in your data range                                    |
    | `median_gap`         | Typical gap size—indicates clumping vs. even spacing                      |
    | `gap percentiles`    | P10, P50, P90 show distribution of gap sizes (small vs. extreme holes)    |
    | `n_gaps_above_thr`   | Number of gaps exceeding a domain-relevant threshold                      |
    | `total_gap_prop`     | Fraction of the overall range with no observations                        |
    | `max_gap_loc`        | Midpoint location of the largest gap—where the biggest hole sits          |
    """
    validate_numeric_named_series(series)
    clean = series.copy().dropna().sort_values()
    n = clean.size
    unique_vals = clean.unique()
    n_unique = unique_vals.size

    # Compute gaps
    if n_unique >= 2:
        gaps = np.diff(unique_vals)
        gaps_list = gaps.tolist()
        max_gap = float(gaps.max())
        median_gap = float(np.median(gaps))
        pct10_gap = float(np.percentile(gaps, 10))
        pct50_gap = float(np.percentile(gaps, 50))
        pct90_gap = float(np.percentile(gaps, 90))
        total_gap_prop = float(gaps.sum() / (unique_vals[-1] - unique_vals[0]))
        # location of max gap midpoint
        idx = int(np.argmax(gaps))
        max_gap_loc = float((unique_vals[idx] + unique_vals[idx+1]) / 2)
        # threshold count
        n_gaps_above = int((gaps > threshold).sum()) if threshold is not None else None
    else:
        # not enough distinct values
        gaps = np.array([])
        gaps_list = []
        max_gap = median_gap = pct10_gap = pct50_gap = pct90_gap = total_gap_prop = max_gap_loc = np.nan
        n_gaps_above = 0 if threshold is not None else None

    # Build ECDF
    ecdf_x = clean.values
    ecdf_y = np.arange(1, n+1) / n if n > 0 else np.array([])

    # Plot
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=figsize)
    if n > 0:
        ax.step(ecdf_x, ecdf_y, where='post', label='ECDF')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Annotate largest gap
    if n_unique >= 2 and not np.isnan(max_gap_loc):
        # ECDF level at value just before gap
        count_le = np.searchsorted(ecdf_x, unique_vals[idx], side='right')
        y_level = count_le / n
        ax.annotate(
            "", xy=(unique_vals[idx], y_level), xytext=(unique_vals[idx+1], y_level),
            arrowprops=dict(arrowstyle='<->', color='red')
        )
        ax.text(
            max_gap_loc, y_level + 0.02,
            f"Max gap = {max_gap:.2f}",
            ha='center', va='bottom', color='red', fontsize='small'
        )

    # Stats textbox
    stats_text = (
        f"n = {n}\n"
        f"n_unique = {n_unique}\n"
        f"max_gap = {max_gap:.2f}\n"
        f"median_gap = {median_gap:.2f}\n"
        f"P10 = {pct10_gap:.2f}, P50 = {pct50_gap:.2f}, P90 = {pct90_gap:.2f}\n"
        f"total_gap_prop = {total_gap_prop:.2f}\n"
        + (f"gaps > {threshold} = {n_gaps_above}" if threshold is not None else "")
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize='small', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    # Data source
    if data_source:
        fig.text(
            0.01, 0.01, f"Source: {data_source}",
            ha='left', va='bottom', fontsize='small', color='gray'
        )

    ax.legend()

    # Optional save
    rel_path = None
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        abs_path = os.path.join(save_path, file_name)
        fig.savefig(abs_path, bbox_inches='tight')
        rel_path = os.path.relpath(abs_path)

    return {
        'descriptive_stats': {
            'n': n,
            'n_unique': n_unique,
            'gaps': gaps_list,
            'max_gap': max_gap,
            'median_gap': median_gap,
            'pct10_gap': pct10_gap,
            'pct50_gap': pct50_gap,
            'pct90_gap': pct90_gap,
            'n_gaps_above_thr': n_gaps_above,
            'total_gap_prop': total_gap_prop,
            'max_gap_loc': max_gap_loc
        },
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'threshold': threshold,
            'relative_path': rel_path
        }
    }
