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
from typing import Any, Dict, Literal, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from ..utils.build_chart_title import build_chart_title
from .validate_numeric_named_series import validate_numeric_named_series

MeanCIMethod = Literal['t', 'bootstrap']
MedianCIMethod = Literal['bootstrap', None]

def plot_central_tendency_violin(
    series: pd.Series,
    mean_ci_method: MeanCIMethod = 't',
    median_ci_method: MedianCIMethod = 'bootstrap',
    alpha: float = 0.05,
    bootstrap_samples: int = 1_000,
    popmean: Optional[float] = None,
    popmedian: Optional[float] = None,
    popvariance: Optional[float] = None,
    title_template: str = "Distribution of {name}{modifiers}: Central Tendency (Violin)",
    name: Optional[str] = None,
    filter_desc: Optional[str] = None,
    transform_desc: Optional[str] = None,
    xlabel: str = "Value",
    ylabel: str = "Density",
    data_source: Optional[str] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    file_name: Optional[str] = None
) -> dict:
    """
    Generate a horizontal violin plot showing distribution with overlaid point and error bars for
    mean and/or median confidence intervals.

    Tests shown when popmean and/or popmedian and/or popvariance are not None:
      - Cohen’s d, one-sample t-test, and (if popvariance) one-sample Z-test for popmean
      - Wilcoxon signed-rank and sign test for popmedian

    Parameters
    ----------
    series : pd.Series
        Numeric dataset to plot. Missing values will be dropped.
    mean_ci_method : {'t', 'bootstrap'}, default='t'
        Which CI to compute for the mean.
    median_ci_method : {'bootstrap', None}, default='bootstrap'
        Which CI to compute for the median, or None.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    bootstrap_samples : int, default=1000
        Number of resamples when using bootstrap methods.
    title_template : str
        Python format string for the chart title.
    name : str, optional
        Override for series.name in title.
    filter_desc : str, optional
    transform_desc : str, optional
    xlabel : str
    ylabel : str
    data_source : str, optional
    figsize : tuple
    save_path : str, optional
    file_name : str, optional

    Returns
    -------
    metadata : dict
        Contains descriptive_stats, tests, and chart_metadata.
    """
    # Validate input
    validate_numeric_named_series(series)
    data = series.dropna()
    n = data.size

    # Build title
    title = build_chart_title(
        name=name,
        series=series,
        filter_desc=filter_desc,
        transform_desc=transform_desc,
        title_template=title_template
    )

    # If empty
    if n == 0:
        return {
            'descriptive_stats': {
                'n': 0,
                'mean': np.nan,
                'median': np.nan,
                'mean_ci': (np.nan, np.nan),
                'median_ci': (np.nan, np.nan),
                'mean_ci_method': mean_ci_method,
                'median_ci_method': median_ci_method,
                'alpha': alpha,
                'bootstrap_samples': bootstrap_samples,
                'popmean': popmean,
                'popmedian': popmedian,
                'popvariance': popvariance
            },
            'tests': {},
            'chart_metadata': {
                'title': title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'data_source': data_source,
                'file_name': file_name
            }
        }

    # Compute central tendencies
    sample_mean = float(data.mean())
    sample_median = float(data.median())

    # Compute mean CI
    if mean_ci_method == 't':
        sem = stats.sem(data, ddof=1)
        mean_ci_low, mean_ci_high = stats.t.interval(1 - alpha, df=n - 1, loc=sample_mean, scale=sem)
    elif mean_ci_method == 'bootstrap':
        rng = np.random.default_rng()
        boot_means = rng.choice(data, size=(bootstrap_samples, n), replace=True).mean(axis=1)
        mean_ci_low, mean_ci_high = np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])
    else:
        raise ValueError("mean_ci_method must be 't' or 'bootstrap'")

    # Compute median CI
    if median_ci_method == 'bootstrap':
        rng = np.random.default_rng()
        boot_meds = rng.choice(data, size=(bootstrap_samples, n), replace=True)
        boot_meds = np.median(boot_meds, axis=1)
        med_ci_low, med_ci_high = np.percentile(boot_meds, [100*alpha/2, 100*(1-alpha/2)])
    else:
        raise ValueError("median_ci_method must be 'bootstrap'")

    # perform population tests
    stats_lines = []
    test_results: Dict[str, Any] = {}

    if popmean is not None:
        # One-Sample Cohen's d
        sd = data.std(ddof=1)
        cohens_d = (sample_mean - popmean) / sd if sd != 0 else None
        test_results['cohens_d'] = cohens_d

        # One-Sample t-Test
        t_stat, t_p = stats.ttest_1samp(data, popmean)
        test_results['t_test'] = {'statistic': float(t_stat), 'p_value': float(t_p), 'reject': bool(t_p < alpha)}
        stats_lines.append(
            f"Mean vs {popmean:.2f}: d={cohens_d:.2f}, "
            f"t={t_stat:.2f}, p={t_p:.3f} "
            f"{'(reject)' if test_results['t_test']['reject'] else '(ns)'}"
        )

        # One-sample Z-test (requires known σ²)
        if popvariance is not None:
            sigma = np.sqrt(popvariance)
            z_stat = (sample_mean - popmean) / (sigma / np.sqrt(n))
            z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            test_results['z_test'] = {'statistic': float(z_stat), 'p_value': float(z_p), 'reject': bool(z_p < alpha)}
            stats_lines.append(
                f"Z-test vs {popmean:.2f}: z={z_stat:.2f}, p={z_p:.3f} "
                f"{'(reject)' if test_results['z_test']['reject'] else '(ns)'}"
            )

    if popmedian is not None:
        # Wilcoxon Signed-Rank Test
        diff = data - popmedian
        stat_wr, p_wr = stats.wilcoxon(diff)
        test_results['wilcoxon'] = {'statistic': float(stat_wr), 'p_value': float(p_wr), 'reject': bool(p_wr < alpha)}
        stats_lines.append(
            f"Median vs {popmedian:.2f}: W={stat_wr:.2f}, p={p_wr:.3f} "
            f"{'(reject)' if test_results['wilcoxon']['reject'] else '(ns)'}"
        )

        # Sign test (binomial on signs)
        nonzero = diff[diff != 0]
        n_sign = len(nonzero)
        if n_sign > 0:
            pos = int((nonzero > 0).sum())
            sign_res = stats.binomtest(pos, n_sign, p=0.5)
            test_results['sign_test'] = {'num_positive': pos, 'num_negative': n_sign - pos, 'n': n_sign, 'p_value': float(sign_res.pvalue), 'reject': bool(sign_res.pvalue < alpha)}
            stats_lines.append(
                f"Sign test: +={pos}, -={n_sign-pos}, p={sign_res.pvalue:.3f} "
                f"{'(reject)' if test_results['sign_test']['reject'] else '(ns)'}"
            )

    # Plot
    sns.set_palette("colorblind")
    palette = sns.color_palette("colorblind")
    mean_col, med_col = palette[0], palette[1]

    fig, ax = plt.subplots(figsize=figsize)
    # Horizontal violin
    sns.violinplot(x=data, orient='h', inner=None, color='lightgray', ax=ax)

    # Overlay mean and its CI
    ax.errorbar(
        x=sample_mean,
        y=0,
        xerr=[[sample_mean - mean_ci_low], [mean_ci_high - sample_mean]],
        fmt='o', capsize=5, color=mean_col, label=f"Mean CI ({mean_ci_method}, {int((1-alpha)*100)}%)"
    )
    ax.axvline(sample_mean, color=mean_col, linestyle='--', label=f"Mean = {sample_mean:.2f}")

    # Overlay median and its CI if requested
    if median_ci_method == 'bootstrap':
        ax.errorbar(
            x=sample_median,
            y=0,
            xerr=[[sample_median - med_ci_low], [med_ci_high - sample_median]],
            fmt='s', capsize=5, color=med_col, label=f"Median CI (bootstrap, {int((1-alpha)*100)}%)"
        )
        ax.axvline(sample_median, color=med_col, linestyle='-.', label=f"Median = {sample_median:.2f}")

    # Labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    if ylabel:
        ax.set_ylabel(ylabel)

    # stats textbox
    if stats_lines:
        textbox = "\n".join(stats_lines)
        ax.text(0.01, 0.95, textbox, transform=ax.transAxes,
                fontsize='small', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))

    # Annotations
    if data_source:
        fig.text(0.01, 0.01, f"Source: {data_source}", ha='left', va='bottom', fontsize='small', color='gray')
    fig.text(0.99, 0.01, f"n = {n}", ha='right', va='bottom', fontsize='small', color='gray')

    ax.legend()

    # Save
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, file_name), bbox_inches='tight')

    # Return metadata
    return {
        'descriptive_stats': {
            'n': n,
            'mean': sample_mean,
            'median': sample_median,
            'mean_ci': (mean_ci_low, mean_ci_high),
            'median_ci': (med_ci_low, med_ci_high),
            'mean_ci_method': mean_ci_method,
            'median_ci_method': median_ci_method,
            'alpha': alpha,
            'bootstrap_samples': bootstrap_samples,
            'popmean': popmean,
            'popmedian': popmedian,
            'popvariance': popvariance
        },
        'tests': test_results,
        'chart_metadata': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_source': data_source,
            'file_name': file_name
        }
    }
