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
import inspect
from typing import Optional, Dict, Any, Callable, Sequence
import pandas as pd

from .plot_central_tendency_histogram import plot_central_tendency_histogram
from .plot_central_tendency_violin import plot_central_tendency_violin
from .plot_dispersion_boxplot       import plot_dispersion_boxplot
from .plot_distribution_ecdf_gap    import plot_distribution_ecdf_gap
from .plot_distribution_ecdf_vs_cdf import plot_distribution_ecdf_vs_cdf
from .plot_distribution_density         import plot_distribution_density
from .plot_distribution_qq_fit import plot_distribution_qq_fit

from .validate_numeric_named_series import validate_numeric_named_series

logger = logging.getLogger(__name__)

def numeric_distribution_analysis(
    series: pd.Series,
    report_path: Path,
    report_log_id: str = str(uuid.uuid4()),
    distribution_names: Sequence[str] = ('norm', 'lognorm', 'gamma', 'expon'),
    evaluate_transforms_fn: Optional[
        Callable[
            [pd.Series, dict, dict, Path],
            dict
        ]
    ] = None,
    plot_central_tendency_histogram_overrides: Optional[Dict[str, Any]] = None,
    plot_central_tendency_violin_overrides: Optional[Dict[str, Any]] = None,
    plot_dispersion_boxplot_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_ecdf_gap_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_ecdf_vs_cdf_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_density_overrides:      Optional[Dict[str, Any]] = None,
    plot_distribution_qq_fit_overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Compute descriptive statistics, assess fit to common distributions, visualize
    distribution shape, and (optionally) evaluate variance-stabilizing transforms.

    Why:
        Provides a one-stop univariate EDA: 
        - central tendency and dispersion,
        - shape & tail characteristics,
        - formal goodness-of-fit to theoretical distributions,
        and—if requested—transform suggestions to improve normality.

    What:
        • Central-tendency histogram via `plot_central_tendency_histogram`
        • Dispersion boxplot via `plot_dispersion_boxplot`
        • ECDF gap analysis via `plot_distribution_ecdf_gap`
        • Density (histogram + KDE) via `plot_distribution_density`
        • ECDF vs. theoretical CDF for each in `distribution_names` via `plot_distribution_ecdf_vs_cdf`
        • Q–Q plot vs. each distribution via `plot_distribution_qq_fit`
        • Optional transform evaluation via `evaluate_transforms_fn`

    How:
        1. Validate and clean data (`validate_numeric_named_series`).
        2. Call each `plot_*` function with `call_plot_with_overrides`, saving results under `report_path`.
        3. Loop over `distribution_names`, fitting and plotting:
           - ECDF vs. CDF (`plot_distribution_ecdf_vs_cdf`)
           - Q–Q fit (`plot_distribution_qq_fit`)
        4. If `evaluate_transforms_fn` is provided, extract the 'norm' Q–Q stats/tests
           and invoke it to generate per-transform analyses.
        5. Return a nested dict with three top‐level keys:
           `'central_tendency'`, `'dispersion'`, and `'shape'`, each containing
           plot metadata and descriptive statistics.
    """
    validate_numeric_named_series(series)

    logger.info(
        "Starting numeric_distribution_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    # Central Tendency
    central_tendency = {}

    central_tendency_hist_over = (plot_central_tendency_histogram_overrides or {}).copy()
    if not central_tendency_hist_over.get('file_name'):
        central_tendency_hist_over['file_name'] = (
            f"Distribution of {series.name} (overview): Central Tendency.png"
    )

    central_tendency['histogram'] = call_plot_with_overrides(
        plot_central_tendency_histogram,
        series,
        overrides=central_tendency_hist_over,
        save_path=report_path,
    )

    central_tendency_violin_over = (plot_central_tendency_violin_overrides or {}).copy()
    if not central_tendency_violin_over.get('file_name'):
        central_tendency_violin_over['file_name'] = (
            f"Distribution of {series.name} (overview): Central Tendency (Violin).png"
    )

    central_tendency['violin'] = call_plot_with_overrides(
        plot_central_tendency_violin,
        series,
        overrides=central_tendency_violin_over,
        save_path=report_path,
    )

    # Dispersion
    dispersion_over = (plot_dispersion_boxplot_overrides or {}).copy()
    if not dispersion_over.get('file_name'):
        dispersion_over['file_name'] = (
            f"Dispersion of {series.name} (overview) (IQR & Outliers).png"
    )

    plot_dispersion_boxplot_meta = call_plot_with_overrides(
        plot_dispersion_boxplot,
        series,
        overrides=dispersion_over,
        save_path=report_path,
    )

    dispersion = {
        'boxplot': plot_dispersion_boxplot_meta,
    }

    # Shape
    shape = {}

    # ECDF gap plot
    ecdf_gap_over = (plot_distribution_ecdf_gap_overrides or {}).copy()

    # only set a default file_name if none was provided
    if not ecdf_gap_over.get('file_name'):
        ecdf_gap_over['file_name'] = (
            f"ECDF Gap Analysis of {series.name}.png"
        )

    # call the plotting helper with the overrides dict
    shape['ecdf_gap'] = call_plot_with_overrides(
        plot_distribution_ecdf_gap,
        series,
        overrides=ecdf_gap_over,
        save_path=report_path,
    )

    # prepare overrides for density plot
    density_over = (plot_distribution_density_overrides or {}).copy()

    # only set a default file_name if none was provided
    if not density_over.get('file_name'):
        density_over['file_name'] = (
            f"Distribution Density of {series.name}.png"
        )

    # call the plotting helper with the overrides dict
    shape['density'] = call_plot_with_overrides(
        plot_distribution_density,
        series,
        overrides=density_over,
        save_path=report_path,
    )

    # Fit each theoretical distribution
    distribution_fits = {}
    for dist in distribution_names:
        # force the distribution_name to the current dist
        ecdf_vs_cdf_over = (plot_distribution_ecdf_vs_cdf_overrides or {}).copy()
        ecdf_vs_cdf_over['distribution_name'] = dist
        ecdf_vs_cdf_over['file_name'] = f"ECDF vs. Theoretical CDF ({dist}).png"

        ecdf_vs_cdf_meta = call_plot_with_overrides(
            plot_distribution_ecdf_vs_cdf,
            series,
            overrides=ecdf_vs_cdf_over,
            save_path=report_path,
        )

        # Q–Q fit
        qq_fit_over = (plot_distribution_qq_fit_overrides or {}).copy()
        qq_fit_over['distribution_name'] = dist
        qq_fit_over['file_name'] = f"Q–Q Plot Fit Assessment for ({dist}).png"

        qq_meta = call_plot_with_overrides(
            plot_distribution_qq_fit,
            series,
            overrides=qq_fit_over,
            save_path=report_path,
        )

        distribution_fits[dist] = {
            'ecdf_vs_cdf': ecdf_vs_cdf_meta,
            'qq': qq_meta
        }
    
    shape['distribution_fits'] = distribution_fits

    # optionally evaluate transforms on the 'norm' residuals
    if evaluate_transforms_fn and 'norm' in distribution_fits:
        norm_qq = distribution_fits['norm']['qq']
        stats   = norm_qq['descriptive_stats']
        tests   = norm_qq.get('tests', {})
        transforms_meta = evaluate_transforms_fn(
            series=series,
            statistics=stats,
            normality_tests=tests,
            report_path=report_path,
            report_log_id=report_log_id,
            distribution_names=distribution_names,
            plot_central_tendency_histogram_overrides=plot_central_tendency_histogram_overrides,
            plot_central_tendency_violin_overrides=plot_central_tendency_violin_overrides,
            plot_dispersion_boxplot_overrides=plot_dispersion_boxplot_overrides,
            plot_distribution_ecdf_gap_overrides=plot_distribution_ecdf_gap_overrides,
            plot_distribution_ecdf_vs_cdf_overrides=plot_distribution_ecdf_vs_cdf_overrides,
            plot_distribution_density_overrides=plot_distribution_density_overrides,
            plot_distribution_qq_fit_overrides=plot_distribution_qq_fit_overrides,
        )
        # expose only the inner mapping of name → analysis
        shape['transforms'] = transforms_meta.get('transforms', {})

    logger.info(
        "Completed numeric_distribution_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    return {
        'report': {
            'central_tendency': central_tendency,
            'dispersion': dispersion,
            'shape': shape
        }
    }

def call_plot_with_overrides(
    plot_func: Callable[..., Dict[str, Any]],
    series: pd.Series,
    overrides: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generic wrapper to call a plotting function with overrideable kwargs + smart file_name defaulting.

    Steps:
      1. Inspect the target func’s signature.
      2. Start with its default parameter values.
      3. Pop out a 'file_name' override if provided.
      4. Apply any other overrides (error on unknown keys).
      5. If no file_name override, default to '{title}.png'.
      6. Invoke plot_func(series, **kwargs, save_path=save_path, file_name=file_name).
    """
    overrides = overrides.copy() if overrides else {}

    # 1. inspect signature
    sig = inspect.signature(plot_func)
    forbidden = {"series", "save_path", "file_name"}
    allowed = {p for p in sig.parameters if p not in forbidden}

    # 2. start with defaults
    plot_kwargs: Dict[str, Any] = {
        name: param.default
        for name, param in sig.parameters.items()
        if name in allowed
    }

    # 3. extract file_name override
    file_name = overrides.pop("file_name", None)

    # 4. apply remaining overrides
    for key, val in overrides.items():
        if key not in allowed:
            raise KeyError(f"'{key}' is not a valid parameter for {plot_func.__name__}")
        plot_kwargs[key] = val

    # 5. call the plot function
    return plot_func(
        series,
        **plot_kwargs,
        save_path=save_path,
        file_name=file_name,
    )
