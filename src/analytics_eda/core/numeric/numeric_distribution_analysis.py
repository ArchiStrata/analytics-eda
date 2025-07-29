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
from typing import Optional, Dict, Any, Callable
import pandas as pd

from .plot_central_tendency_histogram import plot_central_tendency_histogram
from .plot_dispersion_boxplot       import plot_dispersion_boxplot
from .plot_distribution_ecdf_gap    import plot_distribution_ecdf_gap
from .plot_distribution_ecdf_vs_cdf import plot_distribution_ecdf_vs_cdf
from .plot_distribution_shape         import plot_distribution_shape
from .plot_distribution_qq_normality import plot_distribution_qq_normality

from .validate_numeric_named_series import validate_numeric_named_series

logger = logging.getLogger(__name__)

def numeric_distribution_analysis(
    series: pd.Series,
    report_path: Path,
    report_log_id: str = str(uuid.uuid4()),
    plot_central_tendency_histogram_overrides: Optional[Dict[str, Any]] = None,
    plot_dispersion_boxplot_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_ecdf_gap_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_ecdf_vs_cdf_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_shape_overrides:      Optional[Dict[str, Any]] = None,
    plot_distribution_qq_normality_overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Compute descriptive statistics, assess normality, visualize distribution,
    and determine if transformation is needed before fitting alternatives. Drops NAs.
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
    plot_central_tendency_histogram_meta = call_plot_with_overrides(
        plot_central_tendency_histogram,
        series,
        overrides=plot_central_tendency_histogram_overrides,
        save_path=report_path,
    )

    central_tendency = {
        'plot_central_tendency_histogram': plot_central_tendency_histogram_meta,
    }

    # Dispersion
    plot_dispersion_boxplot_meta = call_plot_with_overrides(
        plot_dispersion_boxplot,
        series,
        overrides=plot_dispersion_boxplot_overrides,
        save_path=report_path,
    )

    dispersion = {
        'plot_dispersion_boxplot': plot_dispersion_boxplot_meta,
    }

    # Shape
    ecdf_gap_meta = call_plot_with_overrides(
        plot_distribution_ecdf_gap,
        series,
        overrides=plot_distribution_ecdf_gap_overrides,
        save_path=report_path,
    )

    # ensure default distribution_name for ECDF vs CDF
    ecdf_vs_cdf_over = plot_distribution_ecdf_vs_cdf_overrides.copy() if plot_distribution_ecdf_vs_cdf_overrides else {}
    ecdf_vs_cdf_over.setdefault('distribution_name', 'norm')

    ecdf_vs_cdf_meta = call_plot_with_overrides(
        plot_distribution_ecdf_vs_cdf,
        series,
        overrides=ecdf_vs_cdf_over,
        save_path=report_path,
    )

    distribution_shape_meta = call_plot_with_overrides(
        plot_distribution_shape,
        series,
        overrides=plot_distribution_shape_overrides,
        save_path=report_path,
    )

    qq_meta = call_plot_with_overrides(
        plot_distribution_qq_normality,
        series,
        overrides=plot_distribution_qq_normality_overrides,
        save_path=report_path,
    )

    shape = {
        'plot_distribution_ecdf_gap': ecdf_gap_meta,
        'plot_distribution_ecdf_vs_cdf': ecdf_vs_cdf_meta,
        'plot_distribution_shape': distribution_shape_meta,
        'plot_distribution_qq_normality': qq_meta,
    }

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

    # 5. default file_name from title (if applicable)
    title = plot_kwargs.get("title")
    if file_name is None:
        file_name = f"{title}.png" if title else None

    # 6. call the plot function
    return plot_func(
        series,
        **plot_kwargs,
        save_path=save_path,
        file_name=file_name,
    )
