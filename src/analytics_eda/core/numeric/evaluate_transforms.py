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
"""Evaluate a set of candidate transforms."""

from pathlib import Path
from typing import Any

import pandas as pd

from .numeric_distribution_analysis import numeric_distribution_analysis
from .select_transforms import select_transforms
from .transform_series import transform_series


def evaluate_transforms(
    series: pd.Series,
    is_discrete: bool,
    descriptive_stats: dict[str, Any],
    normality_tests: dict[str, Any],
    report_path: Path,
    report_log_id: str | None = None,
    data_source: str | None = None,
    filter_desc: str | None = None,
    distribution_names=None,
    plot_central_tendency_histogram_overrides=None,
    plot_central_tendency_mean_point_ci_overrides: dict[str, Any] | None = None,
    plot_central_tendency_median_point_ci_overrides: dict[str, Any] | None = None,
    plot_dispersion_boxplot_overrides=None,
    plot_distribution_ecdf_gap_overrides=None,
    plot_distribution_ecdf_vs_cdf_overrides=None,
    plot_distribution_density_overrides=None,
    plot_distribution_qq_fit_overrides=None,
    plot_distribution_probability_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply a suite of candidate transforms to a series and run full distribution analysis on each.

    Parameters
    ----------
    series : pd.Series
        Original numeric data.
    descriptive_stats : dict
        Descriptive statistics for `series`.
    normality_tests : dict
        Results of formal normality tests for `series`.
    report_path : Path
        Base directory where per-transform reports will be saved.

    Returns
    -------
    Dict

    {
       'transforms': dict Mapping from transform name to the metadata returned by `numeric_distribution_analysis`
    }
    """
    # Determine which transforms to try
    candidates = select_transforms(descriptive_stats, normality_tests)
    transforms: dict[str, Any] = {}

    for transform_name in candidates:
        # Prepare a subdirectory for this transform's plots and outputs
        transform_dir = report_path / transform_name
        transform_dir.mkdir(parents=True, exist_ok=True)

        # Apply the transformation
        transformed = transform_series(series, transform_name)

        # Analyze the transformed data
        analysis_meta = numeric_distribution_analysis(
            transformed,
            is_discrete=is_discrete,
            report_path=transform_dir,
            report_log_id=report_log_id,
            data_source=data_source,
            filter_desc=filter_desc,
            transform_desc=f"{transform_name} transform",
            distribution_names=distribution_names,
            plot_central_tendency_histogram_overrides=plot_central_tendency_histogram_overrides,
            plot_central_tendency_mean_point_ci_overrides=plot_central_tendency_mean_point_ci_overrides,
            plot_central_tendency_median_point_ci_overrides=plot_central_tendency_median_point_ci_overrides,
            plot_dispersion_boxplot_overrides=plot_dispersion_boxplot_overrides,
            plot_distribution_ecdf_gap_overrides=plot_distribution_ecdf_gap_overrides,
            plot_distribution_ecdf_vs_cdf_overrides=plot_distribution_ecdf_vs_cdf_overrides,
            plot_distribution_density_overrides=plot_distribution_density_overrides,
            plot_distribution_qq_fit_overrides=plot_distribution_qq_fit_overrides,
            plot_distribution_probability_overrides=plot_distribution_probability_overrides,
        )

        transforms[transform_name] = analysis_meta

    return {"transforms": transforms}
