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
from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from analytics_eda.core.visualization.context import build_plot_context
from analytics_eda.core.visualization.validation import named_only_validator

from ...core.data_quality import (
    MissingDataBarContext,
    MissingDataBarPlot,
    StringCoercionBarContext,
    StringCoercionBarPlot,
)
from ...core.numeric import CardinalityBarContext, CardinalityBarPlot, numeric_distribution_analysis
from ...core.reporting import write_json_report

logger = logging.getLogger(__name__)

def univariate_numeric_analysis(
    series: pd.Series,
    report_root: str = 'reports/eda/univariate/numeric',
    report_log_id = str(uuid.uuid4()),
    data_source: str | None = None,
    filter_desc: str | None = None,
    distribution_names: Sequence[str] = ('norm', 'lognorm', 'gamma', 'expon'),
    plot_central_tendency_histogram_overrides: dict[str, Any] | None = None,
    plot_central_tendency_violin_overrides: dict[str, Any] | None = None,
    plot_dispersion_boxplot_overrides: dict[str, Any] | None = None,
    plot_distribution_ecdf_gap_overrides: dict[str, Any] | None = None,
    plot_distribution_ecdf_vs_cdf_overrides: dict[str, Any] | None = None,
    plot_distribution_density_overrides:      dict[str, Any] | None = None,
    plot_distribution_qq_fit_overrides: dict[str, Any] | None = None,
    plot_distribution_probability_overrides: dict[str, Any] | None = None,

    plot_missing_data_bar_overrides: dict[str, Any] | None = None,
    plot_string_coercion_bar_overrides: dict[str, Any] | None = None,
    plot_cardinality_bar_overrides: dict[str, Any] | None = None,
) -> Path:
    """
    Perform a comprehensive univariate analysis on a numeric pandas Series and
    generate a structured JSON report containing missing data, cardinality, and 
    distribution insights.

    This function validates the input series, computes descriptive statistics,
    conducts normality and distribution tests, generates relevant visualizations,
    and aggregates the results into a single, machine-readable report. The output 
    can be used for exploratory data analysis (EDA), data quality assessment, or 
    as part of automated profiling workflows.

    Analysis includes:
        1. Missing data profiling (counts, percentages, bar chart).
        2. Cardinality profiling (category counts, discrete/continuous flag, bar chart).
        3. Distribution profiling (descriptive stats, normality tests, histogram, 
        violin plot, ECDF, Q–Q plot, and other relevant charts).

    Args:
        series (pd.Series): Numeric series to analyze.
        report_root (str): Directory where report files will be saved.
        report_log_id (str): Identifier for logging and traceability.
        data_source (Optional[str]): Metadata describing the data source.
        distribution_names (Sequence[str]): Statistical distributions to fit 
            during goodness-of-fit analysis.
        *_overrides (dict): Optional keyword overrides for individual plot functions.

    Returns
    -------
        Dict[str, Path]: Dictionary containing the path to the generated JSON report.

    Report structure:
        {
            "metadata": { ... },
            "data": {
                "data_quality": { ... },
                "cardinality": { ... },
                "distribution": { ... }
            }
        }
    """
    # 1. Validation (Named, Typed)
    series_copy = named_only_validator(dropna=False, cast_str=False).validate(series)

    logger.info(
        "Starting univariate_numeric_analysis",
        extra={
            'series_name': series_copy.name,
            'report_log_id': report_log_id
        }
    )

    # Prepare directory
    report_path = Path(report_root) / series_copy.name.replace(' ', '_')
    report_path.mkdir(parents=True, exist_ok=True)

    # Convenience: base context kwargs shared by all plots
    common_base = {
        "save_path": report_path,
        "data_source": data_source,
        "filter_desc": filter_desc,
    }

    # 1. Data Quality & Standardization
    data_quality = {}
    # Missing Data Analysis
    md_ctx = build_plot_context(
        MissingDataBarContext,
        base=common_base,
        overrides=plot_missing_data_bar_overrides,
    )
    md_plot = MissingDataBarPlot(md_ctx)
    data_quality['missing_data_barchart'] = md_plot.run(series_copy)

    # Check for strings in numeric series.
    string_coercion_ctx = build_plot_context(
        StringCoercionBarContext,
        base=common_base,
        overrides=plot_string_coercion_bar_overrides,
    )
    string_coercion_plot = StringCoercionBarPlot(string_coercion_ctx)
    data_quality["string_coercion_barchart"] = string_coercion_plot.run(series_copy)

    series_copy = pd.to_numeric(series_copy, errors="coerce").dropna()

    # Cardinality Analysis
    card_ctx = build_plot_context(
        CardinalityBarContext,
        base=common_base,
        overrides=plot_cardinality_bar_overrides,
    )
    card_plot = CardinalityBarPlot(card_ctx)
    cardinality_bar_plot_result = card_plot.run(series_copy)

    is_discrete = cardinality_bar_plot_result['descriptive_stats']['is_discrete']

    cardinality = {
        'barchart': cardinality_bar_plot_result
    }

    # 2. Distribution Analysis
    distribution_result = numeric_distribution_analysis(
        series_copy,
        is_discrete=is_discrete,
        data_source=data_source,
        filter_desc=filter_desc,
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
        plot_distribution_probability_overrides=plot_distribution_probability_overrides,
    )

    # Generate report
    eda_report = {
        'data_quality': data_quality,
        'cardinality': cardinality,
        'distribution': distribution_result
    }

    full_report = {
        'metadata': {
            'version': '1.0.0',
            'report_name': 'univariate_numeric_analysis',
            'parameters': {
                'series': series_copy.name
            }
        },
        'data': eda_report
    }

    report_file_path = report_path / f"{series_copy.name.replace(' ', '_')}_univariate_analysis_report.json"
    write_json_report(full_report, report_file_path)

    logger.info(
        "Completed univariate_numeric_analysis",
        extra={
            'series_name': series_copy.name,
            'report_log_id': report_log_id
        }
    )

    return {
        'report_file_path': report_file_path
    }
