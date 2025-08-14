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
from typing import Any, Dict, Optional
import uuid
import pandas as pd

from ...core.categorical import validate_categorical_named_series, categorical_distribution_analysis
from ...core.reporting import write_json_report
from ...core.missing_data import MissingDataBarContext, MissingDataBarPlot
from ...core.numeric import CardinalityBarContext, CardinalityBarPlot
from ...core.data_quality import CategoricalCleanlinessBarPlot, CategoricalCleanlinessBarContext

logger = logging.getLogger(__name__)

def univariate_categorical_analysis(
    series: pd.Series,
    report_root: str = 'reports/eda/univariate/categorical',
    report_log_id = str(uuid.uuid4()),
    data_source: Optional[str] = None,
    plot_frequency_pareto_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_density_overrides: Optional[Dict[str, Any]] = None,
    plot_dispersion_boxplot_overrides: Optional[Dict[str, Any]] = None,
    plot_balance_chi_square_uniform_overrides: Optional[Dict[str, Any]] = None,
    plot_balance_lorenz_curve_overrides: Optional[Dict[str, Any]] = None,
    plot_balance_rare_categories_overrides: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Perform a comprehensive univariate analysis of a categorical pandas Series, 
    profiling its quality, cardinality, and distribution, and saving results as plots 
    and a structured JSON report.

    This analysis is designed for exploratory data analysis (EDA) and includes:
        • Missing data profiling — counts, percentages, and a visual barchart.
        • Cardinality assessment — number of distinct categories and frequency distribution.
        • Distribution analysis — frequency/Pareto plots, statistical tests, and balance metrics.

    All outputs are saved to the specified report directory, with key results aggregated 
    into a single JSON file for integration into automated reporting pipelines.

    Args:
        series (pd.Series): Named categorical Series (dtype 'category' or 'object').
        report_root (str, optional): Base directory for saving plots and the report.
        report_log_id (str, optional): Unique identifier for logging/report tracking.
        data_source (str, optional): Optional label for the dataset's origin.
        plot_frequency_pareto_overrides, plot_distribution_density_overrides, 
        plot_dispersion_boxplot_overrides, plot_balance_chi_square_uniform_overrides, plot_balance_rare_categories_overrides,
        plot_balance_lorenz_curve_overrides (dict, optional): 
            Per-plot configuration overrides.

    Returns:
        dict: {
            'report_file_path': Path to the saved JSON report
        }

    JSON report structure:
        {
            'metadata': { ... },
            'data': {
                'missing_data': {...},
                'cardinality': {...},
                'distribution': {...}
            }
        }
    """
    # 1. Validation
    validate_categorical_named_series(series)

    logger.info(
        "Starting univariate_categorical_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    # Always work from a copy
    series_copy = series.copy()

    # Prepare save directory
    report_path = Path(report_root) / series.name.replace(' ', '_')
    report_path.mkdir(parents=True, exist_ok=True)

    # 1. Data Quality & Standardization / categorical_variable_profiling
    # Missing Data Analysis - Detect missingness
    missing_data = {}

    md_ctx = MissingDataBarContext(save_path=report_path, data_source=data_source)
    md_plot = MissingDataBarPlot(md_ctx)
    missing_data['barchart'] = md_plot.run(series_copy)

    # Check categorical cleanliness
    data_quality = {}
    cat_clean_ctx = CategoricalCleanlinessBarContext(save_path=report_path, data_source=data_source)
    cat_clean_plot = CategoricalCleanlinessBarPlot(cat_clean_ctx)
    data_quality['categorical_cleanliness_barchart'] = cat_clean_plot.run(series_copy)
    cleaned, category_clean_meta = cat_clean_plot.clean_series(series_copy)
    data_quality['categorical_cleanliness_barchart']['cleaning_meta'] = category_clean_meta

    # Detect cardinality
    cardinality = {}

    card_ctx = CardinalityBarContext(save_path=report_path, data_source=data_source)
    card_plot = CardinalityBarPlot(card_ctx)
    freq_counts = cleaned.copy().dropna().value_counts()
    cardinality['barchart'] = card_plot.run(freq_counts)

    # TODO: Detect ordinality / monotonicity - Is the variable nominal (unordered) or ordinal (has natural order)?

    # 2. Distribution Analysis
    distribution_result = categorical_distribution_analysis(
        cleaned,
        report_path=report_path,
        report_log_id=report_log_id,
        data_source=data_source,
        plot_frequency_pareto_overrides=plot_frequency_pareto_overrides,
        plot_distribution_density_overrides=plot_distribution_density_overrides,
        plot_dispersion_boxplot_overrides=plot_dispersion_boxplot_overrides,
        plot_balance_chi_square_uniform_overrides=plot_balance_chi_square_uniform_overrides,
        plot_balance_lorenz_curve_overrides=plot_balance_lorenz_curve_overrides,
        plot_balance_rare_categories_overrides=plot_balance_rare_categories_overrides
    )

    # Generate report
    eda_report = {
        'missing_data': missing_data,
        'data_quality': data_quality,
        "cardinality": cardinality,
        'distribution': distribution_result
    }

    full_report = {
        'metadata': {
            'version': '0.1.0',
            'report_name': 'univariate_categorical_analysis',
            'parameters': {
                'series': cleaned.name
            }
        },
        'data': eda_report
    }

    report_file_path = report_path / f"{cleaned.name.replace(' ', '_')}_univariate_analysis_report.json"
    write_json_report(full_report, report_file_path)

    logger.info(
        "Completed univariate_categorical_analysis",
        extra={
            'series_name': cleaned.name,
            'report_log_id': report_log_id
        }
    )

    return {
        'report_file_path': report_file_path
    }
