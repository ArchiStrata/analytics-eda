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
from typing import Any, Dict, Optional
import uuid
from pathlib import Path
import pandas as pd

from .validate_categorical_named_series import validate_categorical_named_series

from .plot_frequency_pareto import plot_frequency_pareto

from ..numeric import plot_distribution_density, plot_dispersion_boxplot
from ..reporting import write_json_report
from ..utils import call_plot_with_overrides

logger = logging.getLogger(__name__)

def categorical_distribution_analysis(
    series: pd.Series,
    report_path: Path,
    report_log_id = str(uuid.uuid4()),
    data_source: Optional[str] = None,
    plot_frequency_pareto_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_density_overrides: Optional[Dict[str, Any]] = None,
    plot_dispersion_boxplot_overrides: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Analyze a categorical pandas Series and produce a structured report with summary
    statistics, a frequency table, and a top-N bar chart (aggregating all others).

    Parameters
    ----------
    series : pd.Series
        Categorical data to analyze (dtype 'category' or 'object').
    report_path : pathlib.Path
    report_log_id (str): report log id.
    """
    # validate input
    validate_categorical_named_series(series)

    logger.info(
        "Starting categorical_distribution_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    # 1. Frequency Distribution
    # What it is: A listing of each category alongside its count and proportion.
    # Why it matters: Shows which categories dominate and which are rare.
    frequency_distribution = {}

    # Pareto
    freq_pareto_over = (plot_frequency_pareto_overrides or {}).copy()
    frequency_distribution['pareto'] = call_plot_with_overrides(
        plot_frequency_pareto,
        series,
        overrides=freq_pareto_over,
        save_path=report_path,
        data_source=data_source,
    )

    # 2. Cardinality & Balance
    # What it is:  
    #   Assessment of category cardinality and balance –  
    #   counting unique categories, identifying rare “tail” categories (often grouped as ‘Others’),  
    #   and quantifying how evenly observations are distributed using metrics like entropy and the Gini index.
    # Why it matters: Tells you if you have too many categories to handle, or if one category overwhelms the rest.

    balance = {}

    freq_counts = series.copy().dropna().value_counts()

    # Density plot (Histogram + KDE)
    balance_density_over = (plot_distribution_density_overrides or {}).copy()
    balance_density_over.setdefault('xlabel', 'Frequency')
    balance['density'] = call_plot_with_overrides(
        plot_distribution_density,
        freq_counts,
        overrides=balance_density_over,
        save_path=report_path,
        data_source=data_source,
    )

    # Boxplot + Violin (Dispersion)
    balance_boxplot_over = (plot_dispersion_boxplot_overrides or {}).copy()
    balance_boxplot_over.setdefault('ylabel', 'Frequency')
    balance['boxplot'] = call_plot_with_overrides(
        plot_dispersion_boxplot,
        freq_counts,
        overrides=balance_boxplot_over,
        save_path=report_path,
        data_source=data_source,
    )

    # TODO: Rare categories (e.g. <1% of total) - bar chart

    # TODO: Side-by-side bar chart with Chi-square goodness-of-fit against a uniform distribution

    # TODO: Lorenz curve with Gini index

    # compile report
    distribution_report = {
        'frequency_distribution': frequency_distribution,
        'balance': balance
    }

    full_report = {
        'metadata': {
            'version': '1.0.0',
            'report_name': 'categorical_distribution_analysis',
            'parameters': {
                'series': series.name
            }
        },
        'data': distribution_report
    }

    logger.info(
        "Completed categorical_distribution_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    report_file_name = f"{series.name.replace(' ', '_')}_categorical_distribution_analysis_report.json"
    report_file_path = report_path / report_file_name
    write_json_report(full_report, report_file_path)

    return {
        'report_file_path': report_file_name
    }
