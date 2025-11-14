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
"""End-to-end categorical distribution analysis with plots and a JSON report."""

import logging
from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from analytics_eda.core.visualization.validation import categorical_validator

from ..numeric import (
    DispersionBoxPlot,
    DispersionBoxPlotContext,
    DistributionDensityContext,
    DistributionDensityPlot,
)
from ..reporting import write_json_report
from ..visualization.context.build_plot_context import build_plot_context
from .balance_chi_square_uniform_plot import (
    BalanceChiSquareUniformContext,
    BalanceChiSquareUniformPlot,
)
from .balance_lorenz_curve_plot import BalanceLorenzCurveContext, BalanceLorenzCurvePlot
from .balance_rare_categories_plot import BalanceRareCategoriesContext, BalanceRareCategoriesPlot
from .frequency_pareto_plot import FrequencyParetoContext, FrequencyParetoPlot

logger = logging.getLogger(__name__)


def categorical_distribution_analysis(
    series: pd.Series,
    report_path: Path,
    report_log_id: str | None = None,
    data_source: str | None = None,
    filter_desc: str | None = None,
    plot_frequency_pareto_overrides: dict[str, Any] | None = None,
    plot_distribution_density_overrides: dict[str, Any] | None = None,
    plot_dispersion_boxplot_overrides: dict[str, Any] | None = None,
    plot_balance_chi_square_uniform_overrides: dict[str, Any] | None = None,
    plot_balance_lorenz_curve_overrides: dict[str, Any] | None = None,
    plot_balance_rare_categories_overrides: dict[str, Any] | None = None,
) -> dict:
    """
    Perform a comprehensive statistical and visual analysis of a categorical Series.

    This function is designed to help identify dominant categories, rare "tail" categories,
    and the overall distribution fairness of categorical data. It produces a combination of
    descriptive statistics, inferential tests, and plots that reveal category proportions,
    variability, and equality of distribution.

    The analysis includes:
      - **Frequency Distribution**: Counts and proportions of each category, visualized via a Pareto chart.
      - **Balance Metrics**:
          * Frequency density (histogram + KDE) of category counts.
          * Boxplot of category frequencies to highlight dispersion.
          * Chi-square goodness-of-fit test against a uniform distribution.
          * Lorenz curve with Gini index to measure category inequality.
          * Rare categories barchart.

    Results are saved to disk as a JSON report containing:
      - Chart metadata for each visualization.
      - Descriptive and inferential statistics for each analysis component.
      - Report metadata (version, parameters, identifiers).

    Parameters
    ----------
    series : pd.Series
        Categorical data to analyze (dtype 'category' or 'object').
    report_path : pathlib.Path
        Directory where the JSON report and generated plots will be saved.
    report_log_id : str, optional
        Unique identifier for logging and traceability.
    data_source : str, optional
        Source description to embed in plots and metadata.
    plot_*_overrides : dict, optional
        Keyword overrides for customizing the context of individual plots
        (e.g., axis labels, titles, save options). Keys must match the
        corresponding `PlotContext` fields.

    Returns
    -------
    dict
        Dictionary with the relative path to the generated JSON report.
    """
    # Generate a stable log id if none was provided.
    if report_log_id is None:
        report_log_id = str(uuid.uuid4())

    # validate input
    cleaned_series = categorical_validator().validate(series)

    logger.info("Starting categorical_distribution_analysis", extra={"series_name": cleaned_series.name, "report_log_id": report_log_id})

    # Convenience: base context kwargs shared by all plots
    common_base = {
        "save_path": report_path,
        "data_source": data_source,
        "filter_desc": filter_desc,
    }

    # 1. Frequency Distribution
    # What it is: A listing of each category alongside its count and proportion.
    # Why it matters: Shows which categories dominate and which are rare.
    frequency_distribution = {}

    # Pareto
    fp_ctx = build_plot_context(
        FrequencyParetoContext,
        base=common_base,
        overrides=plot_frequency_pareto_overrides,
    )
    fp_plot = FrequencyParetoPlot(fp_ctx)
    frequency_distribution["pareto"] = fp_plot.run(cleaned_series)

    # TODO: Word cloud
    # TODO: Categorical time series analysis - Category Drift: Do category definitions or distributions change over time?

    # 2. Balance
    # What it is:
    #   Assessment of category balance –
    #   counting unique categories, identifying rare “tail” categories (often grouped as ‘Others’),
    #   and quantifying how evenly observations are distributed using metrics like entropy and the Gini index.
    # Why it matters: Tells you if you have too many categories to handle, or if one category overwhelms the rest.

    balance = {}

    freq_counts = cleaned_series.value_counts()

    # Density plot (Histogram + KDE)
    dens_ctx = build_plot_context(
        DistributionDensityContext,
        base={**common_base, "xlabel": "Frequency"},
        overrides=plot_distribution_density_overrides,
    )
    dens_plot = DistributionDensityPlot(dens_ctx)
    balance["density"] = dens_plot.run(freq_counts)

    # Boxplot + Violin (Dispersion)
    box_ctx = build_plot_context(
        DispersionBoxPlotContext,
        base={**common_base, "ylabel": "Frequency"},
        overrides=plot_dispersion_boxplot_overrides,
    )
    box_plot = DispersionBoxPlot(box_ctx)
    balance["boxplot"] = box_plot.run(freq_counts)

    # Rare categories
    rare_cat_ctx = build_plot_context(
        BalanceRareCategoriesContext,
        base=common_base,
        overrides=plot_balance_rare_categories_overrides,
    )
    rare_cat_plot = BalanceRareCategoriesPlot(rare_cat_ctx)
    balance["rare_categories"] = rare_cat_plot.run(cleaned_series)

    # Chi-square goodness-of-fit against a uniform distribution
    chi_ctx = build_plot_context(
        BalanceChiSquareUniformContext,
        base={**common_base, "xlabel": "Frequency"},
        overrides=plot_balance_chi_square_uniform_overrides,
    )
    chi_plot = BalanceChiSquareUniformPlot(chi_ctx)
    balance["chi_square_uniform"] = chi_plot.run(cleaned_series)

    # Lorenz curve with Gini index
    lor_ctx = build_plot_context(
        BalanceLorenzCurveContext,
        base=common_base,
        overrides=plot_balance_lorenz_curve_overrides,
    )
    lor_plot = BalanceLorenzCurvePlot(lor_ctx)
    balance["lorenz_curve"] = lor_plot.run(cleaned_series)

    # compile report
    distribution_report = {"frequency_distribution": frequency_distribution, "balance": balance}

    full_report = {"metadata": {"version": "1.0.0", "report_name": "categorical_distribution_analysis", "parameters": {"series": cleaned_series.name}}, "data": distribution_report}

    logger.info("Completed categorical_distribution_analysis", extra={"series_name": cleaned_series.name, "report_log_id": report_log_id})

    report_file_name = f"{cleaned_series.name.replace(' ', '_')}_categorical_distribution_analysis_report.json"
    report_file_path = report_path / report_file_name
    write_json_report(full_report, report_file_path)

    return {"report_file_path": report_file_name}
