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
"""Balance-oriented categorical analysis built on BaseAnalysis."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics_eda.core.numeric import (
    DispersionBoxPlot,
    DispersionBoxPlotContext,
    ShapeDensityContext,
    ShapeDensityPlot,
)
from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.base_analysis import BaseAnalysis
from analytics_eda.core.visualization.context.build_plot_context import build_plot_context
from analytics_eda.core.visualization.validation import categorical_validator

from .balance_chi_square_uniform_plot import (
    BalanceChiSquareUniformPlot,
    BalanceChiSquareUniformPlotContext,
)
from .balance_lorenz_curve_plot import BalanceLorenzCurvePlot, BalanceLorenzCurvePlotContext
from .balance_rare_categories_plot import BalanceRareCategoriesPlot, BalanceRareCategoriesPlotContext


@dataclass
class CategoricalBalanceAnalysisContext(AnalysisContext):
    """
    Context for balance-focused categorical analyses.

    Includes optional per-plot context overrides for the density, boxplot,
    chi-square, Lorenz, and rare-category visualizations.
    """

    report_name: str = "categorical_balance_analysis"
    report_relative_path: str = "balance"
    report_file_name: str = "categorical_balance_analysis.json"
    distribution_density_plot_context: ShapeDensityContext | None = None
    dispersion_boxplot_plot_context: DispersionBoxPlotContext | None = None
    balance_chi_square_uniform_plot_context: BalanceChiSquareUniformPlotContext | None = None
    balance_lorenz_curve_plot_context: BalanceLorenzCurvePlotContext | None = None
    balance_rare_categories_plot_context: BalanceRareCategoriesPlotContext | None = None


class CategoricalBalanceAnalysis(BaseAnalysis):
    """
    Evaluate how evenly observations are distributed across categorical levels.

    Big idea:
        Surface dispersion, skew, and inequality across categories so analysts
        can identify whether the data is balanced, long-tailed, or dominated by
        a handful of categories.

    What this analysis does:
        Builds distribution density (counts of counts), dispersion boxplots,
        rare-category breakdowns, chi-square uniformity tests, and Lorenz curves.

    Why it matters:
        Balance diagnostics inform modeling strategies (e.g., weighting or
        stratification), feature engineering (e.g., grouping rare levels), and
        fairness considerations when categories map to demographic groups.
    """

    semantic_version = "1.0.0"
    context: CategoricalBalanceAnalysisContext

    def __init__(self, context: CategoricalBalanceAnalysisContext) -> None:
        """Instantiate the balance analysis with the supplied context and overrides."""
        super().__init__(context)

    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series:
        """Ensure the incoming data is a named categorical series."""
        return categorical_validator().validate(data_input)

    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """
        Produce balance-focused plots that describe categorical inequality.

        • Distribution Density — Histogram/KDE on frequency counts, revealing
          whether most categories share similar sizes or if counts vary widely.

        • Dispersion Boxplot — Box/violin view on frequency counts to surface
          spread, outliers, and potential dominant categories.

        • Rare Categories — Highlights categories with minimal representation,
          signaling whether long-tail handling or grouping is needed.

        • Chi-Square Uniform — Tests whether category counts deviate from a
          uniform expectation, quantifying imbalance via a statistical test.

        • Lorenz Curve — Visualizes cumulative inequality (plus Gini), providing
          a holistic view of dominance vs. parity across categories.
        """
        assert isinstance(data_input, pd.Series)
        freq_counts = data_input.value_counts()
        base_kwargs = self.base_kwargs()

        dens_ctx = build_plot_context(
            ShapeDensityContext,
            base=self.context.distribution_density_plot_context,
            overrides={**base_kwargs, "xlabel": "Frequency"},
        )
        dens_plot = ShapeDensityPlot(dens_ctx)

        box_ctx = build_plot_context(
            DispersionBoxPlotContext,
            base=self.context.dispersion_boxplot_plot_context,
            overrides={**base_kwargs, "ylabel": "Frequency"},
        )
        box_plot = DispersionBoxPlot(box_ctx)

        rare_ctx = build_plot_context(
            BalanceRareCategoriesPlotContext,
            base=self.context.balance_rare_categories_plot_context,
            overrides=base_kwargs,
        )
        rare_plot = BalanceRareCategoriesPlot(rare_ctx)

        chi_ctx = build_plot_context(
            BalanceChiSquareUniformPlotContext,
            base=self.context.balance_chi_square_uniform_plot_context,
            overrides={**base_kwargs, "xlabel": "Frequency"},
        )
        chi_plot = BalanceChiSquareUniformPlot(chi_ctx)

        lor_ctx = build_plot_context(
            BalanceLorenzCurvePlotContext,
            base=self.context.balance_lorenz_curve_plot_context,
            overrides=base_kwargs,
        )
        lor_plot = BalanceLorenzCurvePlot(lor_ctx)

        return {
            "density": dens_plot.run(freq_counts),
            "boxplot": box_plot.run(freq_counts),
            "rare_categories": rare_plot.run(data_input),
            "chi_square_uniform": chi_plot.run(data_input),
            "lorenz_curve": lor_plot.run(data_input),
        }
