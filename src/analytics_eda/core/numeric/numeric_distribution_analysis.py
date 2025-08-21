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
from typing import Optional, Dict, Any, Callable, Sequence
import pandas as pd

from .central_tendency_histogram_plot import CentralTendencyHistogramContext, CentralTendencyHistogramPlot
from .central_tendency_violin_plot import CentralTendencyViolinContext, CentralTendencyViolinPlot
from .dispersion_box_plot import DispersionBoxplotContext, DispersionBoxPlot
from .distribution_ecdf_gap_plot import DistributionECDFGapContext, DistributionECDFGapPlot
from .distribution_density_plot import DistributionDensityContext, DistributionDensityPlot
from .distribution_ecdf_vs_cdf_plot import DistributionECDFvsCDFContext, DistributionECDFvsCDFPlot
from .distribution_qq_fit_plot import DistributionQqFitContext, DistributionQqFitPlot
from .distribution_probability_function_plot import DistributionProbabilityFunctionContext, DistributionProbabilityFunctionPlot


from .validate_numeric_named_series import validate_numeric_named_series

from ..reporting import write_json_report
from ..utils.build_plot_context import build_plot_context

logger = logging.getLogger(__name__)

def numeric_distribution_analysis(
    series: pd.Series,
    is_discrete: bool,
    report_path: Path,
    report_log_id: str = str(uuid.uuid4()),
    distribution_names: Sequence[str] = ('norm', 'lognorm', 'gamma', 'expon'),
    evaluate_transforms_fn: Optional[
        Callable[
            [pd.Series, dict, dict, Path],
            dict
        ]
    ] = None,
    data_source: Optional[str] = None,
    filter_desc: Optional[str] = None,
    transform_desc: Optional[str] = None,
    plot_central_tendency_histogram_overrides: Optional[Dict[str, Any]] = None,
    plot_central_tendency_violin_overrides: Optional[Dict[str, Any]] = None,
    plot_dispersion_boxplot_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_ecdf_gap_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_ecdf_vs_cdf_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_density_overrides:      Optional[Dict[str, Any]] = None,
    plot_distribution_qq_fit_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_probability_overrides: Optional[Dict[str, Any]] = None,
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

    # Convenience: base context kwargs shared by all plots
    common_base = {
        "save_path": report_path,
        "data_source": data_source,
        "filter_desc": filter_desc,
        "transform_desc": transform_desc,
    }

    # Central Tendency
    central_tendency = {}

    hist_ctx = build_plot_context(
        CentralTendencyHistogramContext,
        base=common_base,
        overrides=plot_central_tendency_histogram_overrides,
    )
    central_tendency["histogram"] = CentralTendencyHistogramPlot(hist_ctx).run(series)


    violin_ctx = build_plot_context(
        CentralTendencyViolinContext,
        base=common_base,
        overrides=plot_central_tendency_violin_overrides,
    )
    central_tendency["violin"] = CentralTendencyViolinPlot(violin_ctx).run(series)

    # TODO: Central Tendency time series analysis

    # Dispersion
    dispersion = {}
    box_ctx = build_plot_context(
        DispersionBoxplotContext,
        base=common_base,
        overrides=plot_dispersion_boxplot_overrides,
    )
    dispersion["boxplot"] = DispersionBoxPlot(box_ctx).run(series)

    # TODO: Dispersion time series analysis

    # Shape
    shape = {}

    # TODO: Shape time series analysis

    # ECDF gap plot
    ecdf_gap_ctx = build_plot_context(
        DistributionECDFGapContext,
        base=common_base,
        overrides=plot_distribution_ecdf_gap_overrides,
    )
    shape["ecdf_gap"] = DistributionECDFGapPlot(ecdf_gap_ctx).run(series)


    # Density plot
    dens_ctx = build_plot_context(
        DistributionDensityContext,
        base=common_base,
        overrides=plot_distribution_density_overrides,
    )
    shape["density"] = DistributionDensityPlot(dens_ctx).run(series)

    # probability
    prob_ctx = build_plot_context(
        DistributionProbabilityFunctionContext,
        base=common_base,
        overrides=plot_distribution_probability_overrides,
    )
    shape["probability"] = DistributionProbabilityFunctionPlot(prob_ctx).run(series)


    # Fit each theoretical distribution
    distribution_fits = {}
    for dist in distribution_names:
        ecdf_vs_cdf_ctx = build_plot_context(
            DistributionECDFvsCDFContext,
            base={**common_base, "distribution_name": dist},
            overrides=plot_distribution_ecdf_vs_cdf_overrides,
        )
        qq_ctx = build_plot_context(
            DistributionQqFitContext,
            base={**common_base, "distribution_name": dist},
            overrides=plot_distribution_qq_fit_overrides,
        )

        distribution_fits[dist] = {
            "ecdf_vs_cdf": DistributionECDFvsCDFPlot(ecdf_vs_cdf_ctx).run(series),
            "qq_fit": DistributionQqFitPlot(qq_ctx).run(series),
        }
    
    shape['distribution_fits'] = distribution_fits

    # optionally evaluate transforms on the 'norm' residuals
    if evaluate_transforms_fn and 'norm' in distribution_fits:
        norm_qq = distribution_fits['norm']['qq_fit']
        descriptive_stats   = norm_qq['descriptive_stats']
        inferential_stats   = norm_qq.get('inferential_stats', {})
        transforms_meta = evaluate_transforms_fn(
            series=series,
            is_discrete=is_discrete,
            descriptive_stats=descriptive_stats,
            normality_tests=inferential_stats,
            report_path=report_path,
            report_log_id=report_log_id,
            data_source=data_source,
            filter_desc=filter_desc,
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
        # expose only the inner mapping of name → analysis
        shape['transforms'] = transforms_meta.get('transforms', {})

    logger.info(
        "Completed numeric_distribution_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    distribution_report = {
            'central_tendency': central_tendency,
            'dispersion': dispersion,
            'shape': shape
        }

    full_report = {
        'metadata': {
            'version': '1.0.0',
            'report_name': 'numeric_distribution_analysis',
            'parameters': {
                'series': series.name,
                'distribution_names': distribution_names
            }
        },
        'data': distribution_report
    }

    report_file_name = f"{series.name.replace(' ', '_')}_numeric_distribution_analysis_report.json"
    report_file_path = report_path / report_file_name
    write_json_report(full_report, report_file_path)

    return {
        'report_file_path': report_file_name
    }
