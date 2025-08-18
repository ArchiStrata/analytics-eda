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
from pandas.api.types import is_numeric_dtype, is_object_dtype

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.magnitude_central_tendency_anova_kruskal_plot import MagnitudeCentralTendencyAnovaKruskalContext, MagnitudeCentralTendencyAnovaKruskalPlot
from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.magnitude_distribution_overlap_density_plot import MagnitudeDistributionOverlapDensityContext, MagnitudeDistributionOverlapDensityPlot
from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.magnitude_effect_size_bar_plot import MagnitudeEffectSizeBarContext, MagnitudeEffectSizeBarPlot
from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.relationship_structure_group_size_bar_plot import RelationshipStructureGroupSizeBarContext, RelationshipStructureGroupSizeBarPlot
from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.relationship_structure_variance_homogeneity_box_plot import RelationshipStructureVarianceHomogeneityBoxPlot, RelationshipStructureVarianceHomogeneityContext
from analytics_eda.core.utils import build_plot_context

from ...univariate import univariate_numeric_analysis
from ....core.reporting import write_json_report

logger = logging.getLogger(__name__)

def categorical_numeric_relationship_analysis(
    df: pd.DataFrame,
    numeric_col: str,
    categorical_col: str,
    report_root: str = 'reports/eda/bivariate/categorical_numeric_relationship_analysis',
    report_log_id = str(uuid.uuid4()),
    data_source: Optional[str] = None,

    plot_relationship_structure_group_size_bar_overrides: Optional[Dict[str, Any]] = None,
    plot_relationship_structure_variance_homogeneity_box_overrides: Optional[Dict[str, Any]] = None,
    plot_magnitude_distribution_overlap_density_overrides: Optional[Dict[str, Any]] = None,
    plot_magnitude_central_tendency_anova_kruskal_overrides: Optional[Dict[str, Any]] = None,
    plot_magnitude_effect_size_barchart_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict:
    """
    Run univariate numeric analysis on segments defined by a categorical column.

    Args:
        df (pd.DataFrame): The dataset.
        numeric_col (str): Numeric column to analyze.
        categorical_col (str): Column to segment by.
        report_root (str): Root directory for saving reports.
        report_log_id (str): report log id.
        **kwargs: Additional arguments passed to univariate_numeric_analysis (e.g., alpha, iqr_multiplier).
    
    Returns:
     Dict:
        - report_file_path: File path to the saved JSON report as written by `write_json_report`.
    """
    logger.info(
        "Starting categorical_numeric_relationship_analysis",
        extra={
            'numeric_col': numeric_col,
            'categorical_col': categorical_col,
            'report_root': report_root,
            'report_log_id': report_log_id
        }
    )

    if categorical_col not in df.columns:
        raise KeyError(f"Categorical column '{categorical_col}' not found.")
    if numeric_col not in df.columns:
        raise KeyError(f"Numeric column '{numeric_col}' not found.")
    
    if not (isinstance(df[categorical_col].dtype, pd.CategoricalDtype) or is_object_dtype(df[categorical_col])):
        raise TypeError(f"Column '{categorical_col}' must be categorical or object.")
    if not is_numeric_dtype(df[numeric_col]):
        raise TypeError(f"Column '{numeric_col}' must be numeric.")

    report_path = Path(report_root) / f"categorical_{categorical_col}_numeric_{numeric_col}_relationship_analysis"
    report_path.mkdir(parents=True, exist_ok=True)

    # Always work from a copy
    df_copy = df.copy()

    # Convenience: base context kwargs shared by all plots
    common_base = {
        "save_path": report_path,
        "data_source": data_source,
    }

    # Relationship Structure - What does the relationship look like?
    relationship_structure = {}

    rs_group_size_bar_ctx = build_plot_context(
        RelationshipStructureGroupSizeBarContext,
        base=common_base,
        overrides=plot_relationship_structure_group_size_bar_overrides,
    )
    rs_group_size_bar_plot = RelationshipStructureGroupSizeBarPlot(rs_group_size_bar_ctx)
    relationship_structure['group_size_barchart'] = rs_group_size_bar_plot.run(df_copy, cols=[categorical_col, numeric_col], role_map={"x": categorical_col, "y": numeric_col})

    rs_var_homogeneity_box_ctx = build_plot_context(
        RelationshipStructureVarianceHomogeneityContext,
        base=common_base,
        overrides=plot_relationship_structure_variance_homogeneity_box_overrides,
    )
    rs_var_homogeneity_box_plot = RelationshipStructureVarianceHomogeneityBoxPlot(rs_var_homogeneity_box_ctx)
    relationship_structure['variance_homogeneity_boxplot'] = rs_var_homogeneity_box_plot.run(df_copy, cols=[categorical_col, numeric_col], role_map={"x": categorical_col, "y": numeric_col})

    numeric_distribution_by_category = {}
    for category, group_df in df_copy.groupby(categorical_col, observed=True):
        category_slug = str(category).replace(" ", "_")
        category_report_root = report_path / f"{categorical_col}_{category_slug}"
        logger.debug("Running univariate analysis for category",
                extra={
                    'category': category,
                    'numeric_col': numeric_col,
                    'categorical_col': categorical_col,
                    'report_log_id': report_log_id
                })

        try:
            report = univariate_numeric_analysis(
                group_df[numeric_col],
                report_root=category_report_root,
                report_log_id=report_log_id,
                data_source=data_source,
                filter_desc=f"filtered by {categorical_col}={category_slug}",
                **kwargs
            )
            numeric_distribution_by_category[category] = report
        except Exception as e:
            # NOTE: If a category analysis fails we still want to continue with the remaining categories.
            logger.exception(
                "univariate_numeric_analysis failed", 
                extra={
                    'category': category,
                    'numeric_col': numeric_col,
                    'categorical_col': categorical_col,
                    'report_log_id': report_log_id
                }
            )
            numeric_distribution_by_category[category] = {
                'error': str(e),
                'report_log_id': report_log_id
            }
    
    relationship_structure['numeric_distribution_by_category'] = numeric_distribution_by_category
    
    # Magnitude of Association - How strongly are the two variables related?
    magnitude_of_association = {}

    mag_dist_overlap_density_ctx = build_plot_context(
        MagnitudeDistributionOverlapDensityContext,
        base=common_base,
        overrides=plot_magnitude_distribution_overlap_density_overrides,
    )
    mag_dist_overlap_density_plot = MagnitudeDistributionOverlapDensityPlot(mag_dist_overlap_density_ctx)
    magnitude_of_association['distribution_overlap_density'] = mag_dist_overlap_density_plot.run(df_copy, cols=[categorical_col, numeric_col], role_map={"x": categorical_col, "y": numeric_col})


    mag_central_tendency_anova_kruskal_ctx = build_plot_context(
        MagnitudeCentralTendencyAnovaKruskalContext,
        base=common_base,
        overrides=plot_magnitude_central_tendency_anova_kruskal_overrides,
    )
    mag_central_tendency_anova_kruskal_plot = MagnitudeCentralTendencyAnovaKruskalPlot(mag_central_tendency_anova_kruskal_ctx)
    magnitude_of_association['central_tendency_anova_kruskal_plot'] = mag_central_tendency_anova_kruskal_plot.run(df_copy, cols=[categorical_col, numeric_col], role_map={"x": categorical_col, "y": numeric_col})


    mag_effect_size_barchart_ctx = build_plot_context(
        MagnitudeEffectSizeBarContext,
        base=common_base,
        overrides=plot_magnitude_effect_size_barchart_overrides,
    )
    mag_effect_size_barchart_plot = MagnitudeEffectSizeBarPlot(mag_effect_size_barchart_ctx)
    magnitude_of_association['effect_size_barchart'] = mag_effect_size_barchart_plot.run(df_copy, cols=[categorical_col, numeric_col], role_map={"x": categorical_col, "y": numeric_col})

    # Direction of Association - Is the relationship positive, negative, or neutral?
    direction_of_association = {}

    # TODO: Post-hoc Pairwise Comparisons: Tukey HSD plot (confidence intervals for mean differences between each pair) and/or Heatmap of pairwise p-values - Tukey’s HSD
    # * DirectionPosthocTukeyHsdPlot

    # Post-hoc Tukey’s HSD (only if ANOVA significant)
                # tukey = pairwise_tukeyhsd(
                #     endog=df[numeric_col],
                #     groups=df[categorical_col],
                #     alpha=alpha
                # )
                # # Convert summary to dict or DataFrame
                # tukey_df = pd.DataFrame(
                #     tukey.summary().data[1:],
                #     columns=tukey.summary().data[0]
                # )
                # results['tukey_hsd'] = {
                #     'pairs': tukey_df.to_dict(orient='records')
                # }

    eda_report = {
        'relationship_structure': relationship_structure,
        'magnitude_of_association': magnitude_of_association,
        'direction_of_association': direction_of_association,
    }

    full_report = {
        'metadata': {
            'version': '0.1.0',
            'report_name': 'categorical_numeric_relationship_analysis',
            'parameters': {
                'numeric_col': numeric_col,
                'categorical_col': categorical_col
            }
        },
        'data': eda_report
    }

    report_file_path = report_path / f"categorical_{categorical_col}_numeric_{numeric_col}_relationship_analysis_report.json"
    full_report = write_json_report(full_report, report_file_path)

    logger.info(
        "Completed categorical_numeric_relationship_analysis",
        extra={
            'numeric_col': numeric_col,
            'categorical_col': categorical_col,
            'report_log_id': report_log_id,
            'report_file_path': str(report_file_path)
        }
    )

    return {
        'report_file_path': report_file_path
    }
