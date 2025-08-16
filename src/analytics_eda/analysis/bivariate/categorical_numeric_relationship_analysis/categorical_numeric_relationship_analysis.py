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
from typing import Optional
import uuid

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype

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
    **kwargs
) -> str:
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
     str: File path to the saved JSON report as written by `write_json_report`.
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

    report_dir = Path(report_root) / f"categorical_{categorical_col}_numeric_{numeric_col}_relationship_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)

    # TODO: relationship_structure - What does the relationship look like?
    relationship_structure = {}
    # TODO: support BivariateGroupSizeBarPlot

    # TODO: Spread & Variance Homogeneity:
    # * Homogeneity of Variances: Boxplots (side-by-side per group to eyeball variance differences) with violin (showing distribution shape + spread) and Error bar plot (mean ± SD per group)
    # * BivariateVarianceHomogeneityBoxPlot - 

        # bart_stat, bart_p = bartlett(*grouped)
        # results['bartlett'] = {
        #     'statistic': float(bart_stat),
        #     'p_value': float(bart_p),
        #     'reject': bool(bart_p < alpha)
        # }

        # lev_stat, lev_p = levene(*grouped)
        # results['levene'] = {
        #     'statistic': float(lev_stat),
        #     'p_value': float(lev_p),
        #     'reject': bool(lev_p < alpha)
        # }

    numeric_distribution_by_category = {}
    for category, group_df in df.groupby(categorical_col, observed=True):
        category_slug = str(category).replace(" ", "_")
        segment_report_root = report_dir / f"{categorical_col}_{category_slug}"
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
                report_root=segment_report_root,
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
    
    # TODO: magnitude_of_association - How strongly are the two variables related?
    magnitude_of_association = {}
    # TODO: support BivariateDistributionOverlapDensityPlot

    # Central Tendency Differences (Global Hypothesis Tests): Boxplots (with group medians highlighted for Kruskal) - ANOVA & Kruskal–Wallis
    # * BivariateGlobalTestAnovaBoxPlot

        # anova_stat, anova_p = f_oneway(*grouped)
        # results['anova'] = {
        #     'statistic': float(anova_stat),
        #     'p_value': float(anova_p),
        #     'reject': bool(anova_p < alpha)
        # }

        # kruskal_stat, kruskal_p = kruskal(*grouped)
        # results['kruskal'] = {
        #     'statistic': float(kruskal_stat),
        #     'p_value': float(kruskal_p),
        #     'reject': bool(kruskal_p < alpha)
        # }

    # Effect Size Estimation: Annotated boxplots (effect size shown in title or subtitle) - Eta-squared (η²), Omega-squared (ω²), Epsilon-squared (ε²)
    # * BivariateEffectSizeBoxPlot

    # Effect Size Estimation

    # Flattened series for total SS
    # all_values = df[numeric_col].dropna()
    # grand_mean = all_values.mean()
    # ss_total = ((all_values - grand_mean) ** 2).sum()

    # # SS_between by looping over grouped + their means
    # ss_between = sum(
    # len(g) * (g.mean() - grand_mean) ** 2
    # for g in grouped
    # )
    # ss_within = ss_total - ss_between

    # k = len(grouped)
    # N = len(all_values)
    # ms_within = ss_within / (N - k)

    # eta2   = ss_between / ss_total if ss_total > 0 else None
    # omega2 = (
    # (ss_between - (k - 1) * ms_within) /
    # (ss_total + ms_within)
    # ) if ss_total + ms_within > 0 else None

    # eps2 = (kruskal_stat - k + 1) / (N - k) if N > k else None

    # results['effect_size'] = {
    #     'eta_squared':   eta2,
    #     'omega_squared': omega2,
    #     'epsilon_squared': eps2
    # }

    # TODO: direction_of_association - Is the relationship positive, negative, or neutral?
    direction_of_association = {}

    # Post-hoc Pairwise Comparisons: Tukey HSD plot (confidence intervals for mean differences between each pair) and/or Heatmap of pairwise p-values - Tukey’s HSD
    # * BivariatePosthocTukeyHsdPlot

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

    report_file_path = report_dir / f"categorical_{categorical_col}_numeric_{numeric_col}_relationship_analysis_report.json"
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
