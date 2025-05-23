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

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import bartlett, f_oneway, kruskal, levene

from .compute_overlap_metrics import compute_overlap_metrics

logger = logging.getLogger(__name__)

def bivariate_numeric_categorical_tests(
        df: pd.DataFrame,
        numeric_col: str,
        categorical_col: str,
        alpha: float = 0.05,
        report_log_id: str = str(uuid.uuid4())) -> dict:
    """
    Perform a comprehensive numeric-vs-categorical bivariate analysis.

    Steps:
      1. Compute metadata:
         - Number of groups
         - Group sizes
      2. Distribution overlap:
         - Overlap Coefficient
         - Bhattacharyya Distance
      3. Homogeneity of variances:
         - Bartlett’s Test (assumes normality)
         - Levene’s Test (robust to non-normality)
      4. Global hypothesis tests:
         - ANOVA (parametric, equal variances)
         - Kruskal–Wallis (non-parametric)
      5. Post-hoc pairwise comparisons:
         - Tukey’s HSD (if ANOVA is significant)
      6. Effect size estimation:
         - Eta-squared (η²)
         - Omega-squared (ω²)
         - Epsilon-squared (ε²)

    Args:
        df: Input DataFrame.
        numeric_col: Name of the numeric column to analyze.
        categorical_col: Name of the categorical column defining groups.
        alpha: Significance level for all hypothesis tests (default: 0.05).
        report_log_id (str): report log id.

    Returns:
        A dict with keys:
          - 'meta': {
                'n_groups': int,
                'group_sizes': list[int]
            }
          - 'distribution_overlap': dict[str, {
                'overlap_coeff': float,
                'bhattacharyya_dist': float
            }]
          - 'bartlett': {statistic, p_value, reject} or {'error': str}
          - 'levene': {statistic, p_value, reject} or {'error': str}
          - 'anova': {statistic, p_value, reject} or {'error': str}
          - 'tukey_hsd': {'pairs': list[dict]} or {'error': str}
          - 'kruskal': {statistic, p_value, reject} or {'error': str}
          - 'effect_size': {
                'eta_squared': float,
                'omega_squared': float,
                'epsilon_squared': float
            }

        If fewer than 2 groups are present, returns {'error': 'Not enough groups…'}.
    """
    logger.info(
        "Starting bivariate_numeric_categorical_tests",
        extra={
            'numeric_col': numeric_col,
            'categorical_col': categorical_col,
            'report_log_id': report_log_id
        }
    )
    if numeric_col not in df.columns:
        raise KeyError(f"Numeric column '{numeric_col}' not found.")
    if categorical_col not in df.columns:
        raise KeyError(f"Categorical column '{categorical_col}' not found.")

    grouped = [
        group[numeric_col].dropna().values
        for _, group in df.groupby(categorical_col, observed=True)
    ]
    results = {}

    if len(grouped) < 2:
        return {"error": "Not enough groups to perform statistical tests."}

    group_sizes = [len(g) for g in grouped]
    results['meta'] = {
        'n_groups': len(grouped),
        'group_sizes': group_sizes
    }

    # Distribution Overlap
    try:
        grouped_dict = {
            str(label): group[numeric_col].dropna().values
            for label, group in df.groupby(categorical_col, observed=True)
        }
        results['distribution_overlap'] = compute_overlap_metrics(grouped_dict)
    except Exception as e:
        logger.exception(
                "bivariate_numeric_categorical_tests failed", 
                extra={
                    'numeric_col': numeric_col,
                    'categorical_col': categorical_col,
                    'report_log_id': report_log_id
                }
            )
        results['distribution_overlap'] = {
            'error': str(e),
            'report_log_id': report_log_id
        }

    # Homogeneity of variances

    # Bartlett’s Test (parametric, assumes normality)
    try:
        bart_stat, bart_p = bartlett(*grouped)
        results['bartlett'] = {
            'statistic': float(bart_stat),
            'p_value': float(bart_p),
            'reject': bool(bart_p < alpha)
        }
    except Exception as e:
        logger.exception(
                "bivariate_numeric_categorical_tests failed", 
                extra={
                    'numeric_col': numeric_col,
                    'categorical_col': categorical_col,
                    'report_log_id': report_log_id
                }
            )
        results['bartlett'] = {
            'error': str(e),
            'report_log_id': report_log_id
        }

    try:
        lev_stat, lev_p = levene(*grouped)
        results['levene'] = {
            'statistic': float(lev_stat),
            'p_value': float(lev_p),
            'reject': bool(lev_p < alpha)
        }
    except Exception as e:
        logger.exception(
                "bivariate_numeric_categorical_tests failed", 
                extra={
                    'numeric_col': numeric_col,
                    'categorical_col': categorical_col,
                    'report_log_id': report_log_id
                }
            )
        results['levene'] = {
            'error': str(e),
            'report_log_id': report_log_id
        }

    # ANOVA (parametric)
    try:
        anova_stat, anova_p = f_oneway(*grouped)
        results['anova'] = {
            'statistic': float(anova_stat),
            'p_value': float(anova_p),
            'reject': bool(anova_p < alpha)
        }

        # Post-hoc Tukey’s HSD (only if ANOVA significant)
        if anova_p < alpha:
            try:
                tukey = pairwise_tukeyhsd(
                    endog=df[numeric_col],
                    groups=df[categorical_col],
                    alpha=alpha
                )
                # Convert summary to dict or DataFrame
                tukey_df = pd.DataFrame(
                    tukey.summary().data[1:],
                    columns=tukey.summary().data[0]
                )
                results['tukey_hsd'] = {
                    'pairs': tukey_df.to_dict(orient='records')
                }
            except Exception as e:
                logger.exception(
                    "bivariate_numeric_categorical_tests failed", 
                    extra={
                        'numeric_col': numeric_col,
                        'categorical_col': categorical_col,
                        'report_log_id': report_log_id
                    }
                )
                results['tukey_hsd'] = {
                    'error': str(e),
                    'report_log_id': report_log_id
                }
    except Exception as e:
        logger.exception(
            "bivariate_numeric_categorical_tests failed", 
            extra={
                'numeric_col': numeric_col,
                'categorical_col': categorical_col,
                'report_log_id': report_log_id
            }
        )
        results['anova'] = {
            'error': str(e),
            'report_log_id': report_log_id
        }

    # Kruskal-Wallis (non-parametric)
    try:
        kruskal_stat, kruskal_p = kruskal(*grouped)
        results['kruskal'] = {
            'statistic': float(kruskal_stat),
            'p_value': float(kruskal_p),
            'reject': bool(kruskal_p < alpha)
        }
    except Exception as e:
        logger.exception(
            "bivariate_numeric_categorical_tests failed", 
            extra={
                'numeric_col': numeric_col,
                'categorical_col': categorical_col,
                'report_log_id': report_log_id
            }
        )
        results['kruskal'] = {
            'error': str(e),
            'report_log_id': report_log_id
        }

    # Effect Size Estimation

    # Flattened series for total SS
    all_values = df[numeric_col].dropna()
    grand_mean = all_values.mean()
    ss_total = ((all_values - grand_mean) ** 2).sum()

    # SS_between by looping over grouped + their means
    ss_between = sum(
    len(g) * (g.mean() - grand_mean) ** 2
    for g in grouped
    )
    ss_within = ss_total - ss_between

    k = len(grouped)
    N = len(all_values)
    ms_within = ss_within / (N - k)

    eta2   = ss_between / ss_total if ss_total > 0 else None
    omega2 = (
    (ss_between - (k - 1) * ms_within) /
    (ss_total + ms_within)
    ) if ss_total + ms_within > 0 else None

    eps2 = (kruskal_stat - k + 1) / (N - k) if N > k else None

    results['effect_size'] = {
        'eta_squared':   eta2,
        'omega_squared': omega2,
        'epsilon_squared': eps2
    }

    logger.info(
        "Completed bivariate_numeric_categorical_tests",
        extra={
            'numeric_col': numeric_col,
            'categorical_col': categorical_col,
            'report_log_id': report_log_id
        }
    )

    return results
