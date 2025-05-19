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

from scipy.stats import chisquare

def categorical_inferential_analysis(freq_table: dict, total: int, alpha: float = 0.05) -> dict:
    """
    Perform inferential analysis on a categorical frequency distribution using
    Chi-square goodness-of-fit against a uniform distribution.

    Args:
        freq_table (dict): Mapping category -> {'count': int, 'proportion': float}
                           or mapping category -> count (int).
        total (int): Total number of observations (including missing if applicable).
        alpha (float, optional): Significance level. Default 0.05.

    Returns:
        dict: {
            'goodness_of_fit': {
                'chi2_statistic': float,
                'p_value': float,
                'alpha': float,
                'reject_null_uniform': bool
            }
        }
    """
    k = len(freq_table)
    expected = [total / k] * k
    observed = [v['count'] for v in freq_table.values()]
    chi2_stat, p_val = chisquare(f_obs=observed, f_exp=expected)

    return {
        'goodness_of_fit': {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_val),
            'alpha': float(alpha),
            'reject_null_uniform': bool(p_val < alpha)
        }
    }
