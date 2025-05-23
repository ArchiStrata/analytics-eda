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
from scipy import stats

from .validate_numeric_named_series import validate_numeric_named_series

logger = logging.getLogger(__name__)

def normality_assessment(
    s: pd.Series,
    alpha: float = 0.05,
    report_log_id: str = str(uuid.uuid4())
) -> dict:
    """
    Perform formal normality tests on a numeric Series (NaNs dropped).

    Assumptions & behavior:
      - Input Series must be numeric; all NaNs are dropped before testing.
      - For n < 50, uses Shapiro–Wilk (best small-sample power).
      - For 20 ≤ n < 50, uses D’Agostino–Pearson omnibus (requires n ≥ 20).
      - For n ≥ 50, runs both D’Agostino–Pearson and Anderson–Darling.
      - For n > 2000, runs Jarque–Bera (moment-based).
      - A top-level 'reject_normality' flag is included if any test rejects H₀ of normality.

    Args:
        s (pd.Series): Numeric data series.
        alpha (float): Significance level for tests (e.g. 0.05).
        report_log_id (str): report log id.

    Returns:
        dict: {
            'n': int,              # sample size after dropping NaNs
            'shapiro'?: {...},     # present if n < 50
            'dagostino_pearson'?: {...},   # present if n ≥ 20
            'anderson': {...},     # always present
            'jarque_bera': {...},  # present if n > 2000
            'reject_normality': bool
        }
        Returns {} if no non-null data.
    """
    validate_numeric_named_series(s)

    logger.info(
        "Starting normality_assessment",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    s_clean = s.dropna()
    n = len(s_clean)
    if n == 0:
        return {}

    result: dict = {'n': n}

    # 1. Shapiro–Wilk for small samples
    if n < 50:
        stat_sw, p_sw = stats.shapiro(s_clean)
        result['shapiro'] = {
            'statistic': float(stat_sw),
            'p_value': float(p_sw),
            'reject': bool(p_sw < alpha)
        }

    # 2. D’Agostino–Pearson omnibus (best for n >= 20)
    if n >= 20:
        stat_dp, p_dp = stats.normaltest(s_clean)
        result['dagostino_pearson'] = {
            'statistic': float(stat_dp),
            'p_value': float(p_dp),
            'reject': bool(p_dp < alpha)
        }

    # 3. Anderson–Darling
    ad = stats.anderson(s_clean, dist='norm')
    # choose the critical value matching alpha (percent)
    pct_levels = [sl / 100.0 for sl in ad.significance_level]
    try:
        idx = pct_levels.index(alpha)
    except ValueError:
        # fallback: pick closest
        idx = min(range(len(pct_levels)),
                  key=lambda i: abs(pct_levels[i] - alpha))
    ad_reject = bool(ad.statistic > ad.critical_values[idx])

    result['anderson'] = {
        'statistic': float(ad.statistic),
        'critical_values': list(map(float, ad.critical_values)),
        'significance_levels': list(map(float, ad.significance_level)),
        'reject': ad_reject
    }

    # 4. Jarque–Bera
    if n > 2000:
        stat, p = stats.jarque_bera(s_clean)
        result['jarque_bera'] = {
            'statistic': float(stat),
            'p_value': float(p),
            'reject': p < alpha
        }

    # 5. Bubble up overall normality decision
    result['reject_normality'] = any(
        info.get('reject', False)
        for info in result.values()
        if isinstance(info, dict)
    )

    logger.info(
        "Completed normality_assessment",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    return result
