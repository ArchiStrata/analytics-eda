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
from collections.abc import Sequence
import logging
import uuid
import warnings
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

def distribution_fit_assessment(
    s: pd.Series,
    alpha: float = 0.05,
    distributions: Sequence[str] = ('norm', 'lognorm', 'gamma', 'expon'),
    report_log_id: str = str(uuid.uuid4())
) -> dict:
    """
    Assess non-normal data by:
      1. Fitting a set of candidate distributions & running KS/AD GOF.
      2. Running a binned χ² goodness-of-fit to a target distribution.

    Args:
        s (pd.Series): The data to assess.
        alpha (float): Significance level for all tests.
        distributions: Names of scipy.stats distributions to fit & test.
        report_log_id (str): report log id.

    Returns:
        dict: {
            'alternative_fits': {
                dist_name: { 'params': tuple, 'ks': {...}, 'ad'?: {...} }, …
            }
        }
    """
    logger.info(
        "Starting distribution_fit_assessment",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    # 1. Fit & GOF for each candidate
    alt_fits = {}
    for name in distributions:
        logger.debug(
            "Performing alternative fit",
            extra={
                'series_name': s.name,
                'report_log_id': report_log_id,
                'distribution_type': name
            }
        )
        dist = getattr(stats, name)

        # skip known-positive-only if data has non-positives
        if name in ('lognorm','gamma') and s.min() <= 0:
            alt_fits[name] = {'error': 'requires positive data'}
            continue

        if name == 'expon':
            # OK to include zeros, just guard against negatives:
            if s.min() < 0:
                alt_fits['expon'] = {'error': 'requires non-negative data'}
                continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                params = dist.fit(s)
        except Exception as e:
            logger.exception(
                "distribution_fit_assessment failed", 
                extra={
                    'series_name': s.name,
                    'report_log_id': report_log_id
                }
            )
            alt_fits[name] = {
                'error': str(e),
                'report_log_id': report_log_id
            }
            continue

        ks_stat, ks_p = stats.kstest(s, name, params)
        alt_fits[name] = {
            'params': params,
            'ks': {'statistic': ks_stat, 'p_value': ks_p, 'reject': ks_p < alpha}
        }

    logger.info(
        "Completed distribution_fit_assessment",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    return {
        'alternative_fits': alt_fits
    }
