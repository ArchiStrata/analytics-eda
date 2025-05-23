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
import numpy as np

from scipy import stats

from .normality_assessment import normality_assessment
from .validate_numeric_named_series import validate_numeric_named_series
from .select_normality_transforms import select_normality_transforms

logger = logging.getLogger(__name__)

def assess_normality_and_transform(
    s: pd.Series,
    statistics: dict,
    normality: dict,
    alpha: float = 0.05,
    skew_thresh: float = 1.0,
    cv_thresh: float = 2.0,
    kurtosis_thresh: float = 1.0,
    report_log_id: str = str(uuid.uuid4())
) -> dict:
    """
    Decide if a univariate series needs a Non-Linear transform, perform it if so,
    and re-assess normality. Returns a assessment dict separate from the transformed Series.

    Args:
        s (pd.Series): Raw data series (numeric).
        statistics (dict): Output from descriptive_statistics(s).
        normality (dict): Output from normality_assessment(s, alpha).
        alpha (float): Significance level for normality re-test.
        skew_thresh (float): Absolute skewness threshold to trigger transform.
        cv_thresh (float): Coefficient of Variation threshold to trigger transform.
        kurtosis_thresh (float): Absolute excess kurtosis threshold to trigger transform.
        report_log_id (str): report log id.

    Returns:
        dict: {
            'assessment': {
                'needs_transform': bool,
                'best_transform': Optional[str],
                'candidates': {
                    transform_name: {
                        'lambda': Optional[float],
                        'normality': dict
                    }, ...
                },
                'normality_raw': dict
            },
            'series': pd.Series
        }
    """
    # Validate input
    validate_numeric_named_series(s)
    logger.info(
        "Starting assess_normality_and_transform",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    # Determine if transform is warranted
    stats_skew = statistics.get('skewness', 0.0)
    stats_cv = statistics.get('cv', 0.0) or 0.0
    stats_kurt = statistics.get('kurtosis', 0.0)

    needs_transform = (
        normality.get('reject_normality', False)
        and (
            abs(stats_skew) > skew_thresh
            or stats_cv > cv_thresh
            or abs(stats_kurt) > kurtosis_thresh
        )
    )

    # Initialize assessment
    assessment = {
        'needs_transform': needs_transform,
        'best_transform': None,
        'candidates': {},
        'normality_raw': normality
    }

    # Start with raw series
    best_series = s.copy()
    best_p = -1.0

    # Perform transform if needed
    if needs_transform:
        candidate_transforms = select_normality_transforms(statistics)

        for name in candidate_transforms:
            candidate = {'lambda': None, 'normality': None}
            try:
                # Apply transform
                match name:
                    case 'box-cox':
                        vals, lam = stats.boxcox(s.values)
                        s_t = pd.Series(vals, index=s.index, name=s.name)
                        candidate['lambda'] = float(lam)
                    case 'yeo-johnson':
                        vals, lam = stats.yeojohnson(s.values)
                        s_t = pd.Series(vals, index=s.index, name=s.name)
                        candidate['lambda'] = float(lam)
                    case 'log':
                        s_t = np.log(s)
                    case 'log1p':
                        s_t = np.log1p(s)
                    case 'sqrt':
                        s_t = np.sqrt(s)
                    case 'reciprocal':
                        s_t = 1.0 / s
                    case 'arcsinh':
                        s_t = np.arcsinh(s)
                    case _:
                        # unsupported transform
                        candidate['error'] = 'not implemented'
                        assessment['candidates'][name] = candidate
                        continue

                # Re-assess normality
                norm_t = normality_assessment(s_t, alpha)
                candidate['normality'] = norm_t

                # Choose p-value for scoring (prefer Shapiro if present)
                p_val = None
                if 'shapiro' in norm_t:
                    p_val = norm_t['shapiro']['p_value']
                elif 'dagostino_pearson' in norm_t:
                    p_val = norm_t['dagostino_pearson']['p_value']

                # Track best
                if p_val is not None and p_val > best_p:
                    best_p = p_val
                    assessment['best_transform'] = name
                    best_series = s_t

            except Exception as e:
                logger.exception(
                    "assess_normality_and_transform failed", 
                    extra={
                        'series_name': s.name,
                        'report_log_id': report_log_id
                    }
                )
                candidate['error'] = str(e)
                candidate['report_log_id'] = report_log_id

            assessment['candidates'][name] = candidate

    logger.info(
        "Completed assess_normality_and_transform",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    # Return final report and chosen series
    return {
        'assessment': assessment,
        'series': best_series
    }
