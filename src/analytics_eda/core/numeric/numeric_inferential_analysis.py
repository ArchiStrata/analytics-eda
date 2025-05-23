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
import numpy as np

from .validate_numeric_named_series import validate_numeric_named_series

logger = logging.getLogger(__name__)

def numeric_inferential_analysis(
    s: pd.Series,
    alpha: float = 0.05,
    popmean: float | None = None,
    popmedian: float | None = None,
    popvariance: float | None = None,
    bootstrap_samples: int = 1_000,
    report_log_id: str = str(uuid.uuid4())
) -> dict:
    """
    Perform a suite of inferential analyses on a numeric series in a DataFrame.

    Drops NaNs, validates the series, then always computes:
      - 95% CI for the mean (t-based)
      - Bootstrap CI for the median
      - Goodness-of-fit (Kolmogorov–Smirnov vs. Normal)
      - Bootstrap-based CIs for both mean and median

    Conditionally computes:
      - One-sample t-test vs. `popmean` (if provided)
      - One-sample Z-test vs. `popmean` using known `popvariance` (if both provided)
      - Wilcoxon signed-rank test vs. `popmedian` (if provided)
      - Sign test vs. `popmedian` (if provided)
      - Effect-size (Cohen's d) vs. `popmean` (if provided)
      - Chi-squared variance test vs. `popvariance` (if provided)

    Args:
        s (pd.Series): Series containing the data.
        alpha (float): Significance/confidence level (e.g. 0.05 for 95%).
        popmean (float|None): Hypothesized population mean for tests.
        popmedian (float|None): Hypothesized population median for tests.
        popvariance (float|None): Hypothesized population variance (σ²) for tests.
        bootstrap_samples (int): Number of resamples for bootstrap CIs.
        report_log_id (str): report log id.

    Returns:
        dict: {
            'ci': {
               'mean_t': [lower, upper],
               'median_boot': [lower, upper]
            },
            'gof': {
               'ks': {'statistic', 'p_value', 'reject'}
            },
            't_test'?: {...},         # if popmean provided
            'z_test'?: {...},         # if popmean & popvariance provided
            'wilcoxon_signed_rank'?: {...}, # if popmedian provided
            'sign_test'?: {...},      # if popmedian provided
            'effect_size'?: {...},    # if popmean provided
            'variance'?: {...},       # if popvariance provided
            'bootstrap': {
               'mean': [lower, upper],
               'median': [lower, upper]
            }
        }
        Returns {} if no non-null data.
    """
    validate_numeric_named_series(s)

    logger.info(
        "Starting numeric_inferential_analysis",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    # 1. Clean & validate
    s_clean = s.dropna()
    n = len(s_clean)
    if n == 0:
        return {}

    result = {}

    # 2. Confidence interval for mean (t-distribution)
    mean = s_clean.mean()
    sem = s_clean.std(ddof=1) / np.sqrt(n)
    ci_mean = stats.t.interval(1 - alpha, df=n-1, loc=mean, scale=sem)
    result.setdefault('ci', {})['mean_t'] = [float(ci_mean[0]), float(ci_mean[1])]

    # 3. Bootstrap CI for median
    medians = []
    for _ in range(bootstrap_samples):
        sample = np.random.choice(s_clean, size=n, replace=True)
        medians.append(np.median(sample))
    lower_med, upper_med = np.percentile(medians, [100*alpha/2, 100*(1-alpha/2)])
    result['ci']['median_boot'] = [float(lower_med), float(upper_med)]

    # 4. Goodness-of-fit: Kolmogorov-Smirnov vs. Normal
    z = (s_clean - mean) / s_clean.std(ddof=1)
    ks_stat, ks_p = stats.kstest(z, 'norm')
    result['gof'] = {
        'ks': {
            'statistic': float(ks_stat),
            'p_value': float(ks_p),
            'reject': bool(ks_p < alpha)
        }
    }

    # 5. Population Variance Tests
    if popvariance is not None:
        # chi-squared
        sample_var = s_clean.var(ddof=1)
        chi2_stat = (n - 1) * sample_var / popvariance
        p_var = stats.chi2.sf(chi2_stat, df=n-1)
        result['variance'] = {
            'chi2': {
                'statistic': float(chi2_stat),
                'p_value': float(p_var),
                'reject': bool(p_var < alpha)
            }
        }

    # 6. Population Mean Tests
    if popmean is not None:
        # One-Sample Cohen's d
        sd = s_clean.std(ddof=1)
        cohens_d = float((mean - popmean) / sd) if sd != 0 else None
        result['effect_size'] = {'cohens_d': cohens_d}

        # One-Sample t-Test
        stat, p = stats.ttest_1samp(s_clean, popmean)
        result['t_test'] = {
            'statistic': float(stat),
            'p_value': float(p),
            'reject': bool(p < alpha)
        }
    
        # One-sample Z-test (requires known σ²)
        if popvariance is not None:
            sigma = np.sqrt(popvariance)
            z_stat = (mean - popmean) / (sigma / np.sqrt(n))
            z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            result['z_test'] = {'statistic': float(z_stat), 'p_value': float(z_p), 'reject': z_p < alpha}

    # 7. Population Median Tests
    if popmedian is not None:
        # Wilcoxon Signed-Rank Test
        diff = s_clean - popmedian
        # only consider non-zero differences
        stat_wr, p_wr = stats.wilcoxon(diff)
        result['wilcoxon_signed_rank'] = {
            'statistic': float(stat_wr), 'p_value': float(p_wr), 'reject': p_wr < alpha
        }

        # Sign test (binomial on signs)
        nonzero = diff[diff != 0]
        n_sign = len(nonzero)
        if n_sign > 0:
            pos = int((nonzero > 0).sum())
            sign_res = stats.binomtest(pos, n_sign, p=0.5)
            result['sign_test'] = {
                'num_positive': pos,
                'num_negative': n_sign - pos,
                'n': n_sign,
                'p_value': float(sign_res.pvalue),
                'reject': sign_res.pvalue < alpha
            }

    # 8. Bootstrap-based CI for the mean (optional)
    means = []
    for _ in range(bootstrap_samples):
        sample = np.random.choice(s_clean, size=n, replace=True)
        means.append(np.mean(sample))
    lower_mean, upper_mean = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    result['bootstrap'] = {
        'mean': [float(lower_mean), float(upper_mean)],
        'median': [float(lower_med), float(upper_med)]
    }

    logger.info(
        "Completed numeric_inferential_analysis",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    return result
