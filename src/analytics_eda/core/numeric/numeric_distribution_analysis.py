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
import pandas as pd

from .descriptive_statistics import descriptive_statistics
from .normality_assessment import normality_assessment
from .distribution_visualizations import distribution_visualizations
from .distribution_fit_assessment import distribution_fit_assessment
from .assess_normality_and_transform import assess_normality_and_transform
from .validate_numeric_named_series import validate_numeric_named_series

def numeric_distribution_analysis(
    s: pd.Series,
    report_dir: Path,
    alpha: float = 0.05
) -> dict:
    """
    Compute descriptive statistics, assess normality, visualize distribution,
    and determine if transformation is needed before fitting alternatives. Drops NAs.

    Args:
        s (pd.Series): Series containing the data.
        report_dir (Path): Directory for saving plots.
        alpha (float): Significance level for normality tests.

    Returns:
        dict: {
            'statistics': dict of descriptive stats,
            'normality_report': {
                'assessment': dict of test results,
                'visualizations': dict of filepaths
            },
            'transform_report: {
                'assessment': dict of test results,
                'visualizations': dict of filepaths
            }
            'alternatives_report': dict or {}
        }
    Raises:
        TypeError: if series is not numeric.
        ValueError: from cleaning.
    """

    print(f"Analyzing Distribution in Series [{s.name}]")

    # 1. Validate
    validate_numeric_named_series(s)

    # 2. Descriptive statistics
    statistics = descriptive_statistics(s, True)

    # 3. TODO: Binning analysis
    # binning_report = report_binning_rules(s, statistics['is_discrete'])

    # 4. TODO: Frequency analysis using binning

    # 5. Normality assessment and raw visualizations
    normality = normality_assessment(s, alpha)
    raw_visualizations = distribution_visualizations(s, report_dir, transform='raw')

    # 6. Non-linear transformations assessment
    transform_result = assess_normality_and_transform(s, statistics, normality, alpha)
    best_series = transform_result['series']

    # if transformed generate distribution_visualizations
    transform_visualizations = None
    best_transform = transform_result['assessment'].get('best_transform')
    if best_transform is not None and best_transform != "":
        transform_visualizations = distribution_visualizations(best_series, report_dir, transform=transform_result['assessment']['best_transform'])

    # 7. TODO: Feature Scaling - Normalization, Standardization: Choose appropriate scaling (min–max normalization, Z-score standardization, robust scaling) for downstream algorithms.

    # 8. Fitting theoretical distributions
    alternatives_assessment = {}
    if normality.get('reject_normality', False):
        alternatives_assessment = distribution_fit_assessment(best_series, alpha)

    return {
        'report': {
            'statistics': statistics,
            'normality_report': {
                'assessment': normality,
                'visualizations': raw_visualizations
            },
            'transformation_report': {
                'assessment': transform_result['assessment'],
                'visualizations': transform_visualizations,
            },
            'distribution_fit_report': {
                'assessment': alternatives_assessment
            }
        },
        'series': best_series
    }
