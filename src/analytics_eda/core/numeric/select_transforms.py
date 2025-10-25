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

from collections.abc import Callable


def select_transforms(
    descriptive_stats: dict,
    normality_tests: dict | None = None
) -> list[str]:
    """
    From descriptive stats and formal normality test results, choose which
    power‐type transforms are valid to try.

    Parameters
    ----------
    descriptive_stats : dict
        Output of descriptive statistics, must contain at least:
          - 'min'       (float)
          - 'skewness'  (float)
          - 'kurtosis'  (float)
    normality_tests : dict, optional
        Results of your normality tests, e.g.:
            {'shapiro': {...}, 'reject_normality': True, ...}

    Returns
    -------
    List[str]
        Candidate transforms, in order of evaluation.
    """
    # pull out the few values we need
    min_val   = descriptive_stats.get('min')
    skew      = descriptive_stats.get('skewness', 0.0)
    kurtosis  = descriptive_stats.get('kurtosis', 0.0)
    tests     = normality_tests or {}
    reject_n  = bool(tests.get('reject_normality', False))

    # each transform defines its own inclusion rule
    transform_rules: dict[str, Callable[[], bool]] = {
        'yeo-johnson': lambda: True,
        'arcsinh':     lambda: True,
        'box-cox':     lambda: (min_val is not None and min_val > 0),
        'log':         lambda: (min_val is not None and (min_val > 0 or skew > 1)),
        'log1p':       lambda: (min_val is not None and min_val >= 0),
        'sqrt':        lambda: (min_val is not None and min_val >= 0),
        'reciprocal':  lambda: (min_val is not None and (min_val > 0 or kurtosis > 1)),
    }

    candidates: list[str] = []
    for name, rule in transform_rules.items():
        if rule():
            candidates.append(name)

    # if the data truly fails normality, make sure we include
    # our most-powerful transforms
    if reject_n:
        for t in ('box-cox', 'yeo-johnson'):
            if t not in candidates:
                candidates.append(t)

    return candidates
