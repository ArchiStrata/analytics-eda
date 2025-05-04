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

def select_normality_transforms(statistics: dict) -> list[str]:
    """
    From descriptive stats, choose which transforms are valid to try.

    Always include:
      - 'yeo-johnson'  (handles all R)
      - 'arcsinh'      (also handles all R)

    Conditionally include:
      - 'box-cox'      (if min > 0)
      - 'log'          (if min > 0)
      - 'reciprocal'   (if min > 0)
      - 'sqrt'         (if min >= 0)
      - 'log1p'        (if min >= 0)

    Args:
        statistics (dict): Output of descriptive_statistics(), must contain 'min'.

    Returns:
        List of transform names to attempt.
    """
    candidates = ['yeo-johnson', 'arcsinh']
    min_val = statistics.get('min', None)

    if min_val is not None:
        if min_val > 0:
            candidates += ['box-cox', 'log', 'reciprocal']
        if min_val >= 0:
            candidates += ['sqrt', 'log1p']

    return candidates
