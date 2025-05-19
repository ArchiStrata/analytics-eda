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
import numpy as np
from scipy.stats import gaussian_kde

def compute_overlap_metrics(grouped_data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """
    Compute distribution overlap metrics for each pair of groups.

    Args:
        grouped_data: Mapping from group label to 1D array of numeric values.

    Returns:
        A dict where each key is "label1 vs label2" and each value contains:
            - 'overlap_coeff': Overlap Coefficient (float)
            - 'bhattacharyya_dist': Bhattacharyya Distance (float)
    """
    labels = list(grouped_data)
    overlaps: dict[str, dict[str, float]] = {}

    for i, label_i in enumerate(labels[:-1]):
        xi = grouped_data[label_i]
        kde_i = gaussian_kde(xi)

        for label_j in labels[i+1:]:
            xj = grouped_data[label_j]
            kde_j = gaussian_kde(xj)

            # Define common evaluation grid
            grid_min = min(xi.min(), xj.min())
            grid_max = max(xi.max(), xj.max())
            grid = np.linspace(grid_min, grid_max, 200)

            # Estimate densities
            pi = kde_i(grid)
            pj = kde_j(grid)

            # Overlap Coefficient: area under min(pi, pj)
            overlap_coeff = np.trapezoid(np.minimum(pi, pj), grid)

            # Bhattacharyya Distance: -ln ∫ sqrt(pi * pj)
            bc = np.trapezoid(np.sqrt(pi * pj), grid)
            bhattacharyya_dist = -np.log(bc) if bc > 0 else np.inf

            overlaps[f"{label_i} vs {label_j}"] = {
                'overlap_coeff': float(overlap_coeff),
                'bhattacharyya_dist': float(bhattacharyya_dist)
            }

    return overlaps
