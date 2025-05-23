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
from pathlib import Path
import pandas as pd
from scipy.stats import zscore

from .validate_numeric_named_series import validate_numeric_named_series

logger = logging.getLogger(__name__)

def numeric_outlier_analysis(
    s: pd.Series,
    report_dir: Path,
    iqr_multiplier: float = 1.5,
    z_thresh: float = 3.0,
    report_log_id: str = str(uuid.uuid4())
) -> dict:
    """
    Detect outliers using IQR, standard Z-score, and robust modified Z-score (MAD-based).
    NaNs are dropped before analysis. Returns {} if no data after dropping NaNs.
    Saves flagged outlier rows to CSVs: <column>_iqr_outliers.csv, <column>_zscore_outliers.csv,
    and <column>_robust_zscore_outliers.csv for manual review.

    Args:
        s (pd.Series): Series containing the data.
        report_dir (Path): Directory for saving outlier CSVs.
        iqr_multiplier (float, optional): IQR fence multiplier (default=1.5).
        z_thresh (float, optional): Threshold for both Z-score and modified Z-score (default=3.0).
        report_log_id (str): report log id.

    Raises:
        KeyError: If `column` is not in `df`.
        TypeError: If `column` exists but isn’t numeric.
    """
    # 1. Validation & prepare series
    validate_numeric_named_series(s)

    logger.info(
        "Starting numeric_outlier_analysis",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    total = len(s)
    if total == 0:
        return {}

    # 2. IQR method
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    mask_iqr = (s < lower) | (s > upper)
    out_iqr = s.loc[mask_iqr.index[mask_iqr]]
    pct_iqr = len(out_iqr) / total
    file_iqr = report_dir / f"{s.name}_iqr_outliers.csv"
    out_iqr.to_csv(file_iqr, index=False)

    # 3. Standard Z-score method
    zs = pd.Series(zscore(s), index=s.index)
    mask_z = zs.abs() > z_thresh
    out_z = s.loc[mask_z.index[mask_z]]
    pct_z = len(out_z) / total
    file_z = report_dir / f"{s.name}_zscore_outliers.csv"
    out_z.to_csv(file_z, index=False)

    # 4. Robust modified Z-score (MAD-based)
    med = s.median()
    mad_val = float((s - med).abs().median())
    if mad_val == 0:
        mask_robust = pd.Series(False, index=s.index)
    else:
        modified_z = 0.6745 * (s - med) / mad_val
        mask_robust = modified_z.abs() > z_thresh
    out_robust = s.loc[mask_robust.index[mask_robust]]
    pct_robust = len(out_robust) / total
    file_robust = report_dir / f"{s.name}_robust_zscore_outliers.csv"
    out_robust.to_csv(file_robust, index=False)

    # 5. Build summary
    summary = {
        "iqr": {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "count": len(out_iqr),
            "pct": pct_iqr,
            "outliers_file": str(file_iqr)
        },
        "zscore": {
            "threshold": z_thresh,
            "count": len(out_z),
            "pct": pct_z,
            "outliers_file": str(file_z)
        },
        "robust_zscore": {
            "threshold": z_thresh,
            "mad": mad_val,
            "count": len(out_robust),
            "pct": pct_robust,
            "outliers_file": str(file_robust)
        }
    }

    logger.info(
        "Completed numeric_outlier_analysis",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    return summary
