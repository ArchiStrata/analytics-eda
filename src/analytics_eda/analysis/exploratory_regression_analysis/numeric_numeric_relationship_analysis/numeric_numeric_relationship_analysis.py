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
import logging
from typing import Any, Dict, Optional
import uuid

import pandas as pd
from pandas.api.types import is_numeric_dtype

from analytics_eda.core.utils import build_plot_context
from ....core.reporting import write_json_report

from .relationship_structure_scatter_plot import (
    RelationshipStructureScatterContext, RelationshipStructureScatterPlot
)
from .relationship_structure_scatter_lowess_plot import (
    RelationshipStructureScatterLowessContext, RelationshipStructureScatterLowessPlot
)
from .magnitude_association_scatter_ols_plot import (
    MagnitudeAssociationScatterOLSContext, MagnitudeAssociationScatterOLSPlot
)
from .magnitude_association_residual_plot import (
    MagnitudeAssociationResidualContext, MagnitudeAssociationResidualPlot
)
from .direction_association_scatter_ols_trend_plot import (
    DirectionAssociationScatterOLSTrendContext, DirectionAssociationScatterOLSTrendPlot
)

logger = logging.getLogger(__name__)


def numeric_numeric_relationship_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    report_root: str = "reports/eda/bivariate/numeric_numeric_relationship_analysis",
    report_log_id: str = str(uuid.uuid4()),
    data_source: Optional[str] = None,

    # per‑plot override dicts
    plot_relationship_structure_scatter_overrides: Optional[Dict[str, Any]] = None,
    plot_relationship_structure_scatter_lowess_overrides: Optional[Dict[str, Any]] = None,
    plot_magnitude_scatter_ols_overrides: Optional[Dict[str, Any]] = None,
    plot_magnitude_residual_overrides: Optional[Dict[str, Any]] = None,
    plot_direction_scatter_ols_trend_overrides: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Run numeric↔numeric relationship analysis (structure, magnitude, direction) and
    write a JSON report bundling plot payloads.

    Returns:
        Dict with 'report_file_path' to the saved JSON report.
    """
    logger.info(
        "Starting numeric_numeric_relationship_analysis",
        extra={"x_col": x_col, "y_col": y_col, "report_root": report_root, "report_log_id": report_log_id},
    )

    # ---- validate inputs ----
    if x_col not in df.columns:
        raise KeyError(f"X column '{x_col}' not found.")
    if y_col not in df.columns:
        raise KeyError(f"Y column '{y_col}' not found.")
    if not is_numeric_dtype(df[x_col]):
        raise TypeError(f"Column '{x_col}' must be numeric.")
    if not is_numeric_dtype(df[y_col]):
        raise TypeError(f"Column '{y_col}' must be numeric.")

    # ---- paths & shared context ----
    report_path = Path(report_root) / f"numeric_{x_col}_numeric_{y_col}_relationship_analysis"
    report_path.mkdir(parents=True, exist_ok=True)

    df_copy = df.copy()
    common_base = {"save_path": report_path, "data_source": data_source}

    # =========================
    # Relationship Structure
    # =========================
    relationship_structure: Dict[str, Any] = {}

    rs_scatter_ctx = build_plot_context(
        RelationshipStructureScatterContext,
        base=common_base,
        overrides=plot_relationship_structure_scatter_overrides,
    )
    rs_scatter_plot = RelationshipStructureScatterPlot(rs_scatter_ctx)
    relationship_structure["scatter"] = rs_scatter_plot.run(
        df_copy, cols=[x_col, y_col], role_map={"x": x_col, "y": y_col}
    )

    rs_lowess_ctx = build_plot_context(
        RelationshipStructureScatterLowessContext,
        base=common_base,
        overrides=plot_relationship_structure_scatter_lowess_overrides,
    )
    rs_lowess_plot = RelationshipStructureScatterLowessPlot(rs_lowess_ctx)
    relationship_structure["scatter_lowess"] = rs_lowess_plot.run(
        df_copy, cols=[x_col, y_col], role_map={"x": x_col, "y": y_col}
    )

    # =========================
    # Magnitude of Association
    # =========================
    magnitude_of_association: Dict[str, Any] = {}

    mag_scatter_ols_ctx = build_plot_context(
        MagnitudeAssociationScatterOLSContext,
        base=common_base,
        overrides=plot_magnitude_scatter_ols_overrides,
    )
    mag_scatter_ols_plot = MagnitudeAssociationScatterOLSPlot(mag_scatter_ols_ctx)
    magnitude_of_association["scatter_ols"] = mag_scatter_ols_plot.run(
        df_copy, cols=[x_col, y_col], role_map={"x": x_col, "y": y_col}
    )

    mag_resid_ctx = build_plot_context(
        MagnitudeAssociationResidualContext,
        base=common_base,
        overrides=plot_magnitude_residual_overrides,
    )
    mag_resid_plot = MagnitudeAssociationResidualPlot(mag_resid_ctx)
    magnitude_of_association["residuals"] = mag_resid_plot.run(
        df_copy, cols=[x_col, y_col], role_map={"x": x_col, "y": y_col}
    )

    # =========================
    # Direction of Association
    # =========================
    direction_of_association: Dict[str, Any] = {}

    dir_trend_ctx = build_plot_context(
        DirectionAssociationScatterOLSTrendContext,
        base=common_base,
        overrides=plot_direction_scatter_ols_trend_overrides,
    )
    dir_trend_plot = DirectionAssociationScatterOLSTrendPlot(dir_trend_ctx)
    direction_of_association["ols_trend"] = dir_trend_plot.run(
        df_copy, cols=[x_col, y_col], role_map={"x": x_col, "y": y_col}
    )

    # ---- bundle report ----
    eda_report = {
        "relationship_structure": relationship_structure,
        "magnitude_of_association": magnitude_of_association,
        "direction_of_association": direction_of_association,
    }

    full_report = {
        "metadata": {
            "version": "0.1.0",
            "report_name": "numeric_numeric_relationship_analysis",
            "parameters": {"x_col": x_col, "y_col": y_col},
        },
        "data": eda_report,
    }

    report_file_path = report_path / f"numeric_{x_col}_numeric_{y_col}_relationship_analysis_report.json"
    write_json_report(full_report, report_file_path)

    logger.info(
        "Completed numeric_numeric_relationship_analysis",
        extra={"x_col": x_col, "y_col": y_col, "report_log_id": report_log_id, "report_file_path": str(report_file_path)},
    )

    return {"report_file_path": report_file_path}
