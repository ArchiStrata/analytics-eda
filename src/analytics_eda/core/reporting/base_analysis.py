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
"""Base class for analyses."""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from analytics_eda.core.reporting.analysis_context import AnalysisContext
from analytics_eda.core.reporting.write_json_report import write_json_report


class BaseAnalysis(ABC):
    """
    Base class for analyses.

      - Use an AnalysisContext for IO and control behavior.
      - Expose a semantic version.
      - Produce a dict of named plots/analyses.
      - Provide a run() method that orchestrates everything.
    """

    def __init__(self, context: AnalysisContext) -> None:
        self.context = context
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self.logger = logging.getLogger(logger_name)

    # --- Semantic version ----------------------------------------------

    @property
    @abstractmethod
    def semantic_version(self) -> str:
        """
        Semantic version of this analysis' logic / report schema.

        Child classes should override this and bump it when:
          - The analysis logic changes; or
          - The report structure (keys, shapes) changes.
        """

    # --- Core analysis -------------------------------------------------

    @abstractmethod
    def validate(self, data_input: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Validate and optionally transform the incoming data for this analysis.

        Parameters
        ----------
        data_input : pd.Series | pd.DataFrame
            Raw data supplied to `run`.

        Returns
        -------
        pd.Series | pd.DataFrame
            The validated (and possibly transformed) data used by downstream artifacts.
        """

    @abstractmethod
    def build_artifacts(self, data_input: pd.Series | pd.DataFrame) -> dict[str, Any]:
        """
        Build the core analysis artifacts.

        Returns
        -------
        Dict[str, Any]
            Mapping from artifact names to objects (plots, tables, metrics, etc.).
        """

    # --- Orchestration -------------------------------------------------

    def run(self, data_input: pd.Series | pd.DataFrame, report_log_id: str | None = None) -> dict[str, Any]:
        """
        Run the analysis.

        Analysis steps:

          1) Validate / prepare the data.
          2) Build the artifacts.
          3) Wrap them with metadata (semantic version).
          4) Save JSON report if configured.
          5) Return either the full report or an empty dict,
             depending on the AnalysisContext.

        Parameters
        ----------
        data_input : pd.Series | pd.DataFrame
            The source data for all artifacts managed by this analysis.
        report_log_id : str | None
            Optional identifier used for logging/report traces.

        Returns
        -------
        Dict[str, Any]
            The full report if context.return_full_report is True,
            otherwise an empty dict.
        """
        if report_log_id is None:
            report_log_id = str(uuid.uuid4())

        self.logger.info(
            "Starting %s",
            self.context.report_name,
            extra={
                "report_log_id": report_log_id,
                "report_name": self.context.report_name,
            },
        )

        validated_data = self.validate(data_input)

        metadata = {
            "version": self.semantic_version,
            "report_name": self.context.report_name,
            "report_relative_path": self.context.report_relative_path,
            "report_file_name": self.context.report_file_name,
            "data_source": self.context.data_source,
            "filter_desc": self.context.filter_desc,
        }

        # Prepare report directory
        report_dir = self.report_dir()
        report_dir.mkdir(parents=True, exist_ok=True)

        report_data = self.build_artifacts(validated_data)

        report: dict[str, Any] = {
            "metadata": metadata,
            "data": report_data,
        }

        report_path = self.report_path()
        if self.context.save_json_report:
            write_json_report(report, report_path)

        self.logger.info(
            "Completed %s",
            self.context.report_name,
            extra={
                "report_log_id": report_log_id,
                "report_name": self.context.report_name,
                "report_path": str(report_path),
            },
        )

        return report if self.context.return_full_report else {"report_path": report_path}

    def report_dir(self) -> Path:
        """Directory where the report will be written."""
        return (self.context.base_dir / self.context.report_relative_path).resolve()

    def report_path(self) -> Path:
        """Full path to the JSON report file."""
        return self.report_dir() / self.context.report_file_name

    def base_kwargs(self) -> dict[str, Any]:
        """Get the base keyword arguments for plot contexts."""
        return {
            "base_dir": self.report_dir(),
            "data_source": self.context.data_source,
            "filter_desc": self.context.filter_desc,
            "auto_file_name": True,
        }
