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
"""Analysis context for running analysis."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnalysisContext:
    """
    Configuration and state for running an analysis.

    Controls:
      - Whether to save a JSON report.
      - Whether to return the full report.
      - Where (relative path) and under what name to save the report.
    """

    # Report name
    report_name: str

    # Subdirectory under base_dir where this analysis report should go
    report_relative_path: str = ""

    # File name of the JSON report
    report_file_name: str = "analysis_report.json"

    # Root directory for all analysis outputs
    base_dir: Path | None = None

    # Behavior flags
    save_json_report: bool = False
    return_full_report: bool = True

    # Data Source
    data_source: str | None = None

    # Filter Description
    filter_desc: str | None = None
