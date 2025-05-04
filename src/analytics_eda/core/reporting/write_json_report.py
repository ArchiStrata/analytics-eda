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
"""
json_report_writer.py

This module provides functionality to sanitize and write JSON reports
to disk in a structured and reliable manner. It ensures that the target
directory exists, the report content is cleaned using a local 
`sanitize_json` utility, and the output is written with UTF-8 encoding 
and consistent formatting.

Functions:
- write_json_report(report, report_path, encoding='utf-8'):
    Sanitizes a report dictionary and writes it as a formatted JSON 
    file to the specified path.

Dependencies:
- sanitize_json: A local module that provides a sanitize_json(report) 
  function to clean or transform data before serialization.
"""
import os
import json
from .sanitize_json import sanitize_json

def write_json_report(report, report_path: str, encoding: str="utf-8"):
    """
    Sanitizes a JSON-serializable report and writes it to the specified file path.

    Args:
        report (dict): The report data to be sanitized and written.
        report_path (str): Destination file path for the JSON report.
        encoding (str): Encoding used when writing the file. Default is "utf-8".
    
    Returns: clean_report
    """
    clean_report = sanitize_json(report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding=encoding) as fp:
        json.dump(clean_report, fp, indent=4, default=str)

    return clean_report
