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
sanitize_json.py

Provides a utility function to recursively sanitize Python data 
structures for JSON serialization.

This module ensures compatibility with JSON encoding by:
- Replacing NaN float values with `None`
- Converting NumPy scalar types (integers, floats, booleans) 
  to their native Python equivalents

Useful for preparing data that includes NumPy types or invalid 
JSON values (e.g., NaN) before serialization.

Functions:
- sanitize_json(o): Recursively converts data into a JSON-serializable form.
"""
import math
import numpy as np

def sanitize_json(o):
    """
    Recursively walk through o and convert:
      - NaN floats to None
      - NumPy scalars to native Python scalars
    """
    # Dictionaries
    if isinstance(o, dict):
        return {k: sanitize_json(v) for k, v in o.items()}

    # Lists
    elif isinstance(o, list):
        return [sanitize_json(v) for v in o]

    # Handle NaN floats
    elif isinstance(o, float) and math.isnan(o):
        return None

    # NumPy integer to int
    elif isinstance(o, (np.integer,)):
        return int(o)

    # NumPy floating to float
    elif isinstance(o, (np.floating,)):
        return float(o)

    # NumPy boolean to bool
    elif isinstance(o, (np.bool_,)):
        return bool(o)

    # Everything else (including built-in bool, int, float, str)
    else:
        return o
