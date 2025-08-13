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
import pandas as pd

class NamedSeriesMixin:
    def validate(self, series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series.")
        if (series.name is None or str(series.name).strip() == ""):
            raise ValueError("Series must have a non-empty 'name' attribute.")
        return series
