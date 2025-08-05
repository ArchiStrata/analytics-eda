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

def build_chart_title(
    *,
    name: str = None,
    series=None,
    filter_desc: str = None,
    transform_desc: str = None,
    title_template: str = "{name}{modifiers}"
) -> str:
    """
    Build a chart title from a template, series name (or fallback), and optional modifiers.

    Parameters
    ----------
    name : str, optional
        Explicit series name to show; falls back to series.name or "Value".
    series : object with .name attribute, optional
        Used only as a fallback if name is None.
    filter_desc : str, optional
        e.g. "filtered by New York"
    transform_desc : str, optional
        e.g. "log-transformed"
    title_template : str
        A Python format-string with placeholders:
          - {name}: the label
          - {modifiers}: space-prefixed "(…)" text or empty

    Returns
    -------
    str
        The rendered title.
    """
    # determine the base label
    label = name or getattr(series, "name", None) or "Value"

    # collect any modifiers
    parts = [d for d in (filter_desc, transform_desc) if d]
    modifiers = f" ({', '.join(parts)})" if parts else ""

    # render the template
    return title_template.format(name=label, modifiers=modifiers)
