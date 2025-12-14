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
"""Helpers to build plot contexts.

Provides `build_plot_context`, which constructs or updates a dataclass-based
plot context from a base instance or dict, applying validated overrides.
"""

from dataclasses import fields, is_dataclass, replace
from typing import Any, TypeVar

T = TypeVar("T")


def build_plot_context(
    ctx_cls: type[T],
    base: T | dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> T:
    """Create or copy a context with validated overrides."""
    overrides = overrides or {}

    # Validate keys
    allowed = {f.name for f in fields(ctx_cls)}
    unknown = set(overrides) - allowed
    if unknown:
        raise KeyError(f"Unknown context fields for {ctx_cls.__name__}: {sorted(unknown)}")

    if base is None:
        return ctx_cls(**overrides)

    if is_dataclass(base) and type(base) is ctx_cls:
        return replace(base, **overrides)

    if isinstance(base, dict):
        data = {k: v for k, v in base.items() if k in allowed}
        data.update(overrides)
        return ctx_cls(**data)

    raise TypeError("base must be None, a dict, or an instance of the target ctx dataclass.")
