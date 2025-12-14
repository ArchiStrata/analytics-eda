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
"""Rendering protocol for Analytics-EDA visualization backends.

Defines the callable signature used by plot drawers and the `RendererProtocol`
interface that coordinates figure/axes creation, metadata, and output.
"""

from collections.abc import Callable
from typing import Any

DrawFnSeries = Callable[[dict[str, Any], dict[str, Any], dict[str, Any], Any, Any, Any], tuple[Any, Any]]
# (desc, inf, chart_md, fig, ax, palette) -> (fig, ax)


class RendererProtocol:
    """Abstract interface for plot rendering.

    A renderer encapsulates figure/axes creation, metadata application,
    subtitles, footers, headroom, legend, formatting, and saving/showing.
    """

    # Core “one-call” API used by BasePlot._pipeline_execute
    def render(
        self,
        *,
        ctx,
        chart_md: dict[str, Any],
        desc: dict[str, Any],
        inf: dict[str, Any],
        draw_fn: DrawFnSeries,
        subtitle_text: str | None,
        footer_text: str | None,
    ) -> str | None:
        """Render a chart and optionally save it to disk.

        Args:
            ctx: Rendering context (renderer-specific).
            chart_md: Chart metadata (titles, labels, sizing, etc.).
            desc: Data description (series roles, kinds, etc.).
            inf: Inference/configuration details used during rendering.
            draw_fn: Callback that draws onto `(fig, ax)` and returns them.
            subtitle_text: Optional subtitle text to place on the figure.
            footer_text: Optional footer text to place on the figure.

        Returns
        -------
            The saved filename if the figure was persisted, otherwise `None`.

        Raises
        ------
            NotImplementedError: If the concrete renderer does not implement it.
        """
        raise NotImplementedError

    # Optional helpers (can be used by plots that need to register text for headroom)
    def register_annotations(self, ax, texts):
        """Register text elements drawn on `ax` to inform layout/headroom.

        Args:
            ax: The Matplotlib axes (or compatible object) where text is drawn.
            texts: Iterable of strings or text artists to be considered in layout.
        """
        ...

    def neutral_grey(self, variant: str = "medium", alpha: float | None = None):
        """Return a neutral grey color suitable for UI chrome and guides.

        Args:
            variant: One of {"light", "medium", "dark"} to pick a shade.
            alpha: Optional opacity to apply to the color.

        Returns
        -------
            A renderer-specific color value (e.g., RGBA tuple or hex string).
        """
        ...
