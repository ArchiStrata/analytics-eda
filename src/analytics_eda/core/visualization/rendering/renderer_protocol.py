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


from typing import Any, Callable, Dict, Tuple

DrawFnSeries = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any], Any, Any, Any], Tuple[Any, Any]]
# (desc, inf, chart_md, fig, ax, palette) -> (fig, ax)

class RendererProtocol:
    """
    A renderer encapsulates all figure/axes creation, metadata application,
    subtitles, footers, headroom, legend, formatting, and saving/showing.
    """

    # Core “one-call” API used by BasePlot._pipeline_execute
    def render(
        self,
        *,
        ctx,
        chart_md: Dict[str, Any],
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        draw_fn: DrawFnSeries,
        subtitle_text: str | None,
        footer_text: str | None,
    ) -> str | None:
        """Return saved filename (or None)."""
        raise NotImplementedError

    # Optional helpers (can be used by plots that need to register text for headroom)
    def register_annotations(self, ax, texts): ...
    def neutral_grey(self, variant: str = "medium", alpha: float | None = None): ...
