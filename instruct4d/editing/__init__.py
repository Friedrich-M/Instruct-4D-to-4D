"""The two consistency mechanisms of the paper.

:class:`KeyFrameEditor`
    Spatial consistency at one timestamp: edit a batch of views together, then
    reconcile and propagate them with depth-based warping (Sections 3.2, 3.4).
:class:`TemporalPropagator`
    Temporal consistency across timestamps: a flow-guided sliding window that
    carries the key frame's edit along the sequence (Section 3.3).
"""

from .buffer import FrameBuffer
from .debug import DebugImageWriter
from .key_view import KeyFrameEditor, compute_view_points
from .propagation import TemporalPropagator
from .warping import apply_warp, project_points_to_view, unproject_depth_ndc

__all__ = [
    "FrameBuffer",
    "DebugImageWriter",
    "KeyFrameEditor",
    "compute_view_points",
    "TemporalPropagator",
    "apply_warp",
    "project_points_to_view",
    "unproject_depth_ndc",
]
