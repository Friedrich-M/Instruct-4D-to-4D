"""Anchor-aware InstructPix2Pix (Section 3.2).

:class:`SequenceInstructPix2Pix` is the entry point used by both editing
pipelines.  :class:`InstructPix2Pix` is the unmodified per-image editor, kept so
the Instruct-NeRF2NeRF baseline can be reproduced.
"""

from .instruct_pix2pix import InstructPix2Pix
from .sequence import LATENT_INIT_MODES, SequenceInstructPix2Pix

__all__ = ["InstructPix2Pix", "SequenceInstructPix2Pix", "LATENT_INIT_MODES"]
