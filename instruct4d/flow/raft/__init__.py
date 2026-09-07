"""RAFT optical flow, vendored from https://github.com/princeton-vl/RAFT.

Only the inference path is kept: the training dataloaders and augmentation
pipeline from the original repository are not shipped.
"""

from .raft import RAFT

__all__ = ["RAFT"]
