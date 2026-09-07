"""Dataset loaders for the multi-view (DyNeRF) setting.

Only the dynamic, multi-view loader is shipped; :data:`DATASETS` is the registry
that ``--dataset_name`` selects from.
"""

from .n3dv import N3DVDynamicDataset

#: Maps the ``--dataset_name`` flag to a loader class.
DATASETS = {
    "n3dv_dynamic": N3DVDynamicDataset,
}

__all__ = ["DATASETS", "N3DVDynamicDataset"]
