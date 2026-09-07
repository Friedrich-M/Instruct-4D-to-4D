"""The streaming TensoRF 4D scene representation.

:data:`FIELDS` is the registry that ``--model_name`` selects from.
"""

from .tensor_base import AlphaGridMask, StreamTensorBase, raw2alpha
from .tensorf import StreamTensorCP, StreamTensorVMSplit

#: Maps the ``--model_name`` flag to a field class.
FIELDS = {
    "StreamTensorVMSplit": StreamTensorVMSplit,
    "StreamTensorCP": StreamTensorCP,
}

__all__ = [
    "FIELDS",
    "StreamTensorBase",
    "StreamTensorVMSplit",
    "StreamTensorCP",
    "AlphaGridMask",
    "raw2alpha",
]
