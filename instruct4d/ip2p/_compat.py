"""Bridges the diffusers API changes the vendored UNet sits across.

The pseudo-3D UNet is a fork of diffusers' 2D one, so it reaches into internals
that have moved between releases.  Rather than pin an old diffusers, resolve the
handful of moved names here and call the version-sensitive loader functions
through wrappers that pass only the arguments the installed version accepts.

Verified against diffusers 0.19 through 0.40.
"""

import inspect
from typing import Any, Callable, Dict

# `maybe_allow_in_graph` and `randn_tensor` moved to diffusers.utils.torch_utils
# in 0.24.
try:
    from diffusers.utils import maybe_allow_in_graph, randn_tensor
except ImportError:  # diffusers >= 0.24
    from diffusers.utils.torch_utils import maybe_allow_in_graph, randn_tensor

# HF_HUB_OFFLINE moved to diffusers.utils.hub_utils.
try:
    from diffusers.utils import HF_HUB_OFFLINE
except ImportError:  # diffusers >= 0.32
    from diffusers.utils.hub_utils import HF_HUB_OFFLINE

# DIFFUSERS_CACHE was removed once diffusers deferred to huggingface_hub's own
# cache. `None` asks the hub for its default, which is what it now resolves to.
try:
    from diffusers.utils import DIFFUSERS_CACHE as DEFAULT_CACHE
except ImportError:  # diffusers >= 0.30
    DEFAULT_CACHE = None

__all__ = [
    "maybe_allow_in_graph",
    "randn_tensor",
    "HF_HUB_OFFLINE",
    "DEFAULT_CACHE",
    "call_supported",
]


def call_supported(func: Callable, /, **kwargs: Any) -> Any:
    """Call ``func`` with only the keyword arguments it actually accepts.

    Lets one call site serve several diffusers versions: ``use_auth_token``
    became ``token``, ``resume_download`` was dropped, and ``load_state_dict``
    lost its ``variant`` argument.  Passing an unsupported name would raise, so
    filter against the real signature instead of branching on a version number.
    """
    parameters = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        accepted: Dict[str, Any] = kwargs
    else:
        accepted = {k: v for k, v in kwargs.items() if k in parameters}
    return func(**accepted)
