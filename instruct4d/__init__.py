"""Instruct 4D-to-4D: editing 4D scenes as pseudo-3D scenes using 2D diffusion.

The package is organised around the three components of the paper:

``instruct4d.ip2p``
    Anchor-aware InstructPix2Pix -- a pseudo-3D inflation of the 2D editor that
    edits a whole batch of frames with a single, shared appearance.
``instruct4d.editing``
    The two consistency mechanisms built on top of it: flow-guided sliding
    window propagation along time, and depth-based warping across views.
``instruct4d.fields`` / ``instruct4d.data`` / ``instruct4d.rendering``
    The streaming TensoRF 4D scene representation that consumes the edited
    images, its DyNeRF data loader, and the volumetric renderer.
"""

__version__ = "1.0.0"
