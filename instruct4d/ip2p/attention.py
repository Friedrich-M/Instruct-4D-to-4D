"""Anchor-aware self-attention for the pseudo-3D InstructPix2Pix UNet.

The stock InstructPix2Pix UNet applies self-attention independently to every
image, so a batch of frames comes back with a batch of unrelated stylisations.
Section 3.2 of the paper replaces that self-attention with *anchor-aware*
attention: queries still come from the frame being denoised, but keys and values
are drawn from a shared anchor frame.  Every frame in the batch is therefore
denoised against the same reference appearance, which is what produces
within-batch consistency without any finetuning.

Three variants are provided; :data:`SELF_ATTENTION_TYPES` maps the name used in
configuration to the class.

``anchor``
    Keys/values come from the anchor frame alone.  The strongest consistency
    constraint; used by the multi-view pipeline, where the anchor is an already
    edited pseudo-view that the rest of the batch must match.
``anchor_self``
    Keys/values are the anchor's concatenated with the frame's own.  Frames keep
    more of their individual content, which matters in the single-view setting
    where consecutive frames differ substantially.
``sparse_causal``
    Keys/values are the first frame's concatenated with the previous frame's,
    the scheme introduced by Tune-A-Video.  Kept for comparison.

All three share the same shape convention.  ``hidden_states`` arrives flattened
as ``(batch * num_frames, height * width, heads * head_dim)``, and ``num_frames``
is threaded down from the UNet so the frame axis can be recovered.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from diffusers.models.attention import AdaLayerNorm, FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.utils import maybe_allow_in_graph
from diffusers.utils.import_utils import is_xformers_available

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

#: Position of the anchor frame within a batch handed to the UNet.  Callers are
#: responsible for putting the frame they want to propagate from at this index.
ANCHOR_INDEX = 0


class _AnchorAttentionBase(Attention):
    """Shared plumbing for the anchor-aware attention variants.

    Subclasses only implement :meth:`gather_context`, which decides which frames
    contribute keys and values.  Everything else -- projection, the scaled
    dot-product, the output projection and the residual -- is identical to
    ``diffusers.models.attention_processor.Attention``.
    """

    def gather_context(self, tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Re-mix a projected key/value tensor across the frame axis.

        Args:
            tensor: ``(batch * num_frames, tokens, heads * head_dim)``.
            num_frames: Length of the frame axis folded into dimension 0.

        Returns:
            A tensor of the same rank whose token axis may be longer, because a
            variant may concatenate context from several frames.
        """
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        if num_frames is None:
            raise ValueError(f"{type(self).__name__} requires num_frames to be passed down from the UNet.")

        residual = hidden_states
        batch_size = hidden_states.shape[0]

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        head_dim = key.shape[-1] // self.heads

        # This is the anchor-aware step: the query stays per-frame while the
        # key/value context is pulled from the anchor.
        key = self.gather_context(key, num_frames)
        value = self.gather_context(value, num_frames)

        # (batch * num_frames, heads, tokens, head_dim)
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)  # linear projection
        hidden_states = self.to_out[1](hidden_states)  # dropout

        if self.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / self.rescale_output_factor


class AnchorAttention(_AnchorAttentionBase):
    """Every frame attends to the anchor frame only."""

    def gather_context(self, tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
        tensor = rearrange(tensor, "(b f) d c -> b f d c", f=num_frames)
        tensor = tensor[:, [ANCHOR_INDEX] * num_frames]
        return rearrange(tensor, "b f d c -> (b f) d c")


class AnchorSelfAttention(_AnchorAttentionBase):
    """Every frame attends to the anchor frame *and* itself.

    Doubles the key/value token count relative to :class:`AnchorAttention`.
    """

    def gather_context(self, tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
        tensor = rearrange(tensor, "(b f) d c -> b f d c", f=num_frames)
        if tensor.shape[0] == 1 and num_frames == 1:
            # Single frame with no companions: there is nothing to anchor to, so
            # fall back to plain self-attention rather than duplicating tokens.
            return rearrange(tensor, "b f d c -> (b f) d c")
        own = torch.arange(num_frames)
        tensor = torch.cat([tensor[:, [ANCHOR_INDEX] * num_frames], tensor[:, own]], dim=2)
        return rearrange(tensor, "b f d c -> (b f) d c")


class SparseCausalAttention(_AnchorAttentionBase):
    """Every frame attends to the first frame and its predecessor.

    The scheme from Tune-A-Video; kept as a baseline for the anchor-aware
    variants above.
    """

    def gather_context(self, tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
        previous = torch.arange(num_frames) - 1
        previous[0] = 0
        tensor = rearrange(tensor, "(b f) d c -> b f d c", f=num_frames)
        tensor = torch.cat([tensor[:, [0] * num_frames], tensor[:, previous]], dim=2)
        return rearrange(tensor, "b f d c -> (b f) d c")


#: Selectable self-attention schemes, keyed by the name used in configuration.
SELF_ATTENTION_TYPES = {
    "anchor": AnchorAttention,
    "anchor_self": AnchorSelfAttention,
    "sparse_causal": SparseCausalAttention,
}

#: Used when a caller does not specify one; matches the multi-view pipeline.
DEFAULT_SELF_ATTENTION = "anchor"


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    """A transformer block whose self-attention is anchor-aware.

    Identical to the diffusers 2D block except that block 1 uses one of
    :data:`SELF_ATTENTION_TYPES` and receives ``sequence_length`` so it can
    recover the frame axis.

    Args:
        self_attention: Key into :data:`SELF_ATTENTION_TYPES`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        self_attention: str = DEFAULT_SELF_ATTENTION,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        if self_attention not in SELF_ATTENTION_TYPES:
            raise ValueError(
                f"self_attention must be one of {sorted(SELF_ATTENTION_TYPES)}, got {self_attention!r}"
            )

        # 1. Anchor-aware self-attention.
        self.norm1 = (
            AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        )
        self.attn1 = SELF_ATTENTION_TYPES[self_attention](
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-attention on the text embedding (unchanged from 2D).
        if cross_attention_dim is not None:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward (unchanged from 2D).
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for how to install xformers",
                name="xformers",
            )
        if not torch.cuda.is_available():
            raise ValueError("xformers' memory efficient attention requires a GPU.")

        # Probe before enabling: xformers builds are easy to mismatch against
        # the installed CUDA/torch, and the failure is clearer here than deep
        # inside the denoising loop.
        xformers.ops.memory_efficient_attention(
            torch.randn((1, 2, 40), device="cuda"),
            torch.randn((1, 2, 40), device="cuda"),
            torch.randn((1, 2, 40), device="cuda"),
        )
        self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        if self.attn2 is not None:
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        sequence_length: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states: ``(batch * sequence_length, height * width, dim)``.
            encoder_hidden_states: ``(batch * sequence_length, max_length, embed_dim)``.
            sequence_length: Number of frames folded into dimension 0.
        """
        # 1. Anchor-aware self-attention.
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask)
                + hidden_states
            )
        else:
            hidden_states = (
                self.attn1(norm_hidden_states, attention_mask=attention_mask, num_frames=sequence_length)
                + hidden_states
            )

        # 2. Cross-attention.
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                + hidden_states
            )

        # 3. Feed-forward.
        return self.ff(self.norm3(hidden_states)) + hidden_states
