"""Anchor-aware attention: the mechanism behind within-batch consistency."""

import pytest
import torch

from instruct4d.ip2p.attention import (
    ANCHOR_INDEX,
    SELF_ATTENTION_TYPES,
    AnchorAttention,
    AnchorSelfAttention,
    BasicTransformerBlock,
)

DIM, HEADS, TOKENS = 32, 4, 12


def make_attention(name):
    return SELF_ATTENTION_TYPES[name](query_dim=DIM, heads=HEADS, dim_head=DIM // HEADS)


@pytest.mark.parametrize("name", sorted(SELF_ATTENTION_TYPES))
def test_output_shape_is_preserved(name):
    attn = make_attention(name)
    x = torch.randn(4, TOKENS, DIM)
    assert attn(x, num_frames=4).shape == x.shape


@pytest.mark.parametrize("name", sorted(SELF_ATTENTION_TYPES))
def test_num_frames_is_required(name):
    with pytest.raises(ValueError, match="num_frames"):
        make_attention(name)(torch.randn(2, TOKENS, DIM))


def test_anchor_attention_reads_only_the_anchor():
    """Every frame's output must depend on the anchor and nothing else."""
    attn = make_attention("anchor")
    torch.manual_seed(0)
    x = torch.randn(3, TOKENS, DIM)
    base = attn(x, num_frames=3)

    perturbed_anchor = x.clone()
    perturbed_anchor[ANCHOR_INDEX] += 5.0
    assert not torch.allclose(base[1], attn(perturbed_anchor, num_frames=3)[1], atol=1e-4)

    perturbed_other = x.clone()
    perturbed_other[2] += 5.0
    assert torch.allclose(base[1], attn(perturbed_other, num_frames=3)[1], atol=1e-5)


def test_anchor_self_attention_also_reads_the_frame_itself():
    attn = make_attention("anchor_self")
    torch.manual_seed(0)
    x = torch.randn(3, TOKENS, DIM)
    base = attn(x, num_frames=3)

    perturbed_other = x.clone()
    perturbed_other[2] += 5.0
    # Frame 2's own content feeds its own output, unlike anchor-only attention.
    assert not torch.allclose(base[2], attn(perturbed_other, num_frames=3)[2], atol=1e-4)
    # But it still cannot see a different non-anchor frame.
    assert torch.allclose(base[1], attn(perturbed_other, num_frames=3)[1], atol=1e-5)


def test_identical_frames_give_identical_outputs():
    attn = make_attention("anchor")
    x = torch.randn(1, TOKENS, DIM).repeat(4, 1, 1)
    out = attn(x, num_frames=4)
    assert torch.allclose(out[0], out[3], atol=1e-5)


def test_single_frame_falls_back_to_self_attention():
    """The single-view pipeline edits the key frame on its own."""
    attn = make_attention("anchor_self")
    x = torch.randn(1, TOKENS, DIM)
    assert attn(x, num_frames=1).shape == x.shape


def test_transformer_block_selects_the_variant():
    block = BasicTransformerBlock(DIM, HEADS, DIM // HEADS, cross_attention_dim=16,
                                  self_attention="anchor_self")
    assert isinstance(block.attn1, AnchorSelfAttention)
    assert isinstance(BasicTransformerBlock(DIM, HEADS, DIM // HEADS).attn1, AnchorAttention)

    x = torch.randn(4, TOKENS, DIM)
    out = block(x, encoder_hidden_states=torch.randn(4, 5, 16), sequence_length=4)
    assert out.shape == x.shape


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="self_attention"):
        BasicTransformerBlock(DIM, HEADS, DIM // HEADS, self_attention="nonexistent")
