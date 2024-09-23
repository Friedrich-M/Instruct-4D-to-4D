from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward, AdaLayerNorm
from diffusers.utils.import_utils import is_xformers_available
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
    
from einops import rearrange

@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # Define 4 blocks. Each block has its own normalization layer. The 4th block is the Temp-Attn block which is not used in the original IP2P model.
        # 1. SC-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)  
        else:
            self.norm1 = nn.LayerNorm(dim)
        
        self.attn1 = TestAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        
        # 2. Cross-Attn
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

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        
        # 4. Temp-Attn
        self.norm_temp = (
            AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        )
        self.attn_temp = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data) # initialize the output weight of Temp-Attn to zero
        
    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(
        self, 
        hidden_states: torch.FloatTensor, 
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        sequence_length: Optional[torch.LongTensor] = None,
    ):
        # hidden_states: (batch*sequence_length, height*weight, inner_dim)
        # encoder_hidden_states: (batch*sequence_length, max_length, embed_dim)
        
        # Notice that normalization is always applied before the real computation in the following blocks.
        
        # 1. Self-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(
                    norm_hidden_states, 
                    encoder_hidden_states, 
                    attention_mask=attention_mask) 
                + hidden_states
            )
        else:
            hidden_states = (
                self.attn1(
                    norm_hidden_states, 
                    attention_mask=attention_mask, 
                    num_frames=sequence_length
                )
                + hidden_states
            ) # (batch*sequence_length, height*width, inner_dim)
            
        # Temporal-Attention
        # d = hidden_states.shape[1]
        # hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=sequence_length) # (batch*height*width, sequence_length, inner_dim)
        # norm_hidden_states = (
        #     self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        # )
        # hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        # hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, 
                    encoder_hidden_states=encoder_hidden_states, 
                    attention_mask=attention_mask,
                    # num_frames=sequence_length
                )
                + hidden_states
            )

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states # (batch*sequence_length, height*width, inner_dim)
        
        # 4. Temporal-Attention
        # d = hidden_states.shape[1]
        # hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=sequence_length) # (batch*height*width, sequence_length, inner_dim)
        # norm_hidden_states = (
        #     self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        # )
        # hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        # hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        
        # 5. 3D-Attention
        # d = hidden_states.shape[1]
        # hidden_states = rearrange(hidden_states, "(b f) d c -> b (f d) c", f=sequence_length) # (batch, height*width*sequence_length, inner_dim)
        # norm_hidden_states = (
        #     self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        # )
        # hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        # hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", d=d)
         
        return hidden_states


class SparseCausalAttention(Attention):
    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states=None, 
        attention_mask=None, 
        num_frames=None,
    ):
        residual = hidden_states
        
        input_ndim = hidden_states.ndim
        
        batch_size, sequence_length, _ = hidden_states.shape
        # hidden_states: (batch*num_frames, height*width, heads*head_dim)
        
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # (batch*num_frames, height*width, heads*head_dim)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states) # (batch*num_frames, max_length, heads*head_dim)
        value = self.to_v(encoder_hidden_states) # (batch*num_frames, max_length, heads*head_dim)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, height*width, head_dim)
                
        former_frame_index = torch.arange(num_frames) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=num_frames)
        key = torch.cat([key[:, [0] * num_frames], key[:, former_frame_index]], dim=2) # (batch, num_frames, 2*max_length, heads*head_dim)
        key = rearrange(key, "b f d c -> (b f) d c") # (batch*num_frames, 2*max_length, heads*head_dim)

        value = rearrange(value, "(b f) d c -> b f d c", f=num_frames)
        value = torch.cat([value[:, [0] * num_frames], value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c") # (batch*num_frames, 2*max_length, heads*head_dim)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, 2*max_length, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, 2*max_length, head_dim)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        ) # (batch*num_frames, heads, height*width, head_dim)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim) # (batch*num_frames, height*width, heads*head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
    
        if self.residual_connection:
            hidden_states = hidden_states + residual
            
        hidden_states = hidden_states / self.rescale_output_factor
            
        return hidden_states
    
    
class KeyFrameAttention(Attention):
    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states=None, 
        attention_mask=None, 
        num_frames=None,
        k=0,
    ):
        residual = hidden_states
        
        input_ndim = hidden_states.ndim
        
        batch_size, sequence_length, _ = hidden_states.shape
        # hidden_states: (batch*num_frames, height*width, heads*head_dim)
        
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # (batch*num_frames, height*width, heads*head_dim)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states) # (batch*num_frames, max_length, heads*head_dim)
        value = self.to_v(encoder_hidden_states) # (batch*num_frames, max_length, heads*head_dim)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, height*width, head_dim)
                
        key = rearrange(key, "(b f) d c -> b f d c", f=num_frames)
        key = key[:, [k] * num_frames] # (batch, num_frames, max_length, heads*head_dim)
        key = rearrange(key, "b f d c -> (b f) d c") # (batch*num_frames, max_length, heads*head_dim)

        value = rearrange(value, "(b f) d c -> b f d c", f=num_frames)
        value = value[:, [k] * num_frames]
        value = rearrange(value, "b f d c -> (b f) d c") # (batch*num_frames, max_length, heads*head_dim)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, max_length, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, max_length, head_dim)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        ) # (batch*num_frames, heads, height*width, head_dim)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim) # (batch*num_frames, height*width, heads*head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
    
        if self.residual_connection:
            hidden_states = hidden_states + residual
            
        hidden_states = hidden_states / self.rescale_output_factor
            
        return hidden_states
    
    
class SlideAttention(Attention):
    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states=None, 
        attention_mask=None, 
        num_frames=None,
        k=0,
    ):
        residual = hidden_states

        input_ndim = hidden_states.ndim

        batch_size, sequence_length, _ = hidden_states.shape
        # hidden_states: (batch*num_frames, height*width, heads*head_dim)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # (batch*num_frames, height*width, heads*head_dim)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states) # (batch*num_frames, max_length, embedding_size)

        key = self.to_k(encoder_hidden_states) # (batch*num_frames, max_length, heads*head_dim)
        value = self.to_v(encoder_hidden_states) # (batch*num_frames, max_length, heads*head_dim)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, height*width, head_dim)
        
        assert num_frames is not None, "num_frames must be provided for SlideAttention"
        
        key = rearrange(key, "(b f) d c -> b f d c", f=num_frames) # (batch, num_frames, max_length, heads*head_dim)
        key[0, k] = key[2, k]
        key = rearrange(key, "b f d c -> (b f) d c") # (batch*num_frames, max_length, heads*head_dim)
        
        value = rearrange(value, "(b f) d c -> b f d c", f=num_frames) # (batch, num_frames, max_length, heads*head_dim)
        value[0, k] = value[2, k]
        value = rearrange(value, "b f d c -> (b f) d c") # (batch*num_frames, max_length, heads*head_dim)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, max_length, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, max_length, head_dim)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        ) # (batch*num_frames, heads, height*width, head_dim)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim) # (batch*num_frames, height*width, heads*head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states) 
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states
            

class TestAttention(Attention):
    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states=None, 
        attention_mask=None, 
        num_frames=None,
        k=0,
    ):
        residual = hidden_states
        
        input_ndim = hidden_states.ndim
        
        batch_size, sequence_length, _ = hidden_states.shape
        # hidden_states: (batch*num_frames, height*width, heads*head_dim)
        
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # (batch*num_frames, height*width, heads*head_dim)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states) # (batch*num_frames, max_length, heads*head_dim)
        value = self.to_v(encoder_hidden_states) # (batch*num_frames, max_length, heads*head_dim)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, height*width, head_dim)
        
        former_frame_index = torch.arange(num_frames) - 1
        former_frame_index[0] = 0
        
        current_frame_index = torch.arange(num_frames)
        
        latter_frame_index = torch.arange(num_frames) + 1
        latter_frame_index[-1] = num_frames - 1
        
        if key.shape[0] > 1:
            key = rearrange(key, "(b f) d c -> b f d c", f=num_frames)
            key = torch.cat([key[:, [k] * num_frames], key[:, current_frame_index]], dim=2) # (batch, num_frames, 4*max_length, heads*head_dim)
            key = rearrange(key, "b f d c -> (b f) d c") # (batch*num_frames, max_length, heads*head_dim)

            value = rearrange(value, "(b f) d c -> b f d c", f=num_frames)
            value = torch.cat([value[:, [k] * num_frames], value[:, current_frame_index]], dim=2)
            value = rearrange(value, "b f d c -> (b f) d c") # (batch*num_frames, max_length, heads*head_dim)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, max_length, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) # (batch*num_frames, heads, max_length, head_dim)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        ) # (batch*num_frames, heads, height*width, head_dim)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim) # (batch*num_frames, height*width, heads*head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
    
        if self.residual_connection:
            hidden_states = hidden_states + residual
            
        hidden_states = hidden_states / self.rescale_output_factor
            
        return hidden_states
