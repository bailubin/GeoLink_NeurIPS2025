# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock

class TwoWayAttentionBlock_Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float=0,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.mlp=nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.norm4=nn.LayerNorm(embedding_dim)


    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, key_padding_mask = None, require_attn=False
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        k = keys + key_pe
        if key_padding_mask is None:
            attn_out, _ = self.self_attn(query=k, key=k, value=keys)
        else:
            attn_out, _ = self.self_attn(query=k, key=k, value=keys, key_padding_mask=key_padding_mask)
        keys = keys + attn_out
        keys = self.norm1(keys)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        if key_padding_mask is None:
            attn_out, _ = self.cross_attn_token_to_image(query=q, key=k, value=keys)
        else:
            attn_out, _ = self.cross_attn_token_to_image(query=q, key=k, value=keys, key_padding_mask=key_padding_mask)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out, attn = self.cross_attn_image_to_token(query=k, key=q, value=queries)
        keys = keys + attn_out
        keys = self.norm3(keys)

        keys=self.mlp(keys)
        keys=self.norm4(keys)

        if require_attn:
            return queries, keys, attn
        else:
            return queries, keys