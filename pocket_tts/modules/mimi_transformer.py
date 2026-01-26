from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from typing_extensions import Self

from pocket_tts.modules.layer_scale import LayerScale
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.utils.config import FlowLMTransformerConfig


class KVCacheResult(NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions.expand(B, -1))


def complete(
    cache: torch.Tensor, end_offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> KVCacheResult:
    capacity = cache.shape[3]
    assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
    B, H, T, D = k.shape
    assert T > 0
    indexes = torch.arange(T, device=end_offset.device, dtype=end_offset.dtype)
    indexes = indexes + end_offset.view(-1, 1)
    indexes = indexes % capacity
    # indexes is [B, T]
    # k is [B, H, T, D]
    # cache is [B, H, T', D]
    this_indexes = indexes.view(B, 1, T, 1)
    this_indexes = this_indexes.expand(-1, H, T, D)
    cache[0].scatter_(2, this_indexes, k)
    cache[1].scatter_(2, this_indexes, v)

    keys = cache[0]
    values = cache[1]

    indexes = torch.arange(capacity, device=end_offset.device, dtype=torch.long)

    # end_index correspond to the actual index where the last value was written.
    last_offset = end_offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes - end_index

    positions = torch.where(delta <= 0, last_offset + delta, last_offset + delta - capacity)
    end_offset[:] = end_offset + T
    invalid = indexes >= end_offset.view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)

    return KVCacheResult(keys, values, positions)


class MimiStreamingMultiheadAttention(StatefulModule):
    def __init__(self, embed_dim: int, num_heads: int, context: int, rope: RotaryEmbedding):
        super().__init__()

        self.embed_dim = embed_dim
        self.context = context
        self.rope = rope
        self.num_heads = num_heads

        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        dim_per_head = self.embed_dim // self.num_heads

        state = {}
        state["offset"] = torch.zeros(batch_size, dtype=torch.long)
        state["cache"] = torch.zeros((2, batch_size, self.num_heads, sequence_length, dim_per_head))
        state["end_offset"] = torch.zeros(batch_size, dtype=torch.long)
        return state

    def increment_step(self, state, increment: int = 1):
        state["offset"] += increment

    def _complete_kv(self, k, v, model_state: dict | None) -> KVCacheResult:
        if model_state is None:
            return KVCacheResult.from_kv(k, v)
        else:
            layer_state = self.get_state(model_state)
            return complete(layer_state["cache"], layer_state["end_offset"], k, v)

    def forward(self, query: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        B, T = query.shape[:2]

        if model_state is None:
            offset = torch.zeros(B, device=query.device, dtype=torch.long)
        else:
            offset = self.get_state(model_state)["offset"]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        projected = torch.cat([q, k, v], dim=2)

        q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)

        # Permute from [b, h, t, d] to [b, t, h, d] for rope
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        q, k = self.rope(q, k, offset)
        # Permute back from [b, t, h, d] to [b, h, t, d]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        k, v, pos_k = self._complete_kv(k, v, model_state)
        pos_k = pos_k[:, None]
        pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(
            -1, 1
        )
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)
        attn_bias = attn_bias & (delta < self.context)
        attn_bias = attn_bias[:, None]

        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class StreamingTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: int | None,
        rope: RotaryEmbedding,
        layer_scale: float | None = None,
        attention_kind: str = "mimi",
    ):
        super().__init__()
        # Redefine self_attn to our streaming multi-head attention
        if attention_kind == "mimi":
            # TODO: we should actually use StreamingMultiheadAttention here and add context window
            # support. And we should then delete MimiStreamingMultiheadAttention.
            # The implementation is really close.
            self.self_attn = MimiStreamingMultiheadAttention(
                context=context, rope=rope, embed_dim=d_model, num_heads=num_heads
            )
        else:
            self.self_attn = StreamingMultiheadAttention(
                rope=rope, embed_dim=d_model, num_heads=num_heads
            )
        self.input_layernorm = nn.LayerNorm(d_model, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-5)

        self.mlp = Mlp(d_model, dim_feedforward, d_model)

        if layer_scale is None:
            self.self_attn_layer_scale = nn.Identity()
            self.mlp_layer_scale = nn.Identity()
        else:
            self.self_attn_layer_scale = LayerScale(d_model, layer_scale)
            self.mlp_layer_scale = LayerScale(d_model, layer_scale)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        x = self.post_attention_layernorm(x)
        update = self.mlp(x)
        return x_orig.to(update) + self.mlp_layer_scale(update)

    def _sa_block(self, x: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        x_orig = x
        x = self.input_layernorm(x)
        update = self.self_attn(x, model_state)
        return x_orig.to(update) + self.self_attn_layer_scale(update)

    def forward(self, x: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        x = self._sa_block(x, model_state)
        x = self._ff_block(x)
        return x

class ProjectedTransformer(nn.Module):
    def __init__(
        self,
        input_dimension: int = 512,
        output_dimensions: tuple[int, ...] = (512,),
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
        layer_scale: float = 0.01,
        context: int = 250,
        max_period: float = 10_000.0,
        dim_feedforward: int = 2048,
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(nn.Linear(d_model, output_dimension, bias=False))

        self.max_period = max_period

        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    rope=self.rope,
                    layer_scale=layer_scale,
                    attention_kind="mimi",
                )
            )

    def forward(self, x, model_state: dict | None):
        x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, model_state)
            
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(x)
            y = y.transpose(1, 2)
            ys.append(y)
        return ys

class StreamingTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float | None = None,
        dim_feedforward: int | list[int] = 2048,
        context: int | None = None,
        max_period: float = 10_000.0,
        kind: str = "mimi",
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period

        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    rope=self.rope,
                    layer_scale=layer_scale,
                    attention_kind=kind,
                )
            )

    @classmethod
    def from_pydantic_config(cls, config: FlowLMTransformerConfig) -> Self:
        dim_feedforward = int(config.d_model * config.hidden_scale)
        return cls(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=dim_feedforward,
            max_period=float(config.max_period),
            kind="flow_lm",
        )

    def forward(self, x: torch.Tensor, model_state: dict | None):
        for layer in self.layers:
            x = layer(x, model_state)
        return x
